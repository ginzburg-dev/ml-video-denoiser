"""Denoiser model architectures.

NEFResidual
    Single-frame spatial denoiser.  UNet encoder-decoder with residual
    learning: the network predicts the noise component, which is subtracted
    from the input to recover the clean image.

NEFTemporal
    Multi-frame temporal denoiser.  Extends NEFResidual with a deformable-
    convolution alignment stage that compensates for inter-frame motion
    before temporal feature aggregation.

Both models accept float32 images and return float32 values. No clamping is applied — the full input range is preserved.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    """Shared configuration for NEFResidual and NEFTemporal.

    Args:
        enc_channels: Number of feature channels at each encoder level.
            Length determines the number of levels (default: 4).
        in_channels: Number of input image channels (default: 3 for RGB).
        out_channels: Number of output channels (default: 3).
        num_frames: Temporal window size for NEFTemporal (default: 5).
        deform_groups: Number of deformable convolution groups (default: 8).
        sigma_min: Minimum noise std used during training (informational).
        sigma_max: Maximum noise std used during training (informational).
    """

    enc_channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    in_channels: int = 3
    out_channels: int = 3
    num_frames: int = 5
    deform_groups: int = 8
    sigma_min: float = 0.0
    sigma_max: float = 75.0 / 255.0

    @classmethod
    def lite(cls) -> "ModelConfig":
        return cls(enc_channels=[32, 64, 128, 256])

    @classmethod
    def standard(cls) -> "ModelConfig":
        return cls(enc_channels=[64, 128, 256, 512])

    @classmethod
    def heavy(cls) -> "ModelConfig":
        return cls(enc_channels=[96, 192, 384, 768])

    @property
    def num_levels(self) -> int:
        return len(self.enc_channels)

    @property
    def pad_multiple(self) -> int:
        """Input spatial dims must be divisible by this for clean skip connections."""
        return 2 ** self.num_levels


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class ConvBnRelu(nn.Module):
    """Conv2d → BatchNorm2d → ReLU block.

    Args:
        in_ch: Input channels.
        out_ch: Output channels.
        kernel_size: Convolution kernel size (default: 3).
        stride: Convolution stride (default: 1).
        padding: Convolution padding (default: 1).
        bias: Whether to add a bias term.  Defaults to False when followed by
            BatchNorm (the BN bias subsumes it).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.relu(self.bn(self.conv(x)))


class EncoderBlock(nn.Module):
    """Two ConvBnRelu layers followed by MaxPool2d.

    The feature maps *before* pooling are returned as skip connections.

    Args:
        in_ch: Input channels.
        out_ch: Output channels after both convolutions.
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = ConvBnRelu(in_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Returns (pooled_features, skip_features)."""
        x = self.conv2(self.conv1(x))
        return self.pool(x), x  # (downsampled, skip)


class DecoderBlock(nn.Module):
    """Bilinear upsample, concat with skip, two ConvBnRelu layers.

    Args:
        in_ch: Channels coming from the previous (deeper) decoder level.
        skip_ch: Channels from the corresponding encoder skip connection.
        out_ch: Output channels.
    """

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv1 = ConvBnRelu(in_ch + skip_ch, out_ch)
        self.conv2 = ConvBnRelu(out_ch, out_ch)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv2(self.conv1(x))


# ---------------------------------------------------------------------------
# Padding utilities
# ---------------------------------------------------------------------------


def _pad_to_multiple(x: Tensor, multiple: int) -> tuple[Tensor, tuple[int, int, int, int]]:
    """Reflect-pad *x* so that H and W are divisible by *multiple*.

    Returns:
        Padded tensor and the padding tuple (left, right, top, bottom) as
        used by F.pad (so it can be stripped after the forward pass).
    """
    _, _, h, w = x.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    # Pad on the right/bottom only — easier to strip
    padding = (0, pad_w, 0, pad_h)  # (left, right, top, bottom)
    return F.pad(x, padding, mode="reflect"), padding


def _unpad(x: Tensor, padding: tuple[int, int, int, int]) -> Tensor:
    """Remove padding added by _pad_to_multiple."""
    _, right, _, bottom = padding
    h, w = x.shape[-2], x.shape[-1]
    return x[..., : h - bottom if bottom else h, : w - right if right else w]


# ---------------------------------------------------------------------------
# NEFResidual
# ---------------------------------------------------------------------------


class NEFResidual(nn.Module):
    """Single-frame UNet denoiser with residual learning.

    The network estimates the noise component and the output is:
        denoised = input − predicted_noise

    This residual formulation improves convergence and prevents the model
    from hallucinating structure that was not in the original image.

    Input spatial dimensions are automatically padded to multiples of
    2^num_levels using reflect padding and stripped before returning.

    Args:
        config: ModelConfig controlling architecture size and behaviour.

    Example::

        model = NEFResidual(ModelConfig.standard())
        noisy = torch.rand(1, 3, 720, 1280)   # any size — auto-padded
        denoised = model(noisy)                # same shape as input
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        cfg = config or ModelConfig.standard()
        self._pad_multiple = cfg.pad_multiple

        enc_ch = cfg.enc_channels
        in_ch = cfg.in_channels
        out_ch = cfg.out_channels

        # Encoder
        self.encoders = nn.ModuleList()
        prev_ch = in_ch
        for ch in enc_ch:
            self.encoders.append(EncoderBlock(prev_ch, ch))
            prev_ch = ch

        # Bottleneck (deepest level, no pooling)
        bot_ch = enc_ch[-1] * 2
        self.bottleneck = nn.Sequential(
            ConvBnRelu(prev_ch, bot_ch),
            ConvBnRelu(bot_ch, bot_ch),
        )

        # Decoder (mirrors encoder in reverse)
        self.decoders = nn.ModuleList()
        prev_ch = bot_ch
        for ch in reversed(enc_ch):
            self.decoders.append(DecoderBlock(in_ch=prev_ch, skip_ch=ch, out_ch=ch))
            prev_ch = ch

        # Head: predict noise residual.
        # Zero-init weights and bias so the model starts as identity
        # (predicted noise ≈ 0 at epoch 0), avoiding random channel bias.
        self.head = nn.Conv2d(enc_ch[0], out_ch, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Denoise *x*.

        Args:
            x: (B, C, H, W) noisy image (float32, any range).

        Returns:
            Denoised image of shape (B, C, H, W), same range as input.
        """
        inp = x
        x, padding = _pad_to_multiple(x, self._pad_multiple)

        # Encoder: collect skip connections
        skips: list[Tensor] = []
        for enc in self.encoders:
            x, skip = enc(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder: use skips in reverse order
        for dec, skip in zip(self.decoders, reversed(skips)):
            x = dec(x, skip)

        # Predict noise, strip padding, subtract from original input
        noise = _unpad(self.head(x), padding)
        return inp - noise


# ---------------------------------------------------------------------------
# Deformable alignment (for NEFTemporal)
# ---------------------------------------------------------------------------


class DeformableAlignment(nn.Module):
    """DCNv2-style deformable convolution alignment.

    Aligns a *neighbour* feature map to a *reference* feature map by
    predicting spatial offsets and modulation masks via a small convolutional
    head, then applying them through torchvision's deformable convolution.

    Args:
        channels: Number of feature channels (same for both inputs).
        kernel_size: Kernel size for deformable convolution (default: 3).
        deform_groups: Number of independent deformable groups (default: 8).
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int = 3,
        deform_groups: int = 8,
    ) -> None:
        super().__init__()
        try:
            from torchvision.ops import deform_conv2d as _  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "torchvision is required for DeformableAlignment. "
                "Install it with: uv add torchvision"
            ) from exc

        pad = kernel_size // 2
        n_offset_ch = 2 * kernel_size * kernel_size * deform_groups
        n_mask_ch = kernel_size * kernel_size * deform_groups

        # Offset + mask prediction head (takes concat of ref and neighbour).
        # Zero-init so offsets start at 0 (no warp) and masks start at 0.5
        # after sigmoid — identity alignment at epoch 0.
        self.offset_conv = nn.Conv2d(
            2 * channels, n_offset_ch, kernel_size, padding=pad, bias=True
        )
        self.mask_conv = nn.Conv2d(
            2 * channels, n_mask_ch, kernel_size, padding=pad, bias=True
        )
        nn.init.zeros_(self.offset_conv.weight)
        nn.init.zeros_(self.offset_conv.bias)
        nn.init.zeros_(self.mask_conv.weight)
        nn.init.zeros_(self.mask_conv.bias)

        # Main deformable convolution weight (same channels in/out)
        self.weight = nn.Parameter(
            torch.empty(channels, channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(channels))

        self._deform_groups = deform_groups
        self._kernel_size = kernel_size
        self._padding = pad

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, ref: Tensor, neighbour: Tensor) -> Tensor:
        """Align *neighbour* towards *ref* and return aligned features.

        Args:
            ref: Reference frame features (B, C, H, W).
            neighbour: Neighbour frame features (B, C, H, W).

        Returns:
            Aligned neighbour features (B, C, H, W).
        """
        from torchvision.ops import deform_conv2d

        concat = torch.cat([ref, neighbour], dim=1)  # (B, 2C, H, W)
        offset = self.offset_conv(concat)
        mask = torch.sigmoid(self.mask_conv(concat))

        return deform_conv2d(
            neighbour,
            offset=offset,
            weight=self.weight,
            bias=self.bias,
            padding=self._padding,
            mask=mask,
        )


# ---------------------------------------------------------------------------
# NEFTemporal
# ---------------------------------------------------------------------------


class NEFTemporal(nn.Module):
    """Multi-frame temporal denoiser with deformable alignment.

    Architecture:
        1. Shared encoder applied to all T frames (as a batch).
        2. Per-level DeformableAlignment: each non-reference frame is aligned
           to the reference at every encoder scale.
        3. Per-level temporal fusion: aligned features + reference are
           concatenated across the channel dimension and projected back.
        4. Shared decoder with fused temporal skip connections.
        5. Same residual head as NEFResidual.

    The reference frame is always the centre frame:
        reference index = num_frames // 2

    Args:
        config: ModelConfig; num_frames and deform_groups are used.

    Example::

        model = NEFTemporal(ModelConfig.standard())
        # (B, T, C, H, W) — 5 consecutive frames
        clip = torch.rand(1, 5, 3, 256, 256)
        denoised = model(clip)  # (B, C, H, W) — denoised centre frame
    """

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        cfg = config or ModelConfig.standard()
        self._pad_multiple = cfg.pad_multiple
        self._num_frames = cfg.num_frames
        self._ref_idx = cfg.num_frames // 2

        enc_ch = cfg.enc_channels
        in_ch = cfg.in_channels
        out_ch = cfg.out_channels
        n_neigh = cfg.num_frames - 1  # number of non-reference frames

        # Shared encoder (same weights for all frames)
        self.encoders = nn.ModuleList()
        prev_ch = in_ch
        for ch in enc_ch:
            self.encoders.append(EncoderBlock(prev_ch, ch))
            prev_ch = ch

        # Per-level deformable alignment (weight-shared across neighbour positions)
        self.align_layers = nn.ModuleList(
            [DeformableAlignment(ch, deform_groups=cfg.deform_groups) for ch in enc_ch]
        )

        # Per-level temporal fusion: (T * ch) → ch
        self.fusion_layers = nn.ModuleList(
            [nn.Conv2d(cfg.num_frames * ch, ch, kernel_size=1) for ch in enc_ch]
        )

        # Bottleneck
        bot_ch = enc_ch[-1] * 2
        self.bottleneck = nn.Sequential(
            ConvBnRelu(enc_ch[-1], bot_ch),
            ConvBnRelu(bot_ch, bot_ch),
        )

        # Decoder
        self.decoders = nn.ModuleList()
        prev_ch = bot_ch
        for ch in reversed(enc_ch):
            self.decoders.append(DecoderBlock(in_ch=prev_ch, skip_ch=ch, out_ch=ch))
            prev_ch = ch

        # Head: zero-init for the same reason as NEFResidual.
        self.head = nn.Conv2d(enc_ch[0], out_ch, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, clip: Tensor) -> Tensor:
        """Denoise the centre frame of *clip*.

        Args:
            clip: (B, T, C, H, W) tensor of T consecutive noisy frames (float32, any range).

        Returns:
            Denoised centre frame (B, C, H, W), same range as input.
        """
        b, t, c, h, w = clip.shape
        assert t == self._num_frames, (
            f"Expected {self._num_frames} frames, got {t}"
        )

        ref_input = clip[:, self._ref_idx]  # (B, C, H, W)

        # Pad all frames identically
        frames_padded = []
        for i in range(t):
            padded, padding = _pad_to_multiple(clip[:, i], self._pad_multiple)
            frames_padded.append(padded)
        # padding is the same for all frames (same H, W)

        # --- Encode all frames (batch trick: (B*T, C, H, W)) ---
        stacked = torch.stack(frames_padded, dim=1)          # (B, T, C, H, W)
        bT = b * t
        x = stacked.view(bT, c, *stacked.shape[-2:])          # (B*T, C, H, W)

        all_skips: list[list[Tensor]] = []  # [level][frame_idx]
        all_pooled: list[Tensor] = []

        for enc in self.encoders:
            x, skip = enc(x)
            # Reshape back to (B, T, C', H', W') and split by frame
            _, cp, hp, wp = skip.shape
            skip_bt = skip.view(b, t, cp, hp, wp)
            all_skips.append([skip_bt[:, i] for i in range(t)])
            # x is still (B*T, ...) — keep for next level
            all_pooled.append(x)  # store intermediate for bottleneck

        # --- Per-level deformable alignment + temporal fusion ---
        fused_skips: list[Tensor] = []
        for level, (align, fuse) in enumerate(zip(self.align_layers, self.fusion_layers)):
            ref_feat = all_skips[level][self._ref_idx]  # (B, C', H', W')
            aligned = []
            for i in range(t):
                if i == self._ref_idx:
                    aligned.append(ref_feat)
                else:
                    aligned.append(align(ref_feat, all_skips[level][i]))
            # Concatenate along channel dim and fuse: (B, T*C', H', W') → (B, C', H', W')
            fused = fuse(torch.cat(aligned, dim=1))
            fused_skips.append(fused)

        # --- Bottleneck (use reference frame's deepest features) ---
        _, cp, hp, wp = all_pooled[-1].shape
        ref_deep = all_pooled[-1].view(b, t, cp, hp, wp)[:, self._ref_idx]
        x = self.bottleneck(ref_deep)

        # --- Decoder ---
        for dec, skip in zip(self.decoders, reversed(fused_skips)):
            x = dec(x, skip)

        # --- Head + residual ---
        noise = _unpad(self.head(x), padding)
        return ref_input - noise
