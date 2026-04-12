"""Denoiser model architectures.

NEFResidual
    Single-frame spatial denoiser.  UNet encoder-decoder with residual
    learning: the network predicts the noise component, which is subtracted
    from the input to recover the clean image.

NEFTemporal
    Multi-frame temporal denoiser.  Encodes all T frames with a shared
    encoder.  At each scale the reference frame features are concatenated
    with the mean of neighbour features and mixed by a learned 1×1 conv.

    Optional learned warp (ModelConfig.use_warp=True): before mixing, each
    neighbour feature map is warped to the reference using a per-level
    offset head — a lightweight Conv2d(2C, 2, 1) that predicts a dense
    (dx, dy) displacement field applied via bilinear grid_sample.  Use this
    for real video with significant camera or object motion.  Leave disabled
    for render sequences where Monte Carlo noise is pixel-independent.

Both models accept float32 images and return float32 values. No clamping is applied — the full input range is preserved.
"""

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
        use_warp: Enable learned per-level warp in NEFTemporal.  Each
            neighbour feature map is warped to the reference before mixing
            using a lightweight Conv2d(2C, 2, 1) offset head.  Recommended
            for real video; leave False for render sequences (default: False).
        sigma_min: Minimum noise std used during training (informational).
        sigma_max: Maximum noise std used during training (informational).
    """

    enc_channels: list[int] = field(default_factory=lambda: [64, 128, 256, 512])
    in_channels: int = 3
    out_channels: int = 3
    num_frames: int = 5
    use_warp: bool = False
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
# Spatial weight transfer and freeze helpers
# ---------------------------------------------------------------------------


def load_spatial_weights(model: "NEFTemporal", path: "Path | str") -> int:
    """Load encoder / bottleneck / decoder / head weights from a NEFResidual checkpoint.

    Keys that exist in both models and have matching shapes are transferred;
    temporal-only keys (temporal_mix, offset_heads) are left at their
    initialised values.

    Args:
        model: A NEFTemporal instance to update in-place.
        path:  Path to a NEFResidual checkpoint saved by training.py
               (must contain a ``model_state_dict`` key).

    Returns:
        Number of parameter tensors transferred.
    """
    from pathlib import Path as _Path

    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    src = ckpt.get("model_state_dict", ckpt)
    dst = model.state_dict()
    transferred = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    dst.update(transferred)
    model.load_state_dict(dst)
    return len(transferred)


def freeze_spatial(model: "NEFTemporal") -> None:
    """Freeze encoder / bottleneck / decoder / head — train temporal components only.

    After calling this, only ``temporal_mix`` (and ``offset_heads`` when
    use_warp=True) have ``requires_grad=True``.
    """
    for module in (model.encoders, model.bottleneck, model.decoders, model.head):
        for p in module.parameters():
            p.requires_grad_(False)


def unfreeze_spatial(model: "NEFTemporal") -> None:
    """Re-enable gradients for all spatial parameters (undo freeze_spatial)."""
    for module in (model.encoders, model.bottleneck, model.decoders, model.head):
        for p in module.parameters():
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# Warp utility
# ---------------------------------------------------------------------------


def _warp(feat: Tensor, offset: Tensor) -> Tensor:
    """Warp *feat* by a dense displacement field *offset*.

    Args:
        feat:   (B, C, H, W) feature map to warp.
        offset: (B, 2, H, W) displacement in pixel units.
                offset[:, 0] = dx (horizontal), offset[:, 1] = dy (vertical).

    Returns:
        Warped feature map (B, C, H, W), sampled bilinearly with border padding.
    """
    b, _, h, w = feat.shape
    # Normalised base grid in [-1, 1]
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, h, device=feat.device, dtype=feat.dtype),
        torch.linspace(-1.0, 1.0, w, device=feat.device, dtype=feat.dtype),
        indexing="ij",
    )
    base = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)
    # Convert pixel-space offset to [-1, 1] normalised space
    norm = torch.stack([offset[:, 0] / (w / 2.0), offset[:, 1] / (h / 2.0)], dim=-1)  # (B, H, W, 2)
    grid = base + norm  # (B, H, W, 2)
    return F.grid_sample(feat, grid, mode="bilinear", padding_mode="border", align_corners=True)


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
# NEFTemporal
# ---------------------------------------------------------------------------


class NEFTemporal(nn.Module):
    """Multi-frame temporal denoiser with simple feature mixing.

    Architecture:
        1. Shared encoder applied to all T frames (as a batch).
        2. Per-level mixing: mean of neighbour features concatenated with
           reference features, projected back to C channels via a 1×1 conv.
        3. Optional warp (use_warp=True): before computing the neighbour mean,
           each neighbour feature is warped to the reference via a per-level
           offset head — Conv2d(2C, 2, 1) predicting a dense (dx, dy) field
           applied with bilinear grid_sample.  Recommended for real video.
        4. Shared decoder with mixed temporal skip connections.
        5. Same residual head as NEFResidual.

    The reference frame is always the centre frame:
        reference index = num_frames // 2

    At epoch 0 the model is identity: temporal_mix layers are initialised so
    the reference-feature slice passes through unchanged (neighbour weight = 0).
    Offset heads are zero-initialised so warp starts as identity.

    Args:
        config: ModelConfig; num_frames and use_warp are used.

    Example::

        model = NEFTemporal(ModelConfig.standard())
        clip = torch.rand(1, 5, 3, 256, 256)
        denoised = model(clip)  # (B, C, H, W) — denoised centre frame

        # With warp for real video:
        cfg = ModelConfig.standard()
        cfg.use_warp = True
        model = NEFTemporal(cfg)
    """

    @classmethod
    def from_residual(
        cls,
        path: "Path | str",
        config: Optional[ModelConfig] = None,
    ) -> "NEFTemporal":
        """Create a NEFTemporal pre-loaded with weights from a NEFResidual checkpoint.

        Temporal components (temporal_mix, offset_heads) keep their identity
        initialisations; the spatial backbone is warm-started from *path*.

        Args:
            path:   Path to a NEFResidual checkpoint produced by training.py.
            config: ModelConfig to use (default: standard).  Must match the
                    architecture used when training the residual model.

        Returns:
            A NEFTemporal instance with spatial weights transferred.

        Example::

            model = NEFTemporal.from_residual("checkpoints/residual/best.pth")
            freeze_spatial(model)   # stage 2: train temporal components only
        """
        model = cls(config)
        n = load_spatial_weights(model, path)
        return model

    def __init__(self, config: Optional[ModelConfig] = None) -> None:
        super().__init__()
        cfg = config or ModelConfig.standard()
        self._pad_multiple = cfg.pad_multiple
        self._num_frames = cfg.num_frames
        self._ref_idx = cfg.num_frames // 2
        self._use_warp = cfg.use_warp

        enc_ch = cfg.enc_channels
        in_ch = cfg.in_channels
        out_ch = cfg.out_channels

        # Shared encoder (same weights for all frames)
        self.encoders = nn.ModuleList()
        prev_ch = in_ch
        for ch in enc_ch:
            self.encoders.append(EncoderBlock(prev_ch, ch))
            prev_ch = ch

        # Per-level offset heads (only created when use_warp=True).
        # Input: cat([ref_feat, neigh_feat]) → 2-channel (dx, dy) displacement.
        # Zero-init: warp starts as identity (zero displacement) at epoch 0.
        if self._use_warp:
            self.offset_heads = nn.ModuleList()
            for ch in enc_ch:
                head = nn.Conv2d(2 * ch, 2, kernel_size=1, bias=True)
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
                self.offset_heads.append(head)

        # Per-level temporal mixing: cat([ref, neigh_mean], dim=1) → ch.
        # Input layout: channels [0..ch-1] = ref, [ch..2*ch-1] = neighbour mean.
        # Identity init: ref slice = eye, neighbour slice = zero.
        # At epoch 0 output == ref features; neighbour influence is learned.
        self.temporal_mix = nn.ModuleList()
        for ch in enc_ch:
            conv = nn.Conv2d(2 * ch, ch, kernel_size=1, bias=True)
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            for c in range(ch):
                conv.weight.data[c, c, 0, 0] = 1.0  # ref slice → identity
            self.temporal_mix.append(conv)

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
        assert t == self._num_frames, f"Expected {self._num_frames} frames, got {t}"

        ref_input = clip[:, self._ref_idx]  # (B, C, H, W)

        # Pad all frames identically (same H, W → same padding)
        frames_padded = []
        for i in range(t):
            padded, padding = _pad_to_multiple(clip[:, i], self._pad_multiple)
            frames_padded.append(padded)

        # --- Encode all frames (batch trick: (B*T, C, H, W)) ---
        stacked = torch.stack(frames_padded, dim=1)   # (B, T, C, H, W)
        x = stacked.view(b * t, c, *stacked.shape[-2:])  # (B*T, C, H, W)

        all_skips: list[list[Tensor]] = []  # [level][frame]
        all_pooled: list[Tensor] = []

        for enc in self.encoders:
            x, skip = enc(x)
            _, cp, hp, wp = skip.shape
            skip_bt = skip.view(b, t, cp, hp, wp)
            all_skips.append([skip_bt[:, i] for i in range(t)])
            all_pooled.append(x)

        # --- Per-level temporal mixing (with optional learned warp) ---
        fused_skips: list[Tensor] = []
        for level, mix in enumerate(self.temporal_mix):
            ref_feat = all_skips[level][self._ref_idx]  # (B, C', H', W')
            neigh_feats = [all_skips[level][i] for i in range(t) if i != self._ref_idx]

            if self._use_warp:
                offset_head = self.offset_heads[level]
                warped = []
                for nf in neigh_feats:
                    offset = offset_head(torch.cat([ref_feat, nf], dim=1))  # (B, 2, H', W')
                    warped.append(_warp(nf, offset))
                neigh_feats = warped

            neigh_mean = torch.stack(neigh_feats, dim=0).mean(dim=0)  # (B, C', H', W')
            fused = mix(torch.cat([ref_feat, neigh_mean], dim=1))
            fused_skips.append(fused)

        # --- Bottleneck (reference frame's deepest features) ---
        _, cp, hp, wp = all_pooled[-1].shape
        ref_deep = all_pooled[-1].view(b, t, cp, hp, wp)[:, self._ref_idx]
        x = self.bottleneck(ref_deep)

        # --- Decoder ---
        for dec, skip in zip(self.decoders, reversed(fused_skips)):
            x = dec(x, skip)

        # --- Head + residual ---
        noise = _unpad(self.head(x), padding)
        return ref_input - noise
