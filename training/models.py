"""Denoiser model architectures.

NAFNet
    Single-frame spatial denoiser using NAFNet blocks (ECCV 2022).

    * LayerNorm2d instead of BatchNorm2d — normalises per image per channel,
      batch-size independent, much better suited to HDR / EXR data.
    * SimpleGate (x₁ × x₂) instead of ReLU — gated, avoids dead neurons.
    * NAFBlock with depthwise conv, simplified channel attention, and dual
      β/γ-scaled residuals that are zero-initialised (identity at epoch 0).

NAFNetTemporal
    Multi-frame temporal denoiser using NAFNet blocks (ECCV 2022).
    Encodes all T frames with a shared NAFNet encoder.  At each scale the
    reference frame skip features are concatenated with the mean of neighbour
    features and mixed by a learned 1×1 conv.

    Optional learned warp (use_warp=True): before mixing, each neighbour
    feature map is warped to the reference using a per-level offset head —
    a lightweight Conv2d(2C, 2, 1) that predicts a dense (dx, dy)
    displacement field applied via bilinear grid_sample.  Use this for real
    video with significant camera or object motion.  Leave disabled for
    render sequences where Monte Carlo noise is pixel-independent.

Both models accept float32 images and return float32 values. No clamping is applied — the full input range is preserved.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


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
# NAFNet building blocks  (NAFNet — ECCV 2022, Chen et al.)
# ---------------------------------------------------------------------------


class LayerNorm2d(nn.Module):
    """Per-channel layer normalisation for 4-D feature maps (B, C, H, W).

    Equivalent to ``nn.LayerNorm(C)`` applied independently at every spatial
    position.  Unlike BatchNorm2d, statistics are computed per image — no
    dependence on batch size or other images.  This makes it robust to HDR /
    EXR data where per-batch statistics are meaningless.

    Args:
        channels: Number of feature channels (C).
        eps: Numerical stability offset (default: 1e-6).
    """

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        # F.layer_norm normalises over the *last* dims, so permute to (B, H, W, C),
        # normalise over C per spatial position, then permute back.
        x = x.permute(0, 2, 3, 1)          # (B, H, W, C)
        x = F.layer_norm(x, (x.shape[-1],), None, None, self.eps)
        x = x.permute(0, 3, 1, 2)          # (B, C, H, W)
        return x * self.weight[None, :, None, None] + self.bias[None, :, None, None]


class SimpleGate(nn.Module):
    """Gated activation: split channels in two and multiply element-wise.

    Given input of shape (B, 2C, H, W) splits into two (B, C, H, W) halves
    and returns their element-wise product.  No learnable parameters.
    Replaces ReLU throughout NAFNet — avoids dead neurons and gives the
    network a natural mechanism to suppress irrelevant activations.
    """

    def forward(self, x: Tensor) -> Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """Core NAFNet processing block.

    Two residual sub-paths, each gated by a learned scalar (β, γ):

    Path 1 (spatial + channel):
        norm1 → 1×1 expand(C→2C) → DW-conv 3×3(2C) → SimpleGate(→C)
               → SCA → 1×1 project(C→C)  scaled by β, added to residual

    Path 2 (feed-forward):
        norm2 → 1×1 expand(C→2C) → SimpleGate(→C) → 1×1 project(C→C)
               scaled by γ, added to residual

    β and γ are zero-initialised → block is identity at epoch 0.
    This eliminates the need for careful LR warm-up.

    Args:
        channels:        Number of input/output channels.
        dw_expand:       Channel multiplier for the depthwise expand step (default: 1).
        ffn_expand:      Channel multiplier for the FFN expand step (default: 2).
        drop_out_rate:   Dropout probability (default: 0.0).
    """

    def __init__(
        self,
        channels: int,
        dw_expand: int = 1,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        super().__init__()
        dw_ch = channels * dw_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch * 2, kernel_size=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(dw_ch * 2, dw_ch * 2, kernel_size=3, padding=1, groups=dw_ch * 2, bias=True)
        self.gate = SimpleGate()
        # Simplified Channel Attention: GAP → FC → scale
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_ch, dw_ch, kernel_size=1, padding=0, bias=True),
        )
        self.conv3 = nn.Conv2d(dw_ch, channels, kernel_size=1, padding=0, bias=True)

        ffn_ch = channels * ffn_expand
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_ch * 2, kernel_size=1, padding=0, bias=True)
        self.conv5 = nn.Conv2d(ffn_ch, channels, kernel_size=1, padding=0, bias=True)

        self.dropout = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0.0 else nn.Identity()

        # β and γ: learned scalars, zero-init → identity block at epoch 0
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # --- Path 1: spatial depthwise + channel attention ---
        identity = x
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.gate(x)          # (B, dw_ch, H, W)
        x = x * self.sca(x)       # simplified channel attention (broadcast)
        x = self.conv3(x)
        x = self.dropout(x)
        x = identity + x * self.beta

        # --- Path 2: feed-forward ---
        identity = x
        x = self.norm2(x)
        x = self.conv4(x)
        x = self.gate(x)          # (B, ffn_ch, H, W)
        x = self.conv5(x)
        x = self.dropout(x)
        return identity + x * self.gamma


class NAFDownsample(nn.Module):
    """Strided 2×2 convolution that halves spatial dims and doubles channels.

    Preferred over MaxPool because it is fully learnable and preserves more
    spatial information.

    Args:
        channels: Input channel count.  Output will be ``channels * 2``.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Conv2d(channels, channels * 2, kernel_size=2, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


class NAFUpsample(nn.Module):
    """1×1 conv + PixelShuffle(2) that doubles spatial dims and halves channels.

    Sub-pixel shuffle avoids checkerboard artefacts introduced by transposed
    convolutions and is faster than bilinear interpolation + conv.

    Args:
        channels: Input channel count.  Output will be ``channels // 2``.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels // 2 * 4, kernel_size=1, padding=0, bias=False),
            nn.PixelShuffle(2),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.body(x)


def _match_spatial_size(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor]:
    """Crop both tensors to the minimum shared (H, W).

    Handles off-by-one differences introduced by strided conv + pixel-shuffle
    on odd spatial dimensions.

    Args:
        a: First tensor (B, C, H_a, W_a).
        b: Second tensor (B, C, H_b, W_b).

    Returns:
        Pair of tensors cropped to (min(H_a,H_b), min(W_a,W_b)).
    """
    h = min(a.shape[-2], b.shape[-2])
    w = min(a.shape[-1], b.shape[-1])
    return a[..., :h, :w], b[..., :h, :w]


# ---------------------------------------------------------------------------
# Spatial weight transfer and freeze helpers
# ---------------------------------------------------------------------------


def load_spatial_weights(model: "NAFNetTemporal", path: "Path | str") -> int:
    """Transfer matching spatial-backbone tensors from a spatial checkpoint into a temporal model.

    Key matching is done by name and shape — temporal-only keys
    (``temporal_mix``, ``offset_heads``) are silently skipped.

    Args:
        model: A ``NAFNetTemporal`` instance to update in-place.
        path:  Path to a matching spatial checkpoint saved by training.py.

    Returns:
        Number of parameter tensors transferred.
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    src = ckpt.get("model_state_dict", ckpt)
    dst = model.state_dict()
    transferred = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    dst.update(transferred)
    model.load_state_dict(dst)
    return len(transferred)


def freeze_spatial(model: "NAFNetTemporal") -> None:
    """Freeze the spatial backbone of a temporal model."""
    for module in (model.intro, model.encoders, model.downs, model.middle, model.ups, model.decoders, model.ending):
        for p in module.parameters():
            p.requires_grad_(False)


def unfreeze_spatial(model: "NAFNetTemporal") -> None:
    """Re-enable gradients for all spatial parameters."""
    for module in (model.intro, model.encoders, model.downs, model.middle, model.ups, model.decoders, model.ending):
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
# NAFNet
# ---------------------------------------------------------------------------


class NAFNetConfig:
    """Configuration for NAFNet.

    Args:
        in_channels:      Input image channels (default: 3).
        base_channels:    Feature channels at the first encoder level (default: 32).
        enc_blocks:       NAFBlocks per encoder level (default: (1, 1, 1, 28)).
        middle_blocks:    NAFBlocks in the bottleneck (default: 1).
        dec_blocks:       NAFBlocks per decoder level (default: (1, 1, 1, 1)).
        dw_expand:        Depthwise expand multiplier in NAFBlock (default: 1).
        ffn_expand:       FFN expand multiplier in NAFBlock (default: 2).
        drop_out_rate:    Dropout probability in NAFBlock (default: 0.0).
    """

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 32,
        enc_blocks: tuple[int, ...] = (1, 1, 1, 28),
        middle_blocks: int = 1,
        dec_blocks: tuple[int, ...] = (1, 1, 1, 1),
        dw_expand: int = 1,
        ffn_expand: int = 2,
        drop_out_rate: float = 0.0,
    ) -> None:
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.enc_blocks = enc_blocks
        self.middle_blocks = middle_blocks
        self.dec_blocks = dec_blocks
        self.dw_expand = dw_expand
        self.ffn_expand = ffn_expand
        self.drop_out_rate = drop_out_rate

    def to_dict(self) -> dict:
        """Serialise the config for checkpoints/manifests."""
        return {
            "in_channels": self.in_channels,
            "base_channels": self.base_channels,
            "enc_blocks": list(self.enc_blocks),
            "middle_blocks": self.middle_blocks,
            "dec_blocks": list(self.dec_blocks),
            "dw_expand": self.dw_expand,
            "ffn_expand": self.ffn_expand,
            "drop_out_rate": self.drop_out_rate,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NAFNetConfig":
        """Rebuild a config from ``to_dict()`` output."""
        return cls(
            in_channels=data["in_channels"],
            base_channels=data["base_channels"],
            enc_blocks=tuple(data["enc_blocks"]),
            middle_blocks=data["middle_blocks"],
            dec_blocks=tuple(data["dec_blocks"]),
            dw_expand=data["dw_expand"],
            ffn_expand=data["ffn_expand"],
            drop_out_rate=data["drop_out_rate"],
        )

    @classmethod
    def tiny(cls) -> "NAFNetConfig":
        """Very fast, good for quick experiments."""
        return cls(base_channels=16, enc_blocks=(1, 1, 1), middle_blocks=1, dec_blocks=(1, 1, 1))

    @classmethod
    def small(cls) -> "NAFNetConfig":
        """NAFNet-S equivalent — good balance of speed and quality."""
        return cls(base_channels=32, enc_blocks=(1, 1, 1, 1), middle_blocks=1, dec_blocks=(1, 1, 1, 1))

    @classmethod
    def standard(cls) -> "NAFNetConfig":
        """NAFNet-baseline (32 base, deeper bottleneck)."""
        return cls(base_channels=32, enc_blocks=(1, 1, 1, 28), middle_blocks=1, dec_blocks=(1, 1, 1, 1))

    @classmethod
    def wide(cls) -> "NAFNetConfig":
        """Wider network — more capacity for complex noise."""
        return cls(base_channels=64, enc_blocks=(1, 1, 1, 28), middle_blocks=1, dec_blocks=(1, 1, 1, 1))

    @property
    def num_levels(self) -> int:
        return len(self.enc_blocks)

    @property
    def pad_multiple(self) -> int:
        """Input must be divisible by this for clean skip connections."""
        return 2 ** self.num_levels


class NAFNet(nn.Module):
    """Single-frame spatial denoiser using NAFNet blocks (ECCV 2022).

    Architecture:
        intro conv (3→base)
        ┌── encoder level k: NAFBlock×enc_blocks[k] → NAFDownsample ──┐
        │                                                               │ (skip)
        bottleneck: NAFBlock×middle_blocks                             │
        │                                                               │
        └── decoder level k: NAFUpsample → (+skip) → NAFBlock×dec_blocks[k]
        ending conv (base→3)  [zero-init → predicted residual ≈ 0 at epoch 0]

    Key design properties:
        * LayerNorm2d instead of BatchNorm — batch-independent, HDR-safe
        * SimpleGate instead of ReLU — gated, avoids dead neurons
        * Additive skip connections (+ not concat) — no extra merge conv
        * Strided conv downsample + PixelShuffle upsample — fully learnable

    Args:
        config: NAFNetConfig controlling block counts and channel widths.

    Example::

        model = NAFNet(NAFNetConfig.standard())
        noisy = torch.rand(1, 3, 720, 1280)  # any size — auto-padded
        denoised = model(noisy)              # same shape as input
    """

    def __init__(self, config: Optional["NAFNetConfig"] = None) -> None:
        super().__init__()
        cfg = config or NAFNetConfig.standard()
        self._config = NAFNetConfig.from_dict(cfg.to_dict())
        self._pad_multiple = cfg.pad_multiple

        kw = {"dw_expand": cfg.dw_expand, "ffn_expand": cfg.ffn_expand, "drop_out_rate": cfg.drop_out_rate}

        self.intro = nn.Conv2d(cfg.in_channels, cfg.base_channels, kernel_size=3, padding=1, bias=True)
        self.ending = nn.Conv2d(cfg.base_channels, 3, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        ch = cfg.base_channels
        for num_blocks in cfg.enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch, **kw) for _ in range(num_blocks)]))
            self.downs.append(NAFDownsample(ch))
            ch *= 2

        self.middle = nn.Sequential(*[NAFBlock(ch, **kw) for _ in range(cfg.middle_blocks)])

        for num_blocks in cfg.dec_blocks:
            self.ups.append(NAFUpsample(ch))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch, **kw) for _ in range(num_blocks)]))

    def forward(self, x: Tensor) -> Tensor:
        """Denoise *x*.

        Args:
            x: (B, C, H, W) noisy image / render (float32, any range).
               C must match ``config.in_channels``.

        Returns:
            Denoised beauty (B, 3, H, W), same spatial size as input.
            Full float range preserved — no clamping.
        """
        # Replace any NaN/Inf before processing (common in raw EXR renders)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        beauty = x[:, :3]
        _, _, h, w = beauty.shape

        x, padding = _pad_to_multiple(x, self._pad_multiple)

        x = self.intro(x)

        skips: list[Tensor] = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        x = self.middle(x)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            x = up(x)
            x, skip = _match_spatial_size(x, skip)
            x = decoder(x + skip)  # additive skip — no channel concat needed

        residual = _unpad(self.ending(x), padding)
        return beauty + residual


# ---------------------------------------------------------------------------
# NAFNetTemporal
# ---------------------------------------------------------------------------


class NAFNetTemporal(nn.Module):
    """Multi-frame temporal denoiser using NAFNet blocks (ECCV 2022).

    Architecture:
        1. Shared ``intro`` conv + per-level NAFBlock encoder applied to all
           T frames as a batch (B×T, C, H, W).
        2. Per-level temporal mixing: mean of neighbour skip features
           concatenated with the reference skip, projected back via 1×1 conv.
        3. Optional learned warp (``use_warp=True``): per-level offset heads —
           Conv2d(2C, 2, 1) → bilinear grid_sample.
        4. NAFBlock bottleneck on the reference frame's deepest features only.
        5. Shared NAFBlock decoder with **additive** temporal skip connections
           (NAFNet style — no channel concat, no extra merge conv).
        6. Zero-init ``ending`` conv — predicted residual ≈ 0 at epoch 0.

    The spatial backbone (intro / encoders / downs / middle / ups / decoders /
    ending) shares identical key names with ``NAFNet``, so weights
    transfer directly via ``load_spatial_weights``.

    Args:
        config:     NAFNetConfig controlling block counts and base channels.
        num_frames: Temporal window size (default: 3; reference = centre frame).
        use_warp:   Enable per-level learned warp for real video with motion.

    Example::

        model = NAFNetTemporal()
        clip = torch.rand(1, 5, 3, 256, 256)
        denoised = model(clip)   # (B, 3, H, W) — denoised centre frame

        # Two-stage training from a NAFNet checkpoint:
        model = NAFNetTemporal.from_spatial("checkpoints/spatial_naf/best.pth")
        freeze_spatial(model)    # train temporal_mix only in stage 2
    """

    @classmethod
    def from_spatial(
        cls,
        path: "Path | str",
        config: Optional[NAFNetConfig] = None,
        num_frames: int = 3,
        use_warp: bool = False,
    ) -> "NAFNetTemporal":
        """Create a NAFNetTemporal pre-loaded with weights from a NAFNet checkpoint.

        Temporal components (``temporal_mix``, ``offset_heads``) keep their
        identity initialisations; the spatial backbone is warm-started.

        Args:
            path:       Path to a NAFNet checkpoint produced by training.py.
            config:     NAFNetConfig to use (must match the training config).
            num_frames: Temporal window (default: 3).
            use_warp:   Enable warp heads (default: False).

        Returns:
            A NAFNetTemporal instance with spatial weights transferred.
        """
        model = cls(config, num_frames=num_frames, use_warp=use_warp)
        load_spatial_weights(model, path)
        return model

    def __init__(
        self,
        config: Optional[NAFNetConfig] = None,
        num_frames: int = 3,
        use_warp: bool = False,
    ) -> None:
        super().__init__()
        validate_temporal_num_frames(num_frames)
        cfg = config or NAFNetConfig.standard()
        self._config = NAFNetConfig.from_dict(cfg.to_dict())
        self._pad_multiple = cfg.pad_multiple
        self._num_frames = num_frames
        self._ref_idx = num_frames // 2
        self._use_warp = use_warp

        kw = {
            "dw_expand": cfg.dw_expand,
            "ffn_expand": cfg.ffn_expand,
            "drop_out_rate": cfg.drop_out_rate,
        }

        # --- Shared spatial backbone (mirrors NAFNet key-for-key) ---
        self.intro = nn.Conv2d(cfg.in_channels, cfg.base_channels, kernel_size=3, padding=1, bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        enc_channels: list[int] = []
        ch = cfg.base_channels
        for num_blocks in cfg.enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch, **kw) for _ in range(num_blocks)]))
            enc_channels.append(ch)
            self.downs.append(NAFDownsample(ch))
            ch *= 2

        self._enc_channels = enc_channels

        self.middle = nn.Sequential(*[NAFBlock(ch, **kw) for _ in range(cfg.middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for num_blocks in cfg.dec_blocks:
            self.ups.append(NAFUpsample(ch))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch, **kw) for _ in range(num_blocks)]))

        # Zero-init: predicted residual ≈ 0 at epoch 0.
        self.ending = nn.Conv2d(cfg.base_channels, 3, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

        # --- Per-level offset heads (use_warp=True only) ---
        # Zero-init: identity displacement at epoch 0.
        if use_warp:
            self.offset_heads = nn.ModuleList()
            for skip_ch in enc_channels:
                head = nn.Conv2d(2 * skip_ch, 2, kernel_size=1, bias=True)
                nn.init.zeros_(head.weight)
                nn.init.zeros_(head.bias)
                self.offset_heads.append(head)

        # --- Per-level temporal mixing ---
        # Input: cat([ref_skip, neigh_mean]) → skip_ch (1×1 conv).
        # Identity init: ref slice is eye, neighbour slice is zero.
        self.temporal_mix = nn.ModuleList()
        for skip_ch in enc_channels:
            conv = nn.Conv2d(2 * skip_ch, skip_ch, kernel_size=1, bias=True)
            nn.init.zeros_(conv.weight)
            nn.init.zeros_(conv.bias)
            for c in range(skip_ch):
                conv.weight.data[c, c, 0, 0] = 1.0
            self.temporal_mix.append(conv)

    def forward(self, clip: Tensor) -> Tensor:
        """Denoise the centre frame of *clip*.

        Args:
            clip: (B, T, C, H, W) noisy frames (float32, any range).

        Returns:
            Denoised centre frame (B, 3, H, W), same range as input.
        """
        b, t, c, h, w = clip.shape
        assert t == self._num_frames, f"Expected {self._num_frames} frames, got {t}"

        # Replace NaN/Inf (can appear in render EXR data)
        clip = torch.nan_to_num(clip, nan=0.0, posinf=0.0, neginf=0.0)

        ref_input = clip[:, self._ref_idx]   # (B, C, H, W)
        beauty = ref_input[:, :3]

        # Pad all frames identically
        frames_padded: list[Tensor] = []
        for i in range(t):
            padded, padding = _pad_to_multiple(clip[:, i], self._pad_multiple)
            frames_padded.append(padded)

        # --- Encode all frames as one batch ---
        stacked = torch.stack(frames_padded, dim=1)          # (B, T, C, H_pad, W_pad)
        x = stacked.view(b * t, c, *stacked.shape[-2:])      # (B*T, C, H_pad, W_pad)

        x = self.intro(x)   # (B*T, base_ch, H_pad, W_pad)

        all_skips: list[list[Tensor]] = []  # [level][frame_idx] → (B, ch, H', W')

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            _, cp, hp, wp = x.shape
            skip_bt = x.view(b, t, cp, hp, wp)
            all_skips.append([skip_bt[:, i] for i in range(t)])
            x = down(x)

        # --- Bottleneck on reference frame only ---
        _, cp, hp, wp = x.shape
        ref_deep = x.view(b, t, cp, hp, wp)[:, self._ref_idx]
        x = self.middle(ref_deep)

        # --- Per-level temporal mixing (with optional learned warp) ---
        fused_skips: list[Tensor] = []
        for level, mix in enumerate(self.temporal_mix):
            ref_feat = all_skips[level][self._ref_idx]
            neigh_feats = [all_skips[level][i] for i in range(t) if i != self._ref_idx]

            if self._use_warp:
                offset_head = self.offset_heads[level]
                warped: list[Tensor] = []
                for nf in neigh_feats:
                    offset = offset_head(torch.cat([ref_feat, nf], dim=1))
                    warped.append(_warp(nf, offset))
                neigh_feats = warped

            neigh_mean = torch.stack(neigh_feats, dim=0).mean(dim=0)
            fused_skips.append(mix(torch.cat([ref_feat, neigh_mean], dim=1)))

        # --- Decoder with additive skips (NAFNet style) ---
        for decoder, up, skip in zip(self.decoders, self.ups, reversed(fused_skips)):
            x = up(x)
            x, skip = _match_spatial_size(x, skip)
            x = decoder(x + skip)

        residual = _unpad(self.ending(x), padding)
        return beauty + residual


def validate_temporal_num_frames(num_frames: int) -> None:
    """Require an odd temporal window with a defined centre frame."""
    if num_frames < 3 or num_frames % 2 == 0:
        raise ValueError("Temporal window size must be an odd integer >= 3.")


def get_model_metadata(model: nn.Module) -> dict:
    """Extract enough metadata to rebuild *model* from a checkpoint."""
    if isinstance(model, NAFNetTemporal):
        return {
            "model_type": "temporal",
            "naf_config": model._config.to_dict(),
            "num_frames": model._num_frames,
            "use_warp": model._use_warp,
        }
    if isinstance(model, NAFNet):
        return {
            "model_type": "spatial",
            "naf_config": model._config.to_dict(),
        }
    raise TypeError(f"Unsupported model type for checkpoint metadata: {type(model)!r}")


def build_model_from_metadata(metadata: dict) -> nn.Module:
    """Rebuild a NAFNet model from checkpoint metadata."""
    cfg = NAFNetConfig.from_dict(metadata["naf_config"])
    model_type = metadata["model_type"]
    if model_type == "spatial":
        return NAFNet(cfg)
    if model_type == "temporal":
        return NAFNetTemporal(
            cfg,
            num_frames=metadata["num_frames"],
            use_warp=metadata.get("use_warp", False),
        )
    raise ValueError(f"Unsupported model_type in checkpoint metadata: {model_type!r}")
