"""NAFNetCascade — two-stage cascade denoiser (future experiment).

Architecture
────────────
    clip (B, T, C, H, W)
        │
        ▼  spatial_stage (shared NAFNet, run T times)
    denoised frames (B, T, 3, H, W)
        │
        ▼  cat([denoised_ref, denoised_others..., raw_center], dim=1)
    (B, (T+1)×3, H, W)
        │
        ▼  temporal_stage (NAFNet with wider intro)
    output (B, 3, H, W)

Identity at init (after loading stage-1 spatial weights)
─────────────────────────────────────────────────────────
    temporal_stage.ending is zero-init.
    Its forward: beauty = input[:, :3] = denoised_ref = spatial(noisy_ref).
    With ending=0, temporal_stage outputs denoised_ref.
    So cascade(clip_identical) ≈ NAFNet(center_frame).  ✓

Training stages
───────────────
    Stage 2: load spatial_stage from stage-1 checkpoint, freeze it,
             train temporal_stage only.
    Stage 3: unfreeze all, joint fine-tune at lower LR.

Use scripts/train_cascade.sh to run this pipeline.
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from models import NAFNet, NAFNetConfig, validate_temporal_num_frames


def _default_temporal_config(num_frames: int, temporal_base: int = 32) -> NAFNetConfig:
    """NAFNet config for the cascade temporal stage.

    Matches exp_053 CascadeNAFTemporalDenoiser exactly:
        base=32  (exp_053 passed base=64 but computed temporal_base = base//2 = 32)
        middle_blocks=2   (exp_053 middle_blk_num=2)
        dw_expand=1       (exp_053 NAFBlock defaults, not the spatial exp048 dw_expand=2)
    """
    return NAFNetConfig(
        in_channels=(num_frames + 1) * 3,  # T denoised frames + raw centre
        base_channels=temporal_base,
        enc_blocks=(1, 1, 1),
        middle_blocks=2,
        dec_blocks=(1, 1, 1),
        dw_expand=1,
        ffn_expand=2,
    )


class NAFNetCascade(nn.Module):
    """Two-stage cascade denoiser: shared spatial NAFNet → temporal NAFNet.

    The spatial stage runs on every frame independently.  The temporal stage
    then fuses the denoised frames (plus the raw centre reference) and
    outputs the final refined centre frame.

    Exposes ``_num_frames`` and ``_ref_idx`` so that ``training.py``'s loop
    can treat this model identically to a plain ``NAFNetTemporal``.

    Args:
        spatial_config:  NAFNetConfig for the shared per-frame denoiser.
        temporal_config: NAFNetConfig for the fusion network.  Defaults to a
                         3-level NAFNet(base=temporal_base, in_ch=(T+1)×3).
        num_frames:      Temporal window size (must be odd ≥ 3).
        temporal_base:   base_channels for the default temporal config.
                         Ignored when ``temporal_config`` is supplied.
    """

    def __init__(
        self,
        spatial_config: Optional[NAFNetConfig] = None,
        temporal_config: Optional[NAFNetConfig] = None,
        num_frames: int = 3,
        temporal_base: int = 32,
    ) -> None:
        super().__init__()
        validate_temporal_num_frames(num_frames)

        sp_cfg = spatial_config or NAFNetConfig.standard()
        self._spatial_config = NAFNetConfig.from_dict(sp_cfg.to_dict())
        self._num_frames = num_frames
        self._ref_idx = num_frames // 2
        self._spatial_frozen = False

        self.spatial_stage = NAFNet(sp_cfg)

        t_cfg = temporal_config or _default_temporal_config(num_frames, temporal_base)
        self._temporal_config = NAFNetConfig.from_dict(t_cfg.to_dict())
        self.temporal_stage = NAFNet(t_cfg)

    def forward(self, clip: Tensor) -> Tensor:
        """Denoise the centre frame of *clip*.

        Args:
            clip: (B, T, C, H, W) noisy frames (float32, any range).

        Returns:
            Denoised centre frame (B, 3, H, W).
        """
        b, t, c, h, w = clip.shape
        assert t == self._num_frames, f"Expected {self._num_frames} frames, got {t}"

        clip = torch.nan_to_num(clip, nan=0.0, posinf=0.0, neginf=0.0)

        # Denoise all frames with the shared spatial stage
        frames_flat = clip.view(b * t, c, h, w)
        denoised_flat = self.spatial_stage(frames_flat)   # (B*T, 3, H, W)
        denoised = denoised_flat.view(b, t, 3, h, w)

        # Arrange input for the temporal stage:
        #   [denoised_ref(3), denoised_others((T-1)×3), raw_center(3)]
        # denoised_ref is first → NAFNet.beauty = denoised_ref at epoch 0.
        ref_denoised = denoised[:, self._ref_idx]
        other_denoised = [denoised[:, i] for i in range(t) if i != self._ref_idx]
        raw_center = clip[:, self._ref_idx, :3]

        temporal_input = torch.cat([ref_denoised] + other_denoised + [raw_center], dim=1)
        return self.temporal_stage(temporal_input)

    def freeze_spatial_stage(self) -> None:
        """Freeze spatial_stage — only temporal_stage receives gradients."""
        for p in self.spatial_stage.parameters():
            p.requires_grad_(False)
        self.spatial_stage.eval()
        self._spatial_frozen = True

        _orig_train = self.__class__.train

        def _train_with_frozen(self: "NAFNetCascade", mode: bool = True):
            _orig_train(self, mode)
            if getattr(self, "_spatial_frozen", False):
                self.spatial_stage.eval()
            return self

        self.__class__.train = _train_with_frozen  # type: ignore[method-assign]

    def unfreeze_spatial_stage(self) -> None:
        """Re-enable gradients for all spatial_stage parameters."""
        for p in self.spatial_stage.parameters():
            p.requires_grad_(True)
        self._spatial_frozen = False

    def load_spatial_stage(self, path: "Path | str") -> int:
        """Transfer matching weights from a NAFNet checkpoint into spatial_stage.

        Args:
            path: Path to a stage-1 NAFNet checkpoint saved by training.py.

        Returns:
            Number of parameter tensors transferred.
        """
        ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
        src = ckpt.get("model_state_dict", ckpt)
        dst = self.spatial_stage.state_dict()
        transferred = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
        dst.update(transferred)
        self.spatial_stage.load_state_dict(dst)
        return len(transferred)


def get_cascade_metadata(model: NAFNetCascade) -> dict:
    """Extract checkpoint metadata for a NAFNetCascade model."""
    return {
        "model_type": "cascade",
        "spatial_config": model._spatial_config.to_dict(),
        "temporal_config": model._temporal_config.to_dict(),
        "num_frames": model._num_frames,
    }


def build_cascade_from_metadata(metadata: dict) -> NAFNetCascade:
    """Rebuild a NAFNetCascade from checkpoint metadata."""
    return NAFNetCascade(
        spatial_config=NAFNetConfig.from_dict(metadata["spatial_config"]),
        temporal_config=NAFNetConfig.from_dict(metadata["temporal_config"]),
        num_frames=metadata["num_frames"],
    )
