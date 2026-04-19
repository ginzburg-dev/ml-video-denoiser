"""NAFNetRefiner — lightweight post-decode refinement stage (stage 4).

Takes the coarse output of a trained NAFNetTemporal (A+B) and the raw
centre frame, and predicts a small residual correction.

    coarse  (B, 3, H, W)  ─┐
                             ├── cat(6ch) ──► tiny NAFNet ──► coarse + Δ
    raw_center (B, 3, H, W) ─┘

Identity at epoch 0: NAFNet.ending is zero-init, so Δ = 0 and the refiner
passes coarse through unchanged — the A+B warm-start is preserved.

NAFNetRefinedTemporal wraps a trained NAFNetTemporal base together with this
refiner into a single nn.Module with the same forward signature as the base:
    forward(clip: (B, T, C, H, W)) → (B, 3, H, W)
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from models import NAFNet, NAFNetConfig, NAFNetTemporal, get_model_metadata


def _refiner_net_config(base_channels: int = 16) -> NAFNetConfig:
    """Compact 2-level NAFNetConfig for the refiner."""
    return NAFNetConfig(
        in_channels=6,        # cat([coarse(3), raw_center(3)])
        base_channels=base_channels,
        enc_blocks=(1, 1),
        middle_blocks=1,
        dec_blocks=(1, 1),
        dw_expand=2,
        ffn_expand=2,
    )


class NAFNetRefiner(nn.Module):
    """Tiny NAFNet that corrects a coarse denoised output using raw pixels.

    Input:  cat([coarse(3), raw_center(3)], dim=1) → (B, 6, H, W).
    Output: (B, 3, H, W) — coarse + learned residual.

    The first 3 channels of the input are ``coarse``, so NAFNet's internal
    ``beauty = x[:, :3]`` equals the coarse output.  At epoch 0 the ending
    conv is zero-init → output equals coarse exactly.
    """

    def __init__(self, base_channels: int = 16) -> None:
        super().__init__()
        self._config = _refiner_net_config(base_channels)
        self.net = NAFNet(self._config)

    def forward(self, coarse: Tensor, raw_center: Tensor) -> Tensor:
        """Refine *coarse* using raw pixel reference *raw_center*.

        Args:
            coarse:     (B, 3, H, W) output of the base temporal model.
            raw_center: (B, 3, H, W) raw (noisy) centre frame.

        Returns:
            (B, 3, H, W) refined output.
        """
        return self.net(torch.cat([coarse, raw_center], dim=1))


class NAFNetRefinedTemporal(nn.Module):
    """Composed: NAFNetTemporal (A+B) base + NAFNetRefiner.

    Forward:
        1. Run base NAFNetTemporal on clip → coarse  (B, 3, H, W)
        2. Run refiner(coarse, raw_center)  → refined (B, 3, H, W)

    Exposes ``_num_frames`` and ``_ref_idx`` so the training loop can treat
    this model identically to a plain NAFNetTemporal.

    Args:
        base_model:           Trained (or randomly initialised) NAFNetTemporal.
        refiner_base_channels: base_channels for the refiner NAFNet (default: 16).
    """

    def __init__(
        self,
        base_model: NAFNetTemporal,
        refiner_base_channels: int = 16,
    ) -> None:
        super().__init__()
        self.base = base_model
        self.refiner = NAFNetRefiner(base_channels=refiner_base_channels)
        self._base_frozen = False

    @property
    def _num_frames(self) -> int:
        return self.base._num_frames

    @property
    def _ref_idx(self) -> int:
        return self.base._ref_idx

    def forward(self, clip: Tensor) -> Tensor:
        """Denoise and refine the centre frame of *clip*.

        Args:
            clip: (B, T, C, H, W) noisy frames (float32, any range).

        Returns:
            Refined centre frame (B, 3, H, W).
        """
        coarse = self.base(clip)
        raw_center = clip[:, self._ref_idx, :3]
        # Crop raw_center to match coarse spatial size (NAFNet may unpad)
        raw_center = raw_center[..., : coarse.shape[-2], : coarse.shape[-1]]
        return self.refiner(coarse, raw_center)

    def freeze_base(self) -> None:
        """Freeze all base parameters — only the refiner receives gradients."""
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.base.eval()
        self._base_frozen = True

        _orig_train = self.__class__.train

        def _train_with_frozen(self: "NAFNetRefinedTemporal", mode: bool = True):
            _orig_train(self, mode)
            if getattr(self, "_base_frozen", False):
                self.base.eval()
            return self

        self.__class__.train = _train_with_frozen  # type: ignore[method-assign]

    def unfreeze_base(self) -> None:
        """Re-enable gradients for all base parameters."""
        for p in self.base.parameters():
            p.requires_grad_(True)
        self._base_frozen = False


def load_base_weights(model: NAFNetRefinedTemporal, path: "Path | str") -> int:
    """Transfer a stage-3 NAFNetTemporal checkpoint into ``model.base``.

    Keys from the checkpoint that match ``model.base`` by name and shape are
    transferred.  Refiner keys are untouched.

    Args:
        model: A ``NAFNetRefinedTemporal`` to update in-place.
        path:  Path to a NAFNetTemporal checkpoint saved by training.py.

    Returns:
        Number of parameter tensors transferred.
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    src = ckpt.get("model_state_dict", ckpt)
    dst = model.base.state_dict()
    transferred = {k: v for k, v in src.items() if k in dst and dst[k].shape == v.shape}
    dst.update(transferred)
    model.base.load_state_dict(dst)
    return len(transferred)


def get_refined_temporal_metadata(model: NAFNetRefinedTemporal) -> dict:
    """Extract checkpoint metadata for a NAFNetRefinedTemporal model."""
    return {
        "model_type": "refined_temporal",
        "base_metadata": get_model_metadata(model.base),
        "refiner_base_channels": model.refiner._config.base_channels,
    }


def build_refined_temporal_from_metadata(metadata: dict) -> NAFNetRefinedTemporal:
    """Rebuild a NAFNetRefinedTemporal from checkpoint metadata."""
    from models import build_model_from_metadata

    base = build_model_from_metadata(metadata["base_metadata"])
    if not isinstance(base, NAFNetTemporal):
        raise ValueError(
            f"refined_temporal base must be NAFNetTemporal, got {type(base)!r}"
        )
    return NAFNetRefinedTemporal(
        base_model=base,
        refiner_base_channels=metadata.get("refiner_base_channels", 16),
    )
