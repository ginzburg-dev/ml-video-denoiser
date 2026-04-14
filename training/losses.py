"""Loss functions for denoiser training.

Available objectives:

``noise-weighted-l1``
    The default paired-data loss. Pixels with higher estimated noise standard
    deviation contribute less to the gradient:

        mean(|output - clean| / (sigma_map + epsilon))

``l1``
    Plain L1 reconstruction loss in linear space.

``log-l1``
    L1 after ``log1p`` compression. This tends to emphasise low / mid
    luminance differences and can produce a visually smoother HDR result.
"""

import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class NoiseWeightedL1Loss(nn.Module):
    """L1 loss weighted inversely by the per-pixel noise standard deviation.

    Lower sigma_map values → higher weight → the loss focuses on regions
    where the noise level is low and reconstruction accuracy should be high.

    Args:
        epsilon: Small constant added to sigma_map to prevent division by zero
            and to bound the maximum weight for near-zero sigma.  A value of
            ~0.01 (in normalised [0, 1] units) corresponds to ~2.5 DN in an
            8-bit image.
        reduction: ``"mean"`` averages over all elements; ``"sum"`` sums them.
    """

    def __init__(self, epsilon: float = 0.01, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in ("mean", "sum"):
            raise ValueError(f"Unsupported reduction: {reduction!r}")
        self._epsilon = epsilon
        self._reduction = reduction

    def forward(
        self,
        output: Tensor,
        clean: Tensor,
        sigma_map: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute noise-weighted L1 loss.

        Args:
            output: Predicted denoised image (B, C, H, W), in [0, 1].
            clean: Ground-truth clean image (B, C, H, W), in [0, 1].
            sigma_map: Per-pixel noise std estimate (B, C, H, W), in [0, 1].

        Returns:
            Scalar loss tensor.
        """
        if sigma_map is None:
            raise ValueError("NoiseWeightedL1Loss requires sigma_map.")
        weights = 1.0 / (sigma_map + self._epsilon)
        elementwise = weights * torch.abs(output - clean)
        if self._reduction == "mean":
            return elementwise.mean()
        return elementwise.sum()


class L1Loss(nn.Module):
    """Plain L1 loss (no noise weighting).

    Useful as a baseline or when sigma_map is not available.
    """

    def forward(
        self,
        output: Tensor,
        clean: Tensor,
        sigma_map: Optional[Tensor] = None,
    ) -> Tensor:
        return torch.abs(output - clean).mean()


class LogL1Loss(nn.Module):
    """Plain L1 in log-compressed space.

    Negative values are clamped to zero before ``log1p`` so the transform stays
    well-defined for HDR inputs with occasional undershoot.
    """

    def forward(
        self,
        output: Tensor,
        clean: Tensor,
        sigma_map: Optional[Tensor] = None,
    ) -> Tensor:
        output_log = torch.log1p(output.clamp_min(0.0))
        clean_log = torch.log1p(clean.clamp_min(0.0))
        return torch.abs(output_log - clean_log).mean()
