"""Loss functions for denoiser training.

The primary loss is noise-weighted L1: pixels with higher estimated noise
standard deviation contribute less to the gradient, preventing the loss from
being dominated by bright, noisy regions and encouraging uniform fidelity
across the dynamic range.

    loss = mean( |output - clean| / (sigma_map + epsilon) )

This is equivalent to maximum-likelihood estimation under a Laplace noise
model with location-dependent scale.
"""

import torch
import torch.nn as nn
from torch import Tensor


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
        self, output: Tensor, clean: Tensor, sigma_map: Tensor
    ) -> Tensor:
        """Compute noise-weighted L1 loss.

        Args:
            output: Predicted denoised image (B, C, H, W), in [0, 1].
            clean: Ground-truth clean image (B, C, H, W), in [0, 1].
            sigma_map: Per-pixel noise std estimate (B, C, H, W), in [0, 1].

        Returns:
            Scalar loss tensor.
        """
        weights = 1.0 / (sigma_map + self._epsilon)
        elementwise = weights * torch.abs(output - clean)
        if self._reduction == "mean":
            return elementwise.mean()
        return elementwise.sum()


class L1Loss(nn.Module):
    """Plain L1 loss (no noise weighting).

    Useful as a baseline or when sigma_map is not available.
    """

    def forward(self, output: Tensor, clean: Tensor) -> Tensor:
        return torch.abs(output - clean).mean()
