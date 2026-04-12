"""Pluggable noise synthesis for training the denoiser.

All generators follow the NoiseGenerator protocol: they accept a clean image
tensor and return (noisy, clean, sigma_map), where sigma_map encodes the local
noise standard deviation — used for pixel-weighted loss during training.

Usage:
    generator = MixedNoiseGenerator.default()
    noisy, clean, sigma_map = generator(clean_patch)

    # Real noise injection (requires a patch pool from noise_profiler.py):
    generator = RealNoiseInjectionGenerator(patch_pool="pools/a7iii_iso3200.npz")
"""

import json
import random
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class NoiseGenerator(Protocol):
    """Protocol for noise synthesis modules.

    Every implementation must be callable and return a 3-tuple of tensors,
    all with the same shape as *clean*:

        (noisy, clean, sigma_map)

    where sigma_map[c, h, w] is the local noise standard deviation estimate
    at channel c, pixel (h, w).  Values are float32; no clamping applied.
    """

    def __call__(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]: ...


# ---------------------------------------------------------------------------
# Gaussian
# ---------------------------------------------------------------------------


class GaussianNoiseGenerator:
    """Additive white Gaussian noise (AWGN).

    Draws sigma uniformly per sample from [sigma_min, sigma_max].  The
    resulting sigma_map is spatially uniform (same value everywhere).

    Args:
        sigma_min: Minimum noise standard deviation (normalised, e.g. 0).
        sigma_max: Maximum noise standard deviation (normalised, e.g. 75/255).
    """

    def __init__(self, sigma_min: float = 0.0, sigma_max: float = 75.0 / 255.0) -> None:
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply AWGN to *clean* and return (noisy, clean, sigma_map)."""
        sigma = random.uniform(self.sigma_min, self.sigma_max)
        noise = torch.randn_like(clean) * sigma
        noisy = clean + noise
        sigma_map = torch.full_like(clean, sigma)
        return noisy, clean, sigma_map


# ---------------------------------------------------------------------------
# Poisson-Gaussian
# ---------------------------------------------------------------------------


class PoissonGaussianNoiseGenerator:
    """Heteroscedastic Poisson-Gaussian noise model.

    Models real sensor noise as the sum of shot noise (Poisson, signal-
    dependent) and read noise (Gaussian, signal-independent):

        noisy = Poisson(clean / K) * K + N(0, sigma_r^2)

    The noise standard deviation varies spatially:

        sigma_map[c, h, w] = sqrt(K * clean[c, h, w] + sigma_r^2)

    This model is most accurate when *clean* represents linear-light values
    (i.e. RAW domain before gamma / tone-mapping).

    Args:
        gain_range: (min, max) range for the Poisson gain K, sampled per call.
            K ≈ electrons/DN.  Typical values: 0.001–0.05.
        read_sigma_range: (min, max) range for read noise std, sampled per call.
            Typical values: 0.0–0.02 (normalised).
    """

    def __init__(
        self,
        gain_range: tuple[float, float] = (0.001, 0.05),
        read_sigma_range: tuple[float, float] = (0.0, 0.02),
    ) -> None:
        self.gain_range = gain_range
        self.read_sigma_range = read_sigma_range

    def __call__(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply Poisson-Gaussian noise and return (noisy, clean, sigma_map)."""
        k = random.uniform(*self.gain_range)
        sigma_r = random.uniform(*self.read_sigma_range)

        # Shot noise: Poisson(lambda=clean/K) * K  (zero-mean after subtraction)
        shot = torch.poisson(clean / k) * k - clean
        read = torch.randn_like(clean) * sigma_r

        noisy = clean + shot + read
        sigma_map = (k * clean + sigma_r**2).sqrt()
        return noisy, clean, sigma_map


# ---------------------------------------------------------------------------
# Real noise injection
# ---------------------------------------------------------------------------


class RealNoiseInjectionGenerator:
    """Injects authentic noise patches sampled from dark-frame recordings.

    Instead of synthesising noise mathematically this generator samples actual
    zero-mean noise residuals captured with a real camera (lens cap on).
    This preserves authentic noise structure that parametric models miss:
    banding, hot pixels, fixed-pattern remnants, chroma correlation, and
    sensor anisotropy.

    The patch pool is produced by noise_profiler.py:

        uv run python noise_profiler.py \\
            --dark dark_sequence/*.png \\
            --save-patches pools/camera_iso3200.npz

    Args:
        patch_pool: Path to an .npz file produced by noise_profiler.py.
            Must contain an array keyed ``"residuals"`` with shape
            (N, H, W, C) of type float32, zero-mean per temporal axis.
        sigma_window: Kernel size for the local sliding-window std estimate
            used to build sigma_map.  Larger values are smoother.
    """

    def __init__(self, patch_pool: str, sigma_window: int = 9) -> None:
        data = np.load(patch_pool)
        # residuals: (N, H, W, C) → convert to (N, C, H, W) for easy cropping
        residuals = data["residuals"].astype(np.float32)
        if residuals.ndim == 4 and residuals.shape[-1] in (1, 3, 4):
            residuals = residuals.transpose(0, 3, 1, 2)
        self._pool = torch.from_numpy(residuals)  # (N, C, H, W)
        self._sigma_window = sigma_window

    def __call__(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Sample a real noise patch, add to *clean*, return (noisy, clean, sigma_map)."""
        _, c, h, w = clean.shape if clean.ndim == 4 else (1, *clean.shape)
        n, pc, ph, pw = self._pool.shape

        if ph < h or pw < w:
            raise ValueError(
                f"Pool patch size ({ph}×{pw}) is smaller than clean patch ({h}×{w}). "
                "Re-extract a larger patch pool."
            )

        # Random pool index and random spatial crop
        idx = random.randrange(n)
        top = random.randint(0, ph - h)
        left = random.randint(0, pw - w)
        noise = self._pool[idx, :, top : top + h, left : left + w]

        # Match channel count (e.g. pool is 3-ch, clean could be 1-ch)
        if pc != c:
            noise = noise[:c] if pc > c else noise.mean(0, keepdim=True).expand(c, -1, -1)

        noise = noise.to(clean.device)
        noisy = clean + noise
        sigma_map = self._local_std(noise.unsqueeze(0)).squeeze(0)
        return noisy, clean, sigma_map

    def _local_std(self, x: Tensor) -> Tensor:
        """Estimate local noise std via sliding-window variance (unfold trick)."""
        k = self._sigma_window
        pad = k // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        patches = x_pad.unfold(2, k, 1).unfold(3, k, 1)  # (B, C, H, W, k, k)
        var = patches.var(dim=(-2, -1), unbiased=False)
        return var.sqrt().squeeze(0)  # (C, H, W)


# ---------------------------------------------------------------------------
# Calibrated parametric real noise
# ---------------------------------------------------------------------------


class RealRAWNoiseGenerator:
    """Calibrated Poisson-Gaussian noise from a noise_profiler.py JSON profile.

    Unlike PoissonGaussianNoiseGenerator, the K and sigma_r parameters here
    are fitted to a specific camera/ISO combination rather than sampled from
    a generic prior.  Multiple ISO profiles can be included in one JSON; the
    generator randomly selects a profile per call for generalization.

    Profile JSON schema (produced by noise_profiler.py):

        {
          "camera": "Sony A7 III",
          "iso_profiles": {
            "iso_800":  {"K": 0.003, "sigma_r": 0.002},
            "iso_3200": {"K": 0.012, "sigma_r": 0.006},
            "iso_12800": {"K": 0.048, "sigma_r": 0.014}
          }
        }

    Args:
        profile_json: Path to the noise profile JSON.
    """

    def __init__(self, profile_json: str) -> None:
        with open(profile_json) as f:
            data = json.load(f)
        profiles = data.get("iso_profiles", {})
        if not profiles:
            raise ValueError(f"No 'iso_profiles' found in {profile_json}")
        self._profiles = list(profiles.values())

    def __call__(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply calibrated Poisson-Gaussian noise; return (noisy, clean, sigma_map)."""
        profile = random.choice(self._profiles)
        k: float = profile["K"]
        sigma_r: float = profile["sigma_r"]

        shot = torch.poisson(clean / k) * k - clean
        read = torch.randn_like(clean) * sigma_r
        noisy = clean + shot + read
        sigma_map = (k * clean + sigma_r**2).sqrt()
        return noisy, clean, sigma_map


# ---------------------------------------------------------------------------
# Mixed
# ---------------------------------------------------------------------------


class MixedNoiseGenerator:
    """Randomly selects among multiple noise generators per sample.

    Args:
        generators: List of noise generator instances.
        weights: Sampling probabilities (must sum to 1.0).  If None, uniform.

    Example::

        gen = MixedNoiseGenerator(
            generators=[
                GaussianNoiseGenerator(0, 75 / 255),
                PoissonGaussianNoiseGenerator(),
            ],
            weights=[0.6, 0.4],
        )
        noisy, clean, sigma_map = gen(clean_patch)
    """

    def __init__(
        self,
        generators: list[NoiseGenerator],
        weights: Optional[list[float]] = None,
    ) -> None:
        if not generators:
            raise ValueError("generators list must not be empty")
        if weights is not None and len(weights) != len(generators):
            raise ValueError("len(weights) must equal len(generators)")
        self._generators = generators
        self._weights = weights

    @classmethod
    def default(
        cls,
        patch_pool: Optional[str] = None,
        profile_json: Optional[str] = None,
    ) -> "MixedNoiseGenerator":
        """Build the recommended four-way mixed generator.

        RealNoiseInjection and RealRAW are included only when the
        corresponding files are provided; their weight is redistributed to
        the synthetic generators otherwise.

        Args:
            patch_pool: Path to .npz patch pool (noise_profiler.py output).
            profile_json: Path to camera noise profile JSON.

        Returns:
            A MixedNoiseGenerator with appropriate generators and weights.
        """
        generators: list[NoiseGenerator] = [
            GaussianNoiseGenerator(0.0, 75.0 / 255.0),
            PoissonGaussianNoiseGenerator(),
        ]
        weights = [0.30, 0.30]
        remaining = 0.40

        if patch_pool is not None:
            generators.append(RealNoiseInjectionGenerator(patch_pool))
            weights.append(0.25)
            remaining -= 0.25

        if profile_json is not None:
            generators.append(RealRAWNoiseGenerator(profile_json))
            weights.append(0.15)
            remaining -= 0.15

        # Redistribute any unclaimed weight to the first two generators
        weights[0] += remaining / 2
        weights[1] += remaining / 2

        return cls(generators=generators, weights=weights)

    def __call__(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Delegate to a randomly selected generator."""
        (gen,) = random.choices(self._generators, weights=self._weights, k=1)
        return gen(clean)
