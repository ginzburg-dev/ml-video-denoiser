"""Pluggable noise synthesis for training the denoiser.

All generators follow the NoiseGenerator protocol: they accept a clean image
tensor of shape (C, H, W) and return a 3-tuple (noisy, clean, sigma_map) where
sigma_map encodes the local noise standard deviation — used for pixel-weighted
loss during training.

Generator overview
------------------
GaussianNoiseGenerator
    Additive white Gaussian noise.  Fast, simple baseline.

PoissonGaussianNoiseGenerator
    Heteroscedastic shot + read noise.  Better match for real sensors,
    especially in linear-light (RAW) domain.

RealNoiseInjectionGenerator
    Injects patches sampled from real captured sequences (dark frames, grain
    plates, render noise, etc.).  Each pool has its own blend mode and weight:

        pools = [
            ("pools/dark_iso800.npz",  "add",     3.0),   # heavy additive residuals
            ("pools/dark_iso3200.npz", "add",     1.0),
            ("pools/grain.npz",        "overlay", 2.0),   # film grain overlay plate
        ]
        gen = RealNoiseInjectionGenerator(pools)

    Blend modes
    ~~~~~~~~~~~
    add        — noisy = clean + patch           use for zero-mean residuals
                                                  (--save-patches default)
    screen     — noisy = 1 - (1-clean)(1-patch)  brightening grain / halation
    overlay    — signal-dependent; grain is       use for 50 %-grey grain plates
                 strongest in midtones            (--save-patches --no-mean-subtract)
    soft_light — gentler overlay (Pegtop)         subtle grain

    Pool weights control how often each pool is sampled relative to the others.
    Values are arbitrary positive numbers — they do not need to sum to 1.

    Pools are built with noise_profiler.py — see that module's docstring for
    capture and extraction instructions.

RealRAWNoiseGenerator
    Calibrated Poisson-Gaussian noise from a JSON profile produced by
    noise_profiler.py Mode 1.  Fitted K and sigma_r per ISO.

CameraNoiseGenerator
    ISO-parameterised Poisson-Gaussian with row-banding fixed-pattern noise.
    Designed for temporal clips: call for_clip() once per clip to hold the
    same ISO and fixed-pattern across all frames.

MixedNoiseGenerator
    Randomly selects among any combination of the above per training sample.
    MixedNoiseGenerator.default() is the recommended entry point and wires
    in whichever sources you provide.

Typical usage
-------------
    # Synthetic only (no real data)
    gen = MixedNoiseGenerator.default()

    # With real noise pools
    gen = MixedNoiseGenerator.default(
        patch_pools=[
            ("pools/dark_iso3200.npz", "add",     2.0),
            ("pools/grain.npz",        "overlay", 1.0),
        ],
    )

    noisy, clean, sigma_map = gen(clean_patch)  # clean_patch: (C, H, W) float32

CLI (training.py)
-----------------
    --patch-pool PATH[:MODE[:WEIGHT]] [...]

    Examples:
        --patch-pool pools/dark.npz
        --patch-pool pools/dark.npz:add:3 pools/grain.npz:overlay:1
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
# MCNoise — Nuke Blink-compatible luminance-dependent grain + fireflies
# ---------------------------------------------------------------------------


class MCNoiseGenerator:
    """Luminance-dependent grain with per-channel chroma spread and firefly spikes.

    Matches the MCNoise Blink kernel exactly:
      - Per-pixel sigma scales with sqrt(lum) — shot-noise characteristic
      - Independent per-channel chroma perturbation on top of shared luma grain
      - Optional dark fade: noise amplitude can taper toward zero in blacks
      - Optional firefly spikes: rare bright outliers with their own dark fade

    MC render noise has R typically 2–4× noisier than G and B.
    Use chroma_spread_r >> chroma_spread_g ≈ chroma_spread_b to match this.

    Args:
        intensity:            Base noise intensity (default 1.0).
        samples:              Virtual sample count; sigma = intensity / sqrt(samples).
        chroma_spread_r:      R channel chroma noise scale (default 0.3).
        chroma_spread_g:      G channel chroma noise scale (default 0.1).
        chroma_spread_b:      B channel chroma noise scale (default 0.08).
        noise_dark_fade:      0 = full noise in shadows, 1 = zero noise in blacks.
        noise_fade_falloff:   Power curve for noise dark fade (1 = linear).
        firefly_thresh:       Firefly spike floor magnitude (default 6.0).
        firefly_prob:         Per-pixel probability of a firefly (default 0.003).
        firefly_chroma:       Chroma spread of firefly spikes (default 0.1).
        firefly_dark_fade:    0 = fireflies everywhere, 1 = none in blacks.
        firefly_fade_falloff: Power curve for firefly dark fade (1 = linear).
    """

    def __init__(
        self,
        intensity: float = 1.0,
        samples: int = 16,
        chroma_spread_r: float = 0.3,
        chroma_spread_g: float = 0.1,
        chroma_spread_b: float = 0.08,
        noise_dark_fade: float = 0.0,
        noise_fade_falloff: float = 1.0,
        firefly_thresh: float = 6.0,
        firefly_prob: float = 0.003,
        firefly_chroma: float = 0.1,
        firefly_dark_fade: float = 0.0,
        firefly_fade_falloff: float = 1.0,
    ) -> None:
        self._sigma = intensity / max(samples, 1) ** 0.5
        self._cs_r = chroma_spread_r
        self._cs_g = chroma_spread_g
        self._cs_b = chroma_spread_b
        self._noise_dark_fade = noise_dark_fade
        self._noise_fade_falloff = max(noise_fade_falloff, 0.001)
        self._firefly_thresh = firefly_thresh
        self._firefly_prob = firefly_prob
        self._firefly_chroma = firefly_chroma
        self._firefly_dark_fade = firefly_dark_fade
        self._firefly_fade_falloff = max(firefly_fade_falloff, 0.001)

    def _fade_mask(self, lum: Tensor, falloff: float) -> Tensor:
        return lum.clamp(0.0, 1.0).pow(falloff)

    def __call__(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        _, h, w = clean.shape
        dev, dtype = clean.device, clean.dtype

        lum = (0.2126 * clean[0] + 0.7152 * clean[1] + 0.0722 * clean[2]).unsqueeze(0)

        noise_mask = 1.0 - self._noise_dark_fade * (1.0 - self._fade_mask(lum, self._noise_fade_falloff))
        dark_floor = 0.05 * (1.0 - self._noise_dark_fade)
        lum_scale = lum.clamp_min(0.0).sqrt() + dark_floor
        noise = self._sigma * lum_scale * noise_mask  # (1, H, W)

        luma_g = torch.randn(1, h, w, device=dev, dtype=dtype)
        chroma = torch.randn(3, h, w, device=dev, dtype=dtype)
        chroma[0] *= self._cs_r
        chroma[1] *= self._cs_g
        chroma[2] *= self._cs_b
        noisy = clean + (luma_g + chroma) * noise

        if self._firefly_prob > 0.0:
            ff_mask = 1.0 - self._firefly_dark_fade * (1.0 - self._fade_mask(lum, self._firefly_fade_falloff))
            fire = torch.rand(1, h, w, device=dev, dtype=dtype) < (self._firefly_prob * ff_mask)
            spike = self._firefly_thresh + torch.rand(1, h, w, device=dev, dtype=dtype) * self._firefly_thresh * 2.0
            ff_chroma = torch.randn(3, h, w, device=dev, dtype=dtype) * self._firefly_chroma
            noisy = noisy + fire * spike * (1.0 + ff_chroma)

        avg_cs = (self._cs_r + self._cs_g + self._cs_b) / 3.0
        sigma_map = (noise * (1.0 + avg_cs)).expand_as(clean)
        return noisy, clean, sigma_map


# ---------------------------------------------------------------------------
# MCNoise preset bank — weighted pool of MCNoiseGenerator configs
# ---------------------------------------------------------------------------

_MC_KNOB_KEYS: tuple[str, ...] = (
    "intensity", "samples",
    "chroma_spread_r", "chroma_spread_g", "chroma_spread_b",
    "noise_dark_fade", "noise_fade_falloff",
    "firefly_thresh", "firefly_prob", "firefly_chroma",
    "firefly_dark_fade", "firefly_fade_falloff",
)


class MCNoisePresetBank:
    """Weighted pool of MCNoiseGenerator presets loaded from JSON.

    JSON format (array of preset objects):
        [
          {"name": "light",  "intensity": 0.5, "samples": 32, "weight": 1},
          {"name": "medium", "intensity": 1.0, "samples": 16, "weight": 3},
          {"name": "heavy",  "intensity": 2.0, "samples": 8,  "weight": 1,
           "firefly_prob": 0.005}
        ]

    All MCNoiseGenerator constructor keys are valid.  ``name`` and ``weight``
    are metadata — ``name`` is informational, ``weight`` controls sampling
    frequency (default 1.0).  Weights do not need to sum to any value.

    Export from Nuke using nuke/export_mc_noise_presets.py.
    """

    def __init__(self, entries: list[tuple["MCNoiseGenerator", float, str]]) -> None:
        self._generators = [(g, name) for g, _, name in entries]
        weights = [w for _, w, _ in entries]
        total = sum(weights)
        self._probs = [w / total for w in weights]

    @classmethod
    def default(cls) -> "MCNoisePresetBank":
        """Light × 1, medium × 3, heavy × 1 — sensible coverage for MC render noise."""
        return cls([
            (MCNoiseGenerator(intensity=0.5,  samples=32,
                              chroma_spread_r=0.3, chroma_spread_g=0.10, chroma_spread_b=0.08,
                              firefly_prob=0.0),                                1.0, "light"),
            (MCNoiseGenerator(intensity=1.0,  samples=16,
                              chroma_spread_r=0.4, chroma_spread_g=0.12, chroma_spread_b=0.10,
                              firefly_prob=0.0),                                3.0, "medium"),
            (MCNoiseGenerator(intensity=2.0,  samples=8,
                              chroma_spread_r=0.5, chroma_spread_g=0.15, chroma_spread_b=0.12,
                              firefly_prob=0.0),                                1.0, "heavy"),
        ])

    @classmethod
    def from_json(cls, path: str) -> "MCNoisePresetBank":
        with open(path) as f:
            data = json.load(f)
        entries: list[tuple[MCNoiseGenerator, float, str]] = []
        for item in data:
            name = item.get("name", "preset")
            weight = float(item.get("weight", 1.0))
            kwargs = {}
            for k in _MC_KNOB_KEYS:
                if k in item:
                    kwargs[k] = int(item[k]) if k == "samples" else float(item[k])
            # backwards compat: old chroma_spread sets all three channels equally
            if "chroma_spread" in item and not any(k in item for k in ("chroma_spread_r", "chroma_spread_g", "chroma_spread_b")):
                cs = float(item["chroma_spread"])
                kwargs.setdefault("chroma_spread_r", cs)
                kwargs.setdefault("chroma_spread_g", cs)
                kwargs.setdefault("chroma_spread_b", cs)
            entries.append((MCNoiseGenerator(**kwargs), weight, name))
        return cls(entries)

    def __call__(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        gen, _ = random.choices(self._generators, weights=self._probs, k=1)[0]
        return gen(clean)


# ---------------------------------------------------------------------------
# Blend modes
# ---------------------------------------------------------------------------

_BLEND_MODES: frozenset[str] = frozenset({"add", "screen", "overlay", "soft_light"})


def _apply_blend(clean: Tensor, patch: Tensor, mode: str) -> Tensor:
    """Apply *patch* onto *clean* using the given blend mode.

    Modes:
        add        — clean + patch  (patch is a zero-mean residual)
        screen     — 1 - (1-clean)(1-patch)
        overlay    — contrast-enhancing; signal-dependent grain strength
        soft_light — gentler overlay (Pegtop formula)
    """
    if mode == "add":
        return clean + patch
    if mode == "screen":
        return 1.0 - (1.0 - clean) * (1.0 - patch)
    if mode == "overlay":
        return torch.where(
            clean < 0.5,
            2.0 * clean * patch,
            1.0 - 2.0 * (1.0 - clean) * (1.0 - patch),
        )
    if mode == "soft_light":
        return (1.0 - 2.0 * patch) * clean ** 2 + 2.0 * patch * clean
    raise ValueError(f"Unknown blend mode {mode!r}. Choose from: {sorted(_BLEND_MODES)}")


# ---------------------------------------------------------------------------
# Real noise injection
# ---------------------------------------------------------------------------


class RealNoiseInjectionGenerator:
    """Injects authentic noise patches sampled from real captured sequences.

    Each pool entry is (path, mode, weight).  On each call one pool is chosen
    by weighted random selection, a patch is cropped, and the blend mode applied.

    Pools are produced by noise_profiler.py:

        # Additive residuals — default, subtracts temporal mean
        python noise_profiler.py --dark dark/*.png --save-patches pools/dark.npz

        # Overlay grain plate — raw pixel values, no mean subtraction
        python noise_profiler.py --dark grain/*.png --no-mean-subtract \\
            --save-patches pools/grain.npz

        gen = RealNoiseInjectionGenerator([
            ("pools/dark_iso800.npz",  "add",     3.0),
            ("pools/dark_iso3200.npz", "add",     1.0),
            ("pools/grain.npz",        "overlay", 2.0),
        ])

    Args:
        pools: List of (npz_path, blend_mode, weight) triples.  weight controls
               relative sampling frequency; values don't need to sum to 1.
               blend_mode is one of ``"add"``, ``"screen"``, ``"overlay"``,
               ``"soft_light"``.  A bare string path defaults to add / weight 1.
        sigma_window: Kernel size for the local std estimate used in sigma_map.
    """

    def __init__(
        self,
        pools: str | list[str] | list[tuple[str, str, float]],
        sigma_window: int = 9,
    ) -> None:
        if isinstance(pools, str):
            specs: list[tuple[str, str, float]] = [(pools, "add", 1.0)]
        elif pools and isinstance(pools[0], str):
            specs = [(p, "add", 1.0) for p in pools]  # type: ignore[arg-type]
        else:
            specs = pools  # type: ignore[assignment]

        self._pools: list[Tensor] = []
        self._modes: list[str] = []
        self._weights: list[float] = []
        for path, mode, weight in specs:
            data = np.load(path)
            residuals = data["residuals"].astype(np.float32)
            if residuals.ndim == 4 and residuals.shape[-1] in (1, 3, 4):
                residuals = residuals.transpose(0, 3, 1, 2)
            self._pools.append(torch.from_numpy(residuals))
            self._modes.append(mode)
            self._weights.append(weight)
        self._sigma_window = sigma_window

    def for_clip(self) -> "_LockedPoolApplier":
        """Return a clip-scoped applier with the pool choice locked for all frames.

        Call once per temporal window; reuse the returned object for every frame
        in that clip.  This ensures all frames share the same noise source and
        blend mode, matching real-world behaviour where a clip is shot at a fixed
        ISO / grain plate pass.

        Spatial crop position is still randomised per frame — only the pool
        selection is locked.
        """
        (i,) = random.choices(range(len(self._pools)), weights=self._weights, k=1)
        return _LockedPoolApplier(self._pools[i], self._modes[i], self._sigma_window)

    def __call__(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Pick a weighted-random pool, crop a patch, blend onto *clean*."""
        return self.for_clip()(clean)

    def _local_std(self, x: Tensor) -> Tensor:
        k = self._sigma_window
        pad = k // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        patches = x_pad.unfold(2, k, 1).unfold(3, k, 1)
        var = patches.var(dim=(-2, -1), unbiased=False)
        return var.sqrt().squeeze(0)


class _LockedPoolApplier:
    """Single-pool noise applier with a fixed pool index for a temporal clip.

    Do not instantiate directly — use RealNoiseInjectionGenerator.for_clip().
    """

    def __init__(self, pool: Tensor, mode: str, sigma_window: int) -> None:
        self._pool = pool
        self._mode = mode
        self._sigma_window = sigma_window

    def __call__(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        n, pc, ph, pw = self._pool.shape
        c, h, w = clean.shape[-3], clean.shape[-2], clean.shape[-1]

        if ph < h or pw < w:
            raise ValueError(
                f"Pool patch size ({ph}×{pw}) is smaller than clean patch ({h}×{w}). "
                "Re-extract a larger patch pool."
            )

        idx = random.randrange(n)
        top = random.randint(0, ph - h)
        left = random.randint(0, pw - w)
        patch = self._pool[idx, :, top : top + h, left : left + w]

        if pc != c:
            patch = patch[:c] if pc > c else patch.mean(0, keepdim=True).expand(c, -1, -1)

        patch = patch.to(clean.device)
        noisy = _apply_blend(clean, patch, self._mode)
        sigma_map = self._local_std((noisy - clean).unsqueeze(0)).squeeze(0)
        return noisy, clean, sigma_map

    def _local_std(self, x: Tensor) -> Tensor:
        k = self._sigma_window
        pad = k // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        patches = x_pad.unfold(2, k, 1).unfold(3, k, 1)
        var = patches.var(dim=(-2, -1), unbiased=False)
        return var.sqrt().squeeze(0)


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
# Camera video noise
# ---------------------------------------------------------------------------


class _ClipNoiseApplier:
    """Per-clip noise state created by ``CameraNoiseGenerator.for_clip()``.

    Holds fixed ISO parameters and lazily generates row-banding fixed-pattern
    noise on the first call so it remains identical across all frames of a clip.
    Do not instantiate directly — use ``CameraNoiseGenerator.for_clip()``.
    """

    def __init__(self, k: float, sigma_r: float, fp_strength: float) -> None:
        self._k = k
        self._sigma_r = sigma_r
        self._fp_strength = fp_strength
        self._fixed_pattern: Optional[Tensor] = None

    def __call__(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self._fixed_pattern is None:
            # Row banding: (…, C, H, 1) broadcasts across W — same for every frame
            fp_shape = list(clean.shape)
            fp_shape[-1] = 1
            self._fixed_pattern = torch.randn(fp_shape, device=clean.device) * self._fp_strength
        fp = self._fixed_pattern.to(clean.device)
        shot = torch.poisson(clean.clamp(min=0.0) / self._k) * self._k - clean
        read = torch.randn_like(clean) * self._sigma_r
        noisy = clean + shot + read + fp
        sigma_map = (self._k * clean.clamp(min=0.0) + self._sigma_r ** 2).sqrt()
        return noisy, clean, sigma_map


class CameraNoiseGenerator:
    """ISO-parameterised Poisson-Gaussian noise model for camera video.

    Maps ISO values to Poisson gain *K* and read noise *sigma_r* via an
    empirical camera noise model:

        K       = K_ref  * (iso / iso_ref)            — shot noise, linear in ISO
        sigma_r = sr_ref * (iso / iso_ref) ** 0.5     — read noise, sub-linear

    Single-frame usage (new ISO drawn per call):

        gen = CameraNoiseGenerator(iso_range=(100, 6400))
        noisy, clean, sigma_map = gen(clean_frame)

    Temporally consistent clip usage (same ISO + fixed-pattern for all frames):

        per_frame = gen.for_clip()
        noisy_frames = [per_frame(f) for f in clean_clip]

    Args:
        iso_range:              (min, max) ISO to sample from (default: 100–6400).
        iso_ref:                Reference ISO at which K_ref / sr_ref are defined.
        K_ref:                  Shot noise gain at iso_ref (electrons/DN equivalent).
        sr_ref:                 Read noise std dev at iso_ref (normalised 0-1 range).
        fixed_pattern_strength: Row-banding amplitude used by ``for_clip()``.
                                0.0 disables fixed-pattern noise.
    """

    def __init__(
        self,
        iso_range: tuple[float, float] = (100.0, 6400.0),
        iso_ref: float = 1600.0,
        K_ref: float = 0.012,
        sr_ref: float = 0.005,
        fixed_pattern_strength: float = 0.002,
    ) -> None:
        self.iso_range = iso_range
        self.iso_ref = iso_ref
        self.K_ref = K_ref
        self.sr_ref = sr_ref
        self.fixed_pattern_strength = fixed_pattern_strength

    def _iso_to_params(self, iso: float) -> tuple[float, float]:
        ratio = iso / self.iso_ref
        return self.K_ref * ratio, self.sr_ref * (ratio ** 0.5)

    def __call__(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Apply Poisson-Gaussian camera noise to a single frame.

        Draws a new ISO per call — for clip-consistent noise use ``for_clip()``.
        """
        iso = random.uniform(*self.iso_range)
        k, sigma_r = self._iso_to_params(iso)
        shot = torch.poisson(clean.clamp(min=0.0) / k) * k - clean
        read = torch.randn_like(clean) * sigma_r
        noisy = clean + shot + read
        sigma_map = (k * clean.clamp(min=0.0) + sigma_r ** 2).sqrt()
        return noisy, clean, sigma_map

    def for_clip(self) -> _ClipNoiseApplier:
        """Return a clip-scoped noise applicator with fixed ISO and fixed-pattern noise.

        Call once per clip; reuse the returned object for every frame in that clip.
        ISO (K, sigma_r) and row-banding are sampled once and held fixed across
        all frames — matching real camera behaviour where ISO is set per shot.
        """
        iso = random.uniform(*self.iso_range)
        k, sigma_r = self._iso_to_params(iso)
        return _ClipNoiseApplier(k, sigma_r, self.fixed_pattern_strength)


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
        patch_pools: Optional[list[tuple[str, str, float]]] = None,
        profile_json: Optional[str] = None,
    ) -> "MixedNoiseGenerator":
        """Build the recommended mixed generator.

        Wires together synthetic and real-noise generators with sensible default
        weights.  Synthetic generators are always included; real-noise sources
        are added only when supplied, with their weight drawn from the shared
        budget (0.40) that would otherwise be split among synthetic generators.

        Weight allocation:
            Gaussian            0.30  (always)
            Poisson-Gaussian    0.30  (always)
            RealNoiseInjection  0.25  (when patch_pools provided)
            RealRAWNoise        0.15  (when profile_json provided)
            Remaining budget redistributed equally to Gaussian / Poisson-Gaussian.

        Args:
            patch_pools: List of (npz_path, blend_mode, weight) triples passed
                to RealNoiseInjectionGenerator.  blend_mode is one of
                ``"add"``, ``"screen"``, ``"overlay"``, ``"soft_light"``.
                The per-pool weight controls relative sampling frequency within
                RealNoiseInjectionGenerator (independent of the 0.25 slot above).
            profile_json: Path to a camera noise profile JSON produced by
                noise_profiler.py Mode 1.
        """
        generators: list[NoiseGenerator] = [
            GaussianNoiseGenerator(0.0, 75.0 / 255.0),
            PoissonGaussianNoiseGenerator(),
        ]
        weights = [0.30, 0.30]
        remaining = 0.40

        if patch_pools:
            generators.append(RealNoiseInjectionGenerator(patch_pools))
            weights.append(0.25)
            remaining -= 0.25

        if profile_json is not None:
            generators.append(RealRAWNoiseGenerator(profile_json))
            weights.append(0.15)
            remaining -= 0.15

        weights[0] += remaining / 2
        weights[1] += remaining / 2

        return cls(generators=generators, weights=weights)

    def for_clip(self) -> "NoiseGenerator":
        """Return a clip-scoped noise applier with the generator locked for all frames.

        Picks one generator (weighted random) and, if that generator supports
        for_clip(), delegates to it so pool / ISO selection is also locked.
        Use this in temporal datasets to ensure consistent noise across frames.
        """
        (gen,) = random.choices(self._generators, weights=self._weights, k=1)
        if hasattr(gen, "for_clip"):
            return gen.for_clip()
        return gen

    def __call__(self, clean: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Delegate to a randomly selected generator."""
        (gen,) = random.choices(self._generators, weights=self._weights, k=1)
        return gen(clean)
