"""Camera noise profiling from dark-frame image sequences.

This module extracts noise statistics from footage recorded with the lens cap
on (or in a completely dark environment).  The results are used to calibrate
the RealRAWNoiseGenerator and RealNoiseInjectionGenerator classes.

Two output modes:

  Mode 1 — Parametric profile (for RealRAWNoiseGenerator):
      Fits scalar K (Poisson gain) and sigma_r (read noise) per ISO profile.
      Requires dark frames; optionally flat-field frames for K estimation.

      uv run python noise_profiler.py \\
          --dark dark_sequence/*.png \\
          --flat flat_sequence/*.png \\
          --iso iso_3200 \\
          --output profiles/sony_a7iii_iso3200.json

  Mode 2 — Patch pool (for RealNoiseInjectionGenerator):
      Subtracts temporal mean from each dark frame, stores zero-mean residuals
      as a compact .npz patch pool.

      uv run python noise_profiler.py \\
          --dark dark_sequence/*.png \\
          --save-patches pools/sony_a7iii_iso3200.npz \\
          --patch-size 128

Both modes can be run together in a single invocation.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------


def _load_frames(paths: list[Path]) -> np.ndarray:
    """Load a list of image files into a float32 array of shape (N, H, W, C).

    Supports PNG, TIFF, EXR, and anything imageio can open.  All frames are
    normalised to [0, 1].  Frames with differing shapes raise ValueError.

    Args:
        paths: Sorted list of image file paths.

    Returns:
        Array of shape (N, H, W, C), dtype float32.
    """
    import imageio.v3 as iio

    frames = []
    ref_shape: Optional[tuple] = None
    for path in paths:
        img = iio.imread(str(path))
        img = img.astype(np.float32)
        if img.ndim == 2:
            img = img[:, :, np.newaxis]
        # Normalise to [0, 1] based on dtype range
        max_val = _dtype_max(img)
        img /= max_val
        if ref_shape is None:
            ref_shape = img.shape
        elif img.shape != ref_shape:
            raise ValueError(
                f"Frame shape mismatch: {path.name} is {img.shape}, expected {ref_shape}"
            )
        frames.append(img)
    if not frames:
        raise ValueError("No frames loaded.")
    return np.stack(frames, axis=0)  # (N, H, W, C)


def _dtype_max(arr: np.ndarray) -> float:
    """Return the normalisation maximum for an array, based on its dtype."""
    if np.issubdtype(arr.dtype, np.floating):
        return 1.0
    if arr.dtype == np.uint8:
        return 255.0
    if arr.dtype == np.uint16:
        return 65535.0
    return float(np.iinfo(arr.dtype).max)


# ---------------------------------------------------------------------------
# Statistical estimators
# ---------------------------------------------------------------------------


def compute_temporal_stats(
    frames: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute per-pixel temporal mean and std from a dark-frame sequence.

    Args:
        frames: (N, H, W, C) float32 array, values in [0, 1].

    Returns:
        mean_dark: (H, W, C) — fixed pattern noise estimate.
        std_map: (H, W, C) — per-pixel temporal noise std.
        sigma_r: scalar — spatially-averaged read noise estimate.
    """
    mean_dark = frames.mean(axis=0)         # (H, W, C)
    std_map = frames.std(axis=0)            # (H, W, C)
    sigma_r = float(std_map.mean())
    return mean_dark, std_map, sigma_r


def estimate_poisson_gain(
    flat_frames: np.ndarray,
    dark_mean: np.ndarray,
    min_samples: int = 5,
) -> Optional[float]:
    """Estimate Poisson gain K via the photon transfer curve (PTC).

    The PTC relationship:  Var(signal) ≈ K * Mean(signal) + sigma_r^2

    We estimate K as the slope of the linear regression Var vs Mean across
    spatial pixels (treating each pixel as an independent measurement at a
    given signal level).

    Args:
        flat_frames: (N, H, W, C) float32 — uniformly lit frames.
        dark_mean: (H, W, C) float32 — dark frame mean to subtract.
        min_samples: Minimum spatial samples needed for a valid fit.

    Returns:
        K (Poisson gain, scalar) or None if estimation fails.
    """
    if flat_frames.shape[0] < 2:
        return None

    signal = flat_frames.mean(axis=0) - dark_mean   # (H, W, C)
    variance = flat_frames.var(axis=0)               # (H, W, C)

    signal_flat = signal.ravel()
    var_flat = variance.ravel()

    # Remove saturated / near-zero pixels
    mask = (signal_flat > 0.01) & (signal_flat < 0.95)
    if mask.sum() < min_samples:
        return None

    x = signal_flat[mask]
    y = var_flat[mask]

    # Ordinary least squares: y = K*x + intercept
    x_mean = x.mean()
    k = float(np.dot(x - x_mean, y - y.mean()) / np.dot(x - x_mean, x - x_mean))
    return max(k, 1e-6)  # guard against negative fits


# ---------------------------------------------------------------------------
# Mode 1: parametric profile
# ---------------------------------------------------------------------------


def build_parametric_profile(
    dark_paths: list[Path],
    flat_paths: list[Path],
    iso_label: str,
    camera: str,
    existing_profile: Optional[dict] = None,
) -> dict:
    """Fit K and sigma_r and merge into a profile dict.

    Args:
        dark_paths: Paths to dark frames.
        flat_paths: Paths to flat-field frames (optional, for K estimation).
        iso_label: ISO label string, e.g. ``"iso_3200"``.
        camera: Camera name for documentation.
        existing_profile: Existing profile dict to merge into.

    Returns:
        Updated profile dict with the new ISO entry.
    """
    print(f"Loading {len(dark_paths)} dark frames…", file=sys.stderr)
    dark_frames = _load_frames(dark_paths)
    mean_dark, _, sigma_r = compute_temporal_stats(dark_frames)
    print(f"  sigma_r = {sigma_r:.6f}", file=sys.stderr)

    k: Optional[float] = None
    if flat_paths:
        print(f"Loading {len(flat_paths)} flat frames…", file=sys.stderr)
        flat_frames = _load_frames(flat_paths)
        k = estimate_poisson_gain(flat_frames, mean_dark)
        if k is not None:
            print(f"  K = {k:.6f}", file=sys.stderr)
        else:
            print("  K estimation failed — too few valid pixels.", file=sys.stderr)

    profile = existing_profile or {"camera": camera, "iso_profiles": {}}
    profile["camera"] = camera
    entry: dict = {"sigma_r": sigma_r}
    if k is not None:
        entry["K"] = k
    profile["iso_profiles"][iso_label] = entry
    return profile


# ---------------------------------------------------------------------------
# Mode 2: patch pool
# ---------------------------------------------------------------------------


def build_patch_pool(
    dark_paths: list[Path],
    patch_size: int,
    stride: Optional[int] = None,
) -> np.ndarray:
    """Extract zero-mean noise residual patches from a dark-frame sequence.

    Subtracts the temporal mean from each frame (removing fixed pattern noise),
    then tiles the result into non-overlapping patches.

    Args:
        dark_paths: Paths to dark frames.
        patch_size: Spatial size of each square patch.
        stride: Extraction stride.  Defaults to *patch_size* (non-overlapping).

    Returns:
        Array of shape (N_patches, C, patch_size, patch_size), dtype float32.
        The ``residuals`` key in the saved .npz matches this layout.
    """
    if stride is None:
        stride = patch_size

    print(f"Loading {len(dark_paths)} dark frames for patch pool…", file=sys.stderr)
    frames = _load_frames(dark_paths)          # (N, H, W, C)
    mean_dark = frames.mean(axis=0)            # (H, W, C) — fixed pattern noise
    residuals = frames - mean_dark[np.newaxis]  # (N, H, W, C) — zero-mean noise

    n_frames, h, w, c = residuals.shape
    patches = []
    for fi in range(n_frames):
        frame = residuals[fi]  # (H, W, C)
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = frame[y : y + patch_size, x : x + patch_size, :]  # (pH, pW, C)
                patches.append(patch.transpose(2, 0, 1))  # (C, pH, pW)

    if not patches:
        raise ValueError(
            f"No patches extracted — frame size ({h}×{w}) may be smaller than "
            f"patch_size ({patch_size})."
        )

    pool = np.stack(patches, axis=0).astype(np.float32)  # (N, C, H, W)
    print(f"  Extracted {len(pool)} patches of size {patch_size}×{patch_size}.", file=sys.stderr)
    return pool


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _glob_paths(patterns: list[str]) -> list[Path]:
    """Expand shell glob patterns into a sorted list of Paths."""
    import glob

    paths = []
    for pattern in patterns:
        expanded = glob.glob(pattern, recursive=True)
        paths.extend(Path(p) for p in expanded)
    return sorted(set(paths))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract a camera noise profile from dark-frame sequences.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    # Input
    parser.add_argument(
        "--dark", nargs="+", required=True, metavar="PATTERN",
        help="Dark frame image files or glob patterns.",
    )
    parser.add_argument(
        "--flat", nargs="+", default=None, metavar="PATTERN",
        help="Flat-field frames for Poisson gain K estimation (optional).",
    )

    # Mode 1: parametric profile
    parser.add_argument(
        "--output", metavar="PATH",
        help="Output JSON profile path (Mode 1: parametric calibration).",
    )
    parser.add_argument(
        "--iso", default="iso_unknown", metavar="LABEL",
        help="ISO label to embed in the profile, e.g. iso_3200.",
    )
    parser.add_argument(
        "--camera", default="Unknown", metavar="NAME",
        help="Camera name for documentation purposes.",
    )

    # Mode 2: patch pool
    parser.add_argument(
        "--save-patches", metavar="PATH",
        help="Output .npz patch pool path (Mode 2: real noise injection).",
    )
    parser.add_argument(
        "--patch-size", type=int, default=128, metavar="N",
        help="Spatial size of extracted patches (default: 128).",
    )
    parser.add_argument(
        "--patch-stride", type=int, default=None, metavar="N",
        help="Stride for patch extraction (default: patch_size, non-overlapping).",
    )

    args = parser.parse_args()

    if args.output is None and args.save_patches is None:
        parser.error("Specify at least one of --output or --save-patches.")

    dark_paths = _glob_paths(args.dark)
    if not dark_paths:
        parser.error(f"No dark frames found for patterns: {args.dark}")

    flat_paths = _glob_paths(args.flat) if args.flat else []

    # --- Mode 1: parametric profile ---
    if args.output is not None:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        existing: Optional[dict] = None
        if out_path.exists():
            with open(out_path) as f:
                existing = json.load(f)
            print(f"Merging into existing profile: {out_path}", file=sys.stderr)

        profile = build_parametric_profile(
            dark_paths=dark_paths,
            flat_paths=flat_paths,
            iso_label=args.iso,
            camera=args.camera,
            existing_profile=existing,
        )
        with open(out_path, "w") as f:
            json.dump(profile, f, indent=2)
        print(f"Wrote parametric profile → {out_path}")

    # --- Mode 2: patch pool ---
    if args.save_patches is not None:
        pool_path = Path(args.save_patches)
        pool_path.parent.mkdir(parents=True, exist_ok=True)
        pool = build_patch_pool(
            dark_paths=dark_paths,
            patch_size=args.patch_size,
            stride=args.patch_stride,
        )
        np.savez_compressed(pool_path, residuals=pool)
        print(f"Wrote patch pool ({len(pool)} patches) → {pool_path}")


if __name__ == "__main__":
    main()
