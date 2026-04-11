"""Tests for noise_profiler.py."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
from noise_profiler import (
    build_parametric_profile,
    build_patch_pool,
    compute_temporal_stats,
    estimate_poisson_gain,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def dark_frame_dir(tmp_path: Path) -> list[Path]:
    """Create 10 synthetic dark frames (PNG) in a temp dir."""
    import imageio.v3 as iio

    paths = []
    rng = np.random.default_rng(42)
    mean_dark = rng.normal(0.02, 0.001, (64, 64, 3)).astype(np.float32).clip(0, 1)
    for i in range(10):
        frame = (mean_dark + rng.normal(0, 0.005, (64, 64, 3))).clip(0, 1)
        frame_u8 = (frame * 255).astype(np.uint8)
        p = tmp_path / f"dark_{i:04d}.png"
        iio.imwrite(str(p), frame_u8)
        paths.append(p)
    return paths


@pytest.fixture()
def flat_frame_dir(tmp_path: Path) -> list[Path]:
    """Create 6 synthetic flat frames for PTC estimation."""
    import imageio.v3 as iio

    paths = []
    rng = np.random.default_rng(99)
    for i in range(6):
        signal = 0.3 + i * 0.1  # vary exposure
        frame = (signal + rng.normal(0, 0.01, (64, 64, 3))).clip(0, 1)
        frame_u8 = (frame * 255).astype(np.uint8)
        p = tmp_path / f"flat_{i:04d}.png"
        iio.imwrite(str(p), frame_u8)
        paths.append(p)
    return paths


# ---------------------------------------------------------------------------
# compute_temporal_stats
# ---------------------------------------------------------------------------


class TestComputeTemporalStats:
    def test_mean_dark_shape(self, dark_frame_dir: list[Path]) -> None:
        import imageio.v3 as iio

        frames = np.stack(
            [iio.imread(str(p)).astype(np.float32) / 255.0 for p in dark_frame_dir]
        )
        mean_dark, std_map, sigma_r = compute_temporal_stats(frames)
        assert mean_dark.shape == frames.shape[1:]
        assert std_map.shape == frames.shape[1:]
        assert sigma_r > 0.0

    def test_sigma_r_reasonable(self, dark_frame_dir: list[Path]) -> None:
        import imageio.v3 as iio

        frames = np.stack(
            [iio.imread(str(p)).astype(np.float32) / 255.0 for p in dark_frame_dir]
        )
        _, _, sigma_r = compute_temporal_stats(frames)
        # Synthetic frames have std ~0.005; allow generous range
        assert 1e-4 < sigma_r < 0.1


# ---------------------------------------------------------------------------
# estimate_poisson_gain
# ---------------------------------------------------------------------------


class TestEstimatePoissonGain:
    def test_returns_positive_float(self, flat_frame_dir: list[Path]) -> None:
        import imageio.v3 as iio

        flat_frames = np.stack(
            [iio.imread(str(p)).astype(np.float32) / 255.0 for p in flat_frame_dir]
        )
        dark_mean = np.zeros(flat_frames.shape[1:], dtype=np.float32)
        k = estimate_poisson_gain(flat_frames, dark_mean)
        assert k is None or k > 0.0

    def test_single_flat_frame_returns_none(self, flat_frame_dir: list[Path]) -> None:
        import imageio.v3 as iio

        flat_frames = np.stack(
            [iio.imread(str(flat_frame_dir[0])).astype(np.float32) / 255.0]
        )
        k = estimate_poisson_gain(flat_frames, np.zeros(flat_frames.shape[1:]))
        assert k is None


# ---------------------------------------------------------------------------
# build_parametric_profile
# ---------------------------------------------------------------------------


class TestBuildParametricProfile:
    def test_output_keys(self, dark_frame_dir: list[Path], tmp_path: Path) -> None:
        profile = build_parametric_profile(
            dark_paths=dark_frame_dir,
            flat_paths=[],
            iso_label="iso_3200",
            camera="TestCam",
        )
        assert "camera" in profile
        assert "iso_profiles" in profile
        assert "iso_3200" in profile["iso_profiles"]
        assert "sigma_r" in profile["iso_profiles"]["iso_3200"]

    def test_merge_into_existing(self, dark_frame_dir: list[Path]) -> None:
        existing = {
            "camera": "TestCam",
            "iso_profiles": {"iso_800": {"K": 0.003, "sigma_r": 0.002}},
        }
        profile = build_parametric_profile(
            dark_paths=dark_frame_dir,
            flat_paths=[],
            iso_label="iso_3200",
            camera="TestCam",
            existing_profile=existing,
        )
        assert "iso_800" in profile["iso_profiles"]
        assert "iso_3200" in profile["iso_profiles"]


# ---------------------------------------------------------------------------
# build_patch_pool
# ---------------------------------------------------------------------------


class TestBuildPatchPool:
    def test_pool_shape(self, dark_frame_dir: list[Path]) -> None:
        pool = build_patch_pool(dark_frame_dir, patch_size=32)
        assert pool.ndim == 4
        assert pool.shape[2] == 32
        assert pool.shape[3] == 32
        assert pool.shape[0] > 0  # at least some patches

    def test_pool_is_zero_mean_approx(self, dark_frame_dir: list[Path]) -> None:
        pool = build_patch_pool(dark_frame_dir, patch_size=32)
        # Residuals should be near-zero mean (fixed pattern removed)
        assert abs(pool.mean()) < 0.05

    def test_patch_larger_than_frame_raises(self, dark_frame_dir: list[Path]) -> None:
        with pytest.raises(ValueError, match="patch_size"):
            build_patch_pool(dark_frame_dir, patch_size=256)  # frames are 64×64
