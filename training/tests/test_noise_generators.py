"""Tests for noise_generators.py."""

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))
from noise_generators import (
    GaussianNoiseGenerator,
    MixedNoiseGenerator,
    NoiseGenerator,
    PoissonGaussianNoiseGenerator,
    RealNoiseInjectionGenerator,
    RealRAWNoiseGenerator,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PATCH = torch.rand(3, 64, 64)  # single clean patch, no batch dim


def _assert_output_shape(noisy: Tensor, clean: Tensor, sigma_map: Tensor) -> None:
    assert noisy.shape == PATCH.shape
    assert clean.shape == PATCH.shape
    assert sigma_map.shape == PATCH.shape


def _assert_range(t: Tensor, lo: float = 0.0, hi: float = 1.0) -> None:
    assert t.min().item() >= lo - 1e-5
    assert t.max().item() <= hi + 1e-5


# ---------------------------------------------------------------------------
# GaussianNoiseGenerator
# ---------------------------------------------------------------------------


class TestGaussianNoiseGenerator:
    def test_output_shapes(self) -> None:
        gen = GaussianNoiseGenerator(0.0, 50.0 / 255.0)
        noisy, clean, sigma_map = gen(PATCH)
        _assert_output_shape(noisy, clean, sigma_map)

    def test_clean_unchanged(self) -> None:
        gen = GaussianNoiseGenerator(0.0, 50.0 / 255.0)
        _, clean, _ = gen(PATCH)
        assert torch.allclose(clean, PATCH)

    def test_noisy_in_range(self) -> None:
        gen = GaussianNoiseGenerator(0.0, 50.0 / 255.0)
        noisy, _, _ = gen(PATCH)
        _assert_range(noisy)

    def test_sigma_map_uniform(self) -> None:
        gen = GaussianNoiseGenerator(10.0 / 255.0, 10.0 / 255.0)
        _, _, sigma_map = gen(PATCH)
        # All values should be the same since sigma_range is a point
        assert sigma_map.std().item() < 1e-6

    def test_zero_sigma_identity(self) -> None:
        gen = GaussianNoiseGenerator(0.0, 0.0)
        noisy, clean, _ = gen(PATCH)
        assert torch.allclose(noisy, clean)

    def test_different_calls_produce_different_noise(self) -> None:
        gen = GaussianNoiseGenerator(20.0 / 255.0, 20.0 / 255.0)
        noisy1, _, _ = gen(PATCH)
        noisy2, _, _ = gen(PATCH)
        assert not torch.allclose(noisy1, noisy2)

    def test_protocol_conformance(self) -> None:
        assert isinstance(GaussianNoiseGenerator(), NoiseGenerator)


# ---------------------------------------------------------------------------
# PoissonGaussianNoiseGenerator
# ---------------------------------------------------------------------------


class TestPoissonGaussianNoiseGenerator:
    def test_output_shapes(self) -> None:
        gen = PoissonGaussianNoiseGenerator()
        noisy, clean, sigma_map = gen(PATCH)
        _assert_output_shape(noisy, clean, sigma_map)

    def test_noisy_in_range(self) -> None:
        gen = PoissonGaussianNoiseGenerator()
        noisy, _, _ = gen(PATCH)
        _assert_range(noisy)

    def test_sigma_map_positive(self) -> None:
        gen = PoissonGaussianNoiseGenerator()
        _, _, sigma_map = gen(PATCH)
        assert (sigma_map >= 0).all()

    def test_clean_unchanged(self) -> None:
        gen = PoissonGaussianNoiseGenerator()
        _, clean, _ = gen(PATCH)
        assert torch.allclose(clean, PATCH)

    def test_protocol_conformance(self) -> None:
        assert isinstance(PoissonGaussianNoiseGenerator(), NoiseGenerator)


# ---------------------------------------------------------------------------
# RealNoiseInjectionGenerator
# ---------------------------------------------------------------------------


@pytest.fixture()
def patch_pool_path(tmp_path: Path) -> Path:
    """Create a synthetic patch pool .npz file."""
    # 20 residual patches of size (3, 128, 128)
    residuals = np.random.normal(0, 0.02, (20, 3, 128, 128)).astype(np.float32)
    pool_path = tmp_path / "pool.npz"
    np.savez_compressed(pool_path, residuals=residuals)
    return pool_path


class TestRealNoiseInjectionGenerator:
    def test_output_shapes(self, patch_pool_path: Path) -> None:
        gen = RealNoiseInjectionGenerator(str(patch_pool_path))
        noisy, clean, sigma_map = gen(PATCH)
        _assert_output_shape(noisy, clean, sigma_map)

    def test_noisy_in_range(self, patch_pool_path: Path) -> None:
        gen = RealNoiseInjectionGenerator(str(patch_pool_path))
        noisy, _, _ = gen(PATCH)
        _assert_range(noisy)

    def test_different_calls_differ(self, patch_pool_path: Path) -> None:
        gen = RealNoiseInjectionGenerator(str(patch_pool_path))
        noisy1, _, _ = gen(PATCH)
        noisy2, _, _ = gen(PATCH)
        # With 20 patches there's very low probability of sampling the same twice
        assert not torch.allclose(noisy1, noisy2)

    def test_sigma_map_nonnegative(self, patch_pool_path: Path) -> None:
        gen = RealNoiseInjectionGenerator(str(patch_pool_path))
        _, _, sigma_map = gen(PATCH)
        assert (sigma_map >= 0).all()

    def test_protocol_conformance(self, patch_pool_path: Path) -> None:
        gen = RealNoiseInjectionGenerator(str(patch_pool_path))
        assert isinstance(gen, NoiseGenerator)


# ---------------------------------------------------------------------------
# RealRAWNoiseGenerator
# ---------------------------------------------------------------------------


@pytest.fixture()
def noise_profile_path(tmp_path: Path) -> Path:
    """Create a minimal noise profile JSON."""
    profile = {
        "camera": "TestCam",
        "iso_profiles": {
            "iso_800": {"K": 0.003, "sigma_r": 0.002},
            "iso_3200": {"K": 0.012, "sigma_r": 0.006},
        },
    }
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile))
    return path


class TestRealRAWNoiseGenerator:
    def test_output_shapes(self, noise_profile_path: Path) -> None:
        gen = RealRAWNoiseGenerator(str(noise_profile_path))
        noisy, clean, sigma_map = gen(PATCH)
        _assert_output_shape(noisy, clean, sigma_map)

    def test_noisy_in_range(self, noise_profile_path: Path) -> None:
        gen = RealRAWNoiseGenerator(str(noise_profile_path))
        noisy, _, _ = gen(PATCH)
        _assert_range(noisy)

    def test_empty_profile_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"camera": "X", "iso_profiles": {}}))
        with pytest.raises(ValueError, match="iso_profiles"):
            RealRAWNoiseGenerator(str(bad))

    def test_protocol_conformance(self, noise_profile_path: Path) -> None:
        gen = RealRAWNoiseGenerator(str(noise_profile_path))
        assert isinstance(gen, NoiseGenerator)


# ---------------------------------------------------------------------------
# MixedNoiseGenerator
# ---------------------------------------------------------------------------


class TestMixedNoiseGenerator:
    def test_output_shapes(self) -> None:
        gen = MixedNoiseGenerator.default()
        noisy, clean, sigma_map = gen(PATCH)
        _assert_output_shape(noisy, clean, sigma_map)

    def test_noisy_in_range(self) -> None:
        gen = MixedNoiseGenerator.default()
        noisy, _, _ = gen(PATCH)
        _assert_range(noisy)

    def test_weights_sum_to_one_two_generators(self) -> None:
        gen = MixedNoiseGenerator.default()
        assert abs(sum(gen._weights) - 1.0) < 1e-5

    def test_with_patch_pool(self, patch_pool_path: Path) -> None:
        gen = MixedNoiseGenerator.default(patch_pool=str(patch_pool_path))
        assert len(gen._generators) == 3
        assert abs(sum(gen._weights) - 1.0) < 1e-5

    def test_with_profile(self, noise_profile_path: Path) -> None:
        gen = MixedNoiseGenerator.default(profile_json=str(noise_profile_path))
        assert len(gen._generators) == 3

    def test_with_both(self, patch_pool_path: Path, noise_profile_path: Path) -> None:
        gen = MixedNoiseGenerator.default(
            patch_pool=str(patch_pool_path), profile_json=str(noise_profile_path)
        )
        assert len(gen._generators) == 4
        assert abs(sum(gen._weights) - 1.0) < 1e-5

    def test_empty_generators_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            MixedNoiseGenerator(generators=[])

    def test_mismatched_weights_raises(self) -> None:
        with pytest.raises(ValueError, match="len"):
            MixedNoiseGenerator(
                generators=[GaussianNoiseGenerator()],
                weights=[0.5, 0.5],
            )

    def test_protocol_conformance(self) -> None:
        gen = MixedNoiseGenerator.default()
        assert isinstance(gen, NoiseGenerator)
