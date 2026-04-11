"""Tests for dataset.py — PatchDataset and VideoSequenceDataset."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import PatchDataset, VideoSequenceDataset
from noise_generators import GaussianNoiseGenerator


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def image_dir(tmp_path: Path) -> Path:
    """Create a directory with 5 synthetic PNG images."""
    import imageio.v3 as iio

    d = tmp_path / "images"
    d.mkdir()
    rng = np.random.default_rng(0)
    for i in range(5):
        img = (rng.random((160, 160, 3)) * 255).astype(np.uint8)
        iio.imwrite(str(d / f"img_{i:04d}.png"), img)
    return d


@pytest.fixture()
def video_seq_dir(tmp_path: Path) -> Path:
    """Create a directory with one sequence of 10 frames."""
    import imageio.v3 as iio

    root = tmp_path / "sequences"
    seq = root / "seq_001"
    seq.mkdir(parents=True)
    rng = np.random.default_rng(1)
    for i in range(10):
        img = (rng.random((96, 96, 3)) * 255).astype(np.uint8)
        iio.imwrite(str(seq / f"frame_{i:06d}.png"), img)
    return root


# ---------------------------------------------------------------------------
# PatchDataset
# ---------------------------------------------------------------------------


class TestPatchDataset:
    def test_length(self, image_dir: Path) -> None:
        ds = PatchDataset([image_dir], patches_per_image=4, patch_size=64)
        assert len(ds) == 5 * 4

    def test_item_shapes(self, image_dir: Path) -> None:
        ds = PatchDataset([image_dir], patches_per_image=2, patch_size=64, augment=False)
        noisy, clean, sigma_map = ds[0]
        assert noisy.shape == (3, 64, 64)
        assert clean.shape == (3, 64, 64)
        assert sigma_map.shape == (3, 64, 64)

    def test_item_in_range(self, image_dir: Path) -> None:
        ds = PatchDataset([image_dir], patches_per_image=2, patch_size=64)
        noisy, clean, _ = ds[0]
        assert noisy.min().item() >= 0.0
        assert noisy.max().item() <= 1.0
        assert clean.min().item() >= 0.0
        assert clean.max().item() <= 1.0

    def test_custom_noise_generator(self, image_dir: Path) -> None:
        gen = GaussianNoiseGenerator(10.0 / 255.0, 10.0 / 255.0)
        ds = PatchDataset([image_dir], noise_generator=gen, patch_size=64, patches_per_image=2)
        _, _, sigma_map = ds[0]
        # Sigma should be near 10/255 everywhere
        assert abs(sigma_map.mean().item() - 10.0 / 255.0) < 1e-3

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="No images"):
            PatchDataset([empty])

    def test_num_images_property(self, image_dir: Path) -> None:
        ds = PatchDataset([image_dir])
        assert ds.num_images == 5

    def test_different_items_from_same_image_differ(self, image_dir: Path) -> None:
        ds = PatchDataset([image_dir], patches_per_image=4, patch_size=64, augment=True)
        noisy0, _, _ = ds[0]
        noisy1, _, _ = ds[1]
        # Different patches from same image should differ (with high probability)
        assert not torch.allclose(noisy0, noisy1)


# ---------------------------------------------------------------------------
# VideoSequenceDataset
# ---------------------------------------------------------------------------


class TestVideoSequenceDataset:
    def test_length(self, video_seq_dir: Path) -> None:
        # seq has 10 frames, T=5 → 6 clips, 4 patches each
        ds = VideoSequenceDataset([video_seq_dir], num_frames=5, patches_per_clip=4, patch_size=32)
        assert len(ds) == 6 * 4

    def test_item_shapes(self, video_seq_dir: Path) -> None:
        ds = VideoSequenceDataset(
            [video_seq_dir], num_frames=5, patches_per_clip=2, patch_size=32, augment=False
        )
        noisy, clean, sigma_map = ds[0]
        assert noisy.shape == (5, 3, 32, 32)
        assert clean.shape == (5, 3, 32, 32)
        assert sigma_map.shape == (5, 3, 32, 32)

    def test_item_in_range(self, video_seq_dir: Path) -> None:
        ds = VideoSequenceDataset([video_seq_dir], num_frames=5, patches_per_clip=2, patch_size=32)
        noisy, clean, _ = ds[0]
        assert noisy.min().item() >= 0.0
        assert noisy.max().item() <= 1.0

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        # Non-existent or empty sequence dirs → no clips → ValueError
        with pytest.raises(ValueError, match="No frame sequences"):
            VideoSequenceDataset([tmp_path / "nonexistent"])  # dir doesn't exist

    def test_num_clips_property(self, video_seq_dir: Path) -> None:
        ds = VideoSequenceDataset([video_seq_dir], num_frames=5)
        assert ds.num_clips == 6  # 10 - 5 + 1
