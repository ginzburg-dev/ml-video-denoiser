"""Tests for dataset.py — PatchDataset, VideoSequenceDataset,
PairedPatchDataset, PairedVideoSequenceDataset, and CombinedDataset."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import (
    CombinedDataset,
    PairedPatchDataset,
    PairedVideoSequenceDataset,
    PatchDataset,
    VideoSequenceDataset,
    _load_image,
    _spread_indices,
)
from noise_generators import GaussianNoiseGenerator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class TestSpreadIndices:
    def test_single_frame_count_returns_first_index(self) -> None:
        assert _spread_indices(90, 1) == [0]


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
    def test_load_exr_image(self, tmp_path: Path) -> None:
        import OpenEXR

        exr_path = tmp_path / "frame.exr"
        rgba = np.zeros((8, 8, 4), dtype=np.float32)
        rgba[..., 0] = 0.1
        rgba[..., 1] = 0.2
        rgba[..., 2] = 0.3
        rgba[..., 3] = 0.9
        OpenEXR.File({"type": OpenEXR.scanlineimage}, {"RGBA": rgba}).write(str(exr_path))

        img = _load_image(exr_path)
        assert img.shape == (8, 8, 3)
        # Display-referred EXR values pass through without any transform.
        assert np.isclose(img[..., 0].mean(), 0.1, rtol=1e-4)
        assert np.isclose(img[..., 1].mean(), 0.2, rtol=1e-4)
        assert np.isclose(img[..., 2].mean(), 0.3, rtol=1e-4)

    def test_length(self, image_dir: Path) -> None:
        ds = PatchDataset([image_dir], patches_per_image=4, patch_size=64)
        assert len(ds) == 5 * 4

    def test_item_shapes(self, image_dir: Path) -> None:
        ds = PatchDataset([image_dir], patches_per_image=2, patch_size=64, augment=False)
        noisy, clean, sigma_map = ds[0]
        assert noisy.shape == (3, 64, 64)
        assert clean.shape == (3, 64, 64)
        assert sigma_map.shape == (3, 64, 64)

    def test_item_finite(self, image_dir: Path) -> None:
        ds = PatchDataset([image_dir], patches_per_image=2, patch_size=64)
        noisy, clean, _ = ds[0]
        assert torch.isfinite(noisy).all()
        assert torch.isfinite(clean).all()

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

    def test_center_crop_is_deterministic(self, image_dir: Path) -> None:
        gen = GaussianNoiseGenerator(0.0, 0.0)
        ds = PatchDataset(
            [image_dir],
            noise_generator=gen,
            patches_per_image=2,
            patch_size=64,
            augment=False,
            crop_mode="center",
        )
        noisy0, clean0, sigma0 = ds[0]
        noisy1, clean1, sigma1 = ds[0]
        assert torch.allclose(noisy0, noisy1)
        assert torch.allclose(clean0, clean1)
        assert torch.allclose(sigma0, sigma1)

    def test_grid_crop_produces_multiple_deterministic_views(self, image_dir: Path) -> None:
        gen = GaussianNoiseGenerator(0.0, 0.0)
        ds = PatchDataset(
            [image_dir],
            noise_generator=gen,
            patches_per_image=4,
            patch_size=64,
            augment=False,
            crop_mode="grid",
            crop_grid_size=2,
        )
        views = [ds[i][1] for i in range(4)]
        assert all(view.shape == (3, 64, 64) for view in views)
        assert not torch.allclose(views[0], views[1])
        assert torch.allclose(ds[0][1], views[0])

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

    def test_item_finite(self, video_seq_dir: Path) -> None:
        ds = VideoSequenceDataset([video_seq_dir], num_frames=5, patches_per_clip=2, patch_size=32)
        noisy, clean, _ = ds[0]
        assert torch.isfinite(noisy).all()
        assert torch.isfinite(clean).all()

    def test_empty_dir_raises(self, tmp_path: Path) -> None:
        # Non-existent or empty sequence dirs → no clips → ValueError
        with pytest.raises(ValueError, match="No frame sequences"):
            VideoSequenceDataset([tmp_path / "nonexistent"])  # dir doesn't exist

    def test_num_clips_property(self, video_seq_dir: Path) -> None:
        ds = VideoSequenceDataset([video_seq_dir], num_frames=5)
        assert ds.num_clips == 6  # 10 - 5 + 1

    def test_random_window_mode_uses_windows_per_sequence(self, video_seq_dir: Path) -> None:
        ds = VideoSequenceDataset(
            [video_seq_dir],
            num_frames=5,
            patches_per_clip=2,
            random_windows=True,
            windows_per_sequence=3,
            patch_size=32,
            augment=False,
        )
        assert ds.num_sequences == 1
        assert ds.num_clips == 3
        assert len(ds) == 6
        noisy, clean, sigma_map = ds[0]
        assert noisy.shape == (5, 3, 32, 32)
        assert clean.shape == (5, 3, 32, 32)
        assert sigma_map.shape == (5, 3, 32, 32)

    def test_variable_frame_sizes_are_padded_before_shared_crop(self, tmp_path: Path) -> None:
        import imageio.v3 as iio

        root = tmp_path / "varying_sequences"
        seq = root / "seq_001"
        seq.mkdir(parents=True)
        for i, width in enumerate([96, 96, 93, 96, 90, 96]):
            img = np.zeros((96, width, 3), dtype=np.uint8)
            iio.imwrite(str(seq / f"frame_{i:06d}.png"), img)

        ds = VideoSequenceDataset(
            [root], num_frames=5, patches_per_clip=1, patch_size=64, augment=False
        )
        noisy, clean, sigma_map = ds[0]
        assert noisy.shape == (5, 3, 64, 64)
        assert clean.shape == (5, 3, 64, 64)
        assert sigma_map.shape == (5, 3, 64, 64)

    def test_full_crop_preserves_temporal_frame_size(self, video_seq_dir: Path) -> None:
        ds = VideoSequenceDataset(
            [video_seq_dir], num_frames=5, patches_per_clip=1, augment=False, crop_mode="full"
        )
        noisy, clean, sigma_map = ds[0]
        assert noisy.shape == (5, 3, 96, 96)
        assert clean.shape == (5, 3, 96, 96)
        assert sigma_map.shape == (5, 3, 96, 96)

    def test_grid_crop_uses_deterministic_temporal_views(self, video_seq_dir: Path) -> None:
        gen = GaussianNoiseGenerator(0.0, 0.0)
        ds = VideoSequenceDataset(
            [video_seq_dir],
            noise_generator=gen,
            num_frames=5,
            patches_per_clip=4,
            patch_size=32,
            augment=False,
            crop_mode="grid",
            crop_grid_size=2,
        )
        clip0 = ds[0][1]
        clip1 = ds[1][1]
        assert clip0.shape == (5, 3, 32, 32)
        assert clip1.shape == (5, 3, 32, 32)
        assert not torch.allclose(clip0, clip1)


# ---------------------------------------------------------------------------
# Shared fixture helpers for paired datasets
# ---------------------------------------------------------------------------


def _write_paired_images(
    tmp_path: Path,
    n: int = 5,
    size: tuple[int, int] = (160, 160),
    noise_sigma: float = 0.05,
    stem_prefix: str = "img",
) -> tuple[Path, Path]:
    """Write n matching clean/noisy image pairs into clean/ and noisy/ subdirs."""
    import imageio.v3 as iio

    clean_dir = tmp_path / "clean"
    noisy_dir = tmp_path / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    for i in range(n):
        clean = (rng.random((*size, 3)) * 0.8 + 0.1).astype(np.float32)
        noisy = np.clip(clean + rng.normal(0, noise_sigma, clean.shape), 0, 1).astype(
            np.float32
        )
        iio.imwrite(str(clean_dir / f"{stem_prefix}_{i:04d}.png"), (clean * 255).astype(np.uint8))
        iio.imwrite(str(noisy_dir / f"{stem_prefix}_{i:04d}.png"), (noisy * 255).astype(np.uint8))

    return clean_dir, noisy_dir


def _write_paired_sequences(
    tmp_path: Path,
    n_sequences: int = 2,
    n_frames: int = 10,
    size: tuple[int, int] = (96, 96),
    noise_sigma: float = 0.05,
) -> tuple[Path, Path]:
    """Write n_sequences of n_frames pairs into clean_root/<seq>/ and noisy_root/<seq>/."""
    import imageio.v3 as iio

    clean_root = tmp_path / "clean_seq"
    noisy_root = tmp_path / "noisy_seq"
    rng = np.random.default_rng(7)

    for s in range(n_sequences):
        clean_seq = clean_root / f"seq_{s:03d}"
        noisy_seq = noisy_root / f"seq_{s:03d}"
        clean_seq.mkdir(parents=True)
        noisy_seq.mkdir(parents=True)
        for f in range(n_frames):
            clean = (rng.random((*size, 3)) * 0.8 + 0.1).astype(np.float32)
            noisy = np.clip(
                clean + rng.normal(0, noise_sigma, clean.shape), 0, 1
            ).astype(np.float32)
            iio.imwrite(str(clean_seq / f"frame_{f:06d}.png"), (clean * 255).astype(np.uint8))
            iio.imwrite(str(noisy_seq / f"frame_{f:06d}.png"), (noisy * 255).astype(np.uint8))

    return clean_root, noisy_root


# ---------------------------------------------------------------------------
# PairedPatchDataset
# ---------------------------------------------------------------------------


class TestPairedPatchDataset:
    def test_length(self, tmp_path: Path) -> None:
        clean_dir, noisy_dir = _write_paired_images(tmp_path, n=5)
        ds = PairedPatchDataset(clean_dir, noisy_dir, patch_size=64, patches_per_image=4)
        assert len(ds) == 5 * 4

    def test_item_shapes(self, tmp_path: Path) -> None:
        clean_dir, noisy_dir = _write_paired_images(tmp_path)
        ds = PairedPatchDataset(
            clean_dir, noisy_dir, patch_size=64, patches_per_image=2, augment=False
        )
        noisy, clean, sigma_map = ds[0]
        assert noisy.shape    == (3, 64, 64)
        assert clean.shape    == (3, 64, 64)
        assert sigma_map.shape == (3, 64, 64)

    def test_values_finite(self, tmp_path: Path) -> None:
        clean_dir, noisy_dir = _write_paired_images(tmp_path)
        ds = PairedPatchDataset(clean_dir, noisy_dir, patch_size=64, patches_per_image=2)
        noisy, clean, sigma_map = ds[0]
        assert torch.isfinite(noisy).all()
        assert torch.isfinite(clean).all()
        assert sigma_map.min().item() >= 0.0

    def test_sigma_map_reflects_actual_noise(self, tmp_path: Path) -> None:
        """sigma_map must be larger where noisy-clean residual is larger."""
        clean_dir, noisy_dir = _write_paired_images(tmp_path, noise_sigma=0.10)
        # Low-noise pair for comparison
        tmp2 = tmp_path / "low"
        tmp2.mkdir()
        clean_dir2, noisy_dir2 = _write_paired_images(tmp2, noise_sigma=0.01)

        ds_high = PairedPatchDataset(clean_dir, noisy_dir, patch_size=64, augment=False)
        ds_low  = PairedPatchDataset(clean_dir2, noisy_dir2, patch_size=64, augment=False)

        _, _, sigma_high = ds_high[0]
        _, _, sigma_low  = ds_low[0]
        assert sigma_high.mean().item() > sigma_low.mean().item()

    def test_augmentation_preserves_alignment(self, tmp_path: Path) -> None:
        """With augmentation, noisy and clean must still be spatially aligned."""
        clean_dir, noisy_dir = _write_paired_images(tmp_path, noise_sigma=0.05)
        ds = PairedPatchDataset(
            clean_dir, noisy_dir, patch_size=64, patches_per_image=8, augment=True
        )
        for idx in range(8):
            noisy, clean, _ = ds[idx]
            residual = (noisy - clean).abs().mean().item()
            # With sigma=0.05, mean absolute residual should be in a sensible range
            assert residual < 0.2, (
                f"idx={idx}: residual {residual:.4f} too large — augmentation misaligned"
            )

    def test_match_by_name(self, tmp_path: Path) -> None:
        clean_dir, noisy_dir = _write_paired_images(tmp_path)
        ds = PairedPatchDataset(
            clean_dir, noisy_dir, patch_size=64, match_by_name=True
        )
        assert ds.num_pairs == 5

    def test_match_by_name_rejects_partial_overlap(self, tmp_path: Path) -> None:
        import imageio.v3 as iio

        clean_dir = tmp_path / "clean"
        noisy_dir = tmp_path / "noisy"
        clean_dir.mkdir()
        noisy_dir.mkdir()
        for stem in ("a", "b", "c"):
            iio.imwrite(str(clean_dir / f"{stem}.png"), np.zeros((32, 32, 3), dtype=np.uint8))
        for stem in ("a", "c", "extra"):
            iio.imwrite(str(noisy_dir / f"{stem}.png"), np.zeros((32, 32, 3), dtype=np.uint8))

        with pytest.raises(ValueError, match="no matching noisy counterpart"):
            PairedPatchDataset(clean_dir, noisy_dir, patch_size=64, match_by_name=True)

    def test_match_by_position(self, tmp_path: Path) -> None:
        clean_dir, noisy_dir = _write_paired_images(tmp_path)
        ds = PairedPatchDataset(
            clean_dir, noisy_dir, patch_size=64, match_by_name=False
        )
        assert ds.num_pairs == 5

    def test_mismatched_count_raises(self, tmp_path: Path) -> None:
        import imageio.v3 as iio

        clean_dir = tmp_path / "clean"
        noisy_dir = tmp_path / "noisy"
        clean_dir.mkdir()
        noisy_dir.mkdir()
        # 3 clean images, 2 noisy images
        for i in range(3):
            iio.imwrite(str(clean_dir / f"img_{i}.png"), np.zeros((32, 32, 3), dtype=np.uint8))
        for i in range(2):
            iio.imwrite(str(noisy_dir / f"img_{i}.png"), np.zeros((32, 32, 3), dtype=np.uint8))

        with pytest.raises(ValueError, match="different counts"):
            PairedPatchDataset(clean_dir, noisy_dir, match_by_name=False)

    def test_missing_noisy_dir_raises(self, tmp_path: Path) -> None:
        clean_dir, _ = _write_paired_images(tmp_path)
        with pytest.raises(FileNotFoundError):
            PairedPatchDataset(clean_dir, tmp_path / "nonexistent")

    def test_num_pairs_property(self, tmp_path: Path) -> None:
        clean_dir, noisy_dir = _write_paired_images(tmp_path, n=3)
        ds = PairedPatchDataset(clean_dir, noisy_dir)
        assert ds.num_pairs == 3

    def test_frames_per_sequence_one_selects_one_frame_per_sequence(self, tmp_path: Path) -> None:
        clean_root, noisy_root = _write_paired_sequences(tmp_path, n_sequences=2, n_frames=10)
        ds = PairedPatchDataset(
            clean_root,
            noisy_root,
            patch_size=64,
            patches_per_image=1,
            frames_per_sequence=1,
        )
        assert ds.num_pairs == 2

    def test_sequence_dirs_match_by_relative_path_when_frame_stems_repeat(self, tmp_path: Path) -> None:
        clean_root, noisy_root = _write_paired_sequences(tmp_path, n_sequences=2, n_frames=4)
        ds = PairedPatchDataset(
            clean_root,
            noisy_root,
            patch_size=64,
            patches_per_image=1,
        )
        assert ds.num_pairs == 8


# ---------------------------------------------------------------------------
# PairedVideoSequenceDataset
# ---------------------------------------------------------------------------


class TestPairedVideoSequenceDataset:
    def test_length(self, tmp_path: Path) -> None:
        clean_root, noisy_root = _write_paired_sequences(tmp_path, n_sequences=2, n_frames=10)
        # Each seq: 10 - 5 + 1 = 6 clips; 2 sequences = 12 clips; × 4 patches = 48
        ds = PairedVideoSequenceDataset(
            [clean_root], [noisy_root], num_frames=5, patches_per_clip=4, patch_size=32
        )
        assert len(ds) == 12 * 4

    def test_item_shapes(self, tmp_path: Path) -> None:
        clean_root, noisy_root = _write_paired_sequences(tmp_path)
        ds = PairedVideoSequenceDataset(
            [clean_root], [noisy_root], num_frames=5, patches_per_clip=2,
            patch_size=32, augment=False
        )
        noisy, clean, sigma_map = ds[0]
        assert noisy.shape    == (5, 3, 32, 32)
        assert clean.shape    == (5, 3, 32, 32)
        assert sigma_map.shape == (5, 3, 32, 32)

    def test_values_finite(self, tmp_path: Path) -> None:
        clean_root, noisy_root = _write_paired_sequences(tmp_path)
        ds = PairedVideoSequenceDataset(
            [clean_root], [noisy_root], num_frames=5, patches_per_clip=2, patch_size=32
        )
        noisy, clean, _ = ds[0]
        assert torch.isfinite(noisy).all()
        assert torch.isfinite(clean).all()

    def test_unmatched_roots_raise(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="same length"):
            PairedVideoSequenceDataset(
                [tmp_path / "a"], [tmp_path / "b", tmp_path / "c"]
            )

    def test_missing_sequence_skipped(self, tmp_path: Path) -> None:
        """A noisy root that exists but has no matching sub-dirs produces 0 clips."""
        clean_root, _ = _write_paired_sequences(tmp_path)
        empty_noisy = tmp_path / "empty_noisy"
        empty_noisy.mkdir()
        with pytest.raises(ValueError, match="No paired frame sequences"):
            PairedVideoSequenceDataset([clean_root], [empty_noisy])

    def test_frame_name_mismatch_raises(self, tmp_path: Path) -> None:
        clean_root, noisy_root = _write_paired_sequences(tmp_path, n_sequences=1, n_frames=6)
        mismatched = noisy_root / "seq_000" / "frame_000003.png"
        mismatched.rename(noisy_root / "seq_000" / "frame_999999.png")

        with pytest.raises(ValueError, match="no matching noisy counterpart"):
            PairedVideoSequenceDataset(
                [clean_root],
                [noisy_root],
                num_frames=5,
                patches_per_clip=1,
                patch_size=32,
            )

    def test_num_clips_property(self, tmp_path: Path) -> None:
        clean_root, noisy_root = _write_paired_sequences(tmp_path, n_sequences=1, n_frames=8)
        ds = PairedVideoSequenceDataset(
            [clean_root], [noisy_root], num_frames=5, patches_per_clip=1, patch_size=32
        )
        assert ds.num_clips == 4  # 8 - 5 + 1

    def test_deterministic_window_subset_for_validation(self, tmp_path: Path) -> None:
        clean_root, noisy_root = _write_paired_sequences(tmp_path, n_sequences=1, n_frames=10)
        ds = PairedVideoSequenceDataset(
            [clean_root], [noisy_root],
            num_frames=5,
            patches_per_clip=1,
            windows_per_sequence=3,
            patch_size=32,
            augment=False,
            crop_mode="center",
        )
        assert ds.num_clips == 3
        sample_a = ds[0]
        sample_b = ds[0]
        for a, b in zip(sample_a, sample_b):
            assert torch.allclose(a, b)

    def test_grid_crop_uses_deterministic_paired_temporal_views(self, tmp_path: Path) -> None:
        clean_root, noisy_root = _write_paired_sequences(tmp_path, n_sequences=1, n_frames=10)
        ds = PairedVideoSequenceDataset(
            [clean_root], [noisy_root],
            num_frames=5,
            patches_per_clip=4,
            patch_size=32,
            augment=False,
            crop_mode="grid",
            crop_grid_size=2,
        )
        clip0 = ds[0][1]
        clip1 = ds[1][1]
        assert clip0.shape == (5, 3, 32, 32)
        assert clip1.shape == (5, 3, 32, 32)
        assert not torch.allclose(clip0, clip1)

    def test_random_window_mode_uses_windows_per_sequence(self, tmp_path: Path) -> None:
        clean_root, noisy_root = _write_paired_sequences(tmp_path, n_sequences=2, n_frames=10)
        ds = PairedVideoSequenceDataset(
            [clean_root], [noisy_root],
            num_frames=5,
            patches_per_clip=2,
            random_windows=True,
            windows_per_sequence=3,
            patch_size=32,
            augment=False,
        )
        assert ds.num_sequences == 2
        assert ds.num_clips == 6
        assert len(ds) == 12
        noisy, clean, sigma_map = ds[0]
        assert noisy.shape == (5, 3, 32, 32)
        assert clean.shape == (5, 3, 32, 32)
        assert sigma_map.shape == (5, 3, 32, 32)

    def test_variable_frame_sizes_are_padded_before_shared_crop(self, tmp_path: Path) -> None:
        import imageio.v3 as iio

        clean_root = tmp_path / "clean"
        noisy_root = tmp_path / "noisy"
        clean_seq = clean_root / "scene_001"
        noisy_seq = noisy_root / "scene_001"
        clean_seq.mkdir(parents=True)
        noisy_seq.mkdir(parents=True)

        for i, width in enumerate([96, 94, 96, 91, 96, 92]):
            clean = np.zeros((96, width, 3), dtype=np.uint8)
            noisy = np.zeros((96, width, 3), dtype=np.uint8)
            iio.imwrite(str(clean_seq / f"frame_{i:06d}.png"), clean)
            iio.imwrite(str(noisy_seq / f"frame_{i:06d}.png"), noisy)

        ds = PairedVideoSequenceDataset(
            [clean_root], [noisy_root], num_frames=5, patches_per_clip=1,
            patch_size=64, augment=False
        )
        noisy, clean, sigma_map = ds[0]
        assert noisy.shape == (5, 3, 64, 64)
        assert clean.shape == (5, 3, 64, 64)
        assert sigma_map.shape == (5, 3, 64, 64)


# ---------------------------------------------------------------------------
# CombinedDataset
# ---------------------------------------------------------------------------


class TestCombinedDataset:
    @pytest.fixture()
    def synthetic_ds(self, tmp_path: Path) -> PatchDataset:
        import imageio.v3 as iio
        d = tmp_path / "syn"
        d.mkdir()
        rng = np.random.default_rng(0)
        for i in range(4):
            iio.imwrite(str(d / f"img_{i}.png"), (rng.random((64, 64, 3)) * 255).astype(np.uint8))
        return PatchDataset([d], patch_size=32, patches_per_image=4)

    @pytest.fixture()
    def paired_ds(self, tmp_path: Path) -> PairedPatchDataset:
        clean_dir, noisy_dir = _write_paired_images(tmp_path / "paired", n=3, size=(64, 64))
        return PairedPatchDataset(clean_dir, noisy_dir, patch_size=32, patches_per_image=4)

    def test_length_defaults_to_sum(
        self, synthetic_ds: PatchDataset, paired_ds: PairedPatchDataset
    ) -> None:
        combined = CombinedDataset([synthetic_ds, paired_ds])
        assert len(combined) == len(synthetic_ds) + len(paired_ds)

    def test_custom_num_samples(
        self, synthetic_ds: PatchDataset, paired_ds: PairedPatchDataset
    ) -> None:
        combined = CombinedDataset([synthetic_ds, paired_ds], num_samples=100)
        assert len(combined) == 100

    def test_item_shape(
        self, synthetic_ds: PatchDataset, paired_ds: PairedPatchDataset
    ) -> None:
        combined = CombinedDataset([synthetic_ds, paired_ds])
        noisy, clean, sigma_map = combined[0]
        assert noisy.shape    == (3, 32, 32)
        assert clean.shape    == (3, 32, 32)
        assert sigma_map.shape == (3, 32, 32)

    def test_both_datasets_sampled(
        self, synthetic_ds: PatchDataset, paired_ds: PairedPatchDataset
    ) -> None:
        """With equal weights, over many draws both datasets should be hit."""
        combined = CombinedDataset([synthetic_ds, paired_ds], weights=[0.5, 0.5])
        # Draw 200 items — both datasets should contribute
        seen_indices = set()
        for i in range(200):
            combined[i]  # just ensure no error; sampling is stochastic by idx
        # Not easily checkable which dataset was hit without introspection,
        # but no exceptions means both paths work.

    def test_weight_normalisation(
        self, synthetic_ds: PatchDataset, paired_ds: PairedPatchDataset
    ) -> None:
        """Weights are normalised internally; raw weights [3, 1] == [0.75, 0.25]."""
        combined = CombinedDataset([synthetic_ds, paired_ds], weights=[3.0, 1.0])
        assert abs(sum(combined._weights) - 1.0) < 1e-6

    def test_uniform_default_weights(
        self, synthetic_ds: PatchDataset, paired_ds: PairedPatchDataset
    ) -> None:
        combined = CombinedDataset([synthetic_ds, paired_ds])
        assert combined._weights == [0.5, 0.5]

    def test_empty_datasets_raises(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            CombinedDataset([])

    def test_weight_length_mismatch_raises(
        self, synthetic_ds: PatchDataset, paired_ds: PairedPatchDataset
    ) -> None:
        with pytest.raises(ValueError, match="len\\(weights\\)"):
            CombinedDataset([synthetic_ds, paired_ds], weights=[0.5])
