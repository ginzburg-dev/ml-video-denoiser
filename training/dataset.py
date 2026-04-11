"""PyTorch datasets for training the denoiser.

Two datasets are provided:

  PatchDataset
      Loads clean images from disk and extracts random patches on-the-fly.
      Noise is applied by a pluggable NoiseGenerator, so only clean images
      need to be stored.  Suitable for training NEFResidual.

  VideoSequenceDataset
      Loads short temporal clips (T consecutive frames) from video frame
      directories.  Applies spatially-consistent noise across all frames in
      a clip (same K / sigma_r per clip).  Suitable for training NEFTemporal.

Both return (noisy, clean, sigma_map) tuples.
"""

import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

from noise_generators import GaussianNoiseGenerator, MixedNoiseGenerator, NoiseGenerator


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".exr"}


def _collect_images(dirs: Sequence[str | Path]) -> list[Path]:
    """Recursively collect all image files from *dirs*, sorted."""
    paths: list[Path] = []
    for d in dirs:
        root = Path(d)
        if not root.exists():
            raise FileNotFoundError(f"Image directory not found: {root}")
        for ext in _IMAGE_EXTENSIONS:
            paths.extend(root.rglob(f"*{ext}"))
            paths.extend(root.rglob(f"*{ext.upper()}"))
    return sorted(set(paths))


def _load_image(path: Path) -> np.ndarray:
    """Load *path* as a float32 HWC array normalised to [0, 1]."""
    import imageio.v3 as iio

    img = iio.imread(str(path))
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if img.shape[2] > 3:
        img = img[:, :, :3]  # drop alpha

    # Normalise
    if img.max() > 1.5:
        dtype_max = 255.0 if img.max() <= 255.5 else 65535.0
        img /= dtype_max
    return img


def _hwc_to_tensor(arr: np.ndarray) -> Tensor:
    """Convert (H, W, C) float32 numpy array to (C, H, W) float32 Tensor."""
    return torch.from_numpy(arr.transpose(2, 0, 1))


def _random_crop(
    img: np.ndarray, patch_size: int
) -> np.ndarray:
    """Return a random (patch_size, patch_size, C) crop from *img*."""
    h, w, _ = img.shape
    if h < patch_size or w < patch_size:
        # Pad with reflection if the image is smaller than the patch
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        h, w, _ = img.shape
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)
    return img[top : top + patch_size, left : left + patch_size, :]


def _augment(img: np.ndarray) -> np.ndarray:
    """Apply random flip / rotation augmentation (in-place safe)."""
    if random.random() < 0.5:
        img = img[::-1, :, :]      # vertical flip
    if random.random() < 0.5:
        img = img[:, ::-1, :]      # horizontal flip
    k = random.randint(0, 3)
    if k:
        img = np.rot90(img, k=k)   # rotate 90 * k degrees
    return np.ascontiguousarray(img)


# ---------------------------------------------------------------------------
# PatchDataset
# ---------------------------------------------------------------------------


class PatchDataset(Dataset):
    """Random-patch dataset for spatial image denoising.

    Loads clean images lazily from *image_dirs*, extracts a random
    (patch_size × patch_size) crop, applies noise via *noise_generator*, and
    returns (noisy_patch, clean_patch, sigma_map).

    Args:
        image_dirs: One or more directories containing clean training images.
            Subdirectories are searched recursively.
        noise_generator: Callable conforming to the NoiseGenerator protocol.
            Defaults to MixedNoiseGenerator.default() (Gaussian + Poisson-Gaussian).
        patch_size: Spatial size of extracted patches (default: 128).
        patches_per_image: How many patches to virtually draw per image per
            epoch.  This multiplies the effective dataset length without
            re-reading files.  (default: 64)
        augment: Whether to apply random flip / rotation augmentation.
    """

    def __init__(
        self,
        image_dirs: Sequence[str | Path],
        noise_generator: Optional[NoiseGenerator] = None,
        patch_size: int = 128,
        patches_per_image: int = 64,
        augment: bool = True,
    ) -> None:
        self._paths = _collect_images(image_dirs)
        if not self._paths:
            raise ValueError(f"No images found in: {list(image_dirs)}")
        self._noise_gen = noise_generator or MixedNoiseGenerator.default()
        self._patch_size = patch_size
        self._patches_per_image = patches_per_image
        self._augment = augment

    def __len__(self) -> int:
        return len(self._paths) * self._patches_per_image

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        img_idx = idx // self._patches_per_image
        img = _load_image(self._paths[img_idx])
        patch = _random_crop(img, self._patch_size)
        if self._augment:
            patch = _augment(patch)
        clean = _hwc_to_tensor(patch)  # (C, H, W)
        noisy, clean, sigma_map = self._noise_gen(clean)
        return noisy, clean, sigma_map

    @property
    def num_images(self) -> int:
        return len(self._paths)


# ---------------------------------------------------------------------------
# VideoSequenceDataset
# ---------------------------------------------------------------------------


class VideoSequenceDataset(Dataset):
    """Temporal clip dataset for video denoising (NEFTemporal).

    Each item is a clip of *num_frames* consecutive frames extracted from a
    video frame directory.  Noise is applied consistently across the clip
    (same gain K and read noise sigma_r per clip) so the model sees realistic
    temporal noise correlation.

    Frame directories are expected to contain numbered image files:
        <video_root>/<sequence_name>/<frame_000001.png>, …

    Args:
        sequence_dirs: Directories where each sub-directory is a frame sequence.
        noise_generator: Noise generator to apply.  For consistent temporal
            noise, wrap the desired generator in ConsistentClipNoiseGenerator.
            If None, defaults to GaussianNoiseGenerator.
        num_frames: Temporal clip length (default: 5).
        patch_size: Spatial patch size (default: 64 — smaller than spatial
            because memory cost is num_frames ×).
        patches_per_clip: Virtual patches per clip per epoch (default: 16).
        augment: Random spatial flip / rotation (applied identically across
            all frames in the clip).
    """

    def __init__(
        self,
        sequence_dirs: Sequence[str | Path],
        noise_generator: Optional[NoiseGenerator] = None,
        num_frames: int = 5,
        patch_size: int = 64,
        patches_per_clip: int = 16,
        augment: bool = True,
    ) -> None:
        self._clips = self._collect_clips(sequence_dirs, num_frames)
        if not self._clips:
            raise ValueError(f"No frame sequences found in: {list(sequence_dirs)}")
        self._noise_gen = noise_generator or GaussianNoiseGenerator(0.0, 50.0 / 255.0)
        self._num_frames = num_frames
        self._patch_size = patch_size
        self._patches_per_clip = patches_per_clip
        self._augment = augment

    @staticmethod
    def _collect_clips(
        sequence_dirs: Sequence[str | Path], num_frames: int
    ) -> list[list[Path]]:
        """Find all valid consecutive-frame windows across all sequence dirs."""
        clips: list[list[Path]] = []
        for seq_dir in sequence_dirs:
            root = Path(seq_dir)
            if not root.exists():
                continue  # skip missing dirs; ValueError raised below if nothing found
            for sub in sorted(root.iterdir()):
                if not sub.is_dir():
                    continue
                frames = sorted(
                    p for p in sub.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
                )
                for i in range(len(frames) - num_frames + 1):
                    clips.append(frames[i : i + num_frames])
        return clips

    def __len__(self) -> int:
        return len(self._clips) * self._patches_per_clip

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return (noisy_clip, clean_clip, sigma_map).

        Shapes:
            noisy_clip:  (T, C, H, W)
            clean_clip:  (T, C, H, W)
            sigma_map:   (T, C, H, W)
        """
        clip_idx = idx // self._patches_per_clip
        frame_paths = self._clips[clip_idx]

        # Load all frames
        frames = [_load_image(p) for p in frame_paths]
        # Determine crop position once (same for all frames)
        h, w, _ = frames[0].shape
        ps = self._patch_size
        top = random.randint(0, max(0, h - ps))
        left = random.randint(0, max(0, w - ps))
        # Determine augmentation seed once
        flip_v = self._augment and random.random() < 0.5
        flip_h = self._augment and random.random() < 0.5
        rot_k = random.randint(0, 3) if self._augment else 0

        noisy_frames, clean_frames, sigma_frames = [], [], []
        for frame in frames:
            # Pad if necessary
            if h < ps or w < ps:
                pad_h = max(0, ps - h)
                pad_w = max(0, ps - w)
                frame = np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            patch = frame[top : top + ps, left : left + ps, :]
            if flip_v:
                patch = patch[::-1, :, :]
            if flip_h:
                patch = patch[:, ::-1, :]
            if rot_k:
                patch = np.rot90(patch, k=rot_k)
            patch = np.ascontiguousarray(patch)
            clean_t = _hwc_to_tensor(patch)
            noisy_t, clean_t, sigma_t = self._noise_gen(clean_t)
            noisy_frames.append(noisy_t)
            clean_frames.append(clean_t)
            sigma_frames.append(sigma_t)

        # Stack: (T, C, H, W)
        return (
            torch.stack(noisy_frames),
            torch.stack(clean_frames),
            torch.stack(sigma_frames),
        )

    @property
    def num_clips(self) -> int:
        return len(self._clips)
