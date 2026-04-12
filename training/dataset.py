"""PyTorch datasets for training the denoiser.

Four datasets are provided:

  PatchDataset
      Loads clean images and synthesises noise on-the-fly via a NoiseGenerator.
      No paired data required.  Suitable for training NEFResidual.

  VideoSequenceDataset
      Loads temporal clips and synthesises noise consistently across frames.
      Suitable for training NEFTemporal.

  PairedPatchDataset
      Loads *matching* clean/noisy image pairs from two parallel directories.
      Noise is already present in the noisy images — no synthesis needed.
      sigma_map is estimated from the actual noise residual |noisy − clean|.

  PairedVideoSequenceDataset
      Same as PairedPatchDataset but for temporal clips.  Requires two
      parallel directory trees with matching sub-directory names.

  CombinedDataset
      Mixes any list of datasets with explicit sampling weights so that
      synthetic and paired data can be drawn from a single DataLoader.

All datasets return (noisy, clean, sigma_map) tuples.
"""

import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
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
    if path.suffix.lower() == ".exr":
        img = _load_exr_image(path)
    else:
        with Image.open(path) as pil_img:
            img = np.array(pil_img)

    source_dtype = img.dtype
    img = img.astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if img.shape[2] > 3:
        img = img[:, :, :3]  # drop alpha

    # Normalize integer-backed images; float-backed inputs like EXR are assumed
    # to already be in scene-linear units and are clipped to the model range.
    if not np.issubdtype(source_dtype, np.floating) and img.max() > 1.5:
        dtype_max = 255.0 if img.max() <= 255.5 else 65535.0
        img /= dtype_max
    return np.clip(img, 0.0, 1.0)


def _load_exr_image(path: Path) -> np.ndarray:
    """Load an EXR image as float32 HWC using the bundled OpenEXR dependency."""
    import OpenEXR

    with OpenEXR.File(str(path)) as exr:
        channels = exr.parts[0].channels
        for key in ("RGBA", "RGB"):
            channel = channels.get(key)
            if channel is not None:
                return np.asarray(channel.pixels, dtype=np.float32)

        names = [name for name in ("R", "G", "B", "A", "Y") if name in channels]
        if not names:
            available = ", ".join(sorted(channels))
            raise ValueError(f"Unsupported EXR channel layout in {path}: {available}")

        planes = [np.asarray(channels[name].pixels, dtype=np.float32) for name in names]
        return np.stack(planes, axis=-1)


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


# ---------------------------------------------------------------------------
# Shared helper for paired datasets
# ---------------------------------------------------------------------------


def _local_std_sigma(noise: Tensor, window: int = 7) -> Tensor:
    """Estimate per-pixel noise std from a noise residual tensor.

    Uses a sliding-window variance over a (window × window) neighbourhood,
    matching the approach used by RealNoiseInjectionGenerator.

    Args:
        noise: (C, H, W) float32 tensor of noise residuals (noisy − clean).
        window: Kernel size for the local variance estimate.

    Returns:
        (C, H, W) sigma_map where each value is the local noise std.
    """
    x = noise.unsqueeze(0)          # → (1, C, H, W)
    pad = window // 2
    x_pad = torch.nn.functional.pad(x, (pad, pad, pad, pad), mode="reflect")
    patches = x_pad.unfold(2, window, 1).unfold(3, window, 1)  # (1,C,H,W,k,k)
    sigma = patches.var(dim=(-2, -1), unbiased=False).sqrt()   # (1, C, H, W)
    return sigma.squeeze(0)         # → (C, H, W)


def _match_pairs(
    clean_dir: Path, noisy_dir: Path, match_by_name: bool
) -> list[tuple[Path, Path]]:
    """Collect matched (clean, noisy) image path pairs from two directories.

    Args:
        clean_dir: Root directory of clean images (searched recursively).
        noisy_dir: Root directory of noisy images (searched recursively).
        match_by_name: If True, match files whose stems are identical.
            If False, match by sorted position — clean[i] ↔ noisy[i].

    Returns:
        Sorted list of (clean_path, noisy_path) pairs.

    Raises:
        ValueError: If no pairs are found or counts differ (position matching).
    """
    clean_paths = _collect_images([clean_dir])
    noisy_paths = _collect_images([noisy_dir])

    if not clean_paths:
        raise ValueError(f"No clean images found in: {clean_dir}")
    if not noisy_paths:
        raise ValueError(f"No noisy images found in: {noisy_dir}")

    if match_by_name:
        noisy_by_stem = {p.stem: p for p in noisy_paths}
        pairs = [
            (c, noisy_by_stem[c.stem])
            for c in clean_paths
            if c.stem in noisy_by_stem
        ]
        if not pairs:
            raise ValueError(
                f"No filename matches between {clean_dir} and {noisy_dir}. "
                "Set match_by_name=False to use sorted-position matching instead."
            )
        return sorted(pairs)
    else:
        if len(clean_paths) != len(noisy_paths):
            raise ValueError(
                f"clean ({len(clean_paths)} images) and noisy "
                f"({len(noisy_paths)} images) directories have different counts. "
                "Use match_by_name=True for explicit filename matching."
            )
        return list(zip(clean_paths, noisy_paths))


# ---------------------------------------------------------------------------
# PairedPatchDataset
# ---------------------------------------------------------------------------


class PairedPatchDataset(Dataset):
    """Random-patch dataset for training on real clean/noisy image pairs.

    Both clean and noisy images are loaded from disk; the noise is therefore
    authentic camera noise rather than a synthetic approximation.  The same
    random crop and augmentation transform is applied identically to both
    images, so spatial alignment is preserved.

    sigma_map is estimated from the actual noise residual |noisy − clean|
    using a local sliding-window standard deviation.

    Args:
        clean_dir: Directory of clean (ground-truth) images.
        noisy_dir: Directory of matching noisy images.
        patch_size: Spatial size of extracted patches (default: 128).
        patches_per_image: Virtual patches per pair per epoch (default: 64).
        augment: Random flip / rotation applied identically to both images.
        match_by_name: If True (default), match pairs by file stem.
            If False, match by sorted position within each directory.
        sigma_window: Kernel size for the local sigma_map estimate (default: 7).

    Example directory layout (match_by_name=True)::

        clean/
          scene_001.png
          scene_002.png
        noisy/
          scene_001.png   ← same stem as clean counterpart
          scene_002.png

    Example layout (match_by_name=False, sorted position)::

        clean/  frame_0001.png  frame_0002.png …
        noisy/  frame_0001.png  frame_0002.png …
    """

    def __init__(
        self,
        clean_dir: str | Path,
        noisy_dir: str | Path,
        patch_size: int = 128,
        patches_per_image: int = 64,
        augment: bool = True,
        match_by_name: bool = True,
        sigma_window: int = 7,
    ) -> None:
        self._pairs = _match_pairs(Path(clean_dir), Path(noisy_dir), match_by_name)
        self._patch_size = patch_size
        self._patches_per_image = patches_per_image
        self._augment = augment
        self._sigma_window = sigma_window

    def __len__(self) -> int:
        return len(self._pairs) * self._patches_per_image

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        pair_idx = idx // self._patches_per_image
        clean_path, noisy_path = self._pairs[pair_idx]

        clean_img = _load_image(clean_path)
        noisy_img = _load_image(noisy_path)

        # Identical crop for both images
        h, w, _ = clean_img.shape
        ps = self._patch_size
        if h < ps or w < ps:
            pad_h = max(0, ps - h)
            pad_w = max(0, ps - w)
            clean_img = np.pad(clean_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            noisy_img = np.pad(noisy_img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
            h, w, _ = clean_img.shape
        top  = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        clean_patch = clean_img[top : top + ps, left : left + ps, :]
        noisy_patch = noisy_img[top : top + ps, left : left + ps, :]

        # Identical augmentation for both images
        if self._augment:
            flip_v = random.random() < 0.5
            flip_h = random.random() < 0.5
            rot_k  = random.randint(0, 3)
            for arr in (clean_patch, noisy_patch):
                if flip_v:
                    arr = arr[::-1, :, :]
                if flip_h:
                    arr = arr[:, ::-1, :]
                if rot_k:
                    arr = np.rot90(arr, k=rot_k)
            # Re-apply in order (the loop above re-binds arr, not the original)
            if flip_v:
                clean_patch = clean_patch[::-1, :, :]
                noisy_patch = noisy_patch[::-1, :, :]
            if flip_h:
                clean_patch = clean_patch[:, ::-1, :]
                noisy_patch = noisy_patch[:, ::-1, :]
            if rot_k:
                clean_patch = np.rot90(clean_patch, k=rot_k)
                noisy_patch = np.rot90(noisy_patch, k=rot_k)

        clean_t = _hwc_to_tensor(np.ascontiguousarray(clean_patch))
        noisy_t = _hwc_to_tensor(np.ascontiguousarray(noisy_patch))
        noise_residual = noisy_t - clean_t
        sigma_map = _local_std_sigma(noise_residual, window=self._sigma_window)
        return noisy_t, clean_t, sigma_map

    @property
    def num_pairs(self) -> int:
        return len(self._pairs)


# ---------------------------------------------------------------------------
# PairedVideoSequenceDataset
# ---------------------------------------------------------------------------


class PairedVideoSequenceDataset(Dataset):
    """Temporal clip dataset for training on real paired noisy/clean video.

    Expects two parallel directory trees with matching sub-directory names:

        clean_root/
          scene_a/  frame_0001.png  frame_0002.png …
          scene_b/  frame_0001.png  …
        noisy_root/
          scene_a/  frame_0001.png  frame_0002.png …  ← same names
          scene_b/  frame_0001.png  …

    Clips of *num_frames* consecutive frames are extracted from each matched
    scene pair.  The same crop position and augmentation are applied to all
    frames in both clean and noisy clips so spatial alignment is preserved.

    sigma_map is estimated per-frame from the actual noise residual.

    Args:
        clean_sequence_dirs: One or more clean root directories.
        noisy_sequence_dirs: Matching noisy root directories.  Must have the
            same number of entries as *clean_sequence_dirs*.
        num_frames: Temporal window length (default: 5).
        patch_size: Spatial patch size (default: 64).
        patches_per_clip: Virtual patches per clip per epoch (default: 16).
        augment: Spatially consistent flip / rotation across all frames.
        sigma_window: Kernel size for sigma_map estimation (default: 7).
    """

    def __init__(
        self,
        clean_sequence_dirs: Sequence[str | Path],
        noisy_sequence_dirs: Sequence[str | Path],
        num_frames: int = 5,
        patch_size: int = 64,
        patches_per_clip: int = 16,
        augment: bool = True,
        sigma_window: int = 7,
    ) -> None:
        if len(clean_sequence_dirs) != len(noisy_sequence_dirs):
            raise ValueError(
                "clean_sequence_dirs and noisy_sequence_dirs must have the same length"
            )
        self._clips = self._collect_paired_clips(
            clean_sequence_dirs, noisy_sequence_dirs, num_frames
        )
        if not self._clips:
            raise ValueError(
                f"No paired frame sequences found in: {list(clean_sequence_dirs)}"
            )
        self._num_frames  = num_frames
        self._patch_size  = patch_size
        self._patches_per_clip = patches_per_clip
        self._augment     = augment
        self._sigma_window = sigma_window

    @staticmethod
    def _collect_paired_clips(
        clean_roots: Sequence[str | Path],
        noisy_roots: Sequence[str | Path],
        num_frames: int,
    ) -> list[tuple[list[Path], list[Path]]]:
        """Collect all (clean_clip, noisy_clip) windows from matching sub-dirs."""
        clips: list[tuple[list[Path], list[Path]]] = []
        for clean_root, noisy_root in zip(clean_roots, noisy_roots):
            clean_root = Path(clean_root)
            noisy_root = Path(noisy_root)
            if not clean_root.exists() or not noisy_root.exists():
                continue
            # Match sub-directories by name
            clean_subdirs = {d.name: d for d in sorted(clean_root.iterdir()) if d.is_dir()}
            for sub_name, clean_sub in clean_subdirs.items():
                noisy_sub = noisy_root / sub_name
                if not noisy_sub.is_dir():
                    continue
                clean_frames = sorted(
                    p for p in clean_sub.iterdir()
                    if p.suffix.lower() in _IMAGE_EXTENSIONS
                )
                noisy_frames = sorted(
                    p for p in noisy_sub.iterdir()
                    if p.suffix.lower() in _IMAGE_EXTENSIONS
                )
                n = min(len(clean_frames), len(noisy_frames))
                for i in range(n - num_frames + 1):
                    clips.append((
                        clean_frames[i : i + num_frames],
                        noisy_frames[i : i + num_frames],
                    ))
        return clips

    def __len__(self) -> int:
        return len(self._clips) * self._patches_per_clip

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return (noisy_clip, clean_clip, sigma_map) each shaped (T, C, H, W)."""
        clip_idx = idx // self._patches_per_clip
        clean_paths, noisy_paths = self._clips[clip_idx]

        # Load first frame to determine crop position
        first_clean = _load_image(clean_paths[0])
        h, w, _ = first_clean.shape
        ps = self._patch_size
        top  = random.randint(0, max(0, h - ps))
        left = random.randint(0, max(0, w - ps))

        # Consistent augmentation across all frames
        flip_v = self._augment and random.random() < 0.5
        flip_h = self._augment and random.random() < 0.5
        rot_k  = random.randint(0, 3) if self._augment else 0

        noisy_frames, clean_frames, sigma_frames = [], [], []
        for clean_p, noisy_p in zip(clean_paths, noisy_paths):
            clean_img = _load_image(clean_p)
            noisy_img = _load_image(noisy_p)

            for arr_list in ([clean_img], [noisy_img]):
                arr = arr_list[0]
                if h < ps or w < ps:
                    pad_h = max(0, ps - h)
                    pad_w = max(0, ps - w)
                    arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
                arr_list[0] = arr[top : top + ps, left : left + ps, :]

            clean_patch, noisy_patch = clean_img, noisy_img
            # Augment: must apply identically to both
            clean_patch = clean_img[top : top + ps, left : left + ps, :]
            noisy_patch = noisy_img[top : top + ps, left : left + ps, :]
            if flip_v:
                clean_patch = clean_patch[::-1, :, :]
                noisy_patch = noisy_patch[::-1, :, :]
            if flip_h:
                clean_patch = clean_patch[:, ::-1, :]
                noisy_patch = noisy_patch[:, ::-1, :]
            if rot_k:
                clean_patch = np.rot90(clean_patch, k=rot_k)
                noisy_patch = np.rot90(noisy_patch, k=rot_k)

            clean_t = _hwc_to_tensor(np.ascontiguousarray(clean_patch))
            noisy_t = _hwc_to_tensor(np.ascontiguousarray(noisy_patch))
            sigma_t = _local_std_sigma(noisy_t - clean_t, window=self._sigma_window)

            clean_frames.append(clean_t)
            noisy_frames.append(noisy_t)
            sigma_frames.append(sigma_t)

        return (
            torch.stack(noisy_frames),   # (T, C, H, W)
            torch.stack(clean_frames),
            torch.stack(sigma_frames),
        )

    @property
    def num_clips(self) -> int:
        return len(self._clips)


# ---------------------------------------------------------------------------
# CombinedDataset
# ---------------------------------------------------------------------------


class CombinedDataset(Dataset):
    """Mixes multiple datasets with explicit sampling weights.

    Each call to ``__getitem__`` randomly selects one of the wrapped datasets
    according to *weights*, then draws a random sample from it.  This lets you
    blend synthetic-noise data (PatchDataset) with real-pair data
    (PairedPatchDataset) in configurable proportions within a single DataLoader.

    All wrapped datasets must return tuples of the same structure and tensor
    shapes so that DataLoader collation works correctly.

    Args:
        datasets: List of datasets to mix.  Must all return the same tuple
            structure.
        weights: Relative sampling probabilities.  Does not need to sum to 1;
            values are normalised internally.  If None, all datasets are
            sampled uniformly.
        num_samples: Total length of the combined dataset (i.e. how many
            samples constitute one "epoch").  Defaults to the sum of individual
            dataset lengths.

    Example::

        synthetic = PatchDataset(clean_dirs, noise_generator=MixedNoiseGenerator.default())
        paired    = PairedPatchDataset("clean/", "noisy/")
        combined  = CombinedDataset(
            datasets=[synthetic, paired],
            weights=[0.6, 0.4],         # 60% synthetic, 40% real pairs
        )
        loader = DataLoader(combined, batch_size=16, shuffle=True, num_workers=4)
    """

    def __init__(
        self,
        datasets: list[Dataset],
        weights: Optional[list[float]] = None,
        num_samples: Optional[int] = None,
    ) -> None:
        if not datasets:
            raise ValueError("datasets must not be empty")
        if weights is not None:
            if len(weights) != len(datasets):
                raise ValueError("len(weights) must equal len(datasets)")
            total = sum(weights)
            if total <= 0:
                raise ValueError("weights must sum to a positive number")
            self._weights = [w / total for w in weights]
        else:
            n = len(datasets)
            self._weights = [1.0 / n] * n

        self._datasets = datasets
        self._num_samples = num_samples if num_samples is not None else sum(
            len(d) for d in datasets  # type: ignore[arg-type]
        )

    def __len__(self) -> int:
        return self._num_samples

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        # idx is used only as a secondary random seed to ensure determinism
        # when num_workers > 0; the primary selection is by weight.
        rng = random.Random(idx)
        (dataset,) = rng.choices(self._datasets, weights=self._weights, k=1)
        item_idx = rng.randrange(len(dataset))  # type: ignore[arg-type]
        return dataset[item_idx]

    @property
    def num_datasets(self) -> int:
        return len(self._datasets)
