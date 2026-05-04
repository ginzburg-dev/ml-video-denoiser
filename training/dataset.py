"""PyTorch datasets for training the denoiser.

Four datasets are provided:

  PatchDataset
      Loads clean images and synthesises noise on-the-fly via a NoiseGenerator.
      No paired data required.  Suitable for training NAFNet residual models.

  VideoSequenceDataset
      Loads temporal clips and synthesises noise consistently across frames.
      Suitable for training NAFNet temporal models.

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

Temporal datasets return :class:`TemporalSample` named tuples.
Spatial datasets return plain ``(noisy, clean, sigma_map)`` tuples.
"""

import random
from pathlib import Path
from typing import NamedTuple, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset

from noise_generators import GaussianNoiseGenerator, MixedNoiseGenerator, NoiseGenerator


# ---------------------------------------------------------------------------
# Temporal batch type
# ---------------------------------------------------------------------------


class TemporalSample(NamedTuple):
    """Batch item returned by temporal datasets.

    ``denoised`` is populated only when a spatial cache is active
    (cascade training with ``--freeze-spatial``).  The training loop
    checks ``batch.denoised is not None`` to decide whether to pass
    pre-computed spatial outputs to the model.
    """

    noisy: Tensor              # (T, C, H, W)
    clean: Tensor              # (T, C, H, W)
    sigma: Tensor              # (T, C, H, W)
    denoised: Optional[Tensor] = None  # (T, 3, H, W) — pre-denoised, or None


def collate_temporal(batch: list[TemporalSample]) -> TemporalSample:
    """DataLoader collate function for :class:`TemporalSample` items."""
    return TemporalSample(
        noisy=torch.stack([s.noisy for s in batch]),
        clean=torch.stack([s.clean for s in batch]),
        sigma=torch.stack([s.sigma for s in batch]),
        denoised=(
            torch.stack([s.denoised for s in batch])  # type: ignore[arg-type]
            if batch[0].denoised is not None
            else None
        ),
    )


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


def _spread_indices(n: int, count: int) -> list[int]:
    """Return *count* evenly spread indices into a sequence of length *n*.

    Uses integer linspace so first and last indices are always included.
    If count >= n, returns all indices.

    Examples::

        _spread_indices(90, 3) → [0, 44, 89]
        _spread_indices(90, 5) → [0, 22, 44, 67, 89]
        _spread_indices(3,  5) → [0, 1, 2]
        _spread_indices(90, 1) → [0]
    """
    if count <= 0:
        raise ValueError("count must be a positive integer")
    if count >= n:
        return list(range(n))
    if count == 1:
        return [0]
    return [int(round(i * (n - 1) / (count - 1))) for i in range(count)]


def _collect_images_spread(
    dirs: Sequence[str | Path],
    frames_per_sequence: Optional[int],
    random_frames: bool = False,
) -> tuple[list[Path], bool]:
    """Collect image files from *dirs*, optionally spread per sequence subdir.

    If *frames_per_sequence* is set:
    - Each subdirectory is treated as one sequence; N evenly spread frames
      are selected from it (or N random frames if *random_frames* is True).
    - Flat directories (no image-containing subdirs) fall back to all images
      with a warning.

    Returns:
        (paths, flat_fallback) where flat_fallback is True if a flat directory
        was encountered and the limit was ignored.
    """
    if frames_per_sequence is None:
        return _collect_images(dirs), False

    paths: list[Path] = []
    flat_fallback = False

    for d in dirs:
        root = Path(d)
        if not root.exists():
            raise FileNotFoundError(f"Image directory not found: {root}")

        # Collect subdirs that contain images
        seq_dirs = sorted(sub for sub in root.iterdir() if sub.is_dir())
        seq_frames: list[list[Path]] = []
        for sub in seq_dirs:
            frames = sorted(
                p for p in sub.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
            )
            if frames:
                seq_frames.append(frames)

        if seq_frames:
            # Sequence-folder structure: spread or randomly sample N frames per sequence
            for frames in seq_frames:
                if random_frames:
                    k = min(frames_per_sequence, len(frames))
                    chosen = random.sample(frames, k)
                    paths.extend(sorted(chosen))
                else:
                    indices = _spread_indices(len(frames), frames_per_sequence)
                    paths.extend(frames[i] for i in indices)
        else:
            # Flat directory — warn and use all images
            flat_imgs = sorted(
                p for p in root.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
            )
            if flat_imgs:
                import warnings
                warnings.warn(
                    f"--frames-per-sequence has no effect on flat directory {root} "
                    "(no sequence subdirectories found). Using all {len(flat_imgs)} images.",
                    UserWarning,
                    stacklevel=4,
                )
                paths.extend(flat_imgs)
                flat_fallback = True
            else:
                raise ValueError(f"No images found in: {root}")

    return sorted(set(paths)), flat_fallback


def _match_keyed_images(
    clean_paths: Sequence[Path],
    noisy_paths: Sequence[Path],
    *,
    clean_root: Path,
    noisy_root: Path,
) -> list[tuple[Path, Path]]:
    """Match image paths by relative subpath + stem, ignoring extension."""

    def _build_keyed_map(paths: Sequence[Path], root: Path) -> dict[str, Path]:
        keyed: dict[str, Path] = {}
        for path in paths:
            key = str(path.relative_to(root).with_suffix(""))
            if key in keyed:
                raise ValueError(
                    "Duplicate image stems detected while matching paired frames by name."
                )
            keyed[key] = path
        return keyed

    clean_by_key = _build_keyed_map(clean_paths, clean_root)
    noisy_by_key = _build_keyed_map(noisy_paths, noisy_root)

    clean_keys = set(clean_by_key)
    noisy_keys = set(noisy_by_key)
    missing_noisy = sorted(clean_keys - noisy_keys)
    if missing_noisy:
        raise ValueError(
            "Clean frames have no matching noisy counterpart: "
            + ", ".join(missing_noisy[:5])
            + (" ..." if len(missing_noisy) > 5 else "")
        )

    # Extra noisy frames (noisy_keys - clean_keys) are silently dropped —
    # this is expected when clean_paths were spread-sampled.
    return [(clean_by_key[key], noisy_by_key[key]) for key in sorted(clean_keys)]


def _load_image(path: Path) -> np.ndarray:
    """Load *path* as a float32 HWC array.

    EXR: values passed through as-is (float32, any range).
    LDR (PNG/JPG/TIFF): PIL converts to RGB, integer values divided by
    255 or 65535 depending on bit depth.
    """
    if path.suffix.lower() == ".exr":
        img = _load_exr_image(path)
        img = img.astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        elif img.shape[2] == 4:
            return img  # keep RGBA — alpha used for noise masking in __getitem__
        elif img.shape[2] > 4:
            img = img[:, :, :4]
        return img

    with Image.open(path) as pil_img:
        # Always convert to RGB so downstream code always sees (H, W, 3)
        # uint8, regardless of source mode (palette, CMYK, RGBA, L, …).
        img = np.array(pil_img.convert("RGB"))

    source_dtype = img.dtype
    img = img.astype(np.float32)
    if img.max() > 1.5:
        dtype_max = 255.0 if img.max() <= 255.5 else 65535.0
        img /= dtype_max
    return img


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


def _apply_alpha_mask(noisy: Tensor, clean: Tensor, patch: np.ndarray) -> Tensor:
    """Premultiply noise contribution by alpha so transparent pixels stay clean.

    When *patch* is RGBA (H, W, 4), the noise added by the noise generator is
    scaled by alpha so alpha=0 areas receive no noise.  Returns the masked
    noisy tensor; clean and sigma are unchanged.
    """
    if patch.shape[2] != 4:
        return noisy
    alpha_np = np.ascontiguousarray(patch[:, :, 3:4].transpose(2, 0, 1))
    alpha = torch.from_numpy(alpha_np)  # (1, H, W)
    return clean + (noisy - clean) * alpha


def _rgb(patch: np.ndarray) -> np.ndarray:
    """Return the RGB channels of a patch, dropping alpha if present."""
    return patch[:, :, :3] if patch.shape[2] == 4 else patch


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


def _center_crop(
    img: np.ndarray, patch_size: int
) -> np.ndarray:
    """Return a centered (patch_size, patch_size, C) crop from *img*."""
    h, w, _ = img.shape
    if h < patch_size or w < patch_size:
        pad_h = max(0, patch_size - h)
        pad_w = max(0, patch_size - w)
        img = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
        h, w, _ = img.shape
    top = (h - patch_size) // 2
    left = (w - patch_size) // 2
    return img[top : top + patch_size, left : left + patch_size, :]


def _crop_image(
    img: np.ndarray,
    patch_size: int,
    crop_mode: str,
) -> np.ndarray:
    """Crop *img* according to the requested mode."""
    if crop_mode == "random":
        return _random_crop(img, patch_size)
    if crop_mode == "center":
        return _center_crop(img, patch_size)
    if crop_mode == "full":
        return img
    if crop_mode == "grid":
        return _center_crop(img, patch_size)
    raise ValueError(f"Unsupported crop_mode: {crop_mode}")


def _grid_start(length: int, patch_size: int, grid_size: int, coord_index: int) -> int:
    """Return a deterministic crop start for one grid coordinate."""
    if grid_size <= 1 or length <= patch_size:
        return 0
    max_start = length - patch_size
    starts = np.linspace(0, max_start, num=grid_size, dtype=int)
    return int(starts[coord_index])


def _pad_frame_to_shape(frame: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Pad a frame on the bottom/right so it reaches the requested shape."""
    h, w, _ = frame.shape
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")


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

    Supports sequence folder structures (subdirectories of frames) via
    *frames_per_sequence*: N evenly spread frames are selected from each
    subdirectory instead of using every frame.  This keeps epoch size
    manageable for long sequences.  Flat directories (no subdirs) fall back
    to all images with a warning.

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
        crop_mode: Spatial crop mode. One of ``"random"``, ``"center"``,
            ``"full"``, or ``"grid"``.
        crop_grid_size: Grid side length used when ``crop_mode="grid"``.
        frames_per_sequence: If set, select this many evenly spread frames
            from each sequence subdirectory.  First and last frames are
            always included.  Ignored (with a warning) for flat directories.
    """

    def __init__(
        self,
        image_dirs: Sequence[str | Path],
        noise_generator: Optional[NoiseGenerator] = None,
        patch_size: int = 128,
        patches_per_image: int = 64,
        augment: bool = True,
        crop_mode: str = "random",
        crop_grid_size: int = 2,
        frames_per_sequence: Optional[int] = None,
        preload: bool = False,
    ) -> None:
        self._paths, _ = _collect_images_spread(image_dirs, frames_per_sequence)
        if not self._paths:
            raise ValueError(f"No images found in: {list(image_dirs)}")
        self._noise_gen = noise_generator or MixedNoiseGenerator.default()
        self._patch_size = patch_size
        self._patches_per_image = patches_per_image
        self._augment = augment
        self._crop_mode = crop_mode
        self._crop_grid_size = crop_grid_size
        self._image_cache: Optional[dict[Path, np.ndarray]] = None
        if preload:
            self._image_cache = {}
            print(f"Preloading {len(self._paths)} images into RAM...", flush=True)
            for i, path in enumerate(self._paths, 1):
                self._image_cache[path] = _load_image(path)
                if i % 20 == 0 or i == len(self._paths):
                    print(f"  {i}/{len(self._paths)} loaded", flush=True)
            print("Preload complete.", flush=True)

    def __len__(self) -> int:
        return len(self._paths) * self._patches_per_image

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        img_idx = idx // self._patches_per_image
        patch_idx = idx % self._patches_per_image
        path = self._paths[img_idx]
        img = self._image_cache[path] if self._image_cache is not None else _load_image(path)
        if self._crop_mode == "grid":
            img = _pad_frame_to_shape(img, max(img.shape[0], self._patch_size), max(img.shape[1], self._patch_size))
            h, w, _ = img.shape
            row = patch_idx // self._crop_grid_size
            col = patch_idx % self._crop_grid_size
            top = _grid_start(h, self._patch_size, self._crop_grid_size, row)
            left = _grid_start(w, self._patch_size, self._crop_grid_size, col)
            patch = img[top : top + self._patch_size, left : left + self._patch_size, :]
        else:
            patch = _crop_image(img, self._patch_size, self._crop_mode)
        if self._augment:
            patch = _augment(patch)
        clean = _hwc_to_tensor(_rgb(patch))
        noisy, clean, sigma_map = self._noise_gen(clean)
        noisy = _apply_alpha_mask(noisy, clean, patch)
        return noisy, clean, sigma_map

    @property
    def num_images(self) -> int:
        return len(self._paths)


# ---------------------------------------------------------------------------
# VideoSequenceDataset
# ---------------------------------------------------------------------------


class VideoSequenceDataset(Dataset):
    """Temporal clip dataset for video denoising (NAFNet temporal).

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
        num_frames: Temporal clip length (default: 3).
        patch_size: Spatial patch size (default: 64 — smaller than spatial
            because memory cost is num_frames ×).
        patches_per_image: Virtual patches per clip per epoch (default: 64).
        random_windows: If True, sample random temporal windows from each
            sequence each epoch instead of enumerating every sliding window.
        windows_per_sequence: Number of temporal windows to draw per sequence
            per epoch when ``random_windows`` is enabled.
        augment: Random spatial flip / rotation (applied identically across
            all frames in the clip).
        crop_mode: Spatial crop mode. One of ``"random"``, ``"center"``,
            ``"full"``, or ``"grid"``.
        crop_grid_size: Grid side length used when ``crop_mode="grid"``.
    """

    def __init__(
        self,
        sequence_dirs: Sequence[str | Path],
        noise_generator: Optional[NoiseGenerator] = None,
        num_frames: int = 3,
        patch_size: int = 64,
        patches_per_image: int = 64,
        random_windows: bool = False,
        windows_per_sequence: Optional[int] = None,
        augment: bool = True,
        crop_mode: str = "random",
        crop_grid_size: int = 2,
        preload: bool = False,
    ) -> None:
        self._sequences = self._collect_sequences(sequence_dirs, num_frames)
        if not self._sequences:
            raise ValueError(f"No frame sequences found in: {list(sequence_dirs)}")
        self._noise_gen = noise_generator or GaussianNoiseGenerator(0.0, 50.0 / 255.0)
        self._num_frames = num_frames
        self._patch_size = patch_size
        self._patches_per_image = patches_per_image
        self._random_windows = random_windows
        self._windows_per_sequence = windows_per_sequence or 1
        self._augment = augment
        self._crop_mode = crop_mode
        self._crop_grid_size = crop_grid_size
        self._num_sequences = len(self._sequences)
        self._image_cache: Optional[dict[Path, np.ndarray]] = None
        if preload:
            all_paths: set[Path] = {p for seq in self._sequences for p in seq}
            self._image_cache = {}
            print(f"Preloading {len(all_paths)} frames into RAM...", flush=True)
            for i, path in enumerate(sorted(all_paths), 1):
                self._image_cache[path] = _load_image(path)
                if i % 20 == 0 or i == len(all_paths):
                    print(f"  {i}/{len(all_paths)} loaded", flush=True)
            print("Preload complete.", flush=True)
        if self._random_windows and self._windows_per_sequence <= 0:
            raise ValueError("windows_per_sequence must be positive when random_windows is enabled")
        self._clips = (
            []
            if self._random_windows
            else self._enumerate_clips(
                self._sequences,
                num_frames,
                windows_per_sequence=windows_per_sequence,
            )
        )

    @staticmethod
    def _collect_sequences(
        sequence_dirs: Sequence[str | Path], num_frames: int
    ) -> list[list[Path]]:
        """Collect all frame sequences that are long enough for sampling."""
        sequences: list[list[Path]] = []
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
                if len(frames) >= num_frames:
                    sequences.append(frames)
        return sequences

    @staticmethod
    def _enumerate_clips(
        sequences: Sequence[Sequence[Path]],
        num_frames: int,
        windows_per_sequence: Optional[int] = None,
    ) -> list[list[Path]]:
        """Expand every valid sliding window, optionally keeping a fixed subset."""
        clips: list[list[Path]] = []
        for frames in sequences:
            n_windows = len(frames) - num_frames + 1
            if windows_per_sequence is None or windows_per_sequence >= n_windows:
                starts = range(n_windows)
            else:
                starts = np.linspace(0, n_windows - 1, num=windows_per_sequence, dtype=int)
            for i in starts:
                i = int(i)
                clips.append(list(frames[i : i + num_frames]))
        return clips

    def _sample_frame_paths(self, idx: int) -> list[Path]:
        """Resolve dataset index to a concrete temporal window."""
        if not self._random_windows:
            clip_idx = idx // self._patches_per_image
            return self._clips[clip_idx]

        windows_per_sequence = self._windows_per_sequence * self._patches_per_image
        sequence_idx = idx // windows_per_sequence
        frames = self._sequences[sequence_idx]
        start = random.randint(0, len(frames) - self._num_frames)
        return list(frames[start : start + self._num_frames])

    def __len__(self) -> int:
        if self._random_windows:
            return self._num_sequences * self._windows_per_sequence * self._patches_per_image
        return len(self._clips) * self._patches_per_image

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        """Return (noisy_clip, clean_clip, sigma_map).

        Shapes:
            noisy_clip:  (T, C, H, W)
            clean_clip:  (T, C, H, W)
            sigma_map:   (T, C, H, W)
        """
        frame_paths = self._sample_frame_paths(idx)

        # Load all frames
        if self._image_cache is not None:
            frames = [self._image_cache[p] for p in frame_paths]
        else:
            frames = [_load_image(p) for p in frame_paths]
        ps = self._patch_size
        target_h = max(ps, max(frame.shape[0] for frame in frames))
        target_w = max(ps, max(frame.shape[1] for frame in frames))
        frames = [_pad_frame_to_shape(frame, target_h, target_w) for frame in frames]

        # Determine crop position once (same for all frames)
        if self._crop_mode == "random":
            top = random.randint(0, target_h - ps)
            left = random.randint(0, target_w - ps)
        elif self._crop_mode == "center":
            top = (target_h - ps) // 2
            left = (target_w - ps) // 2
        elif self._crop_mode == "full":
            top = 0
            left = 0
            ps = None
        elif self._crop_mode == "grid":
            grid_patch_idx = idx % self._patches_per_image
            row = grid_patch_idx // self._crop_grid_size
            col = grid_patch_idx % self._crop_grid_size
            top = _grid_start(target_h, ps, self._crop_grid_size, row)
            left = _grid_start(target_w, ps, self._crop_grid_size, col)
        else:
            raise ValueError(f"Unsupported crop_mode: {self._crop_mode}")
        # Determine augmentation seed once
        flip_v = self._augment and random.random() < 0.5
        flip_h = self._augment and random.random() < 0.5
        rot_k = random.randint(0, 3) if self._augment else 0

        # Lock noise source for the whole clip so all frames share the same
        # pool / ISO — matches real-world behaviour where a clip is one shot.
        noise_gen = self._noise_gen.for_clip() if hasattr(self._noise_gen, "for_clip") else self._noise_gen

        noisy_frames, clean_frames, sigma_frames = [], [], []
        for frame in frames:
            patch = frame if ps is None else frame[top : top + ps, left : left + ps, :]
            if flip_v:
                patch = patch[::-1, :, :]
            if flip_h:
                patch = patch[:, ::-1, :]
            if rot_k:
                patch = np.rot90(patch, k=rot_k)
            patch = np.ascontiguousarray(patch)
            clean_t = _hwc_to_tensor(_rgb(patch))
            noisy_t, clean_t, sigma_t = noise_gen(clean_t)
            noisy_t = _apply_alpha_mask(noisy_t, clean_t, patch)
            noisy_frames.append(noisy_t)
            clean_frames.append(clean_t)
            sigma_frames.append(sigma_t)

        return TemporalSample(
            noisy=torch.stack(noisy_frames),
            clean=torch.stack(clean_frames),
            sigma=torch.stack(sigma_frames),
        )

    @property
    def num_clips(self) -> int:
        if self._random_windows:
            return self._num_sequences * self._windows_per_sequence
        return len(self._clips)

    @property
    def num_sequences(self) -> int:
        return self._num_sequences


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
    clean_dir: Path,
    noisy_dir: Path,
    match_by_name: bool,
    frames_per_sequence: Optional[int] = None,
    random_frames: bool = False,
) -> list[tuple[Path, Path]]:
    """Collect matched (clean, noisy) image path pairs from two directories.

    When *frames_per_sequence* is set, N evenly spread frames are selected
    from each sequence subdirectory in *clean_dir*; the matching noisy paths
    are derived from those same stems/positions.  When *random_frames* is True,
    frames are sampled randomly instead of evenly spread — call again each epoch
    to get a fresh random draw.

    Args:
        clean_dir: Root directory of clean images (searched recursively).
        noisy_dir: Root directory of noisy images (searched recursively).
        match_by_name: If True, match files whose stems are identical.
            If False, match by sorted position — clean[i] ↔ noisy[i].
        frames_per_sequence: If set, select N frames per sequence subdirectory.
        random_frames: If True, pick frames randomly each call instead of
            evenly spreading them.

    Returns:
        Sorted list of (clean_path, noisy_path) pairs.

    Raises:
        ValueError: If no pairs are found or counts differ (position matching).
    """
    clean_paths, _ = _collect_images_spread([clean_dir], frames_per_sequence, random_frames)
    noisy_paths = _collect_images([noisy_dir])

    if not clean_paths:
        raise ValueError(f"No clean images found in: {clean_dir}")
    if not noisy_paths:
        raise ValueError(f"No noisy images found in: {noisy_dir}")

    if match_by_name:
        return _match_keyed_images(
            clean_paths,
            noisy_paths,
            clean_root=Path(clean_dir),
            noisy_root=Path(noisy_dir),
        )
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

    Supports sequence folder structures via *frames_per_sequence*: N evenly
    spread frames are selected from each subdirectory in *clean_dir*; matching
    noisy frames are resolved by the same stem/position logic.

    Args:
        clean_dir: Directory of clean (ground-truth) images.
        noisy_dir: Directory of matching noisy images.
        patch_size: Spatial size of extracted patches (default: 128).
        patches_per_image: Virtual patches per pair per epoch (default: 64).
        augment: Random flip / rotation applied identically to both images.
        match_by_name: If True (default), match pairs by file stem.
            If False, match by sorted position within each directory.
        sigma_window: Kernel size for the local sigma_map estimate (default: 7).
        crop_mode: Spatial crop mode. One of ``"random"``, ``"center"``,
            ``"full"``, or ``"grid"``.
        crop_grid_size: Grid side length used when ``crop_mode="grid"``.
        frames_per_sequence: If set, select N evenly spread frames per
            sequence subdirectory.  Flat directories fall back to all images
            with a warning.

    Example directory layout (match_by_name=True)::

        clean/
          scene_001.png
          scene_002.png
        noisy/
          scene_001.png   ← same stem as clean counterpart
          scene_002.png

    Example layout (sequence folders + frames_per_sequence=3)::

        clean/
          scene_001/  frame_0001.png … frame_0090.png  → picks 3 spread frames
          scene_002/  frame_0001.png … frame_0060.png  → picks 3 spread frames
        noisy/
          scene_001/  frame_0001.png …
          scene_002/  frame_0001.png …
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
        crop_mode: str = "random",
        crop_grid_size: int = 2,
        frames_per_sequence: Optional[int] = None,
        random_frames: bool = False,
        preload: bool = False,
    ) -> None:
        self._clean_dir = Path(clean_dir)
        self._noisy_dir = Path(noisy_dir)
        self._match_by_name = match_by_name
        self._frames_per_sequence = frames_per_sequence
        self._random_frames = random_frames
        self._pairs = _match_pairs(
            self._clean_dir, self._noisy_dir, match_by_name, frames_per_sequence, random_frames
        )
        self._patch_size = patch_size
        self._patches_per_image = patches_per_image
        self._augment = augment
        self._sigma_window = sigma_window
        self._crop_mode = crop_mode
        self._crop_grid_size = crop_grid_size
        self._cache: dict[Path, np.ndarray] | None = None
        if preload:
            self._cache = {}
            all_paths = {p for pair in self._pairs for p in pair}
            print(f"Preloading {len(all_paths)} images into RAM...", flush=True)
            for i, path in enumerate(sorted(all_paths), 1):
                self._cache[path] = _load_image(path)
                if i % 10 == 0 or i == len(all_paths):
                    print(f"  {i}/{len(all_paths)} loaded", flush=True)
            print("Preload complete.", flush=True)

    def resample_frames(self) -> None:
        """Resample frames randomly — call at the start of each epoch when
        *random_frames* is True to get a fresh draw of frames per sequence."""
        if self._random_frames:
            self._pairs = _match_pairs(
                self._clean_dir, self._noisy_dir,
                self._match_by_name, self._frames_per_sequence, random_frames=True,
            )

    def __len__(self) -> int:
        return len(self._pairs) * self._patches_per_image

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        pair_idx = idx // self._patches_per_image
        patch_idx = idx % self._patches_per_image
        clean_path, noisy_path = self._pairs[pair_idx]

        if self._cache is not None:
            clean_img = self._cache[clean_path]
            noisy_img = self._cache[noisy_path]
        else:
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
        if self._crop_mode == "full":
            target_h = max(clean_img.shape[0], noisy_img.shape[0])
            target_w = max(clean_img.shape[1], noisy_img.shape[1])
            clean_patch = _pad_frame_to_shape(clean_img, target_h, target_w)
            noisy_patch = _pad_frame_to_shape(noisy_img, target_h, target_w)
        else:
            target_h = max(clean_img.shape[0], noisy_img.shape[0], ps)
            target_w = max(clean_img.shape[1], noisy_img.shape[1], ps)
            clean_img = _pad_frame_to_shape(clean_img, target_h, target_w)
            noisy_img = _pad_frame_to_shape(noisy_img, target_h, target_w)
            if self._crop_mode == "random":
                top = random.randint(0, target_h - ps)
                left = random.randint(0, target_w - ps)
            elif self._crop_mode == "center":
                top = (target_h - ps) // 2
                left = (target_w - ps) // 2
            elif self._crop_mode == "grid":
                row = patch_idx // self._crop_grid_size
                col = patch_idx % self._crop_grid_size
                top = _grid_start(target_h, ps, self._crop_grid_size, row)
                left = _grid_start(target_w, ps, self._crop_grid_size, col)
            else:
                raise ValueError(f"Unsupported crop_mode: {self._crop_mode}")
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

        clean_t = _hwc_to_tensor(np.ascontiguousarray(_rgb(clean_patch)))
        noisy_t = _hwc_to_tensor(np.ascontiguousarray(_rgb(noisy_patch)))
        if clean_patch.shape[2] == 4:
            noisy_t = _apply_alpha_mask(noisy_t, clean_t, clean_patch)
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
        num_frames: Temporal window length (default: 3).
        patch_size: Spatial patch size (default: 64).
        patches_per_image: Virtual patches per clip per epoch (default: 64).
        random_windows: If True, sample random temporal windows from each
            paired sequence each epoch instead of enumerating all windows.
        windows_per_sequence: Number of temporal windows to draw per paired
            sequence per epoch when ``random_windows`` is enabled.
        augment: Spatially consistent flip / rotation across all frames.
        sigma_window: Kernel size for sigma_map estimation (default: 7).
        crop_mode: Spatial crop mode. One of ``"random"``, ``"center"``,
            ``"full"``, or ``"grid"``.
        crop_grid_size: Grid side length used when ``crop_mode="grid"``.
    """

    def __init__(
        self,
        clean_sequence_dirs: Sequence[str | Path],
        noisy_sequence_dirs: Sequence[str | Path],
        num_frames: int = 3,
        patch_size: int = 64,
        patches_per_image: int = 64,
        random_windows: bool = False,
        windows_per_sequence: Optional[int] = None,
        augment: bool = True,
        sigma_window: int = 7,
        crop_mode: str = "random",
        crop_grid_size: int = 2,
        preload: bool = False,
    ) -> None:
        if len(clean_sequence_dirs) != len(noisy_sequence_dirs):
            raise ValueError(
                "clean_sequence_dirs and noisy_sequence_dirs must have the same length"
            )
        self._sequences = self._collect_paired_sequences(
            clean_sequence_dirs, noisy_sequence_dirs, num_frames
        )
        if not self._sequences:
            raise ValueError(
                f"No paired frame sequences found in: {list(clean_sequence_dirs)}"
            )
        self._num_frames  = num_frames
        self._patch_size  = patch_size
        self._patches_per_image = patches_per_image
        self._random_windows = random_windows
        self._windows_per_sequence = windows_per_sequence or 1
        self._augment     = augment
        self._sigma_window = sigma_window
        self._crop_mode = crop_mode
        self._crop_grid_size = crop_grid_size
        self._num_sequences = len(self._sequences)
        self._spatial_cache: Optional[dict[str, "torch.Tensor"]] = None
        self._image_cache: Optional[dict[Path, np.ndarray]] = None
        if preload:
            all_paths: set[Path] = set()
            for clean_frames, noisy_frames in self._sequences:
                all_paths.update(clean_frames)
                all_paths.update(noisy_frames)
            self._image_cache = {}
            print(f"Preloading {len(all_paths)} frames into RAM...", flush=True)
            for i, path in enumerate(sorted(all_paths), 1):
                self._image_cache[path] = _load_image(path)
                if i % 20 == 0 or i == len(all_paths):
                    print(f"  {i}/{len(all_paths)} loaded", flush=True)
            print("Preload complete.", flush=True)
        if self._random_windows and self._windows_per_sequence <= 0:
            raise ValueError("windows_per_sequence must be positive when random_windows is enabled")
        self._clips = (
            []
            if self._random_windows
            else self._enumerate_paired_clips(
                self._sequences,
                num_frames,
                windows_per_sequence=windows_per_sequence,
            )
        )

    def set_spatial_cache(self, cache: "dict[str, torch.Tensor]") -> None:
        """Set pre-computed spatial stage outputs for all frames.

        When set, ``__getitem__`` returns a 4-tuple
        ``(noisy_clip, denoised_clip, clean_clip, sigma_map)`` instead of the
        usual 3-tuple.  The denoised_clip tensors are cropped and augmented
        identically to noisy_clip so the correspondence is preserved.

        Args:
            cache: Mapping from absolute noisy frame path (str) to a
                ``(3, H, W)`` float32 CPU tensor holding the frozen spatial
                stage output.  Tensors should be in shared memory
                (``tensor.share_memory_()``) so DataLoader workers can access
                them without copying.
        """
        self._spatial_cache = cache

    @staticmethod
    def _collect_paired_sequences(
        clean_roots: Sequence[str | Path],
        noisy_roots: Sequence[str | Path],
        num_frames: int,
    ) -> list[tuple[list[Path], list[Path]]]:
        """Collect all paired sequences that are long enough for sampling."""
        sequences: list[tuple[list[Path], list[Path]]] = []
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
                matched_frames = _match_keyed_images(
                    clean_frames,
                    noisy_frames,
                    clean_root=clean_sub,
                    noisy_root=noisy_sub,
                )
                if len(matched_frames) >= num_frames:
                    matched_clean = [clean_path for clean_path, _ in matched_frames]
                    matched_noisy = [noisy_path for _, noisy_path in matched_frames]
                    sequences.append((matched_clean, matched_noisy))
        return sequences

    @staticmethod
    def _enumerate_paired_clips(
        sequences: Sequence[tuple[Sequence[Path], Sequence[Path]]],
        num_frames: int,
        windows_per_sequence: Optional[int] = None,
    ) -> list[tuple[list[Path], list[Path]]]:
        """Expand every valid sliding window, optionally keeping a fixed subset."""
        clips: list[tuple[list[Path], list[Path]]] = []
        for clean_frames, noisy_frames in sequences:
            n_windows = len(clean_frames) - num_frames + 1
            if windows_per_sequence is None or windows_per_sequence >= n_windows:
                starts = range(n_windows)
            else:
                starts = np.linspace(0, n_windows - 1, num=windows_per_sequence, dtype=int)
            for i in starts:
                i = int(i)
                clips.append((
                    list(clean_frames[i : i + num_frames]),
                    list(noisy_frames[i : i + num_frames]),
                ))
        return clips

    def _sample_frame_paths(self, idx: int) -> tuple[list[Path], list[Path]]:
        """Resolve dataset index to a concrete paired temporal window."""
        if not self._random_windows:
            clip_idx = idx // self._patches_per_image
            return self._clips[clip_idx]

        windows_per_sequence = self._windows_per_sequence * self._patches_per_image
        sequence_idx = idx // windows_per_sequence
        clean_frames, noisy_frames = self._sequences[sequence_idx]
        start = random.randint(0, len(clean_frames) - self._num_frames)
        end = start + self._num_frames
        return list(clean_frames[start:end]), list(noisy_frames[start:end])

    def __len__(self) -> int:
        if self._random_windows:
            return self._num_sequences * self._windows_per_sequence * self._patches_per_image
        return len(self._clips) * self._patches_per_image

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        """Return (noisy_clip, clean_clip, sigma_map) each shaped (T, C, H, W).

        When a spatial cache has been set via :meth:`set_spatial_cache`, returns
        a 4-tuple ``(noisy_clip, denoised_clip, clean_clip, sigma_map)`` where
        *denoised_clip* contains the pre-computed frozen spatial stage outputs
        with the same crop and augmentation applied.
        """
        clean_paths, noisy_paths = self._sample_frame_paths(idx)

        ps = self._patch_size
        if self._image_cache is not None:
            clean_imgs = [self._image_cache[p] for p in clean_paths]
            noisy_imgs = [self._image_cache[p] for p in noisy_paths]
        else:
            clean_imgs = [_load_image(path) for path in clean_paths]
            noisy_imgs = [_load_image(path) for path in noisy_paths]
        target_h = max(
            ps,
            max(img.shape[0] for img in clean_imgs),
            max(img.shape[0] for img in noisy_imgs),
        )
        target_w = max(
            ps,
            max(img.shape[1] for img in clean_imgs),
            max(img.shape[1] for img in noisy_imgs),
        )
        clean_imgs = [_pad_frame_to_shape(img, target_h, target_w) for img in clean_imgs]
        noisy_imgs = [_pad_frame_to_shape(img, target_h, target_w) for img in noisy_imgs]

        if self._crop_mode == "random":
            top = random.randint(0, target_h - ps)
            left = random.randint(0, target_w - ps)
        elif self._crop_mode == "center":
            top = (target_h - ps) // 2
            left = (target_w - ps) // 2
        elif self._crop_mode == "full":
            top = 0
            left = 0
            ps = None
        elif self._crop_mode == "grid":
            grid_patch_idx = idx % self._patches_per_image
            row = grid_patch_idx // self._crop_grid_size
            col = grid_patch_idx % self._crop_grid_size
            top = _grid_start(target_h, ps, self._crop_grid_size, row)
            left = _grid_start(target_w, ps, self._crop_grid_size, col)
        else:
            raise ValueError(f"Unsupported crop_mode: {self._crop_mode}")

        # Consistent augmentation across all frames
        flip_v = self._augment and random.random() < 0.5
        flip_h = self._augment and random.random() < 0.5
        rot_k  = random.randint(0, 3) if self._augment else 0

        noisy_frames, clean_frames, sigma_frames = [], [], []
        for clean_img, noisy_img in zip(clean_imgs, noisy_imgs):
            if ps is None:
                clean_patch = clean_img
                noisy_patch = noisy_img
            else:
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

            clean_t = _hwc_to_tensor(np.ascontiguousarray(_rgb(clean_patch)))
            noisy_t = _hwc_to_tensor(np.ascontiguousarray(_rgb(noisy_patch)))
            if clean_patch.shape[2] == 4:
                noisy_t = _apply_alpha_mask(noisy_t, clean_t, clean_patch)
            sigma_t = _local_std_sigma(noisy_t - clean_t, window=self._sigma_window)

            clean_frames.append(clean_t)
            noisy_frames.append(noisy_t)
            sigma_frames.append(sigma_t)

        denoised: Optional[Tensor] = None
        if self._spatial_cache is not None:
            # Apply the same crop and augmentation to cached denoised frames.
            # Cache tensors are (3, H, W) CPU tensors in model space.
            denoised_frames: list[Tensor] = []
            for noisy_path in noisy_paths:
                d_full = self._spatial_cache[str(noisy_path)]  # (3, H, W)
                _, fh, fw = d_full.shape
                if fh < target_h or fw < target_w:
                    d_full = F.pad(
                        d_full.unsqueeze(0),
                        (0, max(0, target_w - fw), 0, max(0, target_h - fh)),
                        mode="reflect",
                    ).squeeze(0)
                d_crop = d_full if ps is None else d_full[:, top : top + ps, left : left + ps]
                if flip_v:
                    d_crop = d_crop.flip(1)
                if flip_h:
                    d_crop = d_crop.flip(2)
                if rot_k:
                    d_crop = torch.rot90(d_crop, rot_k, [1, 2])
                denoised_frames.append(d_crop.contiguous())
            denoised = torch.stack(denoised_frames)

        return TemporalSample(
            noisy=torch.stack(noisy_frames),
            clean=torch.stack(clean_frames),
            sigma=torch.stack(sigma_frames),
            denoised=denoised,
        )

    @property
    def num_clips(self) -> int:
        if self._random_windows:
            return self._num_sequences * self._windows_per_sequence
        return len(self._clips)

    @property
    def num_sequences(self) -> int:
        return self._num_sequences


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
