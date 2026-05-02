"""Apply a noise patch pool to a full image and save the result.

Tiles random patches from a .npz pool (produced by noise_profiler.py) across
the entire image and saves the noisy result alongside the clean original as a
side-by-side PNG.

Usage:
    uv run python preview_noise.py \\
        --image clean.png \\
        --pool pools/dark_iso3200.npz \\
        --blend add \\
        --output preview.png

    # Grain plate overlay
    uv run python preview_noise.py \\
        --image clean.exr \\
        --pool pools/grain.npz \\
        --blend overlay \\
        --output preview_grain.png
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Blend modes (numpy, operates on float32 HWC arrays)
# ---------------------------------------------------------------------------

def _blend(clean: np.ndarray, patch: np.ndarray, mode: str) -> np.ndarray:
    if mode == "add":
        return clean + patch
    if mode == "screen":
        return 1.0 - (1.0 - clean) * (1.0 - patch)
    if mode == "overlay":
        return np.where(clean < 0.5,
                        2.0 * clean * patch,
                        1.0 - 2.0 * (1.0 - clean) * (1.0 - patch))
    if mode == "soft_light":
        return (1.0 - 2.0 * patch) * clean ** 2 + 2.0 * patch * clean
    raise ValueError(f"Unknown blend mode {mode!r}. Choose: add screen overlay soft_light")


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def apply_pool_to_image(
    img: np.ndarray,
    pool: np.ndarray,
    mode: str,
    patch_size: int,
) -> np.ndarray:
    """Tile random patches from *pool* across *img* using *mode*.

    Args:
        img:        (H, W, C) float32 clean image.
        pool:       (N, C, pH, pW) float32 patch pool.
        mode:       Blend mode — add | screen | overlay | soft_light.
        patch_size: Tile size used when tiling the image.

    Returns:
        (H, W, C) float32 noisy image.
    """
    h, w, c = img.shape
    n, pc, ph, pw = pool.shape
    result = img.copy()

    for y in range(0, h, patch_size):
        for x in range(0, w, patch_size):
            y2 = min(y + patch_size, h)
            x2 = min(x + patch_size, w)
            th = y2 - y
            tw = x2 - x

            if ph < th or pw < tw:
                print(
                    f"Warning: pool patch ({ph}×{pw}) smaller than tile ({th}×{tw}), skipping tile.",
                    file=sys.stderr,
                )
                continue

            idx = random.randrange(n)
            top = random.randint(0, ph - th)
            left = random.randint(0, pw - tw)
            patch = pool[idx, :, top : top + th, left : left + tw]  # (C, th, tw)
            patch = patch.transpose(1, 2, 0)  # (th, tw, C)

            # Match channels
            if pc != c:
                patch = patch[:, :, :c] if pc > c else patch.mean(axis=2, keepdims=True).repeat(c, axis=2)

            result[y:y2, x:x2, :] = _blend(result[y:y2, x:x2, :], patch, mode)

    return result


def load_image(path: Path) -> np.ndarray:
    """Load any image (PNG, EXR, TIFF, …) as float32 (H, W, C) in [0, 1]."""
    from noise_profiler import _read_frame
    return _read_frame(path)


def save_image(img: np.ndarray, path: Path) -> None:
    """Save a float32 (H, W, C) image.  Clamps to [0, 1] before writing."""
    import imageio.v3 as iio
    out = np.clip(img, 0.0, 1.0)
    if path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        out = (out * 255.0).astype(np.uint8)
    elif path.suffix.lower() in (".tif", ".tiff"):
        out = (out * 65535.0).astype(np.uint16)
    iio.imwrite(str(path), out)


def side_by_side(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Concatenate two (H, W, C) images horizontally with a 4-pixel grey divider."""
    divider = np.full((a.shape[0], 4, a.shape[2]), 0.5, dtype=np.float32)
    return np.concatenate([a, divider, b], axis=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a noise patch pool to a full image and save a preview.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", required=True, metavar="PATH",
                        help="Input clean image (PNG, EXR, TIFF, …).")
    parser.add_argument("--pool", required=True, metavar="PATH",
                        help="Patch pool .npz produced by noise_profiler.py.")
    parser.add_argument("--blend", default="add", metavar="MODE",
                        choices=["add", "screen", "overlay", "soft_light"],
                        help="Blend mode (default: add).")
    parser.add_argument("--patch-size", type=int, default=128, metavar="N",
                        help="Tile size for tiling patches over the image (default: 128).")
    parser.add_argument("--output", default=None, metavar="PATH",
                        help="Output path (default: <image_stem>_noisy.png next to input).")
    parser.add_argument("--seed", type=int, default=None, metavar="N",
                        help="Random seed for reproducibility.")

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    image_path = Path(args.image)
    pool_path = Path(args.pool)

    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")
    if not pool_path.exists():
        parser.error(f"Pool not found: {pool_path}")

    out_path = Path(args.output) if args.output else image_path.with_name(
        image_path.stem + "_noisy.png"
    )

    print(f"Loading image: {image_path}", file=sys.stderr)
    img = load_image(image_path)

    print(f"Loading pool:  {pool_path}", file=sys.stderr)
    data = np.load(pool_path)
    pool = data["residuals"].astype(np.float32)
    if pool.ndim == 4 and pool.shape[-1] in (1, 3, 4):
        pool = pool.transpose(0, 3, 1, 2)  # (N, H, W, C) → (N, C, H, W)
    print(f"  {len(pool)} patches of size {pool.shape[-2]}×{pool.shape[-1]}", file=sys.stderr)

    print(f"Applying noise (blend={args.blend}, patch_size={args.patch_size})…", file=sys.stderr)
    noisy = apply_pool_to_image(img, pool, args.blend, args.patch_size)

    preview = side_by_side(img, noisy)
    save_image(preview, out_path)
    print(f"Saved → {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
