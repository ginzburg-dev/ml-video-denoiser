"""Generate synthetic sample images for the training smoke test.

Produces five 128×128 PNG images in tests/fixtures/sample_images/.
All images are pure numpy/Pillow — no external data required.

Run once after cloning:
    cd training && uv run python ../tests/gen_sample_images.py
Or directly:
    python tests/gen_sample_images.py
"""

from pathlib import Path

import numpy as np
from PIL import Image


OUT_DIR = Path(__file__).resolve().parent / "fixtures" / "sample_images"
SIZE = 128


def _save(arr: np.ndarray, name: str) -> None:
    """Save a float [0,1] HxWx3 array as a uint8 PNG."""
    img = (np.clip(arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    path = OUT_DIR / name
    Image.fromarray(img).save(str(path))
    print(f"  {path.name}")


def _horizontal_gradient() -> np.ndarray:
    """Smooth RGB gradient sweeping left→right."""
    x = np.linspace(0.0, 1.0, SIZE, dtype=np.float32)
    r = x
    g = 0.5 - 0.4 * x
    b = 0.3 + 0.5 * (1.0 - x)
    img = np.stack([
        np.broadcast_to(r[np.newaxis, :], (SIZE, SIZE)),
        np.broadcast_to(g[np.newaxis, :], (SIZE, SIZE)),
        np.broadcast_to(b[np.newaxis, :], (SIZE, SIZE)),
    ], axis=-1)
    return img.astype(np.float32)


def _vertical_gradient() -> np.ndarray:
    """Smooth gradient sweeping top→bottom with different hue."""
    y = np.linspace(0.0, 1.0, SIZE, dtype=np.float32)
    r = 0.2 + 0.6 * (1.0 - y)
    g = 0.1 + 0.8 * y
    b = 0.4 * np.ones(SIZE, dtype=np.float32)
    img = np.stack([
        np.broadcast_to(r[:, np.newaxis], (SIZE, SIZE)),
        np.broadcast_to(g[:, np.newaxis], (SIZE, SIZE)),
        np.broadcast_to(b[:, np.newaxis], (SIZE, SIZE)),
    ], axis=-1)
    return img.astype(np.float32)


def _checkerboard() -> np.ndarray:
    """8×8 pixel checker pattern, two complementary hues."""
    block = 8
    img = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    for row in range(SIZE):
        for col in range(SIZE):
            even = ((row // block) + (col // block)) % 2 == 0
            if even:
                img[row, col] = [0.85, 0.20, 0.15]
            else:
                img[row, col] = [0.15, 0.65, 0.80]
    return img


def _color_patches() -> np.ndarray:
    """Four quadrants, each a solid colour with a subtle luminance gradient."""
    img = np.zeros((SIZE, SIZE, 3), dtype=np.float32)
    half = SIZE // 2
    lum = np.linspace(0.85, 1.0, half, dtype=np.float32)
    # top-left: warm orange
    img[:half, :half] = np.outer(lum, lum)[:, :, np.newaxis] * [0.95, 0.55, 0.10]
    # top-right: cool blue
    img[:half, half:] = np.outer(lum, lum)[:, :, np.newaxis] * [0.10, 0.40, 0.90]
    # bottom-left: soft green
    img[half:, :half] = np.outer(lum, lum)[:, :, np.newaxis] * [0.15, 0.80, 0.35]
    # bottom-right: light purple
    img[half:, half:] = np.outer(lum, lum)[:, :, np.newaxis] * [0.70, 0.20, 0.85]
    return img


def _sinusoidal_texture() -> np.ndarray:
    """High-frequency sine texture — exercises the model on fine detail."""
    x = np.linspace(0.0, 4.0 * np.pi, SIZE, dtype=np.float32)
    y = np.linspace(0.0, 4.0 * np.pi, SIZE, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    r = 0.5 + 0.45 * np.sin(xx)
    g = 0.5 + 0.45 * np.sin(yy)
    b = 0.5 + 0.45 * np.sin(xx * 0.7 + yy * 0.7)
    return np.stack([r, g, b], axis=-1).astype(np.float32)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Writing sample images to {OUT_DIR}")
    _save(_horizontal_gradient(), "horizontal_gradient.png")
    _save(_vertical_gradient(),   "vertical_gradient.png")
    _save(_checkerboard(),        "checkerboard.png")
    _save(_color_patches(),       "color_patches.png")
    _save(_sinusoidal_texture(),  "sinusoidal_texture.png")
    print(f"Done — 5 images written.")


if __name__ == "__main__":
    main()
