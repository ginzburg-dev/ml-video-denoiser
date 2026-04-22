"""CLI noise preview: apply noise generators at multiple levels to a user image.

Pass any image and get a labelled PNG grid comparing every generator across
noise levels side-by-side — useful for quickly assessing whether the synthetic
noise looks realistic before starting a training run.

Usage:
    # All generators, default levels:
    uv run python tests/noise_preview.py --image photo.jpg

    # Specific generators and custom ISO range:
    uv run python tests/noise_preview.py \\
        --image photo.jpg \\
        --generators gaussian camera \\
        --iso-min 200 --iso-max 12800

    # Control noise intensity sweep and output location:
    uv run python tests/noise_preview.py \\
        --image photo.jpg \\
        --n-levels 6 \\
        --out /tmp/preview.png

    # Include temporal clip strip (per-frame vs clip-consistent noise):
    uv run python tests/noise_preview.py \\
        --image photo.jpg \\
        --temporal \\
        --n-frames 5

Output:
    <image_stem>_noise_preview.png  — generator × level grid
    <image_stem>_temporal.png       — temporal consistency strip (if --temporal)
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = REPO_ROOT / "training"
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from noise_generators import (
    CameraNoiseGenerator,
    GaussianNoiseGenerator,
    MixedNoiseGenerator,
    PoissonGaussianNoiseGenerator,
)


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------


def _load(path: Path, max_side: int = 512) -> torch.Tensor:
    """Load image → (C, H, W) float32 in [0, 1], optionally downscaled."""
    if path.suffix.lower() == ".exr":
        import OpenEXR
        with OpenEXR.File(str(path)) as exr:
            channels = exr.parts[0].channels
            if "RGB" in channels:
                data = channels["RGB"].pixels
                img = np.stack([np.asarray(data[..., c], dtype=np.float32) for c in range(3)], axis=-1)
            elif "RGBA" in channels:
                data = channels["RGBA"].pixels
                img = np.stack([np.asarray(data[..., c], dtype=np.float32) for c in range(3)], axis=-1)
            else:
                # Try individual R, G, B channels
                r = np.asarray(channels["R"].pixels, dtype=np.float32)
                g = np.asarray(channels["G"].pixels, dtype=np.float32)
                b = np.asarray(channels["B"].pixels, dtype=np.float32)
                img = np.stack([r, g, b], axis=-1)
        # EXR is linear HDR — tone-map to [0,1] for display using log1p
        img = np.log1p(np.clip(img, 0.0, None))
        img = img / (img.max() + 1e-8)
    else:
        import imageio.v3 as iio
        img = iio.imread(str(path)).astype(np.float32)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.shape[2] > 3:
            img = img[:, :, :3]
        if img.max() > 1.5:
            img /= 255.0

    # Downscale long side to max_side for fast preview
    h, w = img.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
        from PIL import Image
        pil = Image.fromarray((img * 255).astype(np.uint8))
        pil = pil.resize((new_w, new_h), Image.LANCZOS)
        img = np.asarray(pil).astype(np.float32) / 255.0

    return torch.from_numpy(img.transpose(2, 0, 1))  # (C, H, W)


def _to_uint8(t: torch.Tensor) -> np.ndarray:
    t = t.float().clamp(0, 1)
    if t.shape[0] == 1:
        t = t.expand(3, -1, -1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _amplify(residual: torch.Tensor, factor: float = 5.0) -> torch.Tensor:
    return (residual * factor + 0.5).clamp(0, 1)


# ---------------------------------------------------------------------------
# Generator factory — builds (label, generator) pairs at multiple levels
# ---------------------------------------------------------------------------


def _gaussian_levels(n: int, sigma_min: float, sigma_max: float):
    sigmas = np.linspace(sigma_min, sigma_max, n)
    return [
        (f"Gaussian σ={s:.3f}", GaussianNoiseGenerator(s, s))
        for s in sigmas
    ]


def _poisson_levels(n: int):
    # K values spanning mild → severe sensor noise
    ks = np.geomspace(0.002, 0.08, n)
    return [
        (f"Poisson K={k:.3f}", PoissonGaussianNoiseGenerator(
            gain_range=(k, k), read_sigma_range=(0.003, 0.003)
        ))
        for k in ks
    ]


def _camera_levels(n: int, iso_min: float, iso_max: float):
    isos = np.geomspace(iso_min, iso_max, n)
    return [
        (f"Camera ISO {int(iso)}", CameraNoiseGenerator(
            iso_range=(iso, iso), fixed_pattern_strength=0.002
        ))
        for iso in isos
    ]


def _mixed_levels(n: int):
    gen = MixedNoiseGenerator.default()
    return [(f"Mixed sample {i+1}", gen) for i in range(n)]


GENERATOR_FACTORIES = {
    "gaussian": _gaussian_levels,
    "poisson":  _poisson_levels,
    "camera":   _camera_levels,
    "mixed":    _mixed_levels,
}


# ---------------------------------------------------------------------------
# Grid drawing
# ---------------------------------------------------------------------------


def _make_grid(
    rows: list[list[np.ndarray]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cell_size: int = 3,
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_size, n_rows * cell_size + 0.7),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")

    for r, (row_data, rlabel) in enumerate(zip(rows, row_labels)):
        for c in range(n_cols):
            ax = axes[r][c]
            if c < len(row_data):
                ax.imshow(row_data[c], interpolation="nearest")
                if r == 0 and c < len(col_labels):
                    ax.set_title(col_labels[c], fontsize=8, pad=3)
            else:
                ax.axis("off")
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(rlabel, fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=80)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main grid: generator × level
# ---------------------------------------------------------------------------


def build_preview_grid(
    clean: torch.Tensor,
    selected: list[str],
    n_levels: int,
    sigma_min: float,
    sigma_max: float,
    iso_min: float,
    iso_max: float,
    amplify: float,
) -> "matplotlib.figure.Figure":
    """One figure: rows = (generator, level), cols = Clean | Noisy | Residual×amp | Sigma."""
    rows: list[list[np.ndarray]] = []
    row_labels: list[str] = []

    # Always put clean as very first row so there's a fixed reference
    rows.append([_to_uint8(clean), _to_uint8(clean), _to_uint8(clean), _to_uint8(clean)])
    row_labels.append("Clean\n(reference)")

    factories = {
        "gaussian": lambda n: _gaussian_levels(n, sigma_min, sigma_max),
        "poisson":  lambda n: _poisson_levels(n),
        "camera":   lambda n: _camera_levels(n, iso_min, iso_max),
        "mixed":    lambda n: _mixed_levels(n),
    }

    with torch.no_grad():
        for gen_key in selected:
            levels = factories[gen_key](n_levels)
            for label, gen in levels:
                noisy, clean_out, sigma = gen(clean)
                residual = _amplify(noisy - clean_out, amplify)
                sigma_disp = (sigma / 0.30).clamp(0, 1)
                rows.append([
                    _to_uint8(clean_out),
                    _to_uint8(noisy),
                    _to_uint8(residual),
                    _to_uint8(sigma_disp),
                ])
                row_labels.append(label)

    col_labels = [
        "Clean",
        "Noisy",
        f"Residual ×{amplify:.0f}\n(grey=zero)",
        "Sigma map\n(white=σ≥0.30)",
    ]
    title = "Noise preview — generator × level"
    return _make_grid(rows, row_labels, col_labels, title)


# ---------------------------------------------------------------------------
# Temporal strip
# ---------------------------------------------------------------------------


def build_temporal_grid(
    clean: torch.Tensor,
    n_frames: int,
    iso_min: float,
    iso_max: float,
    amplify: float = 8.0,
) -> "matplotlib.figure.Figure":
    """Temporal consistency strip: per-frame ISO vs clip-consistent ISO.

    Shows the key difference between applying CameraNoiseGenerator fresh every
    frame (bad — noise level jumps) versus using for_clip() (good — ISO and
    row-banding fixed for the whole clip, matching real camera behaviour).

    Columns: frame 0…N, then a summary cell with mean |Δ| between frames.
    Rows:
        A  Per-frame noisy
        B  Per-frame |frame diff| ×amp
        C  Clip noisy
        D  Clip |frame diff| ×amp
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cam = CameraNoiseGenerator(iso_range=(iso_min, iso_max), fixed_pattern_strength=0.002)

    with torch.no_grad():
        per_noisy = [cam(clean)[0] for _ in range(n_frames)]
        clip_fn = cam.for_clip()
        clip_noisy = [clip_fn(clean)[0] for _ in range(n_frames)]

    def _diffs(frames):
        return [_amplify((b - a).abs(), amplify) for a, b in zip(frames, frames[1:])]

    def _mean_delta(frames) -> str:
        v = np.mean([(b - a).abs().mean().item() for a, b in zip(frames, frames[1:])])
        return f"Δ̄={v:.4f}"

    per_diffs = _diffs(per_noisy)
    clip_diffs = _diffs(clip_noisy)

    cell = 2.2
    n_data_cols = n_frames
    n_cols = n_data_cols + 1  # + summary
    n_rows = 4

    row_labels = [
        "Per-frame\nnoisy",
        f"Per-frame |Δ|\n×{amplify:.0f}",
        "Clip\nnoisy",
        f"Clip |Δ|\n×{amplify:.0f}",
    ]

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell, n_rows * cell + 0.8),
        squeeze=False,
    )
    fig.suptitle(
        "Temporal clip consistency\n"
        "Per-frame: ISO resampled every frame → noise jumps between frames\n"
        "Clip: ISO + banding fixed once → stable, matches real camera",
        fontsize=10, fontweight="bold",
    )

    def _fill(row_idx, frames, summary):
        for c, t in enumerate(frames):
            axes[row_idx][c].imshow(_to_uint8(t), interpolation="nearest")
            axes[row_idx][c].set_xticks([])
            axes[row_idx][c].set_yticks([])
        axes[row_idx][n_cols - 1].text(0.5, 0.5, summary,
            ha="center", va="center", fontsize=10,
            transform=axes[row_idx][n_cols - 1].transAxes)
        axes[row_idx][n_cols - 1].set_xticks([])
        axes[row_idx][n_cols - 1].set_yticks([])
        axes[row_idx][0].set_ylabel(row_labels[row_idx], fontsize=8,
            rotation=0, ha="right", va="center", labelpad=80)

    _fill(0, per_noisy, "—")
    _fill(1, per_diffs, _mean_delta(per_noisy))
    _fill(2, clip_noisy, "—")
    _fill(3, clip_diffs, _mean_delta(clip_noisy))

    for c in range(n_data_cols):
        axes[0][c].set_title(f"frame {c}", fontsize=8, pad=3)
    axes[0][n_cols - 1].set_title("mean |Δ|", fontsize=8, pad=3)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _open(path: Path) -> None:
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        elif platform.system() == "Linux":
            subprocess.Popen(["xdg-open", str(path)])
        elif platform.system() == "Windows":
            subprocess.Popen(["explorer", str(path)])
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preview noise generators on a user-supplied image.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--image", required=True, metavar="PATH",
                        help="Input image (PNG, JPG, TIFF, EXR).")
    parser.add_argument("--generators", nargs="+",
                        choices=["gaussian", "poisson", "camera", "mixed"],
                        default=["gaussian", "poisson", "camera", "mixed"],
                        metavar="NAME",
                        help="Which generators to include. "
                             "Choices: gaussian poisson camera mixed. "
                             "Default: all four.")
    parser.add_argument("--n-levels", type=int, default=4, metavar="N",
                        help="How many noise levels to sweep per generator (default: 4).")
    parser.add_argument("--sigma-min", type=float, default=5.0 / 255.0, metavar="F",
                        help="Min Gaussian sigma (normalised, default: 5/255).")
    parser.add_argument("--sigma-max", type=float, default=75.0 / 255.0, metavar="F",
                        help="Max Gaussian sigma (normalised, default: 75/255).")
    parser.add_argument("--iso-min", type=float, default=200.0, metavar="F",
                        help="Min ISO for camera generator (default: 200).")
    parser.add_argument("--iso-max", type=float, default=12800.0, metavar="F",
                        help="Max ISO for camera generator (default: 12800).")
    parser.add_argument("--amplify", type=float, default=5.0, metavar="F",
                        help="Residual amplification factor for display (default: 5).")
    parser.add_argument("--max-side", type=int, default=512, metavar="PX",
                        help="Downscale image long side to this for speed (default: 512).")
    parser.add_argument("--temporal", action="store_true",
                        help="Also produce a temporal clip consistency strip.")
    parser.add_argument("--n-frames", type=int, default=5, metavar="N",
                        help="Frames in the temporal strip (default: 5).")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Output PNG path. Default: <image_stem>_noise_preview.png "
                             "next to the input image.")
    parser.add_argument("--no-open", action="store_true",
                        help="Skip auto-opening the output PNG.")
    args = parser.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image_path = Path(args.image)
    if not image_path.exists():
        parser.error(f"Image not found: {image_path}")

    out_path = Path(args.out) if args.out else image_path.parent / f"{image_path.stem}_noise_preview.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading {image_path} (max side {args.max_side}px)…")
    clean = _load(image_path, max_side=args.max_side)
    print(f"  Tensor shape: {tuple(clean.shape)}  range [{clean.min():.3f}, {clean.max():.3f}]")

    # --- Main preview grid ---
    print(f"\nGenerating noise preview ({args.generators}, {args.n_levels} levels each)…")
    fig = build_preview_grid(
        clean,
        selected=args.generators,
        n_levels=args.n_levels,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        iso_min=args.iso_min,
        iso_max=args.iso_max,
        amplify=args.amplify,
    )
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    print(f"  Saved: {out_path}")

    written = [out_path]

    # --- Temporal strip ---
    if args.temporal:
        temporal_path = out_path.parent / f"{image_path.stem}_temporal.png"
        print(f"\nGenerating temporal consistency strip ({args.n_frames} frames)…")
        fig = build_temporal_grid(
            clean,
            n_frames=args.n_frames,
            iso_min=args.iso_min,
            iso_max=args.iso_max,
            amplify=8.0,
        )
        fig.savefig(temporal_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"  Saved: {temporal_path}")
        written.append(temporal_path)

    if not args.no_open:
        for p in written:
            _open(p)


if __name__ == "__main__":
    main()
