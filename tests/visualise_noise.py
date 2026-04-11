"""Visual diagnostic for noise generators and paired dataset loading.

Produces PNG grids that let you inspect how each noise type affects images,
what paired dataset samples look like, and how the sigma_map relates to the
actual noise structure.

Run from the repo root:
    cd training && uv run python ../tests/visualise_noise.py

Or with options:
    uv run python ../tests/visualise_noise.py --out-dir /tmp/noise_diagnostics
    uv run python ../tests/visualise_noise.py --patch-pool pools/camera.npz
    uv run python ../tests/visualise_noise.py --no-open   # skip auto-open

Output layout (written to tests/fixtures/noise_diagnostics/ by default):
    noise_types.png        — all generators × all sample images
    generator_<name>.png   — one generator across all images (larger)
    paired_dataset.png     — PairedPatchDataset: clean / noisy / residual / sigma
    patch_pool_residuals.png — raw noise patches from a .npz pool (if provided)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Make sure the training directory is importable
REPO_ROOT  = Path(__file__).resolve().parent.parent
TRAIN_DIR  = REPO_ROOT / "training"
SAMPLE_DIR = REPO_ROOT / "tests" / "fixtures" / "sample_images"
DEFAULT_OUT = REPO_ROOT / "tests" / "fixtures" / "noise_diagnostics"

if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from noise_generators import (
    GaussianNoiseGenerator,
    MixedNoiseGenerator,
    PoissonGaussianNoiseGenerator,
    RealNoiseInjectionGenerator,
    RealRAWNoiseGenerator,
)
from dataset import PairedPatchDataset


# ---------------------------------------------------------------------------
# Image I/O helpers
# ---------------------------------------------------------------------------


def _load_as_tensor(path: Path) -> torch.Tensor:
    """Load a PNG/JPG into a (C, H, W) float32 tensor in [0, 1]."""
    import imageio.v3 as iio
    img = iio.imread(str(path)).astype(np.float32)
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    if img.shape[2] > 3:
        img = img[:, :, :3]
    if img.max() > 1.5:
        img /= 255.0
    return torch.from_numpy(img.transpose(2, 0, 1))  # (C, H, W)


def _tensor_to_uint8(t: torch.Tensor) -> np.ndarray:
    """Convert (C, H, W) float tensor to (H, W, 3) uint8 for display."""
    t = t.float().clamp(0, 1)
    if t.shape[0] == 1:
        t = t.expand(3, -1, -1)
    return (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)


def _amplify_residual(residual: torch.Tensor, factor: float = 5.0) -> torch.Tensor:
    """Shift + scale a noise residual into [0, 1] for visualisation."""
    return (residual * factor + 0.5).clamp(0, 1)


def _sigma_to_display(sigma: torch.Tensor) -> torch.Tensor:
    """Scale sigma_map to [0, 1] for display (maps 0–0.3 to full range)."""
    return (sigma / 0.30).clamp(0, 1)


# ---------------------------------------------------------------------------
# Synthetic patch pool / profile builders (so the script works without
# real camera footage)
# ---------------------------------------------------------------------------


def _make_synthetic_pool(tmp_dir: Path, sigma: float = 0.03) -> Path:
    """Write a synthetic noise patch pool to tmp_dir/pool.npz."""
    rng = np.random.default_rng(0)
    # 40 patches, (C=3, H=128, W=128), Gaussian residuals with spatial structure
    residuals = np.zeros((40, 3, 128, 128), dtype=np.float32)
    for i in range(40):
        base = rng.normal(0, sigma, (3, 128, 128)).astype(np.float32)
        # Add a faint horizontal banding pattern to mimic real sensor structure
        band = rng.normal(0, sigma * 0.3, (3, 128, 1)).astype(np.float32)
        residuals[i] = base + band
    path = tmp_dir / "synthetic_pool.npz"
    np.savez_compressed(path, residuals=residuals)
    return path


def _make_synthetic_profile(tmp_dir: Path) -> Path:
    """Write a synthetic noise profile JSON to tmp_dir/profile.json."""
    profile = {
        "camera": "Synthetic (diagnostic)",
        "iso_profiles": {
            "iso_800":  {"K": 0.003, "sigma_r": 0.002},
            "iso_3200": {"K": 0.012, "sigma_r": 0.006},
            "iso_12800": {"K": 0.048, "sigma_r": 0.014},
        },
    }
    path = tmp_dir / "synthetic_profile.json"
    path.write_text(json.dumps(profile))
    return path


def _make_synthetic_paired_dirs(tmp_dir: Path) -> tuple[Path, Path]:
    """Write 5 synthetic clean/noisy pairs into tmp_dir/clean and tmp_dir/noisy."""
    import imageio.v3 as iio

    clean_dir = tmp_dir / "clean"
    noisy_dir = tmp_dir / "noisy"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noisy_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    for img_path in sorted(SAMPLE_DIR.glob("*.png")):
        import imageio.v3 as iio
        clean = iio.imread(str(img_path)).astype(np.float32) / 255.0
        # Add spatially-structured noise: Gaussian + slight horizontal banding
        noise_base  = rng.normal(0, 0.05, clean.shape).astype(np.float32)
        noise_band  = rng.normal(0, 0.02, (clean.shape[0], 1, clean.shape[2])).astype(np.float32)
        noise       = noise_base + noise_band
        noisy       = np.clip(clean + noise, 0, 1)
        iio.imwrite(str(clean_dir / img_path.name), (clean * 255).astype(np.uint8))
        iio.imwrite(str(noisy_dir / img_path.name), (noisy * 255).astype(np.uint8))

    return clean_dir, noisy_dir


# ---------------------------------------------------------------------------
# Grid drawing helpers
# ---------------------------------------------------------------------------


def _make_grid(
    rows: list[list[np.ndarray]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    cell_size: int = 3,
) -> "matplotlib.figure.Figure":
    """Draw a labelled grid of images.

    Args:
        rows: rows[i][j] is a (H, W, 3) uint8 array for cell (i, j).
        row_labels: Label for each row (left side).
        col_labels: Label for each column (top).
        title: Figure title.
        cell_size: Inches per cell.

    Returns:
        Matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_rows = len(rows)
    n_cols = max(len(r) for r in rows)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell_size, n_rows * cell_size + 0.6),
        squeeze=False,
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)

    for r, (row_data, row_label) in enumerate(zip(rows, row_labels)):
        for c in range(n_cols):
            ax = axes[r][c]
            if c < len(row_data):
                ax.imshow(row_data[c], interpolation="nearest")
                if r == 0 and c < len(col_labels):
                    ax.set_title(col_labels[c], fontsize=9, pad=4)
            else:
                ax.axis("off")
            if c == 0:
                ax.set_ylabel(row_label, fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=60)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.tight_layout()
    return fig


def _four_panel(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    sigma: torch.Tensor,
    noise_amplify: float = 5.0,
) -> list[np.ndarray]:
    """Return [clean, noisy, residual×amp, sigma_map] as uint8 arrays."""
    residual = noisy - clean
    return [
        _tensor_to_uint8(clean),
        _tensor_to_uint8(noisy),
        _tensor_to_uint8(_amplify_residual(residual, noise_amplify)),
        _tensor_to_uint8(_sigma_to_display(sigma)),
    ]


# ---------------------------------------------------------------------------
# Per-generator diagnostic
# ---------------------------------------------------------------------------


def _run_generator_on_images(
    generator,
    images: list[tuple[str, torch.Tensor]],
    noise_amplify: float = 5.0,
) -> list[list[np.ndarray]]:
    """Apply *generator* to each image and return rows for _make_grid."""
    rows = []
    with torch.no_grad():
        for _name, clean in images:
            noisy, clean_out, sigma = generator(clean)
            rows.append(_four_panel(clean_out, noisy, sigma, noise_amplify))
    return rows


def save_generator_grid(
    generator,
    name: str,
    images: list[tuple[str, torch.Tensor]],
    out_dir: Path,
) -> Path:
    """One PNG: all sample images run through *generator*.

    Layout:
        rows    = sample images
        columns = Clean | Noisy | Noise residual ×5 | Sigma map
    """
    rows = _run_generator_on_images(generator, images)
    row_labels = [stem for stem, _ in images]
    col_labels = [
        "Clean",
        "Noisy",
        "Noise residual ×5\n(0.5 = zero)",
        "Sigma map\n(white = σ≥0.30)",
    ]
    fig = _make_grid(rows, row_labels, col_labels, title=f"Noise type: {name}")
    safe = re.sub(r"[^\w\-]", "_", name.lower())
    out_path = out_dir / f"generator_{safe}.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Overview grid: all generators × one representative image
# ---------------------------------------------------------------------------


def save_noise_types_overview(
    generators: list[tuple[str, object]],
    images: list[tuple[str, torch.Tensor]],
    out_dir: Path,
) -> Path:
    """Master overview: rows=generators, cols=images, single clean strip on top."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # We'll show: for each generator, side-by-side noisy versions of all images.
    # Top row: clean originals. Then one row per generator.
    col_labels = [stem for stem, _ in images]
    n_cols     = len(images)
    n_rows     = 1 + len(generators)   # 1 clean row + N generator rows
    cell = 2.2

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * cell, n_rows * cell + 0.5),
        squeeze=False,
    )
    fig.suptitle("Noise types overview  (top row: clean originals)",
                 fontsize=12, fontweight="bold")

    # Row 0: clean originals
    for c, (stem, clean) in enumerate(images):
        axes[0][c].imshow(_tensor_to_uint8(clean), interpolation="nearest")
        axes[0][c].set_title(stem, fontsize=7)
        axes[0][c].set_xticks([])
        axes[0][c].set_yticks([])
    axes[0][0].set_ylabel("Clean", fontsize=8, rotation=0,
                           ha="right", va="center", labelpad=50)

    with torch.no_grad():
        for r, (gen_name, generator) in enumerate(generators, start=1):
            for c, (_stem, clean) in enumerate(images):
                noisy, _, _ = generator(clean)
                axes[r][c].imshow(_tensor_to_uint8(noisy), interpolation="nearest")
                axes[r][c].set_xticks([])
                axes[r][c].set_yticks([])
            axes[r][0].set_ylabel(gen_name, fontsize=8, rotation=0,
                                  ha="right", va="center", labelpad=50)

    fig.tight_layout()
    out_path = out_dir / "noise_types_overview.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Paired dataset diagnostic
# ---------------------------------------------------------------------------


def save_paired_dataset_grid(
    clean_dir: Path,
    noisy_dir: Path,
    out_dir: Path,
    n_samples: int = 5,
    patch_size: int = 128,
) -> Path:
    """Show PairedPatchDataset samples: Clean | Noisy | Residual×5 | Sigma."""
    ds = PairedPatchDataset(
        clean_dir, noisy_dir,
        patch_size=patch_size,
        patches_per_image=4,
        augment=True,
        match_by_name=True,
    )

    rows = []
    row_labels = []
    for i in range(min(n_samples, len(ds))):
        noisy, clean, sigma = ds[i]
        rows.append(_four_panel(clean, noisy, sigma))
        row_labels.append(f"sample {i}")

    col_labels = [
        "Clean",
        "Noisy (real pair)",
        "Residual ×5",
        "Sigma map",
    ]
    fig = _make_grid(
        rows, row_labels, col_labels,
        title="PairedPatchDataset — clean/noisy pairs with augmentation",
    )
    out_path = out_dir / "paired_dataset.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Patch pool residual visualisation
# ---------------------------------------------------------------------------


def save_patch_pool_grid(pool_path: Path, out_dir: Path, n_patches: int = 20) -> Path:
    """Show raw noise residuals stored in a .npz patch pool.

    Each cell shows one patch drawn directly from the pool (amplified ×5
    around 0.5 so positive/negative noise is visible as bright/dark).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data      = np.load(pool_path)
    residuals = data["residuals"]           # (N, C, H, W)
    n_total   = residuals.shape[0]
    n_show    = min(n_patches, n_total)
    n_cols    = 5
    n_rows    = (n_show + n_cols - 1) // n_cols

    indices = np.random.default_rng(0).choice(n_total, n_show, replace=False)
    cell    = 2.5

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(n_cols * cell, n_rows * cell + 0.6),
                             squeeze=False)
    fig.suptitle(
        f"Patch pool residuals from: {pool_path.name}\n"
        f"({n_total} patches total, showing {n_show})  —  "
        "centre grey = zero noise, bright = positive, dark = negative",
        fontsize=10, fontweight="bold",
    )

    for idx, patch_idx in enumerate(indices):
        r, c = divmod(idx, n_cols)
        patch = torch.from_numpy(residuals[patch_idx])   # (C, H, W)
        display = _amplify_residual(patch, factor=5.0)
        axes[r][c].imshow(_tensor_to_uint8(display), interpolation="nearest")
        axes[r][c].set_title(f"#{patch_idx}", fontsize=7)
        axes[r][c].set_xticks([])
        axes[r][c].set_yticks([])

    # Hide unused cells
    for idx in range(len(indices), n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.tight_layout()
    out_path = out_dir / "patch_pool_residuals.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Sigma map detail: compare sigma across generators on one image
# ---------------------------------------------------------------------------


def save_sigma_comparison(
    generators: list[tuple[str, object]],
    clean: torch.Tensor,
    image_name: str,
    out_dir: Path,
) -> Path:
    """Single-image sigma_map comparison across all generators.

    Shows per-pixel noise level estimates side-by-side so you can see how
    spatially uniform Gaussian sigma is versus the signal-dependent structure
    of Poisson-Gaussian, versus the authentic spatial pattern of a real pool.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(generators)
    fig, axes = plt.subplots(2, n, figsize=(n * 3.0, 6.5), squeeze=False)
    fig.suptitle(
        f"Sigma map comparison — image: {image_name}\n"
        "(top: noisy image, bottom: sigma_map — white = σ≥0.30)",
        fontsize=11, fontweight="bold",
    )

    with torch.no_grad():
        for col, (gen_name, generator) in enumerate(generators):
            noisy, _, sigma = generator(clean)

            axes[0][col].imshow(_tensor_to_uint8(noisy), interpolation="nearest")
            axes[0][col].set_title(gen_name, fontsize=9)
            axes[0][col].set_xticks([])
            axes[0][col].set_yticks([])

            sigma_disp = _sigma_to_display(sigma)
            im = axes[1][col].imshow(
                _tensor_to_uint8(sigma_disp), interpolation="nearest"
            )
            axes[1][col].set_xticks([])
            axes[1][col].set_yticks([])
            mean_s = sigma.mean().item()
            axes[1][col].set_xlabel(f"mean σ = {mean_s:.4f}", fontsize=8)

    axes[0][0].set_ylabel("Noisy", fontsize=9, rotation=0,
                          ha="right", va="center", labelpad=45)
    axes[1][0].set_ylabel("Sigma map", fontsize=9, rotation=0,
                          ha="right", va="center", labelpad=45)

    fig.tight_layout()
    out_path = out_dir / "sigma_comparison.png"
    fig.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return out_path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run(
    out_dir: Path,
    pool_path: Optional[Path] = None,
    profile_path: Optional[Path] = None,
    auto_open: bool = True,
) -> list[Path]:
    """Run all diagnostics and return the list of written PNG paths."""
    import matplotlib
    matplotlib.use("Agg")

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="denoiser_diag_"))

    # ------------------------------------------------------------------
    # Load sample images
    # ------------------------------------------------------------------
    if not SAMPLE_DIR.exists() or not list(SAMPLE_DIR.glob("*.png")):
        print(
            f"ERROR: sample images not found in {SAMPLE_DIR}\n"
            "Run:  cd training && uv run python ../tests/gen_sample_images.py",
            file=sys.stderr,
        )
        sys.exit(1)

    images: list[tuple[str, torch.Tensor]] = [
        (p.stem, _load_as_tensor(p))
        for p in sorted(SAMPLE_DIR.glob("*.png"))
    ]
    print(f"Loaded {len(images)} sample images from {SAMPLE_DIR}")

    # ------------------------------------------------------------------
    # Build generators
    # Use caller-supplied pool/profile if given; otherwise synthetic ones
    # ------------------------------------------------------------------
    use_synthetic_pool    = pool_path is None
    use_synthetic_profile = profile_path is None

    if use_synthetic_pool:
        pool_path = _make_synthetic_pool(tmp)
        pool_label = "Real inject (synthetic pool)"
    else:
        pool_label = f"Real inject ({pool_path.name})"

    if use_synthetic_profile:
        profile_path = _make_synthetic_profile(tmp)
        profile_label = "Real RAW (synthetic profile)"
    else:
        profile_label = f"Real RAW ({profile_path.name})"

    generators: list[tuple[str, object]] = [
        ("Gaussian (σ~U[0, 75/255])",
            GaussianNoiseGenerator(0.0, 75.0 / 255.0)),
        ("Gaussian (σ~U[10/255, 25/255])",
            GaussianNoiseGenerator(10.0 / 255.0, 25.0 / 255.0)),
        ("Poisson-Gaussian",
            PoissonGaussianNoiseGenerator()),
        (pool_label,
            RealNoiseInjectionGenerator(str(pool_path))),
        (profile_label,
            RealRAWNoiseGenerator(str(profile_path))),
        ("Mixed (all four)",
            MixedNoiseGenerator.default(
                patch_pool=str(pool_path),
                profile_json=str(profile_path),
            )),
    ]

    # ------------------------------------------------------------------
    # Paired dataset dirs
    # ------------------------------------------------------------------
    paired_clean, paired_noisy = _make_synthetic_paired_dirs(tmp)

    written: list[Path] = []

    # ------------------------------------------------------------------
    # 1.  Per-generator grids
    # ------------------------------------------------------------------
    print("\n── Per-generator grids ──")
    for name, gen in generators:
        written.append(save_generator_grid(gen, name, images, out_dir))

    # ------------------------------------------------------------------
    # 2.  Overview: all generators × all images
    # ------------------------------------------------------------------
    print("\n── Overview grid ──")
    written.append(save_noise_types_overview(generators, images, out_dir))

    # ------------------------------------------------------------------
    # 3.  Paired dataset
    # ------------------------------------------------------------------
    print("\n── Paired dataset ──")
    written.append(
        save_paired_dataset_grid(paired_clean, paired_noisy, out_dir)
    )

    # ------------------------------------------------------------------
    # 4.  Patch pool residuals
    # ------------------------------------------------------------------
    print("\n── Patch pool residuals ──")
    written.append(save_patch_pool_grid(pool_path, out_dir))

    # ------------------------------------------------------------------
    # 5.  Sigma comparison (one representative image)
    # ------------------------------------------------------------------
    print("\n── Sigma map comparison ──")
    checkerboard = next((t for stem, t in images if stem == "checkerboard"), images[0][1])
    written.append(
        save_sigma_comparison(
            generators,
            checkerboard,
            "checkerboard",
            out_dir,
        )
    )

    print(f"\nAll diagnostics written to: {out_dir}")

    if auto_open:
        _try_open(out_dir)

    return written


def _try_open(path: Path) -> None:
    """Best-effort: open the output directory in the system file manager."""
    import subprocess, platform
    try:
        if platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        elif platform.system() == "Linux":
            subprocess.Popen(["xdg-open", str(path)])
        elif platform.system() == "Windows":
            subprocess.Popen(["explorer", str(path)])
    except Exception:
        pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate visual noise diagnostics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--out-dir", default=str(DEFAULT_OUT), metavar="DIR",
        help=f"Output directory (default: {DEFAULT_OUT})",
    )
    parser.add_argument(
        "--patch-pool", default=None, metavar="PATH",
        help="Real .npz patch pool from noise_profiler.py.  "
             "If omitted, a synthetic pool is generated for illustration.",
    )
    parser.add_argument(
        "--noise-profile", default=None, metavar="PATH",
        help="Real noise profile JSON from noise_profiler.py.  "
             "If omitted, a synthetic profile is generated for illustration.",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Do not auto-open the output directory after writing.",
    )
    args = parser.parse_args()

    run(
        out_dir=Path(args.out_dir),
        pool_path=Path(args.patch_pool) if args.patch_pool else None,
        profile_path=Path(args.noise_profile) if args.noise_profile else None,
        auto_open=not args.no_open,
    )


if __name__ == "__main__":
    main()
