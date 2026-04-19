"""Three-way model comparison: A+B vs refined vs cascade.

Evaluates mean PSNR and temporal flicker on a validation clip sequence.
All models are run on the same patches in the same order for a fair comparison.

Usage:
    uv run python compare_models.py \\
        --val-clean /path/to/val_clean \\
        --val-noisy /path/to/val_noisy \\
        --ab-weights      checkpoints/temporal_exp048_stage3/best.pth \\
        --refined-weights checkpoints/refiner_exp048_stage5/best.pth \\
        --cascade-weights checkpoints/cascade_exp048_stage3/best.pth \\
        --num-frames 3 \\
        --num-clips 30

Flicker note:
    Flicker is the mean absolute difference between consecutive output frames.
    Lower = more temporally consistent.  Meaningful only when --num-clips >= 10
    and the clips come from consecutive frames in the same sequence.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from dataset import PairedVideoSequenceDataset
from models import build_model_from_metadata


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _psnr(pred: Tensor, target: Tensor) -> float:
    mse = torch.mean((pred.float() - target.float()) ** 2).item()
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def _flicker(frames: list[Tensor]) -> float:
    """Mean absolute frame-to-frame difference."""
    if len(frames) < 2:
        return 0.0
    diffs = [
        torch.mean(torch.abs(frames[i].float() - frames[i - 1].float())).item()
        for i in range(1, len(frames))
    ]
    return sum(diffs) / len(diffs)


# ---------------------------------------------------------------------------
# Color-space helpers (mirror training.py)
# ---------------------------------------------------------------------------


def _apply_log(x: Tensor) -> Tensor:
    out = x.clone()
    if out.ndim == 4:
        out[:, :3] = torch.log1p(out[:, :3].clamp_min(0.0))
    elif out.ndim == 5:
        out[:, :, :3] = torch.log1p(out[:, :, :3].clamp_min(0.0))
    return out


def _invert_log(x: Tensor) -> Tensor:
    out = x.clone()
    if out.ndim == 4:
        out[:, :3] = torch.expm1(out[:, :3]).clamp_min(0.0)
    return out


def _color_space_from_ckpt(path: Path) -> str:
    """Read the color_space that was used during training."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    cfg = ckpt.get("training_config", {})
    return cfg.get("color_space", "linear")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(path: Path, device: torch.device) -> tuple[nn.Module, str]:
    """Rebuild model from checkpoint metadata; return (model, color_space)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    meta = ckpt.get("model_metadata") or ckpt.get("metadata")
    if meta is None:
        sys.exit(f"ERROR: {path} has no model_metadata — cannot rebuild.")
    model = build_model_from_metadata(meta)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    color_space = (ckpt.get("training_config") or {}).get("color_space", "linear")
    return model.to(device).eval(), color_space


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    color_space: str,
    num_clips: int,
) -> dict:
    psnrs: list[float] = []
    outputs: list[Tensor] = []

    ref_idx: int = model._ref_idx  # type: ignore[attr-defined]

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= num_clips:
                break
            noisy, clean, _ = batch          # (B, T, C, H, W), (B, T, C, H, W), sigma
            noisy = noisy.to(device)
            clean = clean.to(device)

            clean_ref_linear = clean[:, ref_idx]   # (B, C, H, W) — for PSNR

            if color_space == "log":
                noisy = _apply_log(noisy)

            output = model(noisy)            # (B, 3, H, W) — in model space

            if color_space == "log":
                output = _invert_log(output)

            psnrs.append(_psnr(output.cpu(), clean_ref_linear.cpu()))
            outputs.append(output.cpu())

    return {
        "mean_psnr": sum(psnrs) / max(1, len(psnrs)),
        "flicker": _flicker(outputs),
        "n_clips": len(psnrs),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare A+B, refined, and cascade denoiser variants.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    p.add_argument("--val-clean", required=True, nargs="+", metavar="DIR")
    p.add_argument("--val-noisy", required=True, nargs="+", metavar="DIR")
    p.add_argument("--ab-weights",       default=None, metavar="PATH",
                   help="Path to trained NAFNetTemporal (A+B) checkpoint.")
    p.add_argument("--refined-weights",  default=None, metavar="PATH",
                   help="Path to trained NAFNetRefinedTemporal checkpoint.")
    p.add_argument("--cascade-weights",  default=None, metavar="PATH",
                   help="Path to trained NAFNetCascade checkpoint.")
    p.add_argument("--num-frames", type=int, default=3,
                   help="Temporal window size (must match checkpoints, default: 3).")
    p.add_argument("--patch-size", type=int, default=256,
                   help="Spatial crop size for evaluation (default: 256).")
    p.add_argument("--num-clips", type=int, default=30,
                   help="Number of clips to evaluate (default: 30).")
    p.add_argument("--windows-per-sequence", type=int, default=1, metavar="N",
                   help="Deterministic windows per sequence (default: 1).")
    p.add_argument("--workers", type=int, default=4)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not any([args.ab_weights, args.refined_weights, args.cascade_weights]):
        parser.error("Provide at least one of --ab-weights, --refined-weights, --cascade-weights.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    val_ds = PairedVideoSequenceDataset(
        args.val_clean,
        args.val_noisy,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        patches_per_clip=1,
        random_windows=False,
        windows_per_sequence=args.windows_per_sequence,
        augment=False,
        crop_mode="center",
    )
    loader = DataLoader(val_ds, batch_size=1, num_workers=args.workers)
    print(f"Val dataset: {len(val_ds)} clips")

    candidates: list[tuple[str, Optional[str]]] = [
        ("ab",      args.ab_weights),
        ("refined", args.refined_weights),
        ("cascade", args.cascade_weights),
    ]

    results: dict[str, dict] = {}
    for name, path in candidates:
        if path is None:
            continue
        print(f"\n=== {name.upper()}: {path} ===")
        model, color_space = _load_model(Path(path), device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  params: {n_params:,}  color_space: {color_space}")
        r = _evaluate(model, loader, device, color_space, args.num_clips)
        results[name] = r
        print(f"  mean PSNR: {r['mean_psnr']:.3f} dB  ({r['n_clips']} clips)")
        print(f"  flicker:   {r['flicker']:.5f}")

    if not results:
        print("No models evaluated.")
        return

    col_w = 12
    print("\n" + "=" * 46)
    print(f"{'model':<{col_w}}  {'PSNR (dB)':>10}  {'flicker':>10}  {'clips':>6}")
    print("-" * 46)
    for name, r in results.items():
        print(
            f"{name:<{col_w}}  {r['mean_psnr']:>10.3f}  {r['flicker']:>10.5f}  {r['n_clips']:>6}"
        )
    print("=" * 46)

    if len(results) >= 2:
        names = list(results)
        baseline = results[names[0]]["mean_psnr"]
        for name in names[1:]:
            delta = results[name]["mean_psnr"] - baseline
            print(f"  {name} vs {names[0]}: Δ PSNR = {delta:+.3f} dB")


if __name__ == "__main__":
    main()
