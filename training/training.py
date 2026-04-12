"""Training loop for NAFNet denoisers.

Supports:
  - Mixed-precision training via torch.amp
  - Cosine annealing LR schedule with linear warmup
  - Checkpoint save/resume
  - TensorBoard logging
  - Paired clean/noisy datasets mixed with synthetic noise generation
  - Two-stage temporal training: warm-start from a spatial checkpoint,
    optionally freeze spatial layers to train temporal components first

Stage 1 — train spatial model (NAFNet):
    uv run python training.py \\
        --model spatial \\
        --data /path/to/clean/images \\
        --output checkpoints/spatial \\
        --epochs 300

Stage 2 — load spatial weights, train temporal components only:
    uv run python training.py \\
        --model temporal \\
        --data /path/to/sequences \\
        --spatial-weights checkpoints/spatial/best.pth \\
        --freeze-spatial \\
        --output checkpoints/temporal_stage2 \\
        --epochs 100

Stage 3 — unfreeze all, fine-tune jointly at lower lr:
    uv run python training.py \\
        --model temporal \\
        --data /path/to/sequences \\
        --spatial-weights checkpoints/spatial/best.pth \\
        --resume checkpoints/temporal_stage2/best.pth \\
        --output checkpoints/temporal_stage3 \\
        --lr 5e-5 \\
        --epochs 100

Usage — synthetic noise only:
    uv run python training.py \\
        --model spatial \\
        --data /path/to/clean/images \\
        --output checkpoints/spatial \\
        --epochs 300

Usage — paired data only:
    uv run python training.py \\
        --model spatial \\
        --paired-clean /path/to/clean \\
        --paired-noisy /path/to/noisy \\
        --output checkpoints/spatial_paired

Usage — mixed (60% synthetic, 40% paired):
    uv run python training.py \\
        --model spatial \\
        --data /path/to/clean/images \\
        --paired-clean /path/to/clean \\
        --paired-noisy /path/to/noisy \\
        --paired-weight 0.4 \\
        --output checkpoints/spatial_mixed
"""

import argparse
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

from dataset import (
    CombinedDataset,
    PairedPatchDataset,
    PairedVideoSequenceDataset,
    PatchDataset,
    VideoSequenceDataset,
)
from losses import NoiseWeightedL1Loss
from models import NAFNet, NAFNetConfig, NAFNetTemporal, freeze_spatial, load_spatial_weights
from noise_generators import (
    GaussianNoiseGenerator,
    MixedNoiseGenerator,
    PoissonGaussianNoiseGenerator,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def psnr(output: Tensor, clean: Tensor) -> float:
    """Compute PSNR (dB) between *output* and *clean* (both in [0, 1])."""
    mse = torch.mean((output.float() - clean.float()) ** 2).item()
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------


def warmup_cosine_schedule(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    eta_min_ratio: float = 1e-2,
) -> LambdaLR:
    """Linear warmup followed by cosine annealing.

    Args:
        optimizer: The optimiser to schedule.
        warmup_epochs: Number of epochs for the linear warmup phase.
        total_epochs: Total training epochs.
        eta_min_ratio: Minimum LR as a fraction of the initial LR.

    Returns:
        A LambdaLR scheduler.
    """

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    epoch: int,
    best_psnr: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_psnr": best_psnr,
    }

    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_name = tmp_file.name
        torch.save(payload, tmp_name)
        os.replace(tmp_name, path)
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR,
    device: torch.device,
) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["best_psnr"]


def _module_device(module: nn.Module) -> torch.device:
    """Return the device of the first parameter/buffer, defaulting to CPU."""
    tensor = next(module.parameters(), None)
    if tensor is None:
        tensor = next(module.buffers(), None)
    return tensor.device if tensor is not None else torch.device("cpu")


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(
    model: nn.Module,
    loader: DataLoader,
    val_loader: Optional[DataLoader],
    output_dir: Path,
    epochs: int = 300,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 5,
    grad_clip: float = 1.0,
    checkpoint_every: int = 50,
    device: Optional[torch.device] = None,
    resume: Optional[Path] = None,
    use_amp: bool = True,
) -> None:
    """Run the training loop.

    Args:
        model: Spatial or temporal denoiser instance.
        loader: Training DataLoader.
        val_loader: Optional validation DataLoader for PSNR tracking.
        output_dir: Directory to save checkpoints and TensorBoard logs.
        epochs: Total training epochs.
        lr: Initial learning rate for AdamW.
        weight_decay: L2 regularisation coefficient.
        warmup_epochs: Linear warmup duration.
        grad_clip: Maximum gradient norm (0 = disabled).
        checkpoint_every: Save a checkpoint every N epochs.
        device: Target device.  Defaults to CUDA if available, else CPU.
        resume: Path to a checkpoint to resume from.
        use_amp: Whether to use Automatic Mixed Precision training.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_device = _module_device(model)
    model = model.to(device)

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=weight_decay,
    )
    scheduler = warmup_cosine_schedule(optimizer, warmup_epochs, epochs)
    criterion = NoiseWeightedL1Loss(epsilon=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and device.type == "cuda")
    writer = SummaryWriter(log_dir=str(output_dir / "runs"))

    try:
        start_epoch = 0
        best_psnr = 0.0

        if resume is not None and resume.exists():
            start_epoch, best_psnr = _load_checkpoint(
                resume, model, optimizer, scheduler, device
            )
            print(f"Resumed from {resume} at epoch {start_epoch}.")

        output_dir.mkdir(parents=True, exist_ok=True)
        is_temporal = hasattr(model, "_num_frames")

        for epoch in range(start_epoch, epochs):
            model.train()
            epoch_loss = 0.0
            epoch_psnr = 0.0
            t0 = time.perf_counter()

            for step, batch in enumerate(loader):
                if is_temporal:
                    noisy, clean, sigma_map = batch        # (B, T, C, H, W) each
                    noisy = noisy.to(device)
                    clean = clean.to(device)
                    sigma_map = sigma_map.to(device)
                    ref_idx = model._ref_idx
                    clean_ref = clean[:, ref_idx]          # (B, C, H, W) — reference frame
                    sigma_ref = sigma_map[:, ref_idx]
                else:
                    noisy, clean, sigma_map = batch        # (B, C, H, W) each
                    noisy = noisy.to(device)
                    clean = clean.to(device)
                    sigma_map = sigma_map.to(device)
                    clean_ref = clean
                    sigma_ref = sigma_map

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    output = model(noisy)
                    loss = criterion(output, clean_ref, sigma_ref)

                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad], grad_clip
                    )
                scaler.step(optimizer)
                scaler.update()

                epoch_loss += loss.item()
                with torch.no_grad():
                    epoch_psnr += psnr(output, clean_ref)

            scheduler.step()
            n_steps = len(loader)
            avg_loss = epoch_loss / n_steps
            avg_psnr = epoch_psnr / n_steps
            elapsed = time.perf_counter() - t0
            current_lr = scheduler.get_last_lr()[0]

            writer.add_scalar("train/loss", avg_loss, epoch)
            writer.add_scalar("train/psnr", avg_psnr, epoch)
            writer.add_scalar("train/lr", current_lr, epoch)

            print(
                f"Epoch {epoch + 1:4d}/{epochs}  "
                f"loss={avg_loss:.6f}  psnr={avg_psnr:.2f}dB  "
                f"lr={current_lr:.2e}  t={elapsed:.1f}s"
            )

            # Validation
            if val_loader is not None:
                val_psnr = _validate(model, val_loader, device, is_temporal, use_amp)
                writer.add_scalar("val/psnr", val_psnr, epoch)
                print(f"  val psnr={val_psnr:.2f}dB")
                if val_psnr > best_psnr:
                    best_psnr = val_psnr
                    _save_checkpoint(
                        output_dir / "best.pth", model, optimizer, scheduler, epoch, best_psnr
                    )

            # Periodic checkpoint
            if (epoch + 1) % checkpoint_every == 0:
                _save_checkpoint(
                    output_dir / f"epoch_{epoch + 1:04d}.pth",
                    model, optimizer, scheduler, epoch, best_psnr,
                )

        # Final checkpoint
        _save_checkpoint(
            output_dir / "final.pth", model, optimizer, scheduler, epochs - 1, best_psnr
        )
        print(f"Training complete.  Best val PSNR: {best_psnr:.2f} dB")
    finally:
        writer.close()
        if _module_device(model) != original_device:
            model.to(original_device)


def _validate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    is_temporal: bool,
    use_amp: bool,
) -> float:
    model.eval()
    total_psnr = 0.0
    with torch.no_grad():
        for batch in loader:
            if is_temporal:
                noisy, clean, _ = batch
                ref_idx = model._ref_idx
                clean_ref = clean[:, ref_idx].to(device)
            else:
                noisy, clean, _ = batch
                clean_ref = clean.to(device)
            noisy = noisy.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                output = model(noisy)
            total_psnr += psnr(output, clean_ref)
    return total_psnr / max(1, len(loader))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI parser.

    Long-option abbreviation is disabled so flags like ``--noise`` cannot be
    silently interpreted as ``--noise-profile``.
    """
    parser = argparse.ArgumentParser(
        description="Train NAFNet denoisers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--model", choices=["spatial", "temporal"], default="spatial")
    parser.add_argument("--naf-base", type=int, default=32, metavar="C",
                        help="Base channel count for NAFNet residual/temporal models (default: 32).  "
                             "Overrides the selected preset base_channels when both are given.")
    parser.add_argument("--size", choices=["tiny", "small", "standard", "wide"], default=None,
                        help="NAFNet size preset: tiny (~0.4M), small (~7M), standard (~24M), "
                             "wide (~67M).  When set, uses the preset block counts and "
                             "base_channels; --naf-base overrides the base_channels only.")
    parser.add_argument("--num-frames", type=int, default=5, metavar="T",
                        help="Temporal window size for --model temporal (default: 5).")
    parser.add_argument("--use-warp", action="store_true",
                        help="Enable per-level learned warp in the temporal model.  "
                             "Recommended for real video with motion; leave disabled for render sequences.")
    parser.add_argument("--frames-per-sequence", type=int, default=None, metavar="N",
                        help="For spatial training on sequence folder structures: select N evenly "
                             "spread frames from each sequence subdirectory (first, spread, last). "
                             "Requires --model spatial. Flat directories fall back to all images "
                             "with a warning.")
    parser.add_argument("--val-frames-per-sequence", type=int, default=None, metavar="N",
                        help="Same as --frames-per-sequence but applied to the validation dataset. "
                             "Requires --model spatial and validation data.")
    parser.add_argument("--spatial-weights", default=None, metavar="PATH",
                        help="Load a matching spatial checkpoint into the temporal model before training. "
                             "Requires --model temporal.")
    parser.add_argument("--freeze-spatial", action="store_true",
                        help="Freeze spatial layers (encoders, bottleneck, decoders, head) — "
                             "only temporal_mix and offset_heads receive gradients. "
                             "Requires --model temporal.")
    # Synthetic (clean-only) data
    parser.add_argument("--data", nargs="+", default=None, metavar="DIR",
                        help="Directory(ies) of clean images for synthetic noise training.")
    parser.add_argument("--val-data", nargs="+", default=None, metavar="DIR",
                        help="Directory(ies) of clean validation data for synthetic validation.")
    # Paired (real clean/noisy) data
    parser.add_argument("--paired-clean", nargs="+", default=None, metavar="DIR",
                        help="Directory(ies) of clean ground-truth images for paired training.")
    parser.add_argument("--paired-noisy", nargs="+", default=None, metavar="DIR",
                        help="Matching directory(ies) of real noisy images.  "
                             "Must have same number of entries as --paired-clean.")
    parser.add_argument("--val-clean", nargs="+", default=None, metavar="DIR",
                        help="Directory(ies) of clean ground-truth validation images.")
    parser.add_argument("--val-noisy", nargs="+", default=None, metavar="DIR",
                        help="Matching directory(ies) of real noisy validation images.  "
                             "Must have same number of entries as --val-clean.")
    parser.add_argument("--paired-weight", type=float, default=0.5,
                        help="Fraction of each batch drawn from paired data when both "
                             "--data and --paired-clean are supplied (default: 0.5).")
    parser.add_argument("--no-name-match", action="store_true",
                        help="Match paired images by sorted position rather than filename stem.")
    # Common training args
    parser.add_argument("--output", default="checkpoints/run", metavar="DIR")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-image", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP.")
    parser.add_argument("--resume", default=None, metavar="PATH")
    parser.add_argument(
        "--val-windows-per-sequence",
        type=int,
        default=None,
        metavar="N",
        help="For temporal validation, keep N evenly spaced windows per sequence.",
    )
    parser.add_argument(
        "--val-crop-mode",
        choices=["center", "random", "full", "grid"],
        default="random",
        help="Validation crop mode. Use 'center' for deterministic crops or "
             "'full' to validate full-resolution frames.",
    )
    parser.add_argument(
        "--val-grid-size",
        type=int,
        default=2,
        metavar="N",
        help="Validation grid side length when --val-crop-mode grid is used.",
    )
    parser.add_argument(
        "--random-temporal-windows",
        action="store_true",
        help="For temporal training, sample random windows per sequence each epoch "
             "instead of enumerating every sliding window.",
    )
    parser.add_argument(
        "--windows-per-sequence",
        type=int,
        default=None,
        metavar="N",
        help="For temporal training with random windows enabled, draw N temporal "
             "windows from each sequence per epoch.",
    )
    # Synthetic noise options
    parser.add_argument(
        "--noise",
        choices=["gaussian", "poisson-gaussian", "mixed"],
        default="mixed",
        help="Synthetic noise model to use for clean-only training data.",
    )
    parser.add_argument("--patch-pool", default=None, metavar="PATH",
                        help="Path to real noise patch pool (.npz).")
    parser.add_argument("--noise-profile", default=None, metavar="PATH",
                        help="Path to calibrated noise profile (.json).")
    return parser


def _make_noise_generator(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
):
    """Construct the configured synthetic noise generator."""
    if args.noise == "gaussian":
        if args.patch_pool or args.noise_profile:
            parser.error("--patch-pool and --noise-profile require --noise mixed.")
        return GaussianNoiseGenerator(0.0, 75.0 / 255.0)

    if args.noise == "poisson-gaussian":
        if args.patch_pool or args.noise_profile:
            parser.error("--patch-pool and --noise-profile require --noise mixed.")
        return PoissonGaussianNoiseGenerator()

    return MixedNoiseGenerator.default(
        patch_pool=args.patch_pool, profile_json=args.noise_profile
    )


def _validation_mode(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[Optional[str], Optional[tuple[list[str], ...]]]:
    """Validate validation flags and return the selected mode."""
    has_synthetic_val = args.val_data is not None
    has_paired_val = args.val_clean is not None or args.val_noisy is not None

    if args.val_clean is not None and args.val_noisy is None:
        parser.error("--val-clean requires --val-noisy.")
    if args.val_noisy is not None and args.val_clean is None:
        parser.error("--val-noisy requires --val-clean.")
    if has_synthetic_val and has_paired_val:
        parser.error("Use either --val-data or --val-clean/--val-noisy, not both.")

    if has_synthetic_val:
        return "synthetic", (args.val_data,)
    if args.val_clean is not None and args.val_noisy is not None:
        return "paired", (args.val_clean, args.val_noisy)
    return None, None


def _temporal_sampling_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[bool, Optional[int]]:
    """Validate and normalize temporal random-window sampling options."""
    random_windows = args.random_temporal_windows or args.windows_per_sequence is not None
    windows_per_sequence = args.windows_per_sequence

    if windows_per_sequence is not None and windows_per_sequence <= 0:
        parser.error("--windows-per-sequence must be a positive integer.")

    if random_windows and args.model != "temporal":
        parser.error("--random-temporal-windows and --windows-per-sequence require --model temporal.")

    if random_windows and windows_per_sequence is None:
        windows_per_sequence = 1

    return random_windows, windows_per_sequence


def _frames_per_sequence_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> tuple[Optional[int], Optional[int]]:
    """Validate and return (frames_per_sequence, val_frames_per_sequence)."""
    fps = args.frames_per_sequence
    val_fps = args.val_frames_per_sequence

    if fps is not None:
        if fps <= 0:
            parser.error("--frames-per-sequence must be a positive integer.")
        if args.model != "spatial":
            parser.error("--frames-per-sequence requires --model spatial.")

    if val_fps is not None:
        if val_fps <= 0:
            parser.error("--val-frames-per-sequence must be a positive integer.")
        if args.model != "spatial":
            parser.error("--val-frames-per-sequence requires --model spatial.")

    return fps, val_fps


def _validation_temporal_config(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    val_mode: Optional[str],
) -> tuple[Optional[int], str, int]:
    """Validate temporal validation sampling/crop options."""
    if args.val_windows_per_sequence is not None:
        if args.val_windows_per_sequence <= 0:
            parser.error("--val-windows-per-sequence must be a positive integer.")
        if args.model != "temporal":
            parser.error("--val-windows-per-sequence requires --model temporal.")
        if val_mode is None:
            parser.error("--val-windows-per-sequence requires validation data.")
    if args.val_grid_size <= 0:
        parser.error("--val-grid-size must be a positive integer.")
    return args.val_windows_per_sequence, args.val_crop_mode, args.val_grid_size


def _validation_patch_repeats(crop_mode: str, grid_size: int) -> int:
    """Return how many deterministic crops each validation item should use."""
    if crop_mode == "grid":
        return grid_size * grid_size
    return 1


def _config_summary_lines(
    *,
    is_temporal: bool,
    naf_base: Optional[int] = None,
    random_temporal_windows: bool,
    windows_per_sequence: Optional[int],
    frames_per_sequence: Optional[int] = None,
    val_frames_per_sequence: Optional[int] = None,
    val_mode: Optional[str] = None,
    val_windows_per_sequence: Optional[int] = None,
    val_crop_mode: str,
    val_grid_size: int,
) -> list[str]:
    """Return human-readable startup lines for sampling and validation config."""
    lines: list[str] = []

    label = f"base={naf_base}" if naf_base is not None else "preset"
    if is_temporal:
        lines.append(f"Architecture: NAFNet temporal ({label})")
    else:
        lines.append(f"Architecture: NAFNet spatial ({label})")

    if is_temporal:
        if random_temporal_windows:
            lines.append(
                f"Train temporal sampling: random windows ({windows_per_sequence}/sequence/epoch)"
            )
        else:
            lines.append("Train temporal sampling: all sliding windows")
    elif frames_per_sequence is not None:
        lines.append(f"Train spatial sampling: {frames_per_sequence} evenly spread frames/sequence")

    if not is_temporal and val_mode is not None and val_frames_per_sequence is not None:
        lines.append(f"Val spatial sampling: {val_frames_per_sequence} evenly spread frames/sequence")

    if val_mode is not None:
        if is_temporal:
            if val_windows_per_sequence is None:
                lines.append("Val temporal sampling: all sliding windows")
            else:
                lines.append(
                    f"Val temporal sampling: deterministic windows ({val_windows_per_sequence}/sequence)"
                )

        if val_crop_mode == "grid":
            lines.append(f"Val crop mode: grid ({val_grid_size}x{val_grid_size})")
        else:
            lines.append(f"Val crop mode: {val_crop_mode}")

    return lines


def _dataset_summary_lines(name: str, dataset: Dataset) -> list[str]:
    """Return human-readable dataset statistics for startup logging."""
    lines = [f"{name}: {len(dataset)} samples/epoch"]

    if hasattr(dataset, "num_sequences"):
        lines[0] += f" from {dataset.num_sequences} sequences"
        if hasattr(dataset, "num_clips"):
            lines[0] += f" ({dataset.num_clips} temporal windows)"
    elif hasattr(dataset, "num_images"):
        lines[0] += f" from {dataset.num_images} images"
    elif hasattr(dataset, "num_clips"):
        lines[0] += f" from {dataset.num_clips} clips"
    elif hasattr(dataset, "num_pairs"):
        lines[0] += f" from {dataset.num_pairs} pairs"
    elif hasattr(dataset, "num_datasets"):
        lines[0] += f" across {dataset.num_datasets} datasets"

    if isinstance(dataset, CombinedDataset):
        for index, sub_dataset in enumerate(dataset._datasets, start=1):
            weight = dataset._weights[index - 1]
            sub_lines = _dataset_summary_lines(
                f"{name} component {index} ({weight * 100:.0f}%)",
                sub_dataset,
            )
            lines.extend(f"  {line}" for line in sub_lines)

    return lines


def _log_loader_summary(
    name: str,
    dataset: Dataset,
    loader: DataLoader,
) -> None:
    """Print dataset and loader sizing information before training starts."""
    for line in _dataset_summary_lines(name, dataset):
        print(line)
    print(
        f"{name} loader: batch_size={loader.batch_size}  "
        f"steps/epoch={len(loader)}  workers={loader.num_workers}"
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    has_synthetic = args.data is not None
    has_paired    = args.paired_clean is not None and args.paired_noisy is not None

    if not has_synthetic and not has_paired:
        parser.error("Provide --data for synthetic training, --paired-clean/--paired-noisy "
                     "for paired training, or both for mixed training.")

    _naf_preset_map = {
        "tiny": NAFNetConfig.tiny,
        "small": NAFNetConfig.small,
        "standard": NAFNetConfig.standard,
        "wide": NAFNetConfig.wide,
    }
    naf_config = _naf_preset_map[args.size]() if args.size else NAFNetConfig(base_channels=args.naf_base)
    if args.size and args.naf_base != 32:
        naf_config.base_channels = args.naf_base
    match_by_name = not args.no_name_match
    val_mode, val_sources = _validation_mode(args, parser)
    random_temporal_windows, windows_per_sequence = _temporal_sampling_config(args, parser)
    frames_per_sequence, val_frames_per_sequence = _frames_per_sequence_config(args, parser)
    val_windows_per_sequence, val_crop_mode, val_grid_size = _validation_temporal_config(
        args, parser, val_mode
    )
    val_patch_repeats = _validation_patch_repeats(val_crop_mode, val_grid_size)

    noise_gen = _make_noise_generator(args, parser)

    is_temporal = args.model == "temporal"

    # ------------------------------------------------------------------
    # Build training dataset
    # ------------------------------------------------------------------
    def _make_synthetic_ds(dirs: list[str], *, for_validation: bool = False) -> Dataset:
        if is_temporal:
            return VideoSequenceDataset(
                dirs, noise_generator=noise_gen,
                num_frames=args.num_frames, patch_size=64,
                patches_per_clip=val_patch_repeats if for_validation else 16,
                random_windows=random_temporal_windows and not for_validation,
                windows_per_sequence=(
                    windows_per_sequence if not for_validation else val_windows_per_sequence
                ),
                augment=False if for_validation else True,
                crop_mode=val_crop_mode if for_validation else "random",
                crop_grid_size=val_grid_size,
            )
        return PatchDataset(
            dirs, noise_generator=noise_gen,
            patch_size=args.patch_size,
            patches_per_image=val_patch_repeats if for_validation else args.patches_per_image,
            augment=False if for_validation else True,
            crop_mode=val_crop_mode if for_validation else "random",
            crop_grid_size=val_grid_size,
            frames_per_sequence=val_frames_per_sequence if for_validation else frames_per_sequence,
        )

    def _make_paired_ds(
        clean_dirs: list[str],
        noisy_dirs: list[str],
        *,
        for_validation: bool = False,
    ) -> Dataset:
        if len(clean_dirs) != len(noisy_dirs):
            parser.error("--paired-clean and --paired-noisy must have the same number of entries.")
        if is_temporal:
            return PairedVideoSequenceDataset(
                clean_dirs, noisy_dirs,
                num_frames=args.num_frames, patch_size=64,
                patches_per_clip=val_patch_repeats if for_validation else 16,
                random_windows=random_temporal_windows and not for_validation,
                windows_per_sequence=(
                    windows_per_sequence if not for_validation else val_windows_per_sequence
                ),
                augment=False if for_validation else True,
                crop_mode=val_crop_mode if for_validation else "random",
                crop_grid_size=val_grid_size,
            )
        # For spatial: pair each clean/noisy dir entry
        if len(clean_dirs) == 1:
            return PairedPatchDataset(
                clean_dirs[0], noisy_dirs[0],
                patch_size=args.patch_size,
                patches_per_image=val_patch_repeats if for_validation else args.patches_per_image,
                match_by_name=match_by_name,
                augment=False if for_validation else True,
                crop_mode=val_crop_mode if for_validation else "random",
                crop_grid_size=val_grid_size,
                frames_per_sequence=val_frames_per_sequence if for_validation else frames_per_sequence,
            )
        # Multiple paired dirs → combine
        sub_datasets = [
            PairedPatchDataset(
                c, n,
                patch_size=args.patch_size,
                patches_per_image=val_patch_repeats if for_validation else args.patches_per_image,
                match_by_name=match_by_name,
                augment=False if for_validation else True,
                crop_mode=val_crop_mode if for_validation else "random",
                crop_grid_size=val_grid_size,
                frames_per_sequence=val_frames_per_sequence if for_validation else frames_per_sequence,
            )
            for c, n in zip(clean_dirs, noisy_dirs)
        ]
        return CombinedDataset(sub_datasets)

    if has_synthetic and has_paired:
        syn_ds    = _make_synthetic_ds(args.data)
        paired_ds = _make_paired_ds(args.paired_clean, args.paired_noisy)
        pw = max(0.0, min(1.0, args.paired_weight))
        train_ds: Dataset = CombinedDataset(
            datasets=[syn_ds, paired_ds],
            weights=[1.0 - pw, pw],
        )
        print(f"Mixed training: {(1-pw)*100:.0f}% synthetic + {pw*100:.0f}% paired")
    elif has_paired:
        train_ds = _make_paired_ds(args.paired_clean, args.paired_noisy)
        print("Paired training only (no synthetic noise generation)")
    else:
        train_ds = _make_synthetic_ds(args.data)
        print("Synthetic training only (no paired data)")

    # ------------------------------------------------------------------
    # Build validation dataset
    # ------------------------------------------------------------------
    val_ds: Optional[Dataset] = None
    if val_mode == "synthetic":
        (val_dirs,) = val_sources
        val_ds = _make_synthetic_ds(val_dirs, for_validation=True)
    elif val_mode == "paired":
        val_clean_dirs, val_noisy_dirs = val_sources
        val_ds = _make_paired_ds(val_clean_dirs, val_noisy_dirs, for_validation=True)

    if args.spatial_weights and not is_temporal:
        parser.error("--spatial-weights requires --model temporal.")
    if args.freeze_spatial and not is_temporal:
        parser.error("--freeze-spatial requires --model temporal.")

    if is_temporal:
        model_instance = NAFNetTemporal(naf_config, num_frames=args.num_frames, use_warp=args.use_warp)
    else:
        model_instance = NAFNet(naf_config)

    if args.spatial_weights:
        n = load_spatial_weights(model_instance, args.spatial_weights)
        print(f"Loaded {n} spatial weight tensors from {args.spatial_weights}")

    if args.freeze_spatial:
        freeze_spatial(model_instance)
        n_trainable = sum(p.numel() for p in model_instance.parameters() if p.requires_grad)
        n_frozen    = sum(p.numel() for p in model_instance.parameters() if not p.requires_grad)
        print(f"Spatial layers frozen.  Trainable params: {n_trainable:,}  Frozen: {n_frozen:,}")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_batch_size = 1 if val_ds is not None and val_crop_mode == "full" else args.batch_size
    val_loader = DataLoader(val_ds, batch_size=val_batch_size, num_workers=args.workers) if val_ds else None

    for line in _config_summary_lines(
        is_temporal=is_temporal,
        naf_base=args.naf_base,
        random_temporal_windows=random_temporal_windows,
        windows_per_sequence=windows_per_sequence,
        frames_per_sequence=frames_per_sequence,
        val_frames_per_sequence=val_frames_per_sequence,
        val_mode=val_mode,
        val_windows_per_sequence=val_windows_per_sequence,
        val_crop_mode=val_crop_mode,
        val_grid_size=val_grid_size,
    ):
        print(line)

    _log_loader_summary("Train", train_ds, train_loader)
    if val_ds is not None and val_loader is not None:
        _log_loader_summary("Val", val_ds, val_loader)

    train(
        model=model_instance,
        loader=train_loader,
        val_loader=val_loader,
        output_dir=Path(args.output),
        epochs=args.epochs,
        lr=args.lr,
        use_amp=not args.no_amp,
        resume=Path(args.resume) if args.resume else None,
    )


if __name__ == "__main__":
    main()
