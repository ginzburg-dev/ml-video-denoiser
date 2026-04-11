"""Training loop for NEFResidual and NEFTemporal.

Supports:
  - Mixed-precision training via torch.amp
  - Cosine annealing LR schedule with linear warmup
  - Checkpoint save/resume
  - TensorBoard logging
  - Paired clean/noisy datasets mixed with synthetic noise generation

Usage — synthetic noise only:
    uv run python training.py \\
        --model residual \\
        --data /path/to/clean/images \\
        --output checkpoints/residual_standard \\
        --epochs 300

Usage — paired data only:
    uv run python training.py \\
        --model residual \\
        --paired-clean /path/to/clean \\
        --paired-noisy /path/to/noisy \\
        --output checkpoints/residual_paired

Usage — mixed (60% synthetic, 40% paired):
    uv run python training.py \\
        --model residual \\
        --data /path/to/clean/images \\
        --paired-clean /path/to/clean \\
        --paired-noisy /path/to/noisy \\
        --paired-weight 0.4 \\
        --output checkpoints/residual_mixed
"""

import argparse
import math
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import (
    CombinedDataset,
    PairedPatchDataset,
    PairedVideoSequenceDataset,
    PatchDataset,
    VideoSequenceDataset,
)
from losses import NoiseWeightedL1Loss
from models import ModelConfig, NEFResidual, NEFTemporal
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
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_psnr": best_psnr,
        },
        path,
    )


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
        model: NEFResidual or NEFTemporal instance.
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

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
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
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
        description="Train NEFResidual or NEFTemporal.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument("--model", choices=["residual", "temporal"], default="residual")
    parser.add_argument("--size", choices=["lite", "standard", "heavy"], default="standard")
    # Synthetic (clean-only) data
    parser.add_argument("--data", nargs="+", default=None, metavar="DIR",
                        help="Directory(ies) of clean images for synthetic noise training.")
    parser.add_argument("--val-data", nargs="+", default=None, metavar="DIR")
    # Paired (real clean/noisy) data
    parser.add_argument("--paired-clean", nargs="+", default=None, metavar="DIR",
                        help="Directory(ies) of clean ground-truth images for paired training.")
    parser.add_argument("--paired-noisy", nargs="+", default=None, metavar="DIR",
                        help="Matching directory(ies) of real noisy images.  "
                             "Must have same number of entries as --paired-clean.")
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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    has_synthetic = args.data is not None
    has_paired    = args.paired_clean is not None and args.paired_noisy is not None

    if not has_synthetic and not has_paired:
        parser.error("Provide --data for synthetic training, --paired-clean/--paired-noisy "
                     "for paired training, or both for mixed training.")

    cfg_map = {"lite": ModelConfig.lite, "standard": ModelConfig.standard, "heavy": ModelConfig.heavy}
    config = cfg_map[args.size]()
    match_by_name = not args.no_name_match

    noise_gen = _make_noise_generator(args, parser)

    is_temporal = args.model == "temporal"

    # ------------------------------------------------------------------
    # Build training dataset
    # ------------------------------------------------------------------
    def _make_synthetic_ds(dirs: list[str]) -> Dataset:
        if is_temporal:
            return VideoSequenceDataset(
                dirs, noise_generator=noise_gen,
                num_frames=config.num_frames, patch_size=64,
            )
        return PatchDataset(
            dirs, noise_generator=noise_gen,
            patch_size=args.patch_size, patches_per_image=args.patches_per_image,
        )

    def _make_paired_ds(clean_dirs: list[str], noisy_dirs: list[str]) -> Dataset:
        if len(clean_dirs) != len(noisy_dirs):
            parser.error("--paired-clean and --paired-noisy must have the same number of entries.")
        if is_temporal:
            return PairedVideoSequenceDataset(
                clean_dirs, noisy_dirs,
                num_frames=config.num_frames, patch_size=64,
            )
        # For spatial: pair each clean/noisy dir entry
        if len(clean_dirs) == 1:
            return PairedPatchDataset(
                clean_dirs[0], noisy_dirs[0],
                patch_size=args.patch_size,
                patches_per_image=args.patches_per_image,
                match_by_name=match_by_name,
            )
        # Multiple paired dirs → combine
        sub_datasets = [
            PairedPatchDataset(
                c, n,
                patch_size=args.patch_size,
                patches_per_image=args.patches_per_image,
                match_by_name=match_by_name,
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
    # Build validation dataset (synthetic only — ground truth is clean)
    # ------------------------------------------------------------------
    val_ds: Optional[Dataset] = None
    if args.val_data:
        val_ds = _make_synthetic_ds(args.val_data)

    model_instance = NEFTemporal(config) if is_temporal else NEFResidual(config)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)
        if val_ds else None
    )

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
