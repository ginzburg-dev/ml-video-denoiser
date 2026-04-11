"""Training loop for NEFResidual and NEFTemporal.

Supports:
  - Mixed-precision training via torch.cuda.amp
  - Cosine annealing LR schedule with linear warmup
  - Checkpoint save/resume
  - TensorBoard logging

Usage:
    uv run python training.py \\
        --model residual \\
        --data /path/to/clean/images \\
        --output checkpoints/residual_standard \\
        --epochs 300 \\
        --batch-size 16
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

from dataset import PatchDataset, VideoSequenceDataset
from losses import NoiseWeightedL1Loss
from models import ModelConfig, NEFResidual, NEFTemporal
from noise_generators import MixedNoiseGenerator


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
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = warmup_cosine_schedule(optimizer, warmup_epochs, epochs)
    criterion = NoiseWeightedL1Loss(epsilon=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    writer = SummaryWriter(log_dir=str(output_dir / "runs"))

    start_epoch = 0
    best_psnr = 0.0

    if resume is not None and resume.exists():
        start_epoch, best_psnr = _load_checkpoint(resume, model, optimizer, scheduler, device)
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
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
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
    writer.close()
    print(f"Training complete.  Best val PSNR: {best_psnr:.2f} dB")


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
            with torch.cuda.amp.autocast(enabled=use_amp and device.type == "cuda"):
                output = model(noisy)
            total_psnr += psnr(output, clean_ref)
    return total_psnr / max(1, len(loader))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train NEFResidual or NEFTemporal.")
    parser.add_argument("--model", choices=["residual", "temporal"], default="residual")
    parser.add_argument("--size", choices=["lite", "standard", "heavy"], default="standard")
    parser.add_argument("--data", required=True, nargs="+", metavar="DIR",
                        help="Directory(ies) of clean training images.")
    parser.add_argument("--val-data", nargs="+", default=None, metavar="DIR")
    parser.add_argument("--output", default="checkpoints/run", metavar="DIR")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--patches-per-image", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP.")
    parser.add_argument("--resume", default=None, metavar="PATH")
    # Noise options
    parser.add_argument("--patch-pool", default=None, metavar="PATH",
                        help="Path to real noise patch pool (.npz).")
    parser.add_argument("--noise-profile", default=None, metavar="PATH",
                        help="Path to calibrated noise profile (.json).")
    args = parser.parse_args()

    cfg_map = {"lite": ModelConfig.lite, "standard": ModelConfig.standard, "heavy": ModelConfig.heavy}
    config = cfg_map[args.size]()

    noise_gen = MixedNoiseGenerator.default(
        patch_pool=args.patch_pool, profile_json=args.noise_profile
    )

    if args.model == "temporal":
        train_ds = VideoSequenceDataset(
            args.data, noise_generator=noise_gen,
            num_frames=config.num_frames, patch_size=64,
        )
        val_ds = (
            VideoSequenceDataset(args.val_data, num_frames=config.num_frames, patch_size=64)
            if args.val_data else None
        )
        model = NEFTemporal(config)
    else:
        train_ds = PatchDataset(
            args.data, noise_generator=noise_gen,
            patch_size=args.patch_size, patches_per_image=args.patches_per_image,
        )
        val_ds = (
            PatchDataset(args.val_data, patch_size=args.patch_size)
            if args.val_data else None
        )
        model = NEFResidual(config)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, num_workers=args.workers)
        if val_ds else None
    )

    train(
        model=model,
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
