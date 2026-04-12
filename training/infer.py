"""Python inference and quality evaluation.

Runs a trained model on a set of images and reports PSNR / SSIM.
Can also be used to generate output images for visual inspection.

Usage:
    # Evaluate on noisy/clean pairs (PSNR/SSIM):
    uv run python infer.py \\
        --checkpoint checkpoints/residual_standard/best.pth \\
        --noisy /path/to/noisy \\
        --clean /path/to/clean \\
        --sigma 25

    # Denoise a directory of images (no ground truth):
    uv run python infer.py \\
        --checkpoint checkpoints/residual_standard/best.pth \\
        --input /path/to/images \\
        --output /path/to/output
"""

import argparse
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from dataset import _load_image as _load_dataset_image
from dataset import _pad_frame_to_shape
from models import ModelConfig, NEFResidual, NEFTemporal


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_psnr(output: np.ndarray, clean: np.ndarray) -> float:
    """Compute PSNR between two float32 images in [0, 1]."""
    mse = np.mean((output.astype(np.float64) - clean.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim(output: np.ndarray, clean: np.ndarray) -> float:
    """Compute mean SSIM (averaged over channels if multi-channel)."""
    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return float("nan")

    if output.ndim == 3:
        scores = [
            structural_similarity(output[..., c], clean[..., c], data_range=1.0)
            for c in range(output.shape[-1])
        ]
        return float(np.mean(scores))
    return float(structural_similarity(output, clean, data_range=1.0))


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------


def denoise_image(
    model: torch.nn.Module,
    noisy: np.ndarray,
    device: torch.device,
    tile_size: int = 0,
    use_amp: bool = True,
) -> np.ndarray:
    """Run the model on a single HWC float32 image in [0, 1].

    Args:
        model: Trained denoiser (eval mode expected).
        noisy: (H, W, C) float32 in [0, 1].
        device: Target CUDA/CPU device.
        tile_size: Tile-based inference size (0 = full image).
        use_amp: Use AMP for FP16 inference.

    Returns:
        Denoised image as (H, W, C) float32 in [0, 1].
    """
    h, w, c = noisy.shape
    x = torch.from_numpy(noisy.transpose(2, 0, 1)).unsqueeze(0).to(device)  # (1, C, H, W)

    model.eval()
    with torch.no_grad():
        if tile_size > 0:
            output = _tile_inference(model, x, tile_size, use_amp, device)
        else:
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                output = model(x)

    return output.squeeze(0).permute(1, 2, 0).cpu().float().numpy()


def _tile_inference(
    model: torch.nn.Module,
    x: Tensor,
    tile_size: int,
    use_amp: bool,
    device: torch.device,
) -> Tensor:
    """Run inference on overlapping tiles and blend results."""
    _, c, h, w = x.shape
    overlap = tile_size // 8
    stride = tile_size - overlap

    output = torch.zeros_like(x)
    weights = torch.zeros(1, 1, h, w, device=device)

    for y in range(0, h, stride):
        for xi in range(0, w, stride):
            y1, y2 = y, min(y + tile_size, h)
            x1, x2 = xi, min(xi + tile_size, w)
            tile = x[:, :, y1:y2, x1:x2]

            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                pred = model(tile)

            # Linear ramp blend
            ramp = _make_ramp(pred.shape[-2], pred.shape[-1], overlap, device)
            output[:, :, y1:y2, x1:x2] += pred * ramp
            weights[:, :, y1:y2, x1:x2] += ramp

    return output / weights.clamp(min=1e-6)


def _make_ramp(h: int, w: int, overlap: int, device: torch.device) -> Tensor:
    """Create a blending weight ramp (1 in centre, fades at edges)."""
    ry = torch.ones(h, device=device)
    rx = torch.ones(w, device=device)
    if overlap > 0:
        ramp = torch.linspace(0.0, 1.0, overlap, device=device)
        ry[: len(ramp)] = ramp
        ry[-len(ramp) :] = ramp.flip(0)
        rx[: len(ramp)] = ramp
        rx[-len(ramp) :] = ramp.flip(0)
    return (ry.unsqueeze(1) * rx.unsqueeze(0)).unsqueeze(0).unsqueeze(0)


def _clip_indices(length: int, centre_idx: int, num_frames: int) -> list[int]:
    """Return a centred temporal window using edge replication at boundaries."""
    radius = num_frames // 2
    return [min(max(centre_idx + offset, 0), length - 1) for offset in range(-radius, radius + 1)]


def denoise_temporal_frame(
    model: torch.nn.Module,
    sequence: list[np.ndarray],
    frame_idx: int,
    device: torch.device,
    use_amp: bool = True,
) -> np.ndarray:
    """Run a temporal model on the window centred at ``frame_idx``."""
    num_frames = model._num_frames
    frame_indices = _clip_indices(len(sequence), frame_idx, num_frames)
    frames = [sequence[i] for i in frame_indices]

    target_h = max(frame.shape[0] for frame in frames)
    target_w = max(frame.shape[1] for frame in frames)
    padded_frames = [_pad_frame_to_shape(frame, target_h, target_w) for frame in frames]
    clip = np.stack([frame.transpose(2, 0, 1) for frame in padded_frames], axis=0)  # (T, C, H, W)
    x = torch.from_numpy(clip).unsqueeze(0).to(device)  # (1, T, C, H, W)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            output = model(x)

    ref_h, ref_w = sequence[frame_idx].shape[:2]
    return output.squeeze(0).permute(1, 2, 0).cpu().float().numpy()[:ref_h, :ref_w]


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------


def _load_image(path: Path) -> np.ndarray:
    return _load_dataset_image(path)


def _save_image(path: Path, img: np.ndarray) -> None:
    import imageio.v3 as iio

    path.parent.mkdir(parents=True, exist_ok=True)
    out = (img * 255.0).clip(0, 255).astype(np.uint8)
    iio.imwrite(str(path), out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise images and evaluate quality.")
    parser.add_argument("--checkpoint", required=True, metavar="PATH")
    parser.add_argument("--model", choices=["residual", "temporal"], default="residual")
    parser.add_argument("--size", choices=["lite", "standard", "heavy"], default="standard")
    parser.add_argument("--input", default=None, metavar="DIR",
                        help="Input directory of noisy images (no ground truth).")
    parser.add_argument("--noisy", default=None, metavar="DIR",
                        help="Directory of noisy images (with ground truth).")
    parser.add_argument("--clean", default=None, metavar="DIR",
                        help="Directory of clean ground-truth images.")
    parser.add_argument("--output", default=None, metavar="DIR",
                        help="Directory to save denoised images.")
    parser.add_argument("--sigma", type=float, default=0.0,
                        help="Add synthetic AWGN of this sigma (normalised) to input.")
    parser.add_argument("--tile", type=int, default=0, metavar="SIZE")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    cfg_map = {"lite": ModelConfig.lite, "standard": ModelConfig.standard, "heavy": ModelConfig.heavy}
    config = cfg_map[args.size]()
    model_cls = NEFTemporal if args.model == "temporal" else NEFResidual
    model = model_cls(config)

    ckpt = torch.load(args.checkpoint, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded {args.model}/{args.size} from {args.checkpoint}")

    # Determine image pairs
    _IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".exr"}
    if args.noisy and args.clean:
        noisy_dir = Path(args.noisy)
        clean_dir = Path(args.clean)
        pairs = [
            (p, clean_dir / p.name)
            for p in sorted(noisy_dir.iterdir())
            if p.suffix.lower() in _IMAGE_EXT
        ]
    elif args.input:
        pairs = [
            (p, None)
            for p in sorted(Path(args.input).iterdir())
            if p.suffix.lower() in _IMAGE_EXT
        ]
    else:
        parser.error("Specify either --input or both --noisy and --clean.")

    use_amp = not args.no_amp
    psnr_values, ssim_values = [], []

    if args.model == "temporal":
        if args.tile > 0:
            parser.error("--tile is not supported for temporal inference.")
        noisy_paths = [noisy_path for noisy_path, _ in pairs]
        noisy_sequence = [_load_image(path) for path in noisy_paths]
        if args.sigma > 0:
            noisy_sequence = [
                (img + np.random.randn(*img.shape) * args.sigma).clip(0, 1)
                for img in noisy_sequence
            ]

        for frame_idx, (noisy_path, clean_path) in enumerate(pairs):
            output_img = denoise_temporal_frame(model, noisy_sequence, frame_idx, device, use_amp)

            if clean_path is not None and clean_path.exists():
                clean_img = _load_image(clean_path)
                p = compute_psnr(output_img, clean_img)
                s = compute_ssim(output_img, clean_img)
                psnr_values.append(p)
                ssim_values.append(s)
                print(f"{noisy_path.name}: PSNR={p:.2f}dB  SSIM={s:.4f}")
            else:
                print(f"{noisy_path.name}: denoised")

            if args.output:
                _save_image(Path(args.output) / noisy_path.name, output_img)
    else:
        for noisy_path, clean_path in pairs:
            noisy_img = _load_image(noisy_path)
            if args.sigma > 0:
                noisy_img = (noisy_img + np.random.randn(*noisy_img.shape) * args.sigma).clip(0, 1)

            output_img = denoise_image(model, noisy_img, device, args.tile, use_amp)

            if clean_path is not None and clean_path.exists():
                clean_img = _load_image(clean_path)
                p = compute_psnr(output_img, clean_img)
                s = compute_ssim(output_img, clean_img)
                psnr_values.append(p)
                ssim_values.append(s)
                print(f"{noisy_path.name}: PSNR={p:.2f}dB  SSIM={s:.4f}")
            else:
                print(f"{noisy_path.name}: denoised")

            if args.output:
                _save_image(Path(args.output) / noisy_path.name, output_img)

    if psnr_values:
        print(f"\nMean PSNR: {np.mean(psnr_values):.2f} dB")
        print(f"Mean SSIM: {np.mean(ssim_values):.4f}")


if __name__ == "__main__":
    main()
