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
from PIL import Image
from torch import Tensor

from dataset import _load_image as _load_dataset_image
from dataset import _pad_frame_to_shape
from models import NAFNet, NAFNetConfig, NAFNetTemporal


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_psnr(output: np.ndarray, clean: np.ndarray) -> float:
    """Compute PSNR between two float32 images. Data range inferred from clean."""
    mse = np.mean((output.astype(np.float64) - clean.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    data_range = float(clean.max() - clean.min()) or 1.0
    return 10.0 * math.log10(data_range**2 / mse)


def compute_ssim(output: np.ndarray, clean: np.ndarray) -> float:
    """Compute mean SSIM (averaged over channels if multi-channel)."""
    if np.allclose(output, clean):
        return 1.0

    try:
        from skimage.metrics import structural_similarity
    except ImportError:
        return float("nan")

    if output.ndim == 3:
        scores = [
            structural_similarity(output[..., c], clean[..., c], data_range=1.0)
            for c in range(output.shape[-1])
        ]
        scores = [
            score
            if np.isfinite(score)
            else (1.0 if np.allclose(output[..., idx], clean[..., idx]) else 0.0)
            for idx, score in enumerate(scores)
        ]
        mean_score = np.mean(scores)
        if np.isfinite(mean_score):
            return float(mean_score)
        return 0.0
    score = structural_similarity(output, clean, data_range=1.0)
    if np.isfinite(score):
        return float(score)
    return 1.0 if np.allclose(output, clean) else 0.0


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
    """Run the model on a single HWC float32 image.

    Args:
        model: Trained denoiser (eval mode expected).
        noisy: (H, W, C) float32.
        device: Target CUDA/CPU device.
        tile_size: Tile-based inference size (0 = full image).
        use_amp: Use AMP for FP16 inference.

    Returns:
        Denoised image as (H, W, C) float32, same range as input.
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


def _load_alpha(path: Path) -> Optional[np.ndarray]:
    if path.suffix.lower() == ".exr":
        import OpenEXR

        with OpenEXR.File(str(path)) as exr:
            channels = exr.parts[0].channels
            channel = channels.get("RGBA") or channels.get("A")
            if channel is None:
                return None
            alpha = np.asarray(channel.pixels, dtype=np.float32)
            if alpha.ndim == 3:
                alpha = alpha[..., -1]
            return np.clip(alpha, 0.0, 1.0)

    with Image.open(path) as pil_img:
        if "A" not in pil_img.getbands():
            return None
        alpha = np.asarray(pil_img.getchannel("A"), dtype=np.float32)
        if alpha.max() > 1.5:
            alpha /= 255.0 if alpha.max() <= 255.5 else 65535.0
        return np.clip(alpha, 0.0, 1.0)


def _read_exr_header(path: Path) -> dict:
    """Read EXR header attributes to preserve on output.

    Returns a dict containing dataWindow, displayWindow, pixelAspectRatio,
    screenWindowCenter, screenWindowWidth and any other attributes present.
    Returns an empty dict for non-EXR paths.
    """
    if path.suffix.lower() != ".exr":
        return {}
    import OpenEXR
    with OpenEXR.File(str(path)) as exr:
        return dict(exr.parts[0].header)


def _save_image(
    path: Path,
    img: np.ndarray,
    alpha: Optional[np.ndarray] = None,
    exr_header: Optional[dict] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = img.astype(np.float32)

    if path.suffix.lower() == ".exr":
        import OpenEXR

        out = img.astype(np.float32)
        if out.ndim == 2:
            out = np.repeat(out[..., None], 3, axis=2)
        elif out.shape[-1] == 1:
            out = np.repeat(out, 3, axis=2)
        elif out.shape[-1] > 3:
            out = out[..., :3]

        # Build header — start from input header to preserve dataWindow,
        # displayWindow, pixelAspectRatio, etc., then enforce scanlineimage type.
        header: dict = {}
        if exr_header:
            _PRESERVE = {
                "dataWindow", "displayWindow",
                "pixelAspectRatio", "screenWindowCenter", "screenWindowWidth",
            }
            header.update({k: v for k, v in exr_header.items() if k in _PRESERVE})
        header["type"] = OpenEXR.scanlineimage

        if alpha is not None:
            alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
            if alpha.shape != out.shape[:2]:
                raise ValueError("Alpha channel shape does not match RGB output shape.")
            rgba = np.concatenate([out, alpha[..., None]], axis=-1)
            OpenEXR.File(header, {"RGBA": rgba}).write(str(path))
        else:
            OpenEXR.File(header, {"RGB": out}).write(str(path))
        return

    import imageio.v3 as iio

    out = np.clip(img, 0.0, 1.0)
    if alpha is not None:
        alpha = np.clip(alpha.astype(np.float32), 0.0, 1.0)
        if alpha.shape != out.shape[:2]:
            raise ValueError("Alpha channel shape does not match RGB output shape.")
        out = np.concatenate([out, alpha[..., None]], axis=-1)
    out = (out * 255.0).astype(np.uint8)
    iio.imwrite(str(path), out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Denoise images and evaluate quality.")
    parser.add_argument("--checkpoint", required=True, metavar="PATH")
    parser.add_argument("--model", choices=["spatial", "temporal"], default="spatial")
    parser.add_argument("--naf-base", type=int, default=32, metavar="C",
                        help="Base channel count for NAFNet models (default: 32).")
    parser.add_argument("--naf-preset", choices=["tiny", "small", "standard", "wide"], default=None,
                        help="NAFNet size preset (tiny/small/standard/wide).  "
                             "--naf-base overrides base_channels when both are given.")
    parser.add_argument("--num-frames", type=int, default=5, metavar="T",
                        help="Temporal window size for --model temporal (default: 5).")
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
    _preset_map = {
        "tiny": NAFNetConfig.tiny,
        "small": NAFNetConfig.small,
        "standard": NAFNetConfig.standard,
        "wide": NAFNetConfig.wide,
    }
    naf_config = _preset_map[args.naf_preset]() if args.naf_preset else NAFNetConfig(base_channels=args.naf_base)
    if args.naf_preset and args.naf_base != 32:
        naf_config.base_channels = args.naf_base
    if args.model == "temporal":
        model = NAFNetTemporal(naf_config, num_frames=args.num_frames)
    else:
        model = NAFNet(naf_config)

    try:
        ckpt = torch.load(args.checkpoint, map_location=device)
    except (OSError, RuntimeError) as exc:
        parser.error(
            f"Failed to load checkpoint '{args.checkpoint}'. "
            "The file is missing, truncated, or corrupted. "
            "Try another checkpoint such as final.pth or an epoch_XXXX.pth file."
            f" Original error: {exc}"
        )
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded nafnet/{args.model} from {args.checkpoint}")

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
        input_path = Path(args.input)
        if input_path.is_file():
            pairs = [(input_path, None)]
        else:
            pairs = [
                (p, None)
                for p in sorted(input_path.iterdir())
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
                exr_header = _read_exr_header(noisy_path)
                _save_image(Path(args.output) / noisy_path.name, output_img, exr_header=exr_header)
    else:
        single_file = len(pairs) == 1 and Path(args.input).is_file() if args.input else False
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
                out = Path(args.output) if single_file else Path(args.output) / noisy_path.name
                exr_header = _read_exr_header(noisy_path)
                _save_image(out, output_img, exr_header=exr_header)

    if psnr_values:
        print(f"\nMean PSNR: {np.mean(psnr_values):.2f} dB")
        print(f"Mean SSIM: {np.mean(ssim_values):.4f}")


if __name__ == "__main__":
    main()
