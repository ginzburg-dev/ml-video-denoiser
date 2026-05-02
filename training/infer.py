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
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import Tensor

from dataset import _load_image as _load_dataset_image
from dataset import _pad_frame_to_shape
from models import (
    NAFNet,
    NAFNetConfig,
    NAFNetTemporal,
    build_model_from_metadata,
    validate_temporal_num_frames,
)


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


def _apply_color_space_tensor(tensor: Tensor, color_space: str) -> Tensor:
    if color_space == "linear":
        return tensor
    if color_space != "log":
        raise ValueError(f"Unsupported color_space: {color_space!r}")

    transformed = tensor.clone()
    if transformed.ndim == 4:
        transformed[:, :3] = torch.log1p(transformed[:, :3].clamp_min(0.0))
        return transformed
    if transformed.ndim == 5:
        transformed[:, :, :3] = torch.log1p(transformed[:, :, :3].clamp_min(0.0))
        return transformed
    raise ValueError(f"Unsupported tensor rank for color transform: {transformed.ndim}")


def _inverse_color_space_tensor(tensor: Tensor, color_space: str) -> Tensor:
    if color_space == "linear":
        return tensor
    if color_space != "log":
        raise ValueError(f"Unsupported color_space: {color_space!r}")

    restored = tensor.clone()
    if restored.ndim == 4:
        restored[:, :3] = torch.expm1(restored[:, :3]).clamp_min(0.0)
        return restored
    if restored.ndim == 5:
        restored[:, :, :3] = torch.expm1(restored[:, :, :3]).clamp_min(0.0)
        return restored
    raise ValueError(f"Unsupported tensor rank for inverse color transform: {restored.ndim}")


def _resolve_model_type(cli_model: str, metadata: Optional[dict]) -> str:
    """Choose the effective inference mode.

    When checkpoint metadata is present, trust the checkpoint over the CLI
    default so temporal checkpoints cannot be accidentally run through the
    spatial path.
    """
    if metadata is not None:
        return metadata["model_type"]
    return cli_model


def denoise_image(
    model: torch.nn.Module,
    noisy: np.ndarray,
    device: torch.device,
    tile_size: int = 0,
    use_amp: bool = True,
    color_space: str = "linear",
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
    x = _apply_color_space_tensor(x, color_space)

    model.eval()
    with torch.no_grad():
        if tile_size > 0:
            output = _tile_inference(model, x, tile_size, use_amp, device)
        else:
            with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
                output = model(x)
    output = _inverse_color_space_tensor(output, color_space)

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


def denoise_temporal_sequence(
    model: torch.nn.Module,
    sequence: list[np.ndarray],
    device: torch.device,
    use_amp: bool = True,
    color_space: str = "linear",
    temporal_flip: bool = True,
) -> list[np.ndarray]:
    """Denoise every frame in *sequence*, with optional test-time temporal flip.

    **What temporal flip does:**
    The model sees frames in order [i-r, …, i, …, i+r] to produce output_i.
    With temporal_flip=True each frame is also denoised from a time-reversed
    window [i+r, …, i, …, i-r] (neighbours appear in opposite order).  The two
    predictions are averaged:

        final_i = (forward_i + backward_i) / 2

    Because the two passes see different temporal context orderings, any
    systematic bias the model has toward one direction of motion cancels out,
    and per-frame noise variance drops by ~30 %.  Temporal consistency also
    improves because adjacent final frames share a common backward-pass ancestor.

    Cost: 2× inference time.

    Args:
        model:         Trained temporal model (eval mode expected).
        sequence:      List of (H, W, C) float32 noisy frames.
        device:        Target device.
        use_amp:       AMP FP16 inference.
        color_space:   "linear" or "log".
        temporal_flip: Average forward and time-reversed passes (default True).

    Returns:
        List of (H, W, C) float32 denoised frames, same length as *sequence*.
    """
    n = len(sequence)

    forward = [
        denoise_temporal_frame(model, sequence, i, device, use_amp, color_space)
        for i in range(n)
    ]
    if not temporal_flip:
        return forward

    # Reversed sequence: frame i in reversed order ↔ frame (n-1-i) in original
    rev_seq = list(reversed(sequence))
    rev_out = [
        denoise_temporal_frame(model, rev_seq, i, device, use_amp, color_space)
        for i in range(n)
    ]
    backward = list(reversed(rev_out))

    return [(f + b) * 0.5 for f, b in zip(forward, backward)]


def _clip_indices(length: int, centre_idx: int, num_frames: int) -> list[int]:
    """Return a centred temporal window using edge replication at boundaries."""
    validate_temporal_num_frames(num_frames)
    radius = num_frames // 2
    return [min(max(centre_idx + offset, 0), length - 1) for offset in range(-radius, radius + 1)]


def denoise_temporal_frame(
    model: torch.nn.Module,
    sequence: list[np.ndarray],
    frame_idx: int,
    device: torch.device,
    use_amp: bool = True,
    color_space: str = "linear",
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
    x = _apply_color_space_tensor(x, color_space)

    model.eval()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp and device.type == "cuda"):
            output = model(x)
    output = _inverse_color_space_tensor(output, color_space)

    ref_h, ref_w = sequence[frame_idx].shape[:2]
    return output.squeeze(0).permute(1, 2, 0).cpu().float().numpy()[:ref_h, :ref_w]


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------


def _load_image(path: Path) -> np.ndarray:
    img = _load_dataset_image(path)
    return img[:, :, :3] if img.shape[2] > 3 else img


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
    parser.add_argument("--naf-preset", choices=["tiny", "small", "exp048", "standard", "wide"], default=None,
                        help="NAFNet size preset (tiny/small/exp048/standard/wide).  "
                             "--naf-base overrides base_channels when both are given.")
    parser.add_argument("--num-frames", type=int, default=3, metavar="T",
                        help="Temporal window size for --model temporal (default: 3).")
    parser.add_argument("--use-warp", action="store_true",
                        help="Enable learned warp when rebuilding a legacy temporal checkpoint.")
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
    parser.add_argument("--color-space", choices=["linear", "log"], default=None,
                        help="Override checkpoint color space for legacy models.")
    parser.add_argument("--no-temporal-flip", action="store_true",
                        help="Disable test-time temporal flip averaging for temporal models.")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    _preset_map = {
        "tiny": NAFNetConfig.tiny,
        "small": NAFNetConfig.small,
        "exp048": NAFNetConfig.exp048,
        "standard": NAFNetConfig.standard,
        "wide": NAFNetConfig.wide,
    }
    try:
        ckpt = torch.load(args.checkpoint, map_location=device)
    except (OSError, RuntimeError) as exc:
        parser.error(
            f"Failed to load checkpoint '{args.checkpoint}'. "
            "The file is missing, truncated, or corrupted. "
            "Try another checkpoint such as final.pth or an epoch_XXXX.pth file."
            f" Original error: {exc}"
        )
    metadata = ckpt.get("model_metadata") if isinstance(ckpt, dict) else None
    training_config = ckpt.get("training_config", {}) if isinstance(ckpt, dict) else {}
    color_space = args.color_space or training_config.get("color_space", "linear")
    model_type = _resolve_model_type(args.model, metadata)
    if metadata is not None:
        try:
            model = build_model_from_metadata(metadata)
        except (KeyError, TypeError, ValueError) as exc:
            parser.error(f"Checkpoint model metadata is invalid: {exc}")
        _model_type = metadata["model_type"]
        if _model_type in ("spatial", "temporal"):
            _base_ch = metadata["naf_config"]["base_channels"]
        elif _model_type == "cascade":
            _base_ch = metadata["spatial_config"]["base_channels"]
        elif _model_type == "refined_temporal":
            _base_ch = metadata["base_metadata"]["naf_config"]["base_channels"]
        else:
            _base_ch = "?"
        print(
            "Loaded model metadata from checkpoint: "
            f"type={_model_type}, "
            f"base_channels={_base_ch}"
            + (
                f", num_frames={metadata['num_frames']}, use_warp={metadata.get('use_warp', False)}"
                if _model_type == "temporal"
                else ""
            )
            + (
                f", num_frames={metadata['num_frames']}"
                if _model_type in ("cascade", "refined_temporal")
                else ""
            )
            + f", color_space={color_space}"
        )
    else:
        print(
            "Warning: checkpoint has no model_metadata; rebuilding from CLI flags. "
            "If this is a legacy temporal checkpoint, pass the original "
            "--model/--num-frames/--use-warp values.",
            file=sys.stderr,
        )
        naf_config = (
            _preset_map[args.naf_preset]()
            if args.naf_preset
            else NAFNetConfig(base_channels=args.naf_base)
        )
        if args.naf_preset and args.naf_base != 32:
            naf_config.base_channels = args.naf_base
        if model_type == "temporal":
            try:
                validate_temporal_num_frames(args.num_frames)
            except ValueError as exc:
                parser.error(str(exc))
            model = NAFNetTemporal(naf_config, num_frames=args.num_frames, use_warp=args.use_warp)
        else:
            model = NAFNet(naf_config)
        print(
            "Reconstructed model from CLI flags: "
            f"type={model_type}, base_channels={naf_config.base_channels}"
            + (
                f", num_frames={args.num_frames}, use_warp={args.use_warp}"
                if model_type == "temporal"
                else ""
            )
            + f", color_space={color_space}"
        )
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"Loaded nafnet/{model_type} from {args.checkpoint}")

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

    if model_type in ("temporal", "refined_temporal", "cascade"):
        if args.tile > 0:
            parser.error("--tile is not supported for temporal inference.")
        noisy_paths = [noisy_path for noisy_path, _ in pairs]
        noisy_sequence = [_load_image(path) for path in noisy_paths]
        if args.sigma > 0:
            noisy_sequence = [
                (img + np.random.randn(*img.shape) * args.sigma).clip(0, 1)
                for img in noisy_sequence
            ]

        temporal_flip = not args.no_temporal_flip
        if temporal_flip:
            print("Temporal flip averaging enabled (2× inference, --no-temporal-flip to disable).")
        denoised_sequence = denoise_temporal_sequence(
            model, noisy_sequence, device, use_amp,
            color_space=color_space, temporal_flip=temporal_flip,
        )

        for (noisy_path, clean_path), output_img in zip(pairs, denoised_sequence):
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

            output_img = denoise_image(
                model,
                noisy_img,
                device,
                args.tile,
                use_amp,
                color_space=color_space,
            )

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
