"""Export trained model weights for C++ inference.

Writes two artifacts to *output_dir*:

  manifest.json
      JSON file describing every tensor: name, shape, dtype, and the path
      to its binary file.  The C++ WeightStore reads this to locate tensors.

  weights/<name>.bin
      Raw little-endian float arrays (float16 or float32 depending on tensor
      type).  No header — shape information lives in the manifest only.

BN running_mean and running_var are always exported as float32 regardless
of the model dtype, because the C++ BatchNorm kernel applies them in FP32.
All other parameters follow the requested *dtype*.

Usage:
    uv run python export.py \\
        --checkpoint checkpoints/residual_standard/best.pth \\
        --output weights/residual_standard \\
        --dtype float16
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn

from models import NAFNet, NAFNetConfig, NAFNetTemporal


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------


def export_model(
    model: nn.Module,
    output_dir: Path,
    dtype: Literal["float16", "float32"] = "float16",
) -> Path:
    """Export model weights to binary files and a manifest JSON.

    Args:
        model: A trained NAFNet denoiser in eval mode.
        output_dir: Root directory for the export.  Created if absent.
        dtype: Tensor dtype for conv/linear weights.  BN stats always fp32.

    Returns:
        Path to the written manifest.json.
    """
    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    np_dtype = np.float16 if dtype == "float16" else np.float32
    state_dict = model.state_dict()

    # Detect architecture and collect config info
    arch_info = _describe_architecture(model)

    layers = []
    for name, param in state_dict.items():
        # All BN parameters → always float32.
        # The C++ BatchNorm2dLayer applies scale/shift in FP32 internally and
        # checks dtype == kFloat32 for all four BN tensors (weight, bias,
        # running_mean, running_var).  num_batches_tracked is a scalar counter
        # that is never uploaded to the GPU but exported for completeness.
        force_fp32 = ".bn." in name or any(
            name.endswith(suffix)
            for suffix in ("running_mean", "running_var", "num_batches_tracked")
        )
        tensor_dtype = "float32" if force_fp32 else dtype
        out_np_dtype = np.float32 if force_fp32 else np_dtype

        arr = param.cpu().float().numpy().astype(out_np_dtype)
        fname = name.replace(".", "_") + ".bin"
        arr.tofile(weights_dir / fname)

        layers.append(
            {
                "name": name,
                "shape": list(param.shape),
                "dtype": tensor_dtype,
                "file": f"weights/{fname}",
            }
        )

    manifest = {
        "version": "1.0",
        "dtype": dtype,
        "architecture": arch_info,
        "layers": layers,
    }

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    total_mb = sum((weights_dir / l["file"].split("/")[-1]).stat().st_size for l in layers) / 1e6
    print(
        f"Exported {len(layers)} tensors → {output_dir} "
        f"({total_mb:.1f} MB, dtype={dtype})"
    )
    return manifest_path


def _describe_architecture(model: nn.Module) -> dict:
    """Extract architecture metadata from a model instance."""
    if isinstance(model, (NAFNet, NAFNetTemporal)):
        result: dict = {
            "type": "nafnet_temporal" if isinstance(model, NAFNetTemporal) else "nafnet_residual",
            "base_channels": model.intro.out_channels,
            "num_levels": len(model.encoders),
            "in_channels": model.intro.in_channels,
        }
        if isinstance(model, NAFNetTemporal):
            result["num_frames"] = model._num_frames
            result["use_warp"] = model._use_warp
        return result
    return {"type": "unknown"}


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------


def verify_export(
    model: nn.Module,
    manifest_path: Path,
    rtol: float = 1e-2,
) -> bool:
    """Reload exported tensors and verify they match the model state_dict.

    Args:
        model: The original model.
        manifest_path: Path to the written manifest.json.
        rtol: Relative tolerance for comparison (FP16 has ~1e-3 relative error).

    Returns:
        True if all tensors match within tolerance.
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    state = model.state_dict()
    output_dir = manifest_path.parent
    all_ok = True

    for entry in manifest["layers"]:
        name = entry["name"]
        bin_path = output_dir / entry["file"]
        dtype_map = {"float16": np.float16, "float32": np.float32}
        arr = np.fromfile(bin_path, dtype=dtype_map[entry["dtype"]])
        arr = arr.reshape(entry["shape"])

        expected = state[name].cpu().float().numpy()
        loaded = arr.astype(np.float32)

        if not np.allclose(expected, loaded, rtol=rtol, atol=1e-4):
            print(f"  MISMATCH: {name}  max_diff={np.abs(expected - loaded).max():.6f}")
            all_ok = False

    if all_ok:
        print(f"Export verified: all {len(manifest['layers'])} tensors match.")
    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Export model weights for C++ inference.")
    parser.add_argument("--checkpoint", required=True, metavar="PATH")
    parser.add_argument("--model", choices=["spatial", "temporal"], default="spatial")
    parser.add_argument("--naf-base", type=int, default=32, metavar="C",
                        help="Base channel count for NAFNet models (default: 32).")
    parser.add_argument("--naf-preset", choices=["tiny", "small", "standard", "wide"], default=None,
                        help="NAFNet size preset.  --naf-base overrides base_channels when both given.")
    parser.add_argument("--num-frames", type=int, default=5, metavar="T",
                        help="Temporal window size for --model temporal (default: 5).")
    parser.add_argument("--output", required=True, metavar="DIR")
    parser.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--verify", action="store_true",
                        help="Reload and verify exported tensors against state_dict.")
    args = parser.parse_args()

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

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    manifest_path = export_model(model, Path(args.output), dtype=args.dtype)

    if args.verify:
        verify_export(model, manifest_path)


if __name__ == "__main__":
    main()
