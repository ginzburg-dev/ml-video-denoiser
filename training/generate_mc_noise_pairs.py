"""Generate synthetic paired noisy frames using MCNoise presets.

Applies every preset to a single clean image and saves each result as:

    <output>/<preset_name>.exr

Usage:
    python generate_mc_noise_pairs.py \\
        --clean   /workspace/data/TGB_training/train_clean_lit/TGB1004140_mid/TGB1004140.0001.exr \\
        --presets ../nuke/mc_noise_presets_tgb_lit_patch18_shadowface.json \\
        --output  /workspace/data/TGB_training/train_noisy_synth
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from noise_generators import MCNoisePresetBank

_IMAGE_EXTENSIONS = {".exr", ".png", ".tiff", ".tif"}


def _load_exr(path: Path) -> tuple[np.ndarray, np.ndarray | None, dict]:
    import OpenEXR

    with OpenEXR.File(str(path)) as exr:
        channels = exr.parts[0].channels
        header = dict(exr.parts[0].header)

    r = np.asarray(channels["R"].pixels, dtype=np.float32)
    g = np.asarray(channels["G"].pixels, dtype=np.float32)
    b = np.asarray(channels["B"].pixels, dtype=np.float32)
    rgb = np.stack([r, g, b], axis=-1)

    alpha = None
    if "A" in channels:
        alpha = np.clip(np.asarray(channels["A"].pixels, dtype=np.float32), 0.0, 1.0)

    return rgb, alpha, header


def _load_image(path: Path) -> tuple[np.ndarray, np.ndarray | None, dict]:
    if path.suffix.lower() == ".exr":
        return _load_exr(path)
    import imageio.v3 as iio
    img = iio.imread(str(path)).astype(np.float32)
    if img.max() > 1.5:
        img /= 255.0 if img.max() <= 255.5 else 65535.0
    alpha = None
    if img.shape[-1] == 4:
        alpha = img[..., 3]
        img = img[..., :3]
    return img, alpha, {}


def _save_exr(path: Path, rgb: np.ndarray, alpha: np.ndarray | None, header: dict) -> None:
    import OpenEXR

    path.parent.mkdir(parents=True, exist_ok=True)
    out = rgb.astype(np.float32)

    _PRESERVE = {
        "dataWindow", "displayWindow",
        "pixelAspectRatio", "screenWindowCenter", "screenWindowWidth",
    }
    hdr: dict = {k: v for k, v in header.items() if k in _PRESERVE}
    hdr["type"] = OpenEXR.scanlineimage

    if alpha is not None:
        rgba = np.concatenate([out, alpha[..., None]], axis=-1)
        OpenEXR.File(hdr, {"RGBA": rgba}).write(str(path))
    else:
        OpenEXR.File(hdr, {"RGB": out}).write(str(path))



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply MCNoise presets to clean frames and save as paired noisy data."
    )
    parser.add_argument("--clean",        required=True, metavar="FILE",
                        help="Single clean image file.")
    parser.add_argument("--presets",      required=True, metavar="JSON",
                        help="MCNoise preset bank JSON.")
    parser.add_argument("--output",       required=True, metavar="DIR",
                        help="Noisy output dir.  Each preset saved as <preset_name>.exr.")
    parser.add_argument("--output-clean", default=None, metavar="DIR",
                        help="Clean output dir (default: sibling train_clean_synth).")
    parser.add_argument("--device",  default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed",    type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    device = torch.device(args.device)
    clean_path = Path(args.clean)
    output_root = Path(args.output)
    clean_out = Path(args.output_clean) if args.output_clean else output_root.parent / "train_clean_synth"
    preset_path = Path(args.presets)

    if not clean_path.is_file():
        sys.exit(f"ERROR: --clean must be a single image file: {clean_path}")
    if not preset_path.exists():
        sys.exit(f"ERROR: --presets not found: {preset_path}")

    bank = MCNoisePresetBank.from_json(str(preset_path))
    presets = bank._entries  # (MCNoiseGenerator, weight, name)
    print(f"Loaded {len(presets)} presets from {preset_path.name}")

    rgb, alpha, header = _load_image(clean_path)
    clean_t = torch.from_numpy(rgb.transpose(2, 0, 1)).to(device)  # (3, H, W)
    alpha_t = torch.from_numpy(alpha).to(device).unsqueeze(0) if alpha is not None else None

    output_root.mkdir(parents=True, exist_ok=True)
    clean_out.mkdir(parents=True, exist_ok=True)

    clean_np = clean_t.cpu().numpy().transpose(1, 2, 0)

    for i, (gen, _weight, name) in enumerate(presets, 1):
        noisy_path = output_root / f"{name}.exr"
        clean_copy_path = clean_out / f"{name}.exr"

        if args.skip_existing and noisy_path.exists() and clean_copy_path.exists():
            print(f"  skip  {name}  ({i}/{len(presets)})", flush=True)
            continue

        noisy_t, _, _ = gen(clean_t)

        if alpha_t is not None:
            noisy_t = clean_t + (noisy_t - clean_t) * alpha_t

        noisy_np = noisy_t.cpu().numpy().transpose(1, 2, 0)
        _save_exr(noisy_path, noisy_np, alpha, header)
        _save_exr(clean_copy_path, clean_np, alpha, header)
        print(f"  {i:>3}/{len(presets)}  {name}.exr", flush=True)

    print(f"\nDone. {len(presets)} pairs written.")
    print(f"  noisy: {output_root}")
    print(f"  clean: {clean_out}")


if __name__ == "__main__":
    main()
