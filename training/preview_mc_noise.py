"""Preview MCNoise presets on a single image.

Examples:
    python preview_mc_noise.py \
        --image /path/to/clean.exr \
        --mc-config ../nuke/mc_noise_presets_tgb_lit_patch15_balanced.json \
        --output-dir /tmp/mc_preview

    python preview_mc_noise.py \
        --image /path/to/clean.exr \
        --mc-config ../nuke/mc_noise_presets_tgb_lit_patch15_balanced.json \
        --preset-name TGB_LIT_PATCH_P50 \
        --output-dir /tmp/mc_preview
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from dataset import _load_exr_image
from infer import _load_alpha, _read_exr_header, _save_image
from noise_generators import MCNoiseGenerator
from preview_noise import save_image


def _display_tonemap(img: np.ndarray) -> np.ndarray:
    x = np.clip(img, 0.0, None)
    x = x / (1.0 + x)
    return np.power(np.clip(x, 0.0, 1.0), 1.0 / 2.2)


def _fit(im: Image.Image, max_w: int, max_h: int) -> Image.Image:
    w, h = im.size
    scale = min(max_w / w, max_h / h)
    size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return im.resize(size, Image.Resampling.LANCZOS)


def _make_contact_sheet(
    clean_rgb: np.ndarray,
    items: list[tuple[dict, np.ndarray]],
    out_path: Path,
) -> None:
    cols = 3
    thumb_w, thumb_h = 420, 240
    label_h = 64
    pad = 16
    rows = math.ceil((len(items) + 1) / cols)
    sheet = Image.new(
        "RGB",
        (cols * (thumb_w + pad) + pad, rows * (thumb_h + label_h + pad) + pad),
        (20, 20, 20),
    )
    draw = ImageDraw.Draw(sheet)

    display_items = [({"name": "CLEAN_REF"}, clean_rgb)] + items
    for idx, (preset, arr) in enumerate(display_items):
        r = idx // cols
        c = idx % cols
        x = pad + c * (thumb_w + pad)
        y = pad + r * (thumb_h + label_h + pad)
        im = Image.fromarray((_display_tonemap(arr) * 255.0).astype(np.uint8))
        thumb = _fit(im, thumb_w, thumb_h)
        tx = x + (thumb_w - thumb.size[0]) // 2
        sheet.paste(thumb, (tx, y))
        draw.rectangle([x, y + thumb_h, x + thumb_w, y + thumb_h + label_h], fill=(35, 35, 35))
        draw.text((x + 8, y + thumb_h + 8), preset["name"], fill=(230, 230, 230))
        if "intensity" in preset:
            draw.text(
                (x + 8, y + thumb_h + 28),
                f"int={preset['intensity']:.3f} csr={preset['chroma_spread_r']:.2f}",
                fill=(180, 180, 180),
            )

    sheet.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview MCNoise presets on a single image.")
    parser.add_argument("--image", required=True, help="Clean input image (EXR/PNG/TIFF...).")
    parser.add_argument("--mc-config", required=True, help="MCNoise preset JSON.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated previews.")
    parser.add_argument(
        "--preset-name",
        default=None,
        help="Optional preset name to render. If omitted, renders all presets.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=1000,
        help="Base seed used for reproducible noise sampling.",
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    config_path = Path(args.mc_config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean = _load_exr_image(image_path).astype(np.float32)
    clean_rgb = clean[:, :, :3]
    clean_t = torch.from_numpy(clean_rgb.transpose(2, 0, 1)).float()
    alpha = _load_alpha(image_path)
    exr_header = _read_exr_header(image_path)
    noisy_suffix = image_path.suffix.lower() or ".exr"

    presets = json.load(config_path.open())
    if args.preset_name is not None:
        presets = [p for p in presets if p.get("name") == args.preset_name]
        if not presets:
            raise SystemExit(f"Preset not found: {args.preset_name}")

    rendered: list[tuple[dict, np.ndarray]] = []
    for i, preset in enumerate(presets):
        kwargs = {k: v for k, v in preset.items() if k not in ("name", "weight")}
        gen = MCNoiseGenerator(**kwargs)
        torch.manual_seed(args.seed_base + i)
        noisy_t, _, _ = gen(clean_t)
        noisy = noisy_t.permute(1, 2, 0).cpu().numpy()

        stem = f"{i+1:02d}_{preset['name']}"
        _save_image(out_dir / f"{stem}{noisy_suffix}", noisy, alpha=alpha, exr_header=exr_header)
        save_image(_display_tonemap(noisy), out_dir / f"{stem}.png")
        rendered.append((preset, noisy))
        print(f"saved {stem}")

    save_image(_display_tonemap(clean_rgb), out_dir / "00_clean_reference.png")
    _make_contact_sheet(clean_rgb, rendered, out_dir / "contact_sheet.png")
    print(f"saved contact_sheet.png -> {out_dir / 'contact_sheet.png'}")


if __name__ == "__main__":
    main()
