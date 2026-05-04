"""Calibrate MCNoise presets from a clean/noisy EXR pair or sequence.

Analyzes real noise statistics tile-by-tile, reports the full sigma/cs_r
distribution, and generates a JSON preset bank covering all noise varieties.

Usage:
    # Single pair
    python noise_calibrate.py --clean clean.exr --noisy noisy.exr --output presets.json

    # Directories (matched by filename, up to --max-frames frames sampled)
    python noise_calibrate.py --clean clean/ --noisy noisy/ --output presets.json --presets 4 --name TGB --max-frames 20

    # Larger tiles for smoother estimates
    python noise_calibrate.py --clean clean.exr --noisy noisy.exr --output presets.json --tile-size 128
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# EXR loading
# ---------------------------------------------------------------------------

def _load_exr(path: str) -> np.ndarray:
    import OpenEXR
    with OpenEXR.File(str(path)) as f:
        ch = f.parts[0].channels
        keys = list(ch.keys())
        if "RGBA" in keys:
            px = np.asarray(ch["RGBA"].pixels, dtype=np.float32)
            return px[..., :3]
        if all(k in keys for k in ("R", "G", "B")):
            r = np.asarray(ch["R"].pixels, dtype=np.float32)
            g = np.asarray(ch["G"].pixels, dtype=np.float32)
            b = np.asarray(ch["B"].pixels, dtype=np.float32)
            return np.stack([r, g, b], axis=-1)
        first = np.asarray(list(ch.values())[0].pixels, dtype=np.float32)
        if first.ndim == 3:
            return first[..., :3]
        arrays = [np.asarray(v.pixels, dtype=np.float32) for v in list(ch.values())[:3]]
        return np.stack(arrays, axis=-1)


# ---------------------------------------------------------------------------
# Tile-based analysis
# ---------------------------------------------------------------------------

def _tile_stats(pairs: list, tile_size: int = 64) -> dict:
    """Per-tile noise analysis returning full sigma/cs_r/cs_b distributions."""
    sigmas, cs_rs, cs_bs, lums = [], [], [], []
    dark_tiles, mid_tiles = [], []
    ff_px_total, total_px = 0, 0
    ff_thresh_used = 1.0

    for clean, noisy in pairs:
        residual = noisy - clean
        h, w = clean.shape[:2]
        lum = 0.2126 * clean[..., 0] + 0.7152 * clean[..., 1] + 0.0722 * clean[..., 2]
        total_px += lum.size

        for thr in (1.0, 0.5, 0.2):
            fp = int(np.any(np.abs(residual) > thr, axis=-1).sum())
            if fp > 0:
                ff_px_total += fp
                ff_thresh_used = thr
                break

        for y in range(0, h - tile_size + 1, tile_size):
            for x in range(0, w - tile_size + 1, tile_size):
                tl = lum[y:y + tile_size, x:x + tile_size]
                tr = residual[y:y + tile_size, x:x + tile_size]

                avg_lum = float(tl.mean())
                lum_scale = math.sqrt(max(avg_lum, 0.0)) + 0.05

                g_std = float(tr[..., 1].std())
                if g_std < 1e-8:
                    continue
                r_std = float(tr[..., 0].std())
                b_std = float(tr[..., 2].std())

                sigma = g_std / lum_scale
                cs_r = math.sqrt(max((r_std / g_std) ** 2 - 1.0, 0.0))
                cs_b = math.sqrt(max((b_std / g_std) ** 2 - 1.0, 0.0))

                sigmas.append(sigma)
                cs_rs.append(cs_r)
                cs_bs.append(cs_b)
                lums.append(avg_lum)

                if avg_lum < 0.05:
                    dark_tiles.append((avg_lum, g_std))
                elif 0.2 <= avg_lum < 0.5:
                    mid_tiles.append((avg_lum, g_std))

    noise_dark_fade = 0.0
    if dark_tiles and mid_tiles:
        avg_dark_lum = float(np.mean([t[0] for t in dark_tiles]))
        avg_dark_g   = float(np.mean([t[1] for t in dark_tiles]))
        avg_mid_lum  = float(np.mean([t[0] for t in mid_tiles]))
        avg_mid_g    = float(np.mean([t[1] for t in mid_tiles]))
        dark_scale = math.sqrt(max(avg_dark_lum, 0.0)) + 0.05
        mid_scale  = math.sqrt(max(avg_mid_lum,  0.0)) + 0.05
        expected_dark = (avg_mid_g / mid_scale) * dark_scale
        if expected_dark > 1e-8:
            noise_dark_fade = float(max(0.0, min(1.0, 1.0 - avg_dark_g / expected_dark)))

    firefly_prob = float(ff_px_total / total_px) if total_px else 0.0

    return {
        "sigmas":          np.array(sigmas,  dtype=np.float32),
        "cs_rs":           np.array(cs_rs,   dtype=np.float32),
        "cs_bs":           np.array(cs_bs,   dtype=np.float32),
        "lums":            np.array(lums,    dtype=np.float32),
        "noise_dark_fade": noise_dark_fade,
        "firefly_prob":    firefly_prob,
        "has_fireflies":   firefly_prob > 1e-5,
        "ff_thresh_used":  ff_thresh_used,
        "tile_count":      len(sigmas),
    }


def _print_distribution(stats: dict) -> None:
    sigmas = stats["sigmas"]
    cs_rs  = stats["cs_rs"]
    cs_bs  = stats["cs_bs"]
    n      = stats["tile_count"]

    pcts       = [10, 25, 50, 75, 90, 95]
    sigma_pcts = np.percentile(sigmas, pcts)
    cs_r_pcts  = np.percentile(cs_rs,  pcts)
    cs_b_med   = float(np.median(cs_bs))

    print()
    print("  Tile distribution (%d tiles):" % n)
    hdr = "  %-10s" + "  %6s" * len(pcts)
    print(hdr % (("",) + tuple("p%d" % p for p in pcts)))
    print(("  %-10s" + "  %6.4f" * len(pcts)) % (("sigma_G",) + tuple(sigma_pcts)))
    print(("  %-10s" + "  %6.4f" * len(pcts)) % (("cs_R",)    + tuple(cs_r_pcts)))
    print("  %-10s  %.4f (median)" % ("cs_B", cs_b_med))
    print("  %-10s  %.4f" % ("dark_fade", stats["noise_dark_fade"]))
    if stats["has_fireflies"]:
        print("  %-10s  prob=%.6f  thresh>%.2f" % (
            "fireflies", stats["firefly_prob"], stats["ff_thresh_used"]))
    else:
        print("  %-10s  none detected" % "fireflies")

    hist_edges = [0.0, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.25, float("inf")]
    print()
    print("  Sigma histogram:")
    for lo, hi in zip(hist_edges[:-1], hist_edges[1:]):
        count = int(((sigmas >= lo) & (sigmas < hi)).sum())
        bar   = "#" * int(40 * count / max(n, 1))
        hi_lbl = "inf" if math.isinf(hi) else "%.2f" % hi
        print("  [%.2f-%-4s] %5.1f%%  %s" % (lo, hi_lbl, 100.0 * count / n, bar))


# ---------------------------------------------------------------------------
# Preset generation from distribution
# ---------------------------------------------------------------------------

def _intensity_and_samples(sigma: float):
    for samples in (64, 32, 16, 8):
        intensity = sigma * math.sqrt(samples)
        if intensity <= 1.5:
            return round(intensity, 4), samples
    return round(sigma * math.sqrt(8), 4), 8


def _make_presets(stats: dict, n: int, prefix: str) -> list:
    """Generate n presets spanning the measured sigma/cs_r distribution."""
    sigmas = stats["sigmas"]
    cs_rs  = stats["cs_rs"]
    cs_bs  = stats["cs_bs"]
    ndf    = stats["noise_dark_fade"]
    ff_p   = stats["firefly_prob"]
    has_ff = stats["has_fireflies"]
    cs_g   = 0.05
    cs_b   = float(np.median(cs_bs))

    # Each preset targets a sigma percentile band.
    # Weights favour the common mid-range; heavy-tail preset gets weight 1.
    if n == 1:
        sigma_pcts = [50];           cs_r_pcts = [50]
        weights = [1.0];             labels = ["MID"]
    elif n == 2:
        sigma_pcts = [30, 80];       cs_r_pcts = [30, 80]
        weights = [2.0, 1.0];        labels = ["LOW", "HIGH"]
    elif n == 3:
        sigma_pcts = [20, 55, 85];   cs_r_pcts = [20, 55, 85]
        weights = [2.0, 3.0, 1.0];   labels = ["LOW", "MID", "HIGH"]
    else:
        sigma_pcts = [15, 40, 70, 90]; cs_r_pcts = [15, 40, 70, 90]
        weights = [2.0, 3.0, 2.0, 1.0]; labels = ["LOW", "MID_A", "MID_B", "HIGH"]

    sigma_vals = [float(np.percentile(sigmas, p)) for p in sigma_pcts]
    cs_r_vals  = [float(np.percentile(cs_rs,  p)) for p in cs_r_pcts]

    presets = []
    for sigma, cs_r, weight, label in zip(sigma_vals, cs_r_vals, weights, labels):
        intensity, samples = _intensity_and_samples(sigma)
        preset = {
            "name":               "%s_%s" % (prefix, label),
            "weight":             weight,
            "intensity":          intensity,
            "samples":            samples,
            "chroma_spread_r":    round(cs_r, 4),
            "chroma_spread_g":    round(cs_g, 4),
            "chroma_spread_b":    round(cs_b, 4),
            "noise_dark_fade":    round(ndf, 4),
            "noise_fade_falloff": 1.25,
            "firefly_thresh":     round(stats["ff_thresh_used"] * 1.5, 4) if has_ff else 0.0,
            "firefly_prob":       round(min(ff_p, 0.05), 6) if has_ff else 0.0,
            "firefly_chroma":     0.1 if has_ff else 0.0,
            "firefly_dark_fade":  0.3 if has_ff else 1.0,
            "firefly_fade_falloff": 1.2,
        }
        presets.append(preset)
    return presets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_IMAGE_EXT = {".exr", ".png", ".tif", ".tiff"}


def _collect_pairs(clean_path: Path, noisy_path: Path, max_frames: int) -> list:
    clean_files  = sorted(p for p in clean_path.iterdir() if p.suffix.lower() in _IMAGE_EXT)
    noisy_by_name = {p.name: p for p in noisy_path.iterdir() if p.suffix.lower() in _IMAGE_EXT}

    pairs, unmatched = [], []
    for cf in clean_files:
        nf = noisy_by_name.get(cf.name)
        if nf is not None:
            pairs.append((cf, nf))
        else:
            unmatched.append(cf.name)

    if unmatched:
        print("  Warning: %d clean frame(s) had no matching noisy file (e.g. %s)" % (
            len(unmatched), unmatched[0]))
    if not pairs:
        raise RuntimeError("No matched pairs found between %s and %s" % (clean_path, noisy_path))

    if max_frames and len(pairs) > max_frames:
        step = len(pairs) / max_frames
        pairs = [pairs[int(i * step)] for i in range(max_frames)]

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate MCNoise presets from a clean/noisy EXR pair or sequence."
    )
    parser.add_argument("--clean",      required=True, metavar="PATH",
                        help="Clean EXR or directory of EXRs")
    parser.add_argument("--noisy",      required=True, metavar="PATH",
                        help="Noisy EXR or directory of EXRs")
    parser.add_argument("--output",     required=True, metavar="PATH",
                        help="Output JSON preset file")
    parser.add_argument("--presets",    type=int, default=3, choices=[1, 2, 3, 4],
                        help="Number of presets (default: 3)")
    parser.add_argument("--name",       default="PRESET", metavar="PREFIX",
                        help="Preset name prefix (default: PRESET)")
    parser.add_argument("--max-frames", type=int, default=0, metavar="N",
                        help="Max frames to sample from a sequence (0 = all)")
    parser.add_argument("--tile-size",  type=int, default=64, metavar="N",
                        help="Tile size in pixels for per-tile analysis (default: 64)")
    args = parser.parse_args()

    clean_path = Path(args.clean)
    noisy_path = Path(args.noisy)

    if clean_path.is_dir():
        if not noisy_path.is_dir():
            parser.error("--noisy must also be a directory when --clean is a directory")
        print("Scanning %s ..." % clean_path)
        file_pairs = _collect_pairs(clean_path, noisy_path, args.max_frames)
        print("Found %d matched frame pair(s)" % len(file_pairs))
        pairs = []
        for i, (cf, nf) in enumerate(file_pairs):
            print("  [%d/%d] %s" % (i + 1, len(file_pairs), cf.name))
            pairs.append((_load_exr(cf), _load_exr(nf)))
    else:
        print("Loading %s ..." % clean_path)
        clean = _load_exr(clean_path)
        print("Loading %s ..." % noisy_path)
        noisy = _load_exr(noisy_path)
        print("Image: %dx%d" % (clean.shape[1], clean.shape[0]))
        pairs = [(clean, noisy)]

    print("\nTile-based noise analysis (tile_size=%d) ..." % args.tile_size)
    stats = _tile_stats(pairs, tile_size=args.tile_size)
    _print_distribution(stats)

    presets = _make_presets(stats, args.presets, args.name)

    print()
    print("Generated %d preset(s) from distribution percentiles:" % len(presets))
    print("  %-28s  intensity  samples  cs_r   cs_g   cs_b   weight" % "name")
    for p in presets:
        print("  %-28s  %.4f     %-7d  %.4f  %.4f  %.4f  %.1f" % (
            p["name"], p["intensity"], p["samples"],
            p["chroma_spread_r"], p["chroma_spread_g"], p["chroma_spread_b"], p["weight"]))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(presets, f, indent=2)
    print("\nWritten: %s" % out)


if __name__ == "__main__":
    main()
