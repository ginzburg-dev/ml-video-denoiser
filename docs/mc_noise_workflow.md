# MCNoise — Nuke to Training Workflow

MCNoise is a luminance-dependent grain and firefly generator. The Blink kernel
(`nuke/MCNoise.blink`) and the Python implementation (`MCNoiseGenerator` in
`training/noise_generators.py`) share identical math, so noise you preview in
Nuke is the same noise the model trains on.

---

## Files

| File | Purpose |
|------|---------|
| `nuke/MCNoise.blink` | Blink kernel — load into a BlinkScript node |
| `nuke/mc_noise_presets.nk` | Blank Nuke script with light / medium / heavy starting presets |
| `nuke/export_mc_noise_presets.py` | Python script that exports selected nodes to JSON |

---

## Step 1 — Set up presets in Nuke

Open `nuke/mc_noise_presets.nk`. It contains three NoOp nodes
(`MCNoise_light`, `MCNoise_medium`, `MCNoise_heavy`) whose knobs mirror the
Blink kernel parameters exactly.

To use real BlinkScript nodes instead:

1. Create a **BlinkScript** node, paste the contents of `nuke/MCNoise.blink`
   into the kernel source, and recompile.
2. Add a user knob called `mc_weight` (Float, Training tab) to each node.

### Parameters

| Knob | Default | Description |
|------|---------|-------------|
| `intensity` | 1.0 | Base noise amplitude |
| `samples` | 16 | Virtual sample count — sigma = intensity / √samples |
| `chromaSpread` | 0.1 | Independent per-channel chroma noise on top of shared luma grain |
| `noiseDarkFade` | 0.0 | 0 = full noise in blacks, 1 = zero noise in blacks |
| `noiseFadeFalloff` | 1.0 | Power curve for noise dark fade (1 = linear, >1 = sharper cutoff) |
| `fireflyThresh` | 6.0 | Floor magnitude of firefly spikes |
| `fireflyProb` | 0.003 | Per-pixel probability of a firefly |
| `fireflyChroma` | 0.1 | Chroma spread of firefly spikes |
| `fireflyDarkFade` | 0.0 | 0 = fireflies everywhere, 1 = none in blacks |
| `fireflyFadeFalloff` | 1.0 | Power curve for firefly dark fade |
| `mc_weight` | 1.0 | **Training only** — relative sampling weight (see Step 3) |

---

## Step 2 — Export presets to JSON

### Option A — Export button (recommended)

1. Select the nodes you want to export (or select nothing to export all
   MCNoise nodes in the script).
2. Click **Export Presets** on the `ExportPresets` node.
3. Set the `script_dir` knob to the absolute path of `ml-video-denoiser/nuke/`.
4. Choose an output path when prompted.

### Option B — Script Editor

```python
import sys
sys.path.insert(0, "/path/to/ml-video-denoiser/nuke")
from export_mc_noise_presets import export_selected, export_all

# export selected nodes:
export_selected("/path/to/mc_noise_presets.json")

# or export every MCNoise node in the script:
export_all("/path/to/mc_noise_presets.json")
```

### Output JSON format

```json
[
  {
    "name": "MCNoise_light",
    "weight": 1.0,
    "intensity": 0.5,
    "samples": 32,
    "chroma_spread": 0.08,
    "noise_dark_fade": 0.0,
    "noise_fade_falloff": 1.0,
    "firefly_thresh": 6.0,
    "firefly_prob": 0.001,
    "firefly_chroma": 0.1,
    "firefly_dark_fade": 0.0,
    "firefly_fade_falloff": 1.0
  },
  {
    "name": "MCNoise_medium",
    "weight": 3.0,
    ...
  }
]
```

Keys are snake_case (Nuke camelCase knob names are translated automatically).
`name` is informational. `weight` controls how often the preset fires during
training relative to the others — it does not need to sum to any value.

---

## Step 3 — Train with the preset bank

```bash
uv run python training.py \
  --noise mc \
  --noise-mc-config /path/to/mc_noise_presets.json \
  --data /path/to/clean_frames \
  --output checkpoints/mc_noise_run
```

With multiple presets and weights `[1, 3, 1]`, the medium preset fires on
~60 % of training samples, light and heavy each on ~20 %.

### Single preset (no JSON)

If you just want default MCNoise parameters with no preset file:

```bash
--noise mc
```

This uses `MCNoiseGenerator()` with all defaults.

---

## How the math maps between Nuke and Python

The Python `MCNoiseGenerator` mirrors the Blink kernel line-for-line:

| Blink | Python |
|-------|--------|
| `_sigma = intensity / sqrt(samples)` | `self._sigma = intensity / max(samples,1)**0.5` |
| `lumScale = sqrt(lum) + darkFloor` | `lum.clamp_min(0).sqrt() + dark_floor` |
| `noise = _sigma * lumScale * noiseMask` | same |
| `lumaG + cr/cg/cb` per channel | `torch.randn(1,…) + torch.randn(3,…) * chroma_spread` |
| `hash()`-based firefly gate | `torch.rand(…) < firefly_prob * ff_mask` |

The only difference: Nuke uses a deterministic integer hash seeded by pixel
position + `seed` knob; PyTorch uses its global RNG (seeded by `--seed` in
`training.py`). Distributions are identical.
