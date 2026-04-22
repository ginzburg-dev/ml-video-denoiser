# ML Video Denoiser

GPU-accelerated video / image denoiser.  
Python training (NAFNet) → raw weight export → C++ CLI inference with no ONNX / LibTorch dependency.

---

## Repository layout

```
ml-video-denoiser/
├── training/                    Python training pipeline (uv-managed)
│   ├── models.py                NAFNet architectures, spatial weight transfer + freeze helpers
│   ├── dataset.py               PatchDataset, VideoSequenceDataset,
│   │                              PairedPatchDataset, PairedVideoSequenceDataset,
│   │                              CombinedDataset
│   ├── noise_generators.py      Gaussian, Poisson-Gaussian, RealInject, RealRAW, Mixed
│   ├── noise_profiler.py        Dark-frame calibration → JSON profile / patch pool
│   ├── losses.py                NoiseWeightedL1Loss
│   ├── training.py              Training loop (AMP, AdamW, cosine LR, paired support)
│   ├── export.py                Weights → manifest.json + .bin files
│   ├── infer.py                 Python inference + PSNR/SSIM evaluation
│   └── tests/                   117+ pytest tests
│       ├── test_models.py       Model architecture + forward-pass tests
│       ├── test_dataset.py      All five dataset classes
│       ├── test_noise_generators.py  All noise generator classes
│       ├── test_noise_profiler.py    Dark-frame profiling functions
│       ├── test_export.py       Weight export + round-trip verification
│       ├── test_noise_visual.py Noise visualisation diagnostic tests
│       └── test_training_cli.py Training CLI smoke + validation tests
├── src/                         C++ engine source
│   ├── tensor.cu                RAII CUDA tensor
│   ├── weight_loader.cc         mmap + lazy H2D upload
│   ├── layers/                  Conv2d, BatchNorm2d, ReLU, Upsample, MaxPool2d, DeformConv2d
│   ├── models/                  nafnet.cu, nafnet_temporal.cu
│   ├── io/                      image_io.cc (stb), exr_io.cc (tinyexr), video_io.cc (ffmpeg shell)
│   └── kernels/                 bn_inference, concat, model_kernels, deform_im2col
├── include/denoiser/            Public C++ headers
├── cli/main.cc                  denoiser CLI
├── tests/                       C++ gtests + Python integration tests
│   ├── gen_fixtures.py          Generates tiny_unet fixture for parity tests
│   ├── gen_sample_images.py     Generates bundled 128×128 PNGs for smoke tests
│   ├── visualise_noise.py       Visual diagnostic — noise grids + temporal clip consistency PNGs
│   ├── noise_preview.py         CLI noise preview — apply generators at multiple levels to a user image
│   └── test_integration.py      Phase 6 Python ↔ C++ parity tests
├── third_party/                 nlohmann_json, stb, tinyexr, googletest (submodules)
└── CMakeLists.txt
```

---

## Quick start (no external data)

The fastest way to verify everything works after a fresh clone — no dataset download required.

```bash
# 1. Clone with submodules
git clone --recurse-submodules <repo>
cd ml-video-denoiser

# 2. Python environment
cd training && uv sync && cd ..

# 3. C++ engine (adjust CUDA arch for your GPU)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build build -j$(nproc)

# 4. Generate test fixtures (tiny exported model + parity inputs)
cd training && uv run python ../tests/gen_fixtures.py && cd ..

# 5. Run the full test suite — no GPU dataset needed
cd training && uv run pytest ../tests/test_integration.py -v
```

What the integration suite covers without any external data:

| Group | What it tests |
|---|---|
| `TestExportRoundtrip` | `export.py` produces correct manifest + `.bin` files |
| `TestNAFNetPython` | Python model output shape, reflect-padding, finite output |
| `TestCppGtests` | All C++ unit tests (tensor, weight loader, conv, BN, NAFNet parity) |
| `TestPythonCppParity` | Python ↔ C++ pixel-level agreement (max diff < 0.005) |
| `TestCLI` | CLI binary: single image, directory, error handling |
| `TestTrainingSmokeTest` | 2-epoch train on bundled PNGs — loss decreases, checkpoint saves, resume works |
| `TestPSNRRegression` | Random-init model produces no NaN/Inf |

The `TestTrainingSmokeTest` group uses five synthetic 128×128 PNG images committed to
`tests/fixtures/sample_images/`.  They cover horizontal/vertical gradients, a checkerboard,
colour patches, and a sinusoidal texture.

---

## Requirements

| Tool | Version |
|---|---|
| CUDA Toolkit | ≥ 11.8 |
| cuDNN | ≥ 8.0 (≥ 8.5 for cuDNN upsample path) |
| CMake | ≥ 3.22 |
| C++ compiler | GCC ≥ 12 or Clang ≥ 16 (C++23) |
| Python | ≥ 3.11 |
| uv | latest |
| ffmpeg | on PATH (runtime, for video I/O) |

---

## Build

### 1. Clone with submodules

```bash
git clone --recurse-submodules <repo>
cd ml-video-denoiser
```

### 2. Python training environment

```bash
cd training
uv sync                    # installs all deps from uv.lock
uv run pytest tests/ -v    # 117 tests — should all pass
cd ..
```

### 3. C++ engine

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80;86"   # adjust for your GPU
cmake --build build -j$(nproc)
```

Optional flags:
```bash
-DUSE_LIBRAW=ON            # enable LibRaw for camera RAW reading
-DDENOISER_BUILD_TESTS=OFF # skip gtest build
```

### 4. C++ tests

```bash
# Generate fixture (tiny exported model for parity tests)
cd training && uv run python ../tests/gen_fixtures.py && cd ..

# Run C++ unit tests
./build/tests/denoiser_tests

# Run DISABLED parity tests (require fixture)
./build/tests/denoiser_tests \
    --gtest_also_run_disabled_tests \
    --gtest_filter="NAFNetForwardTest.*"
```

### 5. Integration tests (Python + C++)

```bash
cd training
uv run pytest ../tests/test_integration.py -v
```

The bundled sample images are already committed to `tests/fixtures/sample_images/`.
If you need to regenerate them:

```bash
cd training && uv run python ../tests/gen_sample_images.py
```

---

## Tests

### Python unit tests (`training/tests/`)

Run all Python unit tests from the `training/` directory:

```bash
cd training
uv run pytest tests/ -v
```

#### `test_models.py`

Tests for `NAFNet` and `NAFNetTemporal` model architectures.

| Test class | What it covers |
|---|---|
| `TestPadding` | `_pad_to_multiple` / `_unpad` — divisibility, roundtrip, no-op |
| `TestNAFNetForward` | Full forward pass shape, odd spatial dims, batch size 2, gradient flow |
| `TestNAFNetAuxChannels` | `in_channels=9` input → `(B,3,H,W)` output |
| `TestNAFNetIdentityInit` | Output ≈ input[:,:3] at epoch 0 (zero-init ending) |
| `TestNAFNetConfigs` | Tiny / small / standard / wide presets build and run; tiny < wide params |
| `TestNAFNetTemporalForward` | 3-frame clip → single denoised frame, correct shape, identity at init |
| `TestNAFNetTemporalConfigs` | Temporal model with tiny and small presets |
| `TestNAFNetTemporalWarp` | `use_warp=True` — output shape correct |
| `TestSpatialWeightTransfer` | `load_spatial_weights` transfers all matching keys; temporal keys untouched |
| `TestFreezeUnfreeze` | `freeze_spatial` / `unfreeze_spatial` set `requires_grad` correctly |

#### `test_dataset.py`

Tests for all five dataset classes and their helper functions.

| Test class | What it covers |
|---|---|
| `TestLoadImage` | PNG/grayscale loading, normalisation to [0,1], HWC layout |
| `TestRandomCrop` | Crop size, padding when image smaller than patch |
| `TestAugment` | Flip/rotation does not change shape |
| `TestPatchDataset` | Length, output shapes `(C,H,W)`, sigma_map range, noise applied |
| `TestVideoSequenceDataset` | Clip extraction, temporal stack `(T,C,H,W)`, consistent crop/augment |
| `TestPairedPatchDataset` | Paired load by name/position, sigma from real residual, augment alignment |
| `TestPairedVideoSequenceDataset` | Temporal paired clips, sigma from frame residuals |
| `TestCombinedDataset` | Weighted mixing of two datasets, deterministic sampling |

#### `test_noise_generators.py`

Tests for every noise generator class.

| Test class | What it covers |
|---|---|
| `TestGaussianNoiseGenerator` | Output shapes, noisy ∈ [0,1], sigma_map ≥ 0, protocol conformance |
| `TestPoissonGaussianNoiseGenerator` | Heteroscedastic sigma (brighter → more noise) |
| `TestRealNoiseInjectionGenerator` | Pool load, patch add, local-std sigma_map |
| `TestRealRAWNoiseGenerator` | Profile JSON load, multi-ISO sampling |
| `TestCameraNoiseGenerator` | ISO-to-params mapping, single-frame and for_clip() modes, fixed-pattern row consistency |
| `TestMixedNoiseGenerator` | Weight normalisation, all-types combination, missing pool/profile graceful fallback |

#### `test_noise_profiler.py`

Tests for dark-frame noise calibration functions.

| Test class | What it covers |
|---|---|
| `TestComputeTemporalStats` | `mean_dark` shape matches frames, `sigma_r` is a reasonable positive scalar |
| `TestEstimatePoissonGain` | Returns positive float from flat frames; `None` when only one flat frame |
| `TestBuildParametricProfile` | Output JSON keys (`sigma_r`, `K`, `mean_dark_file`), merges into existing file |
| `TestBuildPatchPool` | Pool shape `(N, H, W, C)`, zero-mean residuals, error on oversized patch |

#### `test_export.py`

Tests for `export.py` weight serialisation.

| Test class | What it covers |
|---|---|
| `TestExportWeights` | Manifest JSON structure, `.bin` files present for every layer |
| `TestExportRoundtrip` | `state_dict` values recovered from `.bin` within FP16 tolerance |
| `TestExportBNFloat32` | BN gamma, beta, running_mean, running_var are always FP32 in manifest |
| `TestExportVerify` | `--verify` flag runs Python forward pass and reports max diff |

#### `test_noise_visual.py`

Tests for the noise visualisation diagnostic script (`tests/visualise_noise.py`).
All tests are skipped automatically if the bundled sample images are not present.

| Test | What it covers |
|---|---|
| `test_run_produces_output_pngs` | `run()` returns a non-empty list; every returned path exists on disk |
| `test_all_expected_categories_present` | Overview, paired_dataset, patch_pool_residuals, sigma_comparison, temporal_clip_consistency are all produced |
| `test_generator_grids_cover_all_noise_types` | At least 5 per-generator PNGs (`generator_*.png`) are written |
| `test_no_zero_byte_outputs` | No PNG is empty (matplotlib rendered actual pixel data) |
| `test_sample_image_count_matches_rows` | Still 5 bundled sample images; overview PNG is ≥ 200×200 px |

#### `test_training_cli.py`

Smoke tests for the training CLI and validation pipeline.

| Test class | What it covers |
|---|---|
| `TestTrainingCLI` | Synthetic-only 2-epoch run completes; `final.pth` saved; loss printed to stdout |
| `TestPairedTraining` | Paired-only and mixed (synthetic + paired) runs complete without error |
| `TestValidationPairs` | `--val-clean / --val-noisy` validation runs; val PSNR logged; `best.pth` saved |
| `TestValidationCropModes` | `center`, `grid`, `full`, `random` crop modes all run without error |
| `TestCheckpointResume` | Checkpoint from epoch 1 resumes correctly to epoch 2 |
| `TestSpatialWeightsFlag` | `--spatial-weights` loads correct tensor count; `--freeze-spatial` freezes backbone; both reject `--model temporal` when used without temporal model |
| `TestFramesPerSequence` | `--frames-per-sequence N` selects evenly spread frames from sequence subdirs; flat directory triggers warning and uses all images; rejects `--model temporal` |

### Integration tests (`tests/test_integration.py`)

Run from the `training/` directory after building the C++ engine:

```bash
cd training
uv run pytest ../tests/test_integration.py -v
```

| Test class | What it covers |
|---|---|
| `TestExportRoundtrip` | Python export → C++ weight loader — all tensors match |
| `TestNAFNetPython` | Python model forward pass shape, padding, and output sanity on random input |
| `TestCppGtests` | Invokes `./build/tests/denoiser_tests`; all C++ gtests pass |
| `TestPythonCppParity` | Same input → Python model vs C++ engine → max pixel diff < 0.005 |
| `TestCLI` | `denoiser` binary: single PNG, directory of PNGs, bad-path error |
| `TestTrainingSmokeTest` | 2-epoch train on bundled 128×128 PNGs; loss decreases; checkpoint saves; resume works |
| `TestPSNRRegression` | Random-init model output is finite (no NaN/Inf) on a noisy input |

---

## Training

### Synthetic noise only (default)

The simplest setup — no paired data needed.  Noise is generated on the fly from clean images.

```bash
cd training
uv run python training.py \
    --data /path/to/clean/images \
    --output checkpoints/spatial_standard \
    --epochs 300
```

### Mixed noise (recommended for real-camera generalization)

```bash
uv run python training.py \
    --data /path/to/clean/images \
    --patch-pool pools/camera_iso1600.npz \
    --noise-profile profiles/camera_iso1600.json \
    --output checkpoints/spatial_mixed \
    --epochs 300
```

When `--patch-pool` and/or `--noise-profile` are provided, `MixedNoiseGenerator`
automatically includes `RealNoiseInjectionGenerator` and `RealRAWNoiseGenerator`
alongside the synthetic generators.  Missing paths fall back gracefully to
Gaussian + Poisson-Gaussian only.

### Paired training (real clean/noisy image pairs)

Use paired data when you have matching clean ground-truth and real noisy captures
(e.g., long-exposure clean + short-exposure noisy).

```bash
# Paired only
uv run python training.py \
    --paired-clean /path/to/clean \
    --paired-noisy /path/to/noisy \
    --output checkpoints/spatial_paired \
    --epochs 300

# Mixed: 60% synthetic + 40% paired
uv run python training.py \
    --data /path/to/clean/images \
    --paired-clean /path/to/clean \
    --paired-noisy /path/to/noisy \
    --paired-weight 0.4 \
    --output checkpoints/spatial_mixed \
    --epochs 300
```

Paired image directories:
- Files are matched by **filename stem** by default (e.g., `clean/frame_001.png` ↔
  `noisy/frame_001.png`).
- Pass `--no-name-match` to match by **sorted position** instead.
- Multiple directory pairs are supported:
  `--paired-clean dirA dirB --paired-noisy dirA_noisy dirB_noisy`

The sigma map for paired samples is derived from the actual noise residual
`|noisy − clean|`, not from a parametric model, so the loss weighting reflects
true per-pixel noise levels.

### Paired validation

Validation can use either synthetic or real paired data independently of the
training set:

```bash
# Synthetic validation alongside paired training
uv run python training.py \
    --paired-clean /path/to/train_clean \
    --paired-noisy /path/to/train_noisy \
    --val-data /path/to/clean_val_images \
    --output checkpoints/run

# Paired validation
uv run python training.py \
    --data /path/to/clean/images \
    --val-clean /path/to/val_clean \
    --val-noisy /path/to/val_noisy \
    --output checkpoints/run
```

Validation crop modes (control via `--val-crop-mode`):

| Mode | Behaviour |
|---|---|
| `random` | Random crop each validation step (default) |
| `center` | Fixed centre crop — fully deterministic |
| `grid` | N×N deterministic crop grid per image; set N with `--val-grid-size` |
| `full` | Full-resolution frames; validation batch size forced to 1 |

```bash
# Grid validation: 2×2 = 4 crops per image
uv run python training.py \
    --val-clean /path/to/val_clean \
    --val-noisy /path/to/val_noisy \
    --val-crop-mode grid \
    --val-grid-size 2

# Full-resolution validation
uv run python training.py \
    --val-data /path/to/val_images \
    --val-crop-mode full
```

### Camera video noise

`CameraNoiseGenerator` models real camera sensor noise via an ISO-parameterised Poisson-Gaussian formula:

```
K       = K_ref  × (iso / iso_ref)          — shot noise, linear in ISO
sigma_r = sr_ref × (iso / iso_ref) ^ 0.5   — read noise, sub-linear
noisy   = Poisson(clean / K) × K  +  N(0, sigma_r²)
```

**Single-frame use** (new ISO drawn each call):

```python
gen = CameraNoiseGenerator(iso_range=(100, 6400))
noisy, clean, sigma_map = gen(clean_frame)
```

**Clip-consistent use** — same ISO and row-banding fixed for the entire clip, matching real camera behaviour where the ISO setting does not change between frames:

```python
per_frame = gen.for_clip()          # call once per clip
noisy_frames = [per_frame(f) for f in clean_clip]
```

`for_clip()` returns a `_ClipNoiseApplier` that generates row-banding fixed-pattern noise on the first call and reuses it identically for every subsequent frame in the clip.

**ISO → noise level reference** (default calibration, `iso_ref=1600`):

| ISO | K (shot) | σ_read |
|---|---|---|
| 200 | 0.0015 | 0.0018 |
| 800 | 0.006 | 0.0035 |
| 1600 | 0.012 | 0.005 |
| 6400 | 0.048 | 0.01 |
| 12800 | 0.096 | 0.014 |

Use `for_clip()` whenever training the temporal model so the noise character is consistent across frames — independent ISO resampling per frame causes the noise level to jump between frames, which the model cannot learn to exploit.

---

### Temporal model training

The `--model temporal` flag trains the 3-frame `NAFNetTemporal` model by default.
Training directories must be **sequence roots** whose immediate subdirectories
are frame sequences (one folder per clip):

```text
train_clean/
  scene_001/
    frame_0001.png
    frame_0002.png
    ...
  scene_002/
    ...
train_noisy/
  scene_001/
    frame_0001.png
    ...
```

```bash
# Temporal — paired sequences
uv run python training.py \
    --model temporal \
    --paired-clean /path/to/train_clean \
    --paired-noisy /path/to/train_noisy \
    --val-clean /path/to/val_clean \
    --val-noisy /path/to/val_noisy \
    --val-crop-mode grid \
    --val-grid-size 2 \
    --output checkpoints/temporal_paired \
    --epochs 300

# Temporal — random window sampling (faster epochs on long sequences)
uv run python training.py \
    --model temporal \
    --paired-clean /path/to/train_clean \
    --paired-noisy /path/to/train_noisy \
    --random-temporal-windows \
    --windows-per-sequence 4 \
    --output checkpoints/temporal_random \
    --epochs 300
```

With `--random-temporal-windows`, each sequence contributes
`--windows-per-sequence` randomly selected temporal windows per epoch instead
of all possible sliding windows.

### Spatial training on sequence folder structures

If your clean data is organised as frame sequences (subdirectories of numbered
images) rather than a flat folder of individual images, use
`--frames-per-sequence N` to control how many frames are sampled per sequence
per epoch.

```text
sequences/
  scene_001/  frame_0001.png … frame_0090.png
  scene_002/  frame_0001.png … frame_0060.png
  scene_003/  …
```

```bash
# Pick 10 evenly spread frames from each scene — first, spread, last
uv run python training.py \
    --model spatial \
    --data /path/to/sequences \
    --frames-per-sequence 10 \
    --output checkpoints/spatial \
    --epochs 300
```

Frame selection is deterministic — evenly spread indices using integer linspace,
always including the first and last frame:

| Sequence length | `--frames-per-sequence` | Selected indices |
|---|---|---|
| 90 | 3 | 0, 44, 89 |
| 90 | 5 | 0, 22, 44, 67, 89 |
| 90 | 10 | 0, 10, 20, 30, 40, 50, 60, 70, 80, 89 |
| 3 | 10 | 0, 1, 2 (count capped at sequence length) |

**Flat directory behaviour:** `--frames-per-sequence` only applies to
subdirectory-structured roots.  If you pass a flat directory (images at the
top level, no subdirectories), the flag is ignored and all images are used —
a warning is printed to tell you:

```
UserWarning: --frames-per-sequence has no effect on flat directory /path/to/data
(no sequence subdirectories found). Using all 4200 images.
```

`--frames-per-sequence` controls training only.  Use `--val-frames-per-sequence`
to apply the same spread selection to the validation dataset independently:

```bash
uv run python training.py \
    --model spatial \
    --data /path/to/sequences \
    --frames-per-sequence 10 \
    --val-data /path/to/val_sequences \
    --val-frames-per-sequence 3 \
    --output checkpoints/spatial \
    --epochs 300
```

`--frames-per-sequence` and `--val-frames-per-sequence` are not available for
`--model temporal` — use `--windows-per-sequence` and
`--val-windows-per-sequence` instead.

### Two-stage temporal training

Training `NAFNetTemporal` end-to-end from scratch is less efficient than first
building a strong spatial denoiser, then teaching the temporal components on top
of it.  The `NAFNet` and `NAFNetTemporal` spatial backbone tensors are
architecturally identical, so weights transfer directly.

**Stage 1 — train the spatial model:**

```bash
uv run python training.py \
    --model spatial \
    --data /path/to/clean/images \
    --output checkpoints/spatial \
    --epochs 300
```

**Stage 2 — load spatial weights, train temporal components only:**

`--spatial-weights` transfers the spatial backbone from the
`NAFNet` checkpoint.  `--freeze-spatial` freezes those layers so only
`temporal_mix` (and `offset_heads` when `use_warp=True`) receive gradients.

```bash
uv run python training.py \
    --model temporal \
    --data /path/to/sequences \
    --spatial-weights checkpoints/spatial/best.pth \
    --freeze-spatial \
    --output checkpoints/temporal_stage2 \
    --epochs 100
```

The startup log prints how many tensors were transferred and the trainable /
frozen parameter counts, so you can confirm the freeze is applied correctly:

```
Loaded 110 spatial weight tensors from checkpoints/spatial/best.pth
Spatial layers frozen.  Trainable params: 174,560  Frozen: 7,849,667
```

**Stage 3 — fine-tune all layers jointly at a lower learning rate:**

`--spatial-weights` warm-starts the backbone; omitting `--freeze-spatial`
leaves all parameters trainable.  Use `--resume` to continue from the stage 2
checkpoint.

```bash
uv run python training.py \
    --model temporal \
    --data /path/to/sequences \
    --spatial-weights checkpoints/spatial/best.pth \
    --resume checkpoints/temporal_stage2/best.pth \
    --lr 5e-5 \
    --output checkpoints/temporal_stage3 \
    --epochs 100
```

**Why this works:**  
During stage 2 the backbone already produces high-quality per-frame features.
`temporal_mix` has a clear learning signal from day one — it only needs to learn
how much temporal context to blend in.  Training from scratch forces the encoder
and temporal modules to co-adapt simultaneously, which is slower and less stable.

### Checkpoints

All checkpoints are written under `--output` (default: `checkpoints/run`):

| File | Written when |
|---|---|
| `final.pth` | End of every run |
| `best.pth` | When validation PSNR improves (requires `--val-*`) |
| `epoch_XXXX.pth` | Every 50 epochs (configurable via `--checkpoint-every`) |
| `runs/` | TensorBoard logs (`train/loss`, `train/psnr`, `val/psnr`, `train/lr`) |

Resume training from a checkpoint:

```bash
uv run python training.py \
    --data /path/to/clean/images \
    --resume checkpoints/run/epoch_0050.pth \
    --epochs 300
```

---

## Noise profiling (from dark frames)

`noise_profiler.py` extracts a camera noise model from dark frames (shot with
the lens cap on, no light, at the target ISO).  Two output modes:

### Mode 1 — parametric profile (for `RealRAWNoiseGenerator`)

Extracts read noise `sigma_r` and optionally Poisson gain `K` from flat-field
frames.  The profile is saved as a JSON file referenced at training time via
`--noise-profile`.

```bash
uv run python noise_profiler.py \
    --dark dark_frames/*.png \
    --flat flat_frames/*.png \      # optional — needed for K (shot noise gain)
    --output profiles/camera_iso1600.json
```

The JSON includes `sigma_r` (scalar read noise), `K` (Poisson gain, if flat
frames were provided), and a path to a sidecar `.npz` containing the spatial
mean dark frame.

### Mode 2 — patch pool (for `RealNoiseInjectionGenerator`)

Subtracts the temporal mean from each dark frame to isolate random temporal
noise, then tiles the residuals into a patch pool saved as an `.npz` file.
Referenced at training time via `--patch-pool`.

```bash
uv run python noise_profiler.py \
    --dark dark_frames/*.png \
    --save-patches pools/camera_iso1600.npz \
    --patch-size 128
```

The `.npz` stores zero-mean noise residuals of shape `(N, H, W, C)`.  During
training, random patches are sampled and added directly to clean images,
capturing authentic camera noise structure (banding, hot pixels, chroma noise)
that parametric models miss.

---

## Noise visualisation

`tests/visualise_noise.py` renders PNG diagnostic grids so you can inspect how
each noise type affects images, what paired dataset samples look like, and how
the sigma map relates to the actual noise.

```bash
cd training

# Default — writes PNGs to tests/fixtures/noise_diagnostics/
uv run python ../tests/visualise_noise.py --no-open

# Custom output directory
uv run python ../tests/visualise_noise.py --out-dir /tmp/noise_check

# With a real patch pool and noise profile
uv run python ../tests/visualise_noise.py \
    --patch-pool pools/camera_iso1600.npz \
    --noise-profile profiles/camera_iso1600.json
```

Output files:

| File | Contents |
|---|---|
| `generator_<name>.png` | One grid per noise generator — Clean \| Noisy \| Residual×5 \| Sigma map for all 5 sample images |
| `noise_types_overview.png` | Master grid — all generators × all images side by side |
| `paired_dataset.png` | `PairedPatchDataset` samples — clean \| noisy \| residual \| sigma |
| `patch_pool_residuals.png` | Raw noise patches from the `.npz` pool (amplified ×10 for visibility) |
| `sigma_comparison.png` | Sigma maps from all generators on one image, side by side |
| `temporal_clip_consistency.png` | Per-frame vs clip-consistent `CameraNoiseGenerator` across 5 frames — rows show noisy frames, frame-to-frame \|Δ\| amplified ×8, and noise residuals; last column shows mean \|Δ\| so you can quantify the consistency improvement from `for_clip()` |

When a real `--patch-pool` or `--noise-profile` is not provided, the script
creates synthetic stand-ins automatically so the full set of output PNGs is
always produced.

The same diagnostic is exercised by `test_noise_visual.py` (see Tests section
above), which calls `run()` in a temporary directory and asserts every expected
file is produced with non-zero size.

### Noise preview CLI

`tests/noise_preview.py` takes **your own image** and produces a generator × noise-level grid for rapid visual assessment — useful before committing to a training run.

```bash
cd training

# All generators, 4 levels each — output saved next to input image
uv run python ../tests/noise_preview.py --image /path/to/photo.jpg

# Specific generators and custom ISO sweep
uv run python ../tests/noise_preview.py \
    --image /path/to/photo.jpg \
    --generators gaussian camera \
    --iso-min 200 --iso-max 12800 \
    --n-levels 6

# Include temporal clip strip (per-frame vs clip-consistent noise)
uv run python ../tests/noise_preview.py \
    --image /path/to/photo.jpg \
    --temporal --n-frames 5

# Full control
uv run python ../tests/noise_preview.py \
    --image /path/to/photo.jpg \
    --generators gaussian poisson camera mixed \
    --n-levels 4 \
    --sigma-min 0.01 --sigma-max 0.3 \
    --iso-min 100 --iso-max 6400 \
    --amplify 8 \
    --out /tmp/preview.png
```

Output: `<image_stem>_noise_preview.png` (generator × level grid) and optionally `<image_stem>_temporal.png` (temporal consistency strip).

Grid layout — rows = generator × level, columns = Clean | Noisy | Residual×N | Sigma map.

Temporal strip layout — four rows: per-frame noisy, per-frame |Δ|, clip noisy, clip |Δ|. The last column shows the mean absolute frame-to-frame difference `Δ̄`; a well-configured `for_clip()` should show visibly smaller `Δ̄` than per-frame resampling.

---

## Export trained weights

```bash
uv run python export.py \
    --checkpoint checkpoints/spatial_standard/best.pth \
    --model spatial \
    --size standard \
    --output weights/spatial_standard \
    --dtype float16 \
    --verify
```

`--verify` runs a Python forward pass after export and prints the max absolute
difference between the original and re-loaded model.  Expected max diff < 0.001
for FP16 export.

Weight format: `manifest.json` + flat `.bin` files (little-endian, NCHW).
Conv weights: FP16.  BN gamma, beta, running_mean, running_var: always FP32.

---

## Inference (Python)

```bash
cd training

# Single image
uv run python infer.py \
    --checkpoint checkpoints/spatial_standard/best.pth \
    --input photo.png \
    --output photo_denoised.png

# Directory of images, with PSNR/SSIM against clean reference
uv run python infer.py \
    --checkpoint checkpoints/spatial_standard/best.pth \
    --input noisy_images/ \
    --clean clean_images/ \
    --output denoised/

# Tiled inference for large images
uv run python infer.py \
    --checkpoint checkpoints/spatial_standard/best.pth \
    --input photo_4k.png \
    --tile 512 \
    --output photo_4k_denoised.png

# Temporal model — denoises every frame using a 3-frame window
# Temporal flip averaging is enabled by default (2× inference, improves consistency)
uv run python infer.py \
    --checkpoint checkpoints/temporal/best.pth \
    --input noisy_frames/ \
    --output denoised_frames/

# Disable temporal flip for faster inference (slight flicker increase)
uv run python infer.py \
    --checkpoint checkpoints/temporal/best.pth \
    --input noisy_frames/ \
    --output denoised_frames/ \
    --no-temporal-flip
```

**Temporal flip averaging:** for temporal models, `infer.py` runs the sequence forward and then in reverse (time-flipped), then averages both predictions for each frame.  Because the two passes see different temporal context orderings, per-frame noise variance drops and temporal consistency improves.  The cost is 2× inference time.  Disable with `--no-temporal-flip`.

---

## Inference (CLI)

```bash
# Single image
./build/denoiser \
    --model weights/spatial_standard/manifest.json \
    --input photo.png \
    --output photo_denoised.png

# Image directory
./build/denoiser \
    --model weights/spatial_standard/manifest.json \
    --input frames/ \
    --output frames_denoised/

# Video (requires ffmpeg on PATH)
./build/denoiser \
    --model weights/spatial_standard/manifest.json \
    --input clip.mp4 \
    --output clip_denoised.mp4

# EXR (HDR, linear values preserved)
./build/denoiser \
    --model weights/spatial_standard/manifest.json \
    --input render.exr \
    --output render_denoised.exr
```

CLI options:

```
--model  PATH   manifest.json (required)
--input  PATH   PNG/JPG/EXR, directory, or MP4/MOV
--output PATH   output path (default: <input>_denoised.<ext>)
--mode   MODE   spatial | temporal (default: spatial)
--frames N      temporal window size (default: 3)
--device N      CUDA device index (default: 0)
--prefetch      pre-upload all weights before first inference
```

---

## Architecture

### NAFNet — recommended for EXR / HDR data

Single-frame spatial denoiser based on NAFNet (ECCV 2022, Chen et al.).

Key design choices:

| Component | NAFNet |
|---|---|
| Normalisation | `LayerNorm2d` (per image, per channel) |
| Activation | `SimpleGate` (x₁ × x₂) — no dead neurons |
| Skip connections | additive (`x + skip`) — no extra merge conv |
| Downsample | strided `Conv2d` — fully learnable |
| Upsample | `PixelShuffle` — artefact-free |

`LayerNorm2d` normalises each channel independently **per image**, making it immune to batch-size effects and HDR value range variation — the main reason NAFNet converges faster on EXR render data.

Block-level `β` / `γ` scalars are zero-initialised so every NAFBlock is exact identity at epoch 0; no LR warm-up required.

**Sizes** (configured via `NAFNetConfig`):

| Preset | `base_channels` | Block counts (enc / mid / dec) | Params |
|---|---|---|---|
| `NAFNetConfig.tiny()` | 16 | (1,1,1) / 1 / (1,1,1) | ~0.4 M |
| `NAFNetConfig.small()` | 32 | (1,1,1,1) / 1 / (1,1,1,1) | ~7 M |
| `NAFNetConfig.standard()` | 32 | (1,1,1,28) / 1 / (1,1,1,1) | ~24 M |
| `NAFNetConfig.wide()` | 64 | (1,1,1,28) / 1 / (1,1,1,1) | ~67 M |
| custom (`--naf-base C`) | C | standard block counts | scales with C² |

Train with:
```bash
uv run python training.py \
    --model spatial \
    --data /path/to/sequences \
    --frames-per-sequence 8 \
    --output checkpoints/spatial_naf \
    --epochs 300
```

---

### NAFNetTemporal (multi-frame)

Shared NAFNet encoder applied to all T frames as a batch.
At each encoder scale, neighbour features are averaged and concatenated with the reference frame's features; a 1×1 conv learns how much to pull from the temporal context.
Reference frame = centre frame (index `num_frames // 2`).
No learned warp by default — Monte Carlo render noise is pixel-independent across frames and averages out naturally without warping.

**Optional learned warp** (`--use-warp`):
Each neighbour feature map is warped to the reference before mixing using a per-level MobileNet-style offset head:

```
DW-Conv 3×3 (2C groups)  — spatial context, low cost
PW-Conv 1×1  (2C → C)    — cross-channel mixing
ReLU
PW-Conv 1×1  (C → 2)     — dense (dx, dy) displacement, zero-init
```

The displacement field is applied via bilinear `grid_sample`.  Only the final projection is zero-initialised so warps start as identity at epoch 0; intermediate layers use Kaiming init for stable gradient flow.  Use this for real video with significant camera or object motion; leave disabled for render sequences where Monte Carlo noise is pixel-independent.

At epoch 0 both NAFNet models are exact identity via zero-init ending conv and temporal/reference-preserving initialisation.

### C++ weight format

`manifest.json` + flat `.bin` files (little-endian, NCHW).  
Conv weights: FP16. BN stats: always FP32.

---

## Performance targets (RTX 3090)

| Task | Config | Target |
|---|---|---|
| Spatial 1080p | standard | < 100 ms / frame |
| Spatial 4K tiled | standard | < 400 ms / frame |
| Temporal 1080p | standard | < 300 ms / frame |
| C++ vs Python parity | any | max pixel diff < 0.005 |
| AWGN σ=25, 512×512 | standard trained | PSNR > 31 dB |

---

## Changelog

Generated from conventional commits via `git-cliff`:
```bash
git-cliff --output CHANGELOG.md
```
