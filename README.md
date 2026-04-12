# ML Video Denoiser

GPU-accelerated video / image denoiser.  
Python training (NEFResidual + NEFTemporal) → raw weight export → C++ CLI inference with no ONNX / LibTorch dependency.

---

## Repository layout

```
ml-video-denoiser/
├── training/           Python training pipeline (uv-managed)
│   ├── models.py       NEFResidual, NEFTemporal
│   ├── dataset.py      PatchDataset, VideoSequenceDataset
│   ├── noise_generators.py  Gaussian, Poisson-Gaussian, RealInject, RealRAW, Mixed
│   ├── noise_profiler.py    Dark-frame calibration → JSON profile / patch pool
│   ├── training.py     Training loop (AMP, AdamW, cosine LR)
│   ├── export.py       Weights → manifest.json + .bin files
│   ├── infer.py        Python inference + PSNR/SSIM evaluation
│   └── tests/          88 pytest tests
├── src/                C++ engine source
│   ├── tensor.cu       RAII CUDA tensor
│   ├── weight_loader.cc mmap + lazy H2D upload
│   ├── layers/         Conv2d, BatchNorm2d, ReLU, Upsample, MaxPool2d, DeformConv2d
│   ├── models/         nef_residual.cu, nef_temporal.cu
│   ├── io/             image_io.cc (stb), exr_io.cc (tinyexr), video_io.cc (ffmpeg shell)
│   └── kernels/        bn_inference, concat, model_kernels, deform_im2col
├── include/denoiser/   Public C++ headers
├── cli/main.cc         nef_denoise CLI
├── tests/              C++ gtests + Python integration tests
│   ├── gen_fixtures.py Generates tiny_unet fixture for parity tests
│   └── test_integration.py Phase 6 Python ↔ C++ parity tests
├── third_party/        nlohmann_json, stb, tinyexr, googletest (submodules)
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

# 5. Run the full test suite — 30+ tests, no GPU dataset needed
cd training && uv run pytest ../tests/test_integration.py -v
```

What the suite covers without any external data:

| Group | What it tests |
|---|---|
| `TestExportRoundtrip` | export.py produces correct manifest + .bin files |
| `TestNEFResidualPython` | Python model output shape, clamping, padding |
| `TestCppGtests` | All C++ unit tests (tensor, weight loader, conv, BN, UNet parity) |
| `TestPythonCppParity` | Python ↔ C++ pixel-level agreement |
| `TestCLI` | CLI binary: single image, directory, error handling |
| `TestTrainingSmokeTest` | 2-epoch train on bundled 128×128 PNGs — loss decreases, checkpoint saves, resume works |
| `TestPSNRRegression` | Random-init model produces no NaN/Inf *(PSNR threshold skipped until trained checkpoint exists)* |

The `TestTrainingSmokeTest` uses five synthetic PNG images committed to
`tests/fixtures/sample_images/` — generated once by `tests/gen_sample_images.py`.
They cover horizontal/vertical gradients, a checkerboard, colour patches, and a
sinusoidal texture, giving the training pipeline diverse spatial frequencies to
exercise on.

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
uv run pytest tests/ -v    # 88 tests — should all pass
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
    --gtest_filter="UNetForwardTest.*"
```

### 5. Integration tests (Python + C++)

```bash
cd training
uv run pytest ../tests/test_integration.py -v
```

The `TestTrainingSmokeTest` suite needs the bundled sample images.  They are
already committed to `tests/fixtures/sample_images/`, but if you need to
regenerate them:

```bash
cd training && uv run python ../tests/gen_sample_images.py
```

---

## Training

### Quick start (Gaussian AWGN)

```bash
cd training
uv run python training.py \
    --data /path/to/clean/images \
    --noise gaussian \
    --size standard \
    --epochs 300
```

### Mixed noise (recommended)

```bash
uv run python training.py \
    --data /path/to/clean/images \
    --noise mixed \
    --patch-pool pools/camera_iso1600.npz \
    --noise-profile profiles/camera_iso1600.json
```

### Temporal training with paired validation

```bash
uv run python training.py \
    --paired-clean /path/to/train_clean \
    --paired-noisy /path/to/train_noisy \
    --val-clean /path/to/val_clean \
    --val-noisy /path/to/val_noisy \
    --size standard \
    --model temporal \
    --epochs 300
```

For `--model temporal`, both training and validation directories must be
sequence roots whose immediate subdirectories are clips:

```text
val_clean/
  scene_001/
    frame_0001.exr
    frame_0002.exr
    ...
val_noisy/
  scene_001/
    frame_0001.exr
    frame_0002.exr
    ...
```

### Checkpoints

Training writes checkpoints under `--output` (default: `training/checkpoints/run`):

- `final.pth` at the end of a successful run
- `best.pth` when validation is enabled and validation PSNR improves
- `epoch_XXXX.pth` every 50 epochs
- `runs/` TensorBoard logs

### Noise profiling (from dark frames)

```bash
# Parametric profile (for RealRAWNoiseGenerator)
uv run python noise_profiler.py \
    --dark dark_frames/*.png \
    --flat flat_frames/*.png \
    --output profiles/camera_iso1600.json

# Patch pool (for RealNoiseInjectionGenerator)
uv run python noise_profiler.py \
    --dark dark_frames/*.png \
    --save-patches pools/camera_iso1600.npz
```

### Export trained weights

```bash
uv run python export.py \
    --checkpoint checkpoints/residual_standard/best.pth \
    --model residual \
    --size standard \
    --output weights/residual_standard \
    --dtype float16 \
    --verify
```

---

## Inference (CLI)

```bash
# Single image
./build/nef_denoise \
    --model weights/residual_standard/manifest.json \
    --input photo.png \
    --output photo_denoised.png

# Image directory
./build/nef_denoise \
    --model weights/residual_standard/manifest.json \
    --input frames/ \
    --output frames_denoised/

# Video (requires ffmpeg on PATH)
./build/nef_denoise \
    --model weights/residual_standard/manifest.json \
    --input clip.mp4 \
    --output clip_denoised.mp4

# EXR (HDR, linear values preserved)
./build/nef_denoise \
    --model weights/residual_standard/manifest.json \
    --input render.exr \
    --output render_denoised.exr
```

CLI options:

```
--model  PATH   manifest.json (required)
--input  PATH   PNG/JPG/EXR, directory, or MP4/MOV
--output PATH   output path (default: <input>_denoised.<ext>)
--mode   MODE   spatial | temporal (default: spatial)
--frames N      temporal window size (default: 5)
--device N      CUDA device index (default: 0)
--prefetch      pre-upload all weights before first inference
```

---

## Architecture

### NEFResidual (spatial)
4-level UNet (enc_channels `[64,128,256,512]` standard).  
Output = `clamp(input − predicted_noise, 0, 1)`.  
Auto reflect-pads to multiple of 2^num_levels.

### NEFTemporal (5-frame)
Shared encoder → per-level DCNv2 deformable alignment → temporal fusion → shared decoder.  
Reference frame = centre frame (index 2 of 5).

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
