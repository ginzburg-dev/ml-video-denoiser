# Build Guide

## Prerequisites

| Dependency | Minimum | Notes |
|---|---|---|
| CUDA Toolkit | 11.8 | Includes cuBLAS |
| cuDNN | 8.0 | 8.5+ enables the cuDNN upsample path |
| CMake | 3.22 | |
| C++ compiler | GCC 12 / Clang 16 | C++23 required for `.cc` files |
| Python | 3.11 | Training pipeline only |
| uv | latest | `pip install uv` or `curl -Lsf https://astral.sh/uv/install.sh \| sh` |
| ffmpeg binary | any modern | Runtime only — must be on `PATH` for video I/O |
| LibRaw | 0.21+ | Optional — only for camera RAW reading in the CLI |

---

## 1. Clone

```bash
git clone --recurse-submodules https://github.com/<org>/ml-video-denoiser.git
cd ml-video-denoiser
```

If you already cloned without `--recurse-submodules`:

```bash
git submodule update --init --recursive
```

Submodules used:
- `third_party/nlohmann_json` — JSON manifest parsing
- `third_party/stb` — PNG/JPEG I/O
- `third_party/tinyexr` — EXR I/O
- `third_party/googletest` — C++ unit tests

---

## 2. Python training environment

```bash
cd training
uv sync          # creates .venv and installs all locked dependencies
uv run pytest tests/ -v   # verify: 88 tests should pass
```

To add a new dependency:
```bash
uv add <package>           # updates pyproject.toml + uv.lock
```

---

## 3. C++ engine

### Basic build (Release)

```bash
cmake -B build \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES="80;86"
cmake --build build -j$(nproc)
```

Adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU:

| GPU family | Architecture |
|---|---|
| Turing (RTX 20xx, T4) | `75` |
| Ampere A100 | `80` |
| Ampere RTX 30xx | `86` |
| Ada Lovelace RTX 40xx | `89` |
| Hopper H100 | `90` |

### CMake options

| Option | Default | Description |
|---|---|---|
| `CMAKE_BUILD_TYPE` | `Release` | `Debug` / `RelWithDebInfo` also supported |
| `CMAKE_CUDA_ARCHITECTURES` | `75;80;86` | Semicolon-separated sm values |
| `DENOISER_BUILD_TESTS` | `ON` | Build the gtest suite |
| `USE_LIBRAW` | `OFF` | Enable LibRaw for camera RAW reading |
| `USE_CUDNN_RESAMPLE` | auto | Forced ON/OFF if auto-detection fails |

### cuDNN location

If cuDNN is not found automatically, set:

```bash
cmake -B build \
    -DCUDNN_ROOT_DIR=/path/to/cudnn \
    ...
```

The `cmake/FindCUDNN.cmake` module searches `CUDNN_ROOT_DIR`, common system paths, and `CUDA_TOOLKIT_ROOT_DIR`.

### LibRaw (optional)

```bash
cmake -B build -DUSE_LIBRAW=ON ...
# Requires: apt install libraw-dev  or  brew install libraw
```

### Debug build

```bash
cmake -B build-debug \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build build-debug -j$(nproc)
```

---

## 4. Running tests

### C++ unit tests

```bash
# All tests (skips GPU tests if no CUDA device)
./build/tests/denoiser_tests

# Specific suite
./build/tests/denoiser_tests --gtest_filter="TensorTest.*"

# NAFNet parity tests (need fixture — see step 5)
./build/tests/denoiser_tests \
    --gtest_also_run_disabled_tests \
    --gtest_filter="NAFNetForwardTest.*"
```

### Generate test fixture

```bash
cd training
uv run python ../tests/gen_fixtures.py
# Output: tests/fixtures/tiny_unet/
```

### Python integration tests

```bash
cd training
uv run pytest ../tests/test_integration.py -v
```

This runs:
- Python export round-trip tests
- Python NAFNet correctness tests
- C++ gtest binary (if built)
- Python ↔ C++ parity against the fixture
- CLI smoke tests (if built)
- PSNR regression (if a trained checkpoint exists)

### Full test suite

```bash
cd training
uv run pytest tests/ ../tests/test_integration.py -v
```

---

## 5. Common issues

### `cuDNN not found`

```
CMake Error: Could not find CUDNN
```

Set `CUDNN_ROOT_DIR` explicitly:
```bash
cmake -B build -DCUDNN_ROOT_DIR=/usr/local/cudnn ...
```

Or install via the CUDA installer (includes cuDNN in newer Toolkit versions).

### `nvcc: error: unrecognized option '--std=c++23'`

NVCC does not support C++23 for device code.  The project uses `CMAKE_CXX_STANDARD 23` for host `.cc` files and `CMAKE_CUDA_STANDARD 20` for device `.cu` files.  If your NVCC does not support C++20, upgrade to CUDA Toolkit ≥ 11.8.

### `CUBLAS_STATUS_NOT_INITIALIZED`

Typically caused by a CUDA device/context not being initialised before calling cuBLAS.  Make sure `cudaSetDevice()` is called before constructing any layer.  The CLI does this at startup.

### `stb_image.h: multiple definition`

`STB_IMAGE_IMPLEMENTATION` and `STB_IMAGE_WRITE_IMPLEMENTATION` must be defined in **exactly one** translation unit.  They are defined in `src/io/image_io.cc`.  Do not include `stb_image.h` with those defines elsewhere.

### `tinyexr.h: multiple definition of SaveEXRImageToFile`

Same single-definition-rule issue.  `TINYEXR_IMPLEMENTATION` is defined in `src/io/exr_io.cc` only.

### `ffmpeg not found` at runtime

Install ffmpeg and ensure it is on `PATH`:
```bash
# Linux
apt install ffmpeg
# macOS
brew install ffmpeg
```

The CLI checks for `ffmpeg` / `ffprobe` before attempting any video operation and throws a descriptive error if they are absent.

---

## 6. Install

```bash
cmake --install build --prefix /usr/local
# Installs: /usr/local/bin/denoiser
#           /usr/local/include/denoiser/...
```
