# Architecture

## Overview

The denoiser is split into two independent layers:

1. **Python training pipeline** (`training/`) — PyTorch models, dataset loading, noise synthesis, AMP training loop, weight export.
2. **C++ inference engine** (`src/`, `include/`, `cli/`) — raw CUDA/cuDNN inference from exported weights, zero framework dependencies at runtime.

---

## Models

### NEFResidual (single-frame spatial)

A 4-level encoder-decoder UNet with residual learning.  The network predicts the noise component; the output is obtained by subtracting that from the input:

```
output = clamp(input − predicted_noise, 0, 1)
```

**Encoder** (4 levels, channels `[64, 128, 256, 512]` standard):
```
EncoderBlock:
  ConvBnRelu(in_ch → ch, k=3, pad=1)
  ConvBnRelu(ch    → ch, k=3, pad=1)
  MaxPool2d(2, 2)      ← skip connection taken before pooling
```

**Bottleneck** (no pooling, channels `enc[-1] × 2 = 1024`):
```
ConvBnRelu(512 → 1024)
ConvBnRelu(1024 → 1024)
```

**Decoder** (4 levels, mirrors encoder in reverse):
```
DecoderBlock:
  bilinear_upsample(×2)
  concat(skip)                 ← skip from matching encoder level
  ConvBnRelu(in+skip → ch, k=3, pad=1)
  ConvBnRelu(ch      → ch, k=3, pad=1)
```

**Head**:
```
Conv2d(enc[0]=64 → 3, k=1)    ← 1×1 conv, no BN
```

**Padding**: input is reflect-padded to the nearest multiple of `2^num_levels = 16` before the forward pass and the output is cropped back.

**Parameter count** (standard config): ~31 M parameters.

---

### NEFTemporal (5-frame deformable)

Extends NEFResidual with inter-frame motion compensation using DCNv2-style deformable convolutions.

**Reference frame**: centre frame (index `T//2 = 2`).

```
1. Shared encoder  →  features for all T=5 frames (batch trick: B×T)

2. Per-level DeformableAlignment (weight-shared across neighbour positions):
   concat(ref_feat, nbr_feat)               → (N, 2C, H, W)
   offset_conv: Conv2d(2C → 2·dg·kH·kW)    → spatial offsets
   mask_conv:   Conv2d(2C →  dg·kH·kW)     → modulation masks (pre-sigmoid)
   DCNv2: bilinear_sample(nbr, offsets) × sigmoid(masks)

3. Per-level temporal fusion:
   concat(aligned_t0, …, aligned_t4)         → (N, T·C, H, W)
   Conv2d(T·C → C, k=1)                      → (N, C, H, W)

4. Bottleneck on reference frame's deepest features

5. Shared decoder (uses fused temporal skips)

6. Head + residual subtraction (same as NEFResidual)
```

**Deformable groups**: 8 (default).  Groups partition the channel dimension so each group learns independent offsets, enabling different motion patterns per feature group.

---

## C++ engine

### Tensor (`include/denoiser/tensor.h`)

RAII wrapper around `cudaMalloc`.  Move-only.  NCHW layout.

- `Tensor::empty(shape, dtype)` — uninitialised device allocation
- `Tensor::from_host(ptr, shape, dtype, stream)` — async H2D
- `tensor.to_host(ptr, stream)` — async D2H
- `tensor.slice_channels(start, end)` — non-owning view (zero copy)
- `tensor.make_cudnn_descriptor()` — caller frees with `cudnnDestroyTensorDescriptor`
- `DType`: `kFloat32` (4 bytes) | `kFloat16` (2 bytes)

### WeightStore (`include/denoiser/weight_loader.h`)

- `mmap()` all `.bin` files at construction (lazy OS page-in)
- Lazy H2D: first `get(name)` uploads and caches the device tensor
- `prefetch_all(stream)` uploads everything upfront (useful for benchmarking)
- Thread-safe reads after prefetch; upload path uses a mutex

### Layer implementations

| Layer | Backend | Notes |
|---|---|---|
| `Conv2dLayer` | `cudnnConvolutionForward` | `CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION` (Tensor Cores); per-(H,W) workspace cache |
| `BatchNorm2dLayer` | Custom CUDA kernel | Pre-computed `scale=γ/√(var+ε)`, `shift=β-scale·mean` stored as FP32; kernel: FP16→FP32 MAD→FP16 |
| `ReLULayer` | `cudnnActivationForward` | `CUDNN_ACTIVATION_RELU` |
| `UpsampleLayer` | Custom bilinear kernel | `align_corners=False` (matches PyTorch default); cuDNN path available via `-DUSE_CUDNN_RESAMPLE` if cuDNN ≥ 8.5 |
| `MaxPool2dLayer` | `cudnnPoolingForward` | `CUDNN_POOLING_MAX`, 2×2 stride 2 |
| `DeformConv2dLayer` | Custom im2col + `cublasHgemm` | Offsets computed in FP32 inside kernel; modulation mask sigmoid inline; Tensor Cores via `CUBLAS_TENSOR_OP_MATH` |

### CUDA kernels (`src/kernels/`)

| Kernel | Description |
|---|---|
| `bn_inference.cuh` | BN forward: FP16→FP32 MAD→FP16, per-channel scale/shift |
| `concat.cuh` | Channel-wise concat: `[N,C1,H,W] ++ [N,C2,H,W] → [N,C1+C2,H,W]` |
| `model_kernels.cu` | `reflect_pad`, `crop`, `subtract_clamp` (residual output) |
| `deform_im2col.cuh` | DCNv2: fractional bilinear sampling + modulation → im2col buffer |
| `fp16_utils.cuh` | Device helpers: `half_to_float`, `clamp_half`, `fma_fp32_to_half` |

### Memory layout (NEFResidual, standard, 1080p)

| Tensor | Shape | Size |
|---|---|---|
| Input | (1, 3, 1088, 1920) | ~12 MB FP16 |
| Enc level 0 skip | (1, 64, 1088, 1920) | ~253 MB FP16 |
| Enc level 1 skip | (1, 128, 544, 960) | ~126 MB FP16 |
| Enc level 2 skip | (1, 256, 272, 480) | ~63 MB FP16 |
| Enc level 3 skip | (1, 512, 136, 240) | ~32 MB FP16 |
| Bottleneck | (1, 1024, 68, 120) | ~16 MB FP16 |
| **Peak (skips alive)** | | **~495 MB** |

Skips are freed immediately after the decoder level consumes them, so peak is dominated by level 0 skip.  Use `--tile` (CLI) or the lite config (`[32,64,128,256]`) for GPUs with < 8 GB VRAM.

---

## I/O

| Format | Python | C++ |
|---|---|---|
| PNG / JPEG | `imageio` / `Pillow` | `stb_image` + `stb_image_write` |
| EXR (HDR) | `OpenEXR` + `Imath` | `tinyexr` (header-only) |
| MP4 / MOV | `ffmpeg-python` | Shell out to `ffmpeg` binary |

**LGPL note**: the C++ engine never links against `libavcodec`.  Video decode/encode is performed by invoking the `ffmpeg` binary via `popen()`.  Binary invocation carries no LGPL obligation.

---

## Noise synthesis (training only)

All noise is generated on-the-fly from clean images — the dataset stores only clean frames.

| Generator | Model | Use case |
|---|---|---|
| `GaussianNoiseGenerator` | σ ~ Uniform(σ_min, σ_max) | Baseline AWGN |
| `PoissonGaussianNoiseGenerator` | noisy = Poisson(x/K)·K + N(0,σ_r²) | RAW sensor simulation |
| `RealNoiseInjectionGenerator` | Zero-mean residuals from dark frames | Authentic camera noise structure |
| `RealRAWNoiseGenerator` | Calibrated (K, σ_r) from JSON profile | Generalization across ISOs |
| `MixedNoiseGenerator` | Weighted random selection | Default training mix |

Default training mix: 30% Gaussian + 30% Poisson-Gaussian + 25% RealInject + 15% RealRAW.
