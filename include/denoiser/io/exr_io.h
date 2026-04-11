#pragma once

#include "denoiser/tensor.h"
#include <string>

namespace denoiser::io {

// ---------------------------------------------------------------------------
// OpenEXR I/O via tinyexr (header-only, git submodule)
// ---------------------------------------------------------------------------

// Load a half-float or full-float EXR from *path*.
//
// Returns an FP16 NCHW tensor with shape (1, C, H, W), C = 3 (RGB) or 1.
// HDR values are NOT normalised — EXR linear scene values are preserved.
// Throws std::runtime_error on read failure.
Tensor load_exr(const std::string& path, cudaStream_t stream = nullptr);

// Save an FP16 NCHW tensor (1, C, H, W) to an EXR file at *path*.
//
// Written as 16-bit half-float EXR (HALF precision).  No tone-mapping.
// Throws std::runtime_error on write failure.
void save_exr(const Tensor& tensor, const std::string& path,
              cudaStream_t stream = nullptr);

} // namespace denoiser::io
