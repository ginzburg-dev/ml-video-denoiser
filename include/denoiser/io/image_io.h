#pragma once

#include "denoiser/tensor.h"
#include <string>

namespace denoiser::io {

// ---------------------------------------------------------------------------
// PNG / JPEG image I/O via stb_image / stb_image_write
// ---------------------------------------------------------------------------

// Load an 8-bit PNG or JPEG from *path*.
//
// Returns an FP16 NCHW tensor with shape (1, C, H, W), values normalised to
// [0, 1].  C is 3 for RGB images (alpha channel is dropped if present).
// Throws std::runtime_error on read failure.
Tensor load_image(const std::string& path, cudaStream_t stream = nullptr);

// Save an FP16 NCHW tensor (1, C, H, W) or (1, 1, H, W) to *path*.
//
// Values are clamped to [0, 1] and scaled to uint8 [0, 255].
// Format is inferred from the file extension: .png → PNG, else JPEG.
// Throws std::runtime_error on write failure.
void save_image(const Tensor& tensor, const std::string& path,
                cudaStream_t stream = nullptr);

} // namespace denoiser::io
