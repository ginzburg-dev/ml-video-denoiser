#pragma once

// Host-callable CUDA kernel launchers shared by the model forward passes.
// These declarations use only host-side types so this header can be included
// from plain .cc files as well as .cu files.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace denoiser::kernels {

// ---------------------------------------------------------------------------
// Reflect-pad (right + bottom only — matches PyTorch _pad_to_multiple output)
// ---------------------------------------------------------------------------
// Fills output (N, C, H_out, W_out) from input (N, C, H_in, W_in) where
//   H_out = H_in + pad_h,  W_out = W_in + pad_w,  pad_h/pad_w >= 0.
// When pad_h == 0 and pad_w == 0 the operation is equivalent to memcpy.
// Reflect formula (align_corners=False / PyTorch convention):
//   h >= H_in  →  h_src = 2*(H_in-1) - h
//   w >= W_in  →  w_src = 2*(W_in-1) - w
void launch_reflect_pad(
    const __half* input,
    __half* output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    cudaStream_t stream
);

// ---------------------------------------------------------------------------
// Crop (strip padding added by reflect-pad)
// ---------------------------------------------------------------------------
// Extracts the top-left (N, C, H_out, W_out) sub-region of
// input (N, C, H_in, W_in) into a contiguous output buffer.
void launch_crop(
    const __half* input,
    __half* output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    cudaStream_t stream
);

// ---------------------------------------------------------------------------
// Residual subtraction: out[i] = clamp(a[i] - b[i], 0, 1)
// ---------------------------------------------------------------------------
// Legacy helper for subtraction-style residual heads.
void launch_subtract_clamp(
    const __half* a,
    const __half* b,
    __half* out,
    int64_t n,
    cudaStream_t stream
);

void launch_concat_channels(
    const __half* a,
    const __half* b,
    __half* output,
    int N, int C1, int C2, int H, int W,
    cudaStream_t stream
);

} // namespace denoiser::kernels
