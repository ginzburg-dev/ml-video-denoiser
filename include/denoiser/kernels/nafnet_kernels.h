#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace denoiser::kernels {

void launch_extract_frame(
    const __half* clip,
    __half* frame,
    int N, int T, int C, int H, int W, int frame_idx,
    cudaStream_t stream
);

void launch_select_prefix_channels(
    const __half* input,
    __half* output,
    int N, int C_in, int C_out, int H, int W,
    cudaStream_t stream
);

void launch_layer_norm_affine(
    const __half* input,
    const void* weight,
    const void* bias,
    bool params_fp32,
    __half* output,
    int N, int C, int H, int W,
    cudaStream_t stream
);

void launch_simple_gate(
    const __half* input,
    __half* output,
    int N, int C, int H, int W,
    cudaStream_t stream
);

void launch_global_avg_pool(
    const __half* input,
    __half* output,
    int N, int C, int H, int W,
    cudaStream_t stream
);

void launch_mul_channelwise(
    const __half* input,
    const __half* scale,
    __half* output,
    int N, int C, int H, int W,
    cudaStream_t stream
);

void launch_add(
    const __half* a,
    const __half* b,
    __half* out,
    int64_t n,
    cudaStream_t stream
);

void launch_mul_scalar(
    const __half* input,
    float scalar,
    __half* output,
    int64_t n,
    cudaStream_t stream
);

void launch_scaled_add(
    const __half* identity,
    const __half* residual,
    const void* scale,
    bool scale_fp32,
    __half* output,
    int N, int C, int H, int W,
    cudaStream_t stream
);

void launch_pixel_shuffle2x(
    const __half* input,
    __half* output,
    int N, int C_in, int H, int W,
    cudaStream_t stream
);

void launch_warp_bilinear(
    const __half* feat,
    const __half* offset,
    __half* output,
    int N, int C, int H, int W,
    cudaStream_t stream
);

}  // namespace denoiser::kernels

