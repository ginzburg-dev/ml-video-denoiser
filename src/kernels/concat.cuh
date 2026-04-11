#pragma once

// Channel-wise concatenation kernel for FP16 NCHW tensors.
//
// Combines two tensors along the channel dimension in a single pass,
// avoiding two separate memory copies.
//
//   output[:, 0:C1, :, :] = a
//   output[:, C1:C1+C2, :, :] = b
//
// Both inputs must have the same N, H, W.

#include <cuda_fp16.h>
#include <cstdint>

namespace denoiser::kernels {

// Concatenate *a* (N, C1, H, W) and *b* (N, C2, H, W) into *output* (N, C1+C2, H, W).
__global__ void concat_channels_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ output,
    int N, int C1, int C2, int H, int W
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * (C1 + C2) * H * W;
    if (idx >= total) return;

    const int w = static_cast<int>(idx % W);
    const int h = static_cast<int>((idx / W) % H);
    const int c = static_cast<int>((idx / (static_cast<int64_t>(H) * W)) % (C1 + C2));
    const int n = static_cast<int>(idx / (static_cast<int64_t>(C1 + C2) * H * W));

    if (c < C1) {
        output[idx] = a[((static_cast<int64_t>(n) * C1 + c) * H + h) * W + w];
    } else {
        const int c2 = c - C1;
        output[idx] = b[((static_cast<int64_t>(n) * C2 + c2) * H + h) * W + w];
    }
}

// Host-callable launcher.
void launch_concat_channels(
    const __half* a,
    const __half* b,
    __half* output,
    int N, int C1, int C2, int H, int W,
    cudaStream_t stream
);

} // namespace denoiser::kernels
