#pragma once

// Fused BatchNorm inference kernel for FP16 activations.
//
// During inference BatchNorm reduces to:
//   out[c] = (in[c] - mean[c]) / sqrt(var[c] + eps) * gamma[c] + beta[c]
//          = in[c] * scale[c] + shift[c]
//
// where scale and shift are pre-computed once at layer construction:
//   scale[c] = gamma[c] / sqrt(var[c] + eps)
//   shift[c] = beta[c] - scale[c] * mean[c]
//
// This kernel reads FP16 input, promotes to FP32 for the multiply-add
// (preventing FP16 precision loss in the BN statistics), then writes FP16
// output.  scale[] and shift[] are FP32 device arrays.
//
// Layout: NCHW — channel index is (linear_idx / (H*W)) % C.

#include <cuda_fp16.h>
#include <cstdint>

namespace denoiser::kernels {

// Apply pre-computed scale + shift to an FP16 NCHW tensor.
//
// Args:
//   input   — FP16 device pointer, shape (N, C, H, W)
//   output  — FP16 device pointer, same shape (may alias input for in-place)
//   scale   — FP32 device pointer, shape (C,)
//   shift   — FP32 device pointer, shape (C,)
//   N, C, H, W — tensor dimensions
__global__ void bn_inference_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    const float* __restrict__ scale,
    const float* __restrict__ shift,
    int N, int C, int H, int W
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * C * H * W;
    if (idx >= total) return;

    // Derive channel index from linear index (NCHW)
    const int c = static_cast<int>((idx / (static_cast<int64_t>(H) * W)) % C);

    const float val = __half2float(input[idx]) * scale[c] + shift[c];
    output[idx] = __float2half(val);
}

// Host-callable launcher (declared here, defined in batchnorm2d.cu).
void launch_bn_inference(
    const __half* input,
    __half* output,
    const float* scale,
    const float* shift,
    int N, int C, int H, int W,
    cudaStream_t stream
);

} // namespace denoiser::kernels
