#pragma once

// Helpers for __half (FP16) arithmetic in device code.
// CUDA intrinsics for half-precision are available from sm_53 onward.
// For Ampere (sm_80/86) the __half2 builtins also exercise Tensor Cores
// when used inside WMMA-aware code.

#include <cuda_fp16.h>

namespace denoiser::kernels {

// ---------------------------------------------------------------------------
// Scalar conversions
// ---------------------------------------------------------------------------

__device__ __forceinline__ float half_to_float(__half h) {
    return __half2float(h);
}

__device__ __forceinline__ __half float_to_half(float f) {
    return __float2half(f);
}

// ---------------------------------------------------------------------------
// Clamping
// ---------------------------------------------------------------------------

__device__ __forceinline__ __half clamp_half(__half val, float lo, float hi) {
    float f = __half2float(val);
    f = f < lo ? lo : (f > hi ? hi : f);
    return __float2half(f);
}

// ---------------------------------------------------------------------------
// Fused multiply-add in FP32 (prevents FP16 precision loss in accumulation)
// ---------------------------------------------------------------------------

// Computes: (a * scale + shift) with FP32 precision, returns FP16
__device__ __forceinline__ __half fma_fp32_to_half(__half a, float scale, float shift) {
    return __float2half(__half2float(a) * scale + shift);
}

// ---------------------------------------------------------------------------
// Element-wise subtraction: out[i] = clamp(a[i] - b[i], 0, 1)
// Used in the residual head: denoised = input - predicted_noise
// ---------------------------------------------------------------------------

__global__ void subtract_clamp_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ out,
    int64_t n
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float val = __half2float(a[idx]) - __half2float(b[idx]);
    out[idx] = __float2half(val < 0.f ? 0.f : (val > 1.f ? 1.f : val));
}

} // namespace denoiser::kernels
