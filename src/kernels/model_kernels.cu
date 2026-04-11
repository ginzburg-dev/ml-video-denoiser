#include "denoiser/kernels/model_kernels.h"

namespace denoiser::kernels {

namespace {

// ---------------------------------------------------------------------------
// Reflect-pad kernel
// ---------------------------------------------------------------------------
// Fills a (N, C, H_out, W_out) output from a (N, C, H_in, W_in) input.
// Positions inside [0, H_in) × [0, W_in) are copied directly.
// Positions in the padded region are reflected:
//   h >= H_in  →  h_src = 2*(H_in-1) - h
//   w >= W_in  →  w_src = 2*(W_in-1) - w
__global__ void reflect_pad_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int N, int C, int H_in, int W_in, int H_out, int W_out
) {
    const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * C * H_out * W_out;
    if (idx >= total) return;

    const int w = static_cast<int>(idx % W_out);
    const int h = static_cast<int>((idx / W_out) % H_out);
    const int c = static_cast<int>((idx / (static_cast<int64_t>(H_out) * W_out)) % C);
    const int n = static_cast<int>(idx / (static_cast<int64_t>(C) * H_out * W_out));

    const int h_src = h < H_in ? h : 2 * (H_in - 1) - h;
    const int w_src = w < W_in ? w : 2 * (W_in - 1) - w;

    output[idx] = input[(static_cast<int64_t>(n) * C + c) * H_in * W_in
                        + static_cast<int64_t>(h_src) * W_in + w_src];
}

// ---------------------------------------------------------------------------
// Crop kernel
// ---------------------------------------------------------------------------
// Copies the top-left (H_out × W_out) region of a (H_in × W_in) source.
__global__ void crop_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int N, int C, int H_in, int W_in, int H_out, int W_out
) {
    const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * C * H_out * W_out;
    if (idx >= total) return;

    const int w = static_cast<int>(idx % W_out);
    const int h = static_cast<int>((idx / W_out) % H_out);
    const int c = static_cast<int>((idx / (static_cast<int64_t>(H_out) * W_out)) % C);
    const int n = static_cast<int>(idx / (static_cast<int64_t>(C) * H_out * W_out));

    output[idx] = input[(static_cast<int64_t>(n) * C + c) * H_in * W_in
                        + static_cast<int64_t>(h) * W_in + w];
}

// ---------------------------------------------------------------------------
// Subtract + clamp kernel
// ---------------------------------------------------------------------------
// out[i] = clamp(a[i] - b[i], 0, 1)  — residual denoising final step
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

} // anonymous namespace

// ---------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------

void launch_reflect_pad(
    const __half* input,
    __half* output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    cudaStream_t stream
) {
    const int64_t total = static_cast<int64_t>(N) * C * H_out * W_out;
    constexpr int kBlock = 256;
    const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
    reflect_pad_kernel<<<blocks, kBlock, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out);
}

void launch_crop(
    const __half* input,
    __half* output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    cudaStream_t stream
) {
    const int64_t total = static_cast<int64_t>(N) * C * H_out * W_out;
    constexpr int kBlock = 256;
    const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
    crop_kernel<<<blocks, kBlock, 0, stream>>>(
        input, output, N, C, H_in, W_in, H_out, W_out);
}

void launch_subtract_clamp(
    const __half* a,
    const __half* b,
    __half* out,
    int64_t n,
    cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
    subtract_clamp_kernel<<<blocks, kBlock, 0, stream>>>(a, b, out, n);
}

} // namespace denoiser::kernels
