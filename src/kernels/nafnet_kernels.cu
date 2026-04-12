#include "denoiser/kernels/nafnet_kernels.h"

#include <cuda_fp16.h>

namespace denoiser::kernels {

namespace {

__device__ inline float read_param(const void* ptr, bool is_fp32, int idx) {
    if (is_fp32) {
        return static_cast<const float*>(ptr)[idx];
    }
    return __half2float(static_cast<const __half*>(ptr)[idx]);
}

__global__ void extract_frame_kernel(
    const __half* __restrict__ clip,
    __half* __restrict__ frame,
    int N, int T, int C, int H, int W, int frame_idx
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * C * H * W;
    if (idx >= total) return;

    const int w = static_cast<int>(idx % W);
    const int h = static_cast<int>((idx / W) % H);
    const int c = static_cast<int>((idx / (static_cast<int64_t>(H) * W)) % C);
    const int n = static_cast<int>(idx / (static_cast<int64_t>(C) * H * W));

    const int64_t src_idx =
        ((((static_cast<int64_t>(n) * T) + frame_idx) * C + c) * H + h) * W + w;
    frame[idx] = clip[src_idx];
}

__global__ void select_prefix_channels_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int N, int C_in, int C_out, int H, int W
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * C_out * H * W;
    if (idx >= total) return;

    const int w = static_cast<int>(idx % W);
    const int h = static_cast<int>((idx / W) % H);
    const int c = static_cast<int>((idx / (static_cast<int64_t>(H) * W)) % C_out);
    const int n = static_cast<int>(idx / (static_cast<int64_t>(C_out) * H * W));

    const int64_t src_idx =
        (((static_cast<int64_t>(n) * C_in) + c) * H + h) * W + w;
    output[idx] = input[src_idx];
}

__global__ void layer_norm_affine_kernel(
    const __half* __restrict__ input,
    const void* __restrict__ weight,
    const void* __restrict__ bias,
    bool params_fp32,
    __half* __restrict__ output,
    int N, int C, int H, int W
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * H * W;
    if (idx >= total) return;

    const int w = static_cast<int>(idx % W);
    const int h = static_cast<int>((idx / W) % H);
    const int n = static_cast<int>(idx / (static_cast<int64_t>(H) * W));

    const int64_t base = (static_cast<int64_t>(n) * C * H + h) * W + w;

    float mean = 0.0f;
    for (int c = 0; c < C; ++c) {
        mean += __half2float(input[base + static_cast<int64_t>(c) * H * W]);
    }
    mean /= static_cast<float>(C);

    float var = 0.0f;
    for (int c = 0; c < C; ++c) {
        const float v = __half2float(input[base + static_cast<int64_t>(c) * H * W]) - mean;
        var += v * v;
    }
    const float inv_std = rsqrtf(var / static_cast<float>(C) + 1e-6f);

    for (int c = 0; c < C; ++c) {
        const int64_t offset = base + static_cast<int64_t>(c) * H * W;
        const float norm = (__half2float(input[offset]) - mean) * inv_std;
        const float gamma = read_param(weight, params_fp32, c);
        const float beta = read_param(bias, params_fp32, c);
        output[offset] = __float2half(norm * gamma + beta);
    }
}

__global__ void simple_gate_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int N, int C, int H, int W
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * C * H * W;
    if (idx >= total) return;
    const int64_t plane = static_cast<int64_t>(H) * W;
    const int64_t half_stride = static_cast<int64_t>(C) * plane;
    output[idx] = __float2half(
        __half2float(input[idx]) * __half2float(input[idx + half_stride]));
}

__global__ void global_avg_pool_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int N, int C, int H, int W
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * C;
    if (idx >= total) return;

    const int c = static_cast<int>(idx % C);
    const int n = static_cast<int>(idx / C);
    const int64_t base = (static_cast<int64_t>(n) * C + c) * H * W;

    float sum = 0.0f;
    for (int i = 0; i < H * W; ++i) {
        sum += __half2float(input[base + i]);
    }
    output[idx] = __float2half(sum / static_cast<float>(H * W));
}

__global__ void mul_channelwise_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ scale,
    __half* __restrict__ output,
    int N, int C, int H, int W
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * C * H * W;
    if (idx >= total) return;
    const int c = static_cast<int>((idx / (static_cast<int64_t>(H) * W)) % C);
    const int n = static_cast<int>(idx / (static_cast<int64_t>(C) * H * W));
    const float s = __half2float(scale[static_cast<int64_t>(n) * C + c]);
    output[idx] = __float2half(__half2float(input[idx]) * s);
}

__global__ void add_kernel(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half* __restrict__ out,
    int64_t n
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx] = __float2half(__half2float(a[idx]) + __half2float(b[idx]));
}

__global__ void mul_scalar_kernel(
    const __half* __restrict__ input,
    float scalar,
    __half* __restrict__ output,
    int64_t n
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    output[idx] = __float2half(__half2float(input[idx]) * scalar);
}

__global__ void scaled_add_kernel(
    const __half* __restrict__ identity,
    const __half* __restrict__ residual,
    const void* __restrict__ scale,
    bool scale_fp32,
    __half* __restrict__ output,
    int N, int C, int H, int W
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * C * H * W;
    if (idx >= total) return;

    const int c = static_cast<int>((idx / (static_cast<int64_t>(H) * W)) % C);
    const float s = read_param(scale, scale_fp32, c);
    output[idx] = __float2half(__half2float(identity[idx]) + __half2float(residual[idx]) * s);
}

__global__ void pixel_shuffle2x_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int N, int C_in, int H, int W
) {
    const int C_out = C_in / 4;
    const int H_out = H * 2;
    const int W_out = W * 2;
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * C_out * H_out * W_out;
    if (idx >= total) return;

    const int w_out = static_cast<int>(idx % W_out);
    const int h_out = static_cast<int>((idx / W_out) % H_out);
    const int c_out = static_cast<int>((idx / (static_cast<int64_t>(H_out) * W_out)) % C_out);
    const int n = static_cast<int>(idx / (static_cast<int64_t>(C_out) * H_out * W_out));

    const int r_h = h_out & 1;
    const int r_w = w_out & 1;
    const int h_in = h_out >> 1;
    const int w_in = w_out >> 1;
    const int c_in = c_out * 4 + r_h * 2 + r_w;

    const int64_t src_idx =
        (((static_cast<int64_t>(n) * C_in + c_in) * H + h_in) * W + w_in);
    output[idx] = input[src_idx];
}

__device__ inline float sample_border(
    const __half* feat, int N, int C, int H, int W,
    int n, int c, int y, int x
) {
    const int yy = max(0, min(y, H - 1));
    const int xx = max(0, min(x, W - 1));
    const int64_t idx =
        (((static_cast<int64_t>(n) * C + c) * H + yy) * W + xx);
    return __half2float(feat[idx]);
}

__global__ void warp_bilinear_kernel(
    const __half* __restrict__ feat,
    const __half* __restrict__ offset,
    __half* __restrict__ output,
    int N, int C, int H, int W
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * C * H * W;
    if (idx >= total) return;

    const int w = static_cast<int>(idx % W);
    const int h = static_cast<int>((idx / W) % H);
    const int c = static_cast<int>((idx / (static_cast<int64_t>(H) * W)) % C);
    const int n = static_cast<int>(idx / (static_cast<int64_t>(C) * H * W));

    const int64_t off_base =
        (((static_cast<int64_t>(n) * 2) * H + h) * W + w);
    const float dx = __half2float(offset[off_base]);
    const float dy = __half2float(offset[off_base + static_cast<int64_t>(H) * W]);

    const float x = static_cast<float>(w) + dx;
    const float y = static_cast<float>(h) + dy;

    const int x0 = static_cast<int>(floorf(x));
    const int y0 = static_cast<int>(floorf(y));
    const int x1 = x0 + 1;
    const int y1 = y0 + 1;

    const float wx = x - static_cast<float>(x0);
    const float wy = y - static_cast<float>(y0);

    const float v00 = sample_border(feat, N, C, H, W, n, c, y0, x0);
    const float v01 = sample_border(feat, N, C, H, W, n, c, y0, x1);
    const float v10 = sample_border(feat, N, C, H, W, n, c, y1, x0);
    const float v11 = sample_border(feat, N, C, H, W, n, c, y1, x1);

    const float val = v00 * (1.f - wy) * (1.f - wx)
                    + v01 * (1.f - wy) * wx
                    + v10 * wy * (1.f - wx)
                    + v11 * wy * wx;
    output[idx] = __float2half(val);
}

}  // namespace

void launch_extract_frame(
    const __half* clip,
    __half* frame,
    int N, int T, int C, int H, int W, int frame_idx,
    cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int64_t total = static_cast<int64_t>(N) * C * H * W;
    const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
    extract_frame_kernel<<<blocks, kBlock, 0, stream>>>(clip, frame, N, T, C, H, W, frame_idx);
}

void launch_select_prefix_channels(
    const __half* input,
    __half* output,
    int N, int C_in, int C_out, int H, int W,
    cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int64_t total = static_cast<int64_t>(N) * C_out * H * W;
    const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
    select_prefix_channels_kernel<<<blocks, kBlock, 0, stream>>>(input, output, N, C_in, C_out, H, W);
}

void launch_layer_norm_affine(
    const __half* input,
    const void* weight,
    const void* bias,
    bool params_fp32,
    __half* output,
    int N, int C, int H, int W,
    cudaStream_t stream
) {
    constexpr int kBlock = 64;
    const int64_t total = static_cast<int64_t>(N) * H * W;
    const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
    layer_norm_affine_kernel<<<blocks, kBlock, 0, stream>>>(
        input, weight, bias, params_fp32, output, N, C, H, W);
}

void launch_simple_gate(
    const __half* input,
    __half* output,
    int N, int C, int H, int W,
    cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int64_t total = static_cast<int64_t>(N) * C * H * W;
    const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
    simple_gate_kernel<<<blocks, kBlock, 0, stream>>>(input, output, N, C, H, W);
}

void launch_global_avg_pool(
    const __half* input,
    __half* output,
    int N, int C, int H, int W,
    cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int64_t total = static_cast<int64_t>(N) * C;
    const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
    global_avg_pool_kernel<<<blocks, kBlock, 0, stream>>>(input, output, N, C, H, W);
}

void launch_mul_channelwise(
    const __half* input,
    const __half* scale,
    __half* output,
    int N, int C, int H, int W,
    cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int64_t total = static_cast<int64_t>(N) * C * H * W;
    const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
    mul_channelwise_kernel<<<blocks, kBlock, 0, stream>>>(input, scale, output, N, C, H, W);
}

void launch_add(
    const __half* a,
    const __half* b,
    __half* out,
    int64_t n,
    cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
    add_kernel<<<blocks, kBlock, 0, stream>>>(a, b, out, n);
}

void launch_mul_scalar(
    const __half* input,
    float scalar,
    __half* output,
    int64_t n,
    cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int blocks = static_cast<int>((n + kBlock - 1) / kBlock);
    mul_scalar_kernel<<<blocks, kBlock, 0, stream>>>(input, scalar, output, n);
}

void launch_scaled_add(
    const __half* identity,
    const __half* residual,
    const void* scale,
    bool scale_fp32,
    __half* output,
    int N, int C, int H, int W,
    cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int64_t total = static_cast<int64_t>(N) * C * H * W;
    const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
    scaled_add_kernel<<<blocks, kBlock, 0, stream>>>(
        identity, residual, scale, scale_fp32, output, N, C, H, W);
}

void launch_pixel_shuffle2x(
    const __half* input,
    __half* output,
    int N, int C_in, int H, int W,
    cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int64_t total = static_cast<int64_t>(N) * (C_in / 4) * (H * 2) * (W * 2);
    const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
    pixel_shuffle2x_kernel<<<blocks, kBlock, 0, stream>>>(input, output, N, C_in, H, W);
}

void launch_warp_bilinear(
    const __half* feat,
    const __half* offset,
    __half* output,
    int N, int C, int H, int W,
    cudaStream_t stream
) {
    constexpr int kBlock = 256;
    const int64_t total = static_cast<int64_t>(N) * C * H * W;
    const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
    warp_bilinear_kernel<<<blocks, kBlock, 0, stream>>>(feat, offset, output, N, C, H, W);
}

}  // namespace denoiser::kernels
