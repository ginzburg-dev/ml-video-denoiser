#pragma once

// DCNv2 deformable im2col CUDA kernel.
//
// For each output element (n, c, h_out, w_out) and kernel offset (kh, kw):
//   1. Look up the learned spatial offsets for the deformable group of channel c.
//   2. Compute the fractional input coordinate (h_src, w_src) in FP32.
//   3. Bilinear-sample the FP16 input at that coordinate (FP32 accumulation).
//   4. Multiply by sigmoid(mask) for modulated deformable conv (DCNv2).
//   5. Store in the col buffer at [n*HW + h_out*W_out + w_out, c*kH*kW + kh*kW + kw].
//
// The col buffer is subsequently multiplied by the weight matrix via cuBLAS
// Hgemm (see deform_conv2d.cu) to produce the convolution output.
//
// Offset layout  (N, 2 * deform_groups * kH * kW, H_out, W_out):
//   offset_h for group g, kernel pos (kh,kw):
//     offsets[n, 2*(g*kH*kW + kh*kW + kw),   h_out, w_out]
//   offset_w:
//     offsets[n, 2*(g*kH*kW + kh*kW + kw)+1, h_out, w_out]
//
// Mask layout  (N, deform_groups * kH * kW, H_out, W_out):
//   masks[n, g*kH*kW + kh*kW + kw, h_out, w_out]
//   Masks are raw (pre-sigmoid); sigmoid is applied inside the kernel.

#include <cuda_fp16.h>
#include <cstdint>

namespace denoiser::kernels {

// ---------------------------------------------------------------------------
// Device helper: bilinear sample FP16 input at fractional (h, w) coordinate.
// Returns FP32 interpolated value.  Out-of-bound positions yield 0.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float bilinear_sample(
    const __half* __restrict__ input,
    int C, int H, int W,
    int n, int c,
    float h_f, float w_f
) {
    if (h_f < -1.f || h_f > static_cast<float>(H) ||
        w_f < -1.f || w_f > static_cast<float>(W)) {
        return 0.f;
    }

    const int h0 = static_cast<int>(floorf(h_f));
    const int w0 = static_cast<int>(floorf(w_f));
    const int h1 = h0 + 1;
    const int w1 = w0 + 1;

    const float dh = h_f - static_cast<float>(h0);
    const float dw = w_f - static_cast<float>(w0);

    // Clamp to valid range; out-of-bounds corners contribute 0
    auto safe_val = [&](int r, int c_idx) -> float {
        if (r < 0 || r >= H || c_idx < 0 || c_idx >= W) return 0.f;
        return __half2float(input[(static_cast<int64_t>(n) * C + c) * H * W
                                  + static_cast<int64_t>(r) * W + c_idx]);
    };

    const float v00 = safe_val(h0, w0);
    const float v01 = safe_val(h0, w1);
    const float v10 = safe_val(h1, w0);
    const float v11 = safe_val(h1, w1);

    return v00 * (1.f - dh) * (1.f - dw)
         + v01 * (1.f - dh) * dw
         + v10 * dh          * (1.f - dw)
         + v11 * dh          * dw;
}

// ---------------------------------------------------------------------------
// Deformable im2col kernel
// ---------------------------------------------------------------------------
// Total threads: N * C_in * kH * kW * H_out * W_out
//
// col_buffer layout: (N * H_out * W_out, C_in * kH * kW) FP16
//   stored row-major so cuBLAS can treat it as (K, HW) for the GEMM.
__global__ void deform_im2col_kernel(
    const __half* __restrict__ input,       // (N, C_in, H_in, W_in)
    const __half* __restrict__ offsets,     // (N, 2*dg*kH*kW, H_out, W_out) — FP16, promoted in kernel
    const __half* __restrict__ masks,       // (N, dg*kH*kW, H_out, W_out)   — FP16 pre-sigmoid
    __half* __restrict__       col_buffer,  // (N*H_out*W_out, C_in*kH*kW) FP16
    int N, int C_in, int H_in, int W_in,
    int kH, int kW,
    int H_out, int W_out,
    int stride, int pad, int dilation,
    int deform_groups
) {
    const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * C_in * kH * kW * H_out * W_out;
    if (idx >= total) return;

    // Decode linear index → (n, c, kh, kw, h_out, w_out)
    const int w_out  = static_cast<int>(idx % W_out);
    const int h_out  = static_cast<int>((idx / W_out) % H_out);
    const int kw     = static_cast<int>((idx / (static_cast<int64_t>(W_out) * H_out)) % kW);
    const int kh     = static_cast<int>((idx / (static_cast<int64_t>(W_out) * H_out * kW)) % kH);
    const int c      = static_cast<int>((idx / (static_cast<int64_t>(W_out) * H_out * kW * kH)) % C_in);
    const int n      = static_cast<int>(idx / (static_cast<int64_t>(W_out) * H_out * kW * kH * C_in));

    // Deformable group for this channel
    const int channels_per_group = C_in / deform_groups;
    const int g = c / channels_per_group;

    // Kernel position index within its group
    const int k_idx = kh * kW + kw;

    // Offset tensor: (N, 2*dg*kH*kW, H_out, W_out)
    // Stride: last dim W_out, then H_out, then 2*dg*kH*kW, then N
    const int offset_ch_h = 2 * (g * kH * kW + k_idx);
    const int offset_ch_w = offset_ch_h + 1;
    const int64_t off_base = (static_cast<int64_t>(n) * (2 * deform_groups * kH * kW)
                              + offset_ch_h) * H_out * W_out
                             + static_cast<int64_t>(h_out) * W_out + w_out;

    const float offset_h = __half2float(offsets[off_base]);
    const float offset_w = __half2float(offsets[off_base + static_cast<int64_t>(H_out) * W_out]);

    // Source coordinates in input space
    const float h_src = static_cast<float>(h_out * stride - pad + kh * dilation) + offset_h;
    const float w_src = static_cast<float>(w_out * stride - pad + kw * dilation) + offset_w;

    // Bilinear sample
    const float val = bilinear_sample(input, C_in, H_in, W_in, n, c, h_src, w_src);

    // Mask: (N, dg*kH*kW, H_out, W_out)
    const int64_t mask_idx = (static_cast<int64_t>(n) * (deform_groups * kH * kW)
                              + static_cast<int64_t>(g * kH * kW + k_idx)) * H_out * W_out
                             + static_cast<int64_t>(h_out) * W_out + w_out;
    const float raw_mask = __half2float(masks[mask_idx]);
    const float mask     = 1.f / (1.f + expf(-raw_mask));  // sigmoid

    // Write to col buffer at [n*HW + h_out*W_out + w_out, c*kH*kW + kh*kW + kw]
    // Layout: (N*H_out*W_out, C_in*kH*kW) row-major
    const int64_t col_row = static_cast<int64_t>(n) * H_out * W_out
                            + static_cast<int64_t>(h_out) * W_out + w_out;
    const int     col_col = c * kH * kW + kh * kW + kw;
    const int64_t K       = static_cast<int64_t>(C_in) * kH * kW;

    col_buffer[col_col * (static_cast<int64_t>(N) * H_out * W_out) + col_row] =
        __float2half(val * mask);
}

// Host launcher (defined in deform_conv2d.cu)
void launch_deform_im2col(
    const __half* input,
    const __half* offsets,
    const __half* masks,
    __half* col_buffer,
    int N, int C_in, int H_in, int W_in,
    int kH, int kW,
    int H_out, int W_out,
    int stride, int pad, int dilation,
    int deform_groups,
    cudaStream_t stream
);

} // namespace denoiser::kernels
