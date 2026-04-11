#pragma once

#include "denoiser/tensor.h"
#include "denoiser/weight_loader.h"

#include <cublas_v2.h>
#include <string>

namespace denoiser {

// ---------------------------------------------------------------------------
// DeformConv2dLayer — DCNv2-style modulated deformable convolution
// ---------------------------------------------------------------------------
//
// Computes:
//   output[n, c_out, h, w] = sum_{c_in, kh, kw}
//       weight[c_out, c_in, kh, kw]
//       * bilinear(input[n, c_in], h*s + kh*d + offset_h, w*s + kw*d + offset_w)
//       * sigmoid(mask[n, g(c_in)*kH*kW + kh*kW + kw, h, w])
//
// where offsets and masks are provided by the caller (offset_conv and
// mask_conv in DeformableAlignment).
//
// Implementation uses a custom CUDA im2col kernel followed by cublasHgemm.
// Offsets are FP16 on device and promoted to FP32 inside the kernel for
// accurate coordinate arithmetic.
class DeformConv2dLayer {
public:
    // weight_name: (C_out, C_in, kH, kW) FP16
    // bias_name:   (C_out,) FP16
    DeformConv2dLayer(const WeightStore& store,
                      const std::string& weight_name,
                      const std::string& bias_name,
                      int pad           = 1,
                      int stride        = 1,
                      int dilation      = 1,
                      int deform_groups = 8);
    ~DeformConv2dLayer();

    DeformConv2dLayer(const DeformConv2dLayer&) = delete;
    DeformConv2dLayer& operator=(const DeformConv2dLayer&) = delete;

    // Forward pass.
    //
    // input:   (N, C_in, H_in, W_in) FP16
    // offsets: (N, 2 * deform_groups * kH * kW, H_out, W_out) FP16
    //          — raw output of offset_conv (not scaled)
    // masks:   (N, deform_groups * kH * kW, H_out, W_out) FP16
    //          — raw output of mask_conv (sigmoid applied inside kernel)
    //
    // Returns: (N, C_out, H_out, W_out) FP16
    Tensor forward(const Tensor& input,
                   const Tensor& offsets,
                   const Tensor& masks,
                   cudaStream_t  stream = nullptr) const;

    int out_channels()  const noexcept { return out_channels_; }
    int in_channels()   const noexcept { return in_channels_; }
    int deform_groups() const noexcept { return deform_groups_; }

private:
    cublasHandle_t cublas_ = nullptr;

    const Tensor* weight_ = nullptr;  // non-owning ref
    const Tensor* bias_   = nullptr;

    int out_channels_  = 0;
    int in_channels_   = 0;
    int kernel_h_      = 0;
    int kernel_w_      = 0;
    int pad_           = 1;
    int stride_        = 1;
    int dilation_      = 1;
    int deform_groups_ = 8;
};

} // namespace denoiser
