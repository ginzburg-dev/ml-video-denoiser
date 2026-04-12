#pragma once

#include "denoiser/tensor.h"
#include "denoiser/weight_loader.h"

#include <cudnn.h>
#include <map>
#include <optional>
#include <string>
#include <utility>

namespace denoiser {

// ---------------------------------------------------------------------------
// Conv2dLayer
// ---------------------------------------------------------------------------

// cuDNN-based 2-D convolution with optional bias.
//
// FP16 inference strategy:
//   - Weight and bias tensors are FP16.
//   - Input and output tensors are FP16.
//   - Math type: CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION — cuDNN promotes to
//     FP32 internally for the GEMM accumulation (Tensor Cores on Ampere),
//     then converts the result back to FP16.
//   - Algorithm: CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM (best for
//     small batch sizes with Tensor Cores).
//
// Workspace:
//   A device workspace buffer is required by cuDNN.  It is allocated once
//   per unique (H, W) input size and cached to avoid repeated allocation.
//   The workspace is reused across forward() calls with the same spatial dims.
class Conv2dLayer {
public:
    // Constructs a convolution layer from weights in *store*.
    //
    // Args:
    //   store: Weight store from which to fetch weight and bias tensors.
    //   weight_name: Manifest key for the weight tensor, shape (C_out, C_in, kH, kW).
    //   bias_name: Optional manifest key for the bias tensor, shape (C_out,).
    //   pad: Symmetric padding applied to each spatial dimension.
    //   stride: Convolution stride.
    //   dilation: Convolution dilation.
    Conv2dLayer(const WeightStore& store,
                const std::string& weight_name,
                std::optional<std::string> bias_name = std::nullopt,
                int pad = 1,
                int stride = 1,
                int dilation = 1,
                int groups = 1);

    ~Conv2dLayer();

    // Non-copyable, non-movable (holds cuDNN descriptors).
    Conv2dLayer(const Conv2dLayer&) = delete;
    Conv2dLayer& operator=(const Conv2dLayer&) = delete;

    // Perform forward convolution.
    //
    // input:   FP16 NCHW tensor.
    // stream:  CUDA stream for the operation.
    // Returns: FP16 NCHW output tensor (newly allocated each call).
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) const;

    int out_channels() const noexcept { return out_channels_; }
    int in_channels() const noexcept { return in_channels_; }
    int kernel_h() const noexcept { return kernel_h_; }
    int kernel_w() const noexcept { return kernel_w_; }
    int groups() const noexcept { return groups_; }

private:
    struct Workspace {
        void* ptr = nullptr;
        size_t size = 0;
    };

    cudnnHandle_t cudnn_ = nullptr;
    cudnnFilterDescriptor_t filter_desc_ = nullptr;
    cudnnConvolutionDescriptor_t conv_desc_ = nullptr;
    cudnnTensorDescriptor_t bias_desc_ = nullptr;

    const Tensor* weight_ = nullptr;  // non-owning ref into WeightStore
    const Tensor* bias_ = nullptr;

    int out_channels_ = 0;
    int in_channels_ = 0;
    int kernel_h_ = 0;
    int kernel_w_ = 0;
    int pad_ = 0;
    int stride_ = 0;
    int dilation_ = 0;
    int groups_ = 1;

    // Workspace cache keyed by (H, W)
    mutable std::map<std::pair<int, int>, Workspace> workspace_cache_;

    const Workspace& get_workspace(cudnnTensorDescriptor_t in_desc,
                                   cudnnTensorDescriptor_t out_desc) const;
};

} // namespace denoiser
