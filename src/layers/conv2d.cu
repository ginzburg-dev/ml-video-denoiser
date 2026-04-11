#include "denoiser/layers/conv2d.h"

#include <stdexcept>
#include <string>

namespace denoiser {

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

Conv2dLayer::Conv2dLayer(const WeightStore& store,
                         const std::string& weight_name,
                         std::optional<std::string> bias_name,
                         int pad, int stride, int dilation)
    : pad_(pad), stride_(stride), dilation_(dilation) {

    weight_ = &store.get(weight_name);
    if (bias_name.has_value()) {
        bias_ = &store.get(*bias_name);
    }

    // Infer dimensions from weight shape: (C_out, C_in, kH, kW)
    const auto& sh = weight_->shape();
    if (sh.size() != 4) {
        throw std::runtime_error("Conv2dLayer: weight must be 4-D, got " +
                                 std::to_string(sh.size()) + "-D");
    }
    out_channels_ = static_cast<int>(sh[0]);
    in_channels_  = static_cast<int>(sh[1]);
    kernel_h_     = static_cast<int>(sh[2]);
    kernel_w_     = static_cast<int>(sh[3]);

    // cuDNN handle
    CUDNN_CHECK(cudnnCreate(&cudnn_));

    // Filter descriptor
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filter_desc_));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(
        filter_desc_,
        CUDNN_DATA_HALF,          // FP16 weights
        CUDNN_TENSOR_NCHW,
        out_channels_,
        in_channels_,
        kernel_h_,
        kernel_w_
    ));

    // Convolution descriptor
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&conv_desc_));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(
        conv_desc_,
        pad, pad,
        stride, stride,
        dilation, dilation,
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT           // accumulate in FP32 (required for FP16 TC path)
    ));
    // Enable Tensor Core math (FP16→FP32 promotion happens automatically)
    CUDNN_CHECK(cudnnSetConvolutionMathType(
        conv_desc_, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION));

    // Bias descriptor (1-D along channel dimension)
    if (bias_) {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&bias_desc_));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(
            bias_desc_,
            CUDNN_TENSOR_NCHW,
            CUDNN_DATA_HALF,
            1, out_channels_, 1, 1
        ));
    }
}

Conv2dLayer::~Conv2dLayer() {
    for (auto& [key, ws] : workspace_cache_) {
        if (ws.ptr) cudaFree(ws.ptr);
    }
    if (bias_desc_)  cudnnDestroyTensorDescriptor(bias_desc_);
    if (conv_desc_)  cudnnDestroyConvolutionDescriptor(conv_desc_);
    if (filter_desc_) cudnnDestroyFilterDescriptor(filter_desc_);
    if (cudnn_)      cudnnDestroy(cudnn_);
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

Tensor Conv2dLayer::forward(const Tensor& input, cudaStream_t stream) const {
    if (input.dtype() != DType::kFloat16) {
        throw std::runtime_error("Conv2dLayer::forward: expected FP16 input");
    }

    const int N = static_cast<int>(input.batch());
    const int H = static_cast<int>(input.height());
    const int W = static_cast<int>(input.width());

    // Compute output spatial dimensions
    const int H_out = (H + 2 * pad_ - dilation_ * (kernel_h_ - 1) - 1) / stride_ + 1;
    const int W_out = (W + 2 * pad_ - dilation_ * (kernel_w_ - 1) - 1) / stride_ + 1;

    // Create input / output tensor descriptors for this batch shape
    cudnnTensorDescriptor_t in_desc = input.make_cudnn_descriptor();
    auto output = Tensor::empty({N, out_channels_, H_out, W_out}, DType::kFloat16);
    cudnnTensorDescriptor_t out_desc = output.make_cudnn_descriptor();

    // Set stream on the cuDNN handle
    if (stream) CUDNN_CHECK(cudnnSetStream(cudnn_, stream));

    // Workspace
    const auto& ws = get_workspace(in_desc, out_desc);

    // alpha / beta for: output = alpha * conv(input, weight) + beta * output
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn_,
        &alpha,
        in_desc,  input.data(),
        filter_desc_, weight_->data(),
        conv_desc_,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        ws.ptr, ws.size,
        &beta,
        out_desc, output.data()
    ));

    // Add bias (in-place)
    if (bias_) {
        const float bias_alpha = 1.0f;
        const float bias_beta  = 1.0f;
        CUDNN_CHECK(cudnnAddTensor(
            cudnn_,
            &bias_alpha,
            bias_desc_, bias_->data(),
            &bias_beta,
            out_desc, output.data()
        ));
    }

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    return output;
}

// ---------------------------------------------------------------------------
// Workspace cache
// ---------------------------------------------------------------------------

const Conv2dLayer::Workspace& Conv2dLayer::get_workspace(
    cudnnTensorDescriptor_t in_desc,
    cudnnTensorDescriptor_t out_desc) const {

    int N_out, C_out, H_out, W_out;
    cudnnDataType_t dt;
    int nStr, cStr, hStr, wStr;
    cudnnGetTensor4dDescriptor(out_desc, &dt, &N_out, &C_out, &H_out, &W_out,
                               &nStr, &cStr, &hStr, &wStr);

    const auto key = std::make_pair(H_out, W_out);
    const auto it = workspace_cache_.find(key);
    if (it != workspace_cache_.end()) return it->second;

    // Query required workspace size
    size_t ws_bytes = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn_,
        in_desc,
        filter_desc_,
        conv_desc_,
        out_desc,
        CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        &ws_bytes
    ));

    Workspace ws;
    ws.size = ws_bytes;
    if (ws_bytes > 0) {
        CUDA_CHECK(cudaMalloc(&ws.ptr, ws_bytes));
    }
    workspace_cache_.emplace(key, ws);
    return workspace_cache_.at(key);
}

} // namespace denoiser
