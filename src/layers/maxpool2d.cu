#include "denoiser/layers/maxpool2d.h"
#include <stdexcept>

namespace denoiser {

MaxPool2dLayer::MaxPool2dLayer(int kernel_size, int stride, int pad)
    : kernel_size_(kernel_size), stride_(stride), pad_(pad) {
    CUDNN_CHECK(cudnnCreate(&cudnn_));
    CUDNN_CHECK(cudnnCreatePoolingDescriptor(&pool_desc_));
    CUDNN_CHECK(cudnnSetPooling2dDescriptor(
        pool_desc_,
        CUDNN_POOLING_MAX,
        CUDNN_NOT_PROPAGATE_NAN,
        kernel_size, kernel_size,  // windowHeight, windowWidth
        pad, pad,                  // verticalPadding, horizontalPadding
        stride, stride             // verticalStride, horizontalStride
    ));
}

MaxPool2dLayer::~MaxPool2dLayer() {
    if (pool_desc_) cudnnDestroyPoolingDescriptor(pool_desc_);
    if (cudnn_)     cudnnDestroy(cudnn_);
}

Tensor MaxPool2dLayer::forward(const Tensor& input, cudaStream_t stream) const {
    if (input.dtype() != DType::kFloat16) {
        throw std::runtime_error("MaxPool2dLayer::forward: expected FP16 input");
    }

    const int N = static_cast<int>(input.batch());
    const int C = static_cast<int>(input.channels());
    const int H = static_cast<int>(input.height());
    const int W = static_cast<int>(input.width());

    // Build the input descriptor once; reuse it for both the size query and
    // the actual pooling forward call to avoid a double allocation.
    cudnnTensorDescriptor_t in_desc = input.make_cudnn_descriptor();

    int N_out, C_out, H_out, W_out;
    CUDNN_CHECK(cudnnGetPooling2dForwardOutputDim(
        pool_desc_, in_desc, &N_out, &C_out, &H_out, &W_out));

    auto output = Tensor::empty({N_out, C_out, H_out, W_out}, DType::kFloat16);
    cudnnTensorDescriptor_t out_desc = output.make_cudnn_descriptor();

    if (stream) CUDNN_CHECK(cudnnSetStream(cudnn_, stream));

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    CUDNN_CHECK(cudnnPoolingForward(
        cudnn_, pool_desc_,
        &alpha, in_desc,  input.data(),
        &beta,  out_desc, output.data()
    ));

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    return output;
}

} // namespace denoiser
