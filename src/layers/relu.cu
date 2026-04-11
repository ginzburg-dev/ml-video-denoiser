#include "denoiser/layers/relu.h"
#include <stdexcept>

namespace denoiser {

ReLULayer::ReLULayer() {
    CUDNN_CHECK(cudnnCreate(&cudnn_));
    CUDNN_CHECK(cudnnCreateActivationDescriptor(&act_desc_));
    CUDNN_CHECK(cudnnSetActivationDescriptor(
        act_desc_,
        CUDNN_ACTIVATION_RELU,
        CUDNN_NOT_PROPAGATE_NAN,
        0.0  // coef — unused for ReLU
    ));
}

ReLULayer::~ReLULayer() {
    if (act_desc_) cudnnDestroyActivationDescriptor(act_desc_);
    if (cudnn_)    cudnnDestroy(cudnn_);
}

Tensor ReLULayer::forward(const Tensor& input, cudaStream_t stream) const {
    if (input.dtype() != DType::kFloat16) {
        throw std::runtime_error("ReLULayer::forward: expected FP16 input");
    }
    if (stream) CUDNN_CHECK(cudnnSetStream(cudnn_, stream));

    auto output = Tensor::empty(input.shape(), DType::kFloat16);
    const float alpha = 1.0f;
    const float beta  = 0.0f;

    auto in_desc  = input.make_cudnn_descriptor();
    auto out_desc = output.make_cudnn_descriptor();

    CUDNN_CHECK(cudnnActivationForward(
        cudnn_, act_desc_,
        &alpha, in_desc,  input.data(),
        &beta,  out_desc, output.data()
    ));

    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
    return output;
}

} // namespace denoiser
