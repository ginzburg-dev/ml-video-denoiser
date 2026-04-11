#pragma once

#include "denoiser/tensor.h"
#include <cudnn.h>

namespace denoiser {

// MaxPool2d for FP16 NCHW tensors via cudnnPoolingForward.
//
// Default configuration (kernel=2, stride=2, pad=0) matches PyTorch
// MaxPool2d(2, 2) which halves H and W each call.
class MaxPool2dLayer {
public:
    explicit MaxPool2dLayer(int kernel_size = 2, int stride = 2, int pad = 0);
    ~MaxPool2dLayer();

    MaxPool2dLayer(const MaxPool2dLayer&) = delete;
    MaxPool2dLayer& operator=(const MaxPool2dLayer&) = delete;

    // Spatially pool *input* (FP16 NCHW) and return a new FP16 tensor.
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) const;

private:
    cudnnHandle_t cudnn_ = nullptr;
    cudnnPoolingDescriptor_t pool_desc_ = nullptr;
    int kernel_size_;
    int stride_;
    int pad_;
};

} // namespace denoiser
