#pragma once

#include "denoiser/tensor.h"
#include <cudnn.h>

namespace denoiser {

// Applies ReLU activation to an FP16 NCHW tensor via cudnnActivationForward.
//
// The cuDNN activation is kept as a separate layer (rather than fusing it into
// Conv2d) so it can be replaced or skipped independently.  A future
// optimisation could use cuDNN's fusion API to merge Conv+BN+ReLU.
class ReLULayer {
public:
    ReLULayer();
    ~ReLULayer();

    ReLULayer(const ReLULayer&) = delete;
    ReLULayer& operator=(const ReLULayer&) = delete;

    // Apply ReLU to *input* (FP16 NCHW), return activated output.
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) const;

private:
    cudnnHandle_t cudnn_ = nullptr;
    cudnnActivationDescriptor_t act_desc_ = nullptr;
};

} // namespace denoiser
