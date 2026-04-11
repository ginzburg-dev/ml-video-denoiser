#pragma once

#include "denoiser/tensor.h"
#include <cudnn.h>
#include <cstdint>

namespace denoiser {

// Bilinear 2× upsampling for FP16 NCHW tensors.
//
// Implementation selects at compile time (via CMake define):
//   DENOISER_USE_CUDNN_RESAMPLE (cuDNN >= 8.5)
//     — uses cudnnResampleForward with CUDNN_RESAMPLE_BILINEAR.
//   Otherwise
//     — falls back to a custom CUDA kernel (equivalent result, no version dep).
class UpsampleLayer {
public:
    // scale_factor: spatial multiplier (default: 2 for 2× upsampling).
    explicit UpsampleLayer(int scale_factor = 2);
    ~UpsampleLayer();

    UpsampleLayer(const UpsampleLayer&) = delete;
    UpsampleLayer& operator=(const UpsampleLayer&) = delete;

    // Upsample *input* (FP16 NCHW) by *scale_factor* in H and W.
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) const;

    int scale_factor() const noexcept { return scale_factor_; }

private:
    int scale_factor_ = 2;

#ifdef DENOISER_USE_CUDNN_RESAMPLE
    cudnnHandle_t cudnn_ = nullptr;
    cudnnResampleDescriptor_t resample_desc_ = nullptr;
    cudnnSpatialTransformerDescriptor_t spatial_desc_ = nullptr;
#endif
};

} // namespace denoiser
