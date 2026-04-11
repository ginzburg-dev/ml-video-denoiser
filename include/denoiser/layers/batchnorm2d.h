#pragma once

#include "denoiser/tensor.h"
#include "denoiser/weight_loader.h"

#include <string>

namespace denoiser {

// ---------------------------------------------------------------------------
// BatchNorm2dLayer — inference mode
// ---------------------------------------------------------------------------

// Applies BatchNorm inference to an FP16 NCHW tensor using a custom CUDA
// kernel that computes:
//
//   out[c] = in[c] * scale[c] + shift[c]
//
// where scale and shift are pre-computed at construction from the trained
// running statistics:
//
//   scale[c] = gamma[c] / sqrt(running_var[c] + eps)
//   shift[c] = beta[c]  - scale[c] * running_mean[c]
//
// Both scale and shift are stored as FP32 on the GPU.  The kernel reads FP16
// activations, accumulates in FP32, and writes FP16 output.  This avoids the
// precision loss that would occur if BN statistics were applied in FP16.
class BatchNorm2dLayer {
public:
    // Fetches weight, bias, running_mean, running_var from *store* using
    // the key prefix *base_name* (e.g. "encoder.0.conv1.bn").
    //
    // Expected keys in store:
    //   base_name + ".weight"        — gamma (C,) FP32
    //   base_name + ".bias"          — beta  (C,) FP32
    //   base_name + ".running_mean"  — (C,) FP32
    //   base_name + ".running_var"   — (C,) FP32
    BatchNorm2dLayer(const WeightStore& store,
                     const std::string& base_name,
                     float eps = 1e-5f);

    ~BatchNorm2dLayer();

    BatchNorm2dLayer(const BatchNorm2dLayer&) = delete;
    BatchNorm2dLayer& operator=(const BatchNorm2dLayer&) = delete;

    // Apply BatchNorm in-place on *input* and return the result.
    // input must be an FP16 NCHW tensor.
    // The returned tensor is allocated fresh (not in-place) to preserve the
    // input for gradient debugging; callers may discard the original.
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) const;

    int num_channels() const noexcept { return num_channels_; }

private:
    int num_channels_ = 0;

    // Pre-computed FP32 scale and shift on device
    float* d_scale_ = nullptr;  // (C,)
    float* d_shift_ = nullptr;  // (C,)
};

} // namespace denoiser
