#pragma once

#include "denoiser/tensor.h"
#include "denoiser/weight_loader.h"

#include <memory>
#include <vector>

namespace denoiser {

// ---------------------------------------------------------------------------
// NEFResidual — single-frame UNet denoiser (C++ inference engine)
// ---------------------------------------------------------------------------
//
// Mirrors the Python NEFResidual architecture:
//   Encoder:     N levels, each = 2× ConvBnRelu + MaxPool2d(2)
//   Bottleneck:  2× ConvBnRelu at enc_channels.back() * 2
//   Decoder:     N levels, each = bilinear 2× upsample + concat(skip) + 2× ConvBnRelu
//   Head:        Conv2d(enc_channels[0], out_channels, 1×1)
//   Output:      clamp(input − head(decoder_output), 0, 1)
//
// Input spatial dimensions are padded to a multiple of 2^num_levels with
// reflect padding before processing and stripped from the output.
//
// Weight names loaded from the WeightStore must match the PyTorch export:
//   encoders.{lvl}.conv{1,2}.conv.weight
//   encoders.{lvl}.conv{1,2}.bn.{weight,bias,running_mean,running_var}
//   bottleneck.{0,1}.conv.weight
//   bottleneck.{0,1}.bn.{weight,bias,running_mean,running_var}
//   decoders.{lvl}.conv{1,2}.conv.weight
//   decoders.{lvl}.conv{1,2}.bn.{weight,bias,running_mean,running_var}
//   head.weight,  head.bias
//
// Architecture parameters (enc_channels, num_levels, in_channels,
// out_channels) are read automatically from manifest().architecture.
class NEFResidual {
public:
    // Build from a pre-loaded WeightStore.  Architecture is inferred from
    // manifest().architecture — call WeightStore::prefetch_all() beforehand
    // if low-latency first-inference is required.
    explicit NEFResidual(const WeightStore& store);
    ~NEFResidual();

    NEFResidual(const NEFResidual&) = delete;
    NEFResidual& operator=(const NEFResidual&) = delete;

    // Denoise *input* (FP16 NCHW, values in [0, 1]).
    //
    // The input may be any spatial size — it is automatically reflect-padded
    // to the nearest multiple of 2^num_levels before the forward pass and the
    // output is cropped back to the original size.
    //
    // Returns a new FP16 NCHW tensor of the same shape as *input*, with
    // values clamped to [0, 1].
    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) const;

    // Architecture accessors
    const std::vector<int>& enc_channels() const noexcept;
    int num_levels()   const noexcept;
    int in_channels()  const noexcept;
    int out_channels() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace denoiser
