#pragma once

#include "denoiser/tensor.h"
#include "denoiser/weight_loader.h"

#include <memory>
#include <vector>

namespace denoiser {

// ---------------------------------------------------------------------------
// NEFTemporal — 5-frame deformable UNet denoiser (C++ inference engine)
// ---------------------------------------------------------------------------
//
// Architecture mirrors the Python NEFTemporal:
//   1. Shared encoder (same weights) applied to all T frames as a batch.
//   2. Per-level DeformableAlignment: each non-reference frame's features
//      are warped toward the reference frame using DCNv2.
//   3. Per-level temporal fusion: aligned features concatenated across frames
//      and reduced via a 1×1 conv.
//   4. Bottleneck on the reference frame's deepest features.
//   5. Shared decoder with fused temporal skip connections.
//   6. Head + residual: output = clamp(ref_input − noise, 0, 1).
//
// Reference frame index: num_frames // 2  (middle frame).
//
// Weight name convention (from Python export):
//   encoders.{lvl}.conv{1,2}.conv.weight / .bn.*
//   align_layers.{lvl}.offset_conv.weight / .bias
//   align_layers.{lvl}.mask_conv.weight / .bias
//   align_layers.{lvl}.weight / .bias        (deformable conv)
//   fusion_layers.{lvl}.weight / .bias       (1×1 conv)
//   bottleneck.{0,1}.conv.weight / .bn.*
//   decoders.{lvl}.conv{1,2}.conv.weight / .bn.*
//   head.weight / .bias
class NEFTemporal {
public:
    explicit NEFTemporal(const WeightStore& store);
    ~NEFTemporal();

    NEFTemporal(const NEFTemporal&) = delete;
    NEFTemporal& operator=(const NEFTemporal&) = delete;

    // Denoise the centre frame of *clip*.
    //
    // clip: FP16 tensor of shape (N, T, C, H, W) — T consecutive noisy frames.
    //       Must match num_frames from the manifest architecture.
    //
    // Returns: denoised centre frame (N, C, H, W) FP16, values in [0, 1].
    Tensor forward(const Tensor& clip, cudaStream_t stream = nullptr) const;

    int num_frames() const noexcept;
    int num_levels() const noexcept;
    const std::vector<int>& enc_channels() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace denoiser
