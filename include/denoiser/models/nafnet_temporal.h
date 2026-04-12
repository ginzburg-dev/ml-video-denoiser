#pragma once

#include "denoiser/tensor.h"
#include "denoiser/weight_loader.h"

#include <memory>
#include <vector>

namespace denoiser {

class NAFNetTemporal {
public:
    explicit NAFNetTemporal(const WeightStore& store);
    ~NAFNetTemporal();

    NAFNetTemporal(const NAFNetTemporal&) = delete;
    NAFNetTemporal& operator=(const NAFNetTemporal&) = delete;

    Tensor forward(const Tensor& clip, cudaStream_t stream = nullptr) const;

    int num_frames() const noexcept;
    int num_levels() const noexcept;
    const std::vector<int>& enc_channels() const noexcept;
    bool use_warp() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace denoiser
