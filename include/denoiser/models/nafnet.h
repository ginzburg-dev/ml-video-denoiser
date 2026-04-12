#pragma once

#include "denoiser/tensor.h"
#include "denoiser/weight_loader.h"

#include <memory>
#include <vector>

namespace denoiser {

class NAFNet {
public:
    explicit NAFNet(const WeightStore& store);
    ~NAFNet();

    NAFNet(const NAFNet&) = delete;
    NAFNet& operator=(const NAFNet&) = delete;

    Tensor forward(const Tensor& input, cudaStream_t stream = nullptr) const;

    const std::vector<int>& enc_channels() const noexcept;
    int num_levels() const noexcept;
    int in_channels() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace denoiser

