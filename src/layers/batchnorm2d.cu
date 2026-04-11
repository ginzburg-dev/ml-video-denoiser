#include "denoiser/layers/batchnorm2d.h"
#include "../kernels/bn_inference.cuh"
#include "../kernels/concat.cuh"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace denoiser {

// ---------------------------------------------------------------------------
// Construction — pre-compute scale and shift on GPU
// ---------------------------------------------------------------------------

BatchNorm2dLayer::BatchNorm2dLayer(const WeightStore& store,
                                   const std::string& base_name,
                                   float eps) {
    // Fetch FP32 stats from the weight store
    const Tensor& gamma       = store.get(base_name + ".weight");
    const Tensor& beta        = store.get(base_name + ".bias");
    const Tensor& running_mean = store.get(base_name + ".running_mean");
    const Tensor& running_var  = store.get(base_name + ".running_var");

    if (gamma.dtype() != DType::kFloat32 ||
        running_mean.dtype() != DType::kFloat32) {
        throw std::runtime_error(
            "BatchNorm2dLayer: BN stats must be float32 in the manifest");
    }

    num_channels_ = static_cast<int>(gamma.numel());

    // Download stats to host, compute scale/shift, re-upload
    std::vector<float> h_gamma(num_channels_);
    std::vector<float> h_beta(num_channels_);
    std::vector<float> h_mean(num_channels_);
    std::vector<float> h_var(num_channels_);

    gamma.to_host(h_gamma.data());
    beta.to_host(h_beta.data());
    running_mean.to_host(h_mean.data());
    running_var.to_host(h_var.data());

    std::vector<float> h_scale(num_channels_);
    std::vector<float> h_shift(num_channels_);
    for (int c = 0; c < num_channels_; ++c) {
        h_scale[c] = h_gamma[c] / std::sqrt(h_var[c] + eps);
        h_shift[c] = h_beta[c] - h_scale[c] * h_mean[c];
    }

    const size_t bytes = static_cast<size_t>(num_channels_) * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_scale_, bytes));
    CUDA_CHECK(cudaMalloc(&d_shift_, bytes));
    CUDA_CHECK(cudaMemcpy(d_scale_, h_scale.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_shift_, h_shift.data(), bytes, cudaMemcpyHostToDevice));
}

BatchNorm2dLayer::~BatchNorm2dLayer() {
    if (d_scale_) { cudaFree(d_scale_); d_scale_ = nullptr; }
    if (d_shift_) { cudaFree(d_shift_); d_shift_ = nullptr; }
}

// ---------------------------------------------------------------------------
// Forward
// ---------------------------------------------------------------------------

Tensor BatchNorm2dLayer::forward(const Tensor& input, cudaStream_t stream) const {
    if (input.dtype() != DType::kFloat16) {
        throw std::runtime_error("BatchNorm2dLayer::forward: expected FP16 input");
    }
    if (input.channels() != num_channels_) {
        throw std::runtime_error(
            "BatchNorm2dLayer::forward: channel mismatch — "
            "expected " + std::to_string(num_channels_) +
            ", got "    + std::to_string(input.channels()));
    }

    auto output = Tensor::empty(input.shape(), DType::kFloat16);
    kernels::launch_bn_inference(
        input.data_f16(),
        output.data_f16(),
        d_scale_,
        d_shift_,
        static_cast<int>(input.batch()),
        num_channels_,
        static_cast<int>(input.height()),
        static_cast<int>(input.width()),
        stream
    );
    return output;
}

} // namespace denoiser

// ---------------------------------------------------------------------------
// Kernel launchers (defined in the same .cu to share the CUDA compilation unit)
// ---------------------------------------------------------------------------

namespace denoiser::kernels {

void launch_bn_inference(
    const __half* input,
    __half* output,
    const float* scale,
    const float* shift,
    int N, int C, int H, int W,
    cudaStream_t stream
) {
    const int64_t total = static_cast<int64_t>(N) * C * H * W;
    constexpr int kBlockSize = 256;
    const int blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    bn_inference_kernel<<<blocks, kBlockSize, 0, stream>>>(
        input, output, scale, shift, N, C, H, W);
    // Error check is intentionally deferred to the next CUDA synchronisation
    // point to avoid serialising every kernel launch.
}

void launch_concat_channels(
    const __half* a,
    const __half* b,
    __half* output,
    int N, int C1, int C2, int H, int W,
    cudaStream_t stream
) {
    const int64_t total = static_cast<int64_t>(N) * (C1 + C2) * H * W;
    constexpr int kBlockSize = 256;
    const int blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    concat_channels_kernel<<<blocks, kBlockSize, 0, stream>>>(
        a, b, output, N, C1, C2, H, W);
}

} // namespace denoiser::kernels
