#include "denoiser/layers/upsample.h"

#include <cuda_fp16.h>
#include <stdexcept>

namespace denoiser {

// ---------------------------------------------------------------------------
// Custom bilinear upsample kernel (fallback for cuDNN < 8.5 and default path)
// ---------------------------------------------------------------------------

namespace {

__global__ void bilinear_upsample_kernel(
    const __half* __restrict__ input,
    __half* __restrict__ output,
    int N, int C, int H_in, int W_in, int H_out, int W_out,
    float scale_h, float scale_w
) {
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(N) * C * H_out * W_out;
    if (idx >= total) return;

    const int w_out = static_cast<int>(idx % W_out);
    const int h_out = static_cast<int>((idx / W_out) % H_out);
    const int c     = static_cast<int>((idx / (static_cast<int64_t>(H_out) * W_out)) % C);
    const int n     = static_cast<int>(idx / (static_cast<int64_t>(C) * H_out * W_out));

    // Map output position to input coordinates (align_corners=False, PyTorch default)
    const float h_in_f = (static_cast<float>(h_out) + 0.5f) / scale_h - 0.5f;
    const float w_in_f = (static_cast<float>(w_out) + 0.5f) / scale_w - 0.5f;

    const int h0 = max(0, static_cast<int>(floorf(h_in_f)));
    const int w0 = max(0, static_cast<int>(floorf(w_in_f)));
    const int h1 = min(h0 + 1, H_in - 1);
    const int w1 = min(w0 + 1, W_in - 1);

    const float dh = h_in_f - static_cast<float>(h0);
    const float dw = w_in_f - static_cast<float>(w0);

    // Bilinear interpolation
    const int64_t ch_offset = (static_cast<int64_t>(n) * C + c) * H_in * W_in;
    const float v00 = __half2float(input[ch_offset + static_cast<int64_t>(h0) * W_in + w0]);
    const float v01 = __half2float(input[ch_offset + static_cast<int64_t>(h0) * W_in + w1]);
    const float v10 = __half2float(input[ch_offset + static_cast<int64_t>(h1) * W_in + w0]);
    const float v11 = __half2float(input[ch_offset + static_cast<int64_t>(h1) * W_in + w1]);

    const float val = v00 * (1.f - dh) * (1.f - dw)
                    + v01 * (1.f - dh) * dw
                    + v10 * dh         * (1.f - dw)
                    + v11 * dh         * dw;

    output[idx] = __float2half(val);
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// UpsampleLayer
// ---------------------------------------------------------------------------

UpsampleLayer::UpsampleLayer(int scale_factor) : scale_factor_(scale_factor) {
    if (scale_factor <= 0) {
        throw std::invalid_argument("UpsampleLayer: scale_factor must be > 0");
    }
#ifdef DENOISER_USE_CUDNN_RESAMPLE
    CUDNN_CHECK(cudnnCreate(&cudnn_));
    CUDNN_CHECK(cudnnCreateResampleDescriptor(&resample_desc_));
    CUDNN_CHECK(cudnnSetResampleDescriptor(
        resample_desc_,
        CUDNN_RESAMPLE_BILINEAR,
        CUDNN_PROPAGATE_NAN,
        /*windowHeight=*/0, /*windowWidth=*/0,
        /*verticalPadding=*/0, /*horizontalPadding=*/0,
        /*verticalStride=*/0, /*horizontalStride=*/0
    ));
#endif
}

UpsampleLayer::~UpsampleLayer() {
#ifdef DENOISER_USE_CUDNN_RESAMPLE
    if (resample_desc_) cudnnDestroyResampleDescriptor(resample_desc_);
    if (cudnn_)         cudnnDestroy(cudnn_);
#endif
}

Tensor UpsampleLayer::forward(const Tensor& input, cudaStream_t stream) const {
    if (input.dtype() != DType::kFloat16) {
        throw std::runtime_error("UpsampleLayer::forward: expected FP16 input");
    }

    const int N    = static_cast<int>(input.batch());
    const int C    = static_cast<int>(input.channels());
    const int H_in = static_cast<int>(input.height());
    const int W_in = static_cast<int>(input.width());
    const int H_out = H_in * scale_factor_;
    const int W_out = W_in * scale_factor_;

    auto output = Tensor::empty({N, C, H_out, W_out}, DType::kFloat16);

#ifdef DENOISER_USE_CUDNN_RESAMPLE
    if (stream) CUDNN_CHECK(cudnnSetStream(cudnn_, stream));
    auto in_desc  = input.make_cudnn_descriptor();
    auto out_desc = output.make_cudnn_descriptor();
    const double alpha = 1.0;
    const double beta  = 0.0;
    CUDNN_CHECK(cudnnResampleForward(
        cudnn_, resample_desc_,
        &alpha, in_desc,  input.data(),
        &beta,  out_desc, output.data()
    ));
    cudnnDestroyTensorDescriptor(in_desc);
    cudnnDestroyTensorDescriptor(out_desc);
#else
    const int64_t total = static_cast<int64_t>(N) * C * H_out * W_out;
    constexpr int kBlockSize = 256;
    const int blocks = static_cast<int>((total + kBlockSize - 1) / kBlockSize);
    bilinear_upsample_kernel<<<blocks, kBlockSize, 0, stream>>>(
        input.data_f16(), output.data_f16(),
        N, C, H_in, W_in, H_out, W_out,
        static_cast<float>(scale_factor_), static_cast<float>(scale_factor_)
    );
#endif

    return output;
}

} // namespace denoiser
