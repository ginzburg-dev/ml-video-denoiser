#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG

#include "denoiser/io/image_io.h"
#include "denoiser/tensor.h"

// stb — header-only, single implementation per translation unit
#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace denoiser::io {

namespace {

// Convert uint8 HWC (H × W × C) to FP16 NCHW (1 × C × H × W), normalised /255.
std::vector<__half> hwc_u8_to_nchw_f16(
    const uint8_t* hwc, int H, int W, int C)
{
    const int64_t nchw_count = static_cast<int64_t>(C) * H * W;
    std::vector<__half> nchw(nchw_count);

    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                const uint8_t v = hwc[(h * W + w) * C + c];
                nchw[static_cast<int64_t>(c) * H * W + h * W + w] =
                    __float2half(static_cast<float>(v) / 255.f);
            }
        }
    }
    return nchw;
}

// Convert FP16 NCHW (C × H × W) to uint8 HWC, clamped [0, 255].
std::vector<uint8_t> nchw_f16_to_hwc_u8(
    const __half* nchw, int C, int H, int W)
{
    std::vector<uint8_t> hwc(static_cast<size_t>(H) * W * C);
    for (int c = 0; c < C; ++c) {
        for (int h = 0; h < H; ++h) {
            for (int w = 0; w < W; ++w) {
                const float f = __half2float(
                    nchw[static_cast<int64_t>(c) * H * W + h * W + w]);
                const int v = static_cast<int>(f * 255.f + 0.5f);
                hwc[(h * W + w) * C + c] =
                    static_cast<uint8_t>(std::clamp(v, 0, 255));
            }
        }
    }
    return hwc;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// load_image
// ---------------------------------------------------------------------------

Tensor load_image(const std::string& path, cudaStream_t stream) {
    int W = 0, H = 0, channels = 0;
    uint8_t* data = stbi_load(path.c_str(), &W, &H, &channels, 3);  // force RGB
    if (!data) {
        throw std::runtime_error("load_image: cannot read '" + path +
                                 "': " + stbi_failure_reason());
    }

    const int C = 3;
    std::unique_ptr<uint8_t, decltype(&stbi_image_free)> guard(data, stbi_image_free);

    auto nchw = hwc_u8_to_nchw_f16(data, H, W, C);
    return Tensor::from_host(nchw.data(), {1, C, H, W}, DType::kFloat16, stream);
}

// ---------------------------------------------------------------------------
// save_image
// ---------------------------------------------------------------------------

void save_image(const Tensor& tensor, const std::string& path,
                cudaStream_t stream)
{
    if (tensor.dtype() != DType::kFloat16) {
        throw std::runtime_error("save_image: expected FP16 input");
    }
    if (tensor.shape().size() != 4 || tensor.batch() != 1) {
        throw std::runtime_error("save_image: expected (1, C, H, W) tensor");
    }

    const int C = static_cast<int>(tensor.channels());
    const int H = static_cast<int>(tensor.height());
    const int W = static_cast<int>(tensor.width());

    // Download from GPU
    std::vector<__half> nchw(static_cast<size_t>(C) * H * W);
    tensor.to_host(nchw.data(), stream);
    if (stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto hwc = nchw_f16_to_hwc_u8(nchw.data(), C, H, W);

    // Determine format from extension
    const bool is_png = path.size() >= 4 &&
        (path.substr(path.size() - 4) == ".png" ||
         path.substr(path.size() - 4) == ".PNG");

    int ok = 0;
    if (is_png) {
        ok = stbi_write_png(path.c_str(), W, H, C, hwc.data(), W * C);
    } else {
        ok = stbi_write_jpg(path.c_str(), W, H, C, hwc.data(), /*quality=*/95);
    }

    if (!ok) {
        throw std::runtime_error("save_image: failed to write '" + path + "'");
    }
}

} // namespace denoiser::io
