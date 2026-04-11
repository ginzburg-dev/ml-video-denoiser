#define TINYEXR_IMPLEMENTATION

#include "denoiser/io/exr_io.h"
#include "denoiser/tensor.h"

#include "tinyexr/tinyexr.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace denoiser::io {

namespace {

// Channel name priority lists (tinyexr provides per-channel buffers).
// We try R/G/B first, then Y for greyscale.
constexpr const char* kRgbNames[3] = {"R", "G", "B"};
constexpr const char* kGrayNames[1] = {"Y"};

// Find the index of a channel by name (case-sensitive), returns -1 if absent.
int find_channel(const EXRHeader& header, const char* name) {
    for (int i = 0; i < header.num_channels; ++i) {
        if (std::strcmp(header.channels[i].name, name) == 0) return i;
    }
    return -1;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// load_exr
// ---------------------------------------------------------------------------

Tensor load_exr(const std::string& path, cudaStream_t stream) {
    EXRVersion version;
    if (ParseEXRVersionFromFile(&version, path.c_str()) != TINYEXR_SUCCESS) {
        throw std::runtime_error("load_exr: not a valid EXR file: " + path);
    }

    EXRHeader header;
    InitEXRHeader(&header);
    const char* err = nullptr;
    if (ParseEXRHeaderFromFile(&header, &version, path.c_str(), &err) != TINYEXR_SUCCESS) {
        std::string msg = err ? err : "unknown";
        FreeEXRErrorMessage(err);
        throw std::runtime_error("load_exr: failed to parse header: " + msg);
    }

    // Request all channels as float32 for maximum precision during the load.
    for (int i = 0; i < header.num_channels; ++i) {
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_FLOAT;
    }

    EXRImage image;
    InitEXRImage(&image);
    if (LoadEXRImageFromFile(&image, &header, path.c_str(), &err) != TINYEXR_SUCCESS) {
        std::string msg = err ? err : "unknown";
        FreeEXRErrorMessage(err);
        FreeEXRHeader(&header);
        throw std::runtime_error("load_exr: failed to load image: " + msg);
    }

    const int W = image.width;
    const int H = image.height;

    // Determine channel layout: RGB or greyscale.
    int r_idx = find_channel(header, "R");
    int g_idx = find_channel(header, "G");
    int b_idx = find_channel(header, "B");
    int y_idx = find_channel(header, "Y");

    int C = 0;
    std::vector<int> ch_order;  // indices into image.images[] for each output channel
    if (r_idx >= 0 && g_idx >= 0 && b_idx >= 0) {
        C = 3;
        ch_order = {r_idx, g_idx, b_idx};
    } else if (y_idx >= 0) {
        C = 1;
        ch_order = {y_idx};
    } else if (header.num_channels >= 1) {
        // Fallback: use first channel
        C = 1;
        ch_order = {0};
    } else {
        FreeEXRImage(&image);
        FreeEXRHeader(&header);
        throw std::runtime_error("load_exr: no usable channels in " + path);
    }

    // Build NCHW FP16 buffer
    const int64_t nchw_count = static_cast<int64_t>(C) * H * W;
    std::vector<__half> nchw(nchw_count);

    for (int c = 0; c < C; ++c) {
        const auto* src = reinterpret_cast<const float*>(image.images[ch_order[c]]);
        __half* dst = nchw.data() + static_cast<int64_t>(c) * H * W;
        for (int i = 0; i < H * W; ++i) {
            dst[i] = __float2half(src[i]);
        }
    }

    FreeEXRImage(&image);
    FreeEXRHeader(&header);

    return Tensor::from_host(nchw.data(), {1, C, H, W}, DType::kFloat16, stream);
}

// ---------------------------------------------------------------------------
// save_exr
// ---------------------------------------------------------------------------

void save_exr(const Tensor& tensor, const std::string& path,
              cudaStream_t stream)
{
    if (tensor.dtype() != DType::kFloat16) {
        throw std::runtime_error("save_exr: expected FP16 input");
    }
    if (tensor.shape().size() != 4 || tensor.batch() != 1) {
        throw std::runtime_error("save_exr: expected (1, C, H, W) tensor");
    }

    const int C = static_cast<int>(tensor.channels());
    const int H = static_cast<int>(tensor.height());
    const int W = static_cast<int>(tensor.width());

    if (C != 1 && C != 3) {
        throw std::runtime_error("save_exr: only C=1 or C=3 supported, got C=" +
                                 std::to_string(C));
    }

    // Download from GPU
    std::vector<__half> nchw(static_cast<size_t>(C) * H * W);
    tensor.to_host(nchw.data(), stream);
    if (stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Convert NCHW FP16 → per-channel float32 arrays (tinyexr expects planar layout)
    std::vector<std::vector<float>> ch_data(C, std::vector<float>(H * W));
    for (int c = 0; c < C; ++c) {
        const __half* src = nchw.data() + static_cast<int64_t>(c) * H * W;
        for (int i = 0; i < H * W; ++i) {
            ch_data[c][i] = __half2float(src[i]);
        }
    }

    // Build tinyexr structures
    EXRHeader header;
    InitEXRHeader(&header);
    EXRImage  image;
    InitEXRImage(&image);

    // Channel names: tinyexr stores channels in B, G, R order for RGB EXR
    // (alphabetical by convention).  We reorder accordingly.
    std::vector<const char*> ch_names;
    std::vector<int>         write_order;  // index into ch_data[]
    if (C == 3) {
        ch_names    = {"B", "G", "R"};
        write_order = {2, 1, 0};  // ch_data[0]=R, [1]=G, [2]=B → write BGR
    } else {
        ch_names    = {"Y"};
        write_order = {0};
    }

    const int num_ch = C;
    std::vector<float*> image_ptrs(num_ch);
    for (int i = 0; i < num_ch; ++i) {
        image_ptrs[i] = ch_data[write_order[i]].data();
    }

    image.images    = reinterpret_cast<unsigned char**>(image_ptrs.data());
    image.width     = W;
    image.height    = H;
    image.num_channels = num_ch;

    header.num_channels = num_ch;
    header.channels = static_cast<EXRChannelInfo*>(
        malloc(sizeof(EXRChannelInfo) * num_ch));
    header.pixel_types          = static_cast<int*>(malloc(sizeof(int) * num_ch));
    header.requested_pixel_types= static_cast<int*>(malloc(sizeof(int) * num_ch));

    for (int i = 0; i < num_ch; ++i) {
        strncpy(header.channels[i].name, ch_names[i], 255);
        header.pixel_types[i]           = TINYEXR_PIXELTYPE_FLOAT;
        header.requested_pixel_types[i] = TINYEXR_PIXELTYPE_HALF;  // write as FP16
    }

    const char* err = nullptr;
    const int ret = SaveEXRImageToFile(&image, &header, path.c_str(), &err);
    const std::string err_msg = err ? err : "unknown";
    FreeEXRErrorMessage(err);
    FreeEXRHeader(&header);

    if (ret != TINYEXR_SUCCESS) {
        throw std::runtime_error("save_exr: failed to write '" + path +
                                 "': " + err_msg);
    }
}

} // namespace denoiser::io
