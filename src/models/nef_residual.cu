#include "denoiser/models/nef_residual.h"

#include "denoiser/layers/conv2d.h"
#include "denoiser/layers/batchnorm2d.h"
#include "denoiser/layers/relu.h"
#include "denoiser/layers/upsample.h"
#include "denoiser/layers/maxpool2d.h"
#include "denoiser/kernels/model_kernels.h"

// concat launcher is defined in batchnorm2d.cu; declare it here to avoid
// including the .cuh file (which contains __global__ kernels).
namespace denoiser::kernels {
void launch_concat_channels(
    const __half* a, const __half* b, __half* output,
    int N, int C1, int C2, int H, int W, cudaStream_t stream);
} // namespace denoiser::kernels

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace denoiser {

// ---------------------------------------------------------------------------
// Private implementation (Pimpl)
// ---------------------------------------------------------------------------

struct NEFResidual::Impl {

    // --- Building blocks -----------------------------------------------------

    // Owning wrapper around Conv → BN → ReLU.
    // All members are unique_ptr so the struct is movable.
    struct ConvBnRelu {
        std::unique_ptr<Conv2dLayer>      conv;
        std::unique_ptr<BatchNorm2dLayer> bn;
        std::unique_ptr<ReLULayer>        relu;

        Tensor forward(const Tensor& x, cudaStream_t s) const {
            auto y = conv->forward(x, s);
            y = bn->forward(y, s);
            return relu->forward(y, s);
        }
    };

    // Encoder level: cbr1 → cbr2 → (MaxPool, skip)
    struct EncoderLevel {
        ConvBnRelu                     cbr1, cbr2;
        std::unique_ptr<MaxPool2dLayer> pool;

        // Returns (pooled_tensor, skip_tensor).
        // skip is the output of cbr2 (before pooling), pooled is after pooling.
        std::pair<Tensor, Tensor> forward(const Tensor& x, cudaStream_t s) const {
            auto features = cbr2.forward(cbr1.forward(x, s), s);
            // pool reads features but returns a separate allocation — features is
            // still valid and becomes the skip connection.
            auto pooled = pool->forward(features, s);
            return {std::move(pooled), std::move(features)};
        }
    };

    // Decoder level: upsample → concat(skip) → cbr1 → cbr2
    struct DecoderLevel {
        std::unique_ptr<UpsampleLayer> up;
        ConvBnRelu                     cbr1, cbr2;

        Tensor forward(const Tensor& x, const Tensor& skip, cudaStream_t s) const {
            auto upsampled = up->forward(x, s);

            // Sanity-check spatial dimensions match the skip connection.
            if (upsampled.height() != skip.height() || upsampled.width() != skip.width()) {
                throw std::runtime_error(
                    "DecoderLevel: upsample output size " +
                    upsampled.shape_str() + " does not match skip " + skip.shape_str());
            }

            const int N  = static_cast<int>(upsampled.batch());
            const int C1 = static_cast<int>(upsampled.channels());
            const int C2 = static_cast<int>(skip.channels());
            const int H  = static_cast<int>(upsampled.height());
            const int W  = static_cast<int>(upsampled.width());

            auto cat = Tensor::empty({N, C1 + C2, H, W}, DType::kFloat16);
            kernels::launch_concat_channels(
                upsampled.data_f16(), skip.data_f16(), cat.data_f16(),
                N, C1, C2, H, W, s);

            return cbr2.forward(cbr1.forward(cat, s), s);
        }
    };

    // --- Helpers -------------------------------------------------------------

    static ConvBnRelu make_cbr(const WeightStore& store,
                                const std::string& prefix) {
        ConvBnRelu cbr;
        // Conv2d: bias=False (BatchNorm subsumes the bias)
        cbr.conv = std::make_unique<Conv2dLayer>(
            store, prefix + ".conv.weight",
            std::nullopt,  // no bias
            /*pad=*/1, /*stride=*/1);
        cbr.bn   = std::make_unique<BatchNorm2dLayer>(store, prefix + ".bn");
        cbr.relu = std::make_unique<ReLULayer>();
        return cbr;
    }

    // --- Construction --------------------------------------------------------

    explicit Impl(const WeightStore& store) {
        const auto& arch = store.manifest().architecture;

        if (arch.enc_channels.empty()) {
            throw std::runtime_error("NEFResidual: enc_channels is empty in manifest");
        }

        enc_channels = arch.enc_channels;
        in_channels  = arch.in_channels;
        out_channels = arch.out_channels;
        pad_multiple = 1 << static_cast<int>(enc_channels.size()); // 2^num_levels

        // --- Encoder ---
        encoders.reserve(enc_channels.size());
        for (int lvl = 0; lvl < static_cast<int>(enc_channels.size()); ++lvl) {
            const std::string pfx = "encoders." + std::to_string(lvl);
            EncoderLevel enc;
            enc.cbr1 = make_cbr(store, pfx + ".conv1");
            enc.cbr2 = make_cbr(store, pfx + ".conv2");
            enc.pool = std::make_unique<MaxPool2dLayer>(/*kernel=*/2, /*stride=*/2);
            encoders.push_back(std::move(enc));
        }

        // --- Bottleneck ---
        bot1 = make_cbr(store, "bottleneck.0");
        bot2 = make_cbr(store, "bottleneck.1");

        // --- Decoder ---
        // Levels iterate enc_channels in reverse (deepest first):
        //   dec[0]: in_ch = enc[-1]*2, skip_ch = enc[-1], out_ch = enc[-1]
        //   dec[1]: in_ch = enc[-1],   skip_ch = enc[-2], out_ch = enc[-2]
        //   ...
        decoders.reserve(enc_channels.size());
        for (int lvl = 0; lvl < static_cast<int>(enc_channels.size()); ++lvl) {
            const std::string pfx = "decoders." + std::to_string(lvl);
            DecoderLevel dec;
            dec.up   = std::make_unique<UpsampleLayer>(/*scale_factor=*/2);
            dec.cbr1 = make_cbr(store, pfx + ".conv1");
            dec.cbr2 = make_cbr(store, pfx + ".conv2");
            decoders.push_back(std::move(dec));
        }

        // --- Head: 1×1 conv, with bias (no BN after head) ---
        head = std::make_unique<Conv2dLayer>(
            store, "head.weight",
            std::optional<std::string>("head.bias"),
            /*pad=*/0, /*stride=*/1);
    }

    // --- Data members --------------------------------------------------------

    std::vector<int>         enc_channels;
    int                      in_channels  = 3;
    int                      out_channels = 3;
    int                      pad_multiple = 16;

    std::vector<EncoderLevel> encoders;
    ConvBnRelu                bot1, bot2;
    std::vector<DecoderLevel> decoders;
    std::unique_ptr<Conv2dLayer> head;
};

// ---------------------------------------------------------------------------
// Padding helpers (host-side arithmetic + kernel dispatch)
// ---------------------------------------------------------------------------

namespace {

struct PaddingInfo { int pad_h, pad_w; };

// Reflect-pad *input* to the nearest multiple of *multiple* in H and W.
// Padding is applied to the right column / bottom row only (matching Python).
// Returns the padded tensor and the amount of padding added.
std::pair<Tensor, PaddingInfo> pad_to_multiple(
    const Tensor& input, int multiple, cudaStream_t stream)
{
    const int N    = static_cast<int>(input.batch());
    const int C    = static_cast<int>(input.channels());
    const int H_in = static_cast<int>(input.height());
    const int W_in = static_cast<int>(input.width());

    const int pad_h = (multiple - H_in % multiple) % multiple;
    const int pad_w = (multiple - W_in % multiple) % multiple;
    const int H_out = H_in + pad_h;
    const int W_out = W_in + pad_w;

    auto padded = Tensor::empty({N, C, H_out, W_out}, DType::kFloat16);
    kernels::launch_reflect_pad(
        input.data_f16(), padded.data_f16(),
        N, C, H_in, W_in, H_out, W_out, stream);

    return {std::move(padded), {pad_h, pad_w}};
}

// Strip the padding added by pad_to_multiple, restoring the original shape.
Tensor crop_to_original(
    const Tensor& padded, PaddingInfo padding,
    int orig_h, int orig_w, cudaStream_t stream)
{
    if (padding.pad_h == 0 && padding.pad_w == 0) {
        return padded.clone(stream);
    }

    const int N    = static_cast<int>(padded.batch());
    const int C    = static_cast<int>(padded.channels());
    const int H_in = static_cast<int>(padded.height());
    const int W_in = static_cast<int>(padded.width());

    auto out = Tensor::empty({N, C, orig_h, orig_w}, DType::kFloat16);
    kernels::launch_crop(
        padded.data_f16(), out.data_f16(),
        N, C, H_in, W_in, orig_h, orig_w, stream);
    return out;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// NEFResidual
// ---------------------------------------------------------------------------

NEFResidual::NEFResidual(const WeightStore& store)
    : impl_(std::make_unique<Impl>(store)) {}

NEFResidual::~NEFResidual() = default;

Tensor NEFResidual::forward(const Tensor& input, cudaStream_t stream) const {
    if (input.dtype() != DType::kFloat16) {
        throw std::runtime_error("NEFResidual::forward: expected FP16 input");
    }
    if (input.shape().size() != 4) {
        throw std::runtime_error("NEFResidual::forward: input must be 4-D NCHW");
    }

    const int orig_h = static_cast<int>(input.height());
    const int orig_w = static_cast<int>(input.width());

    // 1. Reflect-pad to multiple of 2^num_levels
    auto [padded, padding] = pad_to_multiple(input, impl_->pad_multiple, stream);

    // 2. Encoder — collect skip connections in order [level 0 .. num_levels-1]
    std::vector<Tensor> skips;
    skips.reserve(impl_->encoders.size());
    Tensor x = std::move(padded);
    for (auto& enc : impl_->encoders) {
        auto [pooled, skip] = enc.forward(x, stream);
        skips.push_back(std::move(skip));
        x = std::move(pooled);
    }

    // 3. Bottleneck
    x = impl_->bot2.forward(impl_->bot1.forward(x, stream), stream);

    // 4. Decoder — use skips in reverse order (deepest first)
    const int num_levels = static_cast<int>(impl_->decoders.size());
    for (int i = 0; i < num_levels; ++i) {
        const int skip_idx = num_levels - 1 - i;
        x = impl_->decoders[i].forward(x, skips[skip_idx], stream);
        // Free this skip immediately — it's no longer needed
        skips[skip_idx] = Tensor{};
    }

    // 5. Head: predict noise residual (still at padded size)
    auto noise_padded = impl_->head->forward(x, stream);

    // 6. Crop noise back to original spatial size
    auto noise = crop_to_original(noise_padded, padding, orig_h, orig_w, stream);

    // 7. Residual subtraction: output = clamp(input - noise, 0, 1)
    auto output = Tensor::empty(input.shape(), DType::kFloat16);
    kernels::launch_subtract_clamp(
        input.data_f16(), noise.data_f16(), output.data_f16(),
        input.numel(), stream);

    return output;
}

const std::vector<int>& NEFResidual::enc_channels() const noexcept {
    return impl_->enc_channels;
}

int NEFResidual::num_levels() const noexcept {
    return static_cast<int>(impl_->enc_channels.size());
}

int NEFResidual::in_channels() const noexcept  { return impl_->in_channels; }
int NEFResidual::out_channels() const noexcept { return impl_->out_channels; }

} // namespace denoiser
