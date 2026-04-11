#include "denoiser/models/nef_temporal.h"

#include "denoiser/layers/conv2d.h"
#include "denoiser/layers/batchnorm2d.h"
#include "denoiser/layers/relu.h"
#include "denoiser/layers/upsample.h"
#include "denoiser/layers/maxpool2d.h"
#include "denoiser/layers/deform_conv2d.h"
#include "denoiser/kernels/model_kernels.h"

// concat launcher (defined in batchnorm2d.cu)
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
// Private implementation
// ---------------------------------------------------------------------------

struct NEFTemporal::Impl {

    // ---- Reused building blocks (identical to NEFResidual::Impl) -----------

    struct ConvBnRelu {
        std::unique_ptr<Conv2dLayer>      conv;
        std::unique_ptr<BatchNorm2dLayer> bn;
        std::unique_ptr<ReLULayer>        relu;

        Tensor forward(const Tensor& x, cudaStream_t s) const {
            return relu->forward(bn->forward(conv->forward(x, s), s), s);
        }
    };

    struct EncoderLevel {
        ConvBnRelu                      cbr1, cbr2;
        std::unique_ptr<MaxPool2dLayer>  pool;

        std::pair<Tensor, Tensor> forward(const Tensor& x, cudaStream_t s) const {
            auto features = cbr2.forward(cbr1.forward(x, s), s);
            return {pool->forward(features, s), std::move(features)};
        }
    };

    struct DecoderLevel {
        std::unique_ptr<UpsampleLayer> up;
        ConvBnRelu cbr1, cbr2;

        Tensor forward(const Tensor& x, const Tensor& skip, cudaStream_t s) const {
            auto upsampled = up->forward(x, s);
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

    // ---- Deformable alignment (per encoder level, weight-shared across
    //      neighbour frames — i.e. one set of weights per level) ------------

    struct AlignLayer {
        std::unique_ptr<Conv2dLayer>       offset_conv;  // 2C → 2*dg*kH*kW, k=3
        std::unique_ptr<Conv2dLayer>       mask_conv;    // 2C → dg*kH*kW, k=3
        std::unique_ptr<DeformConv2dLayer> dcn;          // C → C, k=3

        // Align *neighbour* features towards *ref* features.
        // Returns aligned (N, C, H, W) FP16 tensor.
        Tensor forward(const Tensor& ref, const Tensor& nbr, cudaStream_t s) const {
            const int N  = static_cast<int>(ref.batch());
            const int C  = static_cast<int>(ref.channels());
            const int H  = static_cast<int>(ref.height());
            const int W  = static_cast<int>(ref.width());

            // Concatenate ref + neighbour along channel dim → (N, 2C, H, W)
            auto cat = Tensor::empty({N, 2 * C, H, W}, DType::kFloat16);
            kernels::launch_concat_channels(
                ref.data_f16(), nbr.data_f16(), cat.data_f16(),
                N, C, C, H, W, s);

            auto offsets = offset_conv->forward(cat, s);
            auto masks   = mask_conv->forward(cat, s);
            return dcn->forward(nbr, offsets, masks, s);
        }
    };

    // ---- Helpers -----------------------------------------------------------

    static ConvBnRelu make_cbr(const WeightStore& store,
                                const std::string& prefix) {
        ConvBnRelu cbr;
        cbr.conv = std::make_unique<Conv2dLayer>(
            store, prefix + ".conv.weight", std::nullopt, /*pad=*/1);
        cbr.bn   = std::make_unique<BatchNorm2dLayer>(store, prefix + ".bn");
        cbr.relu = std::make_unique<ReLULayer>();
        return cbr;
    }

    // ---- Construction ------------------------------------------------------

    explicit Impl(const WeightStore& store) {
        const auto& arch = store.manifest().architecture;

        if (arch.enc_channels.empty()) {
            throw std::runtime_error(
                "NEFTemporal: enc_channels is empty in manifest");
        }

        enc_ch_      = arch.enc_channels;
        num_frames_  = arch.num_frames > 0 ? arch.num_frames : 5;
        ref_idx_     = num_frames_ / 2;
        deform_groups_ = arch.deform_groups > 0 ? arch.deform_groups : 8;
        in_channels_ = arch.in_channels;
        out_channels_= arch.out_channels;
        pad_multiple_= 1 << static_cast<int>(enc_ch_.size());

        const int num_levels = static_cast<int>(enc_ch_.size());

        // --- Shared encoder ---
        encoders_.reserve(num_levels);
        for (int lvl = 0; lvl < num_levels; ++lvl) {
            const std::string pfx = "encoders." + std::to_string(lvl);
            EncoderLevel enc;
            enc.cbr1 = make_cbr(store, pfx + ".conv1");
            enc.cbr2 = make_cbr(store, pfx + ".conv2");
            enc.pool = std::make_unique<MaxPool2dLayer>(2, 2);
            encoders_.push_back(std::move(enc));
        }

        // --- Deformable alignment per level ---
        align_.reserve(num_levels);
        for (int lvl = 0; lvl < num_levels; ++lvl) {
            const std::string pfx = "align_layers." + std::to_string(lvl);
            const int ch  = enc_ch_[lvl];
            const int dg  = deform_groups_;
            const int k   = 3;
            AlignLayer al;
            al.offset_conv = std::make_unique<Conv2dLayer>(
                store, pfx + ".offset_conv.weight",
                std::optional<std::string>(pfx + ".offset_conv.bias"),
                /*pad=*/1);
            al.mask_conv = std::make_unique<Conv2dLayer>(
                store, pfx + ".mask_conv.weight",
                std::optional<std::string>(pfx + ".mask_conv.bias"),
                /*pad=*/1);
            al.dcn = std::make_unique<DeformConv2dLayer>(
                store, pfx + ".weight", pfx + ".bias",
                /*pad=*/1, /*stride=*/1, /*dilation=*/1, dg);
            align_.push_back(std::move(al));
        }

        // --- Temporal fusion per level: T*C → C (1×1 conv with bias) ---
        fusion_.reserve(num_levels);
        for (int lvl = 0; lvl < num_levels; ++lvl) {
            const std::string pfx = "fusion_layers." + std::to_string(lvl);
            fusion_.push_back(std::make_unique<Conv2dLayer>(
                store, pfx + ".weight",
                std::optional<std::string>(pfx + ".bias"),
                /*pad=*/0));
        }

        // --- Bottleneck ---
        bot1_ = make_cbr(store, "bottleneck.0");
        bot2_ = make_cbr(store, "bottleneck.1");

        // --- Decoder ---
        decoders_.reserve(num_levels);
        for (int lvl = 0; lvl < num_levels; ++lvl) {
            const std::string pfx = "decoders." + std::to_string(lvl);
            DecoderLevel dec;
            dec.up   = std::make_unique<UpsampleLayer>(2);
            dec.cbr1 = make_cbr(store, pfx + ".conv1");
            dec.cbr2 = make_cbr(store, pfx + ".conv2");
            decoders_.push_back(std::move(dec));
        }

        // --- Head ---
        head_ = std::make_unique<Conv2dLayer>(
            store, "head.weight",
            std::optional<std::string>("head.bias"),
            /*pad=*/0);
    }

    // ---- Data members ------------------------------------------------------

    std::vector<int>   enc_ch_;
    int num_frames_    = 5;
    int ref_idx_       = 2;
    int deform_groups_ = 8;
    int in_channels_   = 3;
    int out_channels_  = 3;
    int pad_multiple_  = 16;

    std::vector<EncoderLevel>             encoders_;
    std::vector<AlignLayer>               align_;
    std::vector<std::unique_ptr<Conv2dLayer>> fusion_;
    ConvBnRelu                            bot1_, bot2_;
    std::vector<DecoderLevel>             decoders_;
    std::unique_ptr<Conv2dLayer>          head_;
};

// ---------------------------------------------------------------------------
// Padding helpers (same as NEFResidual)
// ---------------------------------------------------------------------------

namespace {

struct PaddingInfo { int pad_h, pad_w; };

std::pair<Tensor, PaddingInfo> pad_to_multiple(
    const Tensor& input, int multiple, cudaStream_t stream)
{
    const int N    = static_cast<int>(input.batch());
    const int C    = static_cast<int>(input.channels());
    const int H_in = static_cast<int>(input.height());
    const int W_in = static_cast<int>(input.width());
    const int pad_h = (multiple - H_in % multiple) % multiple;
    const int pad_w = (multiple - W_in % multiple) % multiple;
    auto padded = Tensor::empty({N, C, H_in + pad_h, W_in + pad_w}, DType::kFloat16);
    kernels::launch_reflect_pad(
        input.data_f16(), padded.data_f16(),
        N, C, H_in, W_in, H_in + pad_h, W_in + pad_w, stream);
    return std::make_pair(std::move(padded), PaddingInfo{pad_h, pad_w});
}

Tensor crop_to_original(
    const Tensor& padded, PaddingInfo padding,
    int orig_h, int orig_w, cudaStream_t stream)
{
    if (padding.pad_h == 0 && padding.pad_w == 0) return padded.clone(stream);
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

// Concat T tensors along channel dimension.
// tensors: T tensors each (N, C, H, W) — all same shape.
// Returns (N, T*C, H, W).
Tensor concat_frames(
    const std::vector<Tensor>& tensors, cudaStream_t stream)
{
    const int T  = static_cast<int>(tensors.size());
    const int N  = static_cast<int>(tensors[0].batch());
    const int C  = static_cast<int>(tensors[0].channels());
    const int H  = static_cast<int>(tensors[0].height());
    const int W  = static_cast<int>(tensors[0].width());

    auto result = Tensor::empty({N, T * C, H, W}, DType::kFloat16);
    // Copy each frame's channels into the correct slice of result
    const size_t ch_bytes = static_cast<size_t>(N) * C * H * W * sizeof(__half);
    for (int t = 0; t < T; ++t) {
        __half* dst = result.data_f16() + static_cast<int64_t>(N) * C * H * W * t;
        CUDA_CHECK(cudaMemcpyAsync(
            dst, tensors[t].data_f16(), ch_bytes,
            cudaMemcpyDeviceToDevice, stream));
    }
    return result;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// NEFTemporal
// ---------------------------------------------------------------------------

NEFTemporal::NEFTemporal(const WeightStore& store)
    : impl_(std::make_unique<Impl>(store)) {}

NEFTemporal::~NEFTemporal() = default;

Tensor NEFTemporal::forward(const Tensor& clip, cudaStream_t stream) const {
    if (clip.dtype() != DType::kFloat16) {
        throw std::runtime_error("NEFTemporal::forward: expected FP16 input");
    }
    if (clip.shape().size() != 5) {
        throw std::runtime_error(
            "NEFTemporal::forward: expected 5-D (N, T, C, H, W) clip");
    }

    const int N = static_cast<int>(clip.shape()[0]);
    const int T = static_cast<int>(clip.shape()[1]);
    const int C = static_cast<int>(clip.shape()[2]);
    const int H = static_cast<int>(clip.shape()[3]);
    const int W = static_cast<int>(clip.shape()[4]);

    if (T != impl_->num_frames_) {
        throw std::runtime_error(
            "NEFTemporal::forward: expected T=" +
            std::to_string(impl_->num_frames_) + " frames, got " +
            std::to_string(T));
    }

    const int num_levels = static_cast<int>(impl_->encoders_.size());
    const int ref_idx    = impl_->ref_idx_;

    // Extract the (unpadded) reference frame for residual subtraction at the end.
    // clip is (N, T, C, H, W) stored contiguously — slice out ref frame.
    // Non-owning slice not available for 5-D; clone a contiguous FP16 block.
    const int64_t frame_elems = static_cast<int64_t>(N) * C * H * W;
    // View of reference frame: offset in the flat clip buffer
    // Layout: [n][t][c][h][w] → linear offset for frame t = t * frame_elems
    Tensor ref_input = Tensor::empty({N, C, H, W}, DType::kFloat16);
    CUDA_CHECK(cudaMemcpyAsync(
        ref_input.data_f16(),
        clip.data_f16() + static_cast<int64_t>(ref_idx) * frame_elems,
        static_cast<size_t>(frame_elems) * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream));

    // 1. Pad all frames to a multiple of pad_multiple
    std::vector<Tensor> frames_padded;
    frames_padded.reserve(T);
    PaddingInfo padding{};
    for (int t = 0; t < T; ++t) {
        Tensor frame = Tensor::empty({N, C, H, W}, DType::kFloat16);
        CUDA_CHECK(cudaMemcpyAsync(
            frame.data_f16(),
            clip.data_f16() + static_cast<int64_t>(t) * frame_elems,
            static_cast<size_t>(frame_elems) * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream));
        auto [padded, pad_info] = pad_to_multiple(frame, impl_->pad_multiple_, stream);
        padding = pad_info;
        frames_padded.push_back(std::move(padded));
    }

    // 2. Shared encoder: process all frames together as a big batch (B*T, C, H, W)
    //    Collect skip connections per level per frame.
    //    all_skips[level][frame] = (N, C_lvl, H_lvl, W_lvl)

    // Stack all frames into a single batch tensor
    const int pH = static_cast<int>(frames_padded[0].height());
    const int pW = static_cast<int>(frames_padded[0].width());
    const int BT = N * T;

    auto stacked = Tensor::empty({BT, C, pH, pW}, DType::kFloat16);
    for (int t = 0; t < T; ++t) {
        CUDA_CHECK(cudaMemcpyAsync(
            stacked.data_f16() + static_cast<int64_t>(t) * N * C * pH * pW,
            frames_padded[t].data_f16(),
            static_cast<size_t>(N) * C * pH * pW * sizeof(__half),
            cudaMemcpyDeviceToDevice, stream));
    }
    frames_padded.clear();

    // all_skips[level] = vector<Tensor>(T), each (N, C_lvl, H_lvl, W_lvl)
    std::vector<std::vector<Tensor>> all_skips(num_levels);
    Tensor x = std::move(stacked);

    for (int lvl = 0; lvl < num_levels; ++lvl) {
        auto [pooled, skip_bt] = impl_->encoders_[lvl].forward(x, stream);
        // skip_bt is (BT, C_lvl, H_lvl, W_lvl) — split into T per-frame tensors of (N, ...)
        const int64_t cp = skip_bt.channels();
        const int64_t hp = skip_bt.height();
        const int64_t wp = skip_bt.width();
        const int64_t frame_skip_elems = static_cast<int64_t>(N) * cp * hp * wp;
        all_skips[lvl].resize(T);
        for (int t = 0; t < T; ++t) {
            all_skips[lvl][t] = Tensor::empty({N, cp, hp, wp}, DType::kFloat16);
            CUDA_CHECK(cudaMemcpyAsync(
                all_skips[lvl][t].data_f16(),
                skip_bt.data_f16() + static_cast<int64_t>(t) * frame_skip_elems,
                static_cast<size_t>(frame_skip_elems) * sizeof(__half),
                cudaMemcpyDeviceToDevice, stream));
        }
        x = std::move(pooled);
    }

    // 3. Per-level deformable alignment + temporal fusion
    std::vector<Tensor> fused_skips(num_levels);
    for (int lvl = 0; lvl < num_levels; ++lvl) {
        const Tensor& ref_feat = all_skips[lvl][ref_idx];

        std::vector<Tensor> aligned;
        aligned.reserve(T);
        for (int t = 0; t < T; ++t) {
            if (t == ref_idx) {
                aligned.push_back(ref_feat.clone(stream));
            } else {
                aligned.push_back(
                    impl_->align_[lvl].forward(ref_feat, all_skips[lvl][t], stream));
            }
        }
        all_skips[lvl].clear();  // free per-level skips

        // Concat all T aligned frames along channel dim → (N, T*C_lvl, H_lvl, W_lvl)
        auto cat = concat_frames(aligned, stream);
        // 1×1 fusion conv → (N, C_lvl, H_lvl, W_lvl)
        fused_skips[lvl] = impl_->fusion_[lvl]->forward(cat, stream);
    }

    // 4. Bottleneck on reference frame's deepest features
    //    x is (BT, C_deep, H_deep, W_deep) — extract reference frame's slice
    const int64_t ref_deep_elems =
        static_cast<int64_t>(N) * x.channels() * x.height() * x.width();
    Tensor ref_deep = Tensor::empty(
        {N, x.channels(), x.height(), x.width()}, DType::kFloat16);
    CUDA_CHECK(cudaMemcpyAsync(
        ref_deep.data_f16(),
        x.data_f16() + static_cast<int64_t>(ref_idx) * ref_deep_elems,
        static_cast<size_t>(ref_deep_elems) * sizeof(__half),
        cudaMemcpyDeviceToDevice, stream));
    x = impl_->bot2_.forward(impl_->bot1_.forward(ref_deep, stream), stream);

    // 5. Decoder (deepest skip first)
    for (int i = 0; i < num_levels; ++i) {
        const int skip_idx = num_levels - 1 - i;
        x = impl_->decoders_[i].forward(x, fused_skips[skip_idx], stream);
        fused_skips[skip_idx] = Tensor{};  // free eagerly
    }

    // 6. Head + crop + residual subtraction
    auto noise_padded = impl_->head_->forward(x, stream);
    auto noise = crop_to_original(noise_padded, padding, H, W, stream);

    auto output = Tensor::empty({N, C, H, W}, DType::kFloat16);
    kernels::launch_subtract_clamp(
        ref_input.data_f16(), noise.data_f16(), output.data_f16(),
        ref_input.numel(), stream);

    return output;
}

int NEFTemporal::num_frames() const noexcept { return impl_->num_frames_; }
int NEFTemporal::num_levels() const noexcept {
    return static_cast<int>(impl_->enc_ch_.size());
}
const std::vector<int>& NEFTemporal::enc_channels() const noexcept {
    return impl_->enc_ch_;
}

} // namespace denoiser
