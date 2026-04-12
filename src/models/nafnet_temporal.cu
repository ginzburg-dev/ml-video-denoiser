#include "denoiser/models/nafnet_temporal.h"

#include "denoiser/kernels/model_kernels.h"
#include "denoiser/kernels/nafnet_kernels.h"
#include "denoiser/layers/conv2d.h"

#include <cuda_fp16.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace denoiser {

namespace {

struct PaddingInfo {
    int pad_h = 0;
    int pad_w = 0;
};

Tensor crop_to_size(const Tensor& input, int out_h, int out_w, cudaStream_t stream) {
    if (input.height() == out_h && input.width() == out_w) {
        return input.clone(stream);
    }
    auto out = Tensor::empty(
        {input.batch(), input.channels(), out_h, out_w}, DType::kFloat16);
    kernels::launch_crop(
        input.data_f16(), out.data_f16(),
        static_cast<int>(input.batch()),
        static_cast<int>(input.channels()),
        static_cast<int>(input.height()),
        static_cast<int>(input.width()),
        out_h, out_w, stream);
    return out;
}

std::pair<Tensor, PaddingInfo> pad_to_multiple(
    const Tensor& input, int multiple, cudaStream_t stream
) {
    const int N = static_cast<int>(input.batch());
    const int C = static_cast<int>(input.channels());
    const int H = static_cast<int>(input.height());
    const int W = static_cast<int>(input.width());
    const int pad_h = (multiple - H % multiple) % multiple;
    const int pad_w = (multiple - W % multiple) % multiple;
    auto out = Tensor::empty({N, C, H + pad_h, W + pad_w}, DType::kFloat16);
    kernels::launch_reflect_pad(
        input.data_f16(), out.data_f16(), N, C, H, W, H + pad_h, W + pad_w, stream);
    return {std::move(out), PaddingInfo{pad_h, pad_w}};
}

Tensor unpad(const Tensor& input, PaddingInfo padding, int out_h, int out_w, cudaStream_t stream) {
    if (padding.pad_h == 0 && padding.pad_w == 0) {
        return input.clone(stream);
    }
    return crop_to_size(input, out_h, out_w, stream);
}

Tensor select_beauty(const Tensor& input, cudaStream_t stream) {
    auto out = Tensor::empty({input.batch(), 3, input.height(), input.width()}, DType::kFloat16);
    kernels::launch_select_prefix_channels(
        input.data_f16(), out.data_f16(),
        static_cast<int>(input.batch()),
        static_cast<int>(input.channels()),
        3,
        static_cast<int>(input.height()),
        static_cast<int>(input.width()),
        stream);
    return out;
}

int count_levels(const WeightStore& store, const std::string& root) {
    int levels = 0;
    while (store.contains(root + "." + std::to_string(levels) + ".0.conv1.weight")) {
        ++levels;
    }
    return levels;
}

int count_blocks(const WeightStore& store, const std::string& root) {
    int blocks = 0;
    while (store.contains(root + "." + std::to_string(blocks) + ".conv1.weight")) {
        ++blocks;
    }
    return blocks;
}

bool tensor_is_fp32(const Tensor& t) {
    return t.dtype() == DType::kFloat32;
}

Tensor add_tensors(const Tensor& a, const Tensor& b, cudaStream_t stream) {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("add_tensors: shape mismatch " + a.shape_str() + " vs " + b.shape_str());
    }
    auto out = Tensor::empty(a.shape(), DType::kFloat16);
    kernels::launch_add(a.data_f16(), b.data_f16(), out.data_f16(), a.numel(), stream);
    return out;
}

struct NAFBlock {
    const Tensor* norm1_weight = nullptr;
    const Tensor* norm1_bias = nullptr;
    const Tensor* norm2_weight = nullptr;
    const Tensor* norm2_bias = nullptr;
    const Tensor* beta = nullptr;
    const Tensor* gamma = nullptr;
    std::unique_ptr<Conv2dLayer> conv1;
    std::unique_ptr<Conv2dLayer> conv2;
    std::unique_ptr<Conv2dLayer> sca_conv;
    std::unique_ptr<Conv2dLayer> conv3;
    std::unique_ptr<Conv2dLayer> conv4;
    std::unique_ptr<Conv2dLayer> conv5;
    int channels = 0;

    NAFBlock(const WeightStore& store, const std::string& prefix) {
        norm1_weight = &store.get(prefix + ".norm1.weight");
        norm1_bias = &store.get(prefix + ".norm1.bias");
        norm2_weight = &store.get(prefix + ".norm2.weight");
        norm2_bias = &store.get(prefix + ".norm2.bias");
        beta = &store.get(prefix + ".beta");
        gamma = &store.get(prefix + ".gamma");
        channels = static_cast<int>(norm1_weight->numel());

        conv1 = std::make_unique<Conv2dLayer>(
            store, prefix + ".conv1.weight", std::optional<std::string>(prefix + ".conv1.bias"),
            /*pad=*/0, /*stride=*/1, /*dilation=*/1, /*groups=*/1);

        const auto& conv2_weight = store.get(prefix + ".conv2.weight");
        const int conv2_groups = static_cast<int>(conv2_weight.shape()[0]);
        conv2 = std::make_unique<Conv2dLayer>(
            store, prefix + ".conv2.weight", std::optional<std::string>(prefix + ".conv2.bias"),
            /*pad=*/1, /*stride=*/1, /*dilation=*/1, conv2_groups);

        sca_conv = std::make_unique<Conv2dLayer>(
            store, prefix + ".sca.1.weight", std::optional<std::string>(prefix + ".sca.1.bias"),
            /*pad=*/0, /*stride=*/1, /*dilation=*/1, /*groups=*/1);
        conv3 = std::make_unique<Conv2dLayer>(
            store, prefix + ".conv3.weight", std::optional<std::string>(prefix + ".conv3.bias"),
            /*pad=*/0, /*stride=*/1, /*dilation=*/1, /*groups=*/1);
        conv4 = std::make_unique<Conv2dLayer>(
            store, prefix + ".conv4.weight", std::optional<std::string>(prefix + ".conv4.bias"),
            /*pad=*/0, /*stride=*/1, /*dilation=*/1, /*groups=*/1);
        conv5 = std::make_unique<Conv2dLayer>(
            store, prefix + ".conv5.weight", std::optional<std::string>(prefix + ".conv5.bias"),
            /*pad=*/0, /*stride=*/1, /*dilation=*/1, /*groups=*/1);
    }

    Tensor forward(const Tensor& input, cudaStream_t stream) const {
        auto x = Tensor::empty(input.shape(), DType::kFloat16);
        kernels::launch_layer_norm_affine(
            input.data_f16(), norm1_weight->data(), norm1_bias->data(), tensor_is_fp32(*norm1_weight),
            x.data_f16(),
            static_cast<int>(input.batch()), channels,
            static_cast<int>(input.height()), static_cast<int>(input.width()),
            stream);

        x = conv1->forward(x, stream);
        x = conv2->forward(x, stream);
        auto gated = Tensor::empty({x.batch(), x.channels() / 2, x.height(), x.width()}, DType::kFloat16);
        kernels::launch_simple_gate(
            x.data_f16(), gated.data_f16(),
            static_cast<int>(x.batch()), static_cast<int>(gated.channels()),
            static_cast<int>(x.height()), static_cast<int>(x.width()),
            stream);

        auto pooled = Tensor::empty({gated.batch(), gated.channels(), 1, 1}, DType::kFloat16);
        kernels::launch_global_avg_pool(
            gated.data_f16(), pooled.data_f16(),
            static_cast<int>(gated.batch()), static_cast<int>(gated.channels()),
            static_cast<int>(gated.height()), static_cast<int>(gated.width()),
            stream);
        auto sca = sca_conv->forward(pooled, stream);
        auto attended = Tensor::empty(gated.shape(), DType::kFloat16);
        kernels::launch_mul_channelwise(
            gated.data_f16(), sca.data_f16(), attended.data_f16(),
            static_cast<int>(gated.batch()), static_cast<int>(gated.channels()),
            static_cast<int>(gated.height()), static_cast<int>(gated.width()),
            stream);

        auto residual1 = conv3->forward(attended, stream);
        auto after1 = Tensor::empty(input.shape(), DType::kFloat16);
        kernels::launch_scaled_add(
            input.data_f16(), residual1.data_f16(), beta->data(), tensor_is_fp32(*beta),
            after1.data_f16(),
            static_cast<int>(input.batch()), channels,
            static_cast<int>(input.height()), static_cast<int>(input.width()),
            stream);

        auto norm2 = Tensor::empty(after1.shape(), DType::kFloat16);
        kernels::launch_layer_norm_affine(
            after1.data_f16(), norm2_weight->data(), norm2_bias->data(), tensor_is_fp32(*norm2_weight),
            norm2.data_f16(),
            static_cast<int>(after1.batch()), channels,
            static_cast<int>(after1.height()), static_cast<int>(after1.width()),
            stream);
        auto ff = conv4->forward(norm2, stream);
        auto ff_gated = Tensor::empty({ff.batch(), ff.channels() / 2, ff.height(), ff.width()}, DType::kFloat16);
        kernels::launch_simple_gate(
            ff.data_f16(), ff_gated.data_f16(),
            static_cast<int>(ff.batch()), static_cast<int>(ff_gated.channels()),
            static_cast<int>(ff.height()), static_cast<int>(ff.width()),
            stream);
        auto residual2 = conv5->forward(ff_gated, stream);
        auto out = Tensor::empty(after1.shape(), DType::kFloat16);
        kernels::launch_scaled_add(
            after1.data_f16(), residual2.data_f16(), gamma->data(), tensor_is_fp32(*gamma),
            out.data_f16(),
            static_cast<int>(after1.batch()), channels,
            static_cast<int>(after1.height()), static_cast<int>(after1.width()),
            stream);
        return out;
    }
};

struct NAFDownsample {
    std::unique_ptr<Conv2dLayer> body;

    NAFDownsample(const WeightStore& store, const std::string& prefix) {
        body = std::make_unique<Conv2dLayer>(
            store, prefix + ".body.weight", std::optional<std::string>(prefix + ".body.bias"),
            /*pad=*/0, /*stride=*/2, /*dilation=*/1, /*groups=*/1);
    }

    Tensor forward(const Tensor& input, cudaStream_t stream) const {
        return body->forward(input, stream);
    }
};

struct NAFUpsample {
    std::unique_ptr<Conv2dLayer> conv;

    NAFUpsample(const WeightStore& store, const std::string& prefix) {
        conv = std::make_unique<Conv2dLayer>(
            store, prefix + ".body.0.weight", std::nullopt,
            /*pad=*/0, /*stride=*/1, /*dilation=*/1, /*groups=*/1);
    }

    Tensor forward(const Tensor& input, cudaStream_t stream) const {
        auto expanded = conv->forward(input, stream);
        auto out = Tensor::empty(
            {expanded.batch(), expanded.channels() / 4, expanded.height() * 2, expanded.width() * 2},
            DType::kFloat16);
        kernels::launch_pixel_shuffle2x(
            expanded.data_f16(), out.data_f16(),
            static_cast<int>(expanded.batch()),
            static_cast<int>(expanded.channels()),
            static_cast<int>(expanded.height()),
            static_cast<int>(expanded.width()),
            stream);
        return out;
    }
};

}  // namespace

struct NAFNetTemporal::Impl {
    std::vector<int> enc_channels;
    int num_frames = 5;
    int ref_idx = 2;
    int in_channels = 3;
    int pad_multiple = 1;
    bool use_warp = false;

    std::unique_ptr<Conv2dLayer> intro;
    std::unique_ptr<Conv2dLayer> ending;
    std::vector<std::vector<NAFBlock>> encoders;
    std::vector<NAFDownsample> downs;
    std::vector<NAFBlock> middle;
    std::vector<NAFUpsample> ups;
    std::vector<std::vector<NAFBlock>> decoders;
    std::vector<std::unique_ptr<Conv2dLayer>> temporal_mix;
    std::vector<std::unique_ptr<Conv2dLayer>> offset_heads;

    explicit Impl(const WeightStore& store) {
        const auto& intro_weight = store.get("intro.weight");
        in_channels = static_cast<int>(intro_weight.shape()[1]);

        intro = std::make_unique<Conv2dLayer>(
            store, "intro.weight", std::optional<std::string>("intro.bias"),
            /*pad=*/1, /*stride=*/1, /*dilation=*/1, /*groups=*/1);
        ending = std::make_unique<Conv2dLayer>(
            store, "ending.weight", std::optional<std::string>("ending.bias"),
            /*pad=*/1, /*stride=*/1, /*dilation=*/1, /*groups=*/1);

        const int levels = count_levels(store, "encoders");
        if (levels <= 0) {
            throw std::runtime_error("NAFNetTemporal: no encoder levels found in exported weights");
        }
        pad_multiple = 1 << levels;

        encoders.reserve(levels);
        downs.reserve(levels);
        for (int level = 0; level < levels; ++level) {
            const std::string root = "encoders." + std::to_string(level);
            const int blocks = count_blocks(store, root);
            std::vector<NAFBlock> stage;
            stage.reserve(blocks);
            for (int block = 0; block < blocks; ++block) {
                stage.emplace_back(store, root + "." + std::to_string(block));
            }
            const auto& w = store.get(root + ".0.conv1.weight");
            enc_channels.push_back(static_cast<int>(w.shape()[1]));
            encoders.push_back(std::move(stage));
            downs.emplace_back(store, "downs." + std::to_string(level));
        }

        const int middle_blocks = count_blocks(store, "middle");
        middle.reserve(middle_blocks);
        for (int i = 0; i < middle_blocks; ++i) {
            middle.emplace_back(store, "middle." + std::to_string(i));
        }

        ups.reserve(levels);
        decoders.reserve(levels);
        for (int level = 0; level < levels; ++level) {
            ups.emplace_back(store, "ups." + std::to_string(level));
            const std::string root = "decoders." + std::to_string(level);
            const int blocks = count_blocks(store, root);
            std::vector<NAFBlock> stage;
            stage.reserve(blocks);
            for (int block = 0; block < blocks; ++block) {
                stage.emplace_back(store, root + "." + std::to_string(block));
            }
            decoders.push_back(std::move(stage));
        }

        temporal_mix.reserve(levels);
        for (int level = 0; level < levels; ++level) {
            temporal_mix.push_back(std::make_unique<Conv2dLayer>(
                store, "temporal_mix." + std::to_string(level) + ".weight",
                std::optional<std::string>("temporal_mix." + std::to_string(level) + ".bias"),
                /*pad=*/0, /*stride=*/1, /*dilation=*/1, /*groups=*/1));
        }

        use_warp = store.contains("offset_heads.0.weight");
        if (use_warp) {
            offset_heads.reserve(levels);
            for (int level = 0; level < levels; ++level) {
                offset_heads.push_back(std::make_unique<Conv2dLayer>(
                    store, "offset_heads." + std::to_string(level) + ".weight",
                    std::optional<std::string>("offset_heads." + std::to_string(level) + ".bias"),
                    /*pad=*/0, /*stride=*/1, /*dilation=*/1, /*groups=*/1));
            }
        }

        const auto& arch = store.manifest().architecture;
        if (arch.num_frames > 0) {
            num_frames = arch.num_frames;
        }
        ref_idx = num_frames / 2;
    }
};

NAFNetTemporal::NAFNetTemporal(const WeightStore& store)
    : impl_(std::make_unique<Impl>(store)) {}

NAFNetTemporal::~NAFNetTemporal() = default;

Tensor NAFNetTemporal::forward(const Tensor& clip, cudaStream_t stream) const {
    if (clip.dtype() != DType::kFloat16) {
        throw std::runtime_error("NAFNetTemporal::forward: expected FP16 input");
    }
    if (clip.shape().size() != 5) {
        throw std::runtime_error("NAFNetTemporal::forward: expected 5-D (N,T,C,H,W) clip");
    }

    const int N = static_cast<int>(clip.shape()[0]);
    const int T = static_cast<int>(clip.shape()[1]);
    const int C = static_cast<int>(clip.shape()[2]);
    const int H = static_cast<int>(clip.shape()[3]);
    const int W = static_cast<int>(clip.shape()[4]);
    if (T != impl_->num_frames) {
        throw std::runtime_error("NAFNetTemporal::forward: frame count mismatch");
    }

    auto ref_input = Tensor::empty({N, C, H, W}, DType::kFloat16);
    kernels::launch_extract_frame(
        clip.data_f16(), ref_input.data_f16(), N, T, C, H, W, impl_->ref_idx, stream);
    auto beauty = select_beauty(ref_input, stream);

    std::vector<Tensor> frames;
    frames.reserve(T);
    PaddingInfo padding{};
    for (int t = 0; t < T; ++t) {
        auto frame = Tensor::empty({N, C, H, W}, DType::kFloat16);
        kernels::launch_extract_frame(
            clip.data_f16(), frame.data_f16(), N, T, C, H, W, t, stream);
        auto [padded, pad] = pad_to_multiple(frame, impl_->pad_multiple, stream);
        padding = pad;
        frames.push_back(std::move(padded));
    }

    const int levels = static_cast<int>(impl_->encoders.size());
    std::vector<std::vector<Tensor>> all_skips(levels);
    for (auto& v : all_skips) {
        v.reserve(T);
    }
    std::vector<Tensor> deep_features;
    deep_features.reserve(T);

    for (int t = 0; t < T; ++t) {
        auto x = impl_->intro->forward(frames[t], stream);
        for (int level = 0; level < levels; ++level) {
            for (const auto& block : impl_->encoders[level]) {
                x = block.forward(x, stream);
            }
            all_skips[level].push_back(x.clone(stream));
            x = impl_->downs[level].forward(x, stream);
        }
        deep_features.push_back(std::move(x));
    }

    auto x = deep_features[impl_->ref_idx].clone(stream);
    for (const auto& block : impl_->middle) {
        x = block.forward(x, stream);
    }

    std::vector<Tensor> fused_skips;
    fused_skips.reserve(levels);
    for (int level = 0; level < levels; ++level) {
        const auto& ref_feat = all_skips[level][impl_->ref_idx];
        Tensor neigh_sum;
        int neigh_count = 0;
        for (int t = 0; t < T; ++t) {
            if (t == impl_->ref_idx) {
                continue;
            }
            Tensor neigh = all_skips[level][t].clone(stream);
            if (impl_->use_warp) {
                auto cat = Tensor::empty(
                    {ref_feat.batch(), ref_feat.channels() * 2, ref_feat.height(), ref_feat.width()},
                    DType::kFloat16);
                kernels::launch_concat_channels(
                    ref_feat.data_f16(), neigh.data_f16(), cat.data_f16(),
                    static_cast<int>(ref_feat.batch()),
                    static_cast<int>(ref_feat.channels()),
                    static_cast<int>(neigh.channels()),
                    static_cast<int>(ref_feat.height()),
                    static_cast<int>(ref_feat.width()),
                    stream);
                auto offset = impl_->offset_heads[level]->forward(cat, stream);
                auto warped = Tensor::empty(neigh.shape(), DType::kFloat16);
                kernels::launch_warp_bilinear(
                    neigh.data_f16(), offset.data_f16(), warped.data_f16(),
                    static_cast<int>(neigh.batch()),
                    static_cast<int>(neigh.channels()),
                    static_cast<int>(neigh.height()),
                    static_cast<int>(neigh.width()),
                    stream);
                neigh = std::move(warped);
            }

            if (neigh_count == 0) {
                neigh_sum = std::move(neigh);
            } else {
                neigh_sum = add_tensors(neigh_sum, neigh, stream);
            }
            ++neigh_count;
        }

        if (neigh_count > 1) {
            auto mean = Tensor::empty(neigh_sum.shape(), DType::kFloat16);
            kernels::launch_mul_scalar(
                neigh_sum.data_f16(), 1.0f / static_cast<float>(neigh_count),
                mean.data_f16(), neigh_sum.numel(), stream);
            neigh_sum = std::move(mean);
        }

        auto cat = Tensor::empty(
            {ref_feat.batch(), ref_feat.channels() * 2, ref_feat.height(), ref_feat.width()},
            DType::kFloat16);
        kernels::launch_concat_channels(
            ref_feat.data_f16(), neigh_sum.data_f16(), cat.data_f16(),
            static_cast<int>(ref_feat.batch()),
            static_cast<int>(ref_feat.channels()),
            static_cast<int>(neigh_sum.channels()),
            static_cast<int>(ref_feat.height()),
            static_cast<int>(ref_feat.width()),
            stream);
        fused_skips.push_back(impl_->temporal_mix[level]->forward(cat, stream));
    }

    for (size_t level = 0; level < impl_->decoders.size(); ++level) {
        x = impl_->ups[level].forward(x, stream);
        auto& skip = fused_skips[fused_skips.size() - 1 - level];
        const int h = std::min(static_cast<int>(x.height()), static_cast<int>(skip.height()));
        const int w = std::min(static_cast<int>(x.width()), static_cast<int>(skip.width()));
        auto x_cropped = crop_to_size(x, h, w, stream);
        auto skip_cropped = crop_to_size(skip, h, w, stream);
        x = add_tensors(x_cropped, skip_cropped, stream);
        for (const auto& block : impl_->decoders[level]) {
            x = block.forward(x, stream);
        }
    }

    auto residual = impl_->ending->forward(x, stream);
    residual = unpad(residual, padding, H, W, stream);
    return add_tensors(beauty, residual, stream);
}

int NAFNetTemporal::num_frames() const noexcept {
    return impl_->num_frames;
}

int NAFNetTemporal::num_levels() const noexcept {
    return static_cast<int>(impl_->enc_channels.size());
}

const std::vector<int>& NAFNetTemporal::enc_channels() const noexcept {
    return impl_->enc_channels;
}

bool NAFNetTemporal::use_warp() const noexcept {
    return impl_->use_warp;
}

}  // namespace denoiser
