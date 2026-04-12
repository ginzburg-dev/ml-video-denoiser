#include <gtest/gtest.h>

#include "denoiser/layers/conv2d.h"
#include "denoiser/weight_loader.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cmath>

using namespace denoiser;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

#define SKIP_IF_NO_GPU() \
    if (!has_cuda_device()) GTEST_SKIP() << "No CUDA device available"

static void write_bin(const fs::path& path, const void* data, size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open " + path.string());
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

// Write a manifest.json + weight .bin files for a small 1×1 convolution.
//
// Configuration A (no bias):
//   weight: (C_out=1, C_in=1, 1, 1) FP16 — value 2.0
//   input:  (1, 1, 3, 3) FP16 — values 1..9
//   expected output: 2, 4, 6, 8, 10, 12, 14, 16, 18
//
// Configuration B (with bias):
//   weight: same
//   bias:   (1,) FP16 — value 1.0
//   expected output: 3, 5, 7, 9, 11, 13, 15, 17, 19
//
// Configuration C (multi-channel):
//   weight: (C_out=2, C_in=1, 1, 1) FP16 — channel 0: 1.0, channel 1: -1.0
//   input:  (1, 1, 2, 2) FP16 — values 1..4
//   expected output ch0: 1, 2, 3, 4
//   expected output ch1: -1, -2, -3, -4
static fs::path create_conv_fixture(const fs::path& dir) {
    fs::create_directories(dir / "weights");

    // --- Config A/B weight: (1, 1, 1, 1) FP16 = 2.0 ---
    __half w_scalar = __float2half(2.0f);
    write_bin(dir / "weights" / "conv1x1_w.bin", &w_scalar, sizeof(__half));

    // --- Config B bias: (1,) FP16 = 1.0 ---
    __half bias_scalar = __float2half(1.0f);
    write_bin(dir / "weights" / "conv1x1_b.bin", &bias_scalar, sizeof(__half));

    // --- Config C weight: (2, 1, 1, 1) FP16 = [1.0, -1.0] ---
    std::vector<__half> w_multi = {__float2half(1.0f), __float2half(-1.0f)};
    write_bin(dir / "weights" / "conv_multi_w.bin",
              w_multi.data(), w_multi.size() * sizeof(__half));

    // --- Manifest ---
    const char* json = R"({
  "version": "1.0",
  "model": "test_conv",
  "dtype": "float16",
  "architecture": {
    "type": "nafnet_residual",
    "enc_channels": [32],
    "num_levels": 1,
    "base_channels": 32,
    "in_channels": 1,
    "out_channels": 1
  },
  "layers": [
    { "name": "conv1x1.weight",       "shape": [1, 1, 1, 1], "dtype": "float16", "file": "weights/conv1x1_w.bin"   },
    { "name": "conv1x1.bias",         "shape": [1],          "dtype": "float16", "file": "weights/conv1x1_b.bin"   },
    { "name": "conv_multi.weight",    "shape": [2, 1, 1, 1], "dtype": "float16", "file": "weights/conv_multi_w.bin" }
  ]
})";
    auto path = dir / "manifest.json";
    std::ofstream mf(path);
    mf << json;
    return path;
}

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class Conv2dTest : public ::testing::Test {
protected:
    fs::path tmp_dir_;
    std::unique_ptr<WeightStore> store_;

    void SetUp() override {
        if (!has_cuda_device()) return;
        tmp_dir_ = fs::temp_directory_path() / "denoiser_conv_test";
        fs::remove_all(tmp_dir_);
        auto manifest = create_conv_fixture(tmp_dir_);
        store_ = std::make_unique<WeightStore>(manifest.string());
    }

    void TearDown() override {
        store_.reset();
        fs::remove_all(tmp_dir_);
    }

    // Build a FP16 input tensor from host floats.
    static Tensor make_fp16_input(const std::vector<float>& vals,
                                  const std::vector<int64_t>& shape) {
        std::vector<__half> h(vals.size());
        for (size_t i = 0; i < vals.size(); ++i) h[i] = __float2half(vals[i]);
        auto t = Tensor::from_host(h.data(), shape, DType::kFloat16);
        cudaDeviceSynchronize();
        return t;
    }

    // Copy FP16 tensor to host as floats.
    static std::vector<float> to_host_floats(const Tensor& t) {
        std::vector<__half> raw(static_cast<size_t>(t.numel()));
        t.to_host(raw.data());
        cudaDeviceSynchronize();
        std::vector<float> out(raw.size());
        for (size_t i = 0; i < raw.size(); ++i) out[i] = __half2float(raw[i]);
        return out;
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(Conv2dTest, OutputShapeNoBias) {
    SKIP_IF_NO_GPU();
    // (1,1,3,3) input with 1x1 kernel → (1,1,3,3) output
    Conv2dLayer conv(*store_, "conv1x1.weight", std::nullopt, /*pad=*/0);

    auto input = make_fp16_input({1,2,3,4,5,6,7,8,9}, {1,1,3,3});
    auto output = conv.forward(input);

    EXPECT_EQ(output.batch(),    1);
    EXPECT_EQ(output.channels(), 1);
    EXPECT_EQ(output.height(),   3);
    EXPECT_EQ(output.width(),    3);
    EXPECT_EQ(output.dtype(),    DType::kFloat16);
}

TEST_F(Conv2dTest, NumericalOutputNoBias) {
    SKIP_IF_NO_GPU();
    // weight=2.0, no bias → output[i] = 2 * input[i]
    Conv2dLayer conv(*store_, "conv1x1.weight", std::nullopt, /*pad=*/0);

    auto input  = make_fp16_input({1,2,3,4,5,6,7,8,9}, {1,1,3,3});
    auto output = conv.forward(input);

    auto vals = to_host_floats(output);
    ASSERT_EQ(vals.size(), 9u);
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(vals[i], 2.0f * static_cast<float>(i + 1), 1e-2f)
            << "mismatch at position " << i;
    }
}

TEST_F(Conv2dTest, NumericalOutputWithBias) {
    SKIP_IF_NO_GPU();
    // weight=2.0, bias=1.0 → output[i] = 2 * input[i] + 1
    Conv2dLayer conv(*store_, "conv1x1.weight", "conv1x1.bias", /*pad=*/0);

    auto input  = make_fp16_input({1,2,3,4,5,6,7,8,9}, {1,1,3,3});
    auto output = conv.forward(input);

    auto vals = to_host_floats(output);
    ASSERT_EQ(vals.size(), 9u);
    for (int i = 0; i < 9; ++i) {
        EXPECT_NEAR(vals[i], 2.0f * static_cast<float>(i + 1) + 1.0f, 1e-2f)
            << "mismatch at position " << i;
    }
}

TEST_F(Conv2dTest, MultiChannelOutput) {
    SKIP_IF_NO_GPU();
    // weight[0]=1.0, weight[1]=-1.0; input (1,1,2,2) values 1..4
    // channel 0: [1,2,3,4]   channel 1: [-1,-2,-3,-4]
    Conv2dLayer conv(*store_, "conv_multi.weight", std::nullopt, /*pad=*/0);

    auto input  = make_fp16_input({1,2,3,4}, {1,1,2,2});
    auto output = conv.forward(input);

    EXPECT_EQ(output.channels(), 2);
    EXPECT_EQ(output.height(),   2);
    EXPECT_EQ(output.width(),    2);

    auto vals = to_host_floats(output);
    ASSERT_EQ(vals.size(), 8u);

    // Channel 0 (indices 0..3)
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(vals[i], static_cast<float>(i + 1), 1e-2f)
            << "channel 0, pos " << i;
    }
    // Channel 1 (indices 4..7)
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(vals[4 + i], -static_cast<float>(i + 1), 1e-2f)
            << "channel 1, pos " << i;
    }
}

TEST_F(Conv2dTest, OutputShapeWithPadding) {
    SKIP_IF_NO_GPU();
    // 3x3 kernel on (1,1,5,5) input with pad=1 → (1,1,5,5) output
    // We need a 3x3 kernel weight for this, but our fixture only has 1x1.
    // Use conv_multi weight (2,1,1,1) with pad=0 just for shape — skip if shape wrong.
    //
    // Instead test: 1x1 kernel with pad=1 → H and W each grow by 2
    Conv2dLayer conv(*store_, "conv1x1.weight", std::nullopt, /*pad=*/1);

    auto input  = make_fp16_input(std::vector<float>(9, 1.0f), {1,1,3,3});
    auto output = conv.forward(input);

    // (1,1,3+2,3+2) = (1,1,5,5) for 1x1 kernel with pad=1
    EXPECT_EQ(output.height(), 5);
    EXPECT_EQ(output.width(),  5);
}

TEST_F(Conv2dTest, AccessorsMatchWeight) {
    SKIP_IF_NO_GPU();
    Conv2dLayer conv(*store_, "conv1x1.weight", std::nullopt, /*pad=*/0);
    EXPECT_EQ(conv.out_channels(), 1);
    EXPECT_EQ(conv.in_channels(),  1);
    EXPECT_EQ(conv.kernel_h(),     1);
    EXPECT_EQ(conv.kernel_w(),     1);
}

TEST_F(Conv2dTest, MultiChannelAccessors) {
    SKIP_IF_NO_GPU();
    Conv2dLayer conv(*store_, "conv_multi.weight", std::nullopt, /*pad=*/0);
    EXPECT_EQ(conv.out_channels(), 2);
    EXPECT_EQ(conv.in_channels(),  1);
}

TEST_F(Conv2dTest, RejectsNonFP16Input) {
    SKIP_IF_NO_GPU();
    Conv2dLayer conv(*store_, "conv1x1.weight", std::nullopt, /*pad=*/0);

    // FP32 input must throw
    std::vector<float> data(9, 1.0f);
    auto input = Tensor::from_host(data.data(), {1,1,3,3}, DType::kFloat32);
    cudaDeviceSynchronize();

    EXPECT_THROW(conv.forward(input), std::runtime_error);
}

TEST_F(Conv2dTest, WorkspaceCacheHitOnSameSize) {
    SKIP_IF_NO_GPU();
    Conv2dLayer conv(*store_, "conv1x1.weight", std::nullopt, /*pad=*/0);

    auto input = make_fp16_input(std::vector<float>(9, 1.0f), {1,1,3,3});

    // Both forward calls must succeed (second hits workspace cache)
    EXPECT_NO_THROW(conv.forward(input));
    EXPECT_NO_THROW(conv.forward(input));
}
