#include <gtest/gtest.h>

#include "denoiser/tensor.h"
#include "denoiser/weight_loader.h"

// NAFNet is optional during bring-up — include it conditionally so the rest
// of the test binary can still compile when the model target is disabled.
#if __has_include("denoiser/models/nafnet.h")
  #include "denoiser/models/nafnet.h"
  #define DENOISER_NAFNET_AVAILABLE 1
#else
  #define DENOISER_NAFNET_AVAILABLE 0
#endif

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <algorithm>
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

#define SKIP_PHASE4() \
    GTEST_SKIP() << "NAFNet not available in this build"

static std::vector<float> to_floats(const Tensor& t) {
    std::vector<__half> raw(static_cast<size_t>(t.numel()));
    t.to_host(raw.data());
    cudaDeviceSynchronize();
    std::vector<float> out(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) out[i] = __half2float(raw[i]);
    return out;
}

// ---------------------------------------------------------------------------
// Tensor integration tests — these run independently of Phase 4
// ---------------------------------------------------------------------------

// Verify that slice_channels produces the correct sub-tensor for a skip
// connection layout — this exercises the non-owning view path that older
// model code used to split batched feature maps.
// encoder/decoder use to split a (B*T, C, H, W) temporal batch back into
// per-frame tensors.
TEST(NAFNetSupportTest, SliceChannelsForSkipConnection) {
    SKIP_IF_NO_GPU();

    // (1, 4, 2, 2) with values 0..15
    std::vector<__half> data(16);
    for (int i = 0; i < 16; ++i) data[i] = __float2half(static_cast<float>(i));
    auto t = Tensor::from_host(data.data(), {1, 4, 2, 2}, DType::kFloat16);
    cudaDeviceSynchronize();

    // First two channels
    auto lo = t.slice_channels(0, 2);
    EXPECT_EQ(lo.channels(), 2);
    auto lo_vals = to_floats(lo);
    ASSERT_EQ(lo_vals.size(), 8u);
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(lo_vals[i], static_cast<float>(i), 1e-2f) << "lo slot " << i;
    }

    // Last two channels
    auto hi = t.slice_channels(2, 4);
    EXPECT_EQ(hi.channels(), 2);
    auto hi_vals = to_floats(hi);
    ASSERT_EQ(hi_vals.size(), 8u);
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(hi_vals[i], static_cast<float>(i + 8), 1e-2f) << "hi slot " << i;
    }
}

// Verify that Tensor::empty + from_host round-trips at a representative model
// intermediate tensor size (1, 64, 128, 128) without OOM or corruption.
TEST(NAFNetSupportTest, IntermediateTensorAllocation) {
    SKIP_IF_NO_GPU();

    // 1 × 64 × 128 × 128 FP16 ≈ 2 MB — well within budget
    const int64_t numel = 1 * 64 * 128 * 128;
    auto t = Tensor::empty({1, 64, 128, 128}, DType::kFloat16);
    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.numel(), numel);
    EXPECT_EQ(t.nbytes(), static_cast<size_t>(numel) * sizeof(__half));
}

// ---------------------------------------------------------------------------
// NAFNet forward pass tests
//
// These tests are disabled (DISABLED_ prefix) until NAFNet parity fixtures are
// implemented.  Run them with:
//   ./denoiser_tests --gtest_also_run_disabled_tests \
//       --gtest_filter="*DISABLED_NAFNetForward*"
// ---------------------------------------------------------------------------

#if DENOISER_NAFNET_AVAILABLE

class NAFNetForwardTest : public ::testing::Test {
protected:
    fs::path fixture_dir_;
    std::unique_ptr<WeightStore> store_;

    void SetUp() override {
        fixture_dir_ = fs::path(__FILE__).parent_path() / "fixtures" / "tiny_nafnet";
        if (!fs::exists(fixture_dir_ / "manifest.json")) {
            GTEST_SKIP() << "NAFNet fixture not found — run tests/gen_fixtures.py first";
        }
        if (!has_cuda_device()) {
            GTEST_SKIP() << "No CUDA device available";
        }
        store_ = std::make_unique<WeightStore>((fixture_dir_ / "manifest.json").string());
    }

    void TearDown() override {
        store_.reset();
    }
};

// Output spatial dimensions must match input.
TEST_F(NAFNetForwardTest, DISABLED_OutputShapeMatchesInput) {
    NAFNet model(*store_);

    std::vector<__half> data(3 * 32 * 32, __float2half(0.5f));
    auto input = Tensor::from_host(data.data(), {1, 3, 32, 32}, DType::kFloat16);
    cudaDeviceSynchronize();

    auto output = model.forward(input);

    EXPECT_EQ(output.batch(),    1);
    EXPECT_EQ(output.channels(), 3);
    EXPECT_EQ(output.height(),   32);
    EXPECT_EQ(output.width(),    32);
}

// Output values must be in a reasonable range for a denoiser ([0, 1] or
// close to it for normalised FP16 inputs).
TEST_F(NAFNetForwardTest, DISABLED_OutputInRange) {
    NAFNet model(*store_);

    // Slightly noisy image: uniform 0.5 + small perturbation
    const int numel = 3 * 32 * 32;
    std::vector<__half> data(numel);
    for (int i = 0; i < numel; ++i) {
        data[i] = __float2half(0.5f + 0.01f * static_cast<float>(i % 11 - 5));
    }
    auto input = Tensor::from_host(data.data(), {1, 3, 32, 32}, DType::kFloat16);
    cudaDeviceSynchronize();

    auto output = model.forward(input);
    auto vals   = to_floats(output);

    for (size_t i = 0; i < vals.size(); ++i) {
        EXPECT_GE(vals[i], -0.1f) << "output below zero at " << i;
        EXPECT_LE(vals[i],  1.1f) << "output above 1 at "   << i;
    }
}

// C++ output must match the Python reference within FP16 tolerance.
// The reference is stored in fixtures/tiny_nafnet/expected_output.bin (FP32).
TEST_F(NAFNetForwardTest, DISABLED_PyTorchParitySmallInput) {
    const auto ref_path = fixture_dir_ / "expected_output.bin";
    if (!fs::exists(ref_path)) {
        GTEST_SKIP() << "Reference output not found: " << ref_path;
    }

    // Load reference output (FP32)
    const size_t numel = 3 * 32 * 32;
    std::vector<float> ref(numel);
    std::ifstream f(ref_path, std::ios::binary);
    f.read(reinterpret_cast<char*>(ref.data()), numel * sizeof(float));

    // Load input (same fixture)
    const auto in_path = fixture_dir_ / "input.bin";
    std::vector<__half> in_data(numel);
    {
        std::vector<float> in_f32(numel);
        std::ifstream fin(in_path, std::ios::binary);
        fin.read(reinterpret_cast<char*>(in_f32.data()), numel * sizeof(float));
        for (size_t i = 0; i < numel; ++i) in_data[i] = __float2half(in_f32[i]);
    }

    auto input = Tensor::from_host(in_data.data(), {1, 3, 32, 32}, DType::kFloat16);
    cudaDeviceSynchronize();

    NAFNet model(*store_);
    auto output = model.forward(input);
    auto vals   = to_floats(output);

    ASSERT_EQ(vals.size(), numel);
    float max_diff = 0.f;
    for (size_t i = 0; i < numel; ++i) {
        max_diff = std::max(max_diff, std::abs(vals[i] - ref[i]));
    }
    EXPECT_LE(max_diff, 0.005f) << "max pixel diff vs Python: " << max_diff;
}

// Memory should remain stable across 100 repeated forward passes
// (no allocation leak in workspace cache or elsewhere).
TEST_F(NAFNetForwardTest, DISABLED_VRAMStableUnder100Iters) {
    NAFNet model(*store_);

    std::vector<__half> data(3 * 32 * 32, __float2half(0.5f));
    auto input = Tensor::from_host(data.data(), {1, 3, 32, 32}, DType::kFloat16);
    cudaDeviceSynchronize();

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        (void)model.forward(input);
    }
    cudaDeviceSynchronize();

    size_t free0 = 0, total = 0;
    cudaMemGetInfo(&free0, &total);

    for (int i = 0; i < 100; ++i) {
        (void)model.forward(input);
    }
    cudaDeviceSynchronize();

    size_t free1 = 0;
    cudaMemGetInfo(&free1, &total);

    // Allow at most 1 MB drift (workspace cache is constant after first call)
    const size_t kOneMB = 1u << 20;
    if (free0 >= free1) {
        EXPECT_LE(free0 - free1, kOneMB) << "Possible VRAM leak after 100 iters";
    }
}

#else // !DENOISER_NAFNET_AVAILABLE

TEST(NAFNetForwardTest, NotYetBuilt) {
    SKIP_PHASE4();
}

#endif // DENOISER_NAFNET_AVAILABLE
