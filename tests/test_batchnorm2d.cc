#include <gtest/gtest.h>

#include "denoiser/layers/batchnorm2d.h"
#include "denoiser/weight_loader.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <cmath>
#include <vector>

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

// Build a minimal WeightStore for two BatchNorm configurations:
//
//   "bn_id"  — identity: gamma=1, beta=0, mean=0, var=1
//              → scale = 1/√(1+ε) ≈ 1,  shift = 0
//              → output ≈ input (subject to FP16 precision)
//
//   "bn_nontrivial" — gamma=[1,2], beta=[0,1], mean=[0,0.5], var=[1,0.25], eps=1e-5
//              → scale[0] = 1/√(1+ε) ≈ 1.0,  shift[0] = 0 - 1*0 = 0
//              → scale[1] = 2/√(0.25+ε) ≈ 4.0, shift[1] = 1 - 4*0.5 = -1
//              → ch0 output ≈ input_ch0
//              → ch1 output ≈ 4 * input_ch1 - 1
static fs::path create_bn_fixture(const fs::path& dir) {
    fs::create_directories(dir / "weights");

    // --- bn_id: C=1 ---
    std::vector<float> id_gamma = {1.0f};
    std::vector<float> id_beta  = {0.0f};
    std::vector<float> id_mean  = {0.0f};
    std::vector<float> id_var   = {1.0f};
    write_bin(dir / "weights" / "bn_id_w.bin",    id_gamma.data(), sizeof(float));
    write_bin(dir / "weights" / "bn_id_b.bin",    id_beta.data(),  sizeof(float));
    write_bin(dir / "weights" / "bn_id_mean.bin", id_mean.data(),  sizeof(float));
    write_bin(dir / "weights" / "bn_id_var.bin",  id_var.data(),   sizeof(float));

    // --- bn_nontrivial: C=2 ---
    std::vector<float> nt_gamma = {1.0f, 2.0f};
    std::vector<float> nt_beta  = {0.0f, 1.0f};
    std::vector<float> nt_mean  = {0.0f, 0.5f};
    std::vector<float> nt_var   = {1.0f, 0.25f};
    write_bin(dir / "weights" / "bn_nt_w.bin",    nt_gamma.data(), 2 * sizeof(float));
    write_bin(dir / "weights" / "bn_nt_b.bin",    nt_beta.data(),  2 * sizeof(float));
    write_bin(dir / "weights" / "bn_nt_mean.bin", nt_mean.data(),  2 * sizeof(float));
    write_bin(dir / "weights" / "bn_nt_var.bin",  nt_var.data(),   2 * sizeof(float));

    const char* json = R"({
  "version": "1.0",
  "model": "test_bn",
  "dtype": "float32",
  "architecture": {
    "type": "nafnet_residual",
    "enc_channels": [32],
    "num_levels": 1,
    "base_channels": 32,
    "in_channels": 3,
    "out_channels": 3
  },
  "layers": [
    { "name": "bn_id.weight",       "shape": [1], "dtype": "float32", "file": "weights/bn_id_w.bin"    },
    { "name": "bn_id.bias",         "shape": [1], "dtype": "float32", "file": "weights/bn_id_b.bin"    },
    { "name": "bn_id.running_mean", "shape": [1], "dtype": "float32", "file": "weights/bn_id_mean.bin" },
    { "name": "bn_id.running_var",  "shape": [1], "dtype": "float32", "file": "weights/bn_id_var.bin"  },

    { "name": "bn_nt.weight",       "shape": [2], "dtype": "float32", "file": "weights/bn_nt_w.bin"    },
    { "name": "bn_nt.bias",         "shape": [2], "dtype": "float32", "file": "weights/bn_nt_b.bin"    },
    { "name": "bn_nt.running_mean", "shape": [2], "dtype": "float32", "file": "weights/bn_nt_mean.bin" },
    { "name": "bn_nt.running_var",  "shape": [2], "dtype": "float32", "file": "weights/bn_nt_var.bin"  }
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

class BatchNorm2dTest : public ::testing::Test {
protected:
    fs::path tmp_dir_;
    std::unique_ptr<WeightStore> store_;

    void SetUp() override {
        if (!has_cuda_device()) return;
        tmp_dir_ = fs::temp_directory_path() / "denoiser_bn_test";
        fs::remove_all(tmp_dir_);
        auto manifest = create_bn_fixture(tmp_dir_);
        store_ = std::make_unique<WeightStore>(manifest.string());
    }

    void TearDown() override {
        store_.reset();
        fs::remove_all(tmp_dir_);
    }

    static Tensor make_fp16(const std::vector<float>& vals,
                             const std::vector<int64_t>& shape) {
        std::vector<__half> h(vals.size());
        for (size_t i = 0; i < vals.size(); ++i) h[i] = __float2half(vals[i]);
        auto t = Tensor::from_host(h.data(), shape, DType::kFloat16);
        cudaDeviceSynchronize();
        return t;
    }

    static std::vector<float> to_floats(const Tensor& t) {
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

TEST_F(BatchNorm2dTest, OutputShapeMatchesInput) {
    SKIP_IF_NO_GPU();
    BatchNorm2dLayer bn(*store_, "bn_id");

    auto input  = make_fp16({1.f, 2.f, 3.f, 4.f}, {1, 1, 2, 2});
    auto output = bn.forward(input);

    EXPECT_EQ(output.shape(), input.shape());
    EXPECT_EQ(output.dtype(), DType::kFloat16);
}

TEST_F(BatchNorm2dTest, IdentityParams) {
    SKIP_IF_NO_GPU();
    // gamma=1, beta=0, mean=0, var=1 → output ≈ input (eps only tiny correction)
    BatchNorm2dLayer bn(*store_, "bn_id");

    auto input  = make_fp16({1.f, 2.f, 3.f, 4.f}, {1, 1, 2, 2});
    auto output = bn.forward(input);

    auto vals = to_floats(output);
    const float eps = 1e-5f;
    const float scale = 1.0f / std::sqrt(1.0f + eps);
    for (size_t i = 0; i < vals.size(); ++i) {
        float expected = static_cast<float>(i + 1) * scale;
        EXPECT_NEAR(vals[i], expected, 1e-2f) << "position " << i;
    }
}

TEST_F(BatchNorm2dTest, NumChannelsAccessor) {
    SKIP_IF_NO_GPU();
    BatchNorm2dLayer bn_id(*store_, "bn_id");
    EXPECT_EQ(bn_id.num_channels(), 1);

    BatchNorm2dLayer bn_nt(*store_, "bn_nt");
    EXPECT_EQ(bn_nt.num_channels(), 2);
}

TEST_F(BatchNorm2dTest, NonTrivialParams) {
    SKIP_IF_NO_GPU();
    // C=2 channels, spatial 1×1 for easy per-channel checking:
    //   ch0: scale≈1.0, shift=0.0  → output = input_ch0 * 1.0 + 0.0
    //   ch1: scale≈4.0, shift=-1.0 → output = input_ch1 * 4.0 - 1.0
    BatchNorm2dLayer bn(*store_, "bn_nt");

    // (1, 2, 1, 1): ch0=1.0, ch1=2.0
    auto input  = make_fp16({1.0f, 2.0f}, {1, 2, 1, 1});
    auto output = bn.forward(input);

    auto vals = to_floats(output);
    ASSERT_EQ(vals.size(), 2u);

    const float eps  = 1e-5f;
    float scale0     = 1.0f / std::sqrt(1.0f  + eps);
    float scale1     = 2.0f / std::sqrt(0.25f + eps);
    float shift0     = 0.0f - scale0 * 0.0f;
    float shift1     = 1.0f - scale1 * 0.5f;

    EXPECT_NEAR(vals[0], 1.0f * scale0 + shift0, 1e-2f) << "channel 0";
    EXPECT_NEAR(vals[1], 2.0f * scale1 + shift1, 1e-2f) << "channel 1";
}

TEST_F(BatchNorm2dTest, NonTrivialSpatialBatch) {
    SKIP_IF_NO_GPU();
    // Verify output values across a spatial extent and batch > 1.
    // (2, 2, 2, 2): all ch0 values = 1.0, all ch1 values = 2.0
    BatchNorm2dLayer bn(*store_, "bn_nt");

    std::vector<float> input_vals(2 * 2 * 2 * 2);
    // NCHW layout: ch0 occupies indices [0..3] per batch, ch1 [4..7] per batch
    // batch0: [ch0*4, ch1*4] = [1,1,1,1, 2,2,2,2]
    // batch1: same
    for (int n = 0; n < 2; ++n) {
        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 2; ++w) {
                input_vals[n * 2 * 2 * 2 + 0 * 2 * 2 + h * 2 + w] = 1.0f;  // ch0
                input_vals[n * 2 * 2 * 2 + 1 * 2 * 2 + h * 2 + w] = 2.0f;  // ch1
            }
        }
    }

    auto input  = make_fp16(input_vals, {2, 2, 2, 2});
    auto output = bn.forward(input);

    EXPECT_EQ(output.batch(),    2);
    EXPECT_EQ(output.channels(), 2);
    EXPECT_EQ(output.height(),   2);
    EXPECT_EQ(output.width(),    2);

    auto vals = to_floats(output);
    const float eps  = 1e-5f;
    float scale0     = 1.0f / std::sqrt(1.0f  + eps);
    float scale1     = 2.0f / std::sqrt(0.25f + eps);
    float expected0  = 1.0f * scale0 + (0.0f - scale0 * 0.0f);
    float expected1  = 2.0f * scale1 + (1.0f - scale1 * 0.5f);

    for (int n = 0; n < 2; ++n) {
        for (int h = 0; h < 2; ++h) {
            for (int w = 0; w < 2; ++w) {
                int idx0 = n * 2 * 2 * 2 + 0 * 2 * 2 + h * 2 + w;
                int idx1 = n * 2 * 2 * 2 + 1 * 2 * 2 + h * 2 + w;
                EXPECT_NEAR(vals[idx0], expected0, 1e-2f) << "n=" << n << " h=" << h << " w=" << w << " ch0";
                EXPECT_NEAR(vals[idx1], expected1, 1e-2f) << "n=" << n << " h=" << h << " w=" << w << " ch1";
            }
        }
    }
}

TEST_F(BatchNorm2dTest, RejectsNonFP16Input) {
    SKIP_IF_NO_GPU();
    BatchNorm2dLayer bn(*store_, "bn_id");

    std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
    auto input = Tensor::from_host(data.data(), {1, 1, 2, 2}, DType::kFloat32);
    cudaDeviceSynchronize();

    EXPECT_THROW(bn.forward(input), std::runtime_error);
}

TEST_F(BatchNorm2dTest, RejectsChannelMismatch) {
    SKIP_IF_NO_GPU();
    // bn_id has 1 channel; provide 2-channel input
    BatchNorm2dLayer bn(*store_, "bn_id");

    auto input = make_fp16({1.f, 2.f, 3.f, 4.f}, {1, 2, 1, 2});
    EXPECT_THROW(bn.forward(input), std::runtime_error);
}
