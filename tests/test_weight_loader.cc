#include <gtest/gtest.h>

#include "denoiser/weight_loader.h"

#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <vector>

using namespace denoiser;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Fixture helpers
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

// Create a small test fixture under *dir*:
//   weights/layer0_weight.bin   — 6 FP32 values [1..6], shape [2,3,1,1]
//   weights/layer0_bias.bin     — 2 FP32 values [0.1, 0.2], shape [2]
//   weights/bn_weight.bin       — 2 FP32 values [1.0, 2.0], shape [2]
//   weights/bn_bias.bin         — 2 FP32 values [0.0, 1.0], shape [2]
//   weights/bn_running_mean.bin — 2 FP32 values [0.0, 0.5], shape [2]
//   weights/bn_running_var.bin  — 2 FP32 values [1.0, 0.25], shape [2]
//   manifest.json
//
// Returns the path to manifest.json.
static fs::path create_fixture(const fs::path& dir) {
    fs::create_directories(dir / "weights");

    // Conv weight: [2, 3, 1, 1] FP32
    std::vector<float> w(6);
    for (int i = 0; i < 6; ++i) w[i] = static_cast<float>(i + 1);
    write_bin(dir / "weights" / "layer0_weight.bin", w.data(), w.size() * sizeof(float));

    // Conv bias: [2] FP32
    std::vector<float> b = {0.1f, 0.2f};
    write_bin(dir / "weights" / "layer0_bias.bin", b.data(), b.size() * sizeof(float));

    // BN params: [2] FP32 each
    std::vector<float> bn_gamma  = {1.0f, 2.0f};
    std::vector<float> bn_beta   = {0.0f, 1.0f};
    std::vector<float> bn_mean   = {0.0f, 0.5f};
    std::vector<float> bn_var    = {1.0f, 0.25f};
    write_bin(dir / "weights" / "bn_weight.bin",       bn_gamma.data(), bn_gamma.size() * sizeof(float));
    write_bin(dir / "weights" / "bn_bias.bin",         bn_beta.data(),  bn_beta.size()  * sizeof(float));
    write_bin(dir / "weights" / "bn_running_mean.bin", bn_mean.data(),  bn_mean.size()  * sizeof(float));
    write_bin(dir / "weights" / "bn_running_var.bin",  bn_var.data(),   bn_var.size()   * sizeof(float));

    // manifest.json
    const char* json = R"({
  "version": "1.0",
  "model": "test_model",
  "dtype": "float32",
  "architecture": {
    "type": "nef_residual",
    "enc_channels": [32, 64],
    "num_levels": 2,
    "in_channels": 3,
    "out_channels": 3
  },
  "layers": [
    { "name": "layer0.weight",        "shape": [2, 3, 1, 1], "dtype": "float32", "file": "weights/layer0_weight.bin" },
    { "name": "layer0.bias",          "shape": [2],          "dtype": "float32", "file": "weights/layer0_bias.bin"   },
    { "name": "bn.weight",            "shape": [2],          "dtype": "float32", "file": "weights/bn_weight.bin"     },
    { "name": "bn.bias",              "shape": [2],          "dtype": "float32", "file": "weights/bn_bias.bin"       },
    { "name": "bn.running_mean",      "shape": [2],          "dtype": "float32", "file": "weights/bn_running_mean.bin" },
    { "name": "bn.running_var",       "shape": [2],          "dtype": "float32", "file": "weights/bn_running_var.bin"  }
  ]
})";
    auto manifest_path = dir / "manifest.json";
    std::ofstream mf(manifest_path);
    mf << json;
    return manifest_path;
}

// ---------------------------------------------------------------------------
// Test fixture class
// ---------------------------------------------------------------------------

class WeightLoaderTest : public ::testing::Test {
protected:
    fs::path tmp_dir_;
    fs::path manifest_path_;

    void SetUp() override {
        tmp_dir_ = fs::temp_directory_path() / "denoiser_wl_test";
        fs::remove_all(tmp_dir_);
        manifest_path_ = create_fixture(tmp_dir_);
    }

    void TearDown() override {
        fs::remove_all(tmp_dir_);
    }
};

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

TEST_F(WeightLoaderTest, MissingManifestThrows) {
    EXPECT_THROW(WeightStore("nonexistent/path/manifest.json"), std::runtime_error);
}

TEST_F(WeightLoaderTest, ParsesVersionAndLayerCount) {
    WeightStore store(manifest_path_.string());
    EXPECT_EQ(store.manifest().version, "1.0");
    EXPECT_EQ(store.manifest().layers.size(), 6u);
}

TEST_F(WeightLoaderTest, ArchitectureParsed) {
    WeightStore store(manifest_path_.string());
    const auto& arch = store.manifest().architecture;
    EXPECT_EQ(arch.type, "nef_residual");
    ASSERT_EQ(arch.enc_channels.size(), 2u);
    EXPECT_EQ(arch.enc_channels[0], 32);
    EXPECT_EQ(arch.enc_channels[1], 64);
    EXPECT_EQ(arch.num_levels, 2);
    EXPECT_EQ(arch.in_channels, 3);
    EXPECT_EQ(arch.out_channels, 3);
}

TEST_F(WeightLoaderTest, ContainsKnownKeys) {
    WeightStore store(manifest_path_.string());
    EXPECT_TRUE(store.contains("layer0.weight"));
    EXPECT_TRUE(store.contains("layer0.bias"));
    EXPECT_TRUE(store.contains("bn.weight"));
    EXPECT_TRUE(store.contains("bn.running_mean"));
}

TEST_F(WeightLoaderTest, ContainsReturnsFalseForMissingKey) {
    WeightStore store(manifest_path_.string());
    EXPECT_FALSE(store.contains("does.not.exist"));
}

TEST_F(WeightLoaderTest, GetMissingKeyThrowsOutOfRange) {
    WeightStore store(manifest_path_.string());
    EXPECT_THROW(store.get("not.a.key"), std::out_of_range);
}

TEST_F(WeightLoaderTest, GetUploadsCorrectShape) {
    SKIP_IF_NO_GPU();
    WeightStore store(manifest_path_.string());
    const Tensor& w = store.get("layer0.weight");
    cudaDeviceSynchronize();

    ASSERT_EQ(w.shape().size(), 4u);
    EXPECT_EQ(w.shape()[0], 2);
    EXPECT_EQ(w.shape()[1], 3);
    EXPECT_EQ(w.shape()[2], 1);
    EXPECT_EQ(w.shape()[3], 1);
    EXPECT_EQ(w.dtype(), DType::kFloat32);
}

TEST_F(WeightLoaderTest, GetRoundtripsData) {
    SKIP_IF_NO_GPU();
    WeightStore store(manifest_path_.string());
    const Tensor& w = store.get("layer0.weight");
    cudaDeviceSynchronize();

    std::vector<float> out(6, 0.f);
    w.to_host(out.data());
    cudaDeviceSynchronize();

    for (int i = 0; i < 6; ++i) {
        EXPECT_FLOAT_EQ(out[i], static_cast<float>(i + 1)) << "index " << i;
    }
}

TEST_F(WeightLoaderTest, GetReturnsSameReferenceOnRepeat) {
    SKIP_IF_NO_GPU();
    WeightStore store(manifest_path_.string());
    const Tensor& w1 = store.get("layer0.weight");
    const Tensor& w2 = store.get("layer0.weight");
    EXPECT_EQ(w1.data(), w2.data());  // same device pointer
}

TEST_F(WeightLoaderTest, DeviceBytesGrowsAfterFirstGet) {
    SKIP_IF_NO_GPU();
    WeightStore store(manifest_path_.string());
    size_t before = store.device_bytes_allocated();
    store.get("layer0.weight");
    cudaDeviceSynchronize();
    EXPECT_GT(store.device_bytes_allocated(), before);
}

TEST_F(WeightLoaderTest, PrefetchAllUploadsAllLayers) {
    SKIP_IF_NO_GPU();
    WeightStore store(manifest_path_.string());
    ASSERT_NO_THROW(store.prefetch_all());
    cudaDeviceSynchronize();

    // After prefetch, every layer should already be in cache
    size_t after_prefetch = store.device_bytes_allocated();

    // Repeated get() calls should not increase allocation further
    for (const auto& layer : store.manifest().layers) {
        store.get(layer.name);
    }
    EXPECT_EQ(store.device_bytes_allocated(), after_prefetch);
}

TEST_F(WeightLoaderTest, BiasRoundtripsData) {
    SKIP_IF_NO_GPU();
    WeightStore store(manifest_path_.string());
    const Tensor& b = store.get("layer0.bias");
    cudaDeviceSynchronize();

    ASSERT_EQ(b.numel(), 2);
    std::vector<float> out(2, 0.f);
    b.to_host(out.data());
    cudaDeviceSynchronize();

    EXPECT_NEAR(out[0], 0.1f, 1e-6f);
    EXPECT_NEAR(out[1], 0.2f, 1e-6f);
}
