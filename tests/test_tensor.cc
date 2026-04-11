#include <gtest/gtest.h>

#include "denoiser/tensor.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>
#include <numeric>
#include <vector>

using namespace denoiser;

static bool has_cuda_device() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

#define SKIP_IF_NO_GPU() \
    if (!has_cuda_device()) GTEST_SKIP() << "No CUDA device available"

// ---- Construction -----------------------------------------------------------

TEST(TensorTest, DefaultConstructedIsInvalid) {
    Tensor t;
    EXPECT_FALSE(t.is_valid());
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.nbytes(), 0u);
    EXPECT_EQ(t.data(), nullptr);
}

TEST(TensorTest, EmptyAllocatesShape) {
    SKIP_IF_NO_GPU();
    auto t = Tensor::empty({2, 3, 4, 5}, DType::kFloat32);
    EXPECT_TRUE(t.is_valid());
    ASSERT_EQ(t.shape().size(), 4u);
    EXPECT_EQ(t.batch(), 2);
    EXPECT_EQ(t.channels(), 3);
    EXPECT_EQ(t.height(), 4);
    EXPECT_EQ(t.width(), 5);
    EXPECT_EQ(t.numel(), 2 * 3 * 4 * 5);
    EXPECT_EQ(t.nbytes(), static_cast<size_t>(2 * 3 * 4 * 5) * sizeof(float));
    EXPECT_EQ(t.dtype(), DType::kFloat32);
}

TEST(TensorTest, EmptyFP16CorrectByteSize) {
    SKIP_IF_NO_GPU();
    auto t = Tensor::empty({1, 64, 8, 8}, DType::kFloat16);
    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.nbytes(), static_cast<size_t>(1 * 64 * 8 * 8) * sizeof(__half));
}

TEST(TensorTest, EmptyNonCHWShape) {
    SKIP_IF_NO_GPU();
    // 1-D tensors are valid (used for BN stats)
    auto t = Tensor::empty({64}, DType::kFloat32);
    EXPECT_TRUE(t.is_valid());
    EXPECT_EQ(t.numel(), 64);
}

// ---- H2D / D2H roundtrip ----------------------------------------------------

TEST(TensorTest, FromHostToHostRoundtripFP32) {
    SKIP_IF_NO_GPU();
    std::vector<float> in(12);
    std::iota(in.begin(), in.end(), 1.0f);

    auto t = Tensor::from_host(in.data(), {1, 3, 2, 2}, DType::kFloat32);
    ASSERT_TRUE(t.is_valid());
    cudaDeviceSynchronize();

    std::vector<float> out(12, 0.0f);
    t.to_host(out.data());
    cudaDeviceSynchronize();

    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(out[i], in[i]) << "mismatch at i=" << i;
    }
}

TEST(TensorTest, FromHostToHostRoundtripFP16) {
    SKIP_IF_NO_GPU();
    constexpr int N = 8;
    std::vector<__half> in(N);
    for (int i = 0; i < N; ++i) {
        in[i] = __float2half(static_cast<float>(i) * 0.5f);
    }

    auto t = Tensor::from_host(in.data(), {1, 2, 2, 2}, DType::kFloat16);
    cudaDeviceSynchronize();

    std::vector<__half> out(N);
    t.to_host(out.data());
    cudaDeviceSynchronize();

    for (int i = 0; i < N; ++i) {
        EXPECT_FLOAT_EQ(__half2float(out[i]), __half2float(in[i]))
            << "mismatch at i=" << i;
    }
}

// ---- Clone ------------------------------------------------------------------

TEST(TensorTest, CloneHasDifferentPointer) {
    SKIP_IF_NO_GPU();
    std::vector<float> data = {1.f, 2.f, 3.f, 4.f};
    auto t = Tensor::from_host(data.data(), {1, 1, 2, 2}, DType::kFloat32);
    cudaDeviceSynchronize();

    auto c = t.clone();
    EXPECT_NE(t.data(), c.data());
    EXPECT_EQ(c.shape(), t.shape());
    EXPECT_EQ(c.dtype(), t.dtype());
}

TEST(TensorTest, CloneContentsMatch) {
    SKIP_IF_NO_GPU();
    std::vector<float> data = {10.f, 20.f, 30.f, 40.f};
    auto t = Tensor::from_host(data.data(), {1, 1, 2, 2}, DType::kFloat32);
    cudaDeviceSynchronize();

    auto c = t.clone();
    cudaDeviceSynchronize();

    std::vector<float> out(4, 0.f);
    c.to_host(out.data());
    cudaDeviceSynchronize();

    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(out[i], data[i]);
    }
}

// ---- Move semantics ---------------------------------------------------------

TEST(TensorTest, MoveCtor) {
    SKIP_IF_NO_GPU();
    auto t = Tensor::empty({1, 3, 4, 4}, DType::kFloat32);
    void* ptr = t.data();

    Tensor t2 = std::move(t);
    EXPECT_FALSE(t.is_valid());  // NOLINT(bugprone-use-after-move)
    EXPECT_TRUE(t2.is_valid());
    EXPECT_EQ(t2.data(), ptr);
    EXPECT_EQ(t2.channels(), 3);
}

TEST(TensorTest, MoveAssignment) {
    SKIP_IF_NO_GPU();
    auto t1 = Tensor::empty({1, 1, 4, 4}, DType::kFloat32);
    auto t2 = Tensor::empty({1, 2, 8, 8}, DType::kFloat32);
    void* ptr2 = t2.data();

    t1 = std::move(t2);
    EXPECT_EQ(t1.data(), ptr2);
    EXPECT_FALSE(t2.is_valid());  // NOLINT(bugprone-use-after-move)
    EXPECT_EQ(t1.channels(), 2);
    EXPECT_EQ(t1.height(), 8);
}

TEST(TensorTest, SelfMoveAssignmentIsSafe) {
    SKIP_IF_NO_GPU();
    auto t = Tensor::empty({1, 3, 4, 4}, DType::kFloat32);
    void* ptr = t.data();

    // Simulate self-move — should not crash or double-free
    t = std::move(t);  // NOLINT(clang-diagnostic-self-move)
    // After self-move the state is unspecified but must not crash at destruction
    (void)ptr;
}

// ---- slice_channels ---------------------------------------------------------

TEST(TensorTest, SliceChannelsShape) {
    SKIP_IF_NO_GPU();
    auto t = Tensor::empty({2, 8, 4, 4}, DType::kFloat16);

    auto s = t.slice_channels(0, 4);
    EXPECT_TRUE(s.is_valid());
    EXPECT_EQ(s.batch(), 2);
    EXPECT_EQ(s.channels(), 4);
    EXPECT_EQ(s.height(), 4);
    EXPECT_EQ(s.width(), 4);
}

TEST(TensorTest, SliceChannelsPointerOffset) {
    SKIP_IF_NO_GPU();
    // (1, 4, 1, 1) FP32: each channel is one float at a known offset
    std::vector<float> data = {0.f, 1.f, 2.f, 3.f};
    auto t = Tensor::from_host(data.data(), {1, 4, 1, 1}, DType::kFloat32);
    cudaDeviceSynchronize();

    // Slice [2, 4) → data[2], data[3]
    auto s = t.slice_channels(2, 4);
    EXPECT_EQ(s.channels(), 2);

    // The slice pointer must be offset by 2 floats into the parent buffer
    const float* parent_ptr = static_cast<const float*>(t.data());
    const float* slice_ptr  = static_cast<const float*>(s.data());
    EXPECT_EQ(slice_ptr, parent_ptr + 2);
}

TEST(TensorTest, SliceChannelsDataCorrect) {
    SKIP_IF_NO_GPU();
    std::vector<float> data = {0.f, 1.f, 2.f, 3.f};
    auto t = Tensor::from_host(data.data(), {1, 4, 1, 1}, DType::kFloat32);
    cudaDeviceSynchronize();

    auto s = t.slice_channels(1, 3);
    std::vector<float> out(2, -1.f);
    s.to_host(out.data());
    cudaDeviceSynchronize();

    EXPECT_FLOAT_EQ(out[0], 1.f);
    EXPECT_FLOAT_EQ(out[1], 2.f);
}

// ---- NCHW accessor guards ---------------------------------------------------

TEST(TensorTest, NCHWAccessorsRequire4DShape) {
    SKIP_IF_NO_GPU();
    auto t = Tensor::empty({3, 4}, DType::kFloat32);
    EXPECT_THROW(t.batch(),    std::runtime_error);
    EXPECT_THROW(t.channels(), std::runtime_error);
    EXPECT_THROW(t.height(),   std::runtime_error);
    EXPECT_THROW(t.width(),    std::runtime_error);
}

// ---- cuDNN interop ----------------------------------------------------------

TEST(TensorTest, MakeCudnnDescriptorSucceeds) {
    SKIP_IF_NO_GPU();
    auto t = Tensor::empty({1, 3, 32, 32}, DType::kFloat16);
    cudnnTensorDescriptor_t desc = nullptr;
    ASSERT_NO_THROW(desc = t.make_cudnn_descriptor());
    ASSERT_NE(desc, nullptr);
    cudnnDestroyTensorDescriptor(desc);
}

// ---- shape_str --------------------------------------------------------------

TEST(TensorTest, ShapeStrIsNonEmpty) {
    SKIP_IF_NO_GPU();
    auto t = Tensor::empty({1, 3, 224, 224}, DType::kFloat32);
    EXPECT_FALSE(t.shape_str().empty());
}
