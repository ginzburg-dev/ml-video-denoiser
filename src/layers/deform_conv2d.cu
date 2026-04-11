#include "denoiser/layers/deform_conv2d.h"
#include "../../src/kernels/deform_im2col.cuh"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <stdexcept>
#include <string>

// ---------------------------------------------------------------------------
// File-scope kernels (must be visible before any host function that calls them)
// ---------------------------------------------------------------------------

namespace {

// Broadcast-add bias vector to output.
// output layout: (N, C_out, H_out, W_out) — linear index n*C*HW + c*HW + hw
__global__ void add_bias_kernel(
    __half* __restrict__       output,
    const __half* __restrict__ bias,
    int C_out, int HW, int64_t total
) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int c = static_cast<int>((i / HW) % C_out);
    output[i] = __float2half(__half2float(output[i]) + __half2float(bias[c]));
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// deform_im2col launcher (declared in deform_im2col.cuh, defined here so
// that the kernel __global__ in the .cuh is compiled in this translation unit)
// ---------------------------------------------------------------------------

namespace denoiser::kernels {

void launch_deform_im2col(
    const __half* input,
    const __half* offsets,
    const __half* masks,
    __half*       col_buffer,
    int N, int C_in, int H_in, int W_in,
    int kH, int kW,
    int H_out, int W_out,
    int stride, int pad, int dilation,
    int deform_groups,
    cudaStream_t stream
) {
    const int64_t total = static_cast<int64_t>(N) * C_in * kH * kW * H_out * W_out;
    constexpr int kBlock = 256;
    const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
    deform_im2col_kernel<<<blocks, kBlock, 0, stream>>>(
        input, offsets, masks, col_buffer,
        N, C_in, H_in, W_in,
        kH, kW, H_out, W_out,
        stride, pad, dilation, deform_groups);
}

} // namespace denoiser::kernels

// ---------------------------------------------------------------------------
// DeformConv2dLayer
// ---------------------------------------------------------------------------

namespace denoiser {

static void check_cublas(cublasStatus_t s, const char* file, int line) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) +
            " cuBLAS error " + std::to_string(static_cast<int>(s)));
    }
}
#define CUBLAS_CHECK(expr) check_cublas((expr), __FILE__, __LINE__)

DeformConv2dLayer::DeformConv2dLayer(
    const WeightStore& store,
    const std::string& weight_name,
    const std::string& bias_name,
    int pad, int stride, int dilation, int deform_groups)
    : pad_(pad), stride_(stride), dilation_(dilation), deform_groups_(deform_groups)
{
    weight_ = &store.get(weight_name);
    bias_   = &store.get(bias_name);

    const auto& sh = weight_->shape();
    if (sh.size() != 4) {
        throw std::runtime_error(
            "DeformConv2dLayer: weight must be 4-D, got " +
            std::to_string(sh.size()) + "-D");
    }
    out_channels_ = static_cast<int>(sh[0]);
    in_channels_  = static_cast<int>(sh[1]);
    kernel_h_     = static_cast<int>(sh[2]);
    kernel_w_     = static_cast<int>(sh[3]);

    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUBLAS_CHECK(cublasSetMathMode(cublas_, CUBLAS_TENSOR_OP_MATH));
}

DeformConv2dLayer::~DeformConv2dLayer() {
    if (cublas_) cublasDestroy(cublas_);
}

Tensor DeformConv2dLayer::forward(
    const Tensor& input,
    const Tensor& offsets,
    const Tensor& masks,
    cudaStream_t  stream) const
{
    if (input.dtype()   != DType::kFloat16 ||
        offsets.dtype() != DType::kFloat16 ||
        masks.dtype()   != DType::kFloat16) {
        throw std::runtime_error(
            "DeformConv2dLayer::forward: all inputs must be FP16");
    }

    const int N    = static_cast<int>(input.batch());
    const int H_in = static_cast<int>(input.height());
    const int W_in = static_cast<int>(input.width());
    const int kH   = kernel_h_;
    const int kW   = kernel_w_;
    const int K    = in_channels_ * kH * kW;
    const int HW   = ((H_in + 2 * pad_ - dilation_ * (kH - 1) - 1) / stride_ + 1)
                   * ((W_in + 2 * pad_ - dilation_ * (kW - 1) - 1) / stride_ + 1);
    const int H_out = (H_in + 2 * pad_ - dilation_ * (kH - 1) - 1) / stride_ + 1;
    const int W_out = (W_in + 2 * pad_ - dilation_ * (kW - 1) - 1) / stride_ + 1;

    if (stream) CUBLAS_CHECK(cublasSetStream(cublas_, stream));

    // 1. Im2col → col_buffer (K, N*HW) stored col-major
    //    The kernel writes to col_buffer[col_idx * N*HW + row_idx]
    //    so CUBLAS will see it as a (N*HW × K) col-major matrix (= K × N*HW row-major)
    auto col_buf = Tensor::empty({K, static_cast<int64_t>(N) * HW}, DType::kFloat16);
    kernels::launch_deform_im2col(
        input.data_f16(),
        offsets.data_f16(),
        masks.data_f16(),
        col_buf.data_f16(),
        N, in_channels_, H_in, W_in,
        kH, kW, H_out, W_out,
        stride_, pad_, dilation_,
        deform_groups_,
        stream);

    // 2. GEMM: output[C_out × N*HW] = weight[C_out × K] × col_buf[K × N*HW]
    //
    // cuBLAS col-major trick for row-major A×B = C:
    //   swap operands and pass (n=N*HW, m=C_out, k=K):
    //   CUBLAS: C_col[n×m] = B_col[n×k] × A_col[k×m]
    //   where B_col = col_buf (row-major [K×N*HW] → col-major [N*HW×K])
    //         A_col = weight  (row-major [C_out×K] → col-major [K×C_out])
    //   → row-major result: output[C_out × N*HW] ✓
    auto output = Tensor::empty(
        {N, out_channels_, H_out, W_out}, DType::kFloat16);

    const __half alpha_h = __float2half(1.0f);
    const __half beta_h  = __float2half(0.0f);

    CUBLAS_CHECK(cublasHgemm(
        cublas_,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N * HW,               // n — rows of col_buf (CUBLAS col-major view)
        out_channels_,        // m — cols of weight  (CUBLAS col-major view)
        K,                    // k — inner dim
        &alpha_h,
        col_buf.data_f16(), N * HW,  // B: col_buf, ldb = n = N*HW
        weight_->data_f16(), K,      // A: weight,  lda = k = K
        &beta_h,
        output.data_f16(),  N * HW   // C: output,  ldc = n = N*HW
    ));

    // 3. Broadcast-add bias: output[n,c,h,w] += bias[c]
    if (bias_) {
        const int64_t total = static_cast<int64_t>(N) * out_channels_ * H_out * W_out;
        constexpr int kBlock = 256;
        const int blocks = static_cast<int>((total + kBlock - 1) / kBlock);
        add_bias_kernel<<<blocks, kBlock, 0, stream>>>(
            output.data_f16(), bias_->data_f16(),
            out_channels_, H_out * W_out, total);
    }

    return output;
}

} // namespace denoiser
