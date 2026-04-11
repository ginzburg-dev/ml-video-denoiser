#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cudnn.h>

#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace denoiser {

// ---------------------------------------------------------------------------
// Data type
// ---------------------------------------------------------------------------

enum class DType : uint8_t { kFloat32, kFloat16 };

constexpr size_t dtype_size(DType dtype) noexcept {
    return dtype == DType::kFloat16 ? sizeof(__half) : sizeof(float);
}

constexpr cudnnDataType_t to_cudnn_dtype(DType dtype) noexcept {
    return dtype == DType::kFloat16 ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;
}

// ---------------------------------------------------------------------------
// Tensor
// ---------------------------------------------------------------------------

// RAII wrapper around a CUDA device memory allocation.
//
// Ownership model:
//   - Default-constructed Tensor is empty (no allocation).
//   - Tensors are move-only.  Use clone() for explicit deep copies.
//   - slice_channels() returns a non-owning view — the source Tensor must
//     outlive any views derived from it.
//
// Layout: NCHW (N=batch, C=channels, H=height, W=width).
// All shapes are stored as int64_t vectors matching PyTorch conventions.
class Tensor {
public:
    // --- Construction ----------------------------------------------------------

    Tensor() = default;

    // Allocate uninitialised device memory for the given shape and dtype.
    // Does not synchronise — use with care before kernels that read this buffer.
    static Tensor empty(std::vector<int64_t> shape, DType dtype,
                        cudaStream_t stream = nullptr);

    // Copy *n_bytes* from host *data* to a new device allocation.
    // Performs an async H2D transfer on *stream* (synchronous if nullptr).
    static Tensor from_host(const void* data, std::vector<int64_t> shape,
                            DType dtype, cudaStream_t stream = nullptr);

    // --- Move semantics --------------------------------------------------------

    Tensor(Tensor&& other) noexcept;
    Tensor& operator=(Tensor&& other) noexcept;

    // Copying is disabled — use clone() for an explicit deep copy.
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    ~Tensor();

    // --- Deep copy -------------------------------------------------------------

    // Allocate new device memory and copy this tensor's contents into it.
    Tensor clone(cudaStream_t stream = nullptr) const;

    // --- D2H transfer ----------------------------------------------------------

    // Copy device data to *out* (must point to at least nbytes() host bytes).
    // Performs an async D2H transfer on *stream* (synchronous if nullptr).
    void to_host(void* out, cudaStream_t stream = nullptr) const;

    // --- Accessors -------------------------------------------------------------

    const std::vector<int64_t>& shape() const noexcept { return shape_; }
    DType dtype() const noexcept { return dtype_; }

    // Raw device pointer.  Avoid storing; prefer passing the Tensor itself.
    void* data() noexcept { return d_ptr_; }
    const void* data() const noexcept { return d_ptr_; }

    // Typed device pointers (no runtime type check — caller must ensure dtype).
    float* data_f32() noexcept { return static_cast<float*>(d_ptr_); }
    const float* data_f32() const noexcept { return static_cast<const float*>(d_ptr_); }
    __half* data_f16() noexcept { return static_cast<__half*>(d_ptr_); }
    const __half* data_f16() const noexcept { return static_cast<const __half*>(d_ptr_); }

    bool is_valid() const noexcept { return d_ptr_ != nullptr; }

    // Total number of elements.
    int64_t numel() const noexcept;

    // Total size in bytes.
    size_t nbytes() const noexcept { return static_cast<size_t>(numel()) * dtype_size(dtype_); }

    // Convenience accessors for NCHW dimensions (requires ndim == 4).
    int64_t batch() const { return dim_checked(0); }
    int64_t channels() const { return dim_checked(1); }
    int64_t height() const { return dim_checked(2); }
    int64_t width() const { return dim_checked(3); }

    // --- Non-owning channel slice [start, end) ---------------------------------

    // Returns a view into this tensor's channel dimension.
    // The returned Tensor does NOT own the underlying memory.
    // The source Tensor must remain alive (and unmodified) for the view's lifetime.
    Tensor slice_channels(int64_t start, int64_t end) const;

    // --- cuDNN interop ---------------------------------------------------------

    // Create a cudnnTensorDescriptor_t describing this tensor's shape and dtype.
    // Caller is responsible for calling cudnnDestroyTensorDescriptor().
    cudnnTensorDescriptor_t make_cudnn_descriptor() const;

    // --- Debug -----------------------------------------------------------------

    std::string shape_str() const;

private:
    void* d_ptr_ = nullptr;
    std::vector<int64_t> shape_;
    DType dtype_ = DType::kFloat32;
    bool owns_memory_ = true;

    // Private constructor used by factory methods and slice_channels.
    Tensor(void* d_ptr, std::vector<int64_t> shape, DType dtype, bool owns_memory) noexcept
        : d_ptr_(d_ptr), shape_(std::move(shape)), dtype_(dtype), owns_memory_(owns_memory) {}

    int64_t dim_checked(int i) const;
};

// ---------------------------------------------------------------------------
// CUDA error helpers (used throughout the engine)
// ---------------------------------------------------------------------------

[[nodiscard]] inline std::string cuda_error_string(cudaError_t err) {
    return std::string("CUDA error: ") + cudaGetErrorString(err);
}

inline void check_cuda(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + " " +
            cuda_error_string(err));
    }
}

inline void check_cudnn(cudnnStatus_t status, const char* file, int line) {
    if (status != CUDNN_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) +
            " cuDNN error: " + cudnnGetErrorString(status));
    }
}

#define CUDA_CHECK(expr)  denoiser::check_cuda((expr),  __FILE__, __LINE__)
#define CUDNN_CHECK(expr) denoiser::check_cudnn((expr), __FILE__, __LINE__)

} // namespace denoiser
