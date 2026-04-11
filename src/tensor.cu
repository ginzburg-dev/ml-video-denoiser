#include "denoiser/tensor.h"

#include <cuda_runtime.h>
#include <sstream>
#include <stdexcept>

namespace denoiser {

// ---------------------------------------------------------------------------
// Factory: empty
// ---------------------------------------------------------------------------

Tensor Tensor::empty(std::vector<int64_t> shape, DType dtype, cudaStream_t /*stream*/) {
    int64_t n = 1;
    for (const int64_t d : shape) {
        if (d <= 0) throw std::invalid_argument("Tensor::empty: all dimensions must be > 0");
        n *= d;
    }
    const size_t bytes = static_cast<size_t>(n) * dtype_size(dtype);
    void* ptr = nullptr;
    CUDA_CHECK(cudaMalloc(&ptr, bytes));
    return Tensor(ptr, std::move(shape), dtype, /*owns_memory=*/true);
}

// ---------------------------------------------------------------------------
// Factory: from_host
// ---------------------------------------------------------------------------

Tensor Tensor::from_host(const void* data, std::vector<int64_t> shape,
                          DType dtype, cudaStream_t stream) {
    if (!data) throw std::invalid_argument("Tensor::from_host: data must not be null");
    auto t = Tensor::empty(shape, dtype);
    const size_t bytes = t.nbytes();
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(t.d_ptr_, data, bytes, cudaMemcpyHostToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(t.d_ptr_, data, bytes, cudaMemcpyHostToDevice));
    }
    return t;
}

// ---------------------------------------------------------------------------
// Move
// ---------------------------------------------------------------------------

Tensor::Tensor(Tensor&& other) noexcept
    : d_ptr_(other.d_ptr_),
      shape_(std::move(other.shape_)),
      dtype_(other.dtype_),
      owns_memory_(other.owns_memory_) {
    other.d_ptr_ = nullptr;
    other.owns_memory_ = false;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        if (owns_memory_ && d_ptr_) cudaFree(d_ptr_);
        d_ptr_ = other.d_ptr_;
        shape_ = std::move(other.shape_);
        dtype_ = other.dtype_;
        owns_memory_ = other.owns_memory_;
        other.d_ptr_ = nullptr;
        other.owns_memory_ = false;
    }
    return *this;
}

// ---------------------------------------------------------------------------
// Destructor
// ---------------------------------------------------------------------------

Tensor::~Tensor() {
    if (owns_memory_ && d_ptr_) {
        cudaFree(d_ptr_);  // Intentionally not checking error — destructor must not throw
        d_ptr_ = nullptr;
    }
}

// ---------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------

Tensor Tensor::clone(cudaStream_t stream) const {
    if (!is_valid()) return Tensor{};
    auto t = Tensor::empty(shape_, dtype_);
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(t.d_ptr_, d_ptr_, nbytes(), cudaMemcpyDeviceToDevice, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(t.d_ptr_, d_ptr_, nbytes(), cudaMemcpyDeviceToDevice));
    }
    return t;
}

// ---------------------------------------------------------------------------
// D2H transfer
// ---------------------------------------------------------------------------

void Tensor::to_host(void* out, cudaStream_t stream) const {
    if (!is_valid()) throw std::runtime_error("Tensor::to_host: tensor is empty");
    if (!out) throw std::invalid_argument("Tensor::to_host: output pointer must not be null");
    if (stream) {
        CUDA_CHECK(cudaMemcpyAsync(out, d_ptr_, nbytes(), cudaMemcpyDeviceToHost, stream));
    } else {
        CUDA_CHECK(cudaMemcpy(out, d_ptr_, nbytes(), cudaMemcpyDeviceToHost));
    }
}

// ---------------------------------------------------------------------------
// numel
// ---------------------------------------------------------------------------

int64_t Tensor::numel() const noexcept {
    if (shape_.empty()) return 0;
    int64_t n = 1;
    for (const int64_t d : shape_) n *= d;
    return n;
}

// ---------------------------------------------------------------------------
// Dimension accessors
// ---------------------------------------------------------------------------

int64_t Tensor::dim_checked(int i) const {
    if (shape_.size() != 4) {
        throw std::runtime_error(
            "Tensor: NCHW dimension accessor requires ndim==4, got " +
            std::to_string(shape_.size()));
    }
    return shape_[static_cast<size_t>(i)];
}

// ---------------------------------------------------------------------------
// Non-owning channel slice  [start, end)
// ---------------------------------------------------------------------------

Tensor Tensor::slice_channels(int64_t start, int64_t end) const {
    if (shape_.size() != 4) {
        throw std::runtime_error("slice_channels requires a 4-D (NCHW) tensor");
    }
    const int64_t C = shape_[1];
    if (start < 0 || end > C || start >= end) {
        throw std::out_of_range(
            "slice_channels: invalid range [" + std::to_string(start) +
            ", " + std::to_string(end) + ") for C=" + std::to_string(C));
    }
    const int64_t N = shape_[0], H = shape_[2], W = shape_[3];
    const size_t elem_bytes = dtype_size(dtype_);
    // Offset into the channel dimension: each channel is H*W elements
    uint8_t* base = static_cast<uint8_t*>(d_ptr_);
    // For batch index 0: the slice starts at channel *start*
    // NB: this is only correct when N==1 or when slicing along the channel
    // dimension with the assumption that batches are laid out contiguously.
    // A full strided view would require stride metadata — keep it simple for now.
    void* slice_ptr = base + static_cast<size_t>(start) * static_cast<size_t>(H * W) * elem_bytes;
    std::vector<int64_t> slice_shape = {N, end - start, H, W};
    return Tensor(slice_ptr, std::move(slice_shape), dtype_, /*owns_memory=*/false);
}

// ---------------------------------------------------------------------------
// cuDNN descriptor
// ---------------------------------------------------------------------------

cudnnTensorDescriptor_t Tensor::make_cudnn_descriptor() const {
    if (shape_.size() != 4) {
        throw std::runtime_error("make_cudnn_descriptor requires a 4-D (NCHW) tensor");
    }
    cudnnTensorDescriptor_t desc;
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(
        desc,
        CUDNN_TENSOR_NCHW,
        to_cudnn_dtype(dtype_),
        static_cast<int>(shape_[0]),
        static_cast<int>(shape_[1]),
        static_cast<int>(shape_[2]),
        static_cast<int>(shape_[3])
    ));
    return desc;
}

// ---------------------------------------------------------------------------
// Debug
// ---------------------------------------------------------------------------

std::string Tensor::shape_str() const {
    std::ostringstream oss;
    oss << "(";
    for (size_t i = 0; i < shape_.size(); ++i) {
        oss << shape_[i];
        if (i + 1 < shape_.size()) oss << ", ";
    }
    oss << ")";
    return oss.str();
}

} // namespace denoiser
