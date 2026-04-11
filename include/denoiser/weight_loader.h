#pragma once

#include "denoiser/tensor.h"

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace denoiser {

// ---------------------------------------------------------------------------
// Manifest entry
// ---------------------------------------------------------------------------

struct LayerEntry {
    std::string name;           // e.g. "encoder.0.conv1.weight"
    std::vector<int64_t> shape; // e.g. [64, 3, 3, 3]
    DType dtype;
    std::string file;           // relative path to .bin file from manifest dir
};

struct Manifest {
    std::string version;
    DType dtype;                // default dtype for the model
    std::vector<LayerEntry> layers;

    struct Architecture {
        std::string type;       // "nef_residual" or "nef_temporal"
        std::vector<int> enc_channels;
        int num_levels = 0;
        int in_channels = 3;
        int out_channels = 3;
        int num_frames = 5;     // NEFTemporal only
        int deform_groups = 8;  // NEFTemporal only
    } architecture;
};

// ---------------------------------------------------------------------------
// WeightStore
// ---------------------------------------------------------------------------

// Loads and caches model weights from a manifest.json + directory of .bin files.
//
// Strategy:
//   - .bin files are memory-mapped into the process address space at
//     construction time.  This avoids reading the whole file upfront and lets
//     the OS page data in lazily.
//   - Device tensors (Tensor) are allocated and H2D-copied lazily on the
//     first call to get().  This prevents uploading the full ~500 MB of
//     weights for large models before any layer needs them.
//   - get() is thread-safe for concurrent reads once all weights are uploaded,
//     but the lazy upload path uses a simple mutex.
//
// Usage:
//   WeightStore store("weights/residual_standard/manifest.json");
//   const Tensor& w = store.get("encoder.0.conv1.weight");
class WeightStore {
public:
    explicit WeightStore(const std::string& manifest_path);
    ~WeightStore();

    // Returns a const reference to the device Tensor for *name*.
    // Uploads to GPU on first access.
    // Throws std::out_of_range if *name* is not in the manifest.
    const Tensor& get(const std::string& name) const;

    // Returns true if the manifest contains a layer with this name.
    bool contains(const std::string& name) const;

    // Parsed manifest metadata.
    const Manifest& manifest() const noexcept { return manifest_; }

    // Pre-upload all tensors eagerly (useful for benchmarking cold-start).
    void prefetch_all(cudaStream_t stream = nullptr) const;

    // Total device memory currently allocated by this store (bytes).
    size_t device_bytes_allocated() const;

private:
    struct MappedFile {
        void* ptr = nullptr;
        size_t size = 0;
        int fd = -1;
    };

    Manifest manifest_;
    std::string base_dir_;  // directory containing the .bin files

    // Per-layer host view (mmap'd)
    mutable std::unordered_map<std::string, MappedFile> host_maps_;
    // Per-layer device tensor (lazy)
    mutable std::unordered_map<std::string, Tensor> device_cache_;

    void parse_manifest(const std::string& path);
    const Tensor& upload(const std::string& name) const;
    static MappedFile mmap_file(const std::string& path);
};

} // namespace denoiser
