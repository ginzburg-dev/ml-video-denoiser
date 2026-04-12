#include "denoiser/weight_loader.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace denoiser {

namespace fs = std::filesystem;
using json = nlohmann::json;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static DType dtype_from_string(const std::string& s) {
    if (s == "float16") return DType::kFloat16;
    if (s == "float32") return DType::kFloat32;
    throw std::runtime_error("Unknown dtype string: " + s);
}

// Memory-map a file for read-only access.
WeightStore::MappedFile WeightStore::mmap_file(const std::string& path) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd == -1) {
        throw std::runtime_error("Cannot open weight file: " + path +
                                 " (" + std::string(strerror(errno)) + ")");
    }
    struct stat st{};
    if (::fstat(fd, &st) == -1) {
        ::close(fd);
        throw std::runtime_error("Cannot stat: " + path);
    }
    const size_t size = static_cast<size_t>(st.st_size);
    if (size == 0) {
        // Zero-byte file — valid for num_batches_tracked etc.
        ::close(fd);
        return {nullptr, 0, -1};
    }
    void* ptr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (ptr == MAP_FAILED) {
        ::close(fd);
        throw std::runtime_error("mmap failed for: " + path);
    }
    // fd can be closed after mmap — the mapping persists until munmap
    ::close(fd);
    return {ptr, size, -1};
}

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------

WeightStore::WeightStore(const std::string& manifest_path) {
    const fs::path mp(manifest_path);
    if (!fs::exists(mp)) {
        throw std::runtime_error("Manifest not found: " + manifest_path);
    }
    base_dir_ = mp.parent_path().string();
    parse_manifest(manifest_path);

    // Pre-map all .bin files into host memory
    for (const auto& entry : manifest_.layers) {
        const std::string bin_path = (fs::path(base_dir_) / entry.file).string();
        host_maps_[entry.name] = mmap_file(bin_path);
    }
}

WeightStore::~WeightStore() {
    for (auto& [name, mf] : host_maps_) {
        if (mf.ptr && mf.size > 0) {
            ::munmap(mf.ptr, mf.size);
            mf.ptr = nullptr;
        }
    }
}

// ---------------------------------------------------------------------------
// parse_manifest
// ---------------------------------------------------------------------------

void WeightStore::parse_manifest(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open manifest: " + path);
    }
    const json j = json::parse(f);

    manifest_.version = j.value("version", "1.0");
    manifest_.dtype = dtype_from_string(j.value("dtype", "float16"));

    // Architecture
    if (j.contains("architecture")) {
        const auto& arch = j["architecture"];
        manifest_.architecture.type = arch.value("type", "nafnet_residual");
        manifest_.architecture.base_channels = arch.value("base_channels", 32);
        manifest_.architecture.in_channels = arch.value("in_channels", 3);
        manifest_.architecture.out_channels = arch.value("out_channels", 3);
        manifest_.architecture.num_levels = arch.value("num_levels", 4);
        manifest_.architecture.num_frames = arch.value("num_frames", 5);
        manifest_.architecture.use_warp = arch.value("use_warp", false);
        if (arch.contains("enc_channels")) {
            for (const auto& ch : arch["enc_channels"]) {
                manifest_.architecture.enc_channels.push_back(ch.get<int>());
            }
        }
    }

    // Layers
    for (const auto& layer : j["layers"]) {
        LayerEntry entry;
        entry.name = layer["name"].get<std::string>();
        entry.dtype = dtype_from_string(layer["dtype"].get<std::string>());
        entry.file = layer["file"].get<std::string>();
        for (const auto& d : layer["shape"]) {
            entry.shape.push_back(d.get<int64_t>());
        }
        manifest_.layers.push_back(std::move(entry));
    }
}

// ---------------------------------------------------------------------------
// get / upload
// ---------------------------------------------------------------------------

static std::mutex g_upload_mutex;

const Tensor& WeightStore::get(const std::string& name) const {
    {
        // Fast path: already on device (no lock needed for reads after upload)
        const auto it = device_cache_.find(name);
        if (it != device_cache_.end()) {
            return it->second;
        }
    }
    return upload(name);
}

const Tensor& WeightStore::upload(const std::string& name) const {
    std::lock_guard<std::mutex> lock(g_upload_mutex);

    // Double-checked locking
    const auto it = device_cache_.find(name);
    if (it != device_cache_.end()) return it->second;

    // Find manifest entry
    const LayerEntry* entry = nullptr;
    for (const auto& e : manifest_.layers) {
        if (e.name == name) { entry = &e; break; }
    }
    if (!entry) {
        throw std::out_of_range("WeightStore: layer not found: " + name);
    }

    const auto& mf = host_maps_.at(name);
    if (!mf.ptr || mf.size == 0) {
        // Zero-size tensor (e.g. num_batches_tracked scalar)
        auto t = Tensor::empty(entry->shape, entry->dtype);
        device_cache_.emplace(name, std::move(t));
        return device_cache_.at(name);
    }

    auto t = Tensor::from_host(mf.ptr, entry->shape, entry->dtype);
    device_cache_.emplace(name, std::move(t));
    return device_cache_.at(name);
}

bool WeightStore::contains(const std::string& name) const {
    return host_maps_.count(name) > 0;
}

void WeightStore::prefetch_all(cudaStream_t stream) const {
    for (const auto& entry : manifest_.layers) {
        if (device_cache_.find(entry.name) == device_cache_.end()) {
            upload(entry.name);
        }
    }
    if (stream) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

size_t WeightStore::device_bytes_allocated() const {
    size_t total = 0;
    for (const auto& [name, t] : device_cache_) {
        total += t.nbytes();
    }
    return total;
}

} // namespace denoiser
