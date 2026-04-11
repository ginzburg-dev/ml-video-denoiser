#include "denoiser/models/nef_residual.h"
#include "denoiser/models/nef_temporal.h"
#include "denoiser/weight_loader.h"
#include "denoiser/tensor.h"
#include "denoiser/io/image_io.h"
#include "denoiser/io/exr_io.h"
#include "denoiser/io/video_io.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using namespace denoiser;
using namespace denoiser::io;

// ---------------------------------------------------------------------------
// Argument struct
// ---------------------------------------------------------------------------

struct Args {
    std::string model_path;
    std::string input_path;
    std::string output_path;
    std::string mode = "spatial";   // spatial | temporal
    int  device      = 0;
    int  num_frames  = 5;           // temporal window
    bool prefetch    = false;       // pre-upload all weights
};

// ---------------------------------------------------------------------------
// Usage / help
// ---------------------------------------------------------------------------

static void print_usage(const char* prog) {
    std::cerr <<
        "Usage: " << prog << " --model <manifest.json> --input <path> [options]\n"
        "\n"
        "Required:\n"
        "  --model  PATH   manifest.json from export.py\n"
        "  --input  PATH   image (PNG/JPG/EXR), image directory, or video (MP4/MOV)\n"
        "\n"
        "Options:\n"
        "  --output PATH   output path (default: <input>_denoised.<ext>)\n"
        "  --mode   MODE   spatial | temporal  (default: spatial)\n"
        "  --frames N      temporal window size (default: 5)\n"
        "  --device N      CUDA device index   (default: 0)\n"
        "  --prefetch      upload all weights before first inference\n"
        "  --help\n";
}

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

static Args parse_args(int argc, char** argv) {
    Args args;
    if (argc < 2) {
        print_usage(argv[0]);
        std::exit(1);
    }

    auto next_arg = [&](int& i) -> std::string {
        if (++i >= argc) throw std::runtime_error("missing value for " + std::string(argv[i - 1]));
        return argv[i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string_view flag = argv[i];
        if (flag == "--help" || flag == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else if (flag == "--model")   args.model_path  = next_arg(i);
        else if (flag == "--input")    args.input_path  = next_arg(i);
        else if (flag == "--output")   args.output_path = next_arg(i);
        else if (flag == "--mode")     args.mode        = next_arg(i);
        else if (flag == "--frames")   args.num_frames  = std::stoi(next_arg(i));
        else if (flag == "--device")   args.device      = std::stoi(next_arg(i));
        else if (flag == "--prefetch") args.prefetch     = true;
        else {
            throw std::runtime_error("unknown flag: " + std::string(flag));
        }
    }

    if (args.model_path.empty()) throw std::runtime_error("--model is required");
    if (args.input_path.empty()) throw std::runtime_error("--input is required");

    return args;
}

// ---------------------------------------------------------------------------
// Format detection helpers
// ---------------------------------------------------------------------------

static bool is_exr(const fs::path& p) {
    const auto ext = p.extension().string();
    return ext == ".exr" || ext == ".EXR";
}

static bool is_video(const fs::path& p) {
    const auto ext = p.extension().string();
    return ext == ".mp4" || ext == ".MP4" || ext == ".mov" || ext == ".MOV" ||
           ext == ".avi" || ext == ".AVI" || ext == ".mkv" || ext == ".MKV";
}

static bool is_image(const fs::path& p) {
    const auto ext = p.extension().string();
    return ext == ".png" || ext == ".PNG" ||
           ext == ".jpg" || ext == ".jpg" ||
           ext == ".jpeg" || ext == ".JPEG" ||
           ext == ".exr" || ext == ".EXR";
}

static Tensor load_any_image(const fs::path& p, cudaStream_t stream) {
    if (is_exr(p)) return load_exr(p.string(), stream);
    return load_image(p.string(), stream);
}

static void save_any_image(const Tensor& t, const fs::path& p, cudaStream_t stream) {
    if (is_exr(p)) save_exr(t, p.string(), stream);
    else            save_image(t, p.string(), stream);
}

// Default output path: append "_denoised" before the extension.
static fs::path default_output(const fs::path& input) {
    return input.parent_path() /
           (input.stem().string() + "_denoised" + input.extension().string());
}

// ---------------------------------------------------------------------------
// Inference wrappers
// ---------------------------------------------------------------------------

// Spatial (single-frame) denoising of one image.
static Tensor denoise_spatial(const NEFResidual& model,
                               const Tensor& input,
                               cudaStream_t stream) {
    return model.forward(input, stream);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        print_usage(argv[0]);
        return 1;
    }

    try {
        // --- CUDA device setup ---
        int device_count = 0;
        CUDA_CHECK(cudaGetDeviceCount(&device_count));
        if (args.device >= device_count) {
            throw std::runtime_error("CUDA device " + std::to_string(args.device) +
                                     " not available (" +
                                     std::to_string(device_count) + " devices found)");
        }
        CUDA_CHECK(cudaSetDevice(args.device));

        // Use two streams: one for H2D transfers, one for compute.
        cudaStream_t compute_stream = nullptr;
        cudaStream_t transfer_stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&compute_stream));
        CUDA_CHECK(cudaStreamCreate(&transfer_stream));

        // --- Load weights ---
        std::cout << "Loading model: " << args.model_path << "\n";
        WeightStore store(args.model_path);
        if (args.prefetch) store.prefetch_all(transfer_stream);

        // --- Dispatch on mode ---
        const fs::path input_path(args.input_path);

        if (args.mode == "temporal") {
            // --- Temporal mode (not yet fully wired — placeholder) ---
            throw std::runtime_error(
                "Temporal mode requires a 5-frame clip input. "
                "Pass a directory containing exactly num_frames image files.");
        }

        // --- Spatial mode ---
        NEFResidual model(store);
        std::cout << "Model: NEFResidual, enc_channels=[";
        for (size_t i = 0; i < model.enc_channels().size(); ++i) {
            if (i) std::cout << ",";
            std::cout << model.enc_channels()[i];
        }
        std::cout << "]\n";

        if (is_video(input_path)) {
            // --- Video: extract → denoise frame-by-frame → re-encode ---
            const VideoInfo info = probe_video(input_path.string());
            std::cout << "Video: " << info.width << "×" << info.height
                      << " @ " << info.fps << " fps\n";

            const fs::path tmp_in  = fs::temp_directory_path() / "denoiser_in_frames";
            const fs::path tmp_out = fs::temp_directory_path() / "denoiser_out_frames";
            fs::create_directories(tmp_out);

            std::cout << "Extracting frames...\n";
            auto frame_paths = extract_frames(input_path.string(), tmp_in.string(),
                                              /*fps=*/0.0);
            std::cout << "Denoising " << frame_paths.size() << " frames...\n";

            std::vector<std::string> out_paths;
            out_paths.reserve(frame_paths.size());

            for (size_t idx = 0; idx < frame_paths.size(); ++idx) {
                auto in_t  = load_any_image(frame_paths[idx], transfer_stream);
                CUDA_CHECK(cudaStreamSynchronize(transfer_stream));

                auto out_t = denoise_spatial(model, in_t, compute_stream);
                CUDA_CHECK(cudaStreamSynchronize(compute_stream));

                const fs::path out_frame =
                    tmp_out / fs::path(frame_paths[idx]).filename();
                save_any_image(out_t, out_frame, nullptr);
                out_paths.push_back(out_frame.string());

                if ((idx + 1) % 50 == 0 || idx + 1 == frame_paths.size()) {
                    std::cout << "  " << (idx + 1) << "/" << frame_paths.size()
                              << " frames done\n";
                }
            }

            const fs::path out_path = args.output_path.empty()
                ? default_output(input_path)
                : fs::path(args.output_path);
            std::cout << "Encoding → " << out_path << "\n";
            encode_frames(out_paths, out_path.string(), info.fps);

            fs::remove_all(tmp_in);
            fs::remove_all(tmp_out);

        } else if (fs::is_directory(input_path)) {
            // --- Directory: denoise each image file ---
            const fs::path out_dir = args.output_path.empty()
                ? input_path.parent_path() / (input_path.filename().string() + "_denoised")
                : fs::path(args.output_path);
            fs::create_directories(out_dir);

            std::vector<fs::path> images;
            for (const auto& entry : fs::directory_iterator(input_path)) {
                if (is_image(entry.path())) images.push_back(entry.path());
            }
            std::sort(images.begin(), images.end());

            std::cout << "Processing " << images.size() << " images → " << out_dir << "\n";
            for (const auto& img_path : images) {
                auto in_t  = load_any_image(img_path, transfer_stream);
                CUDA_CHECK(cudaStreamSynchronize(transfer_stream));
                auto out_t = denoise_spatial(model, in_t, compute_stream);
                CUDA_CHECK(cudaStreamSynchronize(compute_stream));
                save_any_image(out_t, out_dir / img_path.filename(), nullptr);
                std::cout << "  " << img_path.filename() << "\n";
            }

        } else {
            // --- Single image ---
            auto in_t  = load_any_image(input_path, transfer_stream);
            CUDA_CHECK(cudaStreamSynchronize(transfer_stream));

            auto out_t = denoise_spatial(model, in_t, compute_stream);
            CUDA_CHECK(cudaStreamSynchronize(compute_stream));

            const fs::path out_path = args.output_path.empty()
                ? default_output(input_path)
                : fs::path(args.output_path);
            save_any_image(out_t, out_path, nullptr);
            std::cout << "Saved: " << out_path << "\n";
        }

        CUDA_CHECK(cudaStreamDestroy(compute_stream));
        CUDA_CHECK(cudaStreamDestroy(transfer_stream));
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        return 1;
    }
}
