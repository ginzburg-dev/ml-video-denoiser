#include "denoiser/io/video_io.h"

#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace denoiser::io {

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

namespace {

// Run a shell command and return stdout.
// Throws std::runtime_error if the command exits non-zero.
std::string run_command(const std::string& cmd) {
    std::string output;
    std::array<char, 4096> buf;

    FILE* pipe = popen((cmd + " 2>&1").c_str(), "r");
    if (!pipe) {
        throw std::runtime_error("video_io: popen failed for: " + cmd);
    }

    while (fgets(buf.data(), static_cast<int>(buf.size()), pipe)) {
        output += buf.data();
    }

    const int exit_code = pclose(pipe);
    if (exit_code != 0) {
        throw std::runtime_error(
            "video_io: command failed (exit " + std::to_string(exit_code) +
            "):\n  " + cmd + "\nOutput:\n" + output);
    }
    return output;
}

// Check that `ffmpeg` (or `ffprobe`) is on PATH.
void check_ffmpeg_available(const std::string& bin) {
#ifdef _WIN32
    const std::string check = "where " + bin + " >NUL 2>&1";
#else
    const std::string check = "command -v " + bin + " >/dev/null 2>&1";
#endif
    FILE* p = popen(check.c_str(), "r");
    const int rc = p ? pclose(p) : -1;
    if (rc != 0) {
        throw std::runtime_error(
            "video_io: '" + bin + "' not found on PATH. "
            "Install ffmpeg and ensure it is accessible.");
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// probe_video
// ---------------------------------------------------------------------------

VideoInfo probe_video(const std::string& path) {
    check_ffmpeg_available("ffprobe");

    // Use ffprobe to extract width, height, fps, and approximate frame count
    // in a machine-readable format.
    const std::string cmd =
        "ffprobe -v error -select_streams v:0 "
        "-show_entries stream=width,height,r_frame_rate,nb_frames "
        "-of default=noprint_wrappers=1 "
        "\"" + path + "\"";

    const std::string raw = run_command(cmd);

    VideoInfo info;
    std::istringstream ss(raw);
    std::string line;
    while (std::getline(ss, line)) {
        auto eq = line.find('=');
        if (eq == std::string::npos) continue;
        const std::string key = line.substr(0, eq);
        const std::string val = line.substr(eq + 1);

        if (key == "width")  info.width  = std::stoi(val);
        if (key == "height") info.height = std::stoi(val);
        if (key == "nb_frames" && val != "N/A") info.num_frames = std::stoi(val);

        if (key == "r_frame_rate") {
            // Format "num/den"
            const auto slash = val.find('/');
            if (slash != std::string::npos) {
                const double num = std::stod(val.substr(0, slash));
                const double den = std::stod(val.substr(slash + 1));
                if (den > 0.0) info.fps = num / den;
            }
        }
    }

    if (info.width <= 0 || info.height <= 0) {
        throw std::runtime_error(
            "probe_video: could not determine dimensions for '" + path + "'");
    }
    return info;
}

// ---------------------------------------------------------------------------
// extract_frames
// ---------------------------------------------------------------------------

std::vector<std::string> extract_frames(
    const std::string& video_path,
    const std::string& output_dir,
    double fps, double start_sec, double end_sec)
{
    check_ffmpeg_available("ffmpeg");

    fs::create_directories(output_dir);

    const std::string frame_pattern =
        (fs::path(output_dir) / "frame_%06d.png").string();

    std::ostringstream cmd;
    cmd << "ffmpeg -y";
    if (start_sec > 0.0) cmd << " -ss " << start_sec;
    cmd << " -i \"" << video_path << "\"";
    if (end_sec > 0.0 && end_sec > start_sec)
        cmd << " -t " << (end_sec - start_sec);
    if (fps > 0.0) cmd << " -vf fps=" << fps;
    cmd << " -pix_fmt rgb24 \"" << frame_pattern << "\"";
    cmd << " -loglevel error";

    run_command(cmd.str());

    // Glob the extracted frames in sorted order
    std::vector<std::string> paths;
    for (const auto& entry : fs::directory_iterator(output_dir)) {
        if (entry.path().extension() == ".png") {
            paths.push_back(entry.path().string());
        }
    }
    std::sort(paths.begin(), paths.end());

    if (paths.empty()) {
        throw std::runtime_error(
            "extract_frames: no frames extracted from '" + video_path + "'");
    }
    return paths;
}

// ---------------------------------------------------------------------------
// encode_frames
// ---------------------------------------------------------------------------

void encode_frames(
    const std::vector<std::string>& frame_paths,
    const std::string& output_path,
    double fps)
{
    if (frame_paths.empty()) {
        throw std::runtime_error("encode_frames: frame_paths is empty");
    }
    check_ffmpeg_available("ffmpeg");

    // Write a temporary file list for ffmpeg concat demuxer
    const fs::path list_file =
        fs::path(output_path).parent_path() / ".denoiser_frame_list.txt";
    {
        std::ofstream f(list_file);
        if (!f.is_open()) {
            throw std::runtime_error(
                "encode_frames: cannot write frame list to " + list_file.string());
        }
        for (const auto& p : frame_paths) {
            f << "file '" << p << "'\n";
            f << "duration " << (1.0 / fps) << "\n";
        }
    }

    std::ostringstream cmd;
    cmd << "ffmpeg -y"
        << " -f concat -safe 0 -i \"" << list_file.string() << "\""
        << " -r " << fps
        << " -c:v libx264 -preset fast -crf 18 -pix_fmt yuv420p"
        << " \"" << output_path << "\""
        << " -loglevel error";

    run_command(cmd.str());
    fs::remove(list_file);
}

} // namespace denoiser::io
