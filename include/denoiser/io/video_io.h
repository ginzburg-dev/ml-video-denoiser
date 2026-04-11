#pragma once

#include "denoiser/tensor.h"
#include <string>
#include <vector>

namespace denoiser::io {

// ---------------------------------------------------------------------------
// Video I/O — extracts / encodes frames by shelling out to the `ffmpeg` binary
// ---------------------------------------------------------------------------
// No libavcodec link — the binary invocation carries no LGPL obligation.
// Requires `ffmpeg` (and `ffprobe`) on PATH at runtime.

struct VideoInfo {
    int    width       = 0;
    int    height      = 0;
    double fps         = 25.0;
    int    num_frames  = 0;   // -1 if unknown
};

// Probe a video file and return basic metadata.
// Throws std::runtime_error if ffprobe fails or the file is not a valid video.
VideoInfo probe_video(const std::string& path);

// Extract all frames from *video_path* to *output_dir* (PNG files).
//
//   output_dir:  directory that will receive frame_000001.png, frame_000002.png …
//                Created if it does not exist.
//   fps:         target frame rate (0 = keep source fps).
//   start_sec:   start time in seconds (0 = beginning).
//   end_sec:     end time in seconds (0 = until end of file).
//
// Returns the list of extracted PNG paths in order.
// Throws std::runtime_error if ffmpeg fails.
std::vector<std::string> extract_frames(
    const std::string& video_path,
    const std::string& output_dir,
    double fps       = 0.0,
    double start_sec = 0.0,
    double end_sec   = 0.0);

// Encode a sequence of image files into a video.
//
//   frame_paths: ordered list of PNG/JPEG frame files.
//   output_path: output video file (.mp4 recommended).
//   fps:         frame rate for the output video.
//
// Throws std::runtime_error if ffmpeg fails.
void encode_frames(
    const std::vector<std::string>& frame_paths,
    const std::string& output_path,
    double fps = 25.0);

} // namespace denoiser::io
