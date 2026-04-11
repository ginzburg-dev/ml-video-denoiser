"""Video frame extraction utility.

Converts MP4 / MOV (or any container ffmpeg supports) to image sequences
by invoking the ``ffmpeg`` binary as a subprocess.  No library linking is
needed — ffmpeg handles all codec complexity.

Usage:
    # Extract all frames as PNG:
    uv run python video_extract.py \\
        --input /path/to/clip.mp4 \\
        --output /path/to/frames

    # Extract at a specific frame rate:
    uv run python video_extract.py \\
        --input /path/to/clip.mp4 \\
        --output /path/to/frames \\
        --fps 24

    # Extract to a temp directory (returns path):
    from video_extract import extract_frames
    frame_dir = extract_frames("clip.mp4", fps=None)
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Core extraction function
# ---------------------------------------------------------------------------


def extract_frames(
    video_path: str | Path,
    output_dir: Optional[str | Path] = None,
    fps: Optional[float] = None,
    quality: int = 1,
    frame_format: str = "png",
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> Path:
    """Extract frames from a video file using the ffmpeg binary.

    Args:
        video_path: Path to the input video file.
        output_dir: Directory to write frames into.  If None, a temporary
            directory is created (caller is responsible for cleanup).
        fps: Output frame rate.  If None, extracts every frame at the
            video's native frame rate.
        quality: JPEG/PNG quality parameter passed to ffmpeg (1 = best).
        frame_format: Output image format: ``"png"`` or ``"jpg"``.
        start_time: Seek to this position in seconds before extracting.
        duration: Maximum duration to extract in seconds.

    Returns:
        Path to the directory containing the extracted frames.

    Raises:
        FileNotFoundError: If *video_path* does not exist.
        RuntimeError: If the ffmpeg binary is not found on PATH or ffmpeg
            returns a non-zero exit code.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    _require_ffmpeg()

    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp(prefix="denoiser_frames_"))
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    pattern = str(output_dir / f"%06d.{frame_format}")

    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y"]

    if start_time is not None:
        cmd += ["-ss", str(start_time)]

    cmd += ["-i", str(video_path)]

    if duration is not None:
        cmd += ["-t", str(duration)]

    if fps is not None:
        cmd += ["-vf", f"fps={fps}"]

    if frame_format == "jpg":
        cmd += ["-q:v", str(quality)]
    else:
        cmd += ["-compression_level", "0"]  # fastest PNG, biggest files

    cmd.append(pattern)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    frames = sorted(output_dir.glob(f"*.{frame_format}"))
    print(
        f"Extracted {len(frames)} frames from {video_path.name} → {output_dir}",
        file=sys.stderr,
    )
    return output_dir


def probe_video(video_path: str | Path) -> dict:
    """Return basic metadata for *video_path* using ffprobe.

    Args:
        video_path: Path to the video file.

    Returns:
        Dict with keys: ``width``, ``height``, ``fps``, ``duration``,
        ``frame_count`` (may be None if unavailable), ``codec``.

    Raises:
        RuntimeError: If ffprobe is not available or fails.
    """
    import json as _json

    _require_ffmpeg(binary="ffprobe")
    cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-select_streams", "v:0",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = _json.loads(result.stdout)
    stream = data["streams"][0] if data.get("streams") else {}

    fps_str = stream.get("avg_frame_rate", "0/1")
    num, den = (int(x) for x in fps_str.split("/"))
    fps = num / den if den else 0.0

    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "fps": fps,
        "duration": float(stream.get("duration", 0)),
        "frame_count": stream.get("nb_frames"),
        "codec": stream.get("codec_name"),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require_ffmpeg(binary: str = "ffmpeg") -> None:
    """Raise RuntimeError if *binary* is not on PATH."""
    if shutil.which(binary) is None:
        raise RuntimeError(
            f"'{binary}' not found on PATH.\n"
            "Install ffmpeg: https://ffmpeg.org/download.html\n"
            "  macOS:   brew install ffmpeg\n"
            "  Ubuntu:  sudo apt-get install ffmpeg\n"
            "  Windows: https://www.gyan.dev/ffmpeg/builds/"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract video frames for denoiser training.")
    parser.add_argument("--input", required=True, metavar="PATH", help="Input video file.")
    parser.add_argument("--output", required=True, metavar="DIR", help="Output frame directory.")
    parser.add_argument("--fps", type=float, default=None,
                        help="Output FPS (default: native video FPS).")
    parser.add_argument("--format", choices=["png", "jpg"], default="png",
                        dest="frame_format", help="Output image format.")
    parser.add_argument("--quality", type=int, default=1,
                        help="Quality level (1 = best, for jpg only).")
    parser.add_argument("--start", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--duration", type=float, default=None, metavar="SECONDS")
    parser.add_argument("--probe", action="store_true",
                        help="Just print video metadata and exit.")
    args = parser.parse_args()

    if args.probe:
        info = probe_video(args.input)
        for k, v in info.items():
            print(f"  {k}: {v}")
        return

    extract_frames(
        video_path=args.input,
        output_dir=args.output,
        fps=args.fps,
        quality=args.quality,
        frame_format=args.frame_format,
        start_time=args.start,
        duration=args.duration,
    )


if __name__ == "__main__":
    main()
