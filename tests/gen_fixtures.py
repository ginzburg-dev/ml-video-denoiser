"""Generate test fixtures for C++ gtest integration tests.

Produces:
    tests/fixtures/tiny_unet/
        manifest.json            — weight store manifest
        weights/*.bin            — exported weight binaries
        input.bin                — FP32 flat array (3 × 32 × 32)
        expected_output.bin      — FP32 flat array (3 × 32 × 32), Python FP16 output

The tiny model uses enc_channels=[8, 16] (2 levels) so the fixture is small
(~50 KB total) and fast to generate / load in tests.

The C++ parity test loads both files, runs the same input through NEFResidual,
and checks that max pixel diff ≤ 0.005.

Usage (from the repo root):
    cd training && uv run python ../tests/gen_fixtures.py
  or from any directory with training/ on the Python path:
    uv run python tests/gen_fixtures.py
"""

import argparse
import sys
import struct
from pathlib import Path

import numpy as np
import torch

# Make sure the training package is importable regardless of cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "training"
if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from models import ModelConfig, NEFResidual  # noqa: E402
from export import export_model              # noqa: E402


# ---------------------------------------------------------------------------
# Tiny model config (not exposed in ModelConfig — defined here only)
# ---------------------------------------------------------------------------

def tiny_config() -> ModelConfig:
    """2-level UNet, very small channels — fast fixture for unit tests."""
    return ModelConfig(
        enc_channels=[8, 16],
        in_channels=3,
        out_channels=3,
    )


# ---------------------------------------------------------------------------
# Fixture generation
# ---------------------------------------------------------------------------

def generate(fixture_dir: Path, seed: int = 42) -> None:
    """Build tiny_unet fixture under *fixture_dir*."""
    fixture_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")  # fixtures are device-agnostic
    cfg    = tiny_config()
    model  = NEFResidual(cfg).to(device).eval()

    # --- Export weights ---
    export_model(model, fixture_dir, dtype="float16")
    print(f"Weights exported → {fixture_dir}/manifest.json")

    # --- Build deterministic input (FP32 in [0, 1]) ---
    # Use a fixed seed so the C++ and Python run against identical data.
    rng = np.random.default_rng(seed)
    input_np = rng.uniform(0.3, 0.7, size=(1, 3, 32, 32)).astype(np.float32)

    # --- Run Python forward in FP16 to match C++ precision ---
    model_fp16  = model.half()
    input_torch = torch.from_numpy(input_np).half()

    with torch.no_grad():
        output_torch = model_fp16(input_torch)  # (1, 3, 32, 32) FP16

    # Convert back to FP32 for storage
    output_np = output_torch.float().numpy()  # (1, 3, 32, 32)

    # --- Save input.bin and expected_output.bin ---
    # Both stored as flat FP32 arrays in NCHW order (batch dim stripped).
    input_flat  = input_np[0].astype(np.float32).flatten()
    output_flat = output_np[0].astype(np.float32).flatten()

    input_path  = fixture_dir / "input.bin"
    output_path = fixture_dir / "expected_output.bin"

    input_flat.tofile(input_path)
    output_flat.tofile(output_path)

    print(f"input.bin          → {input_path}  ({input_flat.nbytes} bytes)")
    print(f"expected_output.bin→ {output_path}  ({output_flat.nbytes} bytes)")

    # --- Sanity checks ---
    assert input_flat.shape  == (3 * 32 * 32,), f"bad input shape: {input_flat.shape}"
    assert output_flat.shape == (3 * 32 * 32,), f"bad output shape: {output_flat.shape}"
    assert output_flat.min() >= -0.01, f"output below 0: {output_flat.min()}"
    assert output_flat.max() <=  1.01, f"output above 1: {output_flat.max()}"

    # Verify round-trip of weights (FP16 ± 1e-2)
    from export import verify_export
    ok = verify_export(model, fixture_dir / "manifest.json", rtol=1e-2)
    if not ok:
        raise RuntimeError("Weight export round-trip verification failed!")

    print(f"\nFixture ready in: {fixture_dir}")
    print(f"  Model:  enc_channels={cfg.enc_channels}, num_levels={cfg.num_levels}")
    print(f"  Input:  shape=(1, 3, 32, 32), range=[{input_np.min():.3f}, {input_np.max():.3f}]")
    print(f"  Output: range=[{output_flat.min():.3f}, {output_flat.max():.3f}]")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: tests/fixtures/tiny_unet next to this script)"
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "fixtures" / "tiny_unet"
    )
    generate(out_dir, seed=args.seed)


if __name__ == "__main__":
    main()
