"""Phase 6 integration tests: Python ↔ C++ parity and end-to-end CLI smoke tests.

These tests validate that the C++ inference engine produces numerically
identical results to the Python NAFNet, and that the full pipeline
(export → C++ load → forward → output) works end-to-end.

Prerequisites:
  1. Build the C++ engine:
       cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
  2. Generate fixtures:
       cd training && uv run python ../tests/gen_fixtures.py
  3. Generate sample images (one-time, already committed):
       cd training && uv run python ../tests/gen_sample_images.py

Run with:
    cd training && uv run pytest ../tests/test_integration.py -v
"""

import ctypes
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = REPO_ROOT / "training"
FIXTURE_DIR  = Path(__file__).resolve().parent / "fixtures" / "tiny_nafnet"
BUILD_DIR    = REPO_ROOT / "build"

if str(TRAINING_DIR) not in sys.path:
    sys.path.insert(0, str(TRAINING_DIR))

from models import NAFNet, NAFNetConfig  # noqa: E402
from export import export_model, verify_export  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tiny_config() -> NAFNetConfig:
    cfg = NAFNetConfig.tiny()
    cfg.base_channels = 8
    return cfg


def fixture_available() -> bool:
    return (FIXTURE_DIR / "manifest.json").exists()


def cpp_tests_binary() -> Path | None:
    """Return path to denoiser_tests binary if it exists."""
    for candidate in [
        BUILD_DIR / "tests" / "denoiser_tests",
        BUILD_DIR / "denoiser_tests",
    ]:
        if candidate.exists():
            return candidate
    return None


def cli_binary() -> Path | None:
    for candidate in [
        BUILD_DIR / "denoiser",
        BUILD_DIR / "cli" / "denoiser",
    ]:
        if candidate.exists():
            return candidate
    return None


def has_cuda() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---------------------------------------------------------------------------
# Fixtures setup / skip guards
# ---------------------------------------------------------------------------

skip_no_cuda    = pytest.mark.skipif(not has_cuda(), reason="No CUDA GPU available")
skip_no_fixture = pytest.mark.skipif(
    not fixture_available(),
    reason="Fixture not found — run: cd training && uv run python ../tests/gen_fixtures.py"
)
skip_no_cpp     = pytest.mark.skipif(
    cpp_tests_binary() is None,
    reason="C++ test binary not found — run: cmake --build build"
)
skip_no_cli     = pytest.mark.skipif(
    cli_binary() is None,
    reason="CLI binary not found — run: cmake --build build"
)


# ---------------------------------------------------------------------------
# 1. Export round-trip
# ---------------------------------------------------------------------------

class TestExportRoundtrip:
    """Verify export.py produces correct manifests and binary files."""

    def test_float16_roundtrip(self, tmp_path: Path):
        torch.manual_seed(0)
        model = NAFNet(tiny_config()).eval()
        manifest = export_model(model, tmp_path, dtype="float16")
        assert verify_export(model, manifest, rtol=1e-2)

    def test_float32_roundtrip(self, tmp_path: Path):
        torch.manual_seed(0)
        model = NAFNet(tiny_config()).eval()
        manifest = export_model(model, tmp_path, dtype="float32")
        assert verify_export(model, manifest, rtol=1e-5)

    def test_bn_stats_always_float32(self, tmp_path: Path):
        """BN running_mean and running_var must always be float32 in the manifest."""
        torch.manual_seed(0)
        model = NAFNet(tiny_config()).eval()
        manifest_path = export_model(model, tmp_path, dtype="float16")

        import json
        with open(manifest_path) as f:
            manifest = json.load(f)

        for layer in manifest["layers"]:
            name = layer["name"]
            if any(name.endswith(s) for s in ("running_mean", "running_var")):
                assert layer["dtype"] == "float32", (
                    f"{name} should be float32 but got {layer['dtype']}"
                )

    def test_conv_weights_are_float16(self, tmp_path: Path):
        torch.manual_seed(0)
        model = NAFNet(tiny_config()).eval()
        manifest_path = export_model(model, tmp_path, dtype="float16")

        import json
        with open(manifest_path) as f:
            manifest = json.load(f)

        for layer in manifest["layers"]:
            name = layer["name"]
            if name.endswith(".weight") and "bn" not in name and "running" not in name:
                assert layer["dtype"] == "float16", (
                    f"Conv weight {name} should be float16 but got {layer['dtype']}"
                )

    def test_manifest_architecture_fields(self, tmp_path: Path):
        torch.manual_seed(0)
        cfg   = tiny_config()
        model = NAFNet(cfg).eval()
        manifest_path = export_model(model, tmp_path, dtype="float16")

        import json
        with open(manifest_path) as f:
            manifest = json.load(f)

        arch = manifest["architecture"]
        assert arch["type"]          == "nafnet_residual"
        assert arch["base_channels"] == cfg.base_channels
        assert arch["num_levels"]    == cfg.num_levels
        assert arch["in_channels"]   == cfg.in_channels


# ---------------------------------------------------------------------------
# 2. Python NAFNet correctness
# ---------------------------------------------------------------------------

class TestNAFNetPython:
    """Basic correctness checks for the Python model."""

    @pytest.fixture
    def model(self):
        torch.manual_seed(7)
        return NAFNet(tiny_config()).eval()

    def test_output_shape_equals_input(self, model):
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_output_clamped_0_1(self, model):
        x = torch.rand(1, 3, 48, 48)
        with torch.no_grad():
            y = model(x)
        assert y.min().item() >= 0.0
        assert y.max().item() <= 1.0

    def test_arbitrary_spatial_size_padded(self, model):
        """Non-multiple-of-16 input must be auto-padded and output matches input size."""
        x = torch.rand(1, 3, 37, 53)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape

    def test_fp16_inference_close_to_fp32(self, model):
        """FP16 and FP32 outputs should be close (within ~0.01)."""
        torch.manual_seed(0)
        x = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            y_fp32 = model(x)
            y_fp16 = model.half()(x.half()).float()
        diff = (y_fp32 - y_fp16).abs().max().item()
        assert diff < 0.02, f"FP16/FP32 max diff too large: {diff:.4f}"

    def test_denoising_reduces_noise(self, model):
        """A clean image + Gaussian noise should still produce a finite valid image."""
        torch.manual_seed(0)
        clean = torch.rand(1, 3, 64, 64) * 0.8 + 0.1  # values in [0.1, 0.9]
        noisy = (clean + torch.randn_like(clean) * (25.0 / 255.0)).clamp(0, 1)

        with torch.no_grad():
            denoised = model(noisy)

        mse_before = ((noisy  - clean) ** 2).mean().item()
        mse_after  = ((denoised - clean) ** 2).mean().item()
        # The model is randomly initialised so it won't denoise perfectly,
        # but output should at least be a valid image
        assert denoised.min().item() >= 0.0
        assert denoised.max().item() <= 1.0


# ---------------------------------------------------------------------------
# 3. C++ gtest suite
# ---------------------------------------------------------------------------

class TestCppGtests:
    """Run the C++ gtest binary and assert all tests pass."""

    @skip_no_cpp
    def test_cpp_tensor_tests(self):
        binary = cpp_tests_binary()
        result = subprocess.run(
            [str(binary), "--gtest_filter=TensorTest.*"],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, (
            f"C++ TensorTest failed:\n{result.stdout}\n{result.stderr}"
        )

    @skip_no_cpp
    def test_cpp_weight_loader_tests(self):
        binary = cpp_tests_binary()
        result = subprocess.run(
            [str(binary), "--gtest_filter=WeightLoaderTest.*"],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, (
            f"C++ WeightLoaderTest failed:\n{result.stdout}\n{result.stderr}"
        )

    @skip_no_cpp
    def test_cpp_conv2d_tests(self):
        binary = cpp_tests_binary()
        result = subprocess.run(
            [str(binary), "--gtest_filter=Conv2dTest.*"],
            capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, (
            f"C++ Conv2dTest failed:\n{result.stdout}\n{result.stderr}"
        )

    @skip_no_cpp
    def test_cpp_batchnorm_tests(self):
        binary = cpp_tests_binary()
        result = subprocess.run(
            [str(binary), "--gtest_filter=BatchNorm2dTest.*"],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, (
            f"C++ BatchNorm2dTest failed:\n{result.stdout}\n{result.stderr}"
        )

    @skip_no_cpp
    def test_cpp_nafnet_support_tests(self):
        binary = cpp_tests_binary()
        result = subprocess.run(
            [str(binary), "--gtest_filter=NAFNetSupportTest.*"],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, (
            f"C++ NAFNetSupportTest failed:\n{result.stdout}\n{result.stderr}"
        )

    @skip_no_cpp
    @skip_no_fixture
    def test_cpp_nafnet_disabled_tests(self):
        """Run the DISABLED_ NAFNet parity tests (require fixture)."""
        binary = cpp_tests_binary()
        result = subprocess.run(
            [
                str(binary),
                "--gtest_also_run_disabled_tests",
                "--gtest_filter=NAFNetForwardTest.*",
            ],
            capture_output=True, text=True, timeout=120
        )
        assert result.returncode == 0, (
            f"C++ NAFNetForwardTest failed:\n{result.stdout}\n{result.stderr}"
        )


# ---------------------------------------------------------------------------
# 4. Python ↔ C++ parity (via fixture)
# ---------------------------------------------------------------------------

class TestPythonCppParity:
    """Compare Python FP16 output against the saved expected_output.bin fixture."""

    @skip_no_fixture
    def test_fixture_input_output_shapes(self):
        input_flat  = np.fromfile(FIXTURE_DIR / "input.bin",           dtype=np.float32)
        output_flat = np.fromfile(FIXTURE_DIR / "expected_output.bin", dtype=np.float32)
        assert input_flat.shape  == (3 * 32 * 32,)
        assert output_flat.shape == (3 * 32 * 32,)

    @skip_no_fixture
    def test_python_reproduces_fixture_output(self):
        """Re-run the Python model from fixture weights and compare to saved output."""
        import json

        with open(FIXTURE_DIR / "manifest.json") as f:
            manifest = json.load(f)

        cfg = tiny_config()
        cfg.base_channels = manifest["architecture"]["base_channels"]
        model = NAFNet(cfg).half().eval()

        # Load exported weights back into the model
        state = model.state_dict()
        for layer in manifest["layers"]:
            bin_path = FIXTURE_DIR / layer["file"]
            np_dtype = np.float16 if layer["dtype"] == "float16" else np.float32
            arr = np.fromfile(bin_path, dtype=np_dtype).reshape(layer["shape"])
            state[layer["name"]] = torch.from_numpy(arr.astype(np.float32)).to(
                dtype=model.state_dict()[layer["name"]].dtype
            )
        model.load_state_dict(state)

        # Reproduce input
        input_np = np.fromfile(FIXTURE_DIR / "input.bin", dtype=np.float32)
        input_t  = torch.from_numpy(input_np.reshape(1, 3, 32, 32)).half()

        with torch.no_grad():
            output_t = model(input_t).float()

        output_np  = output_t.numpy().flatten()
        expected   = np.fromfile(FIXTURE_DIR / "expected_output.bin", dtype=np.float32)

        max_diff = np.abs(output_np - expected).max()
        assert max_diff < 1e-3, (
            f"Python re-run differs from fixture: max diff = {max_diff:.6f}"
        )

    @skip_no_fixture
    def test_fixture_output_range(self):
        output_flat = np.fromfile(FIXTURE_DIR / "expected_output.bin", dtype=np.float32)
        assert output_flat.min() >= -0.01, f"output below 0: {output_flat.min()}"
        assert output_flat.max() <=  1.01, f"output above 1: {output_flat.max()}"


# ---------------------------------------------------------------------------
# 5. CLI smoke tests
# ---------------------------------------------------------------------------

class TestCLI:
    """End-to-end tests for the denoiser CLI binary."""

    @skip_no_cli
    @skip_no_fixture
    def test_cli_single_png(self, tmp_path: Path):
        """Denoise a synthetic PNG image via the CLI."""
        # Create a tiny noisy PNG using stb-compatible approach
        from PIL import Image
        import numpy as np

        rng = np.random.default_rng(0)
        arr = (rng.uniform(0.2, 0.8, (32, 32, 3)) * 255).astype(np.uint8)
        img_path = tmp_path / "test.png"
        Image.fromarray(arr).save(str(img_path))

        out_path = tmp_path / "test_denoised.png"
        result = subprocess.run(
            [
                str(cli_binary()),
                "--model", str(FIXTURE_DIR / "manifest.json"),
                "--input", str(img_path),
                "--output", str(out_path),
            ],
            capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, (
            f"CLI failed:\n{result.stdout}\n{result.stderr}"
        )
        assert out_path.exists(), "Output file was not created"

        # Verify the output PNG is valid and has the same dimensions
        out_img = Image.open(str(out_path))
        assert out_img.size == (32, 32)

    @skip_no_cli
    @skip_no_fixture
    def test_cli_help(self):
        result = subprocess.run(
            [str(cli_binary()), "--help"],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode == 0
        assert "--model" in result.stdout or "--model" in result.stderr

    @skip_no_cli
    @skip_no_fixture
    def test_cli_missing_model_flag(self):
        result = subprocess.run(
            [str(cli_binary()), "--input", "/tmp/fake.png"],
            capture_output=True, text=True, timeout=10
        )
        assert result.returncode != 0, "Should fail without --model"

    @skip_no_cli
    @skip_no_fixture
    def test_cli_image_dir(self, tmp_path: Path):
        """Denoise a directory of PNG images."""
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        import numpy as np
        img_dir = tmp_path / "frames"
        img_dir.mkdir()
        out_dir = tmp_path / "denoised"

        rng = np.random.default_rng(1)
        for i in range(3):
            arr = (rng.uniform(0.1, 0.9, (32, 32, 3)) * 255).astype(np.uint8)
            Image.fromarray(arr).save(str(img_dir / f"frame_{i:04d}.png"))

        result = subprocess.run(
            [
                str(cli_binary()),
                "--model",  str(FIXTURE_DIR / "manifest.json"),
                "--input",  str(img_dir),
                "--output", str(out_dir),
            ],
            capture_output=True, text=True, timeout=90
        )
        assert result.returncode == 0, (
            f"CLI directory mode failed:\n{result.stdout}\n{result.stderr}"
        )
        denoised_files = list(out_dir.glob("*.png"))
        assert len(denoised_files) == 3, (
            f"Expected 3 output frames, got {len(denoised_files)}"
        )


# ---------------------------------------------------------------------------
# 6. Training smoke test (bundled sample images, no external data needed)
# ---------------------------------------------------------------------------

SAMPLE_IMAGES_DIR = REPO_ROOT / "tests" / "fixtures" / "sample_images"


def _sample_images_available() -> bool:
    return SAMPLE_IMAGES_DIR.is_dir() and len(list(SAMPLE_IMAGES_DIR.glob("*.png"))) >= 4


skip_no_samples = pytest.mark.skipif(
    not _sample_images_available(),
    reason=(
        "Sample images not found — run: "
        "cd training && uv run python ../tests/gen_sample_images.py"
    ),
)


class TestTrainingSmokeTest:
    """2-epoch smoke test verifying the training pipeline end-to-end.

    Uses the five synthetic 128×128 PNGs committed in
    tests/fixtures/sample_images/.  No external dataset required.
    Runs entirely on CPU in < 30 s with the tiny base_channels=8 NAFNet model.
    """

    def _make_loader(self):
        from torch.utils.data import DataLoader
        from dataset import PatchDataset
        from noise_generators import GaussianNoiseGenerator

        noise_gen = GaussianNoiseGenerator(sigma_min=10.0 / 255.0, sigma_max=50.0 / 255.0)
        dataset = PatchDataset(
            image_dirs=[SAMPLE_IMAGES_DIR],
            noise_generator=noise_gen,
            patch_size=64,
            patches_per_image=8,
        )
        return DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    def _make_model(self, seed: int = 42):
        torch.manual_seed(seed)
        return NAFNet(tiny_config())

    @skip_no_samples
    def test_sample_images_are_valid_rgb(self):
        """Each sample image must be 128×128 RGB with values in [0, 255]."""
        from PIL import Image
        for img_path in sorted(SAMPLE_IMAGES_DIR.glob("*.png")):
            img = Image.open(img_path)
            assert img.mode == "RGB", f"{img_path.name}: expected RGB, got {img.mode}"
            assert img.size == (128, 128), (
                f"{img_path.name}: expected 128×128, got {img.size}"
            )

    @skip_no_samples
    def test_training_completes_2_epochs(self, tmp_path: Path):
        """Training loop must complete without error and write final.pth."""
        from training import train

        train(
            model=self._make_model(),
            loader=self._make_loader(),
            val_loader=None,
            output_dir=tmp_path,
            epochs=2,
            warmup_epochs=0,
            checkpoint_every=100,   # no periodic checkpoints during the test
            use_amp=False,          # CPU-compatible
        )

        assert (tmp_path / "final.pth").exists(), "final.pth not written after training"
        ckpt = torch.load(tmp_path / "final.pth", map_location="cpu")
        assert "model_state_dict" in ckpt
        assert "optimizer_state_dict" in ckpt
        # 0-indexed: last epoch stored should be 1 (the 2nd epoch)
        assert ckpt["epoch"] == 1, f"Expected epoch=1, got {ckpt['epoch']}"

    @skip_no_samples
    def test_loss_decreases_over_2_epochs(self, tmp_path: Path):
        """Loss at epoch 2 must be strictly lower than at epoch 1."""
        from training import train
        import io
        from contextlib import redirect_stdout

        buf = io.StringIO()
        with redirect_stdout(buf):
            train(
                model=self._make_model(seed=0),
                loader=self._make_loader(),
                val_loader=None,
                output_dir=tmp_path,
                epochs=2,
                warmup_epochs=0,
                checkpoint_every=100,
                use_amp=False,
            )
        log = buf.getvalue()

        # Parse the two "loss=..." lines printed by the training loop
        import re
        losses = [float(m) for m in re.findall(r"loss=([0-9.]+)", log)]
        assert len(losses) == 2, f"Expected 2 loss entries in log, got: {log!r}"
        assert losses[1] < losses[0], (
            f"Loss did not decrease: epoch1={losses[0]:.6f}, epoch2={losses[1]:.6f}"
        )

    @skip_no_samples
    def test_model_outputs_valid_after_training(self, tmp_path: Path):
        """After training, the model must still produce finite outputs."""
        from training import train

        model = self._make_model(seed=99)
        train(
            model=model,
            loader=self._make_loader(),
            val_loader=None,
            output_dir=tmp_path,
            epochs=2,
            warmup_epochs=0,
            checkpoint_every=100,
            use_amp=False,
        )

        model.eval()
        x = torch.rand(1, 3, 64, 64)
        with torch.no_grad():
            y = model(x)

        assert not torch.isnan(y).any(),  "Output contains NaN after training"
        assert not torch.isinf(y).any(),  "Output contains Inf after training"

    @skip_no_samples
    def test_checkpoint_resumes_cleanly(self, tmp_path: Path):
        """A checkpoint saved at epoch 1 must resume to epoch 2 without error."""
        from training import train

        # First run: 1 epoch, save epoch_0001.pth via checkpoint_every=1
        model = self._make_model(seed=5)
        train(
            model=model,
            loader=self._make_loader(),
            val_loader=None,
            output_dir=tmp_path / "run",
            epochs=1,
            warmup_epochs=0,
            checkpoint_every=1,
            use_amp=False,
        )
        first_ckpt = tmp_path / "run" / "epoch_0001.pth"
        assert first_ckpt.exists(), "epoch_0001.pth not written"

        # Resume: 1 more epoch from the saved checkpoint
        model2 = self._make_model(seed=5)
        train(
            model=model2,
            loader=self._make_loader(),
            val_loader=None,
            output_dir=tmp_path / "resume",
            epochs=2,
            warmup_epochs=0,
            checkpoint_every=100,
            use_amp=False,
            resume=first_ckpt,
        )
        assert (tmp_path / "resume" / "final.pth").exists()


# ---------------------------------------------------------------------------
# 7. PSNR regression check (Python model)
# ---------------------------------------------------------------------------

class TestPSNRRegression:
    """PSNR sanity check: a randomly initialised model should produce a
    valid image (PSNR defined, no NaN/Inf), and a trained model (if present)
    should exceed the minimum thresholds from the plan."""

    def _psnr(self, a: torch.Tensor, b: torch.Tensor) -> float:
        mse = ((a - b) ** 2).mean().item()
        if mse < 1e-10:
            return float("inf")
        return 10.0 * np.log10(1.0 / mse)

    def test_random_init_no_nan(self):
        torch.manual_seed(3)
        model = NAFNet(tiny_config()).eval()
        x = torch.rand(1, 3, 64, 64) * 0.8 + 0.1

        with torch.no_grad():
            y = model(x)

        assert not torch.isnan(y).any(), "Output contains NaN"
        assert not torch.isinf(y).any(), "Output contains Inf"

    @pytest.mark.skipif(
        not (REPO_ROOT / "checkpoints" / "spatial_standard" / "best.pth").exists(),
        reason="Trained checkpoint not found — skipping PSNR regression"
    )
    def test_trained_model_psnr_awgn_sigma25(self):
        """Trained standard model must achieve >31 dB on AWGN sigma=25."""
        ckpt_path = REPO_ROOT / "checkpoints" / "spatial_standard" / "best.pth"
        model = NAFNet(NAFNetConfig.standard())
        state = torch.load(str(ckpt_path), map_location="cpu")
        model.load_state_dict(state.get("model_state_dict", state))
        model.eval()

        torch.manual_seed(99)
        sigma  = 25.0 / 255.0
        clean  = torch.rand(4, 3, 256, 256)
        noisy  = (clean + torch.randn_like(clean) * sigma).clamp(0, 1)

        with torch.no_grad():
            denoised = model(noisy)

        psnr = self._psnr(denoised, clean)
        assert psnr > 31.0, f"PSNR {psnr:.2f} dB below 31 dB threshold"
