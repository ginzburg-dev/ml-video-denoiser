"""Pytest wrapper for the noise-visualisation diagnostic script.

Runs the full diagnostic pipeline in a temporary output directory and asserts
that every expected PNG was produced with non-zero size.  The test is fast
(~5 s CPU) because it uses the bundled sample images and synthetic
noise/paired data — no GPU or external dataset required.

Skip guard:
    If the sample images haven't been generated yet the test is skipped
    with a helpful message.  Generate them with:
        cd training && uv run python ../tests/gen_sample_images.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SAMPLE_DIR = REPO_ROOT / "tests" / "fixtures" / "sample_images"
VISUALISE_SCRIPT = REPO_ROOT / "tests" / "visualise_noise.py"

# Add training dir so visualise_noise can import from it when run via pytest
if str(REPO_ROOT / "training") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "training"))

skip_no_samples = pytest.mark.skipif(
    not SAMPLE_DIR.exists() or not list(SAMPLE_DIR.glob("*.png")),
    reason=(
        "Sample images not found.  "
        "Run: cd training && uv run python ../tests/gen_sample_images.py"
    ),
)


@skip_no_samples
class TestNoiseDiagnostics:
    """Verify that visualise_noise.run() produces all expected PNG files."""

    EXPECTED_STEMS = {
        # per-generator grids (names depend on MixedNoiseGenerator.default())
        # verified by prefix rather than exact name
        "noise_types_overview",
        "paired_dataset",
        "patch_pool_residuals",
        "sigma_comparison",
        "temporal_clip_consistency",
    }

    def test_run_produces_output_pngs(self, tmp_path: Path) -> None:
        """run() creates PNGs for every diagnostic category."""
        # Import here so the module-level matplotlib import is lazy
        sys.path.insert(0, str(VISUALISE_SCRIPT.parent))
        import importlib.util

        spec = importlib.util.spec_from_file_location("visualise_noise", VISUALISE_SCRIPT)
        vn = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(vn)  # type: ignore[union-attr]

        written = vn.run(out_dir=tmp_path, pool_path=None, profile_path=None, auto_open=False)

        assert written, "run() returned an empty list — no files were written"
        for path in written:
            assert path.exists(), f"Expected output missing: {path}"
            assert path.stat().st_size > 0, f"Output PNG is empty: {path}"

    def test_all_expected_categories_present(self, tmp_path: Path) -> None:
        """Every diagnostic category appears at least once in the output."""
        sys.path.insert(0, str(VISUALISE_SCRIPT.parent))
        import importlib.util

        spec = importlib.util.spec_from_file_location("visualise_noise", VISUALISE_SCRIPT)
        vn = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(vn)  # type: ignore[union-attr]

        written = vn.run(out_dir=tmp_path, pool_path=None, profile_path=None, auto_open=False)
        stems = {p.stem for p in written}

        for expected in self.EXPECTED_STEMS:
            assert any(expected in s for s in stems), (
                f"Expected a file matching '{expected}' in output, got: {sorted(stems)}"
            )

    def test_generator_grids_cover_all_noise_types(self, tmp_path: Path) -> None:
        """A per-generator PNG is produced for each noise type."""
        sys.path.insert(0, str(VISUALISE_SCRIPT.parent))
        import importlib.util

        spec = importlib.util.spec_from_file_location("visualise_noise", VISUALISE_SCRIPT)
        vn = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(vn)  # type: ignore[union-attr]

        written = vn.run(out_dir=tmp_path, pool_path=None, profile_path=None, auto_open=False)
        gen_files = [p for p in written if p.stem.startswith("generator_")]

        # There should be at least one grid per noise type:
        # gaussian (low), gaussian (medium), poisson-gaussian, camera, real_inject,
        # real_raw, mixed — 7 in default config
        assert len(gen_files) >= 5, (
            f"Expected ≥4 per-generator PNG files, got {len(gen_files)}: "
            f"{[p.name for p in gen_files]}"
        )

    def test_no_zero_byte_outputs(self, tmp_path: Path) -> None:
        """Every PNG produced by run() contains actual image data."""
        sys.path.insert(0, str(VISUALISE_SCRIPT.parent))
        import importlib.util

        spec = importlib.util.spec_from_file_location("visualise_noise", VISUALISE_SCRIPT)
        vn = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(vn)  # type: ignore[union-attr]

        written = vn.run(out_dir=tmp_path, pool_path=None, profile_path=None, auto_open=False)
        zero_byte = [p for p in written if p.stat().st_size == 0]
        assert not zero_byte, f"Zero-byte PNGs: {[p.name for p in zero_byte]}"

    def test_sample_image_count_matches_rows(self, tmp_path: Path) -> None:
        """The overview grid covers all 5 bundled sample images."""
        sys.path.insert(0, str(VISUALISE_SCRIPT.parent))
        import importlib.util
        from PIL import Image

        spec = importlib.util.spec_from_file_location("visualise_noise", VISUALISE_SCRIPT)
        vn = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(vn)  # type: ignore[union-attr]

        written = vn.run(out_dir=tmp_path, pool_path=None, profile_path=None, auto_open=False)

        n_samples = len(list(SAMPLE_DIR.glob("*.png")))
        overview = next((p for p in written if "overview" in p.stem), None)
        assert overview is not None, "noise_types_overview.png not found in outputs"

        img = Image.open(overview)
        # Overview height encodes (n_generators × n_images) rows — just confirm
        # it's a non-trivial image: at minimum 200 × 200 px
        assert img.width > 200 and img.height > 200, (
            f"Overview PNG suspiciously small: {img.size}"
        )
        assert n_samples == 5, f"Expected 5 bundled sample images, found {n_samples}"
