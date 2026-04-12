"""Tests for export.py — weight export and round-trip verification."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from export import export_model, verify_export
from models import NAFNet, NAFNetConfig


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_model() -> NAFNet:
    """A tiny NAFNet for fast export tests."""
    config = NAFNetConfig.tiny()
    config.base_channels = 8
    return NAFNet(config).eval()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


class TestExportModel:
    def test_manifest_created(self, tiny_model: NAFNet, tmp_path: Path) -> None:
        manifest_path = export_model(tiny_model, tmp_path)
        assert manifest_path.exists()

    def test_manifest_has_required_keys(self, tiny_model: NAFNet, tmp_path: Path) -> None:
        export_model(tiny_model, tmp_path)
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)
        assert "version" in manifest
        assert "dtype" in manifest
        assert "architecture" in manifest
        assert "layers" in manifest
        assert len(manifest["layers"]) > 0

    def test_architecture_fields(self, tiny_model: NAFNet, tmp_path: Path) -> None:
        export_model(tiny_model, tmp_path)
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)
        arch = manifest["architecture"]
        assert arch["type"] == "nafnet_residual"
        assert arch["base_channels"] == 8
        assert arch["num_levels"] == 3

    def test_bin_files_created(self, tiny_model: NAFNet, tmp_path: Path) -> None:
        export_model(tiny_model, tmp_path)
        bins = list((tmp_path / "weights").glob("*.bin"))
        assert len(bins) == len(tiny_model.state_dict())

    def test_bn_stats_are_float32(self, tiny_model: NAFNet, tmp_path: Path) -> None:
        export_model(tiny_model, tmp_path, dtype="float16")
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)
        for entry in manifest["layers"]:
            if "running_mean" in entry["name"] or "running_var" in entry["name"]:
                assert entry["dtype"] == "float32", (
                    f"BN stat {entry['name']} should be float32, got {entry['dtype']}"
                )

    def test_conv_weights_are_float16(self, tiny_model: NAFNet, tmp_path: Path) -> None:
        export_model(tiny_model, tmp_path, dtype="float16")
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)
        for entry in manifest["layers"]:
            if entry["name"].endswith(".weight") and "bn" not in entry["name"]:
                assert entry["dtype"] == "float16", (
                    f"Conv weight {entry['name']} should be float16"
                )

    def test_export_float32(self, tiny_model: NAFNet, tmp_path: Path) -> None:
        export_model(tiny_model, tmp_path, dtype="float32")
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["dtype"] == "float32"

    def test_bin_shapes_match_state_dict(self, tiny_model: NAFNet, tmp_path: Path) -> None:
        export_model(tiny_model, tmp_path, dtype="float32")
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)
        state = tiny_model.state_dict()
        for entry in manifest["layers"]:
            name = entry["name"]
            assert name in state, f"Layer {name} not in state_dict"
            assert list(state[name].shape) == entry["shape"], (
                f"Shape mismatch for {name}: manifest={entry['shape']}, "
                f"model={list(state[name].shape)}"
            )


# ---------------------------------------------------------------------------
# Round-trip verification
# ---------------------------------------------------------------------------


class TestVerifyExport:
    def test_verify_passes_float32(self, tiny_model: NAFNet, tmp_path: Path) -> None:
        manifest_path = export_model(tiny_model, tmp_path, dtype="float32")
        assert verify_export(tiny_model, manifest_path, rtol=1e-5)

    def test_verify_passes_float16(self, tiny_model: NAFNet, tmp_path: Path) -> None:
        manifest_path = export_model(tiny_model, tmp_path, dtype="float16")
        # FP16 has ~1e-3 relative error — use generous tolerance
        assert verify_export(tiny_model, manifest_path, rtol=1e-2)

    def test_verify_fails_on_corrupted_bin(self, tiny_model: NAFNet, tmp_path: Path) -> None:
        manifest_path = export_model(tiny_model, tmp_path, dtype="float32")
        with open(tmp_path / "manifest.json") as f:
            manifest = json.load(f)
        # Corrupt the first bin file
        first_bin = tmp_path / manifest["layers"][0]["file"]
        data = np.fromfile(str(first_bin), dtype=np.float32)
        data += 1000.0  # large offset — will definitely fail allclose
        data.tofile(str(first_bin))
        assert not verify_export(tiny_model, manifest_path, rtol=1e-2)
