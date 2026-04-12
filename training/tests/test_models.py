"""Tests for models.py — NAFNet and NAFNetTemporal."""

import sys
from pathlib import Path

import pytest
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import (
    NAFNet,
    NAFNetConfig,
    NAFNetTemporal,
    _pad_to_multiple,
    _unpad,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(b: int = 1, c: int = 3, h: int = 128, w: int = 128) -> Tensor:
    return torch.rand(b, c, h, w)


def _make_clip(b: int = 1, t: int = 5, c: int = 3, h: int = 64, w: int = 64) -> Tensor:
    return torch.rand(b, t, c, h, w)


# ---------------------------------------------------------------------------
# Padding utilities
# ---------------------------------------------------------------------------


class TestPadding:
    @pytest.mark.parametrize("multiple", [16, 32])
    @pytest.mark.parametrize("h, w", [(100, 100), (127, 255), (128, 128)])
    def test_padded_is_divisible(self, h: int, w: int, multiple: int) -> None:
        x = torch.rand(1, 3, h, w)
        padded, _ = _pad_to_multiple(x, multiple)
        assert padded.shape[-2] % multiple == 0
        assert padded.shape[-1] % multiple == 0

    def test_unpad_restores_original_size(self) -> None:
        x = torch.rand(1, 3, 100, 100)
        padded, padding = _pad_to_multiple(x, 16)
        restored = _unpad(padded, padding)
        assert restored.shape == x.shape

    def test_no_padding_needed(self) -> None:
        x = torch.rand(1, 3, 128, 128)
        padded, padding = _pad_to_multiple(x, 16)
        assert padded.shape == x.shape
        assert padding == (0, 0, 0, 0)


# ---------------------------------------------------------------------------
# NAFNet
# ---------------------------------------------------------------------------


class TestNAFNetForward:
    def test_output_shape_standard(self) -> None:
        model = NAFNet(NAFNetConfig.standard()).eval()
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 128, 128)

    def test_output_shape_small(self) -> None:
        model = NAFNet(NAFNetConfig.small()).eval()
        x = _make_batch(h=64, w=64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 64, 64)

    def test_odd_spatial_dims(self) -> None:
        model = NAFNet(NAFNetConfig.small()).eval()
        x = torch.rand(1, 3, 100, 150)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 100, 150)

    def test_output_is_finite(self) -> None:
        model = NAFNet(NAFNetConfig.small()).eval()
        x = _make_batch(h=64, w=64)
        with torch.no_grad():
            out = model(x)
        assert torch.isfinite(out).all()

    def test_batch_size_2(self) -> None:
        model = NAFNet(NAFNetConfig.small()).eval()
        x = _make_batch(b=2, h=64, w=64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3, 64, 64)

    def test_gradient_flows(self) -> None:
        model = NAFNet(NAFNetConfig.tiny())
        x = _make_batch(h=32, w=32)
        out = model(x)
        loss = out.mean()
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert any(g.abs().sum().item() > 0 for g in grads)


class TestNAFNetAuxChannels:
    def test_9ch_input_returns_3ch_output(self) -> None:
        model = NAFNet(NAFNetConfig(in_channels=9, base_channels=16)).eval()
        x = torch.rand(1, 9, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 64, 64)


class TestNAFNetIdentityInit:
    def test_output_close_to_beauty_at_init(self) -> None:
        """At epoch 0 (zero-init ending) output ≈ input[:, :3]."""
        model = NAFNet(NAFNetConfig.tiny()).eval()
        x = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert torch.allclose(out, x[:, :3], atol=1e-5)


class TestNAFNetConfigs:
    @pytest.mark.parametrize("preset", ["tiny", "small", "standard", "wide"])
    def test_preset_builds_and_runs(self, preset: str) -> None:
        cfg = getattr(NAFNetConfig, preset)()
        model = NAFNet(cfg).eval()
        x = torch.rand(1, 3, 32, 32)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 3, 32, 32)

    def test_tiny_fewer_params_than_wide(self) -> None:
        tiny = sum(p.numel() for p in NAFNet(NAFNetConfig.tiny()).parameters())
        wide = sum(p.numel() for p in NAFNet(NAFNetConfig.wide()).parameters())
        assert tiny < wide


# ---------------------------------------------------------------------------
# NAFNetTemporal
# ---------------------------------------------------------------------------


class TestNAFNetTemporalForward:
    def test_output_shape(self) -> None:
        model = NAFNetTemporal(NAFNetConfig.tiny()).eval()
        clip = _make_clip(h=32, w=32)
        with torch.no_grad():
            out = model(clip)
        assert out.shape == (1, 3, 32, 32)

    def test_output_is_finite(self) -> None:
        model = NAFNetTemporal(NAFNetConfig.tiny()).eval()
        clip = _make_clip(h=32, w=32)
        with torch.no_grad():
            out = model(clip)
        assert torch.isfinite(out).all()

    def test_wrong_frame_count_raises(self) -> None:
        model = NAFNetTemporal(NAFNetConfig.tiny(), num_frames=5).eval()
        clip = _make_clip(t=3, h=32, w=32)
        with torch.no_grad(), pytest.raises(AssertionError):
            model(clip)

    def test_identity_init(self) -> None:
        """At epoch 0 the output should ≈ centre frame beauty (zero-init ending)."""
        model = NAFNetTemporal(NAFNetConfig.tiny()).eval()
        clip = torch.rand(1, 5, 3, 32, 32)
        ref = clip[:, 2, :3]   # centre frame beauty
        with torch.no_grad():
            out = model(clip)
        assert torch.allclose(out, ref, atol=1e-5)


class TestNAFNetTemporalConfigs:
    @pytest.mark.parametrize("preset", ["tiny", "small"])
    def test_preset_builds_and_runs(self, preset: str) -> None:
        cfg = getattr(NAFNetConfig, preset)()
        model = NAFNetTemporal(cfg).eval()
        clip = torch.rand(1, 5, 3, 32, 32)
        with torch.no_grad():
            out = model(clip)
        assert out.shape == (1, 3, 32, 32)


class TestNAFNetTemporalWarp:
    def test_use_warp_output_shape(self) -> None:
        model = NAFNetTemporal(NAFNetConfig.tiny(), use_warp=True).eval()
        clip = _make_clip(h=32, w=32)
        with torch.no_grad():
            out = model(clip)
        assert out.shape == (1, 3, 32, 32)


# ---------------------------------------------------------------------------
# Spatial weight transfer and freeze/unfreeze
# ---------------------------------------------------------------------------


class TestSpatialWeightTransfer:
    def test_transfers_all_matching_keys(self) -> None:
        from models import load_spatial_weights
        import tempfile, os

        spatial = NAFNet(NAFNetConfig.tiny())
        temporal = NAFNetTemporal(NAFNetConfig.tiny())

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save({"model_state_dict": spatial.state_dict()}, tmp_path)
            n = load_spatial_weights(temporal, tmp_path)
        finally:
            os.unlink(tmp_path)

        assert n > 0
        # Spatial keys should match
        spatial_keys = set(spatial.state_dict().keys())
        temporal_keys = set(temporal.state_dict().keys())
        overlap = spatial_keys & temporal_keys
        assert n == len(overlap)

    def test_temporal_keys_untouched(self) -> None:
        from models import load_spatial_weights
        import tempfile, os

        spatial = NAFNet(NAFNetConfig.tiny())
        temporal = NAFNetTemporal(NAFNetConfig.tiny())

        # Zero out temporal_mix weights in temporal before loading
        for m in temporal.temporal_mix:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as f:
            tmp_path = f.name
        try:
            torch.save({"model_state_dict": spatial.state_dict()}, tmp_path)
            load_spatial_weights(temporal, tmp_path)
        finally:
            os.unlink(tmp_path)

        # temporal_mix weights should still be zero (not touched by load)
        for m in temporal.temporal_mix:
            assert m.weight.abs().sum().item() == 0.0


import torch.nn as nn


class TestFreezeUnfreeze:
    def test_freeze_spatial_only_temporal_trainable(self) -> None:
        from models import freeze_spatial

        model = NAFNetTemporal(NAFNetConfig.tiny())
        freeze_spatial(model)

        frozen_count = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert frozen_count > 0
        assert trainable_count > 0

        # temporal_mix should remain trainable
        for p in model.temporal_mix.parameters():
            assert p.requires_grad

    def test_unfreeze_restores_all_gradients(self) -> None:
        from models import freeze_spatial, unfreeze_spatial

        model = NAFNetTemporal(NAFNetConfig.tiny())
        freeze_spatial(model)
        unfreeze_spatial(model)

        for p in model.parameters():
            assert p.requires_grad
