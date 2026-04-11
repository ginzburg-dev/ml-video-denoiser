"""Tests for models.py — NEFResidual and NEFTemporal."""

import sys
from pathlib import Path

import pytest
import torch
from torch import Tensor

sys.path.insert(0, str(Path(__file__).parent.parent))
from models import (
    ConvBnRelu,
    DecoderBlock,
    DeformableAlignment,
    EncoderBlock,
    ModelConfig,
    NEFResidual,
    NEFTemporal,
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
# Building blocks
# ---------------------------------------------------------------------------


class TestConvBnRelu:
    def test_output_shape(self) -> None:
        block = ConvBnRelu(3, 64)
        x = _make_batch()
        out = block(x)
        assert out.shape == (1, 64, 128, 128)

    def test_no_bias_by_default(self) -> None:
        block = ConvBnRelu(3, 64)
        assert block.conv.bias is None

    def test_output_nonnegative_after_relu(self) -> None:
        block = ConvBnRelu(3, 64)
        out = block(_make_batch())
        assert out.min().item() >= 0.0


class TestEncoderBlock:
    def test_output_shapes(self) -> None:
        block = EncoderBlock(3, 64)
        pooled, skip = block(_make_batch())
        assert pooled.shape == (1, 64, 64, 64)   # halved spatial
        assert skip.shape == (1, 64, 128, 128)    # before pooling

    def test_eval_mode_no_bn_stats_update(self) -> None:
        block = EncoderBlock(3, 64).eval()
        with torch.no_grad():
            block(_make_batch())  # should not raise


class TestDecoderBlock:
    def test_output_shape(self) -> None:
        block = DecoderBlock(in_ch=128, skip_ch=64, out_ch=64)
        x = torch.rand(1, 128, 32, 32)
        skip = torch.rand(1, 64, 64, 64)
        out = block(x, skip)
        assert out.shape == (1, 64, 64, 64)


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
# NEFResidual
# ---------------------------------------------------------------------------


class TestNEFResidual:
    def test_output_shape_standard(self) -> None:
        model = NEFResidual(ModelConfig.standard()).eval()
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape

    def test_output_shape_lite(self) -> None:
        model = NEFResidual(ModelConfig.lite()).eval()
        x = _make_batch(h=64, w=64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape

    def test_output_in_range(self) -> None:
        model = NEFResidual().eval()
        x = _make_batch()
        with torch.no_grad():
            out = model(x)
        assert out.min().item() >= -1e-5
        assert out.max().item() <= 1.0 + 1e-5

    def test_non_multiple_input(self) -> None:
        """Model should handle any spatial size via auto-padding."""
        model = NEFResidual().eval()
        x = torch.rand(1, 3, 100, 150)
        with torch.no_grad():
            out = model(x)
        assert out.shape == x.shape

    def test_batch_size_2(self) -> None:
        model = NEFResidual().eval()
        x = _make_batch(b=2)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3, 128, 128)

    def test_gradient_flows(self) -> None:
        model = NEFResidual(ModelConfig.lite())
        x = _make_batch(h=64, w=64)
        out = model(x)
        loss = out.mean()
        loss.backward()
        # At least one gradient should be non-zero
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
        assert any(g.abs().sum().item() > 0 for g in grads)

    def test_config_lite_fewer_params_than_standard(self) -> None:
        lite = sum(p.numel() for p in NEFResidual(ModelConfig.lite()).parameters())
        standard = sum(p.numel() for p in NEFResidual(ModelConfig.standard()).parameters())
        assert lite < standard


# ---------------------------------------------------------------------------
# DeformableAlignment
# ---------------------------------------------------------------------------


class TestDeformableAlignment:
    def test_output_shape(self) -> None:
        pytest.importorskip("torchvision")
        align = DeformableAlignment(channels=64)
        ref = torch.rand(1, 64, 32, 32)
        neighbour = torch.rand(1, 64, 32, 32)
        out = align(ref, neighbour)
        assert out.shape == ref.shape

    def test_identity_init_close_to_neighbour(self) -> None:
        """With zero offsets the aligned output should be close to the neighbour."""
        pytest.importorskip("torchvision")
        align = DeformableAlignment(channels=16).eval()
        # Zero-initialise offsets so the alignment is roughly identity
        torch.nn.init.zeros_(align.offset_conv.weight)
        torch.nn.init.zeros_(align.offset_conv.bias)
        torch.nn.init.ones_(align.mask_conv.bias)  # mask=1 → full contribution
        ref = torch.rand(1, 16, 32, 32)
        neighbour = torch.rand(1, 16, 32, 32)
        with torch.no_grad():
            out = align(ref, neighbour)
        assert out.shape == neighbour.shape


# ---------------------------------------------------------------------------
# NEFTemporal
# ---------------------------------------------------------------------------


class TestNEFTemporal:
    def test_output_shape(self) -> None:
        pytest.importorskip("torchvision")
        model = NEFTemporal(ModelConfig.lite()).eval()
        clip = _make_clip(h=64, w=64)
        with torch.no_grad():
            out = model(clip)
        # Output is the denoised centre frame: (B, C, H, W)
        assert out.shape == (1, 3, 64, 64)

    def test_output_in_range(self) -> None:
        pytest.importorskip("torchvision")
        model = NEFTemporal(ModelConfig.lite()).eval()
        clip = _make_clip(h=64, w=64)
        with torch.no_grad():
            out = model(clip)
        assert out.min().item() >= -1e-5
        assert out.max().item() <= 1.0 + 1e-5

    def test_wrong_frame_count_raises(self) -> None:
        pytest.importorskip("torchvision")
        model = NEFTemporal(ModelConfig.lite()).eval()
        clip = _make_clip(t=3, h=64, w=64)  # config expects 5
        with torch.no_grad(), pytest.raises(AssertionError):
            model(clip)
