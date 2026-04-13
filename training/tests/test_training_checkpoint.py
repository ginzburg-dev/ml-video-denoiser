"""Tests for checkpoint metadata and resume semantics."""

from pathlib import Path

import torch
from torch.optim import AdamW

from models import NAFNetConfig, NAFNetTemporal
from training import _load_checkpoint, _save_checkpoint, warmup_cosine_schedule


def _make_temporal_stack() -> tuple[NAFNetTemporal, AdamW, torch.optim.lr_scheduler.LambdaLR]:
    model = NAFNetTemporal(NAFNetConfig.tiny(), num_frames=5, use_warp=True)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = warmup_cosine_schedule(optimizer, warmup_epochs=1, total_epochs=4)
    return model, optimizer, scheduler


class TestTrainingCheckpoint:
    def test_checkpoint_stores_model_metadata(self, tmp_path: Path) -> None:
        model, optimizer, scheduler = _make_temporal_stack()
        ckpt_path = tmp_path / "epoch_0001.pth"

        _save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch=0, best_psnr=12.5)

        payload = torch.load(ckpt_path, map_location="cpu")
        assert payload["model_metadata"]["model_type"] == "temporal"
        assert payload["model_metadata"]["num_frames"] == 5
        assert payload["model_metadata"]["use_warp"] is True
        assert payload["model_metadata"]["naf_config"]["base_channels"] == 16

    def test_load_checkpoint_resumes_from_next_epoch(self, tmp_path: Path) -> None:
        model, optimizer, scheduler = _make_temporal_stack()
        ckpt_path = tmp_path / "epoch_0001.pth"
        _save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch=0, best_psnr=9.0)

        model2, optimizer2, scheduler2 = _make_temporal_stack()
        start_epoch, best_psnr = _load_checkpoint(
            ckpt_path,
            model2,
            optimizer2,
            scheduler2,
            device=torch.device("cpu"),
        )

        assert start_epoch == 1
        assert best_psnr == 9.0
