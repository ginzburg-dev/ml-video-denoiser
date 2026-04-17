"""Tests for checkpoint metadata and resume semantics."""

from pathlib import Path

import torch
from torch.optim import AdamW

from models import NAFNetConfig, NAFNetTemporal, freeze_spatial
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

        _save_checkpoint(
            ckpt_path,
            model,
            optimizer,
            scheduler,
            epoch=0,
            best_psnr=12.5,
            training_config={"color_space": "log", "scheduler_name": "plateau"},
        )

        payload = torch.load(ckpt_path, map_location="cpu")
        assert payload["model_metadata"]["model_type"] == "temporal"
        assert payload["model_metadata"]["num_frames"] == 5
        assert payload["model_metadata"]["use_warp"] is True
        assert payload["model_metadata"]["naf_config"]["base_channels"] == 16
        assert payload["training_config"]["color_space"] == "log"
        assert payload["training_config"]["scheduler_name"] == "plateau"

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

    def test_load_checkpoint_resets_optimizer_when_trainable_params_change(self, tmp_path: Path) -> None:
        model = NAFNetTemporal(NAFNetConfig.tiny(), num_frames=5, use_warp=False)
        freeze_spatial(model)
        optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
        scheduler = warmup_cosine_schedule(optimizer, warmup_epochs=1, total_epochs=4)
        ckpt_path = tmp_path / "stage2_best.pth"
        _save_checkpoint(ckpt_path, model, optimizer, scheduler, epoch=59, best_psnr=38.78)

        model2 = NAFNetTemporal(NAFNetConfig.tiny(), num_frames=5, use_warp=False)
        optimizer2 = AdamW([p for p in model2.parameters() if p.requires_grad], lr=5e-4)
        scheduler2 = warmup_cosine_schedule(optimizer2, warmup_epochs=1, total_epochs=4)

        expected_state = {
            name: tensor.detach().clone()
            for name, tensor in model.state_dict().items()
        }

        start_epoch, best_psnr = _load_checkpoint(
            ckpt_path,
            model2,
            optimizer2,
            scheduler2,
            device=torch.device("cpu"),
        )

        assert start_epoch == 0
        assert best_psnr == 0.0
        for name, tensor in model2.state_dict().items():
            assert torch.equal(tensor, expected_state[name])
