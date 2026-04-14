"""Tests for the training CLI parser and noise selection."""

import pytest
import torch
from torch.utils.data import Dataset

from noise_generators import (
    GaussianNoiseGenerator,
    MixedNoiseGenerator,
    PoissonGaussianNoiseGenerator,
)
from dataset import CombinedDataset
from training import (
    _config_summary_lines,
    _dataset_summary_lines,
    _make_loss,
    _make_noise_generator,
    _temporal_model_config,
    _temporal_sampling_config,
    _validation_patch_repeats,
    _validation_temporal_config,
    _validation_mode,
    build_parser,
)


class _DummyDataset(Dataset):
    def __init__(self, length: int, *, num_images: int | None = None) -> None:
        self._length = length
        self.num_images = num_images

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        return (torch.zeros(1),)


class TestTrainingCli:
    def test_size_flag_selects_model_preset(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data", "images", "--size", "standard"])
        assert args.size == "standard"

    def test_naf_preset_flag_is_rejected(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--data", "images", "--naf-preset", "standard"])

    def test_noise_flag_parses_explicitly(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data", "images", "--noise", "gaussian"])
        assert args.noise == "gaussian"

    def test_loss_flag_parses_explicitly(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data", "images", "--loss", "log-l1"])
        assert args.loss == "log-l1"

    def test_color_space_flag_parses_explicitly(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data", "images", "--color-space", "log"])
        assert args.color_space == "log"

    def test_scheduler_flag_parses_explicitly(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data", "images", "--scheduler", "plateau"])
        assert args.scheduler == "plateau"

    def test_noise_abbreviation_is_rejected(self) -> None:
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--data", "images", "--noi", "gaussian"])

    def test_gaussian_noise_generator_selected(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data", "images", "--noise", "gaussian"])
        assert isinstance(_make_noise_generator(args, parser), GaussianNoiseGenerator)

    def test_poisson_gaussian_noise_generator_selected(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data", "images", "--noise", "poisson-gaussian"])
        assert isinstance(_make_noise_generator(args, parser), PoissonGaussianNoiseGenerator)

    def test_mixed_noise_generator_selected(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data", "images", "--noise", "mixed"])
        assert isinstance(_make_noise_generator(args, parser), MixedNoiseGenerator)

    def test_l1_loss_selected(self) -> None:
        assert isinstance(_make_loss("l1"), torch.nn.Module)

    def test_log_l1_loss_selected(self) -> None:
        assert _make_loss("log-l1").__class__.__name__ == "LogL1Loss"

    def test_real_noise_inputs_require_mixed_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--data", "images", "--noise", "gaussian", "--noise-profile", "profile.json"
        ])
        with pytest.raises(SystemExit):
            _make_noise_generator(args, parser)

    def test_synthetic_validation_mode_selected(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data", "images", "--val-data", "val_images"])
        assert _validation_mode(args, parser) == ("synthetic", (["val_images"],))

    def test_paired_validation_mode_selected(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--data", "images",
            "--val-clean", "val_clean",
            "--val-noisy", "val_noisy",
        ])
        assert _validation_mode(args, parser) == (
            "paired",
            (["val_clean"], ["val_noisy"]),
        )

    def test_mixed_validation_modes_are_rejected(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--data", "images",
            "--val-data", "val_images",
            "--val-clean", "val_clean",
            "--val-noisy", "val_noisy",
        ])
        with pytest.raises(SystemExit):
            _validation_mode(args, parser)

    def test_val_clean_requires_val_noisy(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data", "images", "--val-clean", "val_clean"])
        with pytest.raises(SystemExit):
            _validation_mode(args, parser)

    def test_temporal_sampling_defaults_to_off(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data", "images"])
        assert _temporal_sampling_config(args, parser) == (False, None)

    def test_temporal_model_defaults_to_3_frames(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data", "images", "--model", "temporal"])
        assert args.num_frames == 3

    def test_windows_per_sequence_enables_random_temporal_sampling(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--data", "images",
            "--model", "temporal",
            "--windows-per-sequence", "4",
        ])
        assert _temporal_sampling_config(args, parser) == (True, 4)

    def test_temporal_sampling_requires_temporal_model(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data", "images", "--windows-per-sequence", "4"])
        with pytest.raises(SystemExit):
            _temporal_sampling_config(args, parser)

    def test_windows_per_sequence_must_be_positive(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--data", "images",
            "--model", "temporal",
            "--windows-per-sequence", "0",
        ])
        with pytest.raises(SystemExit):
            _temporal_sampling_config(args, parser)

    def test_even_num_frames_rejected_for_temporal_model(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--data", "images",
            "--model", "temporal",
            "--num-frames", "4",
        ])
        with pytest.raises(SystemExit):
            _temporal_model_config(args, parser)

    def test_val_windows_per_sequence_requires_temporal_model(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--data", "images",
            "--val-data", "val_images",
            "--val-windows-per-sequence", "2",
        ])
        with pytest.raises(SystemExit):
            _validation_temporal_config(args, parser, "synthetic")

    def test_val_windows_per_sequence_requires_validation(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--data", "images",
            "--model", "temporal",
            "--val-windows-per-sequence", "2",
        ])
        with pytest.raises(SystemExit):
            _validation_temporal_config(args, parser, None)

    def test_validation_temporal_config_returns_crop_mode(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--data", "images",
            "--model", "temporal",
            "--val-data", "val_images",
            "--val-windows-per-sequence", "3",
            "--val-crop-mode", "center",
        ])
        assert _validation_temporal_config(args, parser, "synthetic") == (3, "center", 2)

    def test_validation_temporal_config_returns_grid_size(self) -> None:
        parser = build_parser()
        args = parser.parse_args([
            "--data", "images",
            "--model", "temporal",
            "--val-data", "val_images",
            "--val-crop-mode", "grid",
            "--val-grid-size", "3",
        ])
        assert _validation_temporal_config(args, parser, "synthetic") == (None, "grid", 3)

    def test_validation_patch_repeats_matches_grid_area(self) -> None:
        assert _validation_patch_repeats("grid", 3) == 9
        assert _validation_patch_repeats("center", 3) == 1

    def test_config_summary_lines_include_temporal_modes(self) -> None:
        lines = _config_summary_lines(
            is_temporal=True,
            random_temporal_windows=True,
            windows_per_sequence=4,
            val_mode="paired",
            val_windows_per_sequence=3,
            val_crop_mode="grid",
            val_grid_size=2,
        )
        assert "Train temporal sampling: random windows (4/sequence/epoch)" in lines
        assert "Val temporal sampling: deterministic windows (3/sequence)" in lines
        assert "Val crop mode: grid (2x2)" in lines

    def test_config_summary_lines_include_loss(self) -> None:
        lines = _config_summary_lines(
            is_temporal=False,
            random_temporal_windows=False,
            windows_per_sequence=None,
            val_mode=None,
            val_windows_per_sequence=None,
            val_crop_mode="random",
            val_grid_size=2,
            loss_name="log-l1",
            color_space="log",
            scheduler_name="plateau",
        )
        assert "Loss: log-l1" in lines
        assert "Color space: log" in lines
        assert "Scheduler: plateau" in lines

    def test_dataset_summary_uses_image_count_when_available(self) -> None:
        lines = _dataset_summary_lines("Train", _DummyDataset(128, num_images=8))
        assert lines == ["Train: 128 samples/epoch from 8 images"]

    def test_dataset_summary_expands_combined_dataset(self) -> None:
        combined = CombinedDataset(
            datasets=[_DummyDataset(80, num_images=5), _DummyDataset(20, num_images=2)],
            weights=[0.8, 0.2],
            num_samples=100,
        )
        lines = _dataset_summary_lines("Train", combined)
        assert lines[0] == "Train: 100 samples/epoch across 2 datasets"
        assert "Train component 1 (80%): 80 samples/epoch from 5 images" in lines[1]
        assert "Train component 2 (20%): 20 samples/epoch from 2 images" in lines[2]
