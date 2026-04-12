"""Tests for the training CLI parser and noise selection."""

import pytest

from noise_generators import (
    GaussianNoiseGenerator,
    MixedNoiseGenerator,
    PoissonGaussianNoiseGenerator,
)
from training import _make_noise_generator, _validation_mode, build_parser


class TestTrainingCli:
    def test_noise_flag_parses_explicitly(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--data", "images", "--noise", "gaussian"])
        assert args.noise == "gaussian"

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
