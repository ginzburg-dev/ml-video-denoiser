"""Tests for the training CLI parser and noise selection."""

import pytest

from noise_generators import (
    GaussianNoiseGenerator,
    MixedNoiseGenerator,
    PoissonGaussianNoiseGenerator,
)
from training import _make_noise_generator, build_parser


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
