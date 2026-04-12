"""Tests for infer.py helpers."""

from pathlib import Path

import numpy as np
import torch

from infer import _clip_indices, _load_image, denoise_temporal_frame


class TestInferImageLoading:
    def test_load_exr_image(self, tmp_path: Path) -> None:
        import OpenEXR

        exr_path = tmp_path / "frame.exr"
        rgba = np.zeros((8, 8, 4), dtype=np.float32)
        rgba[..., 0] = 0.1
        rgba[..., 1] = 0.2
        rgba[..., 2] = 0.3
        rgba[..., 3] = 0.9
        OpenEXR.File({"type": OpenEXR.scanlineimage}, {"RGBA": rgba}).write(str(exr_path))

        img = _load_image(exr_path)
        assert img.shape == (8, 8, 3)
        assert np.isclose(img[..., 0].mean(), 0.1)
        assert np.isclose(img[..., 1].mean(), 0.2)
        assert np.isclose(img[..., 2].mean(), 0.3)


class _DummyTemporalModel(torch.nn.Module):
    def __init__(self, num_frames: int = 5) -> None:
        super().__init__()
        self._num_frames = num_frames

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        # Return the centre frame so shape handling can be tested directly.
        return clip[:, self._num_frames // 2]


class TestTemporalInfer:
    def test_clip_indices_replicate_edges(self) -> None:
        assert _clip_indices(length=4, centre_idx=0, num_frames=5) == [0, 0, 0, 1, 2]
        assert _clip_indices(length=4, centre_idx=3, num_frames=5) == [1, 2, 3, 3, 3]

    def test_denoise_temporal_frame_builds_5d_clip_and_crops_back(self) -> None:
        model = _DummyTemporalModel(num_frames=5)
        sequence = [
            np.full((6, 5, 3), 0.1, dtype=np.float32),
            np.full((7, 6, 3), 0.2, dtype=np.float32),
            np.full((8, 7, 3), 0.3, dtype=np.float32),
            np.full((7, 6, 3), 0.4, dtype=np.float32),
            np.full((6, 5, 3), 0.5, dtype=np.float32),
        ]

        output = denoise_temporal_frame(model, sequence, frame_idx=2, device=torch.device("cpu"), use_amp=False)

        assert output.shape == sequence[2].shape
        assert np.allclose(output, sequence[2])
