"""Tests for infer.py image loading helpers."""

from pathlib import Path

import numpy as np

from infer import _load_image


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
