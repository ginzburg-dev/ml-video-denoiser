# Changelog

All notable changes to this project will be documented in this file.
Generated from conventional commits via [git-cliff](https://git-cliff.org).

<!-- cliff-start -->
## [Unreleased]

### Added

- **`CameraNoiseGenerator`** (`noise_generators.py`) — ISO-parameterised Poisson-Gaussian noise model for camera video. Maps ISO values to shot noise gain K and read noise σ_r via empirical formulas (`K = K_ref·iso/iso_ref`, `σ_r = sr_ref·√(iso/iso_ref)`). Supports single-frame and clip-consistent modes.
- **`_ClipNoiseApplier`** (`noise_generators.py`) — returned by `CameraNoiseGenerator.for_clip()`; fixes ISO parameters and row-banding fixed-pattern noise for an entire clip so all frames see the same noise character, matching real camera behaviour.
- **`_make_offset_head(skip_ch)`** (`models.py`) — MobileNet-style warp offset head replacing the previous single `Conv2d(2C, 2, 1)`. Architecture: DW-Conv 3×3 → PW-Conv 1×1 → ReLU → PW-Conv 1×1. Gives the head spatial context to estimate motion; only the final projection is zero-initialised.
- **`denoise_temporal_sequence()`** (`infer.py`) — denoises all frames in a sequence with optional test-time temporal flip averaging. Runs the sequence forward and in reverse, then averages both predictions per frame, reducing noise variance and improving temporal consistency at 2× inference cost.
- **`--no-temporal-flip`** flag (`infer.py`) — disables temporal flip averaging for temporal models (enabled by default).
- **`save_temporal_clip_grid()`** (`tests/visualise_noise.py`) — visualises per-frame vs clip-consistent `CameraNoiseGenerator` noise across N synthetic consecutive frames. Produces `temporal_clip_consistency.png` showing noisy frames, frame-to-frame |Δ| ×8, and noise residuals in parallel rows, with mean |Δ̄| in the summary column.
- **`CameraNoiseGenerator` in `visualise_noise.py` generator list** — `Camera ISO 100–6400` row now appears in all diagnostic grids alongside Gaussian, Poisson-Gaussian, and Mixed generators.
- **`tests/noise_preview.py`** — new CLI for rapid noise quality assessment on user-supplied images. Applies selected generators across N noise levels and produces a labelled PNG grid (Clean | Noisy | Residual×N | Sigma map). Optional `--temporal` flag adds a temporal consistency strip showing per-frame vs `for_clip()` noise across N frames.

### Changed

- **`NAFNetTemporal` warp offset heads** (`models.py`) — replaced single `Conv2d(2C, 2, kernel_size=1)` with `_make_offset_head()` (DW 3×3 → PW 1×1 → ReLU → PW 1×1). The deeper head has a 3×3 receptive field to estimate displacements from spatial context; previous single-conv head had no spatial context.
- **Temporal inference loop** (`infer.py`) — `main()` now calls `denoise_temporal_sequence()` instead of a per-frame `denoise_temporal_frame()` loop, enabling temporal flip averaging as the default behaviour.
- **`test_noise_visual.py`** — updated `EXPECTED_STEMS` to include `temporal_clip_consistency`; minimum expected per-generator PNG count raised from 4 to 5.
- **`test_models.py`** — fixed `test_temporal_keys_untouched`: accesses `m[0].weight` on `temporal_mix` entries (which are `nn.Sequential`) instead of `m.weight`.

### Initial
- Initial project scaffold: training pipeline, noise generators, models, C++ engine structure
<!-- cliff-end -->
