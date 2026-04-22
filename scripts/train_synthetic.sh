#!/usr/bin/env bash
# train_synthetic.sh — Three-stage NAFNet training with synthetic camera noise.
#
# Identical pipeline to train.sh but uses on-the-fly CameraNoiseGenerator instead
# of paired clean/noisy data.  Only a clean sequence directory is required.
#
# Stage 1  Spatial      NAFNet spatial denoiser                    (plateau)
# Stage 2  Temporal     Load stage-1 weights, freeze spatial, train temporal_mix only
# Stage 3  Fine-tune    Unfreeze all, joint fine-tune at lower LR (plateau)
#
# Usage:
#   DATA=/path/to/clean/sequences ./scripts/train_synthetic.sh
#
# Override any variable on the command line, e.g.:
#   WORKERS=4 SIZE=small DATA=/my/clips ./scripts/train_synthetic.sh
#   ISO_MIN=200 ISO_MAX=3200 ./scripts/train_synthetic.sh
#
# Skip individual stages:
#   SKIP_STAGE1=1 SPATIAL_WEIGHTS=/path/to/spatial/best.pth ./scripts/train_synthetic.sh
#   SKIP_STAGE2=1 ./scripts/train_synthetic.sh   # requires STAGE2_OUTPUT to already exist
#   SKIP_STAGE3=1 ./scripts/train_synthetic.sh   # stop after stage 2

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

# ---------------------------------------------------------------------------
# Data paths  (clean sequences only — noise is synthesised on the fly)
# ---------------------------------------------------------------------------
DATA="${DATA:-$HOME/data/clean_sequences}"
VAL_DATA="${VAL_DATA:-}"   # optional separate val split; if empty, validation is skipped

# ---------------------------------------------------------------------------
# Noise synthesis
# ---------------------------------------------------------------------------
NOISE="${NOISE:-camera}"         # camera | poisson-gaussian | gaussian | mixed
ISO_MIN="${ISO_MIN:-100}"
ISO_MAX="${ISO_MAX:-3200}"
COLOR_SPACE="${COLOR_SPACE:-log}"   # log recommended for synthetic camera noise

# ---------------------------------------------------------------------------
# Model / architecture
# ---------------------------------------------------------------------------
SIZE="${SIZE:-exp048}"          # tiny | small | exp048 | standard | wide
NUM_FRAMES="${NUM_FRAMES:-3}"   # temporal window (3 = 1 past + ref + 1 future)
USE_WARP="${USE_WARP:-1}"       # 1 = enable learned warp (recommended for camera video)
EXP_NAME="${EXP_NAME:-}"        # optional experiment tag, e.g. EXP_NAME=run01

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_sfx="${EXP_NAME:+_$EXP_NAME}"
STAGE1_OUTPUT="${STAGE1_OUTPUT:-checkpoints/synth_spatial_${SIZE}${_sfx}}"
SPATIAL_WEIGHTS="${SPATIAL_WEIGHTS:-$STAGE1_OUTPUT/best.pth}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-checkpoints/synth_temporal_${SIZE}_stage2${_sfx}}"
STAGE3_OUTPUT="${STAGE3_OUTPUT:-checkpoints/synth_temporal_${SIZE}_stage3${_sfx}}"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
WORKERS="${WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PATCH_SIZE="${PATCH_SIZE:-128}"
PATCHES_PER_IMAGE="${PATCHES_PER_IMAGE:-64}"

SPATIAL_EPOCHS="${SPATIAL_EPOCHS:-150}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-250}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-100}"

SPATIAL_LR="${SPATIAL_LR:-1e-4}"
STAGE2_LR="${STAGE2_LR:-1e-4}"   # ~7.5M fresh params (temporal_mix + bottleneck_mix) — run hot
STAGE3_LR="${STAGE3_LR:-2e-5}"

SPATIAL_LOSS="${SPATIAL_LOSS:-l1}"
STAGE2_LOSS="${STAGE2_LOSS:-l1}"
STAGE3_LOSS="${STAGE3_LOSS:-l1}"

SPATIAL_SCHEDULER="${SPATIAL_SCHEDULER:-plateau}"
SPATIAL_PLATEAU_PATIENCE="${SPATIAL_PLATEAU_PATIENCE:-20}"
STAGE2_SCHEDULER="${STAGE2_SCHEDULER:-plateau}"
STAGE2_PLATEAU_PATIENCE="${STAGE2_PLATEAU_PATIENCE:-20}"
STAGE3_SCHEDULER="${STAGE3_SCHEDULER:-plateau}"
STAGE3_PLATEAU_PATIENCE="${STAGE3_PLATEAU_PATIENCE:-20}"

SPATIAL_FRAMES_PER_SEQUENCE="${SPATIAL_FRAMES_PER_SEQUENCE:-5}"   # 5 random frames/seq/epoch
SPATIAL_VAL_FRAMES_PER_SEQUENCE="${SPATIAL_VAL_FRAMES_PER_SEQUENCE:-3}"
WINDOWS_PER_SEQUENCE="${WINDOWS_PER_SEQUENCE:-4}"                  # 4 random windows/seq/epoch
VAL_WINDOWS_PER_SEQUENCE="${VAL_WINDOWS_PER_SEQUENCE:-3}"

SKIP_STAGE1="${SKIP_STAGE1:-0}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
SKIP_STAGE3="${SKIP_STAGE3:-0}"

# ---------------------------------------------------------------------------

cd "$TRAINING_DIR"

# Build optional val flag (used in all stages)
_val_flags=()
if [[ -n "$VAL_DATA" ]]; then
  _val_flags=(--val-data "$VAL_DATA" --val-crop-mode grid --val-grid-size 3)
fi

# Build optional warp flag (stages 2 & 3)
_warp_flags=()
if [[ "$USE_WARP" == "1" ]]; then
  _warp_flags=(--use-warp)
fi

echo "========================================================"
echo "  NAFNet three-stage synthetic training"
echo "  Size:        $SIZE"
echo "  Num frames:  $NUM_FRAMES"
echo "  Noise:       $NOISE  ISO $ISO_MIN–$ISO_MAX"
echo "  Color space: $COLOR_SPACE"
echo "  Experiment:  ${EXP_NAME:-<default>}"
echo "  Stage 1 output:  $STAGE1_OUTPUT  (${SPATIAL_EPOCHS} epochs)$([ "$SKIP_STAGE1" == "1" ] && echo " [SKIP]")"
echo "  Stage 2 output:  $STAGE2_OUTPUT  (${STAGE2_EPOCHS} epochs)$([ "$SKIP_STAGE2" == "1" ] && echo " [SKIP]")"
echo "  Stage 3 output:  $STAGE3_OUTPUT  (${STAGE3_EPOCHS} epochs)$([ "$SKIP_STAGE3" == "1" ] && echo " [SKIP]")"
echo "========================================================"

# ---------------------------------------------------------------------------
# Stage 1 — Spatial denoiser
# ---------------------------------------------------------------------------
if [[ "$SKIP_STAGE1" == "1" ]]; then
  echo ""
  echo "--- Stage 1 skipped (SKIP_STAGE1=1), using weights: $SPATIAL_WEIGHTS ---"
else
  echo ""
  echo "--- Stage 1: spatial ---"
  uv run python training.py \
    --model spatial \
    --size "$SIZE" \
    --data "$DATA" \
    --noise "$NOISE" \
    --iso-min "$ISO_MIN" \
    --iso-max "$ISO_MAX" \
    --color-space "$COLOR_SPACE" \
    --loss "$SPATIAL_LOSS" \
    --scheduler "$SPATIAL_SCHEDULER" \
    --plateau-patience "$SPATIAL_PLATEAU_PATIENCE" \
    --lr "$SPATIAL_LR" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --frames-per-sequence "$SPATIAL_FRAMES_PER_SEQUENCE" \
    --random-spatial-frames \
    --val-frames-per-sequence "$SPATIAL_VAL_FRAMES_PER_SEQUENCE" \
    "${_val_flags[@]}" \
    --output "$STAGE1_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$SPATIAL_EPOCHS"
fi

# ---------------------------------------------------------------------------
# Stage 2 — Temporal mixing only (spatial backbone frozen)
# ---------------------------------------------------------------------------
if [[ "$SKIP_STAGE2" == "1" ]]; then
  echo ""
  echo "--- Stage 2 skipped (SKIP_STAGE2=1) ---"
else
  echo ""
  echo "--- Stage 2: temporal mix (spatial frozen) ---"
  uv run python training.py \
    --model temporal \
    --size "$SIZE" \
    --num-frames "$NUM_FRAMES" \
    --data "$DATA" \
    --noise "$NOISE" \
    --iso-min "$ISO_MIN" \
    --iso-max "$ISO_MAX" \
    --color-space "$COLOR_SPACE" \
    --loss "$STAGE2_LOSS" \
    --scheduler "$STAGE2_SCHEDULER" \
    --plateau-patience "$STAGE2_PLATEAU_PATIENCE" \
    --lr "$STAGE2_LR" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --patches-per-image "$PATCHES_PER_IMAGE" \
    --random-temporal-windows \
    --windows-per-sequence "$WINDOWS_PER_SEQUENCE" \
    --val-windows-per-sequence "$VAL_WINDOWS_PER_SEQUENCE" \
    "${_val_flags[@]}" \
    --spatial-weights "$SPATIAL_WEIGHTS" \
    --freeze-spatial \
    "${_warp_flags[@]}" \
    --output "$STAGE2_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$STAGE2_EPOCHS"
fi

# ---------------------------------------------------------------------------
# Stage 3 — Joint fine-tune (all layers, half LR)
#
# --resume loads the full model state (spatial + temporal) from stage 2.
# No --spatial-weights here — that would be redundant and confusing.
# No --freeze-spatial — all layers train jointly.
# Optimizer mismatch is expected and handled: stage-2 optimizer was built
# for frozen params only, so training.py falls back to a fresh optimizer
# (model weights are still fully restored from the stage-2 checkpoint).
# ---------------------------------------------------------------------------
if [[ "$SKIP_STAGE3" == "1" ]]; then
  echo ""
  echo "--- Stage 3 skipped (SKIP_STAGE3=1) ---"
else
  echo ""
  echo "--- Stage 3: joint fine-tune ---"
  uv run python training.py \
    --model temporal \
    --size "$SIZE" \
    --num-frames "$NUM_FRAMES" \
    --data "$DATA" \
    --noise "$NOISE" \
    --iso-min "$ISO_MIN" \
    --iso-max "$ISO_MAX" \
    --color-space "$COLOR_SPACE" \
    --loss "$STAGE3_LOSS" \
    --scheduler "$STAGE3_SCHEDULER" \
    --plateau-patience "$STAGE3_PLATEAU_PATIENCE" \
    --lr "$STAGE3_LR" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --patches-per-image "$PATCHES_PER_IMAGE" \
    --random-temporal-windows \
    --windows-per-sequence "$WINDOWS_PER_SEQUENCE" \
    --val-windows-per-sequence "$VAL_WINDOWS_PER_SEQUENCE" \
    "${_val_flags[@]}" \
    --resume "$STAGE2_OUTPUT/best.pth" \
    "${_warp_flags[@]}" \
    --output "$STAGE3_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$STAGE3_EPOCHS"
fi

echo ""
echo "========================================================"
echo "  Done. Final weights: $STAGE3_OUTPUT/best.pth"
echo "========================================================"
