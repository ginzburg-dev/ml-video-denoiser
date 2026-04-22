#!/usr/bin/env bash
# train_cascade.sh — Three-stage NAFNetCascade training pipeline.
#
# Replicates the exp_053 approach:
#   Stage 1  Spatial     Train NAFNet spatial denoiser on paired / synthetic data
#   Stage 2  Cascade     Load stage-1 weights into spatial_stage, freeze it,
#                        train temporal_stage only (small NAFNet fusing T denoised frames)
#   Stage 3  Fine-tune   Unfreeze all, joint fine-tune at lower LR
#
# Usage (paired data):
#   PAIRED_CLEAN=/clean PAIRED_NOISY=/noisy ./scripts/train_cascade.sh
#
# Usage (synthetic noise, clean sequences only):
#   DATA=/clean_seqs NOISE=camera ./scripts/train_cascade.sh
#
# Skip stages:
#   SKIP_STAGE1=1 SPATIAL_WEIGHTS=/path/best.pth ./scripts/train_cascade.sh
#   SKIP_STAGE2=1 ./scripts/train_cascade.sh
#   SKIP_STAGE3=1 ./scripts/train_cascade.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

# ---------------------------------------------------------------------------
# Data — use paired OR synthetic (set DATA= for synthetic, PAIRED_CLEAN= for paired)
# ---------------------------------------------------------------------------
PAIRED_CLEAN="${PAIRED_CLEAN:-}"
PAIRED_NOISY="${PAIRED_NOISY:-}"
VAL_CLEAN="${VAL_CLEAN:-}"
VAL_NOISY="${VAL_NOISY:-}"

DATA="${DATA:-}"           # clean sequences dir (synthetic noise path)
VAL_DATA="${VAL_DATA:-}"
NOISE="${NOISE:-camera}"   # camera | poisson-gaussian | gaussian
ISO_MIN="${ISO_MIN:-100}"
ISO_MAX="${ISO_MAX:-3200}"
COLOR_SPACE="${COLOR_SPACE:-log}"

# ---------------------------------------------------------------------------
# Model / architecture
# ---------------------------------------------------------------------------
SIZE="${SIZE:-exp048}"           # tiny | small | exp048 | standard | wide
NUM_FRAMES="${NUM_FRAMES:-3}"    # temporal window (odd, ≥3)
TEMPORAL_BASE="${TEMPORAL_BASE:-32}"   # base_channels for temporal_stage (exp053 = 32)
EXP_NAME="${EXP_NAME:-}"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
_sfx="${EXP_NAME:+_$EXP_NAME}"
STAGE1_OUTPUT="${STAGE1_OUTPUT:-checkpoints/cascade_spatial_${SIZE}${_sfx}}"
SPATIAL_WEIGHTS="${SPATIAL_WEIGHTS:-$STAGE1_OUTPUT/best.pth}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-checkpoints/cascade_${SIZE}_stage2${_sfx}}"
STAGE3_OUTPUT="${STAGE3_OUTPUT:-checkpoints/cascade_${SIZE}_stage3${_sfx}}"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
WORKERS="${WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PATCH_SIZE="${PATCH_SIZE:-128}"
PATCHES_PER_IMAGE="${PATCHES_PER_IMAGE:-64}"

SPATIAL_EPOCHS="${SPATIAL_EPOCHS:-150}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-100}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-50}"

SPATIAL_LR="${SPATIAL_LR:-1e-4}"
STAGE2_LR="${STAGE2_LR:-1e-4}"
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

SPATIAL_FRAMES_PER_SEQUENCE="${SPATIAL_FRAMES_PER_SEQUENCE:-5}"
SPATIAL_VAL_FRAMES_PER_SEQUENCE="${SPATIAL_VAL_FRAMES_PER_SEQUENCE:-3}"
WINDOWS_PER_SEQUENCE="${WINDOWS_PER_SEQUENCE:-4}"
VAL_WINDOWS_PER_SEQUENCE="${VAL_WINDOWS_PER_SEQUENCE:-3}"

SKIP_STAGE1="${SKIP_STAGE1:-0}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
SKIP_STAGE3="${SKIP_STAGE3:-0}"

# ---------------------------------------------------------------------------

cd "$TRAINING_DIR"

# Build data flags depending on paired vs synthetic mode
_data_flags() {
  if [[ -n "$PAIRED_CLEAN" ]]; then
    echo --paired-clean "$PAIRED_CLEAN" --paired-noisy "$PAIRED_NOISY"
  else
    echo --data "$DATA" --noise "$NOISE" --iso-min "$ISO_MIN" --iso-max "$ISO_MAX"
  fi
}

_val_flags() {
  if [[ -n "$PAIRED_CLEAN" && -n "$VAL_CLEAN" ]]; then
    echo --val-clean "$VAL_CLEAN" --val-noisy "$VAL_NOISY" --val-crop-mode grid --val-grid-size 3
  elif [[ -n "$DATA" && -n "$VAL_DATA" ]]; then
    echo --val-data "$VAL_DATA" --val-crop-mode grid --val-grid-size 3
  fi
}

echo "========================================================"
echo "  NAFNetCascade three-stage training  (exp053 architecture)"
echo "  Size:           $SIZE"
echo "  Num frames:     $NUM_FRAMES"
echo "  Temporal base:  $TEMPORAL_BASE"
if [[ -n "$PAIRED_CLEAN" ]]; then
  echo "  Data mode:      paired  ($PAIRED_CLEAN)"
else
  echo "  Data mode:      synthetic  noise=$NOISE  ISO $ISO_MIN–$ISO_MAX"
fi
echo "  Color space:    $COLOR_SPACE"
echo "  Experiment:     ${EXP_NAME:-<default>}"
echo "  Stage 1 output: $STAGE1_OUTPUT  (${SPATIAL_EPOCHS} epochs)$([ "$SKIP_STAGE1" == "1" ] && echo " [SKIP]")"
echo "  Stage 2 output: $STAGE2_OUTPUT  (${STAGE2_EPOCHS} epochs)$([ "$SKIP_STAGE2" == "1" ] && echo " [SKIP]")"
echo "  Stage 3 output: $STAGE3_OUTPUT  (${STAGE3_EPOCHS} epochs)$([ "$SKIP_STAGE3" == "1" ] && echo " [SKIP]")"
echo "========================================================"

# ---------------------------------------------------------------------------
# Stage 1 — Spatial denoiser  (identical to train.sh / train_synthetic.sh)
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
    --color-space "$COLOR_SPACE" \
    --loss "$SPATIAL_LOSS" \
    --scheduler "$SPATIAL_SCHEDULER" \
    --plateau-patience "$SPATIAL_PLATEAU_PATIENCE" \
    --lr "$SPATIAL_LR" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --patches-per-image "$PATCHES_PER_IMAGE" \
    $(_data_flags) \
    --frames-per-sequence "$SPATIAL_FRAMES_PER_SEQUENCE" \
    --random-spatial-frames \
    --val-frames-per-sequence "$SPATIAL_VAL_FRAMES_PER_SEQUENCE" \
    $(_val_flags) \
    --output "$STAGE1_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$SPATIAL_EPOCHS"
fi

# ---------------------------------------------------------------------------
# Stage 2 — Temporal stage only  (spatial_stage frozen, exp053 style)
#
# --model cascade      → NAFNetCascade
# --spatial-weights    → loaded into spatial_stage, weights matched by name/shape
# --freeze-spatial     → spatial_stage.eval() + requires_grad=False
# --temporal-base      → base_channels for the temporal NAFNet (32 = exp053 default)
# ---------------------------------------------------------------------------
if [[ "$SKIP_STAGE2" == "1" ]]; then
  echo ""
  echo "--- Stage 2 skipped (SKIP_STAGE2=1) ---"
else
  echo ""
  echo "--- Stage 2: cascade temporal stage (spatial frozen) ---"
  uv run python training.py \
    --model cascade \
    --size "$SIZE" \
    --num-frames "$NUM_FRAMES" \
    --temporal-base "$TEMPORAL_BASE" \
    --color-space "$COLOR_SPACE" \
    --loss "$STAGE2_LOSS" \
    --scheduler "$STAGE2_SCHEDULER" \
    --plateau-patience "$STAGE2_PLATEAU_PATIENCE" \
    --lr "$STAGE2_LR" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --patches-per-image "$PATCHES_PER_IMAGE" \
    $(_data_flags) \
    --random-temporal-windows \
    --windows-per-sequence "$WINDOWS_PER_SEQUENCE" \
    --val-windows-per-sequence "$VAL_WINDOWS_PER_SEQUENCE" \
    $(_val_flags) \
    --spatial-weights "$SPATIAL_WEIGHTS" \
    --freeze-spatial \
    --output "$STAGE2_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$STAGE2_EPOCHS"
fi

# ---------------------------------------------------------------------------
# Stage 3 — Joint fine-tune (all layers)
# ---------------------------------------------------------------------------
if [[ "$SKIP_STAGE3" == "1" ]]; then
  echo ""
  echo "--- Stage 3 skipped (SKIP_STAGE3=1) ---"
else
  echo ""
  echo "--- Stage 3: joint fine-tune ---"
  uv run python training.py \
    --model cascade \
    --size "$SIZE" \
    --num-frames "$NUM_FRAMES" \
    --temporal-base "$TEMPORAL_BASE" \
    --color-space "$COLOR_SPACE" \
    --loss "$STAGE3_LOSS" \
    --scheduler "$STAGE3_SCHEDULER" \
    --plateau-patience "$STAGE3_PLATEAU_PATIENCE" \
    --lr "$STAGE3_LR" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --patches-per-image "$PATCHES_PER_IMAGE" \
    $(_data_flags) \
    --random-temporal-windows \
    --windows-per-sequence "$WINDOWS_PER_SEQUENCE" \
    --val-windows-per-sequence "$VAL_WINDOWS_PER_SEQUENCE" \
    $(_val_flags) \
    --resume "$STAGE2_OUTPUT/best.pth" \
    --output "$STAGE3_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$STAGE3_EPOCHS"
fi

echo ""
echo "========================================================"
echo "  Done. Final weights: $STAGE3_OUTPUT/best.pth"
echo "========================================================"
