#!/usr/bin/env bash
# train.sh — Full three-stage NAFNet training pipeline.
#
# Stage 1  Spatial      NAFNet spatial denoiser
# Stage 2  Temporal     Load stage-1 weights, freeze spatial, train temporal_mix only
# Stage 3  Fine-tune    Unfreeze all, joint fine-tune at half LR
#
# Usage:
#   ./scripts/train.sh
#
# Override any variable on the command line, e.g.:
#   WORKERS=4 SIZE=small ./scripts/train.sh
#   PAIRED_CLEAN=/my/clean PAIRED_NOISY=/my/noisy ./scripts/train.sh
#
# Skip individual stages:
#   SKIP_STAGE1=1 SPATIAL_WEIGHTS=/path/to/spatial/best.pth ./scripts/train.sh
#   SKIP_STAGE2=1 ./scripts/train.sh   # requires STAGE2_OUTPUT to already exist
#   SKIP_STAGE3=1 ./scripts/train.sh   # stop after stage 2

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
PAIRED_CLEAN="${PAIRED_CLEAN:-$HOME/data/tgb_train/TGB_training/train_clean}"
PAIRED_NOISY="${PAIRED_NOISY:-$HOME/data/tgb_train/TGB_training/train_noisy}"
VAL_CLEAN="${VAL_CLEAN:-$HOME/data/tgb_train/TGB_training/val_clean}"
VAL_NOISY="${VAL_NOISY:-$HOME/data/tgb_train/TGB_training/val_noisy}"

# ---------------------------------------------------------------------------
# Model / architecture
# ---------------------------------------------------------------------------
SIZE="${SIZE:-exp048}"          # tiny | small | exp048 | standard | wide
NUM_FRAMES="${NUM_FRAMES:-3}"   # temporal window (3 = 1 past + ref + 1 future)

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
STAGE1_OUTPUT="${STAGE1_OUTPUT:-checkpoints/spatial_${SIZE}}"
SPATIAL_WEIGHTS="${SPATIAL_WEIGHTS:-$STAGE1_OUTPUT/best.pth}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-checkpoints/temporal_${SIZE}_stage2}"
STAGE3_OUTPUT="${STAGE3_OUTPUT:-checkpoints/temporal_${SIZE}_stage3}"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
WORKERS="${WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-2}"
PATCH_SIZE="${PATCH_SIZE:-128}"

SPATIAL_EPOCHS="${SPATIAL_EPOCHS:-150}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-60}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-100}"

SPATIAL_LR="${SPATIAL_LR:-1e-4}"
STAGE2_LR="${STAGE2_LR:-1e-4}"
STAGE3_LR="${STAGE3_LR:-5e-5}"

SPATIAL_FRAMES_PER_SEQUENCE="${SPATIAL_FRAMES_PER_SEQUENCE:-10}"
SPATIAL_VAL_FRAMES_PER_SEQUENCE="${SPATIAL_VAL_FRAMES_PER_SEQUENCE:-3}"
WINDOWS_PER_SEQUENCE="${WINDOWS_PER_SEQUENCE:-3}"
VAL_WINDOWS_PER_SEQUENCE="${VAL_WINDOWS_PER_SEQUENCE:-1}"

SKIP_STAGE1="${SKIP_STAGE1:-0}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
SKIP_STAGE3="${SKIP_STAGE3:-0}"

# ---------------------------------------------------------------------------

cd "$TRAINING_DIR"

echo "========================================================"
echo "  NAFNet three-stage training"
echo "  Size:       $SIZE"
echo "  Num frames: $NUM_FRAMES"
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
    --color-space log \
    --loss l1 \
    --scheduler plateau \
    --lr "$SPATIAL_LR" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --paired-clean "$PAIRED_CLEAN" \
    --paired-noisy "$PAIRED_NOISY" \
    --val-clean "$VAL_CLEAN" \
    --val-noisy "$VAL_NOISY" \
    --frames-per-sequence "$SPATIAL_FRAMES_PER_SEQUENCE" \
    --val-frames-per-sequence "$SPATIAL_VAL_FRAMES_PER_SEQUENCE" \
    --val-crop-mode grid \
    --val-grid-size 3 \
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
  --color-space log \
  --loss l1 \
  --scheduler plateau \
  --lr "$STAGE2_LR" \
  --batch-size "$BATCH_SIZE" \
  --patch-size "$PATCH_SIZE" \
  --paired-clean "$PAIRED_CLEAN" \
  --paired-noisy "$PAIRED_NOISY" \
  --val-clean "$VAL_CLEAN" \
  --val-noisy "$VAL_NOISY" \
  --random-temporal-windows \
  --windows-per-sequence "$WINDOWS_PER_SEQUENCE" \
  --val-windows-per-sequence "$VAL_WINDOWS_PER_SEQUENCE" \
  --val-crop-mode center \
  --spatial-weights "$SPATIAL_WEIGHTS" \
  --freeze-spatial \
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
  --color-space log \
  --loss l1 \
  --scheduler plateau \
  --lr "$STAGE3_LR" \
  --batch-size "$BATCH_SIZE" \
  --patch-size "$PATCH_SIZE" \
  --paired-clean "$PAIRED_CLEAN" \
  --paired-noisy "$PAIRED_NOISY" \
  --val-clean "$VAL_CLEAN" \
  --val-noisy "$VAL_NOISY" \
  --random-temporal-windows \
  --windows-per-sequence "$WINDOWS_PER_SEQUENCE" \
  --val-windows-per-sequence "$VAL_WINDOWS_PER_SEQUENCE" \
  --val-crop-mode center \
  --resume "$STAGE2_OUTPUT/best.pth" \
  --output "$STAGE3_OUTPUT" \
  --workers "$WORKERS" \
  --epochs "$STAGE3_EPOCHS"
fi

echo ""
echo "========================================================"
echo "  Done. Final weights: $STAGE3_OUTPUT/best.pth"
echo "========================================================"
