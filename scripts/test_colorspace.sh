#!/usr/bin/env bash
# test_colorspace.sh — Quick A/B test: log vs linear color space.
#
# For each color space trains:
#   Stage 1  Spatial           (short)
#   Stage 2  Cascade temporal  (short, spatial frozen)
#
# Checkpoints go under training/checkpoints/cs_test_{log|linear}_*
# so they never touch production checkpoints.
#
# Usage:
#   ./scripts/test_colorspace.sh
#
# Override epochs for an even quicker smoke test:
#   SPATIAL_EPOCHS=20 TEMPORAL_EPOCHS=20 ./scripts/test_colorspace.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

PAIRED_CLEAN="${PAIRED_CLEAN:-$HOME/data/tgb_train/TGB_training/train_clean}"
PAIRED_NOISY="${PAIRED_NOISY:-$HOME/data/tgb_train/TGB_training/train_noisy}"
VAL_CLEAN="${VAL_CLEAN:-$HOME/data/tgb_train/TGB_training/val_clean}"
VAL_NOISY="${VAL_NOISY:-$HOME/data/tgb_train/TGB_training/val_noisy}"

SIZE="${SIZE:-exp048}"
NUM_FRAMES="${NUM_FRAMES:-3}"
EXP_NAME="${EXP_NAME:-}"        # optional experiment tag, e.g. EXP_NAME=run01

SPATIAL_EPOCHS="${SPATIAL_EPOCHS:-30}"
TEMPORAL_EPOCHS="${TEMPORAL_EPOCHS:-30}"
LR_SPATIAL="${LR_SPATIAL:-1e-4}"
LR_TEMPORAL="${LR_TEMPORAL:-1e-4}"

WORKERS="${WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SPATIAL_PATCH_SIZE="${SPATIAL_PATCH_SIZE:-128}"
TEMPORAL_PATCH_SIZE="${TEMPORAL_PATCH_SIZE:-96}"

cd "$TRAINING_DIR"

echo "========================================================"
echo "  Color space test: linear"
echo "  Spatial:  ${SPATIAL_EPOCHS} epochs"
echo "  Temporal: ${TEMPORAL_EPOCHS} epochs"
echo "  Results:  checkpoints/cs_test_linear_*"
echo "========================================================"

for CS in linear; do
  _sfx="${EXP_NAME:+_$EXP_NAME}"
  SPATIAL_OUT="$ROOT_DIR/training/checkpoints/cs_test_${CS}_spatial${_sfx}"
  TEMPORAL_OUT="$ROOT_DIR/training/checkpoints/cs_test_${CS}_cascade${_sfx}"

  echo ""
  echo "════════════════════════════════"
  echo "  Color space: $CS"
  echo "════════════════════════════════"

  echo ""
  echo "--- [$CS] Stage 1: spatial ---"
  uv run python training.py \
    --model spatial \
    --size "$SIZE" \
    --color-space "$CS" \
    --loss l1 \
    --scheduler none \
    --lr "$LR_SPATIAL" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$SPATIAL_PATCH_SIZE" \
    --paired-clean "$PAIRED_CLEAN" \
    --paired-noisy "$PAIRED_NOISY" \
    --val-clean "$VAL_CLEAN" \
    --val-noisy "$VAL_NOISY" \
    --frames-per-sequence 10 \
    --val-frames-per-sequence 3 \
    --val-crop-mode grid \
    --val-grid-size 3 \
    --output "$SPATIAL_OUT" \
    --workers "$WORKERS" \
    --epochs "$SPATIAL_EPOCHS"

  echo ""
  echo "--- [$CS] Stage 2: cascade temporal (spatial frozen) ---"
  uv run python training.py \
    --model cascade \
    --size "$SIZE" \
    --num-frames "$NUM_FRAMES" \
    --color-space "$CS" \
    --loss l1 \
    --scheduler none \
    --lr "$LR_TEMPORAL" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$TEMPORAL_PATCH_SIZE" \
    --paired-clean "$PAIRED_CLEAN" \
    --paired-noisy "$PAIRED_NOISY" \
    --val-clean "$VAL_CLEAN" \
    --val-noisy "$VAL_NOISY" \
    --random-temporal-windows \
    --windows-per-sequence 6 \
    --val-windows-per-sequence 1 \
    --val-crop-mode center \
    --spatial-weights "$SPATIAL_OUT/best.pth" \
    --freeze-spatial \
    --output "$TEMPORAL_OUT" \
    --workers "$WORKERS" \
    --epochs "$TEMPORAL_EPOCHS"

done

echo ""
echo "========================================================"
echo "  Done. Compare in TensorBoard:"
echo "  uv run tensorboard --logdir \\"
echo "    cs_test_linear_spatial:$ROOT_DIR/training/checkpoints/cs_test_linear_spatial/runs,\\"
echo "    cs_test_linear_cascade:$ROOT_DIR/training/checkpoints/cs_test_linear_cascade/runs"
echo "========================================================"
