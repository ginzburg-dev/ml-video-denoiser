#!/usr/bin/env bash
# train_cascade.sh — Two-stage cascade training (future experiment).
#
# Stage 1  Spatial      Reuse from train.sh (skip if already done)
# Stage 2  Temporal     Load spatial weights, freeze spatial_stage, train temporal_stage
# Stage 3  Fine-tune    Unfreeze all, joint fine-tune at lower LR
#
# Usage:
#   ./scripts/train_cascade.sh
#
# Override any variable on the command line:
#   SKIP_STAGE1=1 SPATIAL_WEIGHTS=/path/to/spatial/best.pth ./scripts/train_cascade.sh
#
# Skip individual stages:
#   SKIP_STAGE1=1 SPATIAL_WEIGHTS=/existing/best.pth ./scripts/train_cascade.sh
#   SKIP_STAGE2=1 ./scripts/train_cascade.sh   # requires STAGE2_OUTPUT to exist
#   SKIP_STAGE3=1 ./scripts/train_cascade.sh   # stop after stage 2

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
# Architecture
# ---------------------------------------------------------------------------
SIZE="${SIZE:-exp048}"
NUM_FRAMES="${NUM_FRAMES:-3}"
TEMPORAL_BASE="${TEMPORAL_BASE:-32}"   # base_channels for cascade temporal_stage

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
STAGE1_OUTPUT="${STAGE1_OUTPUT:-$ROOT_DIR/checkpoints/spatial_${SIZE}}"
SPATIAL_WEIGHTS="${SPATIAL_WEIGHTS:-$STAGE1_OUTPUT/best.pth}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-$ROOT_DIR/checkpoints/cascade_${SIZE}_stage2}"
STAGE3_OUTPUT="${STAGE3_OUTPUT:-$ROOT_DIR/checkpoints/cascade_${SIZE}_stage3}"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
WORKERS="${WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PATCH_SIZE="${PATCH_SIZE:-128}"

SPATIAL_EPOCHS="${SPATIAL_EPOCHS:-150}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-250}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-180}"

SPATIAL_LR="${SPATIAL_LR:-1e-4}"
STAGE2_LR="${STAGE2_LR:-1e-4}"    # temporal_stage is fresh — can run hot
STAGE3_LR="${STAGE3_LR:-3e-5}"

SPATIAL_LOSS="${SPATIAL_LOSS:-l1}"
STAGE2_LOSS="${STAGE2_LOSS:-l1}"
STAGE3_LOSS="${STAGE3_LOSS:-l1}"

SPATIAL_SCHEDULER="${SPATIAL_SCHEDULER:-cosine}"
STAGE2_SCHEDULER="${STAGE2_SCHEDULER:-cosine}"
STAGE3_SCHEDULER="${STAGE3_SCHEDULER:-cosine}"

SPATIAL_FRAMES_PER_SEQUENCE="${SPATIAL_FRAMES_PER_SEQUENCE:-10}"
SPATIAL_VAL_FRAMES_PER_SEQUENCE="${SPATIAL_VAL_FRAMES_PER_SEQUENCE:-3}"
WINDOWS_PER_SEQUENCE="${WINDOWS_PER_SEQUENCE:-6}"
VAL_WINDOWS_PER_SEQUENCE="${VAL_WINDOWS_PER_SEQUENCE:-1}"

SKIP_STAGE1="${SKIP_STAGE1:-0}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
SKIP_STAGE3="${SKIP_STAGE3:-0}"

# ---------------------------------------------------------------------------

cd "$TRAINING_DIR"

echo "========================================================"
echo "  NAFNetCascade three-stage training"
echo "  Size:          $SIZE"
echo "  Num frames:    $NUM_FRAMES"
echo "  Temporal base: $TEMPORAL_BASE"
echo "  Stage 1 output:  $STAGE1_OUTPUT  (${SPATIAL_EPOCHS} epochs)$([ "$SKIP_STAGE1" == "1" ] && echo " [SKIP]")"
echo "  Stage 2 output:  $STAGE2_OUTPUT  (${STAGE2_EPOCHS} epochs)$([ "$SKIP_STAGE2" == "1" ] && echo " [SKIP]")"
echo "  Stage 3 output:  $STAGE3_OUTPUT  (${STAGE3_EPOCHS} epochs)$([ "$SKIP_STAGE3" == "1" ] && echo " [SKIP]")"
echo "========================================================"

# ---------------------------------------------------------------------------
# Stage 1 — Spatial denoiser (identical to train.sh stage 1)
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
    --loss "$SPATIAL_LOSS" \
    --scheduler "$SPATIAL_SCHEDULER" \
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
# Stage 2 — Temporal stage only (spatial_stage frozen)
# ---------------------------------------------------------------------------
if [[ "$SKIP_STAGE2" == "1" ]]; then
  echo ""
  echo "--- Stage 2 skipped (SKIP_STAGE2=1) ---"
else
  echo ""
  echo "--- Stage 2: temporal stage (spatial frozen) ---"
  uv run python training.py \
    --model cascade \
    --size "$SIZE" \
    --num-frames "$NUM_FRAMES" \
    --temporal-base "$TEMPORAL_BASE" \
    --color-space log \
    --loss "$STAGE2_LOSS" \
    --scheduler "$STAGE2_SCHEDULER" \
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
# Stage 3 — Joint fine-tune (all layers, lower LR)
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
    --color-space log \
    --loss "$STAGE3_LOSS" \
    --scheduler "$STAGE3_SCHEDULER" \
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
