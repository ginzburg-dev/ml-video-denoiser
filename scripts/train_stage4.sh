#!/usr/bin/env bash
# train_stage4.sh — Stage-4 post-decode refiner + optional stage-5 joint polish.
#
# Stage 4  Refiner-only    Load stage-3 A+B weights, freeze base, train refiner
# Stage 5  Joint polish    Unfreeze base, fine-tune everything at very low LR
#
# Usage:
#   ./scripts/train_stage4.sh
#
# Override any variable on the command line:
#   BASE_WEIGHTS=/path/to/stage3/best.pth ./scripts/train_stage4.sh
#   SKIP_STAGE5=1 ./scripts/train_stage4.sh   # refiner only, no joint polish

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

# ---------------------------------------------------------------------------
# Data paths (reuse the same defaults as train.sh)
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
REFINER_BASE_CHANNELS="${REFINER_BASE_CHANNELS:-16}"

# ---------------------------------------------------------------------------
# Base model — product of train.sh stage 3
# ---------------------------------------------------------------------------
BASE_WEIGHTS="${BASE_WEIGHTS:-$ROOT_DIR/checkpoints/temporal_${SIZE}_stage3/best.pth}"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
STAGE4_OUTPUT="${STAGE4_OUTPUT:-$ROOT_DIR/checkpoints/refiner_${SIZE}_stage4}"
STAGE5_OUTPUT="${STAGE5_OUTPUT:-$ROOT_DIR/checkpoints/refiner_${SIZE}_stage5}"

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
WORKERS="${WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PATCH_SIZE="${PATCH_SIZE:-128}"
WINDOWS_PER_SEQUENCE="${WINDOWS_PER_SEQUENCE:-6}"
VAL_WINDOWS_PER_SEQUENCE="${VAL_WINDOWS_PER_SEQUENCE:-1}"

# Stage 4: refiner is ~500K fresh params — can run hot
STAGE4_EPOCHS="${STAGE4_EPOCHS:-120}"
STAGE4_LR="${STAGE4_LR:-1e-4}"
STAGE4_SCHEDULER="${STAGE4_SCHEDULER:-cosine}"

# Stage 5: joint polish — gentle on the unfrozen base
STAGE5_EPOCHS="${STAGE5_EPOCHS:-60}"
STAGE5_LR="${STAGE5_LR:-1e-5}"
STAGE5_SCHEDULER="${STAGE5_SCHEDULER:-cosine}"

SKIP_STAGE4="${SKIP_STAGE4:-0}"
SKIP_STAGE5="${SKIP_STAGE5:-0}"

# ---------------------------------------------------------------------------

cd "$TRAINING_DIR"

echo "========================================================"
echo "  Stage-4 refiner training (A+B + refiner)"
echo "  Base weights:   $BASE_WEIGHTS"
echo "  Stage 4 output: $STAGE4_OUTPUT  (${STAGE4_EPOCHS} epochs, LR=${STAGE4_LR})$([ "$SKIP_STAGE4" == "1" ] && echo " [SKIP]")"
echo "  Stage 5 output: $STAGE5_OUTPUT  (${STAGE5_EPOCHS} epochs, LR=${STAGE5_LR})$([ "$SKIP_STAGE5" == "1" ] && echo " [SKIP]")"
echo "========================================================"

# ---------------------------------------------------------------------------
# Stage 4 — Refiner only (base frozen)
# ---------------------------------------------------------------------------
if [[ "$SKIP_STAGE4" == "1" ]]; then
  echo ""
  echo "--- Stage 4 skipped (SKIP_STAGE4=1) ---"
else
  echo ""
  echo "--- Stage 4: train refiner (base frozen) ---"
  uv run python training.py \
    --model refined_temporal \
    --size "$SIZE" \
    --num-frames "$NUM_FRAMES" \
    --color-space log \
    --loss l1 \
    --scheduler "$STAGE4_SCHEDULER" \
    --lr "$STAGE4_LR" \
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
    --base-weights "$BASE_WEIGHTS" \
    --freeze-base \
    --refiner-base-channels "$REFINER_BASE_CHANNELS" \
    --output "$STAGE4_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$STAGE4_EPOCHS"
fi

# ---------------------------------------------------------------------------
# Stage 5 — Joint polish (base unfrozen, very low LR)
# ---------------------------------------------------------------------------
if [[ "$SKIP_STAGE5" == "1" ]]; then
  echo ""
  echo "--- Stage 5 skipped (SKIP_STAGE5=1) ---"
else
  echo ""
  echo "--- Stage 5: joint polish (base unfrozen) ---"
  uv run python training.py \
    --model refined_temporal \
    --size "$SIZE" \
    --num-frames "$NUM_FRAMES" \
    --color-space log \
    --loss l1 \
    --scheduler "$STAGE5_SCHEDULER" \
    --lr "$STAGE5_LR" \
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
    --refiner-base-channels "$REFINER_BASE_CHANNELS" \
    --resume "$STAGE4_OUTPUT/best.pth" \
    --output "$STAGE5_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$STAGE5_EPOCHS"
fi

echo ""
echo "========================================================"
echo "  Done. Final: $STAGE5_OUTPUT/best.pth"
echo "========================================================"
