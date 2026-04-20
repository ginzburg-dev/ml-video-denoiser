#!/usr/bin/env bash
# train_cascade_temporal.sh — Two-stage cascade temporal training.
#
# Assumes train.sh stage 1 has already produced spatial weights.
#
# Stage 2  Freeze spatial_stage, train temporal_stage  (plateau)
# Stage 3  Unfreeze all, joint fine-tune               (cosine)
#
# Usage:
#   ./scripts/train_cascade_temporal.sh
#
# Resume after interruption:
#   SKIP_CASCADE_2=1 ./scripts/train_cascade_temporal.sh

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
TEMPORAL_BASE="${TEMPORAL_BASE:-64}"   # matches exp_053

# ---------------------------------------------------------------------------
# Weights + outputs
# ---------------------------------------------------------------------------
SPATIAL_WEIGHTS="${SPATIAL_WEIGHTS:-$ROOT_DIR/training/checkpoints/spatial_${SIZE}/best.pth}"
CASCADE_STAGE2_OUTPUT="${CASCADE_STAGE2_OUTPUT:-$ROOT_DIR/training/checkpoints/cascade_${SIZE}_stage2}"
CASCADE_STAGE3_OUTPUT="${CASCADE_STAGE3_OUTPUT:-$ROOT_DIR/training/checkpoints/cascade_${SIZE}_stage3}"

# ---------------------------------------------------------------------------
# Hyperparameters
# Stage 2: plateau — fresh 6M temporal_stage params need sustained hot LR
# Stage 3: cosine  — polish phase, guaranteed cool-down tail
# ---------------------------------------------------------------------------
WORKERS="${WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PATCH_SIZE="${PATCH_SIZE:-128}"
WINDOWS_PER_SEQUENCE="${WINDOWS_PER_SEQUENCE:-6}"
VAL_WINDOWS_PER_SEQUENCE="${VAL_WINDOWS_PER_SEQUENCE:-1}"

CASCADE_2_EPOCHS="${CASCADE_2_EPOCHS:-250}"
CASCADE_2_LR="${CASCADE_2_LR:-1e-4}"
CASCADE_2_SCHEDULER="${CASCADE_2_SCHEDULER:-plateau}"

CASCADE_3_EPOCHS="${CASCADE_3_EPOCHS:-100}"
CASCADE_3_LR="${CASCADE_3_LR:-3e-5}"
CASCADE_3_SCHEDULER="${CASCADE_3_SCHEDULER:-cosine}"

SKIP_CASCADE_2="${SKIP_CASCADE_2:-0}"
SKIP_CASCADE_3="${SKIP_CASCADE_3:-0}"

# ---------------------------------------------------------------------------
_skip() { [[ "$1" == "1" ]] && echo " [SKIP]" || echo ""; }

cd "$TRAINING_DIR"

echo "========================================================"
echo "  Cascade temporal training"
echo "  Spatial: $SPATIAL_WEIGHTS"
echo "  Stage 2: $CASCADE_STAGE2_OUTPUT  (${CASCADE_2_EPOCHS} ep, LR=${CASCADE_2_LR}, ${CASCADE_2_SCHEDULER})$(_skip "$SKIP_CASCADE_2")"
echo "  Stage 3: $CASCADE_STAGE3_OUTPUT  (${CASCADE_3_EPOCHS} ep, LR=${CASCADE_3_LR}, ${CASCADE_3_SCHEDULER})$(_skip "$SKIP_CASCADE_3")"
echo "========================================================"

if [[ "$SKIP_CASCADE_2" != "1" && ! -f "$SPATIAL_WEIGHTS" ]]; then
  echo "ERROR: spatial weights not found: $SPATIAL_WEIGHTS"; exit 1
fi

# ---------------------------------------------------------------------------
# Stage 2 — temporal_stage only, spatial frozen
# ---------------------------------------------------------------------------
if [[ "$SKIP_CASCADE_2" == "1" ]]; then
  echo ""; echo "--- Stage 2 skipped ---"
else
  echo ""; echo "--- Stage 2: temporal_stage (spatial frozen, ${CASCADE_2_SCHEDULER}) ---"
  uv run python training.py \
    --model cascade \
    --size "$SIZE" \
    --num-frames "$NUM_FRAMES" \
    --temporal-base "$TEMPORAL_BASE" \
    --color-space log \
    --loss l1 \
    --scheduler "$CASCADE_2_SCHEDULER" \
    --lr "$CASCADE_2_LR" \
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
    --output "$CASCADE_STAGE2_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$CASCADE_2_EPOCHS"
fi

# ---------------------------------------------------------------------------
# Stage 3 — joint fine-tune
# ---------------------------------------------------------------------------
if [[ "$SKIP_CASCADE_3" == "1" ]]; then
  echo ""; echo "--- Stage 3 skipped ---"
else
  echo ""; echo "--- Stage 3: joint fine-tune (${CASCADE_3_SCHEDULER}) ---"
  uv run python training.py \
    --model cascade \
    --size "$SIZE" \
    --num-frames "$NUM_FRAMES" \
    --temporal-base "$TEMPORAL_BASE" \
    --color-space log \
    --loss l1 \
    --scheduler "$CASCADE_3_SCHEDULER" \
    --lr "$CASCADE_3_LR" \
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
    --resume "$CASCADE_STAGE2_OUTPUT/best.pth" \
    --output "$CASCADE_STAGE3_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$CASCADE_3_EPOCHS"
fi

echo ""
echo "========================================================"
echo "  Done. Final: $CASCADE_STAGE3_OUTPUT/best.pth"
echo "========================================================"
