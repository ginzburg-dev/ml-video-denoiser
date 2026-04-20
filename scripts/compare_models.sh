#!/usr/bin/env bash
# compare_models.sh — PSNR + flicker comparison: A+B vs cascade.
#
# Reads whichever --*-weights are available and prints a table.
#
# Usage:
#   ./scripts/compare_models.sh
#
# Override weights:
#   AB_WEIGHTS=/path/to/ab.pth CASCADE_WEIGHTS=/path/to/cascade.pth \
#     ./scripts/compare_models.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
VAL_CLEAN="${VAL_CLEAN:-$HOME/data/tgb_train/TGB_training/val_clean}"
VAL_NOISY="${VAL_NOISY:-$HOME/data/tgb_train/TGB_training/val_noisy}"

# ---------------------------------------------------------------------------
# Weights — point at whichever checkpoints you want to compare
# ---------------------------------------------------------------------------
SIZE="${SIZE:-exp048}"
NUM_FRAMES="${NUM_FRAMES:-3}"

AB_WEIGHTS="${AB_WEIGHTS:-$ROOT_DIR/training/checkpoints/temporal_${SIZE}_stage3/best.pth}"
CASCADE_WEIGHTS="${CASCADE_WEIGHTS:-$ROOT_DIR/training/checkpoints/cascade_${SIZE}_stage3/best.pth}"

# ---------------------------------------------------------------------------
NUM_CLIPS="${NUM_CLIPS:-50}"
PATCH_SIZE="${PATCH_SIZE:-256}"
WORKERS="${WORKERS:-12}"

# ---------------------------------------------------------------------------

cd "$TRAINING_DIR"

echo "========================================================"
echo "  Model comparison"
echo "  A+B:     $AB_WEIGHTS"
echo "  Cascade: $CASCADE_WEIGHTS"
echo "========================================================"

ARGS=(
  --val-clean "$VAL_CLEAN"
  --val-noisy "$VAL_NOISY"
  --num-frames "$NUM_FRAMES"
  --num-clips "$NUM_CLIPS"
  --patch-size "$PATCH_SIZE"
  --workers "$WORKERS"
)

[[ -f "$AB_WEIGHTS" ]]      && ARGS+=(--ab-weights      "$AB_WEIGHTS")      || echo "WARNING: A+B weights not found, skipping"
[[ -f "$CASCADE_WEIGHTS" ]] && ARGS+=(--cascade-weights "$CASCADE_WEIGHTS") || echo "WARNING: cascade weights not found, skipping"

uv run python compare_models.py "${ARGS[@]}"
