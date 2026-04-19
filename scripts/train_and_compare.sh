#!/usr/bin/env bash
# train_and_compare.sh — Full unattended pipeline: refiner (stages 4+5),
# cascade (stages 2+3), then three-way comparison.
#
# Assumes train.sh has already run and produced:
#   checkpoints/spatial_${SIZE}/best.pth         (stage 1 — shared)
#   checkpoints/temporal_${SIZE}_stage3/best.pth (stage 3 — A+B base for refiner)
#
# Pipeline:
#   Refiner stage 4  — load A+B base, freeze it, train refiner only
#   Refiner stage 5  — unfreeze base, joint polish at very low LR
#   Cascade stage 2  — load spatial weights, freeze spatial_stage, train temporal_stage
#   Cascade stage 3  — unfreeze all, joint fine-tune
#   Compare          — PSNR + flicker table: A+B vs refined vs cascade
#
# Skip any stage with SKIP_REFINER_4=1, SKIP_REFINER_5=1,
#                      SKIP_CASCADE_2=1, SKIP_CASCADE_3=1, SKIP_COMPARE=1
#
# Usage:
#   ./scripts/train_and_compare.sh
#
# Resume after an interrupted run:
#   SKIP_REFINER_4=1 SKIP_REFINER_5=1 SKIP_CASCADE_2=1 ./scripts/train_and_compare.sh

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
REFINER_BASE_CHANNELS="${REFINER_BASE_CHANNELS:-16}"
TEMPORAL_BASE="${TEMPORAL_BASE:-32}"

# ---------------------------------------------------------------------------
# Input weights (outputs of train.sh)
# ---------------------------------------------------------------------------
SPATIAL_WEIGHTS="${SPATIAL_WEIGHTS:-$ROOT_DIR/checkpoints/spatial_${SIZE}/best.pth}"
AB_WEIGHTS="${AB_WEIGHTS:-$ROOT_DIR/checkpoints/temporal_${SIZE}_stage2_v2/best.pth}"

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
REFINER_STAGE4_OUTPUT="${REFINER_STAGE4_OUTPUT:-$ROOT_DIR/checkpoints/refiner_${SIZE}_stage4}"
REFINER_STAGE5_OUTPUT="${REFINER_STAGE5_OUTPUT:-$ROOT_DIR/checkpoints/refiner_${SIZE}_stage5}"
CASCADE_STAGE2_OUTPUT="${CASCADE_STAGE2_OUTPUT:-$ROOT_DIR/checkpoints/cascade_${SIZE}_stage2}"
CASCADE_STAGE3_OUTPUT="${CASCADE_STAGE3_OUTPUT:-$ROOT_DIR/checkpoints/cascade_${SIZE}_stage3}"
COMPARE_OUTPUT="${COMPARE_OUTPUT:-$ROOT_DIR/comparisons/${SIZE}}"

# ---------------------------------------------------------------------------
# Shared data loader settings
# ---------------------------------------------------------------------------
WORKERS="${WORKERS:-12}"
BATCH_SIZE="${BATCH_SIZE:-4}"
PATCH_SIZE="${PATCH_SIZE:-128}"
WINDOWS_PER_SEQUENCE="${WINDOWS_PER_SEQUENCE:-6}"
VAL_WINDOWS_PER_SEQUENCE="${VAL_WINDOWS_PER_SEQUENCE:-1}"

# ---------------------------------------------------------------------------
# Refiner hyperparameters
# ---------------------------------------------------------------------------
REFINER_4_EPOCHS="${REFINER_4_EPOCHS:-120}"
REFINER_4_LR="${REFINER_4_LR:-1e-4}"
REFINER_4_SCHEDULER="${REFINER_4_SCHEDULER:-cosine}"

REFINER_5_EPOCHS="${REFINER_5_EPOCHS:-60}"
REFINER_5_LR="${REFINER_5_LR:-1e-5}"
REFINER_5_SCHEDULER="${REFINER_5_SCHEDULER:-cosine}"

# ---------------------------------------------------------------------------
# Cascade hyperparameters
# ---------------------------------------------------------------------------
CASCADE_2_EPOCHS="${CASCADE_2_EPOCHS:-250}"
CASCADE_2_LR="${CASCADE_2_LR:-1e-4}"
CASCADE_2_SCHEDULER="${CASCADE_2_SCHEDULER:-cosine}"

CASCADE_3_EPOCHS="${CASCADE_3_EPOCHS:-180}"
CASCADE_3_LR="${CASCADE_3_LR:-3e-5}"
CASCADE_3_SCHEDULER="${CASCADE_3_SCHEDULER:-cosine}"

# ---------------------------------------------------------------------------
# Skip flags
# ---------------------------------------------------------------------------
SKIP_REFINER_4="${SKIP_REFINER_4:-0}"
SKIP_REFINER_5="${SKIP_REFINER_5:-0}"
SKIP_CASCADE_2="${SKIP_CASCADE_2:-0}"
SKIP_CASCADE_3="${SKIP_CASCADE_3:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_skip_label() { [[ "$1" == "1" ]] && echo " [SKIP]" || echo ""; }
_stage_epochs() { echo "$1 epochs, LR=$2"; }

# ---------------------------------------------------------------------------

cd "$TRAINING_DIR"

echo "========================================================"
echo "  Full pipeline: refiner + cascade + compare"
echo "  Size: $SIZE   Frames: $NUM_FRAMES"
echo ""
echo "  Input weights:"
echo "    Spatial (stage 1): $SPATIAL_WEIGHTS"
echo "    A+B     (stage 3): $AB_WEIGHTS"
echo ""
echo "  Refiner:"
echo "    Stage 4 (refiner frozen base): $REFINER_STAGE4_OUTPUT  ($(_stage_epochs "$REFINER_4_EPOCHS" "$REFINER_4_LR"))$(_skip_label "$SKIP_REFINER_4")"
echo "    Stage 5 (joint polish):        $REFINER_STAGE5_OUTPUT  ($(_stage_epochs "$REFINER_5_EPOCHS" "$REFINER_5_LR"))$(_skip_label "$SKIP_REFINER_5")"
echo ""
echo "  Cascade:"
echo "    Stage 2 (temporal frozen):     $CASCADE_STAGE2_OUTPUT  ($(_stage_epochs "$CASCADE_2_EPOCHS" "$CASCADE_2_LR"))$(_skip_label "$SKIP_CASCADE_2")"
echo "    Stage 3 (joint fine-tune):     $CASCADE_STAGE3_OUTPUT  ($(_stage_epochs "$CASCADE_3_EPOCHS" "$CASCADE_3_LR"))$(_skip_label "$SKIP_CASCADE_3")"
echo ""
echo "  Compare output: $COMPARE_OUTPUT$(_skip_label "$SKIP_COMPARE")"
echo "========================================================"

# Guard: verify input weights exist before starting a long run
_check_weights() {
  local path="$1" label="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: $label not found: $path"
    echo "       Run train.sh first, or set the path explicitly."
    exit 1
  fi
}

[[ "$SKIP_REFINER_4" != "1" ]] && _check_weights "$AB_WEIGHTS"      "A+B weights (--base-weights)"
[[ "$SKIP_CASCADE_2" != "1" ]] && _check_weights "$SPATIAL_WEIGHTS" "Spatial weights (--spatial-weights)"

# ---------------------------------------------------------------------------
# Refiner stage 4 — refiner only, base frozen
# ---------------------------------------------------------------------------
if [[ "$SKIP_REFINER_4" == "1" ]]; then
  echo ""
  echo "--- Refiner stage 4 skipped ---"
else
  echo ""
  echo "================================================================"
  echo "  Refiner stage 4: train refiner (base frozen)"
  echo "================================================================"
  uv run python training.py \
    --model refined_temporal \
    --size "$SIZE" \
    --num-frames "$NUM_FRAMES" \
    --color-space log \
    --loss l1 \
    --scheduler "$REFINER_4_SCHEDULER" \
    --lr "$REFINER_4_LR" \
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
    --base-weights "$AB_WEIGHTS" \
    --freeze-base \
    --refiner-base-channels "$REFINER_BASE_CHANNELS" \
    --output "$REFINER_STAGE4_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$REFINER_4_EPOCHS"
fi

# ---------------------------------------------------------------------------
# Refiner stage 5 — joint polish, base unfrozen
# ---------------------------------------------------------------------------
if [[ "$SKIP_REFINER_5" == "1" ]]; then
  echo ""
  echo "--- Refiner stage 5 skipped ---"
else
  echo ""
  echo "================================================================"
  echo "  Refiner stage 5: joint polish (base unfrozen)"
  echo "================================================================"
  uv run python training.py \
    --model refined_temporal \
    --size "$SIZE" \
    --num-frames "$NUM_FRAMES" \
    --color-space log \
    --loss l1 \
    --scheduler "$REFINER_5_SCHEDULER" \
    --lr "$REFINER_5_LR" \
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
    --resume "$REFINER_STAGE4_OUTPUT/best.pth" \
    --output "$REFINER_STAGE5_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$REFINER_5_EPOCHS"
fi

# ---------------------------------------------------------------------------
# Cascade stage 2 — temporal_stage only, spatial_stage frozen
# ---------------------------------------------------------------------------
if [[ "$SKIP_CASCADE_2" == "1" ]]; then
  echo ""
  echo "--- Cascade stage 2 skipped ---"
else
  echo ""
  echo "================================================================"
  echo "  Cascade stage 2: train temporal_stage (spatial frozen)"
  echo "================================================================"
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
# Cascade stage 3 — joint fine-tune
# ---------------------------------------------------------------------------
if [[ "$SKIP_CASCADE_3" == "1" ]]; then
  echo ""
  echo "--- Cascade stage 3 skipped ---"
else
  echo ""
  echo "================================================================"
  echo "  Cascade stage 3: joint fine-tune"
  echo "================================================================"
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

# ---------------------------------------------------------------------------
# Compare — PSNR + flicker table
# ---------------------------------------------------------------------------
if [[ "$SKIP_COMPARE" == "1" ]]; then
  echo ""
  echo "--- Comparison skipped ---"
else
  echo ""
  echo "================================================================"
  echo "  Comparison: A+B vs refined vs cascade"
  echo "================================================================"

  COMPARE_ARGS=(
    --val-clean "$VAL_CLEAN"
    --val-noisy "$VAL_NOISY"
    --num-frames "$NUM_FRAMES"
    --num-clips 50
    --patch-size 256
    --workers "$WORKERS"
    --out "$COMPARE_OUTPUT"
  )

  # Include whichever checkpoints exist
  [[ -f "$AB_WEIGHTS" ]]                             && COMPARE_ARGS+=(--ab-weights      "$AB_WEIGHTS")
  [[ -f "$REFINER_STAGE5_OUTPUT/best.pth" ]]         && COMPARE_ARGS+=(--refined-weights "$REFINER_STAGE5_OUTPUT/best.pth")
  [[ -f "$CASCADE_STAGE3_OUTPUT/best.pth" ]]         && COMPARE_ARGS+=(--cascade-weights "$CASCADE_STAGE3_OUTPUT/best.pth")

  uv run python compare_models.py "${COMPARE_ARGS[@]}"
fi

echo ""
echo "========================================================"
echo "  All done."
echo "  Refined:  $REFINER_STAGE5_OUTPUT/best.pth"
echo "  Cascade:  $CASCADE_STAGE3_OUTPUT/best.pth"
echo "  Results:  $COMPARE_OUTPUT/"
echo "========================================================"
