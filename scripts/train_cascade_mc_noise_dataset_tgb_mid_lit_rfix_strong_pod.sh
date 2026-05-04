#!/usr/bin/env bash
# train_cascade_mc_noise_dataset_tgb_mid_lit_rfix_strong_pod.sh — Three-stage NAFNetCascade training
# with a stronger lit-calibrated, R-channel-corrected MCNoise bank.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_CLEAN="${DATA_CLEAN:-/workspace/data/TGB_training/train_clean_lit}"
VAL_CLEAN="${VAL_CLEAN:-/workspace/data/TGB_training/val_clean}"
VAL_NOISY="${VAL_NOISY:-/workspace/data/TGB_training/val_noisy}"
NOISE_MC_CONFIG="${NOISE_MC_CONFIG:-$ROOT_DIR/nuke/mc_noise_presets_tgb_mid_lit_rfix_strong.json}"

# ---------------------------------------------------------------------------
# Model / architecture
# ---------------------------------------------------------------------------
SIZE="${SIZE:-exp048}"
NUM_FRAMES="${NUM_FRAMES:-3}"
TEMPORAL_BASE="${TEMPORAL_BASE:-32}"
EXP_NAME="${EXP_NAME:-mc_noise_dataset_tgb_mid_lit_rfix_strong}"

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
PATCHES_PER_IMAGE="${PATCHES_PER_IMAGE:-256}"

SPATIAL_EPOCHS="${SPATIAL_EPOCHS:-50}"
STAGE2_EPOCHS="${STAGE2_EPOCHS:-40}"
STAGE3_EPOCHS="${STAGE3_EPOCHS:-50}"

SPATIAL_LR="${SPATIAL_LR:-1e-4}"
STAGE2_LR="${STAGE2_LR:-1e-4}"
STAGE3_LR="${STAGE3_LR:-2e-5}"

NOISE="${NOISE:-mc}"
SPATIAL_LOSS="${SPATIAL_LOSS:-l1}"
STAGE2_LOSS="${STAGE2_LOSS:-l1}"
STAGE3_LOSS="${STAGE3_LOSS:-l1}"

COLOR_SPACE="${COLOR_SPACE:-linear}"

SPATIAL_SCHEDULER="${SPATIAL_SCHEDULER:-none}"
SPATIAL_PLATEAU_PATIENCE="${SPATIAL_PLATEAU_PATIENCE:-20}"
STAGE2_SCHEDULER="${STAGE2_SCHEDULER:-none}"
STAGE2_PLATEAU_PATIENCE="${STAGE2_PLATEAU_PATIENCE:-20}"
STAGE3_SCHEDULER="${STAGE3_SCHEDULER:-none}"
STAGE3_PLATEAU_PATIENCE="${STAGE3_PLATEAU_PATIENCE:-20}"

SPATIAL_FRAMES_PER_SEQUENCE="${SPATIAL_FRAMES_PER_SEQUENCE:-5}"
SPATIAL_VAL_FRAMES_PER_SEQUENCE="${SPATIAL_VAL_FRAMES_PER_SEQUENCE:-3}"
WINDOWS_PER_SEQUENCE="${WINDOWS_PER_SEQUENCE:-4}"
VAL_WINDOWS_PER_SEQUENCE="${VAL_WINDOWS_PER_SEQUENCE:-3}"

SKIP_STAGE1="${SKIP_STAGE1:-0}"
SKIP_STAGE2="${SKIP_STAGE2:-0}"
SKIP_STAGE3="${SKIP_STAGE3:-1}"
RESUME="${RESUME:-0}"

# ---------------------------------------------------------------------------

cd "$TRAINING_DIR"

if [[ "$NOISE" != "mc" ]]; then
  echo "ERROR: this script is intended for MCNoise training only. Set NOISE=mc." >&2
  exit 1
fi

if [[ ! -f "$NOISE_MC_CONFIG" ]]; then
  echo "ERROR: MCNoise preset bank not found: $NOISE_MC_CONFIG" >&2
  exit 1
fi

_resume_flag() {
  local ckpt="$1/last.pth"
  [[ "$RESUME" == "1" && -f "$ckpt" ]] && echo "--resume $ckpt"
}

echo "========================================================"
echo "  NAFNetCascade — MCNoise MID lit R-fix strong training"
echo "  Size:             $SIZE"
echo "  Num frames:       $NUM_FRAMES"
echo "  Temporal base:    $TEMPORAL_BASE"
echo "  Experiment:       ${EXP_NAME:-<default>}"
echo "  MC config:        $NOISE_MC_CONFIG"
echo "  Stage 1 output:   $STAGE1_OUTPUT  (${SPATIAL_EPOCHS} epochs)$([ "$SKIP_STAGE1" == "1" ] && echo " [SKIP]")"
echo "  Stage 2 output:   $STAGE2_OUTPUT  (${STAGE2_EPOCHS} epochs)$([ "$SKIP_STAGE2" == "1" ] && echo " [SKIP]")"
echo "  Stage 3 output:   $STAGE3_OUTPUT  (${STAGE3_EPOCHS} epochs)$([ "$SKIP_STAGE3" == "1" ] && echo " [SKIP]")"
echo "========================================================"

if [[ "$SKIP_STAGE1" == "1" ]]; then
  echo ""
  echo "--- Stage 1 skipped (SKIP_STAGE1=1), using weights: $SPATIAL_WEIGHTS ---"
else
  echo ""
  echo "--- Stage 1: spatial ---"
  python training.py \
    --model spatial \
    --size "$SIZE" \
    --color-space "$COLOR_SPACE" \
    --noise "$NOISE" \
    --noise-mc-config "$NOISE_MC_CONFIG" \
    --loss "$SPATIAL_LOSS" \
    --scheduler "$SPATIAL_SCHEDULER" \
    --plateau-patience "$SPATIAL_PLATEAU_PATIENCE" \
    --lr "$SPATIAL_LR" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --patches-per-image "$PATCHES_PER_IMAGE" \
    --data "$DATA_CLEAN" \
    --val-clean "$VAL_CLEAN" \
    --val-noisy "$VAL_NOISY" \
    --frames-per-sequence "$SPATIAL_FRAMES_PER_SEQUENCE" \
    --random-spatial-frames \
    --val-frames-per-sequence "$SPATIAL_VAL_FRAMES_PER_SEQUENCE" \
    --val-crop-mode grid \
    --val-grid-size 3 \
    --output "$STAGE1_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$SPATIAL_EPOCHS" \
    $(_resume_flag "$STAGE1_OUTPUT")
fi

if [[ "$SKIP_STAGE2" == "1" ]]; then
  echo ""
  echo "--- Stage 2 skipped (SKIP_STAGE2=1) ---"
else
  echo ""
  echo "--- Stage 2: cascade temporal stage (spatial frozen) ---"
  python training.py \
    --model cascade \
    --size "$SIZE" \
    --num-frames "$NUM_FRAMES" \
    --temporal-base "$TEMPORAL_BASE" \
    --color-space "$COLOR_SPACE" \
    --noise "$NOISE" \
    --noise-mc-config "$NOISE_MC_CONFIG" \
    --loss "$STAGE2_LOSS" \
    --scheduler "$STAGE2_SCHEDULER" \
    --plateau-patience "$STAGE2_PLATEAU_PATIENCE" \
    --lr "$STAGE2_LR" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --patches-per-image "$PATCHES_PER_IMAGE" \
    --data "$DATA_CLEAN" \
    --val-clean "$VAL_CLEAN" \
    --val-noisy "$VAL_NOISY" \
    --random-temporal-windows \
    --windows-per-sequence "$WINDOWS_PER_SEQUENCE" \
    --val-windows-per-sequence "$VAL_WINDOWS_PER_SEQUENCE" \
    --val-crop-mode grid \
    --val-grid-size 3 \
    --spatial-weights "$SPATIAL_WEIGHTS" \
    --freeze-spatial \
    --output "$STAGE2_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$STAGE2_EPOCHS" \
    $(_resume_flag "$STAGE2_OUTPUT")
fi

if [[ "$SKIP_STAGE3" == "1" ]]; then
  echo ""
  echo "--- Stage 3 skipped (SKIP_STAGE3=1) ---"
else
  echo ""
  echo "--- Stage 3: joint fine-tune ---"
  python training.py \
    --model cascade \
    --size "$SIZE" \
    --num-frames "$NUM_FRAMES" \
    --temporal-base "$TEMPORAL_BASE" \
    --color-space "$COLOR_SPACE" \
    --noise "$NOISE" \
    --noise-mc-config "$NOISE_MC_CONFIG" \
    --loss "$STAGE3_LOSS" \
    --scheduler "$STAGE3_SCHEDULER" \
    --plateau-patience "$STAGE3_PLATEAU_PATIENCE" \
    --lr "$STAGE3_LR" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --patches-per-image "$PATCHES_PER_IMAGE" \
    --data "$DATA_CLEAN" \
    --val-clean "$VAL_CLEAN" \
    --val-noisy "$VAL_NOISY" \
    --random-temporal-windows \
    --windows-per-sequence "$WINDOWS_PER_SEQUENCE" \
    --val-windows-per-sequence "$VAL_WINDOWS_PER_SEQUENCE" \
    --val-crop-mode grid \
    --val-grid-size 3 \
    --resume "$([[ "$RESUME" == "1" && -f "$STAGE3_OUTPUT/last.pth" ]] && echo "$STAGE3_OUTPUT/last.pth" || echo "$STAGE2_OUTPUT/best.pth")" \
    --output "$STAGE3_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$STAGE3_EPOCHS"
fi

echo ""
echo "========================================================"
echo "  Done. Final weights: $STAGE3_OUTPUT/best.pth"
echo "========================================================"

runpodctl stop pod $RUNPOD_POD_ID
