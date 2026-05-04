#!/usr/bin/env bash
# train_cascade_paired_synth_shadowface_pod.sh — Three-stage NAFNetCascade training
# on synthetic paired data generated from MCNoise shadowface presets.
#
# Stage 1  Spatial      NAFNet spatial denoiser on synthetic noisy→clean pairs
# Stage 2  Cascade      Load stage-1 weights, freeze spatial, train temporal stage only
# Stage 3  Fine-tune    Unfreeze all, joint fine-tune at lower LR
#
# Usage:
#   ./scripts/train_cascade_paired_synth_shadowface.sh
#
# Skip individual stages:
#   SKIP_STAGE1=1 SPATIAL_WEIGHTS=/path/to/best.pth ./scripts/train_cascade_paired_synth_shadowface.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
_TGB="/workspace/data/TGB_training"
PAIRED_CLEAN="${PAIRED_CLEAN:-$_TGB/train_clean_synth}"
PAIRED_NOISY="${PAIRED_NOISY:-$_TGB/train_noisy_synth}"
VAL_CLEAN="${VAL_CLEAN:-$_TGB/val_clean}"
VAL_NOISY="${VAL_NOISY:-$_TGB/val_noisy}"
INFER_NOISY="${INFER_NOISY:-$_TGB/train_noisy_lit}"

# ---------------------------------------------------------------------------
# Model / architecture
# ---------------------------------------------------------------------------
SIZE="${SIZE:-exp048}"
NUM_FRAMES="${NUM_FRAMES:-3}"
TEMPORAL_BASE="${TEMPORAL_BASE:-32}"
EXP_NAME="${EXP_NAME:-paired_synth_shadowface}"

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
SKIP_STAGE2="${SKIP_STAGE2:-1}"
SKIP_STAGE3="${SKIP_STAGE3:-1}"
RESUME="${RESUME:-0}"
CACHE_DATASET="${CACHE_DATASET:-1}"

# ---------------------------------------------------------------------------

cd "$TRAINING_DIR"

_resume_flag() {
  local ckpt="$1/last.pth"
  [[ "$RESUME" == "1" && -f "$ckpt" ]] && echo "--resume $ckpt"
}

echo "========================================================"
echo "  NAFNetCascade — paired synth shadowface training"
echo "  Size:             $SIZE"
echo "  Num frames:       $NUM_FRAMES"
echo "  Temporal base:    $TEMPORAL_BASE"
echo "  Experiment:       ${EXP_NAME:-<default>}"
echo "  Paired clean:     $PAIRED_CLEAN"
echo "  Paired noisy:     $PAIRED_NOISY"
echo "  Stage 1 output:   $STAGE1_OUTPUT  (${SPATIAL_EPOCHS} epochs)$([ "$SKIP_STAGE1" == "1" ] && echo " [SKIP]")"
echo "  Stage 2 output:   $STAGE2_OUTPUT  (${STAGE2_EPOCHS} epochs)$([ "$SKIP_STAGE2" == "1" ] && echo " [SKIP]")"
echo "  Stage 3 output:   $STAGE3_OUTPUT  (${STAGE3_EPOCHS} epochs)$([ "$SKIP_STAGE3" == "1" ] && echo " [SKIP]")"
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
  python training.py \
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
    --paired-clean "$PAIRED_CLEAN" \
    --paired-noisy "$PAIRED_NOISY" \
    --val-clean "$VAL_CLEAN" \
    --val-noisy "$VAL_NOISY" \
    --frames-per-sequence "$SPATIAL_FRAMES_PER_SEQUENCE" \
    --random-spatial-frames \
    $([ "$CACHE_DATASET" == "1" ] && echo "--cache-dataset") \
    --val-frames-per-sequence "$SPATIAL_VAL_FRAMES_PER_SEQUENCE" \
    --val-crop-mode grid \
    --val-grid-size 3 \
    --output "$STAGE1_OUTPUT" \
    --workers "$WORKERS" \
    --epochs "$SPATIAL_EPOCHS" \
    $(_resume_flag "$STAGE1_OUTPUT")
fi

# ---------------------------------------------------------------------------
# Spatial auto-test — infer first frame of each noisy training sequence
# ---------------------------------------------------------------------------
_SPATIAL_INFER_OUT="$STAGE1_OUTPUT/infer_spatial_auto_test"
if [[ -d "$INFER_NOISY" ]]; then
  echo ""
  echo "--- Spatial auto-test: first frame per sequence from $INFER_NOISY ---"
  mkdir -p "$_SPATIAL_INFER_OUT"
  for _seq_dir in "$INFER_NOISY"/*/; do
    [[ -d "$_seq_dir" ]] || continue
    _seq_name="$(basename "$_seq_dir")"
    _first_frame="$(find "$_seq_dir" -maxdepth 1 -name "*.exr" 2>/dev/null | sort | head -1)"
    [[ -n "$_first_frame" ]] || continue
    if [[ -f "$_SPATIAL_INFER_OUT/$_seq_name" ]]; then
      rm -f "$_SPATIAL_INFER_OUT/$_seq_name"
    fi
    mkdir -p "$_SPATIAL_INFER_OUT/$_seq_name"
    python infer.py \
      --checkpoint "$STAGE1_OUTPUT/best.pth" \
      --input "$_first_frame" \
      --output "$_SPATIAL_INFER_OUT/$_seq_name/$(basename "$_first_frame")"
  done
  echo "--- Spatial auto-test saved to: $_SPATIAL_INFER_OUT ---"
else
  echo ""
  echo "--- Spatial auto-test skipped (INFER_NOISY not found: $INFER_NOISY) ---"
fi

# ---------------------------------------------------------------------------
# Stage 2 — Cascade temporal stage only (spatial frozen)
# ---------------------------------------------------------------------------
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
    --loss "$STAGE2_LOSS" \
    --scheduler "$STAGE2_SCHEDULER" \
    --plateau-patience "$STAGE2_PLATEAU_PATIENCE" \
    --lr "$STAGE2_LR" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --patches-per-image "$PATCHES_PER_IMAGE" \
    --paired-clean "$PAIRED_CLEAN" \
    --paired-noisy "$PAIRED_NOISY" \
    --val-clean "$VAL_CLEAN" \
    --val-noisy "$VAL_NOISY" \
    $([ "$CACHE_DATASET" == "1" ] && echo "--cache-dataset") \
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

# ---------------------------------------------------------------------------
# Stage 3 — Joint fine-tune (all layers)
# ---------------------------------------------------------------------------
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
    --loss "$STAGE3_LOSS" \
    --scheduler "$STAGE3_SCHEDULER" \
    --plateau-patience "$STAGE3_PLATEAU_PATIENCE" \
    --lr "$STAGE3_LR" \
    --batch-size "$BATCH_SIZE" \
    --patch-size "$PATCH_SIZE" \
    --patches-per-image "$PATCHES_PER_IMAGE" \
    --paired-clean "$PAIRED_CLEAN" \
    --paired-noisy "$PAIRED_NOISY" \
    --val-clean "$VAL_CLEAN" \
    --val-noisy "$VAL_NOISY" \
    $([ "$CACHE_DATASET" == "1" ] && echo "--cache-dataset") \
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
