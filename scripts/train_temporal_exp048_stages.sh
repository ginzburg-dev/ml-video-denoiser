#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

PAIRED_CLEAN="${PAIRED_CLEAN:-$HOME/data/tgb_train/TGB_training/train_clean}"
PAIRED_NOISY="${PAIRED_NOISY:-$HOME/data/tgb_train/TGB_training/train_noisy}"
VAL_CLEAN="${VAL_CLEAN:-$HOME/data/tgb_train/TGB_training/val_clean}"
VAL_NOISY="${VAL_NOISY:-$HOME/data/tgb_train/TGB_training/val_noisy}"
SPATIAL_OUTPUT="${SPATIAL_OUTPUT:-checkpoints/spatial_exp048}"
SPATIAL_WEIGHTS="${SPATIAL_WEIGHTS:-$SPATIAL_OUTPUT/best.pth}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-checkpoints/temporal_exp048_stage2}"
STAGE3_OUTPUT="${STAGE3_OUTPUT:-checkpoints/temporal_exp048_stage3}"
WORKERS="${WORKERS:-12}"
SPATIAL_FRAMES_PER_SEQUENCE="${SPATIAL_FRAMES_PER_SEQUENCE:-10}"
SPATIAL_VAL_FRAMES_PER_SEQUENCE="${SPATIAL_VAL_FRAMES_PER_SEQUENCE:-3}"

cd "$TRAINING_DIR"

uv run python training.py \
  --model spatial \
  --size exp048 \
  --color-space log \
  --loss l1 \
  --scheduler plateau \
  --lr 1e-4 \
  --batch-size 2 \
  --patch-size 128 \
  --paired-clean "$PAIRED_CLEAN" \
  --paired-noisy "$PAIRED_NOISY" \
  --val-clean "$VAL_CLEAN" \
  --val-noisy "$VAL_NOISY" \
  --frames-per-sequence "$SPATIAL_FRAMES_PER_SEQUENCE" \
  --val-frames-per-sequence "$SPATIAL_VAL_FRAMES_PER_SEQUENCE" \
  --val-crop-mode grid \
  --val-grid-size 3 \
  --output "$SPATIAL_OUTPUT" \
  --workers "$WORKERS" \
  --epochs 60

uv run python training.py \
  --model temporal \
  --size exp048 \
  --color-space log \
  --loss l1 \
  --scheduler plateau \
  --lr 1e-4 \
  --batch-size 2 \
  --paired-clean "$PAIRED_CLEAN" \
  --paired-noisy "$PAIRED_NOISY" \
  --val-clean "$VAL_CLEAN" \
  --val-noisy "$VAL_NOISY" \
  --spatial-weights "$SPATIAL_WEIGHTS" \
  --freeze-spatial \
  --output "$STAGE2_OUTPUT" \
  --workers "$WORKERS" \
  --epochs 60

uv run python training.py \
  --model temporal \
  --size exp048 \
  --color-space log \
  --loss l1 \
  --scheduler plateau \
  --lr 5e-5 \
  --batch-size 2 \
  --paired-clean "$PAIRED_CLEAN" \
  --paired-noisy "$PAIRED_NOISY" \
  --val-clean "$VAL_CLEAN" \
  --val-noisy "$VAL_NOISY" \
  --spatial-weights "$SPATIAL_WEIGHTS" \
  --resume "$STAGE2_OUTPUT/best.pth" \
  --output "$STAGE3_OUTPUT" \
  --workers "$WORKERS" \
  --epochs 120
