#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

CHECKPOINT="${CHECKPOINT:-checkpoints/temporal_exp048_stage2/best.pth}"
INPUT_TGB1004140_DIR="${INPUT_DIR:-$HOME/data/tgb_train/TGB_training/train_noisy/TGB1004140_mid}"
INPUT_TGB0502050_DIR="${INPUT_DIR:-$HOME/data/tgb_train/TGB_training/val_noisy/TGB0502050_mid}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/temporal_exp048_stage2/infer_test}"

cd "$TRAINING_DIR"

uv run python infer.py \
  --checkpoint "$CHECKPOINT" \
  --input "$INPUT_TGB1004140_DIR" \
  --output "$OUTPUT_DIR"

uv run python infer.py \
  --checkpoint "$CHECKPOINT" \
  --input "$INPUT_TGB0502050_DIR" \
  --output "$OUTPUT_DIR"
