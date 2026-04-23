#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

CHECKPOINT="${CHECKPOINT:-checkpoints/spatial_exp048/best.pth}"
INPUT_DIR="${INPUT_DIR:-$HOME/data/tgb_train/TGB_training/train_noisy/TGB1004140_mid}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/spatial_exp048/infer_test}"

cd "$TRAINING_DIR"

uv run python infer.py \
  --checkpoint "$CHECKPOINT" \
  --input "$INPUT_DIR" \
  --output "$OUTPUT_DIR"
  --no-temporal-flip
