#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

CHECKPOINT_SPATIAL="${CHECKPOINT:-checkpoints/cs_test_linear_spatial/best.pth}"
CHECKPOINT_TEMPORAL="${CHECKPOINT:-checkpoints/cs_test_linear_cascade/best.pth}"
INPUT_TGB1004140_DIR="${INPUT_DIR:-$HOME/data/tgb_train/TGB_training/train_noisy/TGB1004140_mid}"
INPUT_TGB0502050_DIR="${INPUT_DIR:-$HOME/data/tgb_train/TGB_training/val_noisy/TGB0502050_mid}"
OUTPUT_SPATIAL_DIR="${OUTPUT_DIR:-checkpoints/cs_test_linear_spatial/infer_test}"
OUTPUT_TEMPORAL_DIR="${OUTPUT_DIR:-checkpoints/cs_test_linear_cascade/infer_test}"

cd "$TRAINING_DIR"

uv run python infer.py \
  --checkpoint "$OUTPUT_SPATIAL_DIR" \
  --input "$INPUT_TGB1004140_DIR" \
  --output "$OUTPUT_SPATIAL_DIR"

uv run python infer.py \
  --checkpoint "$OUTPUT_TEMPORAL_DIR" \
  --input "$INPUT_TGB1004140_DIR" \
  --output "$OUTPUT_TEMPORAL_DIR"

