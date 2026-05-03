#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

CHECKPOINT="${CHECKPOINT:-checkpoints/spatial_exp048/best.pth}"
INPUT_DIR="${INPUT_DIR:-$HOME/data/tgb_train/TGB_training/train_noisy/TGB1004140_mid}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/spatial_exp048/infer_test}"
TILE="${TILE:-0}"

cd "$TRAINING_DIR"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --input "$INPUT_DIR"
  --output "$OUTPUT_DIR"
)

if [[ "$TILE" != "0" ]]; then
  ARGS+=(--tile "$TILE")
fi

ARGS+=(--no-temporal-flip)

uv run python infer.py \
  "${ARGS[@]}"
