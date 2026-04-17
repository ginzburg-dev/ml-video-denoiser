#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

LOGDIR="${LOGDIR:-checkpoints}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"

cd "$TRAINING_DIR"

uv run tensorboard --logdir "$LOGDIR" --host "$HOST" --port "$PORT"
