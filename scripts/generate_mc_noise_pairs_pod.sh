#!/usr/bin/env bash
# generate_mc_noise_pairs_pod.sh — generate synthetic paired noisy frames
# from clean training sequences using all presets in a MCNoise JSON bank.
#
# Output structure:
#   <OUTPUT_ROOT>/<preset_name>/<seq_name>/<frame.exr>
#
# Usage:
#   ./scripts/generate_mc_noise_pairs_pod.sh
#
# Override:
#   PRESETS=/path/to/presets.json OUTPUT_ROOT=/workspace/data/train_noisy_synth \
#     ./scripts/generate_mc_noise_pairs_pod.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

DATA_CLEAN="${DATA_CLEAN:-/workspace/data/TGB_training/train_clean_lit/TGB1004140_mid}"
PRESETS="${PRESETS:-$ROOT_DIR/nuke/mc_noise_presets_tgb_lit_patch18_shadowface.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/workspace/data/TGB_training/train_noisy_synth}"
SEED="${SEED:-42}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

cd "$TRAINING_DIR"

if [[ ! -d "$DATA_CLEAN" ]]; then
  echo "ERROR: clean data directory not found: $DATA_CLEAN" >&2
  exit 1
fi

if [[ ! -f "$PRESETS" ]]; then
  echo "ERROR: presets file not found: $PRESETS" >&2
  exit 1
fi

echo "========================================================"
echo "  MCNoise synthetic pair generation"
echo "  Clean:    $DATA_CLEAN"
echo "  Presets:  $PRESETS"
echo "  Output:   $OUTPUT_ROOT"
echo "  Seed:     $SEED"
echo "========================================================"

_skip_flag=""
[[ "$SKIP_EXISTING" == "1" ]] && _skip_flag="--skip-existing"

python generate_mc_noise_pairs.py \
  --clean   "$DATA_CLEAN" \
  --presets "$PRESETS" \
  --output  "$OUTPUT_ROOT" \
  --seed    "$SEED" \
  $_skip_flag

echo ""
echo "========================================================"
echo "  Done. Noisy frames at: $OUTPUT_ROOT"
echo "========================================================"
