#!/usr/bin/env bash
# generate_mc_noise_pairs.sh — generate synthetic paired noisy/clean frames
# from a single clean image using all presets in a MCNoise JSON bank.
#
# Output:
#   <OUTPUT_NOISY>/<preset_name>.exr
#   <OUTPUT_CLEAN>/<preset_name>.exr
#
# Usage:
#   ./scripts/generate_mc_noise_pairs.sh
#
# Override:
#   DATA_CLEAN=/path/to/frame.exr PRESETS=/path/to/presets.json \
#     ./scripts/generate_mc_noise_pairs.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAINING_DIR="$ROOT_DIR/training"

LOCAL_STORAGE="~/data"

DATA_CLEAN="${DATA_CLEAN:-$HOME/data/tgb_train/TGB_training/train_clean_lit/TGB1004140_mid/TGB1004140.0001.exr}"
PRESETS="${PRESETS:-$ROOT_DIR/nuke/mc_noise_presets_tgb_lit_patch18_shadowface.json}"
OUTPUT_NOISY="${OUTPUT_NOISY:-$HOME/data/tgb_train/TGB_training/train_noisy_synth}"
OUTPUT_CLEAN="${OUTPUT_CLEAN:-$HOME/data/tgb_train/TGB_training/train_clean_synth}"
SEED="${SEED:-42}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"

cd "$TRAINING_DIR"

if [[ ! -f "$DATA_CLEAN" ]]; then
  echo "ERROR: clean image not found: $DATA_CLEAN" >&2
  exit 1
fi

if [[ ! -f "$PRESETS" ]]; then
  echo "ERROR: presets file not found: $PRESETS" >&2
  exit 1
fi

echo "========================================================"
echo "  MCNoise synthetic pair generation"
echo "  Clean:        $DATA_CLEAN"
echo "  Presets:      $PRESETS"
echo "  Output noisy: $OUTPUT_NOISY"
echo "  Output clean: $OUTPUT_CLEAN"
echo "  Seed:         $SEED"
echo "========================================================"

_skip_flag=""
[[ "$SKIP_EXISTING" == "1" ]] && _skip_flag="--skip-existing"

python generate_mc_noise_pairs.py \
  --clean        "$DATA_CLEAN" \
  --presets      "$PRESETS" \
  --output       "$OUTPUT_NOISY" \
  --output-clean "$OUTPUT_CLEAN" \
  --seed         "$SEED" \
  $_skip_flag

echo ""
echo "========================================================"
echo "  Done."
echo "  Noisy: $OUTPUT_NOISY"
echo "  Clean: $OUTPUT_CLEAN"
echo "========================================================"

cp -r "$OUTPUT_NOISY ~/data/tgb_train/TGB_training/train_noisy_synth"
cp -r "$OUTPUT_CLEAN ~/data/tgb_train/TGB_training/train_clean_synth"

echo ""
echo "========================================================"
echo "  Copied to /data/tgb_train/.."
