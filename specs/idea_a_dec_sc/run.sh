#!/usr/bin/env bash
# Idea A: Decode-Branch Self-Conditioning Evaluation
# Compares baseline ODE vs decode-branch SC on the pretrained ELF-B checkpoint.
# Metrics: Gen.PPL at {1,2,4,8,50} steps (logged by eval.py to output_dir/*/metrics.jsonl).
#
# Usage:
#   CHECKPOINT=<path_to_checkpoint> bash run.sh
#
# CHECKPOINT can be a local path (e.g. ../../models/ELF/outputs/elf_b-owt/checkpoint_19000)
# or a HuggingFace repo id (e.g. embedded-language-flows/elf-b-owt).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ELF_SRC="$(realpath "$SCRIPT_DIR/../../models/ELF/src")"
CONFIG="$ELF_SRC/configs/training_configs/train_owt_ELF-B.yml"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the ELF-B checkpoint path or HF repo id}"
NUM_SAMPLES="${NUM_SAMPLES:-512}"

run_condition() {
    local NAME="$1"
    local SAMPLING_CFG="$2"
    local OUT_DIR="$SCRIPT_DIR/outputs/$NAME"

    echo ""
    echo "========================================"
    echo "  Running: $NAME"
    echo "========================================"

    python "$ELF_SRC/eval.py" \
        --config "$CONFIG" \
        --checkpoint_path "$CHECKPOINT" \
        --config_override "sampling_configs_path=$SAMPLING_CFG" \
        --config_override "output_dir=$OUT_DIR" \
        --config_override "num_samples=$NUM_SAMPLES" \
        --config_override "online_eval=true"
}

run_condition "baseline" "$SCRIPT_DIR/sampling_baseline.yml"
run_condition "dec_sc"   "$SCRIPT_DIR/sampling_dec_sc.yml"

echo ""
echo "========================================"
echo "  Results written to: $SCRIPT_DIR/outputs/"
echo "  metrics.jsonl files per step count:"
find "$SCRIPT_DIR/outputs" -name "metrics.jsonl" 2>/dev/null | sort
echo "========================================"
