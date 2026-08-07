#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 FAMILY [GPU]"
  echo "Families: ct_control kd_jax_full kd_early kd_transition kd_late"
  exit 2
fi

family="$1"
gpu="${2:-0}"

case "$family" in
  ct_control)
    lambda_kd="0.0"; gate_low="0.25"; gate_high="0.95"; gate_k="10.0" ;;
  kd_jax_full)
    lambda_kd="1.0"; gate_low="0.25"; gate_high="0.95"; gate_k="10.0" ;;
  kd_early)
    lambda_kd="1.0"; gate_low="0.05"; gate_high="0.30"; gate_k="40.0" ;;
  kd_transition)
    lambda_kd="1.0"; gate_low="0.30"; gate_high="0.55"; gate_k="40.0" ;;
  kd_late)
    lambda_kd="1.0"; gate_low="0.55"; gate_high="0.80"; gate_k="40.0" ;;
  *)
    echo "Unknown family: $family" >&2
    exit 2 ;;
esac

run_name="exp63_${family}"
mkdir -p logs/exp63

export CUDA_VISIBLE_DEVICES="$gpu"
conda run --no-capture-output -n elf python -u src/train.py \
  --config src/configs/training_configs/finetune_owt_ELF-B-panel.yml \
  --config_override "lambda_kd=${lambda_kd}" \
  --config_override "kd_gate_low=${gate_low}" \
  --config_override "kd_gate_high=${gate_high}" \
  --config_override "kd_gate_k=${gate_k}" \
  --config_override "kd_normalize_active=false" \
  --config_override "output_dir=outputs/${run_name}" \
  --config_override "seed=42" \
  2>&1 | tee "logs/exp63/${run_name}.log"
