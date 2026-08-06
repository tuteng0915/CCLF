#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  echo "Usage: $0 FAMILY [SEED] [GPU]"
  echo "Families: ct_control kd_full kd_early kd_transition kd_late"
  exit 2
fi

family="$1"
seed="${2:-42}"
gpu="${3:-0}"

case "$family" in
  ct_control)
    lambda_kd="0.0"; gate_low="0.0"; gate_high="1.0" ;;
  kd_full)
    lambda_kd="1.0"; gate_low="0.0"; gate_high="1.0" ;;
  kd_early)
    lambda_kd="1.0"; gate_low="0.05"; gate_high="0.30" ;;
  kd_transition)
    lambda_kd="1.0"; gate_low="0.30"; gate_high="0.55" ;;
  kd_late)
    lambda_kd="1.0"; gate_low="0.55"; gate_high="0.80" ;;
  *)
    echo "Unknown family: $family" >&2
    exit 2 ;;
esac

run_name="exp62_${family}_s${seed}"
output_dir="outputs/${run_name}"
log_dir="logs/exp62"
mkdir -p "$log_dir"

export CUDA_VISIBLE_DEVICES="$gpu"
conda run -n elf python -u src/train.py \
  --config src/configs/training_configs/finetune_owt_ELF-B-panel.yml \
  --config_override "lambda_kd=${lambda_kd}" \
  --config_override "kd_gate_low=${gate_low}" \
  --config_override "kd_gate_high=${gate_high}" \
  --config_override "output_dir=${output_dir}" \
  --config_override "seed=${seed}" \
  2>&1 | tee "${log_dir}/${run_name}.log"
