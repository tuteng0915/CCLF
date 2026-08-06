#!/usr/bin/env bash
set -euo pipefail

# EXP-61 Stage 1: isolate the initial-noise-scale protocol change.
# Run from the ELF-torch repository root inside the `elf` environment, e.g.
# CUDA_VISIBLE_DEVICES=5 bash experiments/probe_elf/run_exp61_stage1.sh

exp61_checkpoint="${EXP61_CHECKPOINT:-kd_cr}"
exp61_seed="${EXP61_SEED:-42}"
exp61_n_seq="${EXP61_N_SEQ:-64}"
exp61_log_dir="logs/exp61_stage1"

mkdir -p "${exp61_log_dir}"

run_cell() {
  local exp61_weights="$1"
  local exp61_noise_scale="$2"
  local exp61_cell="$3"
  local exp61_log="${exp61_log_dir}/${exp61_cell}_${exp61_checkpoint}_${exp61_weights}_ns${exp61_noise_scale}_seed${exp61_seed}_n${exp61_n_seq}.log"

  python experiments/probe_elf/pipeline_native_revalidation_exp61.py \
    --checkpoint "${exp61_checkpoint}" \
    --weights "${exp61_weights}" \
    --noise_scale "${exp61_noise_scale}" \
    --seed "${exp61_seed}" \
    --n_seq "${exp61_n_seq}" \
    --label "stage1_${exp61_cell}" \
    2>&1 | tee "${exp61_log}"
}

run_cell auto 1.0 cell_a_legacy_noise
run_cell auto 2.0 cell_b_native_noise
