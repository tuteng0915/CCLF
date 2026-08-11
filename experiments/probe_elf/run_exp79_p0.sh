#!/usr/bin/env bash
set -euo pipefail

# Run from the ELF-torch repository root inside the `elf` environment.
# The smoke gate omits GPT-2 PPL; the decisive pilot evaluates the complete
# panel only after native-runner, condition-restore, and freeze-A gates pass.

python experiments/probe_elf/late_coupled_blocks_exp79.py \
  --checkpoint baseline \
  --device cuda:0 \
  --seed 42 \
  --n_samples 8 \
  --batch_size 2 \
  --maturities 24 28 \
  --parallel_steps 60 \
  --freeze_a_maturities 28 \
  --representations reencoded \
  --skip_ppl \
  --label p0_smoke_reencoded

python experiments/probe_elf/late_coupled_blocks_exp79.py \
  --checkpoint baseline \
  --device cuda:0 \
  --seed 42 \
  --n_samples 128 \
  --batch_size 4 \
  --maturities 24 28 \
  --parallel_steps 60 \
  --freeze_a_maturities 28 \
  --representations reencoded \
  --label p0_decisive_reencoded
