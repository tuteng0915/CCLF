#!/usr/bin/env bash
set -euo pipefail

# EXP-79 conditional P1: a fixed 64-token Gutenberg prefix conditions a
# 192-token continuation. The first 64 generated positions form block A and
# the remaining 128 positions form block B.

python experiments/probe_elf/late_coupled_blocks_exp79.py \
  --checkpoint baseline \
  --device "${1:-cuda:0}" \
  --seed 42 \
  --n_samples 8 \
  --batch_size 2 \
  --conditional \
  --prefix_length 64 \
  --maturities 24 28 \
  --parallel_steps 60 \
  --freeze_a_maturities 28 \
  --representations reencoded \
  --skip_ppl \
  --label p1_conditional_smoke

python experiments/probe_elf/late_coupled_blocks_exp79.py \
  --checkpoint baseline \
  --device "${1:-cuda:0}" \
  --seed 42 \
  --n_samples 128 \
  --batch_size 4 \
  --conditional \
  --prefix_length 64 \
  --maturities 24 28 \
  --parallel_steps 60 \
  --freeze_a_maturities 28 \
  --representations reencoded \
  --label p1_conditional_decisive
