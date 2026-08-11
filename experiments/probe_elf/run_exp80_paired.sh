#!/usr/bin/env bash
set -euo pipefail

device="${1:-cuda:0}"

python experiments/probe_elf/paired_conditional_revalidation_exp80.py \
  --checkpoint baseline \
  --device "${device}" \
  --seed 42 \
  --n_uncond 4 \
  --n_cond 4 \
  --batch_size 2 \
  --conditional_dataset owt \
  --arms standard32 standard64 pipeline_local_refine8 soft_ltr soft_random canonical_ltr_refine8 unlock4 \
  --skip_ppl \
  --label smoke

python experiments/probe_elf/paired_conditional_revalidation_exp80.py \
  --checkpoint baseline \
  --device "${device}" \
  --seed 42 \
  --n_uncond 64 \
  --n_cond 64 \
  --batch_size 4 \
  --conditional_dataset owt \
  --arms standard32 standard64 standard136 pipeline_local_refine8 soft_ltr soft_random canonical_ltr_refine8 unlock4 \
  --label p0_owt
