# EXP-68 Spec — Native SDE Fidelity for Calibrated Commitment

**Status:** READY
**Priority:** P0
**Script:** `models/ELF-torch/experiments/probe_elf/native_sde_commit_eval_exp68.py`

## Question

The positive EXP-65/66 method results use uniform ODE-32. Do the same frozen
hard-commit policies survive ELF's native stochastic sampling path?

## Fixed native protocol

- sequence length 1024;
- native SDE-32 with logit-normal time grid;
- `P_mean=-0.8`, `P_std=0.8`, `sde_gamma=1.5`;
- initial noise scale 2 and SC-CFG 3;
- 256 unconditional samples and 128 conditioned continuations;
- seed-42 held-out bank;
- complete PPL/diversity/repetition/degeneration/ROUGE-L panel.

Standard and hard-commit arms share the initial latent, sampled time grid, and
every per-step SDE noise tensor. The confidence readout introduces no extra
random draw, so the comparison is strictly paired.

## Checkpoints and frozen policies

| Checkpoint family | `t_c` | `gamma_conf` |
|---|---:|---:|
| ELF baseline | 0.40 | 0.60 |
| EXP-63 continued-training controls, both training seeds | 0.40 | 0.60 |
| EXP-63 Early-KD, both training seeds | 0.40 | 0.60 |

No SDE result is used to retune these values.

## Decision rule

A fidelity claim passes only if hard commitment improves PPL without material
loss of diversity, repetition, degeneration, or conditioned ROUGE-L. A sign
reversal is reported as solver-specificity, not averaged with ODE. EXP-67 uses
only checkpoint/sampler combinations that survive this gate.
