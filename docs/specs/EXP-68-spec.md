# EXP-68 Spec — Native SDE Fidelity for Calibrated Commitment

**Status:** COMPLETE / SAME SIGN, NEGLIGIBLE NATIVE-SDE EFFECT
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

## Results

All five frozen configurations completed the native SDE-32 panel:

| Checkpoint | Standard PPL | Hard PPL | Delta | Cond. PPL delta | Commit fraction |
|---|---:|---:|---:|---:|---:|
| ELF baseline | 30.72 | 30.42 | -0.29 | +0.17 | .992 |
| control | 30.01 | 29.75 | -0.26 | +0.76 | .990 |
| Early-KD | 27.56 | 27.46 | -0.10 | +0.55 | .991 |
| control, train seed 7 | 30.05 | 29.85 | -0.21 | +0.77 | .990 |
| Early-KD, train seed 7 | 27.58 | 27.35 | -0.23 | +0.49 | .991 |

Unconditional PPL retains the favorable sign for all checkpoints, and D1/D2,
repetition, degeneration, unigram collapse, and conditioned ROUGE-L are
effectively unchanged. However, the ODE improvement magnitude does not
survive: the native-SDE deltas are only `-0.10` to `-0.29`, while conditioned
PPL is slightly worse by `+0.17` to `+0.77`.

The logit-normal grid also makes the frozen `t_c=0.40` policy commit about 99%
of unconditional positions at its first crossing. There is consequently
little unresolved suffix left to benefit from anchors. This is a mechanistic
reason for the vanishing effect, not evidence for an equally strong
sampler-independent method gain.

## Decision

The policy passes a weak sign/quality fidelity gate but fails magnitude
fidelity. The paper should call hard commitment a clean deterministic ODE
intervention whose benefit is highly solver-dependent. EXP-67 therefore audits
the deterministic ODE mechanism and must not present its result as a native-SDE
explanation.
