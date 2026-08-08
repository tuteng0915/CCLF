# EXP-66 Spec — Native Early-KD x Commitment Interaction

**Status:** COMPLETE / POSITIVE MAIN EFFECTS / WEAK INTERACTION
**Priority:** P0

## Question

The corrected early-window KD intervention improves short-context ODE quality
and true-rollout commitment timing in two independent training runs. Does that
effect survive native length, and is it complementary to confidence-gated hard
commitment?

## Checkpoints

Use the two matched training pairs from EXP-63:

| Training seed | Continued-training control | Early-window KD |
|---:|---|---|
| primary | `outputs/exp63_ct_control/checkpoint_2000` | `outputs/exp63_kd_early/checkpoint_2000` |
| 7 | `outputs/exp63_ct_control_seed7/checkpoint_2000` | `outputs/exp63_kd_early_seed7/checkpoint_2000` |

All evaluations use EMA weights.

## Stage A: shared commitment calibration

Use only the primary training pair and the EXP-65 calibration noise bank. Run
standard decoding plus

```text
t_c in {0.30, 0.40, 0.50, 0.60}
gamma in {0.60, 0.70, 0.80}
```

under length 128, native noise scale 2, SC-CFG 3, and ODE-32. Select one
*shared* `(t_c, gamma)` for both control and Early-KD. This prevents the
interaction result from being created by separately tuning each model.

## Stage B: native-length 2 x 2 panel

Freeze the shared policy, then evaluate all four checkpoints at length 1024:

| Training | Inference |
|---|---|
| matched continued training | standard |
| early-window KD | standard |
| matched continued training | calibrated hard commit |
| early-window KD | calibrated hard commit |

Use 256 paired unconditional samples, 128 fixed conditioned continuations,
uniform ODE-32, and the complete EXP-64 metric panel. The primary estimands are
the Early-KD effect under each sampler and the difference-in-differences:

```text
Delta_interaction =
  (EarlyKD+Commit - Control+Commit)
  - (EarlyKD+Standard - Control+Standard).
```

Report PPL together with D1/D2, repeated 4-grams, unigram collapse,
degeneration, and conditioned ROUGE-L. A PPL-only interaction is not a pass.

## Decision rule

- If Early-KD improves both standard and committed decoding, retain it as a
  training-time method result.
- If only commitment improves, retain hard commitment and treat Early-KD as a
  short-context mechanism intervention.
- If the combination loses diversity or increases premature locking, report
  the anti-synergy rather than choosing the best cell post hoc.

## Stage-A result and frozen shared policy

Both primary checkpoints favor the same quality-preserving grid point:

| Checkpoint | Standard PPL | Hard commit `(0.40, 0.60)` | D1 change | D2 change | Deg. change |
|---|---:|---:|---:|---:|---:|
| continued-training control | 277.6 | **213.0** | .455 -> .444 | .888 -> .887 | .031 -> .023 |
| Early-KD | 224.3 | **172.7** | .440 -> .429 | .878 -> .872 | .047 -> .047 |

The shared Stage-B policy is therefore frozen at `t_c=0.40`, `gamma=0.60`.
The selection was made before inspecting any length-1024 result. Stage B runs
the two training seeds with this one policy and the untouched seed-42 noise
bank.

## Stage-B length-1024 result

All four checkpoints completed the fixed 256-sample unconditional and
128-sample conditioned panel:

| Training checkpoint | Inference | PPL | D1 | D2 | Deg. | Cond. PPL | Cond. R-L |
|---|---|---:|---:|---:|---:|---:|---:|
| control | standard | 130.1 | .170 | .699 | .000 | 257.4 | .105 |
| control | hard commit | **120.8** | .173 | .695 | .000 | **229.9** | .106 |
| Early-KD | standard | 120.8 | .168 | .699 | .000 | 253.4 | .103 |
| Early-KD | hard commit | **111.3** | .172 | .693 | .000 | **229.3** | .105 |
| control, train seed 7 | standard | 130.2 | .170 | .701 | .000 | 257.7 | .104 |
| control, train seed 7 | hard commit | **120.9** | .173 | .696 | .000 | **229.4** | .106 |
| Early-KD, train seed 7 | standard | 127.9 | .168 | .692 | .000 | 266.7 | .104 |
| Early-KD, train seed 7 | hard commit | **117.4** | .171 | .687 | .000 | **238.1** | .105 |

Hard commitment gives a clean unconditional PPL improvement for all four
checkpoints (`-9.3`, `-9.5`, `-9.3`, and `-10.5`) with no degeneration or
material diversity loss. Early-KD also improves unconditional PPL relative to
its matched control for both training seeds (`-9.3` and `-2.3` under standard
decoding; `-9.5` and `-3.5` under commitment).

The interaction itself is weak: the unconditional PPL
difference-in-differences is approximately `-0.2` for the primary pair and
`-1.2` for training seed 7. The effects are therefore mostly additive rather
than synergistic. Early-KD does not robustly improve conditioned PPL: it is
slightly better for the primary seed and worse for seed 7, while conditioned
ROUGE-L remains nearly unchanged.

## Final decision

- retain hard commitment as a clean ODE-32 method result on the corrected
  checkpoints, pending the native-SDE fidelity check;
- retain Early-KD as a training-time unconditional-quality result whose effect
  is positive but training-seed dependent;
- do not claim a strong KD-by-commitment synergy or a robust conditioned
  Early-KD improvement.
