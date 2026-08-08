# EXP-66 Spec — Native Early-KD x Commitment Interaction

**Status:** QUEUED AFTER EXP-65
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
