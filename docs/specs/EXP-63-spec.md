# EXP-63 Spec — Corrected JAX-Aligned KD Control

**Status**: COMPLETE / POSITIVE EARLY-WINDOW EFFECT / P0
**Models**: matched ELF-B continued-training control and clean-teacher KD

## Why EXP-62 must not answer the KD-window question

The EXP-62 PyTorch objective did not match the historical JAX KD recipe. For
denoiser rows it used the decoder logits from the same noisy mixed forward as
the teacher for the linear branch. The JAX implementation instead performs a
separate deterministic forward on clean `x0` at `t=1` and stop-gradients that
decoder distribution. EXP-62 also replaced JAX's smooth temporal gate and
ordinary-token normalization with a hard gate normalized over active tokens.

Consequently, EXP-62 is valid evidence that the noisy-head self-distillation
variant produces low-PPL but visibly degenerate text. It is not evidence about
which time window matters for the original KD objective.

## Corrected objective

For clean decoder teacher `p_dec(.|x0,t=1)`, noisy-state linear student
`p_lin(.|z_t,t)`, temperature `tau=4`, and

```text
omega(t) = sigmoid(k(t - t_low)) * [1 - sigmoid(k(t - t_high))],
```

the corrected auxiliary loss is

```text
L_KD = tau^2 * mean_masked[
  omega(t) KL(stopgrad(p_dec) || p_lin)
].
```

The required JAX-aligned settings are `k=10`, `t_low=0.25`,
`t_high=0.95`, and normalization by the ordinary token mask rather than the
sum of gate weights.

## Stage A — matched corrected replication

Train from the same converted baseline with identical data order, seed,
optimizer, learning rate, batch size, sequence length 128, and 2,000 steps:

| family | KD | purpose |
|---|---|---|
| `ct_control` | off | continued-training drift under the corrected code |
| `kd_jax_full` | clean teacher, smooth `[0.25,0.95]` gate | corrected KD effect |

Before launch:

1. unit-test zero KL for identical logits, gradient flow to the student, and
   the smooth gate shape;
2. run a two-step GPU smoke and require finite base/KD losses;
3. verify the KD metric is true KL rather than cross-entropy with a large
   teacher-entropy offset.

## Evaluation and decision

Use EMA weights, the fixed seed-42 initial-noise bank, native `z0=2 epsilon`,
SC-CFG 3, length 128, and ODE-32. Report PPL, D1/D2, rep-4, degeneration,
tokenization/format diagnostics, and at least eight unselected samples.

- Proceed to temporal windows only if corrected KD improves or preserves
  distributional and qualitative quality relative to the matched control.
- Stop if PPL falls while samples become fragments, code-like strings, or
  repetitive pseudo-text.
- If Stage A survives, run solver 16/64 and the true-rollout first/stable/
  revision fingerprint before training any length-1024 model.

Runner:

```bash
bash experiments/probe_elf/run_exp63_corrected_kd.sh ct_control 0
bash experiments/probe_elf/run_exp63_corrected_kd.sh kd_jax_full 1
```

## Stage A result

The corrected objective removes the catastrophic low-PPL/fragmented-text
failure of EXP-62, but its effect after 2,000 matched continued-training steps
is modest rather than method-level:

| checkpoint | ODE-16 PPL | ODE-32 PPL | ODE-64 PPL | D1 / D2 (ODE-32) |
|---|---:|---:|---:|---:|
| control | 775.4 | 261.8 | 109.0 | .394 / .860 |
| corrected KD | 736.0 | 261.6 | 98.5 | .423 / .877 |

On true rollout, corrected KD changes mean first-endpoint time from `.318` to
`.310` and stable-endpoint time from `.347` to `.338`, while revisions change
from `5.74` to `5.90`. Thus KD preserves qualitative quality and gives a small,
solver-consistent signal, but it does not establish a strong improvement.

## Stage B — temporal localization

Run three equal-width windows from the same baseline and data order:

| family | gate interval | question |
|---|---:|---|
| `kd_early` | `[0.05, 0.30]` | Does KD before the transition matter? |
| `kd_transition` | `[0.30, 0.55]` | Is the coordination window causal? |
| `kd_late` | `[0.55, 0.80]` | Is late lexical cleanup sufficient? |

Use the corrected clean teacher, true KL, ordinary-token normalization, and
the historical broad gate `[0.25, 0.95]` with `k=10`. Each intervention
multiplies that broad weight by a local selector with `k=40`; the selector
therefore isolates a subset of the historical objective instead of replacing
and accidentally amplifying it. The three selectors have equal width and use
identical `lambda_kd=1`. Evaluate ODE-32 and true-rollout dynamics first. Only
run solver 16/64 for a window that improves both quality and the commitment
fingerprint.

## Stage B result

Only the early selector improves both generation and rollout dynamics:

| checkpoint | ODE-32 PPL | first endpoint | stable endpoint | revisions |
|---|---:|---:|---:|---:|
| control | 261.8 | .318 | .347 | 5.74 |
| broad corrected KD | 261.6 | .310 | .338 | 5.90 |
| early | **211.0** | **.301** | **.329** | **5.55** |
| transition | 278.6 | .319 | .345 | 5.76 |
| late | 254.9 | .316 | .344 | 5.78 |

The early-control paired sequence bootstrap gives 95% intervals
`[-.0219,-.0126]` for first-endpoint progress,
`[-.0227,-.0127]` for stable-endpoint progress, and `[-.265,-.113]` for
revisions. Early KD also preserves its PPL advantage at ODE-16/64
(`634.8/82.6` versus `775.4/109.0`) and on a second seed-123 initial-noise bank
(`198.4` versus `270.3` at ODE-32).

This supports an early lexical-conditioning effect, not a transition-window
alignment effect. Before finer time slicing, replicate matched control/early
continued training with an independent training seed.

## Independent training-seed replication

With training seed 7 (new data order, training noise, and randomly initialized
linear student branch), the result replicates:

| checkpoint | ODE-32 PPL | first endpoint | stable endpoint | revisions |
|---|---:|---:|---:|---:|
| seed-7 control | 257.4 | .319 | .348 | 5.79 |
| seed-7 early | **224.2** | **.302** | **.330** | **5.47** |

The seed-7 paired bootstrap intervals are `[-.0225,-.0128]` for first,
`[-.0232,-.0127]` for stable, and `[-.394,-.253]` for revisions. The early
effect therefore survives training seed, generation-noise seed, and ODE solver
changes.

## Conclusion and boundary

Correct clean-teacher KD is not broadly beneficial at every denoising time.
Its useful causal support is concentrated before the measured collective
transition: early localization improves endpoint quality and makes commitment
earlier and less revision-heavy, whereas transition-only KD worsens PPL and
late-only KD is nearly neutral. This does **not** yet identify the narrowest
sub-window or establish length-1024 scaling; those are follow-ups, not part of
the present positive claim.
