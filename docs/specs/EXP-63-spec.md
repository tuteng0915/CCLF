# EXP-63 Spec — Corrected JAX-Aligned KD Control

**Status**: READY / P0
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
