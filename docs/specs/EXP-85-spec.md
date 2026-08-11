# EXP-85 Spec — Triggered Anchor Distillation

**Status:** GATED / DO NOT LAUNCH BEFORE EXP-82 OR EXP-83 PASSES

## Goal

Train a single forward transition to absorb the useful asymmetric condition
identified by temporary anchoring, rather than training another local-clock
pipeline.

At a calibrated transition state, divide eligible positions into reliable
anchors `A` and unresolved positions `U`. A frozen teacher performs the
temporary-anchor branch. The student sees the same state and anchor context,
but is supervised only on the unresolved transition:

```text
L_async = mean_U ||v_student(z, x_hat_A, t) - v_teacher_unlock(z, x_hat_A, t)||^2
L_clean = mean_U ||x_student - x_teacher_unlock||^2
L = L_async + lambda_x L_clean + lambda_sync L_sync.
```

At least half of batches remain ordinary synchronous ELF transitions.

## Stages

1. **Functional overfit:** frozen backbone plus a small anchor-context adapter,
   200 steps, fixed train bank and held-out noise.
2. **Bounded rollout pilot:** unfreeze only the top four blocks for 500 steps
   if held-out unresolved-transition MSE falls by at least 40% and standard
   generation remains within 10% PPL.
3. **Formal U/C evaluation:** compare matched synchronous distillation,
   triggered distillation, shuffled-anchor teacher, and the inference-time
   Unlock policy.

## Promotion gate

The method passes only if it improves the triggered-minus-standard interaction
over synchronous distillation, retains native standard sampling, improves
conditional generation, and does not reproduce the diversity/repetition cost.
Otherwise keep temporary anchoring as an inference-only diagnostic.
