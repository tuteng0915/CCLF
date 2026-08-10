# EXP-76 Spec — Clock-Adapter Bootstrapping

**Status:** ACTIVE / P1
**Start checkpoint:** healthy ELF base with EXP-72 layerwise time adapters
**Purpose:** distinguish an unlearnable local-time parameterization from an
optimization failure caused by attempting to update the whole model at once.

## Stage 0: fixed-shard overfit

Freeze the ELF backbone and train only:

- layerwise local-time projections and scales;
- optional relative-time attention bias in the follow-up arm.

Generate a fixed bank of paired synchronous teacher trajectories. Assemble
LTR and RTL teacher-wave states from different local checkpoints, and
supervise the per-position teacher velocity at each position's own time:

```text
v_target[i] = interp(v_teacher[:,i], tau_i)
L_clock = mean_i ||v_student(z_wave,tau)[i] - v_target[i]||^2.
```

Use balanced LTR/RTL examples but retain an explicit order label in the
diagnostics; do not average the two target fields before computing loss.
Train 200 steps on a fixed 32-noise bank. Compare:

1. frozen randomly initialized adapter;
2. layerwise additive adapter;
3. additive adapter plus relative-time attention bias.

## Mandatory functional gates

The bootstrap passes only if all are true on held-out noise:

```text
MSE_clock < 0.5 * frozen_adapter_MSE
cos(v_LTR, v_RTL) < .99
S_tau > 1.25 * synchronous_control_S_tau
```

Also require the unmodified standard ODE sampler to stay within 10% PPL of the
base checkpoint. Parameter movement alone is not success.

## Stage 1

Only after Stage 0 passes, unfreeze the top four transformer blocks for a
500-step mixed synchronous/heterogeneous pilot. At least half of batches stay
synchronous. Re-evaluate the same clock gates and the full EXP-72 sampler
panel.

## Decision rule

- **Bootstrap success:** functional LTR/RTL separation generalizes to held-out
  noise without damaging standard generation; use this checkpoint for EXP-77.
- **Adapter-only success:** frozen-stage gate passes but disappears after
  unfreezing; retain adapter separation and redesign joint optimization.
- **Failure:** even fixed-shard adapter overfit cannot learn the teacher field.
  Reject this local-time parameterization rather than training longer.

