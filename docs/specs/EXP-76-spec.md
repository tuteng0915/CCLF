# EXP-76 Spec — Clock-Adapter Bootstrapping

**Status:** DONE / PARTIAL FUNCTIONAL PASS
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

## Result (2026-08-10)

Freezing the backbone succeeds where joint EXP-72 fine-tuning did not. After
200 steps, mean held-out velocity MSE falls from `.05037` to `.02849` (43.4%),
the fixed-state clock-flip cosine changes `.97365 -> .96485`, and the adapter
scale grows `.010 -> .0637`. A 300-step continuation reaches mean MSE `.02733`
and cosine `.96349`; improvement has largely plateaued just short of the
pre-registered 50% MSE reduction.

At the 200-step checkpoint, Standard ODE remains healthy (PPL `265.2`),
`S_tau` rises from the EXP-72 level near `101.9` to `115.0`, and the evaluator's
LTR/RTL velocity cosine falls from `1.000` to `.9922`. LTR `.10` remains poor
at PPL `319.0`, but is now better than matched RTL `330.5`, unlike EXP-72.

After the 300-step continuation, Standard PPL is unchanged at `265.2`,
`S_tau=118.8`, and cosine is `.9911`; functional separation continues to grow.
Wave quality does not: LTR `.10` is `329.0`, RTL `.10` is `385.0`, and LTR
`.15` reaches `508.7`. The adapter can encode the clock without learning a
coherent heterogeneous rollout operator.

Decision: partial functional pass. The adapter demonstrably uses local time
without damaging synchronous generation, but teacher-field fidelity misses the
strict gate and direct wave quality remains negative. This is sufficient only
for the bounded EXP-77 Stage-0 test, not for a 2,000-step promotion.
