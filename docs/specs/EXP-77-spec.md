# EXP-77 Spec — Asynchronous Block Transition Distillation

**Status:** CONDITIONAL / P2
**Launch condition:** EXP-76 produces a held-out, direction-sensitive clock
adapter; canonical context from EXP-75 is used if it passes its vector gate.
**Purpose:** train the actual staggered block transition, rather than applying
ordinary synchronous trajectory/consistency distillation.

## Asynchronous teacher target

Run a healthy scalar-time teacher on a dense synchronous trajectory and retain
`z_T(t)`, `x_T(t)`, and `v_T(t)`. Sample a staggered block configuration with
local clocks `tau_1,...,tau_G`. For one active block `g`, assemble its current
state and next local target:

```text
z_wave[i] = interp(z_T[:,i], tau_group(i))
z_target[g] = interp(z_T[:,g], tau_g + delta)
x_target[g] = interp(x_T[:,g], tau_g).
```

The student sees the entire heterogeneous configuration but loss and state
update apply only to `g`:

```text
z_student_next[g] = z_wave[g] + delta * v_student[g]

L_transition = ||z_student_next[g] - z_target[g]||^2
L_clean      = ||x_hat_student[g] - x_target[g]||^2
L_sync       = matched synchronous transition preservation
L = L_transition + lambda_x L_clean + lambda_sync L_sync.
```

This differs from classic trajectory distillation in two essential ways:

1. positions occupy different local times in the same forward pass;
2. supervision is for the next block-local transition in a fill/drain wave,
   not a larger synchronous solver step.

If EXP-75 passes, non-target context is represented by its predicted-clean
state; otherwise retain raw context and report that choice explicitly.

## Training arms

Start from the successful EXP-76 adapter and compare 500-step matched pilots:

1. synchronous transition-distillation control;
2. off-policy asynchronous block transitions assembled from the teacher bank;
3. scheduled on-policy block transitions, replacing teacher states with
   detached student states with probability `0 -> .5`;
4. RTL target-order control.

Use LTR fill/drain schedules with eight final synchronous refinement steps.
No position is frozen or irreversibly decoded.

## Evaluation

Use Standard-32/64, LTR/RTL/random asynchronous samplers, paired noise, and the
complete quality/timing panel. Add:

- block-local transition MSE on held-out teacher waves;
- LTR/RTL velocity separation;
- distance from the paired teacher trajectory;
- prefix/middle/suffix quality;
- `tau_first`, `tau_stable`, revisions, and endpoint-affinity entropy.

## Decision rule

- **Asynchronous-distillation success:** asynchronous training improves the
  LTR-minus-standard interaction over the synchronous distillation control,
  retains standard quality, and beats RTL/random on held-out generation.
- **Exposure-bias support:** scheduled on-policy improves over off-policy on
  both transition distance and final quality.
- **Canonical-context dependence:** improvement appears only with canonical
  context; make that representation the method, not local clocks alone.
- **Negative:** a verified clock and block-local supervision still cannot beat
  compute-matched synchronous decoding. Close the asynchronous method line.

