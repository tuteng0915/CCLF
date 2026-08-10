# EXP-77 Spec — Asynchronous Block Transition Distillation

**Status:** DONE / NEGATIVE (Stage 0, seed 42)
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

## Result (2026-08-10)

All four 200-step arms retain healthy synchronous generation but fail under the
31-call fill/drain block sampler.

| Training arm | Standard-32 | Standard-64 | Block LTR | Block RTL | Block random | Clock cosine |
|---|---:|---:|---:|---:|---:|---:|
| Sync transition control | 288.0 | 109.2 | 3849.9 | 3431.9 | 3515.0 | .9907 |
| Off-policy LTR | 282.9 | 107.4 | 3911.8 | 3528.0 | 3398.5 | .9906 |
| Scheduled on-policy LTR | 287.2 | 99.8 | **3658.0** | 3477.3 | 3435.6 | .9906 |
| Off-policy RTL | 272.9 | 106.7 | 3886.6 | 3566.9 | 3420.3 | .9905 |

Entries are PPL on paired `n=64` generations. Block outputs have D2 near
`.991--.996` and zero Rep-4 because they are incoherent high-diversity word
salad, not because diversity improves. Scheduled on-policy training gives only
a small relative LTR repair and no LTR ordering advantage; RTL/random remain
better in most comparisons.

The clock response survives training and Standard quality is healthy, so the
failure is not an unused-clock or catastrophic-forgetting explanation. A
learnable isolated block transition does not compose into a coherent parallel
fill/drain rollout. Stop at Stage 0 and close the current asynchronous block
method line.
