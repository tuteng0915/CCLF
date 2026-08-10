# EXP-78 Spec — Robust and Revisable Post-Transition Commitment

**Status:** ACTIVE / P0
**Motivation:** EXP-74 is the only positive method signal after the controlled
Pipeline follow-ups. Promote it through multi-seed, conditioned, stochastic,
and reversibility tests before modifying training.

## Core question

Does a single post-transition hard commitment improve generation because it
turns reliable lexical states into useful conditions, or only because an
irreversible ODE intervention happens to lower the unconditional PPL metric?

## Arms

Use paired initial noise and a shared solver grid:

1. **Standard**: no commitment.
2. **Hard-highconf**: at the first checkpoint after `t=.40`, permanently
   condition positions with decoder confidence at least `.90`.
3. **Hard-stable**: read at `.30` and `.40`; permanently condition positions
   whose token identity agrees and whose final confidence is at least `.60`.
4. **Unlock-4**: use Hard-highconf, freeze for four solver intervals, then
   return those positions to continuous joint refinement.
5. **Unlock-8**: same, with an eight-interval lock.

Conditioned prefix positions are never counted as selected anchors. Report the
selected fraction among eligible suffix positions.

## Stage A — deterministic ODE robustness

```text
length=128, ODE-32 uniform, noise=2, SC-CFG=3
n_unconditional=128, n_conditioned=64
seeds={42,123,456}
checkpoints={ELF base, Control, Early-KD}
```

Report the complete shared quality panel, conditioned suffix PPL and ROUGE-L,
exact prefix preservation, calls, and anchor fraction. Promotion requires a
favorable paired sign in all three seeds and no systematic conditioned or
degeneration regression. ODE-64 remains the compute/quality reference, not an
arm to beat at 32 calls.

## Stage B — native SDE fidelity

Run only the best persistent and best unlock arm from Stage A:

```text
length=1024, native SDE-32 logit-normal, gamma=1.5
n_unconditional=128, n_conditioned=64, seed=42
```

Use paired per-step SDE noise. Do not infer stochastic fidelity from ODE.

## Mechanism and timing audit

For the best arm record `tau_first`, `tau_stable`, revisions, and endpoint
agreement separately for selected and unselected positions. The desired result
is later reliable conversion rather than premature guessing:

```text
tau_stable(unselected) decreases,
tau_first(unselected) does not move substantially earlier.
```

## Decision rule

- **Persistent method:** Hard-highconf/stable improves unconditional and
  conditioned quality across seeds and survives native SDE.
- **Revisable method:** Unlock-4/8 retains most quality gain, showing that
  temporary anchors are sufficient and irreversible commitment is unnecessary.
- **ODE-only method:** deterministic gains replicate but vanish or reverse in
  native SDE; present as solver-specific inference intervention only.
- **Metric artifact:** PPL improves while degeneration, continuation, or
  repetition worsens. Reject.

