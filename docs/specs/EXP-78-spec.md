# EXP-78 Spec — Robust and Revisable Post-Transition Commitment

**Status:** DONE / ODE-ONLY REVISABLE POSITIVE
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

## Result (2026-08-10)

Stage A passes strongly on deterministic ODE. Across all three checkpoints and
all three seeds, every anchor arm lowers both unconditional and conditioned
PPL. Unlock-4 is the best mean arm:

| Checkpoint | Standard U-PPL | Unlock-4 U-PPL | Standard C-PPL | Unlock-4 C-PPL |
|---|---:|---:|---:|---:|
| ELF base | 285.4 | **208.5** | 509.6 | **379.7** |
| Control | 264.6 | **203.5** | 477.4 | **362.0** |
| Early-KD | 204.2 | **165.9** | 384.4 | **305.3** |

The favorable paired PPL sign holds in every seed. Mean conditioned ROUGE-L
changes by `+.0006/+.0007/+.0018`; degeneration does not systematically
worsen. Unlock-4 selects `87--88%` of eligible positions.

Stage B does **not** reproduce the effect under native SDE-32 at length 1024.
Unlock-4 changes unconditional PPL by only `-.20/-.21/-.28` and conditioned
PPL by `+.55/-.01/-.70` for Base/Control/Early-KD. Selection is saturated at
`93--98%`, so this is an inert intervention rather than a failure to trigger.

The timing audit confirms genuine, but limited, reversibility. After release,
`8.1--10.0%` of selected anchors finish at a different token. Revisions at
unselected positions fall by only `.09--.20` per token; their stable time moves
by `-.012/-.010/+.006`. Therefore the experiment supports temporary reliable
states as useful ODE conditions, but does not show a checkpoint-independent
advance of global coordination.

**Decision:** retain Unlock-4 as an ODE-specific inference intervention and a
method clue. Do not present it as a sampler-independent solution or as evidence
that the global coordination bottleneck has already been solved.
