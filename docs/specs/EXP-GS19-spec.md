# EXP-GS19 Spec — Order-Controlled Asynchronous Denoising Ablation

**Status**: AFTER P0
**Priority**: P2 — run after GS16--GS17
**Models**: ELF first; other models only after a native intervention is defined
**Proposed script**: `experiments/interventions/eval_async_schedule.py`

## Question

If synchronous denoising creates a coordination bottleneck, does asynchrony
help because of linguistic direction, symmetry breaking, or reliable anchors?

## Intervention

Existing checkpoints take scalar time, not a vector of local times. Therefore
this is explicitly a heterogeneous-state intervention with train--test
mismatch, not a native WFF sampler.

At each global step:

1. run the ordinary scalar-time model and obtain `xhat`;
2. estimate the native noise component;
3. perform the ordinary solver update;
4. reconstruct each position at scheduled local progress `tau_i` using the
   same `(xhat_i, epshat_i)` pair;
5. feed the heterogeneous state to the next scalar-time call.

For position rank `q_i` and width `Delta`:

```text
tau_i(s) = clip(s + Delta*(1 - 2*q_i/(L-1)), 0, 1)
```

Compare schedules with the same multiset of local times:

1. synchronous;
2. left-to-right;
3. right-to-left;
4. fixed random and reversed-random;
5. block-random;
6. confidence-adaptive.

Use `Delta in {0.05,0.10,0.20,0.30}` on normalized log-SNR progress. Every arm
gets identical initial noise, backbone-call count, and four final synchronous
refinement steps. Add a norm-matched random state perturbation control.

## Metrics

- Gen.PPL, distributional quality, distinct-n, repetition, degeneration rate;
- `tau_first`, `tau_stable`, and revision count;
- endpoint specificity/velocity from GS16--GS17;
- premature locking and branch entropy.

Desired signature:

```text
tau_stable decreases
tau_first changes little
generation quality does not degrade
```

## Scale and decision

- smoke: 64 samples;
- pilot: 256 paired samples × 3 seeds, all orderings at Delta 0.10/0.20;
- formal: 1000 samples for retained arms and two step budgets.

Interpretation:

| result | interpretation |
|---|---|
| only LTR wins | linguistic direction matters |
| LTR/RTL/random all win | breaking synchrony is sufficient |
| adaptive wins | reliable anchors matter most |
| quality improves but mechanism metrics do not | sampler heuristic, not mechanism validation |
| all fail | do not train Wavefront Flow Forcing |

---

## Pilot Results (2026-08-01, ELF baseline)

**Implementation**: `experiments/interventions/eval_async_schedule.py`,
n_samples=32, single seed, Delta=0.20 (spec pilot asks 256 paired x 3 seeds
at Delta 0.10 AND 0.20), 5 orderings out of the spec's 6+1
(synchronous/ltr/rtl/fixed_random/confidence_adaptive implemented;
reversed_random, block_random, and the separate norm-matched random
perturbation control NOT implemented). 28 async steps + 4 synchronous
refinement steps (32 total backbone calls, matched across arms).

**Correctness bug found and fixed before this run**: the first implementation
derived `xhat` via `adapter.forward_state`, which uses a simpler, always-
deterministic self-conditioning path -- NOT the same computation as
`adapter.solver_step`'s actual sampling machinery
(`_ode_step`/`_forward_sample`, which mixes conditional/unconditional passes
based on `self_cond_prob`). This made the `delta=0`/synchronous arm diverge
from a plain rollout on the same noise (verified: only ~8% token agreement
with a plain `solver_step`-driven rollout on 2 test sequences). Fixed by
deriving `xhat` from `adapter.solver_step`'s own return value; verified the
fix gives exactly 100% token agreement with a plain rollout before rerunning
the pilot.

**Result: "all fail" (spec's final decision-table row) -- do not pursue
Wavefront Flow Forcing based on this evidence.**

| ordering | gen_ppl | rep_rate | degenerate_frac | tau_stable_step | revisions |
|---|---|---|---|---|---|
| synchronous (baseline) | 76.8 | 0.029 | 0.00 | 16.79 | 5.57 |
| ltr | 143.8 | 0.230 | 0.09 | 21.68 | 9.73 |
| rtl | 190.6 | 0.326 | 0.75 | 20.52 | 8.18 |
| fixed_random | 375.2 | 0.001 | 0.00 | 19.82 | 8.72 |
| confidence_adaptive | 441.6 | 0.023 | 0.00 | 21.54 | 9.53 |

None of the desired-signature criteria are met by any ordering:
- `tau_stable` should decrease -- instead it INCREASES for all 4 orderings
  (16.79 -> 19.8-21.7).
- `tau_first` should change little -- instead it also increases
  (15.39 -> 18.9-20.7, not reported in the table above but in the raw JSON).
- generation quality should not degrade -- instead `gen_ppl` roughly
  doubles-to-quintuples (76.8 -> 144-442), and revision count nearly doubles
  (5.57 -> 8.2-9.7), meaning positions flip their top-1 guess far more often
  under async schedules, not less. RTL is catastrophic (75% of samples
  classified degenerate by 4-gram repetition rate > 0.3).

This is consistent and unambiguous across all four orderings tested (not
just one type failing while another succeeds) and across all three
signature metrics simultaneously -- not a mixed or marginal signal.

**Interpretation per the spec's own decision table**: "all fail -> do not
train Wavefront Flow Forcing." The heterogeneous-state reconstruction trick
(feeding a position its own locally-advanced/delayed noise-clean blend built
from the SAME xhat/eps_hat pair the model produced under normal scalar-time
conditioning) appears to actively hurt a model that was only ever trained
with a single global scalar time per forward call -- consistent with the
spec's own caveat that this is "a heterogeneous-state intervention with
train-test mismatch, not a native WFF sampler." The result does not tell us
whether a MODEL NATIVELY TRAINED with per-position local time would show the
same coordination-bottleneck relief GS16-18 might otherwise motivate
investigating -- only that grafting asynchrony onto an already-trained
scalar-time model does not work.

**Scale caveat**: pilot only (n=32, 1 seed, 1 Delta value, 5 of 6+1
orderings) -- well below the spec's own pilot scale. Given how unambiguous
and consistent the failure is across every ordering and every metric,
running the fuller pilot scale is unlikely to reverse the qualitative
conclusion, but has not been done.

---

## Cross-architecture replication on Plaid (2026-08-03, part of GS20)

Generalized the script (now `--model {elf,plaid}`) so the per-position
reconstruction step uses an adapter-appropriate `(alpha,sigma)` schedule
(`noise_params()`, dispatched by `adapter.name`) instead of ELF's hardcoded
linear `(tau, 1-tau)` formula. Verified the refactor is a pure
generalization, not a behavior change: ELF's `synchronous` arm reproduces
byte-identical `gen_ppl` (253.6) to the pre-refactor run. Two missing
`plaid` env dependencies (`sacrebleu`, transitively required by
`metrics_utils.py`'s `Metrics` class for Gen.PPL) were installed.

**n_samples=32, same 5 orderings, Delta=0.20, 28+4 steps -- clean POSITIVE
cross-architecture confirmation, if anything more decisive than ELF**:

| ordering | gen_ppl | vs. baseline | tau_first | tau_stable | revisions |
|---|---|---|---|---|---|
| synchronous (baseline) | 255.5 | -- | 18.96 | 21.73 | 11.31 |
| ltr | 779.0 | 3.0x worse | 19.15 | 22.54 | 10.70 |
| rtl | 1410.2 | 5.5x worse | 19.28 | 22.72 | 10.54 |
| fixed_random | 2479.3 | 9.7x worse | 19.59 | 22.64 | 11.68 |
| confidence_adaptive | 3688.3 | **14.4x worse** | 17.02 | **19.70** | 8.00 |

All four orderings fail the "all fail" bar again -- generation quality
collapses for every ordering (3x to 14.4x worse Gen.PPL than the
synchronous baseline), a substantially larger degradation than ELF's 2x-6x
range. One nuance worth noting: `confidence_adaptive` is the only ordering
that actually satisfies part of the desired signature (`tau_stable`
decreases from 21.73 to 19.70, fewer revisions) -- but pairs this with by
far the WORST quality collapse (14.4x). This is a clean illustration that
"nominally faster commitment" and "better generation" are not the same
thing here; the ordering that looks best on the timing metrics is the worst
on the metric that actually matters. **Conclusion unchanged and now
cross-architecture: do not pursue Wavefront Flow Forcing based on this
evidence, on either architecture tested.**
