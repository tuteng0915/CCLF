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
