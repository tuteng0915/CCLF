# EXP-GS18 Spec — Conditional Reviewer Controls

**Status**: CONDITIONAL — run only if the corresponding claim remains central
**Priority**: P1
**Models**: ELF first; cross-model replication only after a positive result

This file consolidates the former GS20 and GS21. These controls strengthen two
supporting claims, but neither should delay the core GS16--GS17 mechanism test.

## A. Rank- and energy-matched residual control

### Question

GS12 compares a rank-8 component with its much larger complementary residual.
Does the residual retain more token information because lexical information is
distributed, or simply because it keeps more dimensions and energy?

### Protocol

For centered `U_c = U - mean_position(U)`, evaluate equal-dimensional:

- top-`k`, middle-`k`, bottom-`k`, and random-`k` subspaces;
- `k in {1,2,4,8,16,32,64,128}`;
- raw oracle and predicted-clean representations.

Run both:

1. **reconstruction**: add the same position mean and measure native token
   recovery, terminal-token margin, POS-histogram R2, and retained energy;
2. **causal removal**: remove equal-dimensional top/bottom/random components
   from the native state and measure the change in token margin.

Report rank-matched and energy-matched versions. Use 10 random subspaces per
condition, `n_sequences>=128`, and sequence-level bootstrap intervals.

### Decision

Use the strong “distributed high-rank lexical code” claim only if non-top
components outperform rank- and energy-matched top components and their removal
causes greater damage. Otherwise retain the narrower statement:

> the leading rank-8 centered component is insufficient, while the
> complementary residual retains native token readability.

**Proposed script**:
`experiments/global_state/analyze_rank_matched_modes.py`

## B. Common-factor-controlled collective dynamics

### Question

Does the GS5 collective peak survive removal of document difficulty, global
confidence, logit scale, token frequency, and other sequence-wide factors?

### Protocol

Run on true rollout states from GS17; oracle states are a secondary comparison.
For margin increments `dm_{n,i,t}`, construct:

```text
M0 = raw dm
M1 = dm - mean_i(dm) within sequence
M2 = M1 residualized by frequency, POS, position, and current margin
M3 = M2 residualized by sequence logit norm, mean entropy, and mean margin
```

Measure connected spatial correlation and correlation length from `M3`.
Compare against:

- POS/frequency-stratified position shuffle;
- sequence shuffle at matched position/frequency;
- circular shift;
- sign-flip and marginal-variance-matched Gaussian nulls.

Use `n_sequences>=128`, at least 33 checkpoints, 1000 null permutations, and
sequence-level bootstrap bands across time and distance.

### Decision

Use “collective coordination” only if the peak survives M3, exceeds every
matched null, appears on true rollouts, and aligns with stable commitment in
GS17. If only M0 survives, call it a shared sequence-level fluctuation.

**Proposed script**:
`experiments/global_state/analyze_connected_coupling.py`

## Stop rule

Do not run both parts automatically. Run Part A only if the paper retains the
high-rank claim. Run Part B only if the paper retains “collective coordination”
as a headline mechanism claim.

---

## Pilot Results (2026-08-01, ELF baseline)

### Part A: Rank/energy-matched residual control

**Implementation**: `experiments/global_state/analyze_rank_matched_modes.py`,
n_samples=64, t=0.28, k in {1,2,4,8,16,32,64,128}, n_random=5. (Initial run
had a real performance bug -- the full per-sequence SVD was recomputed from
scratch for every (k, kind) combination, ~8x redundant work; fixed by
precomputing each sequence's SVD once and slicing/reusing it, verified the
fix reproduces identical numbers on a smoke test before rerunning at scale.)

**Result: the decision rule is NOT met.** At every single k tested, on both
raw and predicted-clean representations, `top-k` clearly dominates
`middle-k`/`bottom-k`/`random-k` on every metric (retained energy,
reconstruction token accuracy, structural R2):

| k | top tok_acc | middle tok_acc | bottom tok_acc | random tok_acc (mean of 5) |
|---|---|---|---|---|
| 8 (raw) | 0.010 | 0.000 | 0.000 | 0.000 |
| 32 (raw) | 0.021 | 0.000 | 0.000 | 0.001 |
| 128 (raw) | 0.067 | 0.007 | 0.000 | 0.020 |
| 8 (model) | 0.086 | 0.001 | 0.001 | 0.001 |
| 32 (model) | 0.307 | 0.001 | 0.000 | 0.001 |
| 128 (model) | 0.495 | 0.001 | 0.001 | 0.066 |

Non-top components do **not** outperform rank-matched top components at any
k, and (with the single exception of very small k on the predicted-clean
representation, where removing top-k hurts margin as expected) removing top-k
does **not** cause more damage than removing middle/bottom/random -- if
anything the opposite: removing a large top-k block *increases* the
measured margin (see caveat below).

**Per the spec's own decision rule, the strong "lexical information is
distributed and specially encoded outside the leading directions" claim is
NOT supported.** The narrower statement should be retained: *a small
rank-8-style top-k truncation is insufficient for token recovery, and the
(much larger) complementary residual retains native readability -- but this
is very likely explained mostly by the residual simply having far more
dimensions and far more retained energy than a small top-k slice, not by
lexical information being specially encoded in non-leading SVD directions.*
This narrows (does not overturn) GS3/GS12's "token identity depends on the
high-rank residual" finding: it is still true that you need many dimensions,
but "many dimensions" is doing the work, not "specifically the non-top
ones."

**Caveat on the removal-margin paradox**: at large k (>=16) on the
predicted-clean representation, removing the top-k block *increases* the
measured true-token margin substantially (e.g. k=128: delta_margin=+16.2).
This is likely a metric artifact, not evidence that the model "does better"
without its own leading structure: the default-competitor token `f_i` used
in the margin is fixed from the UNPERTURBED state, and once a large,
high-energy component is removed the resulting near-degenerate/near-mean
input may make `f_i` itself much less likely under the model, inflating the
margin numerically without the true token actually becoming more probable
in any meaningful sense. This should not be read as "the leading directions
are actively harmful."

### Part B: Common-factor-controlled collective dynamics

**Implementation**: `experiments/global_state/analyze_connected_coupling.py`,
n_traj=32, 17 dense free-running checkpoints, n_perm=200 per null (vs. spec's
1000). M2/M3 residualize by position index + current margin, then by
per-sequence logit norm/mean entropy/mean margin -- frequency and POS
covariates are NOT implemented (no ground-truth text to align against for
free-running trajectories); the "stratified" position-shuffle null is
correspondingly a plain shuffle.

**Result: the M3-residualized correlation length exceeds all 5 matched null
models' 95th percentile at 13 of 16 checkpoints** (the 3 exceptions are
t=0.461-0.520, t=0.579-0.637, both mid-trajectory where the raw signal is
already weak, and t=0.931-0.990, the last grid step, which is affected by
the same near-t_end numerical fragility documented in EXP-GS17-spec.md).
This is the "survives M3, exceeds every matched null, appears on true
rollouts" part of the decision rule.

**But the specific temporal SHAPE does not replicate GS5's single-peak
story.** `xi_M3` is highest EARLY (t=0.05-0.34, values 2.7-3.5, spanning and
extending well before GS17's median `tau_50_stable`~0.24-0.30), drops to a
trough mid-trajectory (t=0.40-0.70, values 0.5-0.7), then rises again late
(t=0.75-0.93, up to 1.6) before crashing at the final grid point. GS5's
oracle-state analysis found a single sharp susceptibility peak right AFTER
the cliff (t=0.28-0.39); here, on true free-running rollouts with three
additional confound-removal steps, the strongest signal is concentrated
BEFORE/DURING the commitment window rather than sharply after it, and a
second, weaker elevated region appears much later for reasons not yet
understood (possibly a distinct late-stage consistency effect, possibly
partly numerical).

**Verdict**: "collective coordination" (spatial correlation beyond what
several matched null models predict) survives the stricter M3 test and can
be retained as a real phenomenon on true rollouts -- but the "single peak
tied to the commitment cliff" framing from GS5 should be revised to
"broadly elevated collective correlation across the pre-commitment and
commitment period, with a currently unexplained secondary late-trajectory
rise," not a single well-localized event.

### Scale caveats (both parts)

Both parts are well below the spec's formal-scale minimums (Part A:
n_sequences=64 vs >=128, single t; Part B: n_traj=32 vs >=128, 17 checkpoints
vs >=33, 200 null permutations vs 1000). The qualitative conclusions above
(Part A's negative result for the "distributed lexical code" claim; Part B's
positive-but-reshaped "collective coordination" result) are clear enough at
pilot scale to be reported, but formal-scale reruns (and, per the spec, ELF
+ LangFlow cross-architecture replication) would be needed before citing
specific numbers in the paper.

---

## Cross-architecture replication on Plaid (2026-08-03, part of GS20)

Both parts rerun on Plaid (`--model plaid`) after adding it to these
scripts' `--model` choices. Two new environment issues surfaced (neither
existed for GS16/17/19, which never exercise these code paths) and were
fixed:
  - `analyze_rank_matched_modes.py` needs `nltk` (POS tagging) and
    `scikit-learn` (structural probe) -- not in the `plaid` conda env's
    original requirements.txt; installed directly.
  - Installing `nltk` triggered a `libstdc++` ABI mismatch
    (`CXXABI_1.3.15' not found`, via nltk's transitive `sqlite3` ->
    `libicui18n` import chain resolving to an old system libstdc++ instead
    of the conda env's newer one, despite the conda env's own copy having
    the required symbol) -- worked around with
    `LD_PRELOAD=$CONDA_PREFIX/lib/libstdc++.so.6` at invocation time rather
    than chasing the root RPATH/load-order cause.
  - A real bug was found and fixed in `PlaidAdapter.make_oracle_state`
    (missing `@torch.no_grad()`): calling Plaid's learned `gamma_bounds`/
    `noise_schedule` nn.Modules without no_grad leaves the output
    graph-tracked, which crashes on `.numpy()`. GS16/17/19 never hit this
    because they start from pure noise and never call `make_oracle_state`;
    GS18 Part A is the first script needing real oracle-corrupted text on
    Plaid. Fixed in `plaid_adapter.py` (`_gamma`, `make_oracle_state`,
    `make_null_state` all now `@torch.no_grad()`).

### Part A: n_samples=64, `--ks 1 2 4 8 16` (adjusted for Plaid's
embed_dim=16 -- k=16 is the full space, so all four subspace kinds
trivially converge there; k in {1,2,4,8} is the informative range).

**Clean POSITIVE cross-architecture confirmation of the ELF finding**: at
every k in {1,2,4,8}, on both raw and predicted-clean representations,
`top-k` dominates `middle-k`/`bottom-k`/`random-k` on every metric (retained
energy, reconstruction token accuracy, causal-removal damage) -- e.g. k=8,
raw: top tok_acc=0.096 vs middle=0.015, bottom=0.014, random=0.034; k=8,
model: top tok_acc=0.097 vs middle=0.034, bottom=0.023, random=0.041. This
replicates ELF's negative result for the decision rule (non-top components
do NOT outperform rank-matched top components) even though Plaid's ambient
space is 32x smaller (16 vs 512 dims) and the k-range tested is
correspondingly far narrower. **The narrowed claim from the ELF pilot now
has independent cross-architecture support**: "leading/high-variance
directions are the most information-dense per dimension for both structural
and lexical signal" generalizes across two architecturally distinct
continuous diffusion LMs; the stronger "lexical information is specially
encoded away from the leading directions" claim remains unsupported on
both.

### Part B: n_traj=32, n_states=17, n_perm=200.

**Disagreement / boundary condition, not a replication**: the M3-residualized
correlation length exceeds all 5 null models' 95th percentile at only **1 of
16** checkpoints on Plaid (t=0.520-0.579), versus **13 of 16** on ELF. Most
`xi_M3` values sit essentially AT their corresponding null p95 (e.g.
t=0.109-0.168: xi_M3=0.668 vs null p95=0.669) rather than clearly exceeding
it. **Leading hypothesis, not yet confirmed**: Plaid's native `solver_step`
is a stochastic ancestral sampler that injects independent Gaussian noise
into EVERY position at EVERY step (unlike ELF/LangFlow's deterministic Euler/
EDM steps) -- this per-position independent noise is a direct, structural
reason to expect the measured spatial correlation of margin increments to be
diluted, regardless of whether the underlying "collective coordination"
mechanism itself differs. Not disentangled here (would need e.g. computing
margin increments from the deterministic-drift component only, if Plaid
exposes one, or comparing against a null that specifically models this
per-step noise injection rather than the more generic 5 nulls already used).
Per GS20's own alignment rule, this is reported as an open boundary
condition, not evidence that "collective coordination" is ELF-specific.
