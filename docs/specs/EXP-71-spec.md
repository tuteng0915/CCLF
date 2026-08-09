# EXP-71 Spec — Synchronous Soft-Anchor Pipeline

**Status:** ACTIVE / P0  
**Model:** ELF base, continued-training Control, corrected Early-KD  
**Planned script:** `models/ELF-torch/experiments/probe_elf/soft_anchor_pipeline_exp71.py`  
**Purpose:** test the core “prefix provides better conditions for the suffix”
idea without assigning different denoising times to different positions.

## Hypothesis

EXP-67 shows that position-correct predicted-clean anchors causally stabilize
unresolved tokens under deterministic ODE, while shuffled anchors reverse the
effect. The harmful part of Pipeline may therefore be heterogeneous clocks,
not directional information flow.

Keep every position at the same global time `t_k`. Let a revisable wave select
a leader set `A_k`. A first forward produces a current predicted-clean state;
only the leaders receive this fresh state in a second, same-time forward:

```text
x_sc_i = x_hat_current_i       if i in A_k
         x_hat_previous_i      otherwise

(v_k, x_hat_k) = f_theta([z_k, x_sc], t_k)
z_{k+1} = z_k + (t_{k+1}-t_k) v_k.
```

No latent is frozen, no token is discretized, and the leader set is recomputed
at every step. The entire sequence remains jointly revisable.

## Wave schedule

For normalized solver progress `s=k/K`, use

```text
a(s) = clip((s - .15) / .60, 0, 1)
|A_k| = floor(a(s) * L).
```

The primary LTR arm takes the first `|A_k|` eligible positions. Direction and
selection controls use the same leader count:

- RTL suffix wave;
- one fixed random permutation per sequence;
- confidence-ranked leaders;
- shuffled-content leaders: preserve leader positions/confidence bins but
  permute their fresh predicted-clean vectors across sequences.

The shuffled-content arm is a mechanism control, not a proposed sampler.

## Compute controls and arms

Use paired native ODE noise and compare:

1. **Standard-32:** ordinary one-pass ODE-32.
2. **Standard-64:** ordinary ODE-64, matched approximately to the two-forward
   cost of the soft-anchor arms.
3. **Two-forward null:** two forwards at each ODE-32 time, but refresh all or
   no self-conditioning positions uniformly; isolates extra compute.
4. **Soft-anchor LTR.**
5. **Soft-anchor RTL.**
6. **Soft-anchor random.**
7. **Soft-anchor confidence.**
8. **Soft-anchor shuffled-content.**

Screen arms 1--8 on 32 sequences. Promote at most the best directional arm,
the random arm, and the shuffled-content null to `n=128`; do not run a large
grid before checking generated samples.

## Primary estimands

On positions outside `A_k`, measure the next-step causal response:

```text
Delta margin_own
Delta entropy
Delta tau_first
Delta tau_stable
Delta revisions.
```

For final quality report the complete shared panel: PPL, D1/D2, Rep-4,
degeneration, words, MaxShare, UniqueRatio, unigram collapse, conditioned
suffix PPL/ROUGE-L, and exact prefix preservation. Report denoiser calls and
wall-clock latency separately from quality.

## Decision rule

- **Directional soft anchoring works:** LTR beats Standard-64, the two-forward
  null, RTL/random, and shuffled-content on quality and unresolved-position
  stability.
- **Symmetry breaking works:** LTR/RTL/random are all better than synchronous,
  with no reliable directional ordering.
- **Only correct content matters:** position-correct arms beat shuffled content
  but do not beat the compute controls. Retain as mechanism evidence, not a
  method result.
- **Negative:** no position-correct soft-anchor arm retains quality. Then the
  EXP-67 one-shot anchor effect does not compose into a repeated sampler.

If positive, the next implementation should distill the two-forward update
into a single pass; do not claim efficiency from this diagnostic version.
