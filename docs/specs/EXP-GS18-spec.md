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
