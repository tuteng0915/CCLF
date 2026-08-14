# EXP-106 Spec — Noise-Averaged Causal Response

**Status:** IMPLEMENTED / DIAGNOSTIC RUNNING
**Purpose:** test whether EXP-105 fails because one stochastic Plaid probe is a
high-variance estimate of an intervention's expected short-horizon response.

## Motivation

EXP-102's single-probe entropy response transfers in aggregate, but EXP-105's
quality-safe threshold selects only four trajectories per bank. All four
calibration selections improve final NLL (mean `-.198` nats), whereas one of
four final selections worsens by `+.264` nats and reduces the selected-case
mean to `-.035`. The offending event has a large measured response (`1.45`),
so a larger scalar threshold alone cannot explain the error.

Plaid injects ancestral noise at every native update. A single paired
anchor/control fork cancels common noise within that fork, but its response can
still depend strongly on the sampled future. Estimate expected response with
four independent probe futures:

```text
r_bar(k) = (1 / 4) sum_j [H(control_j) - H(anchor_j)],
```

where anchor and control share noise within replicate `j`, while the four
replicates use distinct deterministic seed routes. The current state, anchor
mask, and final trigger bank remain unchanged.

## Stage 0 diagnostic

Rebuild local signals only for the already opened seed-2027 and seed-2028
banks. Compare one-probe and four-probe response on:

- within-trajectory pairwise accuracy against final trigger utility;
- Spearman correlation and selected-event win fraction;
- response variance across the four probe futures;
- whether the seed-2028 high-response harmful event is attenuated.

This stage cannot establish a new method because seed 2028 has already been
opened. It decides whether a fresh calibration/final sequence is warranted.
Proceed only if four-probe response improves pairwise accuracy in both banks
and the pooled improvement is at least `.03`; otherwise close response-noise
averaging before any new generation bank.

If Stage 0 passes, freeze replicate count at four, fit one scalar threshold on
a new seed-2029 calibration bank, and evaluate once on seed 2030. Report the
extra native calls explicitly; the goal is causal signal validation, not an
efficiency claim.

Implementation: the backward-compatible `--probe_replicates` and
`--probe_seed_base` options in
`experiments/interventions/build_plaid_local_trigger_utility_exp102.py`.
