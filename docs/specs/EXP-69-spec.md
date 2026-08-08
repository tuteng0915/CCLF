# EXP-69 Spec — Native-SDE Anchor-Density Calibration

**Status:** CALIBRATION IMPLEMENTED
**Priority:** P0
**Script:** `models/ELF-torch/experiments/probe_elf/native_sde_anchor_calibration_exp69.py`

## Motivation

EXP-68 applied the ODE-selected nominal policy (`t_c=0.40`, confidence
threshold `.60`) to ELF's stochastic logit-normal SDE grid. It selected about
99% of positions at the first crossing, left almost no unresolved set, and
reduced unconditional PPL by only `0.10--0.29`. This does not constitute a
fair test of the anchor-to-unresolved coordination mechanism.

## Stage A: intervention-free calibration

Run the standard native SDE-32 trajectory without writing any token. At fixed
completed solver-step indices

```text
s in {4, 8, 12, 16, 20, 24, 28},
```

measure the full lexical-confidence distribution and anchor fractions under

```text
gamma in {.60, .70, .80, .90, .95, .99}.
```

Use the ELF baseline, length 1024, 32 trajectories, native noise scale 2,
SC-CFG 3, logit-normal schedule (`P_mean=-0.8`, `P_std=0.8`), and SDE gamma
1.5. Fixed step index, rather than nominal time, is the primary clock because
the sampled time grid varies across batches.

Select at most three `(step, gamma)` cells spanning approximately 25%, 50%,
and 75% anchors. No quality result is inspected during this selection.

## Stage B: quality and mechanism gate

For selected cells, run strictly paired standard/true-anchor continuations
with shared initial latent, sampled time grid, and future SDE noise. First use
a 64-sample quality screen; only a cell that leaves a substantial unresolved
set and does not visibly collapse proceeds to the matched shuffled-anchor
mechanism audit.

The causal comparison must report, on positions unresolved at the fork:

- first and stable commitment shifts;
- revision-count shift;
- first-post-fork margin to each branch's own endpoint;
- endpoint agreement across branches;
- PPL, D1/D2, repeated 4-grams, and degeneration.

## Decision rule

- **Native-SDE support:** true anchors beat natural and confidence/frequency-
  matched shuffled anchors on stability and own-endpoint margin without a
  quality collapse.
- **Solver-specific mechanism:** no anchor-density-matched cell reproduces the
  ODE causal signature.
- **Inconclusive:** the policy cannot leave a useful unresolved set while
  maintaining coherent samples.

This is a newly calibrated SDE intervention. It must not be presented as a
post-hoc rescue of the frozen EXP-68 fidelity test.
