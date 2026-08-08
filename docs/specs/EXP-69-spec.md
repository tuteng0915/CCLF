# EXP-69 Spec — Native-SDE Anchor-Density Calibration

**Status:** QUALITY GATE FAILED / MECHANISM AUDIT NOT LAUNCHED
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

## Stage-A result

The formal baseline calibration completed on 32 length-1024 trajectories.
Confidence rises extremely quickly under the native sampler:

| Completed SDE step | Mean sampled time | Fraction at `.60` | Fraction at `.95` | Fraction at `.99` |
|---:|---:|---:|---:|---:|
| 4 | .139 | .256 | .105 | .068 |
| 8 | .197 | .726 | .501 | .422 |
| 12 | .239 | .902 | .760 | .699 |
| 16 | .303 | .966 | .900 | .869 |
| 20 | .341 | .987 | .955 | .937 |
| 24 | .425 | .994 | .975 | .963 |
| 28 | .525 | .996 | .981 | .973 |

This directly explains the EXP-68 saturation: around its nominal
`t_c=0.40`, even a `.99` threshold leaves only about 4% of positions
unresolved. The three quality-screen cells were frozen before generation as
step 4 / `.60`, step 8 / `.95`, and step 12 / `.95`, targeting approximately
25%, 50%, and 75% anchors.

## Stage-B quality screen

The three cells used the same 64 unconditional and 32 conditioned examples.
The standard arm is bit-for-bit reproducible at the metric level across all
three independent jobs (`PPL=29.31`, `D1=.156`, `D2=.625`), confirming the
paired latent, grid, and SDE-noise protocol.

| Cell | Actual uncond. anchors | Uncond. PPL | Delta | Cond. anchors | Cond. PPL | Delta | Cond. R-L |
|---|---:|---:|---:|---:|---:|---:|---:|
| standard | -- | **29.31** | -- | -- | **45.89** | -- | .116 |
| step 4 / `.60` | .252 | 49.10 | +19.79 | .233 | 68.24 | +22.35 | .115 |
| step 8 / `.95` | .524 | 43.54 | +14.23 | .300 | 57.04 | +11.15 | .117 |
| step 12 / `.95` | .785 | 36.80 | +7.49 | .524 | 60.67 | +14.78 | .117 |

All calibrated early-commit cells substantially worsen unconditional and
conditioned PPL. None produces unigram collapse or degeneration, and
repetition and D1/D2 remain healthy, so the failure is loss of contextual
coherence rather than a trivial repetition loop. The least harmful cell is
the latest and densest one; damage shrinks as the intervention approaches the
nearly saturated EXP-68 regime.

## Decision

No anchor-density-matched cell passes the native-SDE quality gate. The matched
shuffled-anchor mechanism panel is therefore not launched: it would explain
an intervention already known to harm generation and cannot establish a
usable sampler-independent method result.

Together, EXP-68 and EXP-69 show a solver-specific boundary:

- late native-SDE commitment is almost inert because nearly every position is
  already confident;
- early native-SDE commitment leaves an unresolved set but destroys coherence;
- the positive hard-commit method result and the EXP-67 causal mechanism
  should remain explicitly scoped to deterministic ODE rollout.
