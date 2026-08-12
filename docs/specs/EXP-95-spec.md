# EXP-95 Spec — Plaid Temporary-Anchor Pareto

**Status:** ACTIVE / STAGE-1 SCREEN
**Purpose:** find whether the portable temporary-anchor clue can improve
Plaid quality without the diversity trade-off seen in ELF.

Use Plaid-native transition calibration and the EXP-90 paired conditional
protocol. Screen

```text
density in {.125, .25, .50, .75},
horizon in {1, 2, 4, 8},
trigger in {pre-transition, transition, post-transition}.
```

Use successive halving: seed 42, `n_U=n_C=16` for the grid; `n=32` for at most
four cells; then seeds 42/123/456 for at most two frozen cells. Every promoted
cell includes native Standard, readout-only sham, random position-correct
predicted-clean anchors, top-confidence anchors, and same-mask shuffled
content. Exact ancestral noise and prompts are shared within a cell.

Promote only if

```text
Delta C-PPL < 0,  Delta D1 >= 0,  Delta degeneration <= 0
```

in at least `2/3` seeds, with no material Rep-4 or prompt-gain regression.
This is a Pareto search, not evidence for a new mechanism if only PPL improves.

Stage-1 runner:
`experiments/interventions/eval_plaid_anchor_pareto_exp95.py`. It evaluates
Standard once and the random-anchor grid with exactly shared prompts, initial
noise, and Plaid ancestral noise. Trigger steps are frozen at `14/18/22`
(pre/transition/post); complete top-confidence, shuffled-content, and sham
controls are added only for promoted cells.
