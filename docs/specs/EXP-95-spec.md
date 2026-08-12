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

## Stage-1 result and promotion

All 48 random-anchor cells completed on the same seed-42 `n_U=n_C=16` panel.
Eight cells pass the screen gate (`Delta C-PPL<0`, `Delta D1>=0`, no larger
degeneration, `Delta Rep-4<=.001`, and prompt-gain regression smaller than
`.01`). The signal concentrates at the pre-transition trigger `step=14`
(`t_native=.4652`). The leading cell is:

```text
step 14, density .50, horizon 1:
Delta U-PPL = -34.56, Delta C-PPL = -45.58,
Delta C-D1 = +.0082, Delta degeneration = 0,
Delta prompt gain = +.0869.
```

Promote at most four complementary cells to `n_U=n_C=32` with Standard,
readout sham, random, top-confidence, and shuffled-content controls:

1. `step=14, density=.50, H=1` — strongest complete screen;
2. `step=14, density=.75, H=1` — coverage control;
3. `step=14, density=.50, H=4` — horizon control;
4. `step=18, density=.75, H=2` — only transition-trigger strict pass retained
   for timing contrast.

Post-transition cells can improve PPL strongly but all leading cells reduce
D1, so none are promoted.
