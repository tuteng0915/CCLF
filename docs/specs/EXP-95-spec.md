# EXP-95 Spec — Plaid Temporary-Anchor Pareto

**Status:** STAGE 2 POSITIVE / TWO SETTINGS IN FORMAL REPLICATION
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

## Stage-2 complete-control result

The four promoted cells use `n_U=n_C=32`, seed 42. Standard and readout sham
agree exactly (`U-PPL=138.01`, `C-PPL=111.83`, gain `.4786`, C-D1 `.6281`,
degeneration `.0313`). Key correct-content arms are:

| Setting | Selector | U-PPL | C-PPL | Gain | C-D1 | Rep-4 | Deg. | Revision |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| t14, d=.50, H=1 | random | 107.58 | 81.85 | .5600 | .6413 | .0000 | .0313 | .769 |
| t14, d=.50, H=1 | confidence | 105.58 | 87.93 | .5125 | **.6464** | .0000 | .0313 | .605 |
| t14, d=.75, H=1 | random | 101.32 | 86.54 | .5327 | .6398 | .0000 | .0313 | .775 |
| t14, d=.75, H=1 | confidence | **98.80** | **78.88** | .5466 | .6439 | .0007 | .0313 | .705 |
| t14, d=.50, H=4 | random | 117.39 | 87.04 | .5386 | .6372 | .0000 | .0313 | .758 |
| t18, d=.75, H=2 | confidence | 111.10 | 90.78 | .5230 | .6318 | .0007 | .0313 | .568 |

Shuffled-content C-PPL is `216.80/134.21/287.60/330.47` for the four settings,
ruling out a generic perturbation/diversity explanation. Correct anchors also
revise heavily after release, so the gain is not irreversible locking.

Freeze `t14,d=.50,H=1` and `t14,d=.75,H=1` for seeds 123/456 with the same five
arms and `n_U=n_C=32`. The first setting tests random broad coverage; the
second tests whether high coverage makes confidence selection useful. Do not
retune after seeing those seeds.
