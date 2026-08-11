# EXP-89 Spec — Unlock Scale and Prefix-Ratio Panel

**Status:** DONE / PPL SIGN SCALES, DIVERSITY TRADE-OFF REMAINS
**Policy:** EXP-82 random-position, position-correct predicted-clean anchors at
`t=.30`, density `.50`, temporary horizon `H=4`.

## Design

Compare Standard-32 with the frozen policy at lengths `128/256/512/1024`.
Every conditional panel uses native `cond_seq` and prefix ratios
`.25/.50/.75`; every job also evaluates paired unconditional generation.
Sample counts are scaled to processed-token cost:

| Length | U/C samples per cell |
|---:|---:|
| 128 | 128 (reuse EXP-82 P1 for ratio .50) |
| 256 | 64 |
| 512 | 32 |
| 1024 | 16 |

Use seed 42, OWT offset 15000 for new cells, noise 2, SC-CFG 3, and the full
EXP-80 metric panel. Report prompt-conditioned/shuffled PPL, prompt gain,
ROUGE-L, diversity/repetition/collapse, degeneration, exact prompt clamp,
anchor fraction, calls, processed-token calls, and wall time.

## Decision

- **Scaling support:** the U/C PPL sign survives every length and at least two
  prefix ratios without worsening repetition/degeneration as length grows.
- **Short-context sampler:** gains shrink or reverse beyond length 256.
- **Conditioning boundary:** unconditional gains survive but conditional signs
  depend strongly on prefix ratio.

Runner: `transition_unlock_pareto_exp82.py` with the frozen arm and
length/prefix arguments; no new policy tuning is allowed in this panel.

## Result (2026-08-11)

The frozen policy improves PPL in all nine new length/prefix cells. Reported
below are random-anchor minus Standard-32 deltas:

| length | prefix ratio | `Delta` U-PPL | `Delta` C-PPL | `Delta` prompt gain | `Delta` C-RL | `Delta` C-D1 | `Delta` C-Rep4 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | .25 | -25.2 | -41.7 | -.006 | +.004 | -.011 | +.003 |
| 256 | .50 | -25.2 | -96.5 | +.012 | +.004 | -.016 | +.003 |
| 256 | .75 | -25.2 | -175.3 | +.018 | +.003 | -.020 | +.001 |
| 512 | .25 | -9.8 | -18.7 | .000 | +.003 | -.010 | +.001 |
| 512 | .50 | -9.8 | -52.9 | +.005 | +.005 | -.011 | .000 |
| 512 | .75 | -9.8 | -126.0 | +.011 | +.003 | -.005 | .000 |
| 1024 | .25 | -2.5 | -13.0 | .000 | +.003 | .000 | -.001 |
| 1024 | .50 | -2.5 | -42.4 | +.027 | +.001 | -.012 | -.002 |
| 1024 | .75 | -2.5 | -184.0 | -.002 | +.002 | -.013 | +.003 |

The unconditional benefit shrinks sharply with length, whereas conditional
benefit grows with the amount of observed prefix. No new degeneration appears,
but the small D1 loss remains. The `n=16` length-1024 cells are preliminary as
specified; the result supports scale robustness of the PPL sign, not a clean
all-metric Pareto improvement.
