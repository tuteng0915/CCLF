# EXP-89 Spec — Unlock Scale and Prefix-Ratio Panel

**Status:** RUNNING / FROZEN POLICY
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
