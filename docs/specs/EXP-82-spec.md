# EXP-82 Spec — Transition Unlock Pareto Screen

**Status:** DONE / RANDOM TEMPORARY ANCHORS POSITIVE WITH TRADE-OFF
**Purpose:** determine whether Unlock-4 has a useful quality/compute Pareto
point below its current 87--88% anchor density.

Use paired ELF unconditional and fixed-prefix conditional generation. Compare
Standard-32, a readout-only sham, confidence-percentile anchors at transition
times `.30/.40/.50` and densities `.25/.50/.75/.875`, horizons `1/4/8` for
selected cells, plus random-mask and shuffled-content controls. Every arm uses
the same 32 denoiser calls; lexical readouts are counted separately.

Report the complete EXP-80 quality panel, prompt-conditioned and shuffled-
prompt PPL, prompt gain, ROUGE-L, anchor fraction, revisions, calls, token-calls,
and wall time.

- **Useful Pareto method:** density below `.60` retains a material PPL gain
  without systematic D1/D2, repetition, degeneration, or ROUGE-L regression.
- **Compression trade-off:** gain grows only with density while diversity and
  repetition worsen.
- **Content mechanism:** top-confidence content beats random selection and
  matched shuffled content.
- **Readout artifact:** readout-only sham matches anchor arms.

Runner:
`models/ELF-torch/experiments/probe_elf/transition_unlock_pareto_exp82.py`.

## Result (2026-08-11)

The P0 sweep selects `t=.30`, density `.50`, and `H=4`. The formal comparison
uses three paired `n_U=n_C=128` panels (Gutenberg and two OWT blocks):

| Panel | Arm | U-PPL | C-PPL | Gain | C-RL | C-D1 | C-Rep4 | C-Deg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Gutenberg | Standard-32 | 296.7 | 572.2 | .1517 | .0870 | .4514 | .0075 | .0234 |
|  | top-confidence | 243.0 | 424.6 | .1783 | .0899 | .4372 | .0128 | .0156 |
|  | random position | **206.5** | **392.4** | **.1838** | **.0920** | .4281 | .0120 | .0078 |
| OWT-42 | Standard-32 | 296.7 | 728.6 | .2251 | .0812 | .5262 | .0090 | .0234 |
|  | top-confidence | 243.0 | 517.9 | .2483 | .0854 | .5176 | .0129 | .0312 |
|  | random position | **206.5** | **512.7** | **.2642** | **.0876** | .5145 | .0103 | .0391 |
| OWT-43 | Standard-32 | 287.0 | 572.7 | .2535 | .0846 | .4916 | .0133 | .0547 |
|  | top-confidence | 240.5 | 430.9 | **.2754** | .0879 | .4800 | .0175 | .0625 |
|  | random position | **203.7** | **380.2** | .2739 | **.0888** | .4744 | .0173 | .0547 |

Random position-correct anchors are consistently best or near-best despite
lower trigger confidence (`.887` versus `.999` in P0). Matched shuffled
content is catastrophic (for example Gutenberg C-PPL `1554.1`), so correct
content is essential while high confidence is not. The PPL/ROUGE-L signal is
real in deterministic ELF ODE, but D1 falls and Rep-4 often rises. This is a
coverage-over-confidence mechanism clue, not an all-metric method win.
