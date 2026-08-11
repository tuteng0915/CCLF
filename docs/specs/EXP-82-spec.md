# EXP-82 Spec — Transition Unlock Pareto Screen

**Status:** IMPLEMENTED / P0
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
