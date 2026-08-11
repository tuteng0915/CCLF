# EXP-84 Spec — Scale and Architecture Portability

**Status:** READY AFTER EXP-82/83 CELL FREEZE

## Question

Does the best calibrated temporary-anchor policy survive longer sequences,
different observed-prefix ratios, and different continuous-language-model
samplers?

## ELF scaling panel

Freeze one policy before inspecting scale results. Compare Standard-32 and the
frozen policy at lengths `128/256/512/1024`. For conditional generation use
prefix ratios `.25/.50/.75`; use paired suffix noise and at least 64 samples per
cell where memory permits. Report the full U/C panel, prompt gain, boundary
bands, anchor fraction, repetition/collapse, and processed-token calls.

## Cross-architecture panel

Test deterministic LangFlow first, then Plaid. Do not transfer ELF's nominal
time. Calibrate trigger checkpoint and confidence threshold without changing
text quality so that anchor density matches the frozen ELF cell. Compare:

```text
native standard / readout-only / calibrated temporary anchor /
same-position shuffled-content control.
```

Both unconditional and native-prefix conditional generation are mandatory.
Plaid must reuse paired ancestral noise after the fork.

## Decision

A general method claim requires a favorable conditional sign beyond ELF and
no architecture-specific degeneration. ELF-only length scaling supports an
ELF sampler method; failure under stochastic Plaid defines a solver boundary.

Planned runners:
`scale_unlock_exp84.py` and architecture-specific wrappers.
