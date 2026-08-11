# EXP-81 Spec — Prompt-Use Decomposition

**Status:** DONE / GENERIC-QUALITY EFFECT
**Purpose:** determine where Unlock-4's conditional PPL gain comes from and
whether it consistently increases use of the true prompt.

This analysis reuses completed EXP-80 continuations; it does not regenerate
samples. Rebuild the exact true and shuffled prompts, compute token-level NLL,
and decompose prompt gain into boundary tokens 1--8, middle tokens 9--32, late
tokens 33+, and the full continuation.

Compare Standard-32 and Unlock-4 per sample. Report hierarchical-bootstrap
confidence intervals for paired NLL/PPL and prompt-gain differences, sign
consistency across the two OWT panels and Gutenberg, and correlations between
prompt-gain changes and D1/Rep-4/degeneration changes.

- **Prompt mechanism:** Unlock-4 increases true-vs-shuffled prompt gain in the
  boundary band and the sign replicates across panels.
- **Generic sample-quality effect:** true and shuffled NLL improve similarly,
  with no stable prompt-gain increase.
- **Trade-off:** gains concentrate in repetitive/low-diversity samples.

Runner:
`models/ELF-torch/experiments/probe_elf/prompt_use_decomposition_exp81.py`.

## Result (2026-08-11)

The analysis pooled `384` paired continuations from the two OWT panels and
Gutenberg. Unlock-4 lowers true-prompt NLL in every band, but does not produce
a statistically reliable prompt-gain increase:

| GPT-2 suffix band | `Delta` true NLL | 95% CI | `Delta` prompt gain | 95% CI |
|---|---:|---:|---:|---:|
| tokens 1--8 | `-.2735` | `[-.3528,-.1945]` | `+.0465` | `[-.0071,+.0985]` |
| tokens 9--32 | `-.2369` | `[-.2930,-.1815]` | `-.0076` | `[-.0300,+.0152]` |
| tokens 33+ | `-.3749` | `[-.4292,-.3224]` | `-.0002` | `[-.0161,+.0162]` |
| full suffix | `-.3009` | `[-.3363,-.2653]` | `+.0045` | `[-.0085,+.0181]` |

Full-suffix prompt-gain deltas are `+.0096`, `-.0023`, and `+.0063` in the
three panels, and every confidence interval crosses zero. The boundary hint is
the only positive localized signal, and its pooled interval also crosses zero.
Therefore Unlock-4's replicated conditional PPL gain is primarily a generic
sample-likelihood effect, not evidence of stronger prompt use.
