# EXP-81 Spec — Prompt-Use Decomposition

**Status:** IMPLEMENTED / P0 ANALYSIS
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
