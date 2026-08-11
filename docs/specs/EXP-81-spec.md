# EXP-81 Spec — Prompt-Use Decomposition

**Status:** READY / ANALYSIS-ONLY P0

## Question

EXP-80 shows a replicated Unlock-4 improvement in unconditional and
prompt-conditioned PPL, but not a robust improvement in aggregate prompt gain.
Does Unlock-4 improve local prompt use, or merely produce more probable text?

## Protocol

Reuse the three completed EXP-80 P1 panels and their paired Standard-32 and
Unlock-4 continuations. Reconstruct the exact true and cyclically shuffled
prompts from each panel. Score every continuation with the same frozen
GPT-2-large evaluator used by EXP-80.

For sample `m` and suffix band `b`, define

```text
g_m,b = NLL(y_m,b | shuffled(c_m)) - NLL(y_m,b | c_m).
```

Use GPT-2 suffix-token bands `1--8`, `9--32`, and `33+`, plus the full suffix.
Report per-sample NLL, token counts, prompt gain, and the paired
`Unlock-4 - Standard-32` difference. Construct 95% document-bootstrap
intervals with 10,000 resamples within each panel and across panels.

## Decision

- **Prompt-use improvement:** paired prompt-gain CI is positive, especially at
  the boundary, without reversing later in the suffix.
- **Fluency-only improvement:** conditional NLL improves but paired prompt gain
  is null or negative.
- **Mixed/local effect:** only the boundary band improves; scope the claim to
  short-range continuation conditioning.

No new generation is required. Runner:
`models/ELF-torch/experiments/probe_elf/prompt_use_decomposition_exp81.py`.
