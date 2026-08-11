# EXP-87 Spec — Conditional Plaid Late Coupling

**Status:** IMPLEMENTED / P1 BOUNDED VERIFICATION
**Purpose:** decide whether Plaid hard-`m=24` survives a real-prefix test.

Use a fixed 64-token Gutenberg prefix, a 192-token continuation, paired suffix
and ancestral noise, at least 128 examples per seed for seeds 42/123/456.
Compare only Parallel-32, Block-SAR-64, and raw/continuous/hard `m=24`.

Report prompt-conditioned and shuffled-prompt PPL, prompt gain, ROUGE-L,
A-to-B boundary PPL, D1/D2/Rep-4/degeneration/collapse, prefix/suffix revision,
prompt preservation, and compute.

- **Conditional compute lead:** hard `m=24` beats Block-SAR on paired
  prompt-conditioned and boundary PPL without a quality regression in all
  seeds while using 56 rather than 64 calls.
- **Unconditional-only lead:** signs disappear or reverse under real prompts.
- **Metric trade-off:** PPL improves while prompt gain, ROUGE-L, diversity, or
  repetition worsens.

No further maturity sweep follows a negative result. Runner:
`experiments/interventions/eval_plaid_conditional_late_coupling.py`.
