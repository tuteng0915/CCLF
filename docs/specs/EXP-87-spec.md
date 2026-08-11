# EXP-87 Spec — Conditional Plaid Late Coupling

**Status:** DONE / THREE-SEED POSITIVE
**Purpose:** decide whether Plaid hard-`m=24` survives a real-prefix test.

Use a fixed 64-token OWT-family prefix, a 192-token continuation, paired suffix
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

## Formal result

All rows are means over seeds `42/123/456`, `n=128` per seed. Prompt latent
clamp error is exactly zero and native-reference agreement is `1.0`.

| Arm | Calls | Standalone suffix PPL | Prompt PPL | Shuffled | Gain | Boundary PPL | R-L | D1 | D2 | Rep-4 | Deg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Parallel | 32 | 107.36 | 105.32 | 127.10 | .1881 | 130.27 | .1034 | .4380 | .8967 | .0001 | .0260 |
| Block-SAR | 64 | 104.41 | 100.87 | 124.38 | .2096 | 149.82 | .1080 | .4326 | .8947 | .0010 | .0130 |
| Late raw | 56 | **98.93** | **95.38** | **118.44** | .2166 | 133.52 | **.1087** | .4304 | .8916 | .0010 | .0104 |
| Late continuous | 56 | 99.98 | 96.18 | 119.68 | **.2185** | 126.58 | .1086 | .4295 | .8921 | .0012 | **.0078** |
| Late hard | 56 | 100.19 | 96.06 | 119.27 | .2164 | **122.51** | .1085 | .4290 | .8917 | .0010 | .0156 |

Every late arm beats Block-SAR on prompt-conditioned PPL in every seed while
using eight fewer calls. Raw has the best mean suffix/PPL, continuous has the
best prompt gain and degeneration, and hard has the best boundary PPL. The
shared positive result is late coupling with joint revision, not a unique hard
commitment advantage. D1/D2 decrease slightly, so the claim is a robust
compute-quality lead over Block-SAR rather than dominance on every metric.
