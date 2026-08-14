# EXP-101 Spec — Plaid Adaptive Temporary-Anchor Trigger

**Status:** IMPLEMENTED / TRIGGER-HEADROOM GATE PENDING  
**Purpose:** test whether the remaining Plaid intervention problem can be
reduced from selecting a high-dimensional anchor subset to selecting one
trajectory-specific trigger time.

## Question

EXP-95 establishes a fixed policy: at native step 14 of 32, expose the top
confidence 75% suffix positions for one solver interval, then release all
positions. EXP-99 shows that anchor identity matters, but EXP-100 shows that a
joint final-NLL subset reranker does not generalize. EXP-101 freezes anchor
content, density, and horizon and varies only *when* the intervention occurs.

The first gate is deliberately diagnostic:

```text
Does per-trajectory best-of-trigger materially beat fixed step 14?
```

If not, no learned adaptive trigger is warranted.

## Frozen protocol

- model: Plaid baseline, native 32-step ancestral sampler;
- conditional panel: 64-token real prefix plus 64-token generated suffix;
- selector: top confidence;
- density: `.75`;
- horizon: one native solver interval;
- candidate trigger steps: `8,10,12,14,16,18,20,22`;
- fixed baseline: step `14`;
- discovery: seed 42 / panel offset 0;
- validation: seed 123 / panel offset 1000;
- `n=64` trajectories per bank;
- prompt, initial latent, and every ancestral noise draw are exactly paired
  across trigger candidates.

The oracle chooses the candidate trigger with lowest final conditional
sequence NLL separately for each trajectory. It is an upper bound, not an
inference algorithm. Report fixed, every candidate trigger, oracle, oracle
winner histogram, paired NLL bootstrap, and the complete matched-size text
quality panel.

## Inference-available event statistics

A separate standard replay records low-dimensional suffix summaries at every
candidate trigger. No candidate final text is used to construct them:

```text
mean / q10 confidence,
mean entropy,
mean top-1--top-2 margin,
one-native-step lexical revision rate,
one-native-step predicted-clean cosine instability,
confidence and entropy change.
```

If headroom passes, fit only a one-statistic threshold policy on discovery:
choose the earliest candidate time at which confidence/margin exceeds a
threshold or instability/revision/entropy falls below a threshold. Freeze the
statistic, direction, threshold, persistence, and fallback before validation.
This low-capacity policy is intentionally distinguishable from EXP-100's
high-dimensional subset reranker.

## Gates

Open the adaptive-policy stage only if best-of-trigger:

1. improves C-PPL over fixed step 14 by at least 5% on both banks;
2. has a trajectory-bootstrap 95% NLL interval excluding zero on both banks;
3. does not obtain the likelihood gain solely through a material D1, Rep-4,
   or degeneration regression.

Promote a threshold policy only if the discovery-frozen rule beats fixed step
14 on validation with an NLL interval excluding zero and no material quality
regression (`Delta D1 >= -.005`, `Delta Rep-4 <= .005`,
`Delta degeneration <= .015`, and `Delta prompt gain >= -.01`). Otherwise
close adaptive trigger timing and move to native
short-horizon trajectory supervision. Do not tune on a final bank after a
failed validation gate.

## Implementation

- bank runner:
  `experiments/interventions/eval_plaid_adaptive_trigger_exp101.py`;
- threshold fitter/evaluator:
  `experiments/interventions/fit_plaid_trigger_rule_exp101.py`.
