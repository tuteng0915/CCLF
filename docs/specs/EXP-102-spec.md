# EXP-102 Spec — Native Short-Horizon Trigger Utility

**Status:** DONE / LIKELIHOOD SIGNAL REPLICATES, FULL QUALITY GATE FAILS
**Purpose:** determine whether a short native counterfactual trajectory gives
a more useful trigger signal than EXP-101's instantaneous confidence,
entropy, or revision thresholds.

## Motivation

EXP-101 shows replicated per-trajectory trigger headroom but rejects the
discovery-frozen one-statistic event rule. Trigger utility is therefore not a
simple function of maturity at the current state. Before training another
predictor, test whether the consequence of anchoring becomes visible over a
few native solver steps.

## Frozen protocol

Reuse the two EXP-101 banks and their final conditional sequence NLL labels.
For each candidate trigger `8,10,12,14,16,18,20,22`:

1. replay the shared native Plaid trajectory to that trigger;
2. fork an unmodified control and the frozen EXP-101 top-confidence `.75`,
   horizon-one anchor intervention;
3. pair all ancestral noise between the two forks;
4. release anchors after one interval and continue both forks natively;
5. read out after `0,1,2,4` additional post-release updates.

At each lookahead, measure on the unresolved suffix positions:

```text
confidence gain,
entropy reduction,
top-1/top-2 margin gain,
predicted-clean cosine change relative to the control,
lexical disagreement with the control.
```

The statistics use no reference continuation, final generated text, or
external language model. Final NLL is attached only after the local signals
have been computed.

## Decision test

On discovery, choose one `(lookahead, statistic, direction)` using mean final
NLL of the trigger selected per trajectory. Freeze it and evaluate once on
validation. Report within-trajectory Spearman and pairwise ranking against
final trigger utility, selected C-PPL versus fixed step 14, bootstrap NLL, and
the resulting complete text-quality panel from EXP-101.

This is a teacher-signal audit, not yet a deployable sampler: evaluating every
candidate trigger costs multiple counterfactual denoiser calls. Promote only
if the frozen local signal:

- improves validation NLL over fixed step 14 with a 95% interval excluding
  zero;
- has pairwise ranking accuracy above `.55` on both banks;
- passes the EXP-101 D1/Rep-4/degeneration/prompt-gain quality tolerances.

If it passes, distill that native local utility into a single-pass trigger
predictor using disjoint banks. If it fails, close trigger prediction from
both instantaneous and short-horizon generic confidence signals; the next
method must learn a trajectory transition objective jointly with the model.

## Implementation

- builder: `experiments/interventions/build_plaid_local_trigger_utility_exp102.py`;
- frozen evaluator: `experiments/interventions/eval_plaid_local_trigger_utility_exp102.py`.

## Result

The discovery-frozen signal is unresolved-position entropy reduction after
four post-release native steps. It is the first tested trigger statistic that
generalizes:

| Bank | Fixed step-14 | Local-signal choice | NLL delta [95% CI] | Pair acc. | Spearman |
|---|---:|---:|---:|---:|---:|
| discovery | 78.44 | **64.62** | `-.194 [-.339,-.047]` | .609 | .291 |
| validation | 86.85 | **69.93** | `-.212 [-.362,-.065]` | .614 | .279 |

The raw argmax policy misses the complete gate. On validation D1 changes only
`-.0032` and Rep-4 improves, but degeneration increases by `.0313` and prompt
gain changes by `-.0159`. The signal is therefore promoted as a useful
counterfactual teacher, not as the final sampler. EXP-103 tests a conservative
fallback before any distillation claim.
