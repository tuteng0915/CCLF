# EXP-93 Spec — Subset Utility and Selector Headroom

**Status:** ACTIVE / P0 ORACLE-HEADROOM TEST
**Purpose:** determine whether temporary random anchoring is merely a strong
average policy or already close to the best achievable subset, then replace
single-token confidence with a subset-level utility selector.

## Core correction

Random selection is not a proposed optimum. Its advantage over top confidence
shows that selected-token reliability alone is not the correct objective. For
a trigger state `s` and subset `A`, define conditional utility by the paired
change in final suffix NLL:

```text
U(A; s) = NLL_standard(s) - NLL_anchor(A; s).
```

The immediate question is the gap

```text
H_M = max_{m <= M} U(A_m; s) - mean_{m <= M} U(A_m; s),
```

for `M` independently drawn random masks at fixed density, trigger, horizon,
initial noise, and prompt. A positive held-out `H_M` establishes selector
headroom. The oracle-best mask is an upper bound, not a deployable method.

## Stage 1: mask-utility distribution

Freeze the current cross-architecture default policy except for mask identity:

- native trigger; density `.50`; hold horizon `4`;
- fixed 32-call base rollout and common initial/ancestral noise;
- conditional generation is primary; unconditional is diagnostic;
- `M=16` random masks for each of `n=64` trajectories;
- exact position-correct predicted-clean anchor content;
- record per-sequence GPT-2 NLL, ROUGE-L, D1/Rep-4/degeneration, and revision.

Report mean-random, median-random, oracle-best-of-16, oracle-worst-of-16, the
interquartile range of mask utility, and the probability that a random mask
beats top confidence. Bootstrap over trajectories, not over the `16` correlated
masks.

Decision:

- if oracle-best improves conditional NLL by less than `2%` over mean-random,
  selection is not the main bottleneck at density `.50`;
- if the gap is at least `5%`, proceed immediately to proxy prediction;
- between `2--5%`, proceed only if mask ranking is predictable out of sample.

## Stage 2: explain and predict subset utility

At the trigger, compute only inference-available features:

- **reliability:** mean/min/quantiles of lexical confidence in `A`;
- **spatial coverage:** maximum uncovered gap, span, and block occupancy;
- **representation coverage:** facility-location coverage of unresolved
  predicted-clean states by selected states;
- **redundancy:** mean/max pairwise selected-state cosine;
- **lexical coverage:** selected token-frequency quantiles and unique predicted
  token ratio;
- **causal influence:** one shadow-step change in unresolved confidence,
  entropy, and top-1 identity under the candidate mask.

Fit mask-utility predictors only on a training trajectory bank. Use a held-out
bank for feature selection and freeze the predictor before the final quality
panel. Report pairwise ranking accuracy and Spearman correlation; training
fit is not evidence.

The selector must score whole subsets, because utility is non-additive. A
useful structural objective is

```text
Score(A) = alpha Reliability(A)
         + beta  Coverage(A -> U)
         + gamma Influence(A -> U)
         - delta Redundancy(A).
```

This replaces the invalid assumption `Score(A)=sum_i confidence(i)`.

## Stage 3: deployable selector comparison

Compare at identical anchor density and trigger:

1. top confidence;
2. one random mask;
3. spatially stratified confidence;
4. confidence-constrained facility location;
5. learned best-of-16 reranking;
6. oracle best-of-16, labeled explicitly as an upper bound.

The learned reranker may batch its shadow probes, but report effective model
calls and FLOPs. It cannot be called a same-budget win if it uses unreported
lookahead compute. A cheap selector is preferred if its conditional NLL lies
within the oracle confidence interval.

Run ELF first. Promote to LangFlow and Plaid only if a frozen non-oracle
selector beats one-random in at least `2/3` ELF seeds with no material prompt-
gain, D1, Rep-4, or degeneration regression.

## Interpretation

- **Large oracle gap + predictable ranking:** random exposed a real selection
  problem; promote subset-level utility selection.
- **Large oracle gap + unpredictable ranking:** a better subset exists, but
  current trigger features do not identify it; learn an influence estimator or
  accept extra lookahead compute.
- **Small oracle gap:** random is not mathematically optimal, but selection is
  practically saturated at this density; improve content, trigger, or horizon
  instead.

