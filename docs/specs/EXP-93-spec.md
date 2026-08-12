# EXP-93 Spec — Subset Utility and Selector Headroom

**Status:** STAGE 2 CLOSED / REPLICATED ORACLE GAP, CURRENT SELECTORS FAIL
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

### Stage-1 result (ELF baseline, seed 42)

The formal conditional panel uses `n=64`, `M=16`, OWT offset `30000`, and all
paired controls above. Mean-random C-PPL is `335.67`, versus `379.43` for
top-confidence and `553.43` for Standard-32. Oracle-best-of-16 reaches
`210.19`, a `37.38%` reduction from mean-random. Paired sequence-NLL headroom
is `.4767 [.4409,.5126]` nats under trajectory bootstrap; random beats
top-confidence in `.6113` of the `1024` mask/trajectory cells.

The selector-headroom gate passes. This remains an oracle result: its D1
`.5273`, Rep-4 `.0294`, and degeneration `.1250` are worse than the average
random mask (`.5473/.0230/.1074`). Stage 2 must therefore predict held-out
utility without final-text information and retain the complete quality panel.

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

### Stage-2 result

Grouped four-fold trajectory OOF prediction is at chance on two independent
utility banks:

| Utility bank | Spearman | Pairwise accuracy | Top-1 | Selected C-PPL | Mean-random | Oracle |
|---|---:|---:|---:|---:|---:|---:|
| seed 42 | .0003 | .5013 | .0313 | 339.74 | 335.67 | 210.19 |
| seed 123 | .0367 | .5128 | .0625 | 381.69 | 390.12 | 240.52 |

No individual inference-time proxy is materially better: confidence,
coverage, and negative redundancy have pairwise accuracy `.477--.506`; the
best shadow entropy reduction reaches only `.523/.527` on the two banks. The
learned selector is slightly worse than mean-random on seed 42 and only
slightly better on seed 123, far from the oracle gap. Coefficients are not
stable enough to freeze a cross-bank selector.

Decision: Stage 3 is closed for these features. Subset identity has large
oracle headroom, but its utility is strongly nonlocal and is not recoverable
from the tested trigger-time summaries or one-step shadow features.

### Stronger dynamic and dependency-aware follow-ups

The one-step result prompted two escalations.

1. **Multi-step future-context lookahead.** Each candidate mask is held for
   `h in {2,4,8}` steps, released for one step, and scored by selected-token
   consistency plus unresolved confidence/entropy change. Discovery selects
   `h=8` release unresolved confidence gain as primary and entropy reduction
   as secondary. Their discovery C-PPL is `303.77/308.84` versus mean-random
   `335.67`. On the independent seed-123/offset-31000 bank, they reach only
   `384.17/373.37` versus `390.12`; paired NLL intervals
   `[-.0859,+.0513]` and `[-.1093,+.0199]` cross zero. This does not validate,
   and scoring all 16 candidates uses nine extra denoiser calls per candidate.
2. **Single-position causal dependency graph.** For 16 trajectories per bank,
   anchor each of 64 suffix positions separately for four steps, record its
   confidence/entropy/top-1 effect on all other positions, and aggregate the
   resulting graph over each candidate subset. The discovery leader,
   selected-to-unresolved mean confidence influence, falls from Spearman
   `.113` / pairwise `.540` to `.006` / `.504` on the independent bank and
   selects C-PPL `458.76`. The held-out grouped OOF graph model has Spearman
   `.041`, pairwise `.520`, and C-PPL `436.48`.

Decision: close static, short-lookahead, and additive pairwise selectors. The
remaining research question is whether non-additive subset interaction can be
learned on a trajectory-disjoint training bank and validated without final-NLL
selection leakage.

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
