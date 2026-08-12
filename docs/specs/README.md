# CCLF Experiment Specs — Start Here

This directory is an experiment ledger, not a flat to-do list. Most files are
completed historical records. The active paper plan is deliberately small.

For the paper-facing numeric summary and complete formal evaluation tables,
start with [`../result.md`](../result.md). This directory retains the detailed
per-experiment protocols and the full historical audit trail.

## Status vocabulary

| status | meaning |
|---|---|
| **ACTIVE** | next experiment needed to close the paper's main claim |
| **CONDITIONAL** | run only if the corresponding supporting claim stays in the paper |
| **DEFERRED** | useful later, but must not delay the core mechanism result |
| **DONE** | evidence already available; rerun only for a named robustness gap |
| **SUPERSEDED** | old protocol replaced by a corrected version |
| **INVALID** | numbers must not be used |
| **NEGATIVE / DEAD HYPOTHESIS** | valid experiment that rejected its motivating story |

See [DEAD-ENDS.md](DEAD-ENDS.md) before reviving an old experiment.
`EXP-INDEX.md` remains the complete historical ledger.
Use [EVAL-PROTOCOL.md](EVAL-PROTOCOL.md) for the shared generation lengths,
solver budgets, fixed sampling bank, and promotion rules; do not infer a universal default
from one historical script.

## Current active queue

GS16--GS20 and the corrected temporal-KD panel are complete. The Pipeline
factorization and its four targeted follow-ups are also complete. The current
method ledger is:

| package | status | question | stop condition |
|---|---|---|---|
| [EXP-70](EXP-70-spec.md) | **DONE / NEGATIVE** | Does target-clock aliasing, heterogeneous context, or missing joint refinement dominate current Pipeline failure? | mixed-state error dominates; true local clocks and refinement do not rescue it |
| [EXP-71](EXP-71-spec.md) | **DONE / NEGATIVE** | Can a revisable prefix help the suffix while every position stays at the same global time? | correct content matters, but every soft arm loses to compute-matched ODE-64; no LTR advantage |
| [EXP-72](EXP-72-spec.md) | **DONE / STOPPED AT 500** | Can a model with deep, verified per-token time conditioning learn a native wave? | functional local-clock gate failed and LTR interaction worsened |
| [EXP-73](EXP-73-spec.md) | **IMPLEMENTED / NOT LAUNCHED** | If the clock is learned, can on-policy trajectory distillation close the remaining rollout gap? | runner smoke-tested; EXP-72 prerequisite failed, so formal training would not isolate exposure bias |
| [EXP-74](EXP-74-spec.md) | **DONE / HARD-ONLY POSITIVE** | Can sparse event-triggered anchoring retain the EXP-67 causal benefit? | soft expiry fails; one persistent hard anchor improves all three checkpoints and density controls |
| [EXP-75](EXP-75-spec.md) | **DONE / NEGATIVE** | Can predicted-clean context canonicalize heterogeneous attention? | PPL partly improves, but vector error and incoherence remain |
| [EXP-76](EXP-76-spec.md) | **DONE / PARTIAL PASS** | Can freezing the backbone force a functional local clock? | yes partially without Standard-quality loss; wave quality remains negative |
| [EXP-77](EXP-77-spec.md) | **DONE / NEGATIVE** | Does block-local asynchronous transition distillation work once the clock is functional? | all block samplers remain at PPL `3400--3900`; stop at Stage 0 |
| [EXP-78](EXP-78-spec.md) | **DONE / ODE-ONLY REVISABLE POSITIVE** | Does post-transition hard anchoring survive multi-seed, conditioned, stochastic, and reversibility tests? | robust ODE gain and real revision after release; native-SDE effect is inert |
| [EXP-79](EXP-79-spec.md) | **DONE / CONDITIONAL NEGATIVE** | Can late synchronization make block decoding revisable without raw mixed-time interaction? | fixed-prefix P1 confirms that late coupling only matches Semi-AR and loses to parallel on prompt-conditioned PPL, ROUGE-L, and boundary quality |
| [EXP-80](EXP-80-spec.md) | **DONE / ASYNC NEGATIVE; UNLOCK PPL GAIN REPLICATED** | Do key ELF methods change rank under real prefix conditioning? | conditioning does not rescue soft/local/canonical arms; Unlock-4's same-call U/C PPL gain replicates on two new OWT panels and Gutenberg, but prompt-gain improvement is not robust and diversity/repetition trade-offs remain |
| [EXP-81](EXP-81-spec.md) | **DONE / GENERIC-QUALITY EFFECT** | Is Unlock-4's conditional gain concentrated at the prompt boundary? | true-prompt NLL improves in every suffix band, but pooled prompt-gain CIs cross zero; this is not evidence of stronger prompt use |
| [EXP-82](EXP-82-spec.md) | **DONE / ODE PPL POSITIVE WITH TRADE-OFF** | Can lower-density temporary anchors retain the gain? | random 50% position-correct anchors at `t=.30`, `H=4` beat confidence selection on PPL/ROUGE-L in three panels; D1 falls and Rep-4 often rises |
| [EXP-83](EXP-83-spec.md) | **SEED REPLICATION RUNNING** | Is ELF's narrow endpoint-affinity collapse cross-architecture? | seed-42 says no: LangFlow remains ambiguous until the tautological endpoint and paired-noise Plaid becomes self-specific gradually without entropy collapse |
| [EXP-84](EXP-84-spec.md) | **P0 MIXED** | Can endpoint-specific residual directions causally redirect the future? | yes direction-specifically, but `epsilon_50(t)` is non-monotone, so simple basin rigidity is not confirmed |
| [EXP-85](EXP-85-spec.md) | **TWO-SEED PARTIAL POSITIVE** | Do anchors causally reduce endpoint uncertainty? | correct 50% anchors add pre-transition entropy collapse; alternative anchors redirect more than random controls, but shuffled effects are non-negligible |
| [EXP-86](EXP-86-spec.md) | **SMOKE DONE / FORMAL PENDING** | Does noise-debiased Plaid drift recover deterministic-model dynamics? | estimator is measurable; formal analysis waits for corrected paired-noise endpoint banks |
| [EXP-87](EXP-87-spec.md) | **DONE / THREE-SEED POSITIVE** | Does Plaid late coupling survive real-prefix evaluation? | yes: all 56-call late arms beat Block-SAR-64 C-PPL in every seed; raw has mean `95.38` versus `100.87` |
| [EXP-88](EXP-88-spec.md) | **DONE / GATE FAILED** | Can shadow-validated rollback remove harmful temporary anchors? | identity/combined rollback releases about one third and further lowers PPL, but D1 worsens; no multi-seed promotion |
| [EXP-89](EXP-89-spec.md) | **DONE / SCALE SIGN POSITIVE** | Does the frozen random-anchor policy survive length and prefix-ratio changes? | PPL improves in all 9 new cells through length 1024; unconditional gain shrinks with length and the small diversity trade-off remains |
| [EXP-90](EXP-90-spec.md) | **DONE / CONDITIONAL PORTABILITY POSITIVE** | Is temporary predicted-clean anchoring portable beyond ELF? | random correct anchors improve C-PPL in 3/3 seeds on both LangFlow and Plaid; shuffled content is catastrophic, while U-PPL and full Pareto behavior remain architecture-dependent |
| [EXP-91](EXP-91-spec.md) | **DONE / THREE-SEED NEGATIVE** | Does training on predicted-clean subsets turn the inference clue into a learned capability? | no: mean U/C PPL interactions are slightly unfavorable, prompt-gain interaction falls, and conditioned degeneration worsens on every inference seed |
| [EXP-92](EXP-92-spec.md) | **DONE / STRAIGHT-ENDPOINT TARGET REJECTED** | Does conditional, on-policy subset training retain the portable anchor benefit without damaging Standard generation? | no: `lambda_mix=1` fails, and the fixed `.25` follow-up worsens C-PPL interaction in 3/3 seeds and prompt gain in 3/3; preserve the curved teacher field in any successor |
| [EXP-93](EXP-93-spec.md) | **STAGE 1 DONE / LARGE ORACLE GAP; STAGE 2 ACTIVE** | Is random anchoring close to subset-optimal, and can trigger-time features predict better masks? | best-of-16 cuts C-PPL `37.38%` versus mean-random; test whether trigger-time subset features predict that utility out of sample |
| [EXP-94](EXP-94-spec.md) | **DONE / EXTRA DENOISING EXPLAINS GAIN** | Does Plaid late coupling beat a strictly token-compute-matched parallel sampler? | no: at identical 11264 token-calls, Parallel-44 C-PPL is `90.63` versus late raw/continuous `114.54`; retain only as a Block-SAR replacement |
| [EXP-95](EXP-95-spec.md) | **STAGE 1 DONE / FOUR CELLS PROMOTED** | Is there a Plaid temporary-anchor setting that improves PPL without the ELF diversity trade-off? | eight of 48 cells pass the paired screen; four early/transition cells receive larger complete-control panels |
| [EXP-96](EXP-96-spec.md) | **CLOSED BY EXP-94** | Can a per-sample maturity event beat the best fixed coupling schedule at matched average work? | fixed coupling has no compute-matched headroom |
| [EXP-97](EXP-97-spec.md) | **CLOSED BY EXP-94** | Does revisable coupling scale from two blocks to a three/four-block wave? | do not scale a two-block schedule that loses to compute-matched parallel refinement |
| [EXP-98](EXP-98-spec.md) | **CLOSED BY EXP-94** | Can on-policy trajectory distillation compress a verified coupling teacher while preserving its curved field? | no verified compute-matched teacher remains |

Completed decision:

| package | status | result |
|---|---|---|
| [EXP-61](EXP-61-spec.md) | **DONE / NEGATIVE** | Pipeline improves legacy noise-scale-1 ODE but worsens native-noise ODE by +197 PPL at n=256; do not promote the current sampler |
| [EXP-60](EXP-60-spec.md) | **DONE / NEGATIVE** | WFF training worsens both LTR sampler interactions and standard ODE quality; the local-time gate remains near zero |
| [EXP-62](EXP-62-spec.md) | **SUPERSEDED / NEGATIVE** | noisy-head self-distillation lowers PPL but produces fragmented pseudo-text; implementation does not match the original clean-teacher JAX KD objective |
| [EXP-63](EXP-63-spec.md) | **DONE** | corrected clean-teacher Early-KD improves unconditional ODE quality and commitment timing in two training seeds; conditioned gains are not robust |

Conditional after the active experiments:

| package | status | purpose |
|---|---|---|
| GS16/17 cross-architecture formalization | **DEFERRED / P2** | recalibrated endpoint-bank timing on a second architecture if the mechanism paper needs a general claim |

The asynchronous queue completed at its gates: mixed-state factorization and
canonical context are negative; adapter bootstrapping learns a clock, but
block-local transition distillation still does not compose. Do not launch a
broader clock/schedule sweep in the current architecture. The only positive
inference-time signal is now bounded precisely: EXP-78 confirms anchoring
across ODE seeds and conditioned generation, and shows that a four-step lock
can be released, but the effect is essentially absent under native SDE. Treat
this as an ODE method clue, not a universal sampler result.

## Evidence already supporting the paper

### Measurement and representation

- **GS11**: position averaging can create apparently early global retrieval.
- **GS12**: coarse POS statistics are largely mean-explainable; the leading
  rank-8 component is insufficient for exact token readout.
- **GS7**: oracle recoverability and true-rollout token recovery differ sharply.

### Context and rollout dynamics

- **GS13**: non-target states causally affect a fixed target margin, with a
  weaker and non-monotone interpretation than the original topic-direction story.
- **GS14**: lexical branch consensus contracts on true trajectories.
- **GS15**: rollout residuals remain below a direct chord reference; this is a
  global curvature description, not by itself a mechanism test.

These results should be reused rather than reimplemented. GS16--GS17 are the
minimal experiments needed to close the remaining mechanism ambiguity.

## Themes in the historical archive

| theme | relevant specs | current role |
|---|---|---|
| oracle/readout calibration | EXP-01--07, PT1--PT5 | background and measurement controls |
| token timing and stability | EXP-14/16, PT6/7/9, GS7/14/15 | supporting evidence for delayed stabilization |
| global-state formation | GS1--GS15 | hypothesis history; use corrected GS11--GS15 results |
| self-conditioning/method | EXP-31--54 | separate method lineage; several pipeline-specific failures |
| asynchronous denoising | spec-11, EXP-31/36/37, GS19 | prior pilots and current controlled ablation |
| cross-model replication | EXP-02/03/21--30/52/53, GS20 | LangFlow evidence and future CDCD boundary test |

## Rules for adding another spec

Add a new file only if all are true:

1. it answers a decision not already covered by GS16--GS20;
2. a positive and a negative result would change the next action;
3. its independent statistical unit and null are specified;
4. it cannot be a stage or control inside an existing package.

Otherwise append a stage to the existing package. Do not create more topic
probes, static representation visualizations, or nominal-time cross-model
comparisons for the current paper.
