# CCLF Experiment Specs — Start Here

This directory is an experiment ledger, not a flat to-do list. Most files are
completed historical records. The active paper plan is deliberately small.

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
solver budgets, seeds, and promotion rules; do not infer a universal default
from one historical script.

## Current active queue

GS16--GS20 are now completed at pilot or formal scale. The active queue is
about validating the method claims exposed during presentation assembly:

| package | status | question | stop condition |
|---|---|---|---|
| [EXP-61](EXP-61-spec.md) | **READY / P0** | Does Pipeline ODE survive the native ELF evaluation path (`noise_scale=2`, EMA), and does it work on baseline? | legacy result is reproduced and the 3-checkpoint native comparison is complete |
| [EXP-60](EXP-60-spec.md) | **IMPLEMENTED / P1** | Did inference-only asynchronous schedules fail merely because local time was unseen during training? | paired synchronous/WFF fine-tunes and sampler interaction are evaluated |
| [EXP-62](EXP-62-spec.md) | **READY / P1** | Are KD effects larger than ordinary continued-training drift and training-seed variation? | matched control/KD replicas complete before temporal-window variants are promoted |

Conditional after EXP-61:

| package | status | purpose |
|---|---|---|
| EXP-61 Stage 3 | **CONDITIONAL / P1** | multi-seed conditional semantic-quality validation for any surviving Pipeline arm |
| GS16/17 cross-architecture formalization | **DEFERRED / P2** | recalibrated endpoint-bank timing on a second architecture if the mechanism paper needs a general claim |

Recommended order: `EXP-61 smoke -> EXP-61 native formal + EXP-60 paired training -> conditional quality / cross-architecture only for surviving claims`.

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
