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

## Current active queue

There are only **two core mechanism experiments**:

| package | status | question | stop condition |
|---|---|---|---|
| [GS16](EXP-GS16-spec.md) | **ACTIVE / P0** | Is the final endpoint already specific, or does endpoint affinity collapse late? | fixed endpoint bank is calibrated and self-specificity is estimated |
| [GS17](EXP-GS17-spec.md) | **ACTIVE / P0** | Is motion endpoint-parallel or orthogonal, and when does stable commitment occur? | local velocity and event timing distinguish curved transport from late selection |

After GS16--GS17:

| package | status | purpose |
|---|---|---|
| [GS18](EXP-GS18-spec.md) | **CONDITIONAL / P1** | rank/energy control and common-factor control; run only for claims retained in the paper |
| [GS19](EXP-GS19-spec.md) | **ACTIVE AFTER P0 / P2** | asynchronous schedule ablation before Wavefront Flow Forcing training |
| [GS20](EXP-GS20-spec.md) | **DEFERRED / P2** | minimal CDCD replication after the ELF mechanism is identified |

Recommended order:

```text
GS16 calibrated endpoint bank + specificity
    -> GS17 velocity + unified timing
    -> decide which GS18 controls are still necessary
    -> GS19 asynchronous intervention
    -> GS20 CDCD replication
```

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

