# EXP-97 Spec — Multi-Stage Revisable Progressive Coupling

**Status:** GATED BY EXP-94/96
**Purpose:** test whether the two-block Plaid result scales into a genuine
revisable wave rather than a single boundary trick.

For 384- and 512-token sequences, compare compute-matched Parallel, Block-SAR,
two-stage coupling, and progressive three/four-block schedules

```text
A -> [A,B] -> [A,B,C] -> [A,B,C,D] -> global refinement.
```

Earlier blocks are never frozen after admission. Include an ablation without
final global refinement and record blockwise revision, boundary PPL for every
join, prompt gain, the full quality panel, token-calls, wall time, and peak
memory. Use a frozen schedule inherited from EXP-94/96; do not retune it per
length.

Promote only if the advantage over compute-matched Parallel survives both
lengths and at least `2/3` seeds. If two blocks work but three/four do not,
scope the method as boundary refinement rather than wavefront generation.

