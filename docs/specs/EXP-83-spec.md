# EXP-83 Spec — Adaptive Rollback Anchoring

**Status:** READY AFTER EXP-82 SCREEN

## Question

Can dynamic release retain the Unlock PPL improvement while removing its
distinctness and repetition cost?

## Arms

Start from the best non-saturated EXP-82 density/trigger cell and compare:

1. Standard-32;
2. fixed Unlock-4;
3. confidence rollback: release when confidence falls below its trigger value
   by margin `delta`;
4. identity rollback: release after a fresh lexical readout disagrees with the
   anchored identity;
5. combined rollback;
6. fixed-budget rollback: release the least stable anchors first so the active
   budget never exceeds `q`.

Every arm uses the same scheduled readout points and reports them separately
from denoiser calls. Released positions return to continuous joint refinement;
they are never irreversibly remasked or decoded.

## Gate

Use paired `n=64+64` U/C screen, then three `n=128+128` replications for at
most one rollback arm. A pass requires

```text
delta PPL < 0,
delta D1 >= -.005,
delta Rep-4 <= +.002,
delta degeneration <= +.01
```

against Standard-32, plus nonzero release and final revision rates. If no arm
beats fixed Unlock-4 on the quality Pareto frontier, retain the trade-off as a
method boundary rather than tuning further.

Planned runner:
`models/ELF-torch/experiments/probe_elf/adaptive_rollback_exp83.py`.
