# EXP-94 Spec — Compute-Matched Plaid Coupling Frontier

**Status:** ACTIVE / P0
**Purpose:** decide whether the positive Plaid late-coupling result is a real
allocation advantage or only a comparison against an inefficient Block-SAR
baseline.

## Audit before promotion

For the current `m=24`, 32-step schedule over two 128-token blocks, the actual
denoiser work is

```text
C_late = 24*128 + 24*256 + 8*256 = 11264 token-calls,
```

because suffix maturation evaluates the full 256-token context even though the
first block is clamped. The relevant parallel control is therefore
`Parallel-44`, not only `Parallel-32`; `Block-SAR-64` uses 12288 token-calls.

Run fixed 64-token-prefix/192-token-continuation panels with shared documents
and initial noise. Plaid ancestral noise is exactly shared between arms only
when their time grids coincide; do not claim step-noise pairing across solvers
with different numbers of steps.

### Stage A — decisive compute audit

Compare Parallel-32, Parallel-44, Parallel-56, Block-SAR-64, and Late
raw/continuous/hard at `m=24`, joint refinement `J=8`. Parallel-44 exactly
matches Late-24 token-calls; Parallel-56 matches its sequential denoiser calls.

Report the complete EXP-87 conditional panel plus denoiser calls, token-calls,
wall time, and peak CUDA memory. Primary endpoints are paired prompt-
conditioned PPL and boundary PPL. D1/D2, Rep-4, degeneration, prompt gain, and
ROUGE-L are mandatory Pareto controls.

### Stage B — fixed-token-compute scheduling frontier

Run only if Stage A shows a meaningful advantage over Parallel-44 or a clear
latency/memory advantage. Keep `C=11264` token-calls and test raw/continuous
coupling schedules

```text
(m, J) in {(0,44), (8,32), (16,20), (24,8), (28,2)},
C(m,J) = 384m + 256J.
```

The pre-coupling grid ends at the corresponding native 32-step clock fraction,
and the joint grid covers the remaining interval. `m=0,J=44` is exactly the
Parallel-44 control. Different schedules share prompts and initial noise;
representation arms at the same schedule additionally share every ancestral
step noise.

## Decision

- **Scheduling advantage:** a late arm beats Parallel-44 on C-PPL in at least
  `2/3` formal seeds, has favorable mean C-PPL and boundary PPL, and does not
  materially regress degeneration or repetition.
- **Semi-AR replacement only:** it beats Block-SAR but not Parallel-44.
- **Extra denoising only:** Parallel-44/56 explains the gain.
- **Premature-condition failure:** quality improves monotonically toward the
  latest coupling point or Block-SAR.

Stage A starts with seed 42, `n=32`; promote to `n=128`, seeds 42/123/456 only
if the primary sign is favorable. Stage B uses successive halving: seed 42,
`n=32`, then at most the best two non-parallel schedules receive formal runs.

Stage A runner: the extended
`experiments/interventions/eval_plaid_conditional_late_coupling.py` with
`--parallel_steps_extra 44,56`. Stage B receives a separate runner only after
Stage A passes.
