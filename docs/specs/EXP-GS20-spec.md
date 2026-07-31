# EXP-GS20 Spec — CDCD Cross-Family Replication

**Status**: DEFERRED until ELF core mechanism is resolved
**Priority**: P2
**Proposed adapter**: `experiments/phase_transition/adapters/cdcd_adapter.py`

## Purpose

Test whether the GS16--GS17 mechanism survives a third continuous categorical
diffusion family. CDCD is planned replication, not current evidence.

## Adapter gate

Implement the shared interface where meaningful:

```text
load
encode_clean / decode
make_oracle_state
forward_state
solver_step
native_logsnr
sample_initial_noise
```

Before interpretation, verify:

1. clean states decode correctly;
2. oracle corruption matches the original implementation;
3. one solver step matches the reference sampler;
4. fixed seeds reproduce expected behavior;
5. generation quality matches the reference checkpoint within tolerance;
6. native log-SNR is monotone.

Do not force CDCD into ELF's linear corruption convention.

## Minimal replication

Only replicate the paper-critical quantities:

1. dense `tau_first` and `tau_stable` from GS17;
2. local residual velocity and self-endpoint progress from GS17;
3. calibrated fixed-bank endpoint specificity from GS16;
4. GS15 chord metric as descriptive context only.

GS11--GS13 and all static topic probes are out of scope unless CDCD produces a
specific contradiction that requires them.

## Alignment and claim rule

Compare models by native log-SNR percentile, normalized arc length, and
event-aligned time relative to `tau_50_stable`; never by nominal diffusion time.

Call a result cross-architecture only if the same operational metric is
informative in ELF, LangFlow, and CDCD and trajectory-level confidence intervals
show sign consistency. A CDCD disagreement is a boundary condition, not a
failed replication to hide.

Pilot: 8 trajectories, 65 states, 6 calibrated branches. Formal: at least 32
trajectories × 3 seeds and 8 branches.
