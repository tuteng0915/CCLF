# EXP-GS23 Spec — CDCD Adapter and Cross-Family Mechanism Replication (P2)

**Status**: PLANNED  
**Priority**: P2 — establish whether the mechanism generalizes beyond ELF/LangFlow  
**Model**: CDCD continuous categorical diffusion  
**Proposed adapter**: `experiments/phase_transition/adapters/cdcd_adapter.py`  
**Output**: `results/global_state/cdcd/<checkpoint>/...`

## 1. Scope

CDCD is not currently implemented in the shared adapter layer. This spec first
defines the interface and then runs a minimal replication matrix. CDCD results
must not appear as completed evidence until every adapter sanity check passes.

## 2. Required adapter interface

Match the existing ELF/LangFlow adapter methods where meaningful:

```text
load
encode_clean / decode
make_oracle_state
forward_state -> predicted_clean, logits
solver_step
native_logsnr
sample_initial_noise
```

Also expose:

- native state parameterization;
- exact forward corruption coefficients;
- whether the reverse sampler is deterministic for fixed randomness;
- model velocity/score when available;
- the categorical/embedding readout used by the original CDCD implementation.

Do not force CDCD into ELF's linear `z=t*x+(1-t)*eps` convention.

## 3. Adapter sanity checks

1. terminal clean state decodes to the supplied tokens;
2. oracle corruption matches the original CDCD implementation numerically;
3. one native solver step matches the reference sampler;
4. fixed seed gives deterministic reproduction when expected;
5. generated quality and token statistics match the published/reference
   checkpoint within tolerance;
6. log-SNR is monotone and covers the comparison range used by ELF/LangFlow.

Record tolerances and raw comparisons in the spec result section before any
mechanism experiment is interpreted.

## 4. Minimal replication matrix

Run in this order:

1. **GS18-lite**: dense top-1 revision, `tau_first`, and `tau_stable`;
2. **GS17**: local residual velocity and self-endpoint progress;
3. **GS19**: calibrated true-trajectory branching;
4. **GS16**: fixed-bank endpoint specificity;
5. **GS15 only as a descriptive comparison**, not the decisive mechanism test;
6. GS21 if the earlier results reveal a comparable transition window.

GS11--GS13 are optional and should not delay the true-rollout experiments.

## 5. Cross-family alignment

Architectures are compared at:

- native log-SNR percentiles;
- normalized cumulative arc length;
- event-aligned time relative to median `tau_stable`.

Never compare raw nominal diffusion `t`. Report raw-state and model-prediction
representations separately; if a metric saturates, mark it uninformative rather
than substituting a different metric only for one model without disclosure.

## 6. Scale

Pilot:

- `n_traj=8`;
- 65 saved states;
- `K=6` calibrated branches.

Formal:

- `n_traj>=32`, 3 seeds;
- `K>=8`;
- same bootstrap and independent-unit rules as GS16--GS19.

## 7. Cross-family claim rule

A phenomenon may be called cross-architecture only if:

1. it appears in at least ELF, LangFlow, and CDCD;
2. the same operational metric is informative in all three;
3. sign consistency has trajectory-level confidence intervals;
4. the result survives log-SNR or event-time alignment.

If CDCD differs, treat the difference as a useful boundary condition and
analyze which parameterization/readout property predicts it.

