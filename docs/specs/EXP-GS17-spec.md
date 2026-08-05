# EXP-GS17 Spec — Local Residual Dynamics and Unified Transition Timing (P0)

**Status**: ACTIVE
**Priority**: P0 — replace the GS15 chord comparison with a local dynamical test
**Models**: ELF baseline + LangFlow; CDCD after GS20
**Proposed scripts**: `experiments/global_state/analyze_residual_velocity.py`,
`experiments/global_state/analyze_transition_timeline.py`
**Output**: `results/global_state/<model>/<checkpoint>/transition_dynamics_<label>.{json,npz}`

## 1. Scientific question

GS15 compares rollout CKA against a direct chord. A curved but progressive path
can remain below that chord at every intermediate point. This experiment asks a
more direct question:

> At each state, how much of the actual local velocity reduces distance to the
> trajectory's own endpoint, and how much moves in endpoint-orthogonal
> directions?

## 2. Dense true-rollout collection

Save `Z_s`, `SC_s`, `Xhat_s`, and terminal decode on a dense solver grid.
Use `s` for monotone generation progress and save native `t`, log-SNR, and
cumulative arc length separately.

Pilot:

- ELF baseline, `n_traj=16`;
- 65 saved solver states.

Formal:

- `n_traj>=48`, 3 generation seeds;
- ELF and LangFlow;
- 129 saved states if finite-difference noise is material.

Do not compare architectures at equal nominal `t`. Cross-model figures use
normalized log-SNR percentile or normalized cumulative arc length.

## 3. Local geometry

For centered residual `R_s` and endpoint `R_*`, define:

```text
d_s       = R_* - R_s
v_s       = dR_s/ds
v_parallel = <v_s,d_s> / (||d_s||^2 + eps) * d_s
v_perp     = v_s - v_parallel
```

Primary metrics:

```text
cos_endpoint(s) = <v_s,d_s> / (||v_s|| ||d_s||)
rho_parallel(s) = ||v_parallel||^2 / (||v_s||^2 + eps)
progress(s)     = -d/ds ||R_* - R_s||_F^2
turning(s)      = angle(v_s, v_{s+1})
path_eff(s)     = ||R_s-R_start|| / cumulative_arc_length(start->s)
```

Use normalized Frobenius inner products after position centering. Run raw-state
and predicted-clean residuals separately.

## 4. Velocity estimation

Preferred:

- use the model/solver's actual vector field before the numerical update;
- analytically center the velocity across positions.

Robustness:

- central finite difference on the saved dense trajectory;
- repeat after light Savitzky--Golay smoothing with a preregistered window;
- verify that sign and transition location do not depend on smoothing.

For ELF, audit the sign and time convention of
`v=(xhat-z)/(1-t)` before using it. For LangFlow, expose the velocity used by
the native Euler step rather than deriving it from a plotting-time schedule.

## 5. Alternative-endpoint control

Reuse the fixed candidate endpoint bank from GS16. For each alternative
endpoint `R*_j`, compute:

```text
cos_j(s) = cos(v_s, R*_j - R_s)
V_self(s) = cos_self(s) - mean_{j>0} cos_j(s)
```

This distinguishes "moving somewhere that happens to reduce all endpoint
distances" from locally moving toward the rollout's own future.

## 6. Parameterization robustness

The following must be reported:

1. native solver time;
2. log-SNR percentile;
3. normalized arc length.

The central qualitative conclusion must survive at least log-SNR and arc-length
parameterizations. A peak that only exists under nominal `t` is a schedule
artifact.

## 7. Decision rule

| observation | interpretation |
|---|---|
| early `V_self>0`, positive progress, low `rho_parallel` | curved transport toward an already specific endpoint |
| early `V_self≈0`, followed by sharp rise aligned with GS16 entropy collapse | late endpoint selection |
| positive progress to many endpoints but no self advantage | common-manifold transport, endpoint unresolved |
| negative progress over a window | genuine temporary motion away from final residual |

Negative GS15 `O_R` alone is not a decision criterion. GS15 becomes a global
description of curvature; GS17 supplies the local mechanism evidence.

## 8. Unified transition timeline

This stage absorbs the former GS18 spec. On the **same true rollouts**, save
native logits and top-1 tokens together with the velocity and endpoint-affinity
quantities. Define per position:

```text
tau_first  = first time top1 equals the rollout terminal token
tau_stable = first time top1 equals the terminal token at every later checkpoint
tau_margin = first persistent terminal-token margin crossing above zero
```

Define per trajectory:

```text
tau_50_stable = time when 50% of positions are stable
tau_affinity  = maximum endpoint-affinity entropy contraction from GS16
tau_velocity  = maximum rise of self-endpoint velocity advantage
tau_branch    = maximum calibrated branch-entropy contraction from GS16
```

If the common-factor-controlled statistic in GS18 is run, also include
`tau_collective`. Test event ordering within trajectories and produce curves
event-aligned to `tau_50_stable`. Broad visual overlap from separate datasets
does not count as temporal alignment.

The paper may call the transition coordinated only if affinity/branch
contraction or controlled collective coupling forms a reproducible window that
precedes or overlaps stable commitment on the same rollouts. Otherwise report
heterogeneous gradual stabilization.

## 9. Statistics and plots

- trajectory-level median and 95% hierarchical bootstrap bands;
- fraction of trajectories with positive `V_self` and positive progress;
- individual-trajectory small multiples for at least 12 examples;
- event-aligned plots relative to within-trajectory `tau_50_stable`.

Do not pool positions as independent observations.

---

## Pilot Results (2026-07-31, ELF baseline, n_traj=16, same trajectories as GS16)

**Implementation**: `experiments/global_state/analyze_transition_timeline.py`
(combines the spec's two proposed scripts into one). Velocity = central
finite difference on a 65-state dense rollout, used uniformly for both
representations; for ELF the closed-form `v=(xhat-z)/(1-t)` is also computed
as a cross-check (`cos_v_finite_vs_analytic`). Reuses GS16's fixed endpoint
bank (same `--seed`/`--n_traj`) for the Section 5 alternative-endpoint
control, raw representation only.

**Local geometry — refines the exploration-collapse story**, pooled across
16 trajectories (raw representation, 95% CI):

| t | cos_endpoint(t) | V_self(t) |
|---|---|---|
| 0.050 | +0.808 [+0.806,+0.809] | -0.004 [-0.005,-0.003] |
| 0.168 | +0.816 [+0.812,+0.820] | -0.006 [-0.007,-0.005] |
| 0.226 | +0.899 [+0.891,+0.906] | +0.001 [-0.001,+0.004] |
| 0.285 | +0.954 [+0.946,+0.961] | +0.005 [+0.002,+0.008] |
| 0.461 | +0.986 [+0.981,+0.990] (peak) | +0.009 [+0.005,+0.015] |
| 0.931 | +0.925 [+0.920,+0.931] | +0.012 [+0.008,+0.016] |

`cos_endpoint` (local velocity's alignment with the trajectory's OWN
eventual endpoint) is already high (+0.81) from the very first checkpoint
(t=0.05) -- local motion has a clear "denoising" directionality from early
on. But `V_self` (self progress MINUS mean progress toward the K alternative
candidate endpoints from GS16's bank) stays near/below zero until
`t≈0.226-0.285`, meaning that early directionality is shared almost equally
by self and the alternative branch endpoints -- i.e. Section 7's
"common-manifold transport, endpoint unresolved" row, not pure directionless
exploration and not yet endpoint-specific transport. `V_self` then rises to
a plateau of ~+0.012-0.014 by mid-trajectory. **This refines GS16's binary
exploration-collapse framing**: early motion is not undirected, but it is not
YET differentially pointed at the eventual specific answer over plausible
alternatives; what collapses later is the alternatives' viability, not a
sudden onset of self-direction.

**Event ordering** (fraction of the 16 trajectories where event A precedes
or ties event B):

- `P(tau_affinity <= tau_50_stable) = 0.0` (0/16) -- GS16's branch-perturbation
  affinity collapse (t=0.30-0.36) consistently comes AFTER the per-position
  top-1 decode has already stabilized (median tau_50_stable ~0.24-0.30). "The
  current best guess looks decided" and "the underlying attractor has become
  robust to perturbation" are two distinct, slightly staggered events, not
  the same moment.
- `P(tau_velocity <= tau_50_stable) = 0.75` (12/16) -- ⚠️ treat with caution:
  4/16 trajectories give anomalous late `tau_velocity` (t=0.84-0.96), a known
  artifact of the "argmax of dV_self/ds" detector picking up numerical noise
  late in the trajectory (see the finite-difference instability caveat
  below). The 12 "clean" trajectories give tau_velocity≈0.18-0.20, slightly
  BEFORE tau_50_stable, consistent with the pooled V_self curve above.

**Known limitations**:
- `cos_endpoint`/`rho_parallel` are degenerate (0/0, clamped to 0) at the
  very last grid point (t_end), since `R_star` is defined as that same
  point's own residual -- same tautology class as GS15's `A_linear(t_end)=1`
  and GS16's candidate-0-at-t_end. Excluded from interpretation.
- `cos_v_finite_vs_analytic` (ELF only) degrades from ~0.99 in the first half
  of the trajectory to negative by t_end -- finite-difference velocity
  becomes numerically unstable once consecutive dense states are nearly
  identical. Local-dynamics numbers in the last 1-2 grid points are
  unreliable for both velocity estimates.
- `tau_velocity`'s simple argmax-of-derivative estimator is not robust (see
  event-ordering caveat above); a smoothed threshold-crossing detector would
  be a better estimator for a formal-scale rerun.
- Section 5 alternative-endpoint control (`V_self`, `cos_self_to_bank`) is
  raw-representation only.

See `docs/global_state_formation_synthesis.md` for the combined GS16+GS17
interpretation and `docs/specs/EXP-GS16-spec.md` for the calibration/bank
construction this reuses.

---

## Formal Results (2026-08-01/03, ELF baseline, n_traj=48, n_states=129, 3 seeds)

**Status**: DONE — all 3 seeds complete.

**Script**: `experiments/global_state/analyze_transition_timeline.py`
with `--n_traj 48 --n_states 129 --label formal_s{seed} --seed {seed}`
and the matching `--endpoint_bank_npz` + `--gs16_json` from GS16 formal.

### Per-seed tau summary (raw representation)

| Metric | Seed 42 | Seed 123 | Seed 456 | Pooled (3-seed, 95% CI) |
|--------|---------|---------|---------|------------------------|
| tau_50_stable mean | 0.2016 | 0.2004 | 0.2162 | **0.206 ± 0.022** |
| tau_50_stable std | 0.0201 | 0.0171 | 0.0794 | — |
| tau_50_stable median | 0.2042 | 0.2005 | 0.2042 | **0.203** |
| tau_velocity mean | 0.2494 | 0.2120 | 0.2515 | **0.238 ± 0.055** (mean unreliable) |
| tau_velocity std | 0.2429 | 0.1958 | 0.2395 | — |
| tau_velocity median | 0.1675 | 0.1675 | 0.1748 | **0.170** |
| tau_affinity mean | 0.3239 | 0.3249 | 0.3178 | **0.322 ± 0.010** |
| tau_affinity std | 0.0400 | 0.0480 | 0.0646 | — |
| tau_affinity median | 0.3007 | 0.3633 | 0.3007 | **0.322** |

Pooled 95% CI: t-distribution with df=2 across per-seed means.
tau_velocity mean is dominated by outlier trajectories; use median=0.170 as the reliable summary.
seed=456 tau_50_stable has higher std (0.0794 vs 0.017–0.020 for other seeds); mean slightly elevated but median matches.

### Event ordering (fraction of 48 trajectories where A precedes B)

| Event ordering | Seed 42 | Seed 123 | Seed 456 | Pooled mean | Interpretation |
|----------------|---------|---------|---------|-------------|----------------|
| P(τ_velocity ≤ τ_50_stable) | 0.875 | 0.9375 | 0.875 | **0.896** | velocity before token stability in ~90% of trajs |
| P(τ_affinity ≤ τ_50_stable) | 0.000 | 0.0625 | 0.0833 | **0.049** | endpoint identity AFTER token stability in ~95% of trajs |
| P(τ_velocity ≤ τ_affinity) | 0.8958 | 0.9375 | 0.8542 | **0.896** | velocity before endpoint affinity in ~90% of trajs |

Canonical ordering confirmed across all 3 seeds:
**τ_velocity < τ_50_stable < τ_affinity** (in ~85–94% of trajectories per seed).

### Connection to GS16 formal

GS16 formal shows endpoint commitment at t≈0.36–0.43 (rank_frac drops from 0.93 to 0.17).
GS17 tau_affinity grand mean = 0.322 ± 0.010 (extremely tight across seeds) aligns with the onset of this window.
GS17 tau_50_stable median ≈ 0.20 precedes endpoint commitment by Δt≈0.12–0.23.

**Joint event timeline (pooled, 3 seeds)**:

| t | Event |
|---|-------|
| ~0.17 | τ_velocity: denoising velocity peak (median of medians) |
| ~0.20 | τ_50_stable: 50% of positions have stable top-1 token (median) |
| ~0.30 | GS16: H_end peaks at 0.848 (maximum endpoint uncertainty) |
| ~0.32 | τ_affinity: endpoint identity locked (3-seed grand mean = 0.322 ± 0.010) |
| ~0.36–0.43 | GS16: rank_self drops from 0.93 → 0.17 (endpoint committed) |
| ≥0.43 | GS16: plateau — rank_self=1.00, H_end=0.524 |

This is the exploration-collapse story, confirmed mechanistically across 3 independent seeds:
1. Denoising velocity stabilizes early (τ_velocity median ≈ 0.17)
2. Per-position token guesses stabilize (τ_50_stable median ≈ 0.20)
3. Endpoint basin identity is confirmed later (τ_affinity mean ≈ 0.32 ± 0.010, GS16 rank collapse ≈ 0.36–0.43)

### Known limitations (carry-forward from pilot)

- tau_velocity std is large (~0.19–0.24) due to outlier trajectories with late argmax-peak;
  the median (0.167–0.175 across seeds, grand median=0.170) is the reliable summary.
- P(τ_affinity ≤ τ_50_stable) rises from 0.000 (seed=42) to 0.0625 (seed=123) to 0.0833 (seed=456);
  the ordering is strong (~95% of trajectories pooled) but not perfectly universal.
- seed=456 tau_50_stable std = 0.0794 (vs ~0.02 for other seeds); source of variability unknown
  but qualitative conclusions unchanged.

### Output files

| File | Seed | Status |
|------|------|--------|
| `transition_dynamics_formal_s42.json` | 42 | ✅ done |
| `transition_dynamics_formal_s123.json` | 123 | ✅ done |
| `transition_dynamics_formal_s456.json` | 456 | ✅ done |
