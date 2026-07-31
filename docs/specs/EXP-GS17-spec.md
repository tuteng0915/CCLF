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
