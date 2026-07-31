# EXP-GS16 Spec — Fixed-Bank Endpoint Specificity and Affinity Collapse (P0)

**Status**: PLANNED  
**Priority**: P0 — 决定论文能否区分 curved transport 与 exploration--collapse  
**Models**: ELF baseline first; then LangFlow; CDCD after GS23 adapter  
**Proposed script**: `experiments/global_state/analyze_endpoint_specificity.py`  
**Output**: `results/global_state/<model>/<checkpoint>/endpoint_specificity_<label>.{json,npz}`

## 1. Scientific question

GS15 only shows that a rollout is less similar to its endpoint than a direct
chord. It does **not** tell us whether the endpoint is already selected.
This experiment asks:

> At an intermediate state, is the trajectory already more compatible with
> its own eventual endpoint than with other locally reachable endpoints?

The two competing mechanisms make different predictions:

| mechanism | early self-endpoint advantage | affinity entropy | relation to stable token time |
|---|---:|---:|---|
| curved transport | positive early | low or smoothly decreasing | self-specificity precedes stability |
| exploration--collapse | near zero early | high, then sharp fall | entropy collapse aligns with stability |

## 2. Why the candidate endpoint bank must be fixed

Do **not** create a new branch set independently at every checkpoint and then
compare entropy across time. Later checkpoints have less remaining integration
time, so their branch endpoints will mechanically be more similar.

For each base trajectory `n`, create one fixed candidate endpoint bank from an
early split `t_bank`:

1. Run the unperturbed rollout and save dense states `(Z_t, SC_t)`.
2. At `t_bank`, produce `K` calibrated perturbations and continue all branches.
3. Candidate `j=0` is the unperturbed rollout endpoint.
4. Candidates `j=1..K` are the branch endpoints.
5. Deduplicate endpoints by exact token sequence and retain multiplicities.
6. Use this **same endpoint bank** to score every later state of the
   unperturbed base trajectory.

Primary `t_bank=0.20` for ELF. Also run `t_bank=0.05` as a robustness check.
For other models, match `t_bank` by native log-SNR rather than nominal `t`.

## 3. Representations and similarities

For every saved state and endpoint, compute centered residuals

```text
R_t = Z_t - mean_position(Z_t)
R*_j = Z*_j - mean_position(Z*_j)
```

Run the analysis separately on:

- raw sampler residual `R_t^Z`;
- predicted-clean residual `R_t^Xhat`.

Use two primary similarities:

1. normalized Frobenius cosine after position centering;
2. linear CKA, only as a robustness metric.

The cosine is primary because CKA is a representation-similarity statistic, not
a vector-direction measure. All claims must survive both or be described as
metric-dependent.

## 4. Primary metrics

For trajectory `n`:

```text
a_j(t)       = similarity(R_t, R*_j)
S_self(t)    = a_0(t) - mean_{j>0} a_j(t)
rank_self(t) = rank of a_0 among all candidate affinities
```

Affinity entropy:

```text
p_j(t; beta) = softmax(beta * zscore_j(a_j(t)))
H_end(t)     = -sum_j p_j log p_j / log(K+1)
N_eff(t)     = exp(H_raw(t))
```

Report a beta sweep `{0.5, 1, 2, 4}`. No beta may be selected using the desired
collapse time. The non-parametric `rank_self` and `S_self` are the primary
claims; entropy is secondary.

Subtract the start-state baseline as a robustness check:

```text
Delta a_j(t) = a_j(t) - a_j(t_bank)
```

This prevents shared early noise from being mistaken for endpoint specificity.

## 5. Controls

1. **Cross-trajectory endpoint null**: replace candidates with endpoints from
   other trajectories matched by sequence length and terminal token entropy.
2. **Lexical-distance stratification**: report results separately for
   alternative endpoints with low, medium, and high token Hamming distance.
3. **Multiplicity-aware analysis**: if several branches produce the same
   endpoint, report both unique-endpoint entropy and basin-mass entropy.
4. **Position-shuffled endpoint**: shuffle positions within `R*_j`; affinity
   should collapse if the metric is genuinely position-specific.
5. **Mean-only analysis**: repeat on broadcast means to show the result is not
   document-level mean similarity.

## 6. Scale and statistics

Pilot:

- ELF baseline;
- `n_traj=16`, `K=8`;
- 17 checkpoints covering the full rollout;
- one perturbation scale calibrated by GS19.

Formal:

- `n_traj>=48`, `K=12`;
- at least 3 independent generation seeds;
- ELF and LangFlow, then CDCD.

The independent unit is the **base trajectory**. Bootstrap trajectories
hierarchically; do not treat tokens, branches, or checkpoints as independent.
Save all per-trajectory affinity matrices in `.npz`.

## 7. Decision rule

Evidence for curved transport requires all of:

1. median `S_self(t)>0` significantly before median stable-token time;
2. self endpoint ranks above chance before branch entropy contracts;
3. early velocity is mostly orthogonal but still has positive endpoint progress
   in GS17.

Evidence for exploration--collapse requires all of:

1. early `S_self(t)` is indistinguishable from zero;
2. `H_end(t)` or `rank_self(t)` changes sharply in a narrow window;
3. that window aligns with stable commitment in GS18.

If self-specificity is metric-dependent or only appears after stable
commitment, report the mechanism as unresolved.

## 8. Failure interpretation

- No distinct branch endpoints: perturbation is too weak or the bank was built
  too late; rerun GS19 calibration before changing the hypothesis.
- Strong early self-specificity only in predicted-clean but not raw state:
  the model prediction contains an endpoint proposal that has not yet been
  integrated into the sampler state.
- High early self-specificity under both real and position-shuffled endpoints:
  the similarity is dominated by global statistics and is invalid.

