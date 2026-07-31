# EXP-GS19 Spec — Calibrated Branching and Remaining-Time Controls (P0 Prerequisite)

**Status**: PLANNED  
**Priority**: P0 prerequisite — calibrate the endpoint bank used by GS16 and rule out the trivial explanation of GS14 branch contraction  
**Models**: ELF baseline first; LangFlow after pilot  
**Proposed script**: `experiments/global_state/branch_calibrated_trajectory.py`  
**Output**: `results/global_state/<model>/<checkpoint>/branch_calibrated_<label>.{json,npz}`

## 1. Confound being tested

GS14 perturbs checkpoints with a fixed state-relative scale. Later checkpoints
have less remaining integration time and may amplify perturbations less.
Increasing branch consensus can therefore occur without any lexical
exploration--collapse.

GS19 separates:

1. local sensitivity of the flow;
2. remaining-time amplification;
3. genuine contraction of reachable lexical endpoints.

## 2. Three perturbation protocols

For the same base rollout and direction bank `u_k`, run:

### A. Fixed state-relative norm

Existing reference:

```text
delta_tk = eta * ||Z_t,i|| * unit(u_k)
```

### B. One-step matched impact

Choose `eta_t` by pilot bisection so the perturbation produces a fixed median
relative divergence after one native solver step:

```text
median ||step(Z_t+delta)-step(Z_t)|| / ||step(Z_t)|| = kappa_step
```

Use `kappa_step in {1e-4, 3e-4, 1e-3}`.

### C. Terminal-linearized matched impact

Estimate random-direction amplification of the remaining flow with JVPs:

```text
g_t(u) = ||J_{t->*} u|| / ||u||
eta_t  = kappa_terminal / median_u g_t(u)
```

Then branch with `eta_t` so the predicted continuous terminal displacement is
matched across split times. If full-rollout JVP is too expensive, use a
finite-difference estimate on 8 pilot directions and mark it as approximate.

## 3. Metrics

- immediate continuous divergence;
- terminal continuous divergence;
- lexical entropy and normalized consensus across `K` endpoints;
- agreement with the unperturbed endpoint;
- pairwise branch agreement;
- number of unique endpoint sequences;
- per-position terminal token entropy;
- semantic distance and exact-token Hamming distance.

Use Miller--Madow entropy correction because `K` is small. Report raw counts in
addition to normalized entropy.

## 4. Scale

Pilot:

- ELF baseline;
- `n_traj=12`, `K=8`;
- split checkpoints matched to GS14;
- three calibration targets.

Formal:

- `n_traj>=32`, `K=12`;
- 3 seeds;
- bootstrap base trajectories, keeping all branches nested.

## 5. Decision rule

Evidence for genuine lexical basin contraction requires branch entropy to
decrease with generation progress under protocols B and C, not only A.

| result | interpretation |
|---|---|
| contraction survives B and C | reachable lexical futures genuinely contract |
| disappears under C | GS14 was mainly remaining-time/Jacobian amplification |
| continuous divergence matched but lexical entropy falls | specifically lexical basin formation |
| pairwise branches agree but differ from original endpoint | bifurcation into a common alternative basin |

GS19's calibrated branches become the fixed endpoint bank used by GS16.
