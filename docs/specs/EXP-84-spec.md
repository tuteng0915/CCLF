# EXP-84 Spec — Counterfactual Endpoint Steering

**Status:** P0 DONE / CAUSAL DIRECTION POSITIVE, RIGIDITY NON-MONOTONE
**Purpose:** convert endpoint collapse from an observational curve into a
causal basin-rigidity test.

For state `z_t`, self endpoint `e_0`, and reachable alternative `e_j`, perturb
the centered residual along the normalized contrast `e_j-e_0` using magnitude
`epsilon ||R_t||_F`. Compare alternative, opposite/self, random-orthogonal,
position-shuffled, and no-perturbation directions at times on both sides of the
collapse window. Continue with paired native rollout.

Report alternative capture, self retention, token agreement with both
endpoints, and

```text
epsilon_50(t) = min epsilon such that P_redirect(t, epsilon) >= .5.
```

- **Causal exploration--collapse:** alternative steering works before collapse,
  `epsilon_50` rises across the collapse window, and matched nulls do not.
- **Predetermined transport:** alternative steering is weak from the earliest
  time.
- **Generic fragility:** random and endpoint directions redirect equally.

Start with deterministic ELF formal GS16 banks, then promote to LangFlow.
Runner: `experiments/global_state/intervene_endpoint_steering.py`.
