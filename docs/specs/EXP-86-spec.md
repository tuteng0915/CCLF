# EXP-86 Spec — Noise-Debiased Plaid Drift

**Status:** IMPLEMENTED / P1
**Purpose:** separate genuine Plaid dynamics from ancestral-noise contamination
of adjacent-state finite differences.

At each saved Plaid state, estimate conditional mean drift using at least 16
exact antithetic `xi/-xi` one-step transitions. Report diffusion covariance,
Monte Carlo error, and convergence at `K={4,8,16,32}` on a fixed panel.

Recompute endpoint-parallel/orthogonal velocity, candidate-specific `V_self`,
`tau_velocity`, and its order relative to stability/affinity. A second stage
recomputes GS18-B susceptibility from mean drift increments.

- **Noise confound:** debiasing recovers ELF-like dynamics while single-path
  finite differences remain discrepant.
- **Architecture boundary:** the discrepancy survives at converged K and
  adequate drift SNR.
- **Unresolved:** estimator variance remains comparable to the effect.

Runner: `experiments/global_state/analyze_plaid_debiased_drift.py`.
