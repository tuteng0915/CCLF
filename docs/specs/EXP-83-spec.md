# EXP-83 Spec — Formal Cross-Architecture Endpoint Collapse

**Status:** READY / P0
**Purpose:** test whether endpoint exploration--collapse is cross-architecture
rather than an ELF-only observation.

Run LangFlow and Plaid with model-specific branch-point and log-SNR
calibration. Never reuse an ELF nominal time. Use at least 32 trajectories,
8 deduplicated branches, and seeds 42/123/456. Plaid pairs ancestral noise.

Report centered-cosine self specificity, self-endpoint rank, normalized
endpoint entropy, effective candidate count, and affinity-collapse time, with
position-shuffled and norm-matched random endpoint controls.

Gates require native reference agreement 1.0, at least four distinct endpoints
on average, model-native time calibration, and shuffled specificity near zero.

- **Confirmation:** early self specificity is indistinguishable from zero and
  rank/entropy collapse in a narrow, seed-stable window on both models.
- **Architecture boundary:** the effect survives on only one model after native
  calibration.
- **Measurement failure:** branch diversity or null gates fail.

Runner: the existing
`experiments/global_state/analyze_endpoint_specificity.py` via LangFlow/Plaid
adapters, preceded by bounded branch-point calibration pilots.
