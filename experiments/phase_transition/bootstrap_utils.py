"""Shared sequence-level bootstrap CI utility for the phase_transition suite.

The suite doc's shared protocol (docs/phase_transition_experiment_suite.md
section 2) explicitly requires "Bootstrap confidence intervals by sequence,
not by token position" -- this was not implemented anywhere in the initial
PT1-10 pass (see EXP-INDEX.md's rigor audit). This module is the shared
utility for retrofitting it; per-experiment scripts import bootstrap_ci().

Resampling by SEQUENCE (not by position) matters because positions within
the same sequence share context and are not independent -- resampling
individual positions would understate the true uncertainty.
"""

import numpy as np


def bootstrap_ci(values_per_seq, n_boot=2000, ci=0.95, seed=42, statistic=np.mean):
    """values_per_seq: (N,) array, one scalar summary per SEQUENCE (already
    reduced over positions within that sequence). Returns
    (point_estimate, lo, hi, boot_std)."""
    values_per_seq = np.asarray(values_per_seq)
    values_per_seq = values_per_seq[np.isfinite(values_per_seq)]
    N = len(values_per_seq)
    if N < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boot_stats[b] = statistic(values_per_seq[idx])
    lo, hi = np.percentile(boot_stats, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(statistic(values_per_seq)), float(lo), float(hi), float(boot_stats.std())


def bootstrap_ratio_ci(numer_per_seq, denom_per_seq, n_boot=2000, ci=0.95, seed=42):
    """CI for a ratio of two per-sequence quantities (e.g. m_res/m_raw),
    resampling sequences jointly (same bootstrap indices for both arrays)."""
    numer = np.asarray(numer_per_seq)
    denom = np.asarray(denom_per_seq)
    valid = np.isfinite(numer) & np.isfinite(denom) & (denom != 0)
    numer, denom = numer[valid], denom[valid]
    N = len(numer)
    if N < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot_ratios = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        boot_ratios[b] = numer[idx].mean() / denom[idx].mean()
    lo, hi = np.percentile(boot_ratios, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    point = numer.mean() / denom.mean()
    return float(point), float(lo), float(hi)
