# EXP-59: Multi-Seed Validation of kd_cr Pipeline-Avg

**Status**: DONE  
**Date**: 2026-08-01  
**Model**: kd_cr only  
**Script**: `experiments/probe_elf/pipeline_avg_multiseed_exp59.py`  
**Results**: `models/ELF-torch/results/exp59_pipeline_avg_multiseed/results_seed{42,123,456}.json`

---

## Motivation

EXP-58 identified kd_cr pipeline_avg as the single honest positive result
(I=−142, D1/D2 both up, ROUGE-L not catastrophic). That measurement was a
single seed (42). EXP-59 replicates with 3 seeds to establish variance and
95% CI, confirming the result is robust before reporting in paper.

---

## Setup

| Parameter | Value |
|-----------|-------|
| Arms | standard (ODE-32), pipeline_avg (T=16, 31 calls) |
| Model | kd_cr only |
| Seeds | 42, 123, 456 |
| N per seed | 256 sequences |
| MAX_LENGTH | 128 tokens |
| T_PIPE | 16 groups |
| Statistical test | pooled t (df=2, t₀.₀₂₅=4.303) |

---

## Results

### Per-seed raw

| Seed | Arm | PPL | I | D1 | D2 |
|------|-----|-----|---|----|----|
| 42 | standard | 338.1 | 0 | 0.197 | 0.780 |
| 42 | pipeline_avg | 196.3 | −141.8 | 0.310 | 0.809 |
| 123 | standard | 363.6 | 0 | 0.199 | 0.778 |
| 123 | pipeline_avg | 197.2 | −166.4 | 0.309 | 0.814 |
| 456 | standard | 343.0 | 0 | 0.204 | 0.790 |
| 456 | pipeline_avg | 190.4 | −152.6 | 0.304 | 0.807 |

### Pooled (mean ± 95% CI, n=3, df=2)

| Arm | PPL | I | D1 | D2 |
|-----|-----|---|----|----|
| standard | 348.2 ± 33.7 | 0 | 0.200 ± 0.009 | 0.783 ± 0.016 |
| **pipeline_avg** | **194.6 ± 9.2** | **−153.6 ± 30.7** | **0.308 ± 0.009** | **0.810 ± 0.009** |

---

## Analysis

### Robust improvement confirmed

The I=−142 single-seed result from EXP-58 replicates cleanly:
- Mean I=−153.6 (slightly larger improvement than seed-42 alone)
- 95% CI: [−184.3, −122.9] — entirely negative; effect is significant
- All 3 seeds agree in direction (−141.8, −166.4, −152.6); low variance (σ≈12)

### Diversity signal is healthy and consistent

- D1 rises from 0.200 → 0.308 across ALL 3 seeds (+54% mean)
- D2 rises from 0.783 → 0.810 across ALL 3 seeds (+3.4%)
- The D1 increase means pipeline_avg generates substantially more distinct unigrams
  than standard ODE — the opposite of PPL hacking (which collapses D1)

### No seed sensitivity

Standard deviation of I across seeds: σ=12.3 → CV=8%. This is low enough that
the effect is not a statistical accident. The pipeline_avg schedule is
structurally reliable for kd_cr.

### Comparison to EXP-54b best result

EXP-54b h10_only (kd2): I=−130 ± 9 (95% CI), validated as genuine.
EXP-59 kd_cr pipeline_avg: I=−154 ± 31 — stronger improvement, wider CI
(fewer seeds in EXP-54b was n=5, df=4). CI overlaps; pipeline_avg is not
definitively stronger but is in the same regime and genuinely honest.

---

## Key Conclusions

1. **kd_cr pipeline_avg is the strongest validated result in the series**:
   I=−153.6 ± 30.7 (95% CI), D1/D2 both consistently up.

2. **Effect is robust to seed**: σ(I)=12 across seeds; all 3 seeds show
   clear improvement with no outlier.

3. **31 model calls vs 32** for standard ODE means pipeline_avg achieves
   better PPL at slightly lower compute (though note: 31 calls at varying t
   vs 32 uniform steps — the comparison is not fully apples-to-apples).

4. **Paper-ready**: this is the result to report as the primary finding for
   pipeline ODE. I=−153.6 ± 30.7 with D1↑54% and D2↑3%.

---

## Reference Baselines

| Method | Model | I | D1 | D2 | Notes |
|--------|-------|---|----|----|-------|
| EXP-54b h10_only | kd2 | −130 ± 9 | — | — | Valid; 5 seeds |
| EXP-58 pipeline_avg | kd_cr | −142 | 0.310 | 0.809 | Single seed (seed=42) |
| **EXP-59 pipeline_avg** | **kd_cr** | **−154 ± 31** | **0.308 ± 0.009** | **0.810 ± 0.009** | **3-seed validated** |
| EXP-58 pipeline_global | kd_cr | −301 | 0.211 | 0.604 | PPL hacking (ROUGE-L=0.012) |
