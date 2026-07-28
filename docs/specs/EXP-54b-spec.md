# EXP-54b: Multi-Seed Variance Validation for h₁₀ SC

**Status**: DONE  
**Date**: 2026-07-27  
**Script**: `models/ELF-torch/experiments/probe_elf/h10_sc_exp54b_multiseed.py`  
**Results**: `models/ELF-torch/results/exp54b_multiseed/results.json`

## Motivation

EXP-54 (single seed=42, N=256) gave I = −129.3 for h₁₀ SC sccfg=1, but with no confidence
interval. Two rigor questions:
1. Is "natural sccfg=3 slightly worse than sccfg=1" (ΔPPL=+11) a stable finding or seed noise?
2. What is the 95% CI on I(h₁₀ SC)?

## Protocol

- **Checkpoint**: kd2 (`converted/elf_b-owt-kd2_torch.pt`)
- **Seeds**: 42, 123, 456 (each N=256 sequences)
- **Arms**: natural sccfg=1 (reference), natural sccfg=3, h₁₀ SC sccfg=1
- **ODE steps**: 32, time_schedule=uniform, SC_T_MIN=0.5

## Results

### Per-Seed PPL

| Seed | natural sccfg=1 | natural sccfg=3 | h₁₀ SC sccfg=1 | I (h₁₀ − nat1) |
|------|-----------------|-----------------|-----------------|-----------------|
| 42   | 284.7           | 295.7           | 155.4           | −129.3          |
| 123  | 308.0           | 314.3           | 174.3           | −133.6          |
| 456  | 306.7           | 318.2           | 180.6           | −126.2          |

### Aggregated (mean ± 95% CI, t-distribution with 2 df)

| Arm | Mean | 95% CI margin | Std |
|-----|------|--------------|-----|
| natural sccfg=1 | 299.8 | ±32.5 | 13.1 |
| natural sccfg=3 | 309.4 | ±29.9 | 12.0 |
| h₁₀ SC sccfg=1 | 170.1 | ±32.6 | 13.1 |
| **I(h₁₀ − natural)** | **−129.7** | **±9.3** | **3.7** |

**Delta(nat_sccfg3 − nat_sccfg1) = +9.6** across all seeds.

## Key Findings

1. **I is robustly −130**: The 95% CI is [−120.4, −139.0]. Even at the conservative lower bound,
   h₁₀ SC gives >40% PPL reduction. The EXP-54 single-seed estimate (I=−129) was accurate.

2. **sccfg=3 consistently hurts natural SC**: ΔPPL = +11.0, +6.3, +11.5 across seeds (mean +9.6).
   This is a stable finding, not seed noise. Confirms EXP-44 B11 anti-correlation for kd2.

3. **Seed variability is large in absolute PPL (σ≈13), small in I (σ=3.7)**:
   The natural PPL ranges from 284.7 to 308.0 (a 23 PPL swing), but the improvement
   I(h₁₀ SC) is remarkably stable: std=3.7, which is 2.9% of the mean I=−129.7.
   This means h₁₀ SC consistently improves by the same amount regardless of how good the
   natural generation happens to be on a given seed.

4. **Paper claim updated**: "45% PPL reduction" → using mean values: 129.7/299.8 = 43.3%.
   The paper can safely cite "~43% PPL reduction (I = −130 ± 9, 95% CI over 3 seeds)."

## Paper Implications

- Replace EXP-54 single-seed citation with EXP-54b aggregate where possible
- Cite: "h₁₀ SC reduces kd2 PPL by 43% (I = −130 ± 9, 95% CI, N=768 sequences, 3 seeds)"
- The "sccfg=3 makes natural arm worse" finding (EXP-44) is confirmed with 3 seeds
