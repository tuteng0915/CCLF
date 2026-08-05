# EXP-58: Pipeline ODE (Diffusion Forcing Approximation) + Quality Evaluation

**Status**: DONE  
**Date**: 2026-08-01  
**Models**: kd_cr, kd2  
**Script**: `experiments/probe_elf/pipeline_ode_exp58.py`  
**Results**: `results/exp58_pipeline_ode/results.json`

---

## Motivation

EXP-56b showed I=−186 for progressive commitment (kd2 prog_t30_c70), but quality eval revealed
PPL hacking: D1 dropped 44%, func_frac=60.7% of committed tokens were function words ("and and on
on..."). Pipeline ODE is a conceptually different approach — no commitment/locking, just asynchronous
denoising where leading position groups run ahead by one step at a time.

User's concept: "前序的token步长大一些，后续的token慢慢跟上，流水线作业" — leading positions
get more denoising steps early, trailing positions attend to them via attention and benefit from
richer context. Not per-position t_i (would require ELF architectural change); instead, a shared
scalar t per model call with only active positions updated.

---

## Setup

| Parameter | Value |
|-----------|-------|
| T_PIPE | 16 groups → 2T-1=31 total model calls |
| N_UNCOND | 256 sequences |
| N_COND | 128 prefix+suffix pairs |
| MAX_LENGTH | 128 tokens |
| PREFIX_LEN | 64 tokens (NLTK gutenberg) |
| seed | 42 |
| Baseline | ODE-32 (standard N_STEPS=32) |

Two pipeline variants:
- **global_t**: t = k/31, uniform 0→1 over 31 steps; trailing positions only see Δt_total≈0.97; decoded from x_pred (trailing z doesn't reach t=1.0)
- **avg_t**: t = avg effective t of active groups; each group gets Δt=1/T per step, total Δt=1.0; decoded from z

Conditioned evaluation: T5-small encoder → x0 = T5_output / latent_std(0.2); prefix positions set as cond_seq; ODE run with cond_mask. Metrics on generated suffix only.

---

## Results

### kd_cr

| Arm | PPL | I | D1 | D2 | ROUGE-L |
|-----|-----|---|----|----|---------|
| standard | 338.1 | 0 | 0.197 | 0.780 | — |
| **pipeline_global** | **37.0** | **−301.0** | 0.211 | 0.604 | — |
| pipeline_avg | 196.3 | −141.8 | 0.310 | 0.809 | — |
| standard_cond | 628.5 | 0 | 0.231 | 0.736 | 0.050 |
| pipeline_global_cond | 210.2 | −418.3 | 0.458 | 0.913 | **0.012** |
| pipeline_avg_cond | 235.9 | −392.6 | 0.481 | 0.942 | 0.027 |

### kd2

| Arm | PPL | I | D1 | D2 | ROUGE-L |
|-----|-----|---|----|----|---------|
| standard | 284.1 | 0 | 0.194 | 0.660 | — |
| pipeline_global | 833.1 | **+549.0** | 0.429 | 0.924 | — |
| pipeline_avg | 484.2 | +200.0 | 0.368 | 0.858 | — |
| standard_cond | 318.4 | 0 | 0.247 | 0.713 | 0.060 |
| pipeline_global_cond | 1246.5 | +928.1 | 0.574 | 0.982 | 0.016 |
| pipeline_avg_cond | 618.7 | +300.3 | 0.541 | 0.961 | 0.027 |

---

## Analysis

### kd_cr pipeline_global: suspicious PPL=37

A PPL of 37.0 for unconditionally generated text is extraordinary — GPT-2 large on real OWT text
scores ~20-30 PPL, so generated text reaching 37 implies something unusual. Key signals:

- D2 drops 22% (0.780→0.604): bigram diversity collapses while D1 stays flat
- Conditioned ROUGE-L = 0.012: the model is NOT capturing prefix content (vs standard_cond 0.050)
- Conditioned D1/D2 both INCREASE (0.458/0.913 vs 0.231/0.736): more diverse but less coherent with prefix

Interpretation: kd_cr pipeline_global appears to be a different form of PPL hacking. With global_t
schedule, leading position groups (j=0-7) run from t=0 to t≈0.48 and effectively "lock in" to some
representation early. These leading positions then provide stable context for trailing positions via
attention. The resulting text may be highly predictable from GPT-2's perspective without being
semantically meaningful continuation of any real prefix.

The conditioned evaluation exposes this: ROUGE-L=0.012 (vs 0.050 for standard ODE) means the
pipeline ODE's generated suffix is essentially uncorrelated with the true continuation, despite
having much lower PPL.

### kd_cr pipeline_avg: healthiest signal

- PPL improves moderately (338→196, I=−142)
- D1 rises 57% (0.197→0.310): substantially more diverse unigrams
- D2 rises 4% (0.780→0.809): no bigram collapse
- ROUGE-L=0.027 (vs standard_cond 0.050): some reduction but not catastrophic

This is the most honest result: avg_t ensures each group accumulates Δt=1.0 total, so the
denoising budget is correct. The improvement (I=−142) is real and not obviously PPL hacking.
The diversity increase is healthy — pipeline ODE generates more varied content than the standard
baseline.

However, I=−142 is weaker than the best progressive commitment result (EXP-56b, I=−186), and that
commitment result was itself PPL hacking. So pipeline_avg kd_cr may still be the most honest
improvement in the series, though the conditioned ROUGE-L (0.027) suggests it still doesn't
generate coherent prefix continuations.

Note on conditioned PPL: standard_cond has PPL=628 >> unconditional 338, because NLTK gutenberg is
heavily OOD from OWT. The pipeline ODE's I=−393 in conditioned setting is relative to this already-
degraded baseline; absolute conditioned PPL (235) is within normal range.

### kd2: catastrophic pipeline failure

Both pipeline variants fail dramatically:
- Global: PPL jumps from 284→833 (I=+549)
- Avg: PPL jumps from 284→484 (I=+200)

D1/D2 both increase substantially, meaning the generated text is MORE diverse but also more
incoherent. kd2's generation quality completely collapses under the asynchronous denoising schedule.

This mirrors the finding from EXP-54c: kd2's h10 SC gate is critically sensitive to the t-schedule
(tmin=0.5 is required; earlier activation is catastrophic). Pipeline ODE violates the implicit
assumption that the model processes each position with a clean monotonic t trajectory from 0 to 1.
For kd2, the asynchronous schedule — where trailing positions see t computed from the global or
average schedule rather than their own denoising progress — breaks the SC interaction.

kd_cr appears more robust to this schedule perturbation, likely because kd_cr's SC interaction is
less sensitive to the precise t value at intermediate steps.

---

## Comparison with Reference Baselines

| Method | Model | I | D1 | D2 | Notes |
|--------|-------|---|----|----|-------|
| EXP-56b prog_t30_c70 | kd2 | −186 | 0.108 | 0.416 | PPL hacking confirmed |
| EXP-56b prog_t40_c70 | kd_cr | −175 | 0.156 | 0.619 | Mild PPL hacking |
| EXP-54b h10_only | kd2 | −130 | — | — | Valid (no diversity collapse) |
| **EXP-58 pipeline_global** | **kd_cr** | **−301** | **0.211** | **0.604** | **Suspicious (D2↓, ROUGE-L=0.012)** |
| EXP-58 pipeline_avg | kd_cr | −142 | 0.310 | 0.809 | Healthiest; D1/D2 both UP |
| EXP-58 pipeline_global | kd2 | +549 | 0.429 | 0.924 | Failed |
| EXP-58 pipeline_avg | kd2 | +200 | 0.368 | 0.858 | Failed |

---

## Key Conclusions

1. **kd_cr pipeline_avg is the one valid positive result**: I=−142, D1/D2 both increase, 31 model
   calls vs 32 for standard. This is a real improvement but weaker than (fraudulent) prog commitment.

2. **kd_cr pipeline_global: suspicious PPL gaming**: PPL=37 is too low, D2 drops, ROUGE-L=0.012.
   The global schedule's rapid leading-position "lock-in" creates a different PPL-hacking mechanism.

3. **kd2 is incompatible with pipeline ODE**: Both variants fail catastrophically (+200/+549 in PPL).
   The asynchronous t-schedule breaks kd2's SC interaction.

4. **Conditioned evaluation is informative**: ROUGE-L correctly distinguishes genuine prefix
   continuation (standard_cond ~0.05) from spurious low-PPL generation (pipeline variants ~0.01-0.03).
   Standard ELF with conditioning scores 0.05-0.06 ROUGE-L on gutenberg text — a valid baseline.

5. **Pipeline ODE is not a drop-in commitment replacement**: It's an architectural choice (fewer
   serial steps) that only works for kd_cr and only in the avg variant. The core PPL improvement
   mechanism is not fully understood.

---

## Next Steps

- EXP-58b (if needed): larger N, multi-seed for pipeline_avg kd_cr to confirm I=−142±?
- EXP-59 candidate: identify WHY kd_cr handles pipeline ODE but kd2 doesn't
  (hypothesis: kd_cr SC is robust to t-schedule variation; kd2's h10 SC gate is not)
- Paper section: pipeline ODE as efficiency method (fewer serial steps) rather than quality method
