# CCLF Research Notes
# Last updated: 2026-06-13

---

## SESSION HANDOFF (read this first)

**Paper:** `/home/wjzhang/tt_workspace/model/CCLF/ICLR27-CCLF/` — all sections written, bib fixed (13 entries, no orphans)  
**Code:** `/home/wjzhang/tt_workspace/model/CCLF/CCLF/`

### What is done
- Probe experiments complete: ELF, LangFlow, MDLM, DUO all probed (64 seqs, 21 t-steps, 4 noise seeds)
- All paper sections written (intro, related work, prelim, method, experiments, conclusion, appendix)
- Six method ideas fully analyzed and documented (see below)
- DUO citation added (was missing entirely), bib cross-checked

### What to do next (in order)

**Session 1 — Idea A (inference only, no training needed):**
- Modify ELF sampling to use decode-branch SC signal
- File: `models/ELF/src/train_step.py`
- Current SC: `jnp.concatenate([z, x_pred_cond], axis=-1)` where x_pred_cond = x̂_t^(1) from first pass
- Change: add a decode-branch pass (backbone at t=1 applied to x̂_t^(1)) → x̂_t^dec, use that instead
- ⚠️ Cost: 3 forward passes per step (was 2). NOT zero cost.
- Evaluate: G(t), Rec@1(t), Gen.PPL at {1,2,4,8} steps

**Session 2 — Idea 3 + C (first training run):**
- Add L_KD to ELF training: KL(sg(p_1^dec) ‖ p_t) with gate ω(t)=[0.25,0.95]
- Add cliff importance sampling: t ~ dG/dt + ε (weights from probe_geo.json)
- ⚠️ Watch: importance sampling + gates may double-concentrate on cliff region — monitor t distribution

**Session 3 — Idea 2 (joint objective):**
- Add L_ce with two-stage α_nm(t) = α1·σ(k1(t−0.30)) − α2·σ(k2(t−0.60))
- Run ablations: KD-only vs KD+CE vs CE-only; two-stage vs three-stage α_nm

**Session 4 — Idea B (refinement):**
- Add per-position KD weight: w_{t,i} = (1−H/logV)·1[argmax p_{t,i} ≠ argmax p_1^dec]

**Session 5 — Idea 1 (ablation):**
- Cosine L_anc with β(t) gate [0.20,0.55]; treat as regularizer ablation only

### Key open questions
1. Does p_t^dec (intermediate teacher) outperform p_1^dec? (+50% training cost but stronger signal)
2. Does importance sampling interact well with ω(t)/β(t) gates (double-counting risk)?
3. Three-stage α_nm: ablate whether t3≈0.90 recommit helps or hurts (CE vs KD on plateau)

---

## Core Finding (Probe Study)

The central finding is a **geometric–distributional gap**:

> ELF's backbone already encodes geometric token identity in intermediate representations (G(t) = 60.8% at t=0.30), but the linear unembedding cannot yet extract it (Rec@1 = 45.5% at the same step). The decode branch closes much of this gap (70% at t=0.30, 86.8% at t=0.35, +27.6 pp advantage).

All six design ideas are different angles of attack on this same root problem.

---

## Six Method Ideas: Full Analysis

### Problem taxonomy

```
几何信号已有 (G(t) high)
    └── decode branch 能读出来  →  Idea A (推理时直接用)
    └── linear branch 读不出来  →  Idea 3 (训练时蒸馏)
                                   Idea B (加重 stuck positions)
Commitment cliff 可干预
    └── wrong→correct rate 高   →  Idea 2 (L_ce on cliff)
    └── 采样效率低              →  Idea C (importance sampling)
Residual 结构性大 (ρ ≥ 0.82)
    └── L2 anchor 低效          →  Idea 1 (改 cosine)
```

---

### Idea A: Decode-branch self-conditioning (inference-time, no retraining)

**What ELF self-conditioning actually does (verified from train_step.py):**
ELF self-conditioning is in **continuous embedding space**, NOT token probability space.
The backbone input is: `[z_t, sg(x̂_t^(1))]` where x̂_t^(1) ∈ R^(L×d) is the predicted clean embedding from a preliminary forward pass (NOT p_t).
Concretely:
```
Pass 1: z_t → backbone(t) → x̂_t^(1)          [standard forward, embedding space]
Pass 2: [z_t, x̂_t^(1)] → backbone(t) → x̂_t  [main pass with SC signal]
```

**Proposed change:**
Replace x̂_t^(1) (from standard forward) with x̂_t^dec (from decode branch):
```
Pass 1: z_t → backbone(t) → x̂_t^(1)          [same as before]
Pass 1b: x̂_t^(1) → backbone(t=1) → x̂_t^dec  [NEW: decode branch in embedding space]
Pass 2: [z_t, x̂_t^dec] → backbone(t) → x̂_t  [main pass with better SC signal]
```

**Cost:** NOT zero. Requires 3 forward passes per sampling step vs. 2 currently (~50% more compute per step). No retraining required.

**Motivation:** Decode branch at t=0.35 achieves 86.8% top-1 vs 59.2% for linear branch. The embedding x̂_t^dec coming out of the decode branch should encode token identity more cleanly than x̂_t^(1), providing a better self-conditioning signal for the main pass.

**Why it matters:** Either (i) helps → decode branch SC signal is a better guide, deployable without training; or (ii) doesn't help → the geometric–distributional gap is a training problem, not a forward-pass architecture problem, which strengthens Idea 3's necessity.

**Probe support:** ★★ Decoder advantage is real; whether it transfers to SC signal quality is a hypothesis.

**⚠️ Previous error:** Originally described as "zero cost" and as replacing p_t (probability space). Both were wrong. The SC mechanism is in embedding space (R^d), and the cost is +50% per sampling step.

---

### Idea 2: L_ce with commit–release schedule α_nm(t)

**Motivation:** wrong→correct correction rate peaks at 20.8%/step at t=0.30 (commitment cliff), drops to 1–1.5%/step at plateau. CE supervision is ~10-20x more effective at the cliff.

**Design:** Two-stage α_nm(t) = α1·σ(k1(t−0.30)) − α2·σ(k2(t−0.60))
- Stage 1 rises at t≈0.30: supervise token crystallization during the cliff
- Stage 2 falls at t≈0.60: release CE as G(t) plateaus and wrong-committed positions freeze

**Three-stage variant** (adding +α3·σ(k3(t−0.90))): treat as ablation. The argument AGAINST t3 has two parts:
1. L_KD's ω(t) gate already covers [0.25, 0.95], providing trajectory supervision across the plateau — CE there is redundant with KD.
2. Wrong-committed positions in the plateau are frozen (wrong→correct = 1–1.5%/step); adding CE there pushes them harder into incorrect anchors, risking over-commitment rather than correction.

**⚠️ Previous error:** The original argument was "final decoder handles +16.8pp so t3 is unnecessary." This is circular — if L_KD succeeds, the final correction spike should shrink, making that argument moot. The correct argument is about over-commitment risk + redundancy with L_KD.

**Cost:** Low (add CE term + α schedule).

**Probe support:** ★★ Commitment cliff identification is clear; quantitative correction rate supports schedule boundaries.

---

### Idea 3: L_KD — decode-teacher distillation

**Motivation:** Internalizes the decode branch's distributional advantage (+27.6pp at t=0.35) into the linear branch via KL divergence.

**Formula:** L_KD(t) = τ_KD² · (1/L) Σ KL(p_1,i^dec ‖ p_t,i)

**Two teacher variants:**
- **Default:** p_1^dec (final decode distribution, computed once) — efficient
- **Strong variant:** p_t^dec (decode at each intermediate t) — costs one extra forward pass but provides a locally-calibrated teacher at every t (the +27pp advantage is NOT only at t=1, it's present all the way from t=0.25 to t=0.95)

**Gate:** ω(t) = σ(k_ω(t−0.25))·(1−σ(k_ω(t−0.95))) — covers the entire stable-but-imperfect plateau

**Cost:** Medium. Default variant: ~+0% compute (p_1^dec computed once per batch). Strong variant: ~+50% (extra forward pass at each t).

**Probe support:** ★★★ Most strongly probe-supported component. Decoder advantage table directly motivates this.

---

### Idea C: Commitment cliff importance sampling

**Motivation:** Same t matters very differently. The cliff (t∈[0.10,0.35]) has 10-20x more correction potential than the plateau. Current uniform t-sampling wastes compute on the plateau.

**Design:** Sample t ~ p(t) where p(t) ∝ dG/dt + ε. Use probe-measured G(t) curve to compute weights offline, then use as a fixed sampling distribution during training.

**Cost:** Essentially zero — just change the t-sampling distribution. No change to loss or architecture.

**Probe support:** ★★ dG/dt is directly measured.

**Note:** Can be stacked with any other training component at zero additional cost.

---

### Idea B: Wrong-committed position entropy mask for L_KD

**Motivation:** 13–17% of positions in the stable-but-imperfect plateau are wrong-committed (low H, wrong top-1). These are the highest-value targets for KD but currently receive equal weight.

**Design:** Per-position KD weight:
```
w_{t,i} = (1 − H(p_{t,i}) / log V) · 1[argmax p_{t,i} ≠ argmax p_1,i^dec]
```
= high entropy positions get low weight (they're still forming)
= wrong-committed positions (low H, wrong top-1 vs teacher) get high weight

**Cost:** Low (per-position weight computation is trivial given p_t is already computed).

**Probe support:** ★★ The 13–17% wrong-committed fraction is directly measured.

**Dependency:** Requires Idea 3 (L_KD) as a base.

---

### Idea 1: Cosine L_anc

**Motivation:** ρ(t) = ‖r_t‖/‖x̂_t‖ ≥ 0.82 throughout — the L2 distance between x̂_t and a_t has an irreducible floor driven by the structural contextual residual. An L2 anchor loss fights this floor.

**Design:** Replace L2 with:
```
L_anc(t) = (1/L) Σ (1 − cos(x̂_{t,i}, sg(a_{t,i})))
```
Targets angular alignment only, not magnitude.

**Gate:** β(t) = σ(k_β(t−0.20))·(1−σ(k_β(t−0.55))) — active only during commitment-formation window

**Expected impact:** Low. This is primarily a regularizer. λ_anc should be much smaller than λ_KD. Treat as ablation.

**Probe support:** ★ ρ(t) finding motivates cosine over L2, but overall component is weakest.

---

## Priority Ordering

### By expected impact
1. Idea 3 (L_KD) — high, strongest probe support, core training component
2. Idea A (decode-branch SC) — medium~high, no training but +50% inference cost
3. Idea 2 (L_ce + schedule) — medium, probe-motivated cliff supervision
4. Idea B (position mask) — medium, targeted refinement of L_KD
5. Idea C (importance sampling) — medium, free efficiency gain
6. Idea 1 (cosine L_anc) — low, regularizer only

### By research sequencing (recommended execution order)
```
Week 1:  Idea A   → decode-branch SC at inference (no retraining, 3 passes/step)
         → validates that geometric–distributional gap is real and exploitable
         → ⚠️ NOT zero cost: +50% per sampling step

Week 2:  Idea 3+C → L_KD + cliff importance sampling (free to combine)
         → first full training run; establishes core result

Week 3:  Idea 2   → add L_ce with commit–release (two-stage), run joint objective
         → ablate: KD-only vs. KD+CE vs. CE-only
         → also ablate: two-stage vs three-stage alpha_nm

Week 4:  Idea B   → add position-adaptive KD mask
         → ablate: uniform KD vs. masked KD

Later:   Idea 1   → cosine L_anc as ablation baseline
         → ablate: no-L_anc vs. L2 L_anc vs. cosine L_anc
```

---

## Paper Narrative (how the six ideas fit the story)

The probe reveals a geometric–distributional gap. This motivates a two-pronged approach:

**Inference-time** (Idea A): The decode branch can close the gap without training. This establishes the gap is real and quantifies the ceiling for training-time methods.

**Training-time** (Ideas 2, 3): L_KD internalizes the decode branch's advantage; L_ce with commit–release steers crystallization at the cliff. Together they close the gap.

**Refinements** (Ideas B, C): Importance sampling and position-adaptive masking make the training more efficient and targeted. Low-cost improvements on top of the core method.

**Regularizer** (Idea 1): Cosine L_anc provides a directional nudge without fighting the structural residual. Weakest component but conceptually principled.

---

## Key Numbers to Remember (from probe_geo results)

| Metric | Value | t |
|--------|-------|---|
| G(t) peak | 90.4% | t≈0.55–0.60 |
| G(t) at cliff start | 60.8% | t=0.30 |
| Rec@1 at cliff start | 45.5% | t=0.30 |
| Decode branch top-1 | 70.0% | t=0.30 |
| Decode branch top-1 | 86.8% | t=0.35 |
| Decode advantage peak | +27.6 pp | t=0.35 |
| ρ(t) minimum | 0.819 | t≈0.60 |
| ρ(t) at t=1 | 1.00 | t=1.0 |
| Final decoder correction | +16.8 pp | t=1.0 jump |
| Wrong-committed fraction | 13–17% | plateau |
| Wrong→correct rate peak | 20.8%/step | t≈0.30 |
| Wrong→correct rate plateau | 1–1.5%/step | t=0.35–0.95 |

---

## Open Questions

1. Does p_t^dec (intermediate decode teacher) significantly outperform p_1^dec? Cost is 2x training, but signal strength is much higher.
2. Does importance sampling by dG/dt interact well with the ω(t) and β(t) gates? (The gates already concentrate attention on certain t regions; importance sampling might double-count.)
3. Is L_KD useful for LangFlow despite the architectural asymmetry (no intermediate x̂_t)? Could apply to z_t instead — but the motivation is weaker.
4. Three-stage vs. two-stage α_nm: the final recommit at t3≈0.90 might still help if the model has underfitted the cliff — to be determined by ablation.
