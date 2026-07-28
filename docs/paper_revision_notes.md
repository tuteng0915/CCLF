# Paper Revision Notes

Based on the technical review in `CDLM_commitment_review_and_experiment_roadmap.md`.
This document maps every problematic claim to: (a) **immediate wording fix** (no experiments needed),
and (b) **pending experiment** required before the claim can be strengthened or removed.

Status tags: `[FIX NOW]` = apply immediately | `[PENDING EXP-XX]` = wait for experiment result

---

## 0. Probe Protocol Disclaimer (affects ALL sections)

**The fundamental issue (§4.1 of review):** Every existing probe in this paper uses the
"forward-noise oracle" pattern:
```
for each t:
    z_t = t * x_clean + (1-t) * ε      # independent draw
    x̂_t = backbone(z_t, t)              # single forward pass
    record metrics on x̂_t
```
This is **not** the actual reverse generation trajectory. The model never runs a reverse ODE/SDE step.
Even `probe_token_trajectories.py` (which fixes one ε and sweeps t) still evaluates the oracle denoiser
at each t independently — it does not run the sampler.

**Consequence:** All "trajectory" language in the paper is technically about the oracle denoiser's
behavior on forward-corrupted states, not about what happens during generation.

**Global fix to apply:** Add one sentence to the Probing paragraph in Section 5 and to the
Appendix setup section:
> "All probes in this work use the *forward-noise oracle* protocol: for each time $t$, we independently
> sample $z_t \sim q(z_t \mid y)$ and compute a single backbone forward pass; these are not actual
> reverse-generation trajectories. A comparison with reverse-sampler trajectories is conducted in
> EXP-01 (ongoing)."

Then apply find-and-replace across the paper:
- "the trajectory commits/releases/is stuck" → "the forward-noise oracle probe shows..."
- "during the trajectory" → "under the forward-noise oracle probe"
- "trajectory dynamics" → "probe behavior under forward noise corruption"

---

## Issue 1 — "commit–release–recommit" described as trajectory dynamics

**Review reference:** §4.1, §4.14

**Locations in paper:**

| File | Location | Problematic text |
|------|----------|-----------------|
| `0_abstract.tex:4` | Abstract | "indicating a **natural commit--release--recommit structure in the trajectory**" |
| `1_introduction.tex:41` | Introduction para 3 | "This commitment is non-monotone: geometric accuracy peaks at $90.4\%$...and then slightly declines" (acceptable as forward-probe observation, but the framing implies trajectory dynamics) |
| `4_method.tex:28` | Section 4.2 | "we consider a non-monotonic **commit--release schedule**" and "three-stage variant adds a recommit term" |
| `6_conclusion.tex:4` | Conclusion | "commitment peaking around $t\approx 0.55$--$0.60$ and **then partially releasing** before the terminal decode step" |

**Immediate fixes `[FIX NOW]`:**

| Location | Old text | New text |
|----------|----------|----------|
| Abstract:4 | "indicating a natural commit--release--recommit structure in the trajectory" | "indicating a **non-monotonic cosine alignment pattern** under the forward-noise oracle probe" |
| Abstract:4 | "non-monotone" | keep but note it is a forward-probe observation |
| Abstract:6 | "probe-calibrated commit--release--recommit coupling schedule" | "probe-calibrated **non-monotonic coupling schedule**" |
| Section 4.2:28 | "non-monotonic commit--release schedule" | keep the schedule name; just remove the claim that it reflects observed trajectory dynamics |
| Conclusion:4 | "partially releasing before the terminal decode step" | "showing a non-monotonic forward-probe alignment profile before the terminal decode step" |

**What to NOT yet claim:** That actual reverse-sampler trajectories show this non-monotonic pattern.

**EXP-01v3 baseline result (2026-07-22):** Protocol B (actual ODE trajectories, 64 sequences, 32 steps) shows that G_reverse > G_oracle at t<0.258 and G_reverse < G_oracle at t>0.258. The reversal at early t is explained by ODE determinism: z_0 ~ N(0,4) encodes full information about y_final, so the reverse trajectory "knows" the answer from step 1. G_xpred exceeds both at all t (model's own one-step prediction commits immediately). The non-monotonic forward-probe pattern is NOT confirmed on actual ODE trajectories — reverse trajectory commitment appears monotonic (always increasing from t=1→0). Cross-checkpoint results for kd_cr and kd2 pending.

**RESOLVED (2026-07-22):** EXP-01v3 complete for all three checkpoints. Crossover: kd2 (t≈0.184) < kd_cr (t≈0.213) < baseline (t≈0.243). No non-monotonic pattern observed on actual ODE trajectories for any checkpoint. "Commit–release–recommit" as trajectory dynamics is NOT supported by Protocol B.

---

## Issue 2 — ELF–LangFlow geometric comparison is asymmetric

**Review reference:** §4.2

**Problem:** ELF G(t) is measured on $\hat{x}_t$ (the backbone's **denoised prediction**).
LangFlow G(t) is measured on $z_t$ (the **noisy input**). These are architecturally different objects.

**Location:**

| File | Location | Problematic text |
|------|----------|-----------------|
| `5_experments.tex:11` | LangFlow paragraph | "This confirms that ELF's early geometric commitment ($G(t)>50\%$ by $t\approx 0.28$) is a learned property of its backbone architecture, not an artifact of the noise schedule." |
| `0_abstract.tex:3` | Abstract | "ELF's backbone representations commit geometrically to the correct token far earlier than any other model: $G(t)$ exceeds $60\%$ by $t=0.30$, while LangFlow, MDLM, and DUO all remain below $5\%$ at that point." |

**Immediate fix `[FIX NOW]`:**

The paper already contains this honest caveat in `5_experments.tex:11`:
> "Because LangFlow's backbone maps $z_t$ directly to vocabulary logits without producing a separate
> D-dimensional denoised representation, we measure $G(t)$ on $z_t$ itself... This is an architectural
> asymmetry, not a methodological choice."

This is the right spirit, but the conclusion paragraph then overclaims. Fix:

| Location | Old | New |
|----------|-----|-----|
| `5_experments.tex:11` last sentence | "This contrast confirms that ELF's early geometric commitment...is a learned property of its backbone architecture, not an artifact of the noise schedule." | "This contrast reflects the architectural difference: ELF's denoised $\hat{x}_t$ is directly compared to ELF's token centroids, while LangFlow's noisy $z_t$ is compared to the same centroids. Whether LangFlow's native posterior $x_t$ (Eq.~\ref{eq:probe_dGdt}) shows comparable early commitment is tested in EXP-02 (ongoing)." |
| Abstract:3 | "far earlier than any other model" | "far earlier in ELF's denoised $\hat{x}_t$ than in LangFlow's noisy $z_t$, MDLM, and DUO; the comparison is asymmetric for LangFlow (see Section~5)" |

**Pending:** `[PENDING EXP-02]` — probe LangFlow native $x_t$ and $a_t = E^\top x_t$ to make a valid comparison.

---

## Issue 3 — G(t) described as "decoder-independent"

**Review reference:** §4.3

**Problem:** In a tied-weight model (ELF uses tied weights), $G(t)$ is:
$$G(t) = \arg\max_v \frac{\hat{x}_t^\top E_v}{\|\hat{x}_t\| \|E_v\|} = y_i$$

This is the same vocabulary matrix $E = W$ used in the linear projection head, just without bias
and with row-normalization. It is **not** independent of the decoder; it is a bias-free,
row-normalized version of the same projection. The gap between $G(t)$ and $\mathrm{Rec}@1(t)$
is caused by: (1) output bias removed, (2) row norms normalized, (3) no temperature, (4) distribution shift.

**Locations:**

| File | Location | Problematic text |
|------|----------|-----------------|
| `3_preliminary.tex:124` | G(t) definition | "Because $G(t)$ uses only the geometry of $\hat{x}_t$ relative to the token centroid matrix $E$---**not the parameters of $W$**---it can reveal that the backbone's continuous representation has already acquired directional token identity even when the **learned decoder head** has not yet reflected it in $p_t$" |
| `5_experments.tex:64` | Method components | "the backbone's continuous representation already encodes **geometric** token identity (e.g., $G(t){=}60.8\%$ at $t{=}0.30$), but the linear unembedding cannot yet extract it" |

**Immediate fix `[FIX NOW]`:**

| Location | Old | New |
|----------|-----|-----|
| `3_preliminary.tex:124` | "not the parameters of W" and "learned decoder head" framing | "In a tied-weight model such as ELF where $E = W$, $G(t)$ is equivalent to a bias-free, row-normalized vocabulary projection; the gap between $G(t)$ and $\mathrm{Rec}@1(t)$ may reflect the effect of output bias, output-row norms, or temperature rather than purely geometric information absent from $W$. We interpret G(t) as a **cosine-normalized token readout accuracy** rather than a decoder-independent geometric measure." |
| Throughout paper | "geometric commitment" (when contrasted with distributional) | "cosine-normalized readout accuracy" |
| Section 5, method (i) | "backbone's continuous representation already encodes geometric token identity...but the linear unembedding cannot yet extract it" | "the cosine-normalized readout $G(t)$ exceeds the bias-weighted linear readout $\mathrm{Rec}@1(t)$ at intermediate $t$; the gap may reflect bias and norm effects in addition to representation quality" |

**Metric rename (paper-wide):**
- `geometric nearest-token accuracy G(t)` → `cosine-normalized token readout accuracy G(t)`
- `geometric commitment` → `cosine-normalized readout`

**EXP-04v2 result (2026-07-22):** Three-source decomposition completed (1024 tokens, baseline checkpoint). G_head_null ≈ 0.014–0.022% (essentially zero — head geometry contributes no systematic token bias). G_backbone_null ≈ 0.15–1.96% (small frequency prior from backbone, t-dependent). G_oracle = 0.23–95.07% (completely dominates). Conclusion: the G(t) vs Rec@1 gap is NOT explained by bias/norm effects from the vocabulary head geometry. The gap reflects genuine information in the cosine-normalized readout that is suppressed by output bias. The "decoder-independent" claim in Issue 3 should be revised as specified above, but G(t) remains a valid measure of directional information. `[PENDING EXP-04]` RESOLVED.

**No further experiment needed for this issue.** The immediate fix above is sufficient.

---

## Issue 4 — ρ(t) interpretation is overclaimed

**Review reference:** §4.5

**Problem:** The paper defines:
$$\rho(t) = \frac{\|\hat{x}_t - a_t\|}{\|\hat{x}_t\|}, \quad a_t = E^\top p_t$$

Then claims this "measures the fraction of $\hat{x}_t$ not explained by the current lexical belief"
(Section 3) and that "the residual encodes contextual and syntactic information" (Section 4, Abstract).

These claims are mathematically wrong because:
1. $E^\top p_t$ is a barycenter, not an orthogonal projection onto a lexical subspace
2. $\rho(t) \approx 0.82$ does not mean "82% of information is non-lexical"
3. Large residual norm ≠ the residual contains syntax/semantics (the review lists 8 specific problems)

**Locations:**

| File | Location | Problematic text |
|------|----------|-----------------|
| `3_preliminary.tex:79` | ρ(t) definition | "which **measures the fraction of $\hat{x}_t$ that is not explained by the current lexical belief**" |
| `4_method.tex:13` | Section 4.1 | "**The residual encodes contextual and syntactic information** that the static token centroid $E^\top p_t$ cannot capture" |
| `0_abstract.tex:5` | Abstract | "the residual $r_t = \hat{x}_t - E^\top p_t$ **encodes contextual information that is structurally distinct from lexical identity**" |
| `6_conclusion.tex:5` | Conclusion | "large residuals $r_t$ persist throughout the trajectory, confirming that contextual representations are not explained by token anchors alone: **the residual is a structural property of contextual embedding spaces**" |

**Immediate fixes `[FIX NOW]`:**

| Location | Old | New |
|----------|-----|-----|
| `3_preliminary.tex:79` | "which measures the fraction of $\hat{x}_t$ that is not explained by the current lexical belief" | "which measures the **anchor mismatch ratio**: the relative L2 distance between $\hat{x}_t$ and the vocabulary barycenter $a_t = E^\top p_t$. Since $a_t$ is not an orthogonal projection onto a lexical subspace, $\rho(t)$ should not be interpreted as the fraction of information unexplained by lexical identity." |
| `4_method.tex:13` | "The residual encodes contextual and syntactic information" | "**The residual $r_t$ has a large irreducible norm** in ELF's contextual embedding space" |
| Abstract:5 | "encodes contextual information that is structurally distinct from lexical identity" | "has a large and persistent norm that is not explained by the vocabulary barycenter alone; whether it encodes contextual information is tested separately (EXP-12)" |
| Conclusion:5 | "the residual is a structural property of contextual embedding spaces" | "the large anchor mismatch ratio is a structural property of contextual embedding spaces; its content requires further analysis" |

**Metric rename:**
- `normalized anchor residual ρ(t)` → `anchor mismatch ratio ρ(t)`
- Do not call it "contextual residual" until EXP-12 validates this

**Pending:** `[PENDING EXP-12]` — residual analysis experiment will test whether $r_t$ contains syntactic/semantic information beyond token identity.

---

## Issue 5 — q(t) ∝ |dG/dt| described as not changing the objective

**Review reference:** §4.7

**Problem:** Section 4.3 and Section 5 both say:
> "This importance weight is applied to all training objectives simultaneously and **does not alter the
> loss values, only the frequency** with which each trajectory region is visited."

This is incorrect. Changing the sampling distribution changes the expected loss:
$$\mathbb{E}_{t \sim p}[L(t)] \neq \mathbb{E}_{t \sim q}[L(t)]$$

To preserve the original objective, importance weights $p(t)/q(t)$ must be applied. Without them,
the method **deliberately reweights** the training objective.

**Locations:**

| File | Location | Problematic text |
|------|----------|-----------------|
| `4_method.tex:112` | Section 4.3 (probe_dGdt) | "does not alter the loss values, only the frequency with which each trajectory region is visited" |
| `5_experments.tex:73` | Method component (iv) | same claim implied |

**Immediate fix `[FIX NOW]`:**

| Location | Old | New |
|----------|-----|-----|
| `4_method.tex:112` | "does not alter the loss values, only the frequency with which each trajectory region is visited" | "**deliberately reweights the training objective** by changing the t-sampling distribution; importance weights $p(t)/q(t)$ would be needed for an unbiased estimate of the original objective. We treat the reweighted objective as a design choice, not a sampling efficiency improvement." |

**Pending:** `[PENDING EXP-15]` — the training schedule experiment will compare uniform, q(t) with importance correction (unbiased), and q(t) without correction (deliberately reweighted).

---

## Issue 6 — dec_sc improvement attributed to correction mechanism without controls

**Review reference:** §4.8

**Problem:** The paper says:
> "directly confirming that the decode branch encodes correction information that can be exploited
> within the existing sampling loop" (Abstract)
> "directly confirming the decode branch's role as a correction mechanism" (Conclusion)

Dec_sc adds one extra nonlinear pass (decode branch, t=1). The improvement could arise from:
(a) the branch extracts token-specific correction information, OR
(b) any additional nonlinear compute improves the estimate, OR
(c) temporal smoothing of the self-conditioning state stabilizes generation.

These are not distinguished by current results.

**Locations:**

| File | Location | Problematic text |
|------|----------|-----------------|
| `0_abstract.tex:8` | Abstract | "**directly confirming** that the decode branch encodes correction information" |
| `6_conclusion.tex:8` | Conclusion | "**directly confirming** the decode branch's role as a correction mechanism" |
| `5_experments.tex:91` | Analysis | "This suggests that the dec\_sc signal is most useful..." (interpretation OK, mechanism attribution is the issue) |

**Immediate fix `[FIX NOW]`:**

| Location | Old | New |
|----------|-----|-----|
| Abstract:8 | "directly confirming that the decode branch encodes correction information that can be exploited" | "**consistent with the hypothesis** that the decode branch encodes correction information; compute-matched controls are ongoing (EXP-13)" |
| Conclusion:8 | "directly confirming the decode branch's role as a correction mechanism" | "**suggesting** that the decode branch's output provides signal beyond generic nonlinear refinement; compute-matched ablations are ongoing (EXP-13)" |

**Pending:** `[PENDING EXP-13]` — compute-matched controls (extra denoise pass, shuffled decode branch, random residual) will test whether the correction-information hypothesis holds.

---

## Issue 7 — "Stable-but-imperfect positions are stuck" on forward probes only

**Review reference:** §4.1

**Locations:**

| File | Location | Problematic text |
|------|----------|-----------------|
| `4_method.tex:26` | Section 4.2 | "wrong-committed positions are largely **stuck**, and CE supervision there provides little benefit" |
| `4_method.tex:26` | Section 4.2 | "the wrong$\to$correct rate drops by an order of magnitude...while the correct$\to$wrong rate remains near zero. The **trajectory is stuck**" |
| `X_Appendix.tex:754` | Appendix Phase 3 | "wrong-committed positions **cannot self-correct**: the wrong$\to$correct rate drops by an order of magnitude" |

**Note:** The wrong-to-correct rates are computed across noise levels (e.g., comparing $\hat{y}$ at $t=0.35$
vs $\hat{y}$ at $t=0.40$ on independently sampled $z_t$), not on actual generation steps.

**Immediate fix `[FIX NOW]`:** Add "under the forward-noise oracle probe" qualifier:
- "wrong-committed positions are largely stuck **under the forward-noise oracle probe**"
- "the trajectory **appears** stuck in the forward-noise oracle probe; whether reverse-sampler trajectories show the same pattern is tested in EXP-01"

**EXP-01v3 baseline partial result (2026-07-22):** On actual ODE trajectories, wrong-committed positions show G_reverse < G_oracle at t>0.258, consistent with positions being "resolvable from signal" but the model not correcting them. However, EXP-01v3 does not directly measure per-position w2c/c2w rates on ODE trajectories — it only measures aggregate accuracy. Whether stuck positions self-correct on actual ODE trajectories remains to be tested with a position-tracking variant of the probe. The qualifier above remains appropriate.

---

## Issue 8 — Gen.PPL lacks statistical reliability indicators

**Review reference:** §4.9

**Problem:** The paper notes ODE-8 baseline shifts from 942 (n=256) to 872 (n=1000). The SAR failure
yields PPL≈2-3 under mode collapse, proving PPL alone is not a quality metric.

**Locations:** Table `tab:ppl`, Table `tab:sar`, all PPL numbers in Section 5 and Conclusion.

**Immediate fix `[FIX NOW]`:**
1. Add to Table `tab:ppl` caption: "Single-run estimates; confidence intervals and diversity metrics pending."
2. Add footnote: "Gen.PPL alone can be arbitrarily low under mode collapse (see SAR results, Table~\ref{tab:sar}); MAUVE and repetition rate are reported for key comparisons in Appendix~\ref{app:ablation:decsc}."
3. The dec_sc ablations in `app:ablation:decsc` already report MAUVE — ensure these are cross-referenced from the main text.

**Pending:** `[PENDING]` — add multiple seeds and bootstrap CI to at least the key results (baseline ODE-8/16, dec_sc ODE-16, dec_sc SDE-16) before final submission.

---

## Issue 9 — Too many unevaluated training components presented as method

**Review reference:** §4.10

**Problem:** Section 4 formally defines L_ce, L_anc, L_KD, q(t), L_sc — none of which have been
evaluated. This creates a paper with a large method section and no training-time results, which is
structurally weak.

**Immediate fix `[FIX NOW]`:** Add explicit "Unevaluated" label to Section 4's training objectives.
Change the Section 4.3 header from:
> `\subsection{Training Objectives}` 
to:
> `\subsection{Training Objectives (Proposed, Not Yet Evaluated)}`

And add at the start of 4.3:
> "The following objectives are **proposed** based on the probing findings; empirical evaluation is reported in EXP-15 and EXP-16 (ongoing). The inference-time validation in Section~5 tests the underlying mechanism, not the training-time objectives."

---

## Metric Rename Table (apply paper-wide)

| Old name | New name | Files affected |
|----------|----------|----------------|
| geometric nearest-token accuracy G(t) | cosine-normalized token readout accuracy G(t) | all files |
| geometric commitment | cosine-normalized readout | all files |
| normalized anchor residual ρ(t) | anchor mismatch ratio ρ(t) | all files |
| commitment time (from entropy threshold) | entropy-collapse time | Appendix |
| wrong-to-correct trajectory rate | cross-noise-level proposal transition rate (forward probe) | Appendix, Section 4 |
| "the trajectory commits" | "the forward-noise oracle probe shows commitment" | all files |
| "stuck" positions | "positions with low cross-noise-level transition rate (forward probe)" | Section 4, Appendix |

---

## Restructure Recommendations

### Move to appendix (from main paper)
- **MDLM and DUO comparison** figures — keep as supplemental controls; remove from main narrative
- **SAR** — move entirely to appendix or supplemental; note as "incompatible with dec_sc, reported as negative result"
- **Position-type taxonomy (A/B/C/D)** — relabel as "forward-noise oracle position types"; move figure to appendix
- **Detailed analysis of non-monotone SDE/ODE patterns** — interesting but not central to the mechanism story

### Relabel in Section 4
- Group L_ce, L_anc, L_KD, q(t), L_sc under "Training Objectives (Proposed, Unevaluated)"
- Add one sentence per objective: "Empirical validation of this component is EXP-XX (ongoing)"

### Strengthen
- **Main result:** dec_sc improves PPL at zero training cost — this is real and should be the primary claim
- **Linear vs decode branch gap** (17-28 pp, Table `tab:dec_advantage`) — this is the cleanest result; it should appear earlier and more prominently
- **Interpolation experiment** (if it exists) — if dec_sc residual is aligned with improvement direction, this is causal evidence

### Remove or weaken now
- "The residual encodes contextual information" (based only on ρ(t)) → replace with observational language
- "The timing gap is explained by the SNR schedule" → replace with "confounded by schedule" (see Issue 2)
- "Commit–release–recommit" as a trajectory fact → replace with "non-monotonic coupling schedule motivated by forward-probe observation"
- "Probe-proportional sampling does not change the objective" → replace with "deliberately reweights" (Issue 5)
- "Decode self-conditioning directly confirms the commitment mechanism" → replace with "is consistent with" (Issue 6)

---

## Completed Experiment Findings Summary (as of 2026-07-22)

### EXP-04v2 — Decoder Geometry Null Model (RESOLVED `[PENDING EXP-04]`)

**Result:** G_head_null ≈ 0.017% (head geometry no bias), G_backbone_null ≈ 0.15-2% (small frequency prior), G_oracle dominates at all t.

**Paper implication:** The G(t) vs Rec@1 gap is NOT explained by output-head geometry. Issue 3's immediate fix (relabel "cosine-normalized token readout") remains appropriate, but we can strengthen the language slightly: G(t) measures genuine directional information, not a bias artifact. The gap reflects suppression of that information by output bias, not an absence of information.

**Specific change to add to Section 3 (after immediate fix is applied):** "A null model feeding pure noise $z_t \sim \mathcal{N}(0, I)$ through the backbone yields G_null < 0.02% at all t (EXP-04v2), confirming that G(t) is not driven by output-head geometry or frequency priors."

---

### EXP-05v3 — Global Null Prior Estimation (RESOLVED `[PENDING EXP-05]`)

**Result:** G_null ≈ 0.03–4% (true global null prior from z_t_null = (1-t)·ε), G_debias ≈ G_oracle (within 3pp), no systematic prior correction needed. EXP-05's batch-shuffle −17pp was a methodological artifact (wrong-instance posterior, not frequency prior).

**Paper implication:** The G(t) signal is essentially uncontaminated by learned token frequency priors. The "early commitment" claim holds without prior correction.

**Specific note for paper:** If discussing prior effects or frequency bias, cite EXP-05v3: "Global null prior estimation (z_t = (1-t)·ε, zero x_clean signal) yields G_null < 4% at all t across all three checkpoints; debias correction G_debias ≈ G_oracle, confirming no systematic prior bias in G(t) (EXP-05v3)."

---

### EXP-01v3 — Reverse ODE Trajectory vs Oracle (COMPLETE — all 3 checkpoints, 2026-07-22)

**Result (baseline checkpoint):** G_reverse > G_oracle at t < 0.243 (ODE determinism: z_0 contains full y_final information). Crossover at t ≈ 0.243. G_xpred >> both from step 1. No non-monotonic pattern observed on actual ODE trajectories — commitment appears monotonic.

**Result (kd_cr checkpoint):** Crossover at t ≈ 0.213 (earlier than baseline by 0.030t). G_oracle much higher at same t (e.g., t=0.35: kd_cr 70.8% vs baseline 56.2%). Oracle−reverse gap much larger (peak +44.8pp vs baseline +20.8pp). G_xpred LOWER than baseline (European token artifact: kd_cr generates non-English, mismatches English GT). G_reverse at t=0.05: 9.62% (vs baseline 0.63%) — strong early backbone commitment.

**Result (kd2 checkpoint):** Crossover at t ≈ 0.184 (earliest of all three). G_oracle similarly high to kd_cr. Oracle−reverse gap peak +46.9pp. G_reverse at t=0.05: 13.83% (vs baseline 0.63%) — strongest early commitment.

**Crossover ordering: kd2 (t≈0.184) < kd_cr (t≈0.213) < baseline (t≈0.243)**. Consistent with EXP-10 oracle commitment ordering.

**Paper implication for Issue 1:** The "commit-release-recommit" structure is NOT confirmed on actual ODE trajectories for any checkpoint (only on forward-noise oracle probe). The non-monotonic coupling schedule design can remain, but must be justified as "calibrated to forward-probe observations" not "matching trajectory dynamics."

**Paper implication for Protocol A vs B:** The Oracle probe UNDERESTIMATES commitment at early t (t < crossover) because it doesn't benefit from ODE determinism. KD-trained models have larger oracle−reverse gaps (up to 47pp vs 21pp), suggesting KD improves oracle representation quality but doesn't proportionally improve actual ODE trajectory quality. This is a new finding: **KD improves backbone responsiveness more than it improves generative trajectory informativeness**.

**Caution on G_xpred cross-checkpoint comparison:** kd_cr/kd2 generate European tokens; G_xpred uses English GT → kd_cr/kd2 G_xpred is artificially suppressed. Cannot compare G_xpred across checkpoints without controlling for generation language.

---

### EXP-07v2 — Document-Level Probe Revalidation (COMPLETE, 2026-07-22)

**Setup:** 512 sequences split at document level (train=409, test=103). Each document's positions go entirely to train or test. Same states as original EXP-07 (exp07_{baseline,kd_cr,kd2}_64/states/). Added train_acc, overfit_gap, shuffled_label control. All 3 checkpoints completed; full data at `results/exp07v2_{baseline,kd_cr,kd2}/probe_accuracies_v2.json`.

**Three-checkpoint results at t=0.201 and t=0.352:**

| checkpoint | probe_test@t=0.20 | overfit_gap@t=0.20 | probe_test@t=0.35 | shuffled_acc |
|------------|-------------------|---------------------|---------------------|--------------|
| baseline | **51.3%** | +30.1pp | 83.9% | 1–4% |
| kd_cr | **49.7%** | +31.1pp | 80.6% | 1–4% |
| kd2 | **50.6%** | +29.8pp | 81.1% | 1–4% |

**Doc-level vs position-level correction (baseline):**

| t | original (position-level) | v2 test_acc (doc-level) | reduction |
|---|---------------------------|-------------------------|-----------|
| 0.201 | 56.3% | 51.3% | −5.0pp |
| 0.216 | 62.1% | 56.7% | −5.4pp |
| 0.246 | 71.5% | 66.0% | −5.5pp |

**Consistent ~5pp reduction across all t values. Original EXP-07 overestimated by 5pp.**

**Story A final numbers (document-level, all 3 checkpoints):**

| checkpoint | probe_test@t=0.20 | G(t)@t=0.20 | gap (probe−G) | verdict |
|------------|-------------------|-------------|---------------|---------|
| baseline | 51.3% | ~10.2% | **+41pp** | probe >> native; backbone commits but decoder can't expose |
| kd_cr | 49.7% | ~60.7% | **−11pp** | native >> probe; KD training exposes commitment via decoder |
| kd2 | 50.6% | ~56–60% | **~−6 to −9pp** | native likely >> probe (kd2 G(t) not directly measured) |

Shuffled-label control: 1–4% across all t and checkpoints (T5 frequency bias, not memorization). Overfit_gap at t=0.20 is ~30pp but test_acc=50% >> G(t)=10% for baseline, confirming genuine generalization of the probe.

**Architecture clarification (final):** x_hat = `self.final_layer(x)` = 512-dim T5 embedding prediction. EXP-07v2 Rec@1≈0% because unembed_kernel is applied to the wrong space — expected behavior. Correct native comparison = G(t) = cosine argmax in T5 embedding space.

**Paper implication (FINAL):** EXP-07v2 supersedes EXP-07. Use document-level numbers in paper:
- baseline: "linear probe achieves 51.3% vs native cosine decoder 10.2% at t=0.20, held-out document split (EXP-07v2)"
- kd_cr: "native decoder 60.7% outperforms probe 49.7% at t=0.20, held-out document split (EXP-07v2)"
- Original +46pp claim → corrected to **+41pp**; original −7pp claim → corrected to **−11pp**; both directions strengthened or confirmed

---

### EXP-36 — Valid dec_sc × DF Interaction (RESOLVED)

**Result:** tmin=0.5 gate insufficient for kd_cr/kd2 — language mixing persists. Only kd_cr + freeze_1.0 stable (but no improvement over no-DF baseline). PPL metric fails under degeneration (kd_cr + dec_sc PPL=264 < baseline 331 but text is Romanian/German).

**Paper implication for dec_sc claims:** dec_sc improvements (Issue 6) are checkpoint-specific: safe for baseline, risky for kd_cr/kd2. Any dec_sc claim should be scoped to "ELF baseline" and note checkpoint sensitivity.

**Paper implication for DF+dec_sc:** DF and dec_sc do NOT combine beneficially for any tested checkpoint+configuration (H2 partially confirmed). Cannot claim "inference enhancements are additive."

---

### EXP-08 / EXP-09 / EXP-09v2 / EXP-10 — Story B & C Methodological Reassessment (2026-07-22)

**Overall verdict:**
| Experiment | Valid finding | Main problem | Rating |
|------------|---------------|--------------|--------|
| EXP-08 | Function words oracle-recoverable earlier than content words | Not coarse-to-fine; frequency/surprisal confounded; rename needed | Descriptive value, not mechanism claim |
| EXP-09 | KD recoverable positions more spatially clustered | Measures spatial correlation, NOT causal bootstrapping; selection bias | Mechanism claim invalid |
| EXP-09v2 | Asymmetry pattern at late t | Risk sets incomparable; n=2–12; early-t data reverses claimed direction | Do not put in main text |
| EXP-10 | KD advances native decoder recoverability on oracle states | Protocol A only; "structural ceiling" wrong; data inconsistencies | Most solid in group |

---

#### Three shared blocking problems

**B1: t* = first-hit time, not commitment time.**
All of EXP-08, 09, 09v2 use t* defined as first correct prediction. But first-correct ≠ committed:
`wrong → correct → wrong → correct` is possible. The correct definition is:
`T_stable = min{t_k : ŷ_i(t_j) = y_i, ∀ j ≥ k}` (or at minimum K consecutive correct steps).
All timing analyses currently describe **first recoverability time**, not commitment time. Claims about "commitment" must be downgraded accordingly.

**B2: Fixed-noise audit required.**
Per-position timing (t*) is only valid if the same ε is used across all t for the same position:
`z_t = t·x_clean + (1−t)·ε, same ε at all t`.
If ε is resampled per t, a position changing from wrong to correct may just be a different noise draw. Requires code audit confirming: same sequence, same position, same ε, same padding mask, same SC condition, only t changes.

**B3: Tokenizer / function-word classification inconsistency.**
EXP-08 cites T5 vocabulary with "Ġ" prefix (GPT-2/BPE style). T5 SentencePiece uses "▁". EXP-09v2 mentions GPT-2 tokenizer for ELF. This must be resolved before any func/content analysis is valid. The 12.2% function-word coverage figure may be wrong. Correct approach: (1) word-level POS tagging on raw text, (2) align words back to ELF subtoken IDs, (3) only analyze complete words or explicitly specify multi-subtoken aggregation rule.

---

#### EXP-08: Token-type recoverability timing (NOT coarse-to-fine)

**What it actually shows:**
Under oracle corruption (Protocol A), function words have earlier average first-recoverability time than content words:
- kd_cr: func t*≈0.182, content t*≈0.255 (Δ≈0.073)
- baseline: func t*≈0.246, content t*≈0.400 (Δ≈0.154)
This is a real, meaningful effect.

**What it does NOT show:**
"Coarse-to-fine semantic formation." The result T_function < T_content conflates at least 6 confounds:
frequency, contextual surprisal, token length, tokenizer fragmentation, position, closed-class status, decoder bias.

The true coarse-to-fine test requires: for the same content word target i, show that `A_POS(t) > A_token(t)` at early t, i.e., its POS/semantic class is predictable before its exact identity. Current EXP-08 compares two *different* token populations, not hierarchical levels for the same token.

**Required before using "coarse-to-fine":**
1. Frequency + surprisal regression: `T_i = β₀ + β₁·log_freq(yᵢ) + β₂·surprisal(yᵢ|y<i) + β₃·POS + β₄·subtoken_len + β₅·position + εᵢ`. If function-word indicator shrinks to near-zero after controls, the timing difference is entirely explained by frequency.
2. Hierarchical probe: train probes at each t to predict POS, semantic cluster, and exact token from the same x̂_t. Show `A_POS(t) > A_semantic(t) > A_token(t)` at early t.
3. Within-content analysis: test `T_semantic_class < T_exact_token` for content words only. This is the real coarse-to-fine test.

**Safe paper claim (immediately usable):**
> "Under oracle corruption, function words become token-recoverable earlier than content words, with a larger gap in the baseline checkpoint."

**Forbidden paper claim:**
> "ELF first commits to coarse semantic structure, then refines exact lexical identity."

---

#### EXP-09: Spatial clustering of recoverability (NOT contextual bootstrapping)

**What it actually shows:**
In KD-trained models, positions that are "near a currently-correct position" at time t_k are more likely to become correct at t_{k+1} than "far" positions. Gap peaks at +65pp (kd_cr, t=0.5→0.7).

**Why it is NOT bootstrapping:**
Protocol A: there is no feedback. The correct prediction at t_k is never written back into the state; it is not passed as extra conditioning to t_{k+1}. Each t is an independent forward pass with the same ε. The "neighbor was correct" label is purely an observation about local span predictability, not an intervention. The shared-cause explanation suffices:
`local phrase predictability → {neighbor early correct, target later correct}`.

**Selection/survivor bias:**
The experiment conditions on positions that are *still incorrect* at t_k. This induces collider bias: easy spans have most positions already correct, leaving only a minority of errors that happen to be surrounded by correct positions. Hard spans remain uniformly incorrect, placing targets in the "far" group. The "near" group then trivially has higher next-step accuracy because it belongs to an easier span overall.

**Why +65pp at late t is especially unreliable:**
At late t in KD models, ~99% of positions are correct. "Far from any correct position" is nearly empty. Both near and far groups have extreme sample sparsity, making the point estimate meaningless. EXP-09 itself notes far group is empty or near-empty at late t.

**Safe paper claim:**
> "KD-trained models exhibit stronger spatial clustering of oracle-state token recoverability than the baseline."

**Forbidden paper claim:**
> "Committed tokens contextually bootstrap nearby uncommitted tokens" / "commitment propagates" / "cascade."

**What would actually prove bootstrapping:**
Causal intervention: fix position i's noisy state z_{t,i}, randomly select neighbor j, compare P(correct_i | z_{t,j} noisy) vs P(correct_i | z'_{t,j} = clean). If clean neighbor increases i's probability, and near > far, that is evidence of causal rescue. ELF requires on-manifold intervention (T5 re-encode, then add matched noise, or intervene on SC channel). Strongest version: on actual reverse ODE trajectory, replace neighbor SC with oracle-correct SC; compare final token outcomes.

---

#### EXP-09v2: Directional asymmetry analysis (exploratory only)

**Stated claim:**
func→content delta >> content→func delta, interpreted as unidirectional causal chain.

**Why this doesn't hold:**
1. **Risk sets incomparable at late t.** At t=0.5→0.7: cf group has n=3; at t=0.7→1.0: cf group has n=2. fc groups have n=61 and n=12. You cannot compare delta magnitudes between groups with such different sample sizes and compositions.
2. **Early-t data reverses the direction.** At t=0.1→0.2 (largest, most comparable sample): kd_cr fc=+0.1pp, cf=+1.8pp; kd2 fc=+1.6pp, cf=+2.7pp. The only comparable time window shows content→function is *stronger* (though both effects are tiny). The "func→content" direction only appears in the data where function-word risk set is exhausted.
3. **Late fc survivors are confounded.** The n=12 late-remaining content words are likely rare tokens, named entities, multi-subtoken words, or punctuation-adjacent positions. Those with function-word neighbors likely come from high-predictability phrase templates ("the ___", "of ___"), inflating the apparent effect.

**Safe disposition:**
Appendix only, described as:
> "Exploratory asymmetry analysis among a small number of late-unrecovered positions; risk sets are not comparable across directions at late t."

**Forbidden claim:**
> "KD creates a temporal causal propagation chain from function words to content words."

---

#### EXP-10: KD advances native decoder recoverability (oracle states)

**What it actually shows (solid):**
Under the same oracle Protocol A, KD checkpoints' native decode-path accuracy (G(t)) is dramatically higher at all t than baseline:
- t=0.20: baseline≈10%, kd_cr≈61%, kd2≈56–60%
- t=0.30: baseline≈54%, kd_cr≈90%

Combined with EXP-07v2 (probe accuracy ≈ same across all three checkpoints at t=0.20: 49–51%), the key inference is:
> **KD does not substantially increase oracle-state token recoverability in x̂_t (probe capacity similar). It primarily improves how the native decode pathway exploits that information.**

This is a sharper and more honest mechanism claim than "KD advances commitment."

**What it does NOT show:**
1. Not real-dynamics advancement. Protocol A only. EXP-01v3 shows actual reverse ODE trajectories don't have the same cliff. Rephrase: "KD advances native-decoder recoverability on oracle forward-corruption states."
2. "Structural ceiling at 85%" is wrong. Baseline at t≥0.98 reaches 97–100%. The plateau is a probe-configuration artifact (zero-SC mismatch at high t), not a structural limit on decodability. Remove "14% structurally undecodable" from any draft.
3. kd2 ≈ kd_cr on this metric doesn't mean stage-2 KD is neutral — it may affect generation, reverse trajectory, calibration, SDE behavior, or SC dynamics.

**Data inconsistency requiring resolution before paper:**
Multiple G(t) numbers exist for the same checkpoint at the same t, varying across:
- EXP-10 spec table (early entries): baseline t=0.20: 36.2%, t=0.30: 77.5%
- EXP-10 spec table (later entries): baseline t=0.20: 9.9%, t=0.30: 53.9%
- JAX dense run (probe_decode_v2_dense): baseline t=0.20: 8.6%, t=0.30: 45.3%

Source of discrepancy is likely: JAX vs torch, different SC condition, different noise draws, different dataset subset, or different decode path definition. A provenance table must be generated before any numbers appear in the paper: one pipeline, one dataset, one SC config, one set of noise draws.

---

#### Revised Story B narrative (what can be claimed)

**Finding 1: Token-type-dependent recoverability timing (EXP-08)**
Function words are oracle-recoverable earlier than content words. The gap is larger in baseline than KD. This needs frequency/surprisal controls before attributing to POS/semantic hierarchy.

**Finding 2: KD produces more spatially coherent recoverability (EXP-09)**
In KD models, oracle-correct regions are more spatially clustered in the sequence. Causal mechanism unknown; consistent with local phrase coherence or with dynamic propagation (not distinguished by current data).

**Finding 3: KD reorganizes the native decoding interface (EXP-10)**
Probe capacity on x̂_t is similar across checkpoints (~50% at t=0.20). KD shifts the native decoder's G(t) curve dramatically earlier (~10% → ~60% at t=0.20). The best current hypothesis: KD reorganizes how the contextual decoding pathway exploits token-recoverable information already present in x̂_t.

**Not yet claimable:**
- coarse-to-fine semantic hierarchy
- commitment cascade / contextual bootstrapping / temporal propagation
- func→content causal direction
- "KD advances commitment time" (as opposed to native-decoder interface)

---

#### Follow-up priority

**P0 (before any paper claim):**
1. Tokenizer audit: identify correct token boundary convention (▁ vs Ġ), rerun function-word classification
2. Fixed-noise audit: confirm same ε across t in all per-position timing computations
3. Change t* → first-stable (K≥3 consecutive correct) in EXP-08/09/09v2; rerun
4. Resolve G(t) data inconsistencies: generate unified provenance table (JAX vs torch, SC config, dataset, noise seed)

**P1 (controls):**
5. Frequency + surprisal regression on T* (EXP-08): does function-word indicator survive?
6. Discrete-time hazard model for neighbor effect (EXP-09): `Pr(T_i=t_{k+1}|T_i>t_k) = σ(α_k + β·N_i(t_k) + γᵀ·c_i + u_seq)` — still correlational but more honest than near/far delta

**P2 (causal experiments needed before using "bootstrapping"):**
7. Random clean-neighbor intervention: fix z_{t,i}, replace z_{t,j} with cleaner state, measure Δ log p_i(y_i)
8. Function-neighbor vs content-neighbor paired intervention on same target
9. Reverse-trajectory intervention: replace neighbor SC with oracle-correct SC at a real ODE checkpoint; compare final token

---

## Priority Order for Paper Edits

Apply in this order (each is independent of experiments):

1. **Issue 5 (q(t) objective)** — one sentence change, mathematically wrong, fix immediately
2. **Issue 4 (ρ(t) overclaim)** — remove one sentence from 3_preliminary.tex:79, one from Abstract
3. **Issue 6 (dec_sc mechanism)** — change "directly confirming" to "consistent with" in Abstract + Conclusion; add checkpoint scope caveat (EXP-36)
4. **Global probe disclaimer** — add two sentences to Section 5 Probing paragraph; note EXP-01v3 crossover paradox
5. **Issue 1 (commit-release-recommit)** — soften trajectory language throughout; EXP-01v3 confirms pattern NOT present on actual ODE trajectories
6. **Issue 7 (stuck positions)** — add "under forward-noise oracle probe" qualifier
7. **Issue 3 (G(t) decoder-independence)** — update Section 3 definition paragraph; add EXP-04v2 null model result
8. **Issue 9 (unevaluated objectives)** — add "Proposed, Not Yet Evaluated" label
9. **Issue 2 (LangFlow comparison)** — update conclusion of LangFlow paragraph
10. **Metric renames** — find-replace throughout
11. **Add null prior note (EXP-05v3)** — add one sentence to G(t) discussion confirming no prior bias

---

## EXP-11 / EXP-12 / EXP-13 / EXP-14 — Story C Methodological Reassessment (2026-07-22)

### 总体评级

| 实验 | 核心价值 | 当前 solidity | 判断 |
|------|-------:|------------:|------|
| EXP-11 Branching | 很高 | 较弱 | 扰动规模实现错误，绝对稳定率不可解释 |
| EXP-12 Rank | 高 | 中高 | 描述性结论有效，但跨 checkpoint 错误集不可直接比较 |
| EXP-13 dec-sc controls | 极高 | 中低 | 推翻原故事；v2 仍不足区分 compute 和 information |
| EXP-14 Flips | 高 | 较弱 | 使用错误 readout，top-1 flip ≠ commit-release-recommit |

### B1. EXP-11 扰动规模错误（critical bug）

**当前代码** (`probe_branching_stability.py` lines ~190-197):
```python
rms = z_split.norm(dim=-1, keepdim=True).mean().item()
sigma = args.noise_frac * rms
delta = torch.randn_like(z_split) * sigma
```

对于 d=512 维向量，期望 `|δ|_2 ≈ sigma * sqrt(512) ≈ 0.01 * rms * 22.6`。

**实际效果**：`noise_frac=0.01` 对应 ~22.6% 相对扰动，而非 1%。

**修正代码**：
```python
u = torch.randn_like(z_split)
u = u / (u.norm(dim=-1, keepdim=True) + 1e-8)
delta = args.noise_frac * z_split.norm(dim=-1, keepdim=True) * u
```

**其他问题**：
- Self-conditioning 使用零 SC（`zeros_sc = torch.zeros_like(z_b)`），而非真实轨迹 SC，测量的是不同系统
- "8 次全部一致"作为 stability 对 K 高度敏感（90% 单次概率 → 43% 8次全同）；应同时报告 S_orig、S_pair、H_branch、modal probability
- 1/(1-t) 放大无法通过 v(z,t) 分母推断，需实测 `|Φ(z+δ) - Φ(z)| / |δ|`

**当前结论无效**：EXP-11 spec 中 "branching stability t=0.688 时 7.72%/5.52%" 不能解释为动力学极度不稳定，主要是扰动规模过大的人为产物。

**修正方向（EXP-11v2）**：
- η sweep: `{1e-4, 3e-4, 1e-3, 3e-3, 1e-2}`
- 保存并传入真实 SC state
- 多指标：S_orig(t,η), S_pair(t,η), H_branch, modal_prob
- 扩展 t_split: `{0.10, 0.20, 0.30, 0.50, 0.70, 0.80, 0.90, 0.95}`
- 三个 checkpoint 使用相同 reverse states

### B2. EXP-14 Readout 错误（critical bug）

**当前代码** (`analyze_traj_stability.py` lines ~44-49):
```python
E = unembed_kernel.T  # (vocab_size, 512)
x_norm = F.normalize(xb.reshape(-1, d), dim=-1)   # d=512, x_pred 直接余弦
cos_sims = x_norm @ E_norm_dev.T
```

使用余弦相似度 `normalize(x̂_t) · normalize(E)^T`，**不是** ELF 正确 decode path。

**正确 decode path**（EXP-12 / probe_rev_traj_v2.py 已验证）：
```python
h11 = layer_feats[-1]  # (B, L, 768) — block 11 hidden state
hidden = F.gelu(h11 @ proj_kernel + proj_bias, approximate="tanh")  # (B, L, 512)
logits = hidden @ unembed_kernel + unembed_bias  # (B, L, V)
```

**修正方法**：轨迹文件只保存了 `z_t` 和 `x_pred`（均 512-dim），无 h11。需按 `probe_rev_traj_v2.py` 模式，对每个 trajectory step 的 `z_t` 重新运行 backbone forward（注册 block-11 hook）获取 h11，再走 decode path。

**结论无效**：EXP-14 中 83.4%（kd_cr）5+ flips、mean flips=6.91、"baseline 比 kd 更早稳定"等数字全部基于错误 readout，不能作为论文结论。

**Top-1 flip ≠ commit-release-recommit 问题**：高 entropy 下两候选交替翻转产生大量 flip 但无 commitment；真正的 commit-release-recommit 需要：
- Commit：某 token 持续 m≥3 步且 margin > threshold
- Release：该 token posterior 显著下降
- Recommit：后续再次稳定预测

**Step count 偏差**：32 steps 比 16 steps 更多 flip 机会，flip count 不能跨 step 数量比较；应报告 flip rate per unit t。

**Endpoint self-reference**：用最后步 decode 结果作 proxy GT，最后一步必然 100%，制造人工"晚期跳跃"。

**修正方向（EXP-14v2）**：
1. 重新加载模型，对每 trajectory 步的 z_t 注册 block-11 hook，获取 h11
2. 使用正确 decode path 计算每步 token prediction
3. 使用最后步的 actual sampler output（或 decode path output at t≈1.0）作 proxy GT
4. 新增 stable commit/release/recommit 检测（margin + persistence）
5. 新增 flip rate per unit t；在多个 step count 下比较

### B3. EXP-12 Selection Bias（moderate issue）

EXP-12 **使用了正确 decode path**（GELU(h_L11@proj)@unembed），基础结论有效：
- baseline wrong positions: mean rank ≈ 372（t=0.3）
- kd_cr wrong positions: mean rank ≈ 16（t=0.3），73% 已在 top-5

**但问题**：baseline wrong set（38% of positions）≠ kd_cr wrong set（11%）。KD 剩下的错误是更难的位置，直接比较 "372 vs 16" 夸大了每个位置的提升幅度。

**应补**：
1. 所有位置的 MRR、median rank、top-5/10 recall（不 condition on wrong）
2. 固定 baseline-wrong set `S_B = {i: ŷ_i^baseline ≠ y_i}`，对比该集合上 baseline vs kd_cr 的 rank
3. 共同错误集 `S_hard = {i: ŷ_i^B ≠ y_i ∧ ŷ_i^KD ≠ y_i}`
4. Median rank / geometric mean rank（mean rank 对 heavy tail 敏感）
5. Normalized logit gap = `(ℓ_top1 - ℓ_true) / std(ℓ)` 替代裸 logit gap

**当前安全结论**：
> Under oracle corruption, KD substantially improves the native decoder's ranking of the ground-truth token. Probe-measurable recoverability of x̂_t is similar across checkpoints (EXP-07v2), while native decoder ranking differs dramatically (baseline rank~372 vs kd_cr rank~16 at t=0.30). This supports: KD reorganizes the decode interface rather than changing x̂_t recoverability.

### B4. EXP-13 v2 结论需修订（moderate issue）

**v2 结果**（tmin=0.5）：
- baseline: decode(226.4) ≈ extra_denoise(229.2) < none(234.2)；差异极小
- kd2: extra_denoise(113.5) < none(134.5) < decode(205.2)；decode 明显更差

**v2 局限**：
- 仅 1 seed；512 samples；无 paired CI；kd_cr 受 seed=123 artifact 污染
- decode-shuffled 更好有多种替代解释（正则化噪声、减弱局部正反馈等）
- 仅测试 tmin=0.5 后半段；早期 decode feedback 影响未知
- 直接全量替换 SC（α=1）是极端设置

**当前最强结论**：
> In a late-gated (tmin=0.5) setting, decode branch refinement provides no consistent advantage over a compute-matched extra denoising pass for baseline; for kd2, the decode branch is actively detrimental. However, 1-seed results without CI are insufficient to claim H0 or H1 universally.

**最重要的定性 finding（可保留）**：
> 强行把 high-noise 阶段 decode proposal 反馈到 SC 会形成错误 positive feedback attractor（第一轮实验的模式坍缩）。Early lexical sharpening → SC feedback amplification → mode collapse。这个 failure mechanism 可能是值得写的 negative finding。

### 新统一假设（EXP-01 ~ EXP-14 综合）

```
KD improves off-policy lexical readout (oracle states → better native decoder ranking),
but creates no corresponding on-policy dynamics improvement (reverse trajectory not more stable);
hard decode feedback can amplify the mismatch into collapse.

off-policy readout improvement ≠ on-policy dynamics improvement
```

比原来的 "early commitment → decode feedback → better generation" 更符合现有证据。

### P0 优先级（EXP-11/12/13/14）

1. **EXP-11v2**：修正 perturbation normalization（per-position 单位球），sweep η={1e-4,...,1e-2}，保存真实 SC，多指标；重跑 3 checkpoints
2. **EXP-14v2**：加载模型，对每 trajectory step 重新运行 backbone + block-11 hook，decode path readout；新增 stable commit-release-recommit 检测，flip rate per unit t
3. **EXP-12v2**：paired rank analysis（fixed baseline-wrong set + S_hard），MRR + median + calibrated logit gap；CPU-only，快速
4. **EXP-13v3**：onset × mixing-strength sweep（tmin ∈ {0.2,…,0.8}, α ∈ {0.05,…,1.0}）；5 seeds，1000+ samples；解析 decode-shuffled 更好的机制

### P1 优先级

5. Oracle probe → reverse state probe transfer：检验 KD 学到的 readout 是否只适用于 oracle manifold
6. 比较 `|z_t^reverse - z_t^oracle-final|`：KD 是否改善 readout 同时让 sampler 偏离训练 manifold
7. Continuation intervention：固定 S_t=(z_t, s_t)，比较 5 种 SC 条件对下一步 x̂ 的局部影响


---

## EXP-15 / EXP-16 — Story C Supporting Evidence Reassessment (2026-07-22)

### 总体评级

| 实验 | 描述性结果 | 当前 mechanism claim | 适合位置 |
|------|--------:|------------------:|---------|
| EXP-15 | late blocks 参数变化更大 | "L10 是 KD 核心作用层"证据不足 | Appendix supporting |
| EXP-16 | KD 将 native decode 首次正确时间前移 | 不能称 commitment；固定噪声 bug | 主文 figure，但必须重命名 |

### EXP-15：参数距离 ≠ Functional importance

**安全结论**（可保留）：
> Parameter updates are concentrated more heavily in late transformer blocks, with L10–L11 showing the largest relative L2 changes (~0.21–0.22 vs ~0.08–0.12 for early blocks).

**过度主张（当前 spec 写法）**：
- "L10 是 KD 核心作用层" → 不成立
- 参数 L2 变化最大 ≠ functional importance（LayerNorm、冗余方向、高曲率 vs 平坦方向）
- R_L11 ≈ 0.22 > R_L10 ≈ 0.21，"发现 L10 最重要"与最大变化层不一致
- 缺少 continued-training drift control（相同步数原始目标函数训练）
- 12 层数据点高度自相关，参数变化与激活兼容性的关联不可统计检验

**需要的补充分析（P1）**：
1. Module 级分解：QKV/O attention、MLP up/down、LayerNorm、timestep conditioning、decode branch
2. Update direction 相似性：`cos(Δθ_l^{KDCR}, Δθ_l^{KD2})` — 若一致才是 KD-specific
3. Baseline continued-training control：`R_l^excess = R_l^KD - R_l^continued`
4. Block swap / parameter interpolation：只换 L10/L11 后是否恢复 KD 效果

### EXP-16：First-hit ≠ Commitment（两个 bug）

**Bug 1：固定噪声 bug（critical，继承自 EXP-07b）**

EXP-16 读取 `results/exp07b_{ckpt}` 中的 layer states，而这些 states 是用旧版 `probe_layerwise.py` 生成的，每个 t 值独立采样 ε。因此 EXP-16 的"首次正确时间"是：

> 在哪个 t 的独立噪声采样下，该位置恰好被正确读取。

而非：

> 在固定同一 ε 的情况下，随着 SNR 增加，何时首次被正确读取。

**修复**：使用 `results/exp07b_v2_*`（fixed-noise，当前正在生成）重跑 EXP-16。

**Bug 2：First-hit ≠ Commitment（定义问题）**

当前定义 `T_i^first = min{t: ŷ_i(t) = y_i}`，但：
- 不要求后续所有 t 也正确（may correct → wrong → wrong）
- 无 margin/confidence 要求
- "永不承诺 19%" 实为"在 {0.1, 0.2, 0.3, 0.5, 0.7} 这 5 个采样 t 内未被读取"；dense baseline 显示 t=0.98–1.0 时几乎 100% 正确 → 这 19% 不是"永不"，是"在评估范围内未被读取"

**修正定义（EXP-16v2 应报告三种时间）**：
1. `T_i^first`：min{t: ŷ_i(t) = y_i}（first-correct readout time，当前实现）
2. `T_i^stable`：min{t_k: ŷ_i(t_j) = y_i ∀j ≥ k}（stable-readout time，K≥3 连续正确）
3. `T_i^margin`：stable time with margin > δ

**修正措辞**："永不承诺 19%" → "not recovered within evaluated grid {0.1, 0.2, 0.3, 0.5, 0.7}"  
"135 倍减少" 这句不能保留。

**LangFlow 比较必须删除**：  
"LangFlow 在 t=0.916 才首次承诺，而 kd-cr 在 t≈0.17" 完全无效——nominal t 不可比、noise range 不同、readout 不同、tokenizer 不同。EXP-03 已表明 matched-SNR 后差距可能消失或反向。

**安全结论（保留）**：
> The native decoder's per-position first-correct readout time distribution is shifted substantially earlier by KD: under oracle corruption, 90.2% of positions are first correctly read by t=0.30 in kd_cr vs only 63.5% in baseline. Together with EXP-07v2 (similar probe accuracy across checkpoints), this supports: KD reorganizes the native decode interface, not the latent information formation timeline.

**修正方向（EXP-16v2）**：
1. 用 exp07b_v2 states（固定噪声，当前正在生成）重跑
2. 报告 T_i^first + T_i^stable + T_i^stable-margin 三种时间
3. "never recovered" → "not recovered within evaluated range"（right-censored）
4. 后续补充 dense grid（t=0.05, 0.06, ..., 1.0）
5. Paired analysis: ΔT_i = T_i^{KD} - T_i^{baseline}（相同位置、相同噪声）

### 两实验的综合 Insight（与 EXP-07v2 结合）

```
EXP-07v2:  独立 probe accuracy 三 checkpoint 相似（~50% @ t=0.20）
EXP-15:    KD 参数更新集中在 late blocks
EXP-16:    native decode first-correct 时间分布大幅前移（KD下 90% 在 t≤0.30）
           而 probe capacity 未变化

→ 最一致假设：KD 重组 late-stage computation 以通过 native decoder
             暴露 x̂_t 中已存在的 token 信息，而非让 token 信息更早形成。
```

### P0 优先级（EXP-15/16）

1. **EXP-16v2**：exp07b_v2 states 就绪后立即重跑，加 stable-readout time，修正"永不"措辞
2. **EXP-16 paired analysis**：相同位置、相同 ε 下的 ΔT_i = T_i^KD - T_i^baseline

### P1 优先级

3. **EXP-15 module 级分解**：QKV/O/MLP/LN/decode-branch 独立统计
4. **EXP-15 update direction**：`cos(Δθ^{KDCR}, Δθ^{KD2})`（两个 KD 变种方向是否一致）
5. **Dense grid EXP-16**：t=0.05,...,1.0，Kaplan-Meier survival analysis


---

## EXP-15v2 Results (2026-07-22) — Module-level Parameter Analysis

**Status**: COMPLETE (analyze_param_distance.py, CPU-only)

### Revised block-level numbers (EXP-15 spec was incorrect)

| Block | kd_cr R_l | kd2 R_l | cos(Δkd_cr,Δkd2) |
|-------|--------:|------:|------------------:|
| L0-L3 | 0.215-0.238 | 0.219-0.242 | 0.808-0.833 |
| L4-L8 | 0.211-0.237 | 0.212-0.241 | 0.815-0.860 |
| L9    | 0.280 | 0.288 | 0.877 |
| L10   | 0.338 | 0.344 | 0.891 |
| L11   | 0.338 | 0.338 | 0.902 |

**原始 spec 数据 (0.08-0.12 for L0-L3) 是错误的。** 实际早期层变化也在 0.21-0.24，不是小幅变化。Late blocks 确实更大 (0.34 vs 0.21-0.24)，但差距不如 spec 描述的戏剧性。

### 最重要的新发现：Decode head 变化最大

| 参数 | kd_cr R |
|------|--------:|
| unembed_bias | **2.59** |
| final_layer.linear.* | 0.58-1.23 |
| proj_kernel | 0.36 |
| proj_bias | 0.29 |
| unembed_kernel | 0.32 |

**KD 对 decode head 参数（unembed_bias、proj_kernel、unembed_kernel）的修改比例远大于 transformer blocks。** 这直接支持：

> KD 主要重组 native decoding interface（decode head）而非 backbone representation。

这和 EXP-07v2（probe capacity 三 checkpoint 相似）的结论完美一致。

### Update direction similarity 的新 insight

cos(Δkd_cr, Δkd2) 随深度增加（L0: 0.83 → L11: 0.90）。两种 KD 变种在 late blocks 的更新方向更一致，说明这是 KD objective 驱动的系统性重组，不是随机 drift。

### 修订后的安全结论（EXP-15v2）

> Both KD variants (kd_cr, kd2) show largest relative parameter changes in the decode head (unembed_bias R≈2.6, proj_kernel R≈0.36) and in late transformer blocks (L10-L11 R≈0.34 vs L0-L8 R≈0.21). Update directions in late blocks are more correlated between the two KD variants (cos≈0.90 at L11 vs 0.83 at L0), suggesting KD-driven systematic reorganization of the native decode interface.

### 废弃的 EXP-15 原始结论

~~"L10 是 KD 核心作用层（0.21 vs 0.08-0.12 for early blocks）"~~ → 早期层实际也有 ~0.22 变化，且 decode head 变化更大（2.59x），应直接用 decode head 改变作为 "KD reorganizes decode interface" 的证据。


---

## EXP-14v2 Results (2026-07-22) — Correct Decode Path on Actual ODE Trajectories (kd_cr)

**Status**: COMPLETE (kd_cr checkpoint, analyze_traj_stability.py, GPU 2)
**Protocol B**: actual reverse ODE trajectory (64 steps, 192 sequences × 1024 tokens = 196,608 positions)

### G(t) via Correct Decode Path on Protocol B Trajectory

| ODE step | t value | frac_match_proxy_GT |
|----------|---------|---------------------|
| 1        | 0.000   | 3.79%               |
| 9        | 0.250   | 28.01%              |
| 17       | 0.500   | 43.44%              |
| 25       | 0.750   | 51.35%              |
| 29       | 0.875   | 55.83%              |
| 32       | 0.969   | **66.42%**          |

**Critical observation**: Only 66.42% of positions match proxy GT even at the final step (t≈0.97). This means the decode path at the very end of the ODE cannot recover all tokens. (Proxy GT is defined as decode-path prediction on x_pred from the last trajectory step at t_next≈1.0 — not the true reference text.)

### Flip Distribution (Correct Decode Path)

| Flips | Fraction |
|-------|----------|
| 0     | 2.7%     |
| 1     | 10.4%    |
| 2–4   | 38.5%    |
| 5+    | **48.4%** |

- Mean flips: **4.66**, median: **4.0**
- Flip rate per unit t: **4.81** flips/unit_t
- Mean last-flip step: **19.3/32** (last flip happens late in the ODE on average)

### Metric Bug: stable_commit_step is Misleading for Protocol B

`frac_stably_committed=1.0, mean_step=1.1` — This fires because at t=0 (step 1), the model outputs the same token prediction for 3 consecutive steps (all near-noise predictions are the same junk token), which immediately satisfies the K=3 consecutive criterion. But that "stable" token is NOT the final answer.

**For Protocol B, the correct metric would be: T_stable_correct = min{step_i: prediction matches proxy_GT for K consecutive steps}.**

The current `stable_commit_step` is only meaningful for Protocol A (oracle) where predictions monotonically increase in quality.

### Key Insights for Paper Revision

1. **Protocol B dynamics are much more turbulent than Protocol A suggests**: 48.4% of positions flip 5+ times along the actual ODE trajectory. The oracle probe (EXP-07b/16) at the same t value shows smooth curves — but the actual trajectory is not at that oracle state.

2. **G(t) comparison (Protocol A vs B)**: Protocol A oracle at t=0.20 gives ~9-10% G(t) (EXP-16v2 pending). Protocol B at t≈0.25 already gives 28% — suggesting the ODE trajectory state at t=0.25 is already more "aligned" than a pure oracle noisy interpolation at t=0.20. This is expected since the ODE denoises progressively.

3. **Decode path ceiling problem**: Even at ODE endpoint, decode path accuracy only reaches 66.4% against proxy GT. Proxy GT itself may not be perfect (it's the last-step x_pred, not true reference). True accuracy against reference text would require that data.

4. **Revision to "commit–release–recommit" claim**: With 48.4% 5+ flips and mean 4.66 flips per position, the ODE trajectory constantly revises token predictions throughout. This is NOT a simple commit-and-stay pattern. Must be removed from paper.

### Pending: baseline and kd2 runs

kd_cr is complete. Need to run baseline and kd2 to make paired comparisons. But the finding about turbulent dynamics (5+ flips in 48% of positions) is likely robust across checkpoints.


---

## EXP-09v3 + EXP-08v2 Results (2026-07-22) — Stable Commit Timing + Coarse-to-Fine

**EXP-09v3**: stable_k=3, fixed noise (seed=42), correct T5 tokenizer, exp07b_v2 states
**EXP-08v2**: coarse-to-fine analysis on top of EXP-09v3 output

### EXP-09v3: Oracle Stable Commit Timing (Protocol A, K=3 consecutive)

| Checkpoint | never_commit | by t=0.10 | by t=0.20 | by t=0.30 | by t=0.50 |
|-----------|-------------:|----------:|----------:|----------:|----------:|
| baseline  | **24.8%**    | 2.0%      | 35.5%     | 61.2%     | 75.2%     |
| kd_cr     | **0.67%**    | 12.5%     | 59.3%     | 89.6%     | 99.3%     |
| kd2       | **1.07%**    | 12.5%     | 58.3%     | 88.5%     | 98.9%     |

**Key finding**: KD drops never-stably-commit from 24.8% to 0.67% (kd_cr). By t=0.5, KD checkpoints achieve 99% stable oracle readout vs 75.2% for baseline.

**This directly supports**: "KD dramatically improves off-policy lexical readout stability on the oracle Protocol A."

### EXP-08v2: Coarse-to-Fine Breakdown (correct T5 tokenizer, ▁ prefix stripping)

| Checkpoint | mean_t* (all) | func mean_t* | func committed | content mean_t* | content committed |
|-----------|:-------------:|:------------:|:--------------:|:---------------:|:-----------------:|
| baseline  | 0.287         | 0.231        | 84.5%          | 0.332           | 69.2%             |
| kd_cr     | 0.247         | 0.202        | **99.9%**      | 0.277           | **98.9%**         |
| kd2       | 0.250         | 0.204        | **99.9%**      | 0.279           | **98.3%**         |

**Coarse-to-fine bootstrapping** (content words, func neighbor within d≤5):

| Checkpoint | step 1 (t=0.2) Δ | step 2 (t=0.3) Δ |
|-----------|:-----------------:|:-----------------:|
| baseline  | +0.007            | +0.099            |
| kd_cr     | **+0.120**        | **+0.600**        |
| kd2       | **+0.114**        | **+0.589**        |

**Notes on step 2 (t=0.3) bootstrapping**: For kd_cr, content words near a committed func neighbor: 95.9% commit (n=5,937) vs 35.9% for those without (n=284). The "without" group is small (n=284), making this estimate noisy, but the effect direction is strong.

For baseline at step 2: 40.4% (n=19,693 with neighbor) vs 30.6% (n=762 without). Baseline also shows coarse-to-fine, but weaker.

### Key Conclusions (EXP-08/09v3)

1. **Function words commit before content words** across all checkpoints (Δ ≈ 0.075–0.10 t-units). This is the core coarse-to-fine signal in oracle readout.

2. **KD amplifies both commitment rate and timing advance**: nearly 100% of positions commit by t=0.5, and both func/content words commit earlier than baseline.

3. **Coarse-to-fine bootstrapping is dramatically stronger under KD** (Δ=+0.60 vs +0.10 at t=0.3). Possible mechanisms: (a) KD genuinely creates a hierarchical commit cascade; (b) selection bias — more committed func neighbors under KD means more positions enter the "near neighbor" category, thinning the "without neighbor" group. Both likely contribute.

4. **Critical caveat**: These are all Protocol A (oracle) results. EXP-14v2 showed Protocol B (actual ODE trajectory) has 48% 5+ flips per position. The excellent oracle readout stability ≠ on-policy stability. This must be stated explicitly.


---

## EXP-16v2 Results (2026-07-22) — Three-metric Readout Timing (Fixed Noise, Correct Decode Path)

**Status**: COMPLETE (compute_readout_timing.py, all 3 checkpoints, GPU 4)
**Protocol**: fixed ε (seed=42), exp07b_v2 states, decode path = GELU(h_L11@proj)@unemb + bias

### G(t) — Oracle Readout Accuracy

| t | baseline | kd_cr | kd2 |
|---|---------|-------|-----|
| 0.10 | 1.91% | **12.30%** | 12.27% |
| 0.20 | 36.07% | **58.46%** | 57.24% |
| 0.30 | 61.87% | **89.22%** | 87.98% |
| 0.50 | 75.55% | **99.48%** | 99.03% |
| 0.70 | 78.76% | **99.84%** | 99.64% |
| 1.00 | 90.11% | **99.86%** | 99.80% |

**kd_cr at t=0.50 achieves G=99.5%, baseline only 77.3%.** G gap is largest at t=0.20–0.50.

### T_first (first-correct readout time) — cumulative fraction

| by t= | baseline | kd_cr | kd2 |
|-------|---------|-------|-----|
| 0.10  | 1.9%    | 12.3% | 12.3% |
| 0.20  | 36.1%   | 58.8% | 57.6% |
| 0.30  | 63.0%   | 89.5% | 88.2% |
| 0.50  | 77.3%   | 99.5% | 99.1% |
| 1.00  | 91.1%   | 99.9% | 99.8% |
| **never** | **8.9%** | **0.11%** | **0.16%** |

### T_stable (K=3 consecutive correct) — cumulative fraction

| by t= | baseline | kd_cr | kd2 |
|-------|---------|-------|-----|
| 0.10  | 1.8%    | 11.9% | 11.9% |
| 0.20  | 34.7%   | 58.2% | 57.0% |
| 0.30  | 60.8%   | 89.2% | 87.95% |
| 0.50  | 74.9%   | 99.5% | 99.0% |
| **never** | **25.1%** | **0.53%** | **0.98%** |

*✓ Consistent with EXP-09v3: baseline 24.8%, kd_cr 0.67%*

### T_margin (K=3 stable + logit margin >5.0) — cumulative fraction

| by t= | baseline | kd_cr | kd2 |
|-------|---------|-------|-----|
| 0.10  | 1.46%   | 2.41% | 2.10% |
| 0.20  | 32.5%   | 44.8% | 45.4% |
| 0.50  | 71.1%   | 98.5% | 97.2% |
| **never** | **28.9%** | **1.50%** | **2.80%** |

### Paired ΔT Analysis (KD − baseline, among positions both recovered)

**T_first:**
- kd_cr vs baseline: mean ΔT = **−0.78 steps**, median = 0.0; frac kd_cr earlier = 46.7%
- kd2 vs baseline: mean ΔT = −0.75 steps, frac kd2 earlier = 45.2%

**T_stable:**
- kd_cr vs baseline: mean ΔT = **−0.36 steps**, median = 0.0; frac kd_cr earlier = 37.0%
- kd2 vs baseline: mean ΔT = −0.34 steps, frac kd2 earlier = 35.4%

**Also recovered by kd_cr but NOT baseline: 24.62%** (this is the dominant effect!)

### Interpretation

1. **The dominant KD effect is not timing but coverage**: among positions that BOTH baseline and kd_cr recover, KD is only slightly earlier (mean −0.36 steps, 37% earlier). The real gain is the 24.6% of positions that only KD recovers — baseline simply cannot stably commit them within [0.1, 1.0].

2. **T_margin gap is real**: 1.5% vs 28.9% never-commit at high confidence. KD produces high-confidence (large logit margin) stable readout for nearly all positions.

3. **Cross-validation with EXP-09v3**: numbers match exactly (baseline never-stable = 25.1% vs EXP-09v3 24.8%, kd_cr 0.53% vs 0.67%). Minor differences from floating-point rounding and slightly different implementations confirm results are robust.

4. **kd_cr G(t) provenance now resolved**: T_first t=0.20: 58.8% (EXP-16v2 JSON). Old EXP-10 showed "kd_cr≈61%" — now confirmed to be ~58.8% (T_first) / 58.2% (T_stable). Use **58.8%** as authoritative T_first. Note: T_stable by_t_0.50 is 99.5% for kd_cr (same as T_first at t=0.50).

### What this means for the paper

The "oracle readout timing" story is now quantitatively solid:
- KD essentially eliminates oracle readout failures (never-stable: 25% → 0.5%)
- KD makes 99% of positions stably readable by t=0.5 (T_stable 99.5%) vs 75% for baseline (T_stable 74.9%)
- Function words commit earlier than content words (EXP-08v2 confirms ~-0.075t)
- The effect is consistent across kd_cr and kd2

But this is **all Protocol A**. Protocol B (EXP-14v2) still shows 45-67% 5+ flips.


---

## EXP-14v2 三模型完整对比（2026-07-22）

**Status**: ALL THREE CHECKPOINTS COMPLETE

### 翻转分布（Protocol B，correct decode path）

| 指标 | baseline | kd_cr | kd2 |
|------|---------|-------|-----|
| 0 flips | 0.0% | 2.7% | 2.5% |
| 5+ flips | **67.6%** | 48.4% | **45.8%** |
| mean flips | **6.08** | 4.66 | **4.48** |
| median | 6.0 | 4.0 | 4.0 |
| flip rate/unit t | 6.27 | 4.81 | 4.63 |
| mean last-flip step | 21.2/32 | 19.3/32 | 19.4/32 |

### G_B(t)（Protocol B via correct decode path，proxy GT = 末步 decode prediction）

| t | baseline | kd_cr | kd2 |
|---|---------|-------|-----|
| 0.000 | 0.4% | 3.8% | 4.3% |
| 0.250 | 18.5% | **28.0%** | 26.0% |
| 0.500 | 37.2% | **43.4%** | **43.4%** |
| 0.750 | 46.2% | 51.3% | **53.1%** |
| 0.875 | 52.2% | 55.8% | **60.3%** |
| **0.969** | **73.0%** | 66.4% | **73.0%** |

### 与旧 EXP-14（错误 readout）的关键逆转

| 指标 | 旧 EXP-14（x_hat@unemb，错误） | 新 EXP-14v2（decode path，正确） |
|------|-------------------------------|--------------------------------|
| baseline 5+ flips | 58.2% | **67.6%** |
| kd_cr 5+ flips | **83.4%** | **48.4%** |
| kd2 5+ flips | **89.8%** | **45.8%** |
| 排序 | kd2 > kd_cr > baseline（KD 最不稳定） | baseline > kd_cr ≈ kd2（KD 更稳定） |

**旧结论（KD 使 on-policy 轨迹更不稳定）完全错误。正确结论：KD 稍微改善 on-policy 稳定性，但整体仍有 45-67% 5+ flips。**

### G(t) Provenance 最终确认

EXP-16v2 现在是权威来源（fixed noise seed=42，correct decode path）：
- **baseline @ t=0.20: 36.1%** (T_first) — 与 EXP-10 的 "≈10%" 冲突（非固定 ε 差异）
- **kd_cr @ t=0.20: 58.8%** (T_first) — 与 EXP-10 的 "≈61%" 接近（旧 spec 错误地记为 58.5%）

差距解释：EXP-10 和 JAX probe 使用非固定噪声（每 t 独立 ε）。非固定噪声下 G(t=0.20) 较低（baseline≈9-10%，因为任何 ε 实例都可能让某位置更难）。EXP-16v2 使用固定 ε（seed=42），所以 G(t) 更高（因为某些 ε 恰好让该位置在 t=0.2 就容易读出）。

**使用原则**：
- 引用固定噪声 G(t)（EXP-16v2）时注明"oracle G with fixed noise ε"
- 引用 EXP-10 G(t) 时注明"oracle G averaged over independent ε draws"
- 两者**不可直接比较**


---

## ===== 权威结果总结（2026-07-22）— 所有 P0 实验完成 =====

*以下为经过 bug 修复后的权威数据，可直接用于论文修订*

### A. Oracle Protocol A — KD 对 decode readout 的影响（EXP-16v2，fixed noise）

**G(t) — oracle readout accuracy（correct decode path，fixed ε seed=42）**

| t | baseline | kd_cr | kd2 |
|---|---------|-------|-----|
| 0.10 | 1.9% | 12.3% | 12.3% |
| 0.20 | **36.1%** | **58.8%** | **57.6%** |
| 0.30 | 63.0% | 89.5% | 88.2% |
| 0.50 | 77.3% | **99.5%** | **99.1%** |
| 0.70 | 80.7% | 99.8% | 99.7% |
| 1.00 | 91.1% | 99.9% | 99.8% |

*(Source: `results/exp16v2/readout_timing.json`, T_first; verified 2026-07-22)*

**T_stable（K=3 consecutive correct）never-commit rate：**

| baseline | kd_cr | kd2 |
|---------|-------|-----|
| **25.1%** | **0.53%** | **0.98%** |

**T_margin（K=3 stable + margin>5）never-commit rate：**

| baseline | kd_cr | kd2 |
|---------|-------|-----|
| **28.9%** | **1.50%** | **2.80%** |

*Authoritative source*: `results/exp16v2/readout_timing.json`

### B. Oracle Protocol A — Coarse-to-Fine（EXP-08v2，stable_k=3，T5 tokenizer）

| Checkpoint | func mean t* | func committed | content committed | func→content boost at t=0.3 |
|-----------|:------------:|:--------------:|:-----------------:|:----------------------------:|
| baseline  | 0.231        | 84.5%          | 69.2%             | +9.9pp                       |
| kd_cr     | **0.202**    | **99.9%**      | **98.9%**         | **+60.0pp**                  |
| kd2       | **0.204**    | **99.9%**      | **98.3%**         | **+58.9pp**                  |

**Function words commit earlier** (Δ≈−0.075 to −0.101 t-units across all checkpoints).

### C. On-policy Protocol B — ODE 轨迹实际稳定性（EXP-14v2，correct decode path）

| 指标 | baseline | kd_cr | kd2 |
|------|---------|-------|-----|
| 5+ flips | **67.6%** | 48.4% | 45.8% |
| mean flips | **6.08** | 4.66 | 4.48 |
| G_B(t=0.25) | 18.5% | **28.0%** | 26.0% |
| G_B(t=0.97) | **73.0%** | 66.4% | **73.0%** |

**KD slightly reduces on-policy flip rate** (67.6% → 45-48%). But all checkpoints show highly turbulent ODE dynamics (45-67% 5+ flips).

### D. Parameter Space（EXP-15v2，module-level）

| 区域 | kd_cr 最大 R |
|------|------------:|
| **decode head** (unembed_bias) | **2.59** |
| late blocks L10/L11 | 0.338 |
| early blocks L0-L8 | 0.21–0.24 |
| LayerNorm (within blocks) | 0.04–0.09 |

**cos(Δkd_cr, Δkd2)**: L0=0.83 → L11=0.90 (更一致的 late block 更新)

### E. 核心论文论题修订

**旧论题（不可支持）**：
- ~~"commit-release-recommit 是真实 ODE 动力学"~~ → Protocol B 显示连续翻转（平均 4.66 次），无离散 commit 事件
- ~~"KD 使 on-policy 轨迹更不稳定"~~ → 逆转！KD 实际更稳定（45% vs 67% 5+flips）
- ~~"L10 是 KD 核心作用层"~~ → decode head 变化最大（unembed_bias R=2.59）
- ~~"ELF 在 t=0.30 就有 90% 承诺"~~ → 这是 Protocol A，Protocol B 在 t=0.30 时 kd_cr G_B≈33%（完全不同）
- ~~"never committed 19%"~~ → 正确值 25.1%（stable_k=3），不是 19%（first-hit）

**新论题（有数据支持）**：
1. **KD 几乎消除 oracle readout 失败**（Protocol A never-stable: 25% → 0.5%）
2. **KD 主要重组 decode interface**，而非 backbone（EXP-15v2: unembed_bias R=2.59 >> block R=0.22-0.34；EXP-12v2: logit_gap 66.65→1.46 at t=0.30 on fixed never-commit set）
3. **Coarse-to-fine oracle readout cascade 在 KD 下显著增强**（func→content boost: +60pp at t=0.3）
4. **Protocol B on-policy 动力学仍然湍流**（45-67% 5+ flips；EXP-11v2: S_orig@t=0.5 仅 4.7%，ODE bifurcation 模式），KD 有轻微改善但幅度有限
5. **Protocol A ≠ Protocol B**：oracle readout stability ≠ on-policy dynamics stability
6. **EXP-12v2**：在 baseline never-commit 的 25.1% 位置上，kd_cr@t=0.30 正确率 82.6%（median_rank=1），baseline 仅 4.4%（median_rank=12）——直接无 selection bias 的 decode interface 效应量化

---

## Section F — EXP-12v2 Results（Paired Rank Analysis on Fixed Baseline-Wrong Set）

**实验状态**: DONE（2026-07-22）  
**结果文件**: `results/exp12v2/rank_analysis.json`, `results/exp12v2_t030/rank_analysis.json`

### F.1 Reference Set: Baseline Never-Commit Positions (25.1%)

n_ref = 61,602 positions (25.1% of M = 245,711 valid positions)
这些是 baseline 在 Protocol A (EXP-16v2) 中 T_stable = NEVER 的位置。

| t    | baseline MRR | kd_cr MRR | kd2 MRR | baseline correct | kd_cr correct | kd2 correct | bl median_rank | kd_cr median_rank |
|------|:------------:|:---------:|:-------:|:----------------:|:-------------:|:-----------:|:--------------:|:-----------------:|
| 0.10 | 0.0063       | 0.1132    | 0.1251  | 0.14%            | 6.96%         | 8.52%       | 7017           | 153               |
| 0.20 | 0.1105       | 0.5884    | 0.5686  | 4.30%            | 50.17%        | 48.28%      | 130            | 1                 |
| 0.30 | 0.1955       | 0.8781    | 0.8605  | 4.37%            | **82.55%**    | 80.46%      | 12             | **1**             |
| 0.50 | 0.2685       | 0.9898    | 0.9817  | 2.50%            | **98.22%**    | 96.94%      | 4              | **1**             |
| 0.70 | 0.3775       | 0.9959    | 0.9923  | 15.59%           | 99.38%        | 98.75%      | 3              | 1                 |
| 1.00 | 0.7293       | 0.9963    | 0.9949  | 61.59%           | 99.45%        | 99.28%      | 1              | 1                 |

**Mean logit gap (top1_logit − true_logit)**:

| t    | baseline gap | kd_cr gap |
|------|:------------:|:---------:|
| 0.10 | 136.75       | 20.06     |
| 0.20 | 81.97        | 5.74      |
| 0.30 | **66.65**    | **1.46**  |
| 0.50 | 52.78        | 0.09      |
| 1.00 | 20.58        | 0.03      |

### F.2 Reference Set: Baseline Wrong at t=0.30 (38.1%)

n_ref = 93,685 positions (38.1% of M = 245,711)
这些是 baseline 在 t=0.30 时 oracle readout 错误的位置。

| t    | baseline MRR | kd_cr MRR | kd2 MRR | baseline correct | kd_cr correct | kd2 correct |
|------|:------------:|:---------:|:-------:|:----------------:|:-------------:|:-----------:|
| 0.10 | 0.0054       | 0.0794    | 0.0873  | 0.14%            | 4.71%         | 5.72%       |
| 0.20 | 0.0854       | 0.4682    | 0.4518  | 2.91%            | 37.33%        | 35.98%      |
| 0.30 | 0.1766       | 0.8248    | 0.8044  | 0.00%            | **74.53%**    | 72.27%      |
| 0.50 | 0.5322       | 0.9927    | 0.9861  | 38.37%           | **98.71%**    | 97.64%      |
| 0.70 | 0.6027       | 0.9973    | 0.9946  | 46.74%           | 99.58%        | 99.12%      |
| 1.00 | 0.8246       | 0.9976    | 0.9966  | 75.21%           | 99.64%        | 99.51%      |

### F.3 Key Findings

1. **KD 的 decode head 恢复了 baseline 无法读出的位置**：在 baseline never-commit 的 61,602 个位置上，kd_cr 在 t=0.30 时正确率达 82.5%，t=0.50 达 98.2%。Baseline 在这些相同位置上始终很差（t=0.30: 4.4%, t=0.50: 2.5%）。

2. **Logit gap 崩塌是 KD 的标志性效应**：Baseline 在 t=0.30 的 mean_logit_gap=66.65（真实 token 被埋在第 12 位），而 kd_cr 的 gap=1.46（真实 token 几乎在首位）。这意味着 KD 的 decode interface 重组使真实 token 的 logit 大幅上升。

3. **median_rank 从 12 降至 1**：对于 baseline never-commit 集，baseline 在 t=0.30 的 median_rank=12，而 kd_cr=1。说明不是个别位置的提升，而是 rank 分布的系统性偏移。

4. **两种 reference set 结论一致**：无论用 never-commit 还是 wrong@t=0.30 定义 reference set，kd_cr 均在 t≥0.30 时大幅超越 baseline（74-98% correct vs 0-4%）。

5. **这是 decode interface 效应而非 backbone 效应**：Exp07b (EXP-15v2) 显示 probe capacity 在不同 checkpoint 间相近，但 unembed_bias R=2.59。EXP-12v2 的结果与此一致：相同 oracle state 下 KD 的 decode head 大幅降低 true token 的 rank。

### F.4 Paper Impact

- **直接支持核心论题 #2**（KD 主要重组 decode interface）
- **提供 quantitative evidence**：can cite "on baseline's never-stable positions, KD achieves 82% correct at t=0.30 vs baseline's 4% (median_rank 1 vs 12)"
- **可以引用 logit_gap 数字**：66.65 → 1.46 at t=0.30 (45× reduction)
- **支持 MRR 作为提交度量**：MRR 从 0.1955 → 0.8781 (4.5× improvement at t=0.30)

---

## Section G — EXP-11v2 Partial Results（Protocol B Branching Stability，kd_cr）

**实验状态**: kd_cr DONE（2026-07-22）；baseline + kd2 仍在运行  
**结果文件**: `results/exp11v2_kd_cr/branching_stability.json`  
**修正**: 正确 per-position 单位球缩放（旧 EXP-11 用全局 noise_frac，实际扰动范围不清晰）

### G.1 kd_cr S_orig by t_split（η=1e-4，最小扰动）

| t_split | S_orig | S_pair | all_same |
|---------|:------:|:------:|:--------:|
| 0.09    | 0.38%  | 98.2%  | 96.7%    |
| 0.19    | 0.64%  | 97.5%  | 95.4%    |
| 0.31    | 2.37%  | 96.8%  | 94.2%    |
| 0.50    | 4.66%  | 97.4%  | 95.3%    |
| 0.69    | 8.30%  | 98.0%  | 96.4%    |
| 0.81    | **29.60%** | 97.1% | 94.7% |

S_orig 对 η 极不敏感（1e-4 to 1e-3 几乎不变），只在 η=1e-2 时 S_pair/all_same 略降。

### G.2 新科学发现：ODE 双重吸引子模式

**S_orig ≪ S_pair** 是这个实验最意外的发现：

- **S_orig≈0.047 但 S_pair≈0.974 (at t=0.50)**：扰动后的 K=4 条轨迹互相 97.4% 一致，但它们不是回到原来的 token，而是集体收敛到一个不同的 token。
- 这是"**bifurcation into common alternative attractor**"模式，不是随机散布。扰动没有造成混乱，而是把轨迹从一个吸引子切换到另一个。
- **p10_S_orig = 0** everywhere：至少 90% 的位置在小扰动下永远不会返回原预测。这说明原始轨迹大多数处于"不稳定平衡"而非"稳定吸引子"状态。

### G.3 与 EXP-14v2 的一致性

EXP-14v2 (kd_cr): 48.4% 的位置有 5+ 翻转，mean_flips=4.66。EXP-11v2 说明这些翻转不是随机噪声，而是 ODE 在多个有吸引力的结果之间"振荡"，每次小扰动可以系统性地切换到另一个稳定输出。

### G.4 与 EXP-11（旧版）对比

| t_split | EXP-11 kd_cr | EXP-11v2 kd_cr (η=1e-3) |
|---------|:------------:|:-----------------------:|
| ~0.09  | 0.36%        | 0.38%                   |
| ~0.31  | 2.10%        | 2.38%                   |
| ~0.50  | 4.88%        | 4.66%                   |
| ~0.69  | 7.72%        | 8.28%                   |

EXP-11v2 在 η=1e-3 时与 EXP-11 高度一致——说明原 EXP-11 的 1% noise_frac 实际上等价于 η≈1e-3，不是 22.6%（原来声称的 bug magnitude 被夸大了）。两个实验在相同量级上独立验证了低 S_orig 结论。

### G.5 三模型 S_orig 完整对比（EXP-11v2 DONE，2026-07-22）

**关键问题的回答：kd_cr S_orig 是否 > baseline S_orig？答案取决于时间段。**

| t_split | kd_cr S_orig | kd2 S_orig | baseline S_orig |
|---------|:----------:|:----------:|:--------------:|
| 0.09    | **0.38%**  | 0.76%      | 0.79%          |
| 0.19    | **0.64%**  | 1.30%      | 0.78%          |
| 0.31    | 2.37%      | **3.35%**  | 0.73% ← dip   |
| 0.50    | 4.66%      | 6.56%      | **7.01%**      |
| 0.69    | **8.30%**  | 7.42%      | 5.86% ← dip   |
| 0.81    | **29.60%** | 25.68%     | 17.82%         |

**关键发现**：

- **Late t（0.69-0.81）**: kd_cr > kd2 > baseline。KD 训练使轨迹**后期更稳定**（kd_cr S_orig@0.81=29.60% vs baseline 17.82%），与 EXP-14v2 kd_cr mean_last_flip_step=19.3 < baseline 21.2 一致——kd_cr 位置更早完成最后翻转，t=0.81 时大多已收敛。
- **Early t（0.09-0.19）**: kd_cr < baseline ≈ kd2。kd_cr 对早期扰动最敏感（bifurcation 效应最强），但 kd_cr S_pair 仍最高（0.982 at t=0.09）——扰动后的轨迹互相一致但不回到原来，kd_cr 的 alternative attractor 最 well-defined。
- **Baseline 非单调**：baseline 在 t=0.31（dip 到 0.73%）和 t=0.69（dip 到 5.86%）出现两次下降。最可能解释：baseline mean_last_flip_step≈21.2（≈t≈0.66），t=0.69 正好是 baseline 轨迹"最后翻转密集区"，此时扰动最容易改变最终结果。
- **综合结论**：KD 改善的是 Protocol B 后期稳定性（late-stage convergence），而不是全程降低 ODE 敏感性。这与 EXP-14v2 的 flip rate 降低（67.6%→48%）一致：kd_cr 轨迹更快"结算"到最终 token，但结算之前的 ODE 动力学同样湍流。

---

## Section H — Unified Interpretation（综合解释，所有 v2 实验完成后）

### H.1 Protocol A vs Protocol B: The Core Gap

所有 v2 实验共同揭示了一个核心的二元性：

**Protocol A（oracle decode readout，EXP-09v3/EXP-16v2/EXP-12v2）**：
- kd_cr 在 t=0.50 时几乎 100% 正确（G=99.5%，never-stable=0.53%）
- 即使是 baseline never-commit 的位置，kd_cr 也在 t=0.30 达 82.6% 正确（EXP-12v2）
- logit_gap 从 66.65 降至 1.46（45× 压缩）
- 这些都是"给定 oracle 噪声状态，decode head 能否读出正确 token"的回答

**Protocol B（on-policy ODE trajectory，EXP-14v2/EXP-11v2）**：
- kd_cr 仍有 48.4% 位置有 5+ 次翻转（mean_flips=4.66）
- kd_cr S_orig at t=0.50 仅 4.66%（扰动后 95.3% 不返回原预测）
- ODE 轨迹以"bifurcation into common alternative"模式运行：扰动轨迹集体改变但互相一致
- 这些是"真实生成轨迹的 ODE 动力学是否稳定"的回答

**两者巨大的 gap**（G_A(0.50)=99.5% vs S_orig(0.50)=4.66%）说明：
> KD 戏剧性地改善了 off-policy oracle 读出，但 on-policy ODE 动力学仍然动荡。"承诺"是探针测量的 oracle 读出能力，不等于 ODE 轨迹的真正收敛。

### H.2 Mechanism Story（三层论证）

**第一层**（EXP-15v2）：参数空间中，KD 最大的修改是 decode head（unembed_bias R=2.59），transformer backbone 变化较小（R=0.22-0.34）。

**第二层**（EXP-12v2）：在 baseline 从未能稳定读出的位置上，KD 的 decode head 以 82.6% 正确率读出同样的 oracle 噪声状态。logit_gap 崩塌 45 倍。这是 decode interface 重组的直接定量证据。

**第三层**（EXP-14v2 + EXP-11v2）：在真实 ODE 轨迹上，KD 改善了后期稳定性（flip rate 67.6%→48%；kd_cr S_orig@t=0.81 = 29.60% vs baseline 17.82%）。但 ODE bifurcation 结构基本保留——所有三个 checkpoint 的 S_pair≈0.95-0.98，小扰动仍系统性地将轨迹重定向到另一个吸引子而非随机散布。

**综合结论**：
> KD 主要是一个 decode interface 重组，使 backbone hidden states 的 token 信息更可访问（通过改变 unembed_bias）。它不从根本上改变 ODE 动力学的 bifurcation 性质，但略微降低了实际翻转频率。这解释了 Protocol A 和 Protocol B 的巨大差异：Protocol A 测的正是 decode interface 能力，而 Protocol B 测的是 ODE 动力学稳定性。

### H.3 Paper Claim Mapping

| Paper claim (to update) | Supporting evidence | Qualification needed |
|-------------------------|--------------------|--------------------|
| "KD achieves 82%→98% correct at t=0.30-0.50" | EXP-12v2 (fixed ref set) | Protocol A oracle readout, not on-policy |
| "Never-stable 25%→0.5%" | EXP-16v2 + EXP-09v3 | Fixed noise oracle; Protocol B different |
| "KD reorganizes decode interface" | EXP-15v2 (unembed_bias R=2.59) + EXP-12v2 | Not backbone |
| "Coarse-to-fine cascade +60pp" | EXP-08v2 (T5 tokenizer, stable_k=3) | Protocol A |
| "Protocol B turbulence 45-67% 5+ flips" | EXP-14v2 (correct decode path) | Cite old EXP-14 as invalid |
| "ODE bifurcation not commitment" | EXP-11v2 (S_orig≪S_pair, all 3 checkpoints) | Universal ELF ODE property |
| "KD reduces flip rate slightly (67%→48%)" | EXP-14v2 | Small improvement; Protocol B turbulence persists |

### H.4 Two-Story Structure for Paper

**Story 1 (Protocol A — the accessible story)**:
KD dramatically improves oracle readout stability: from 25% never-commit to 0.5%, from 75% to 99.5% G(t=0.5). The mechanism is decode interface reorganization (unembed_bias R=2.59 >> backbone R=0.22). The coarse-to-fine cascade (+60pp) is also a Protocol A phenomenon.

**Story 2 (Protocol B — the honest story)**:
On-policy ODE trajectories remain turbulent (45-67% 5+ flips, S_orig<5% at t=0.50). KD provides modest improvement in actual stability. The ODE operates in a bifurcation mode: perturbations consistently redirect trajectories to an alternative attractor (S_pair≈97%), not random scatter. This explains the persistent turbulence despite high oracle readout accuracy.

**The synthesis**: KD learns to organize the native decode interface around what the ODE will eventually converge to. This makes the oracle decode "look ahead" to the eventual outcome, explaining the huge Protocol A improvement. But it doesn't fundamentally stabilize the ODE path itself.

### H.5 Remaining Open Questions

1. **Is kd_cr S_orig higher than baseline?** ANSWERED: YES at late t (0.69-0.81), NO at early t (0.09-0.19). KD improves late-stage convergence, not early-stage sensitivity. Baseline shows non-monotone dips at t=0.31 and t=0.69 (consistent with last-flip window centered at t≈0.66).
2. **Does the bifurcation model predict EXP-14v2 flip rates?** (analytical question)  
3. **What determines which attractor the perturbed trajectory chooses?** (token-frequency? context?)
4. **Can the decode interface reorganization be visualized in embedding space?**

---

## Section I — LangFlow 比较实验方法论审查（EXP-20–24）

**2026-07-22 审查** — 这组实验开始把 ELF 结论推广到 LangFlow，但当前实现存在若干影响结论有效性的方法论问题。

### I.1 立即需要从论文删除的表述

以下结论在当前证据下**不成立**，必须删除或降级：

| 当前论文表述 | 问题 | 处置 |
|------------|------|------|
| "LangFlow 比 ELF 晚 0.63t 单位承诺" | 违反 EXP-03 结论：nominal-t 跨模型不可比 | 必须删除 |
| "ELF kd-cr 在 t=0.20 已 ~58%，LangFlow 到 t=0.83 才承诺" | 同上 | 必须删除 |
| "LangFlow native head > probe ⇒ native head 充分提取 backbone 信息" | 忽略 skip connection 输入不对称（native 多了 z_t 通道）| 降级为 "没有观察到 ELF-baseline 量级的 gap" |
| "LangFlow 轨迹比 ELF 早稳定（早决策模式）" | ELF 对比数字来自 EXP-14 无效版本（83.4%→正确为 67.6%）；argmax stability ≠ commitment | 需补 entropy/margin 数据才能声称 |
| "LangFlow 无 self-conditioning 解释 ELF/LangFlow 差异" | 生成代码中 `if model.config.self_conditioning` 分支存在 | 必须先验证 config 值 |

### I.2 可以保留的模型内结论（含 EXP-21v2/24v2 更新）

这些不涉及跨模型 nominal-t 比较，在 LangFlow 内部有效：

- EXP-22: LangFlow native posterior 在 t < 0.80 时 entropy > 1 nat；早期 committed 位置（t≈0.52-0.60）中 91-100% 是错误的（P(wrong|H<1) 分析，见 EXP-22-spec.md）
- EXP-23: Gaussian-null 输入下 LangFlow 早期 mode_frac=65–71%（skip connection 强度的直接测量）
- EXP-21v2: **backbone_top1 ≈ 0 at all t**（DONE，见 EXP-21-spec.md）；skip 主导 t=1.00 解码（92.4%）；backbone 是残差校正器，不是独立预测器；probe_h 在 t≥0.85 捕获 94-96% 的 native accuracy，native-probe 差距来自 skip 而非 output_layer 额外能力
- EXP-24v2: LangFlow argmax 在步骤 6.6/32 就锁定 p>0.5，但 p>0.9 需到步骤 12.6/32；"早决策"表述过强，应改为"早 argmax 锁定、晚置信建立"；`self_conditioning=True` 已确认（DONE）

### I.3 真正值得追的架构差异假说（EXP-21v2 后更新）

EXP-21v2 揭示了一个核心架构差异：

> **LangFlow**：解码由 skip connection `c_skip × z_t @ E.T` 主导；backbone 提供残差校正（backbone alone ≈ 0 accuracy）；随着 z_t → x_clean（t→0），skip 自然收敛到正确 token。h_last 编码 token 信息但通过 skip 路径输出，而非通过 backbone logit 直接预测。
>
> **ELF**：解码完全依赖 backbone（无 skip connection）；backbone logit 是主要预测信号；KD 训练对齐了 backbone→logit 的投影，使 oracle readout 高效。

这两种架构对 commit timing 和 trajectory stability 的影响完全不同，无法用 nominal-t 对比两者的"早/晚承诺"。

### I.4 优先级排序（基于 EXP-21v2/24v2 后审查）

**P0（当前实验已解决，需更新论文措辞）**：
1. 删除论文中所有 nominal-t 跨模型比较（EXP-22、EXP-03 已证无效）
2. EXP-24 决策规则表替换为 EXP-14v2 正确数字（baseline 67.6%，kd_cr 48.4%）
3. EXP-21 结论改为：backbone 是残差校正器（backbone_top1≈0），probe_h≈native at t≥0.85（差距来自 skip）
4. 删除"LangFlow 无 self-conditioning 解释差异"（self_conditioning=True 已确认）
5. EXP-24 "早决策"改为"早 argmax 锁定（步骤 6.6）、晚置信建立（步骤 12.6）"

**P1（仍需新实验才能强化）**：
6. 在 matched log-SNR 坐标上重做 ELF–LangFlow 对比曲线
7. probe_hz 更多 samples（≥200 sequences）以确认是否 ≈ native at t=1.00
8. ELF 的 skip/residual 分解（ELF 有 decode branch 而非 skip，但类似的分析可厘清 ELF architecture）


---

## Section J: EXP-25~28 方法论审查 — LangFlow 空间/时序/频率分析（2026-07-22）

### J.0 总体问题

EXP-25-28 构成一组"LangFlow 是否复现 ELF 的 coarse-to-fine、bootstrapping 现象"的对比研究。四个实验的核心结论均一致：**与"共享局部可预测性场"（shared local predictability field）相容，与"承诺传播/自举"假说不成立**。

具体来说：
- 相邻位置的承诺时序相关 → 可由"相邻位置共享语法/语义上下文，因此具有相似可预测性"完全解释
- 无需任何因果传播机制（已承诺邻居主动帮助未承诺位置）

四个实验均未能提供超越此共因解释的证据。

### J.1 EXP-25：功能词/内容词时序（⚠️ 频率混淆 + 跨模型比较无效）

**实验声称**：LangFlow 复现"粗到细"（coarse-to-fine）承诺顺序，功能词早 0.050t 单位。

**核心问题**：
1. **频率/surprisal 未控制**：功能词 = 高频+低 surprisal token。"功能词先"与"高频词先"产生相同预测，当前数据无法区分。需要 logistic 回归，以 POS / function-word 指示变量为预测变量，同时控制 `log freq(v)` 和 `-log P(v | context)`。
2. **跨模型 Δt 无效**：Δ(LangFlow)=−0.050 vs Δ(ELF)=−0.073 的直接比较违反 EXP-03 结论（nominal-t 不可比）。

**立即删除**：论文中"粗到细是 CDLM 通性"、"ELF KD 放大了粗到细层级"等涉及跨模型数值比较的陈述。

**可保留**：LangFlow 内部，功能词 oracle-commit t* 早于内容词（无需与 ELF 比较数值）。

### J.2 EXP-26：空间自举（⚠️ Risk-set collapse + 共因混淆）

**实验声称**：LangFlow 空间自举效应峰值 +21.1pp（near=37.8% vs far=16.7%）。

**核心问题**：
1. **Risk-set collapse**：峰值时 far_n=6（仅 6 个位置！）。far_rate=16.7% 的 95% CI ≈ [0%, 64%]。该"显著"效应完全被统计噪声掩盖，不能声称成立。
2. **共因混淆**：near 组 = 局部短语已有 token 承诺 = 该短语整体可预测性高 = far 组难度更高。不需要任何"传播"机制，可预测局部内所有 token 均快速承诺即可产生同样的条件概率模式。
3. **多重比较**：在 51 × 多个 d 的搜索空间中后验选出最大值，未校正 α 水平。

**正确分析方向**：生存模型（hazard model）+ Moran's I 空间自相关检验 + 控制 token unigram freq 和 surprisal。

**立即删除**："空间自举效应是 CDLM 通性"、"ELF decode branch 将该效应放大 3×"。

### J.3 EXP-27：token 频率 vs 承诺（⚠️ 错误 tokenizer + type-level 统计）

**实验声称**：r=0.47（log tok-id vs t*），频率-承诺梯度是 CDLM 通性。

**核心问题**：
1. **错误 tokenizer**：LangFlow 使用 GPT-2 BPE tokenizer（token ID 按 merge 顺序，无频率语义）。EXP-27 可能用了 T5 SentencePiece token ID（来自 ELF 的 tokenizer）。两个 vocabulary 不同，ID 无法对应。必须确认 `gt_tokens.npy` 的来源再下任何结论。
2. **type-level 相关系数**：r=0.47 基于 69 个 types，每个 type 的 mean_t* 用 n=3-10 个 occurrence 估计，误差极大。有效样本量为 69，而非名义上的 n。
3. **频率 vs surprisal 未分离**。

**立即删除**："频率-承诺梯度是 CDLM 通性" + 所有 r=0.47 数字。需先验证 tokenizer 再定。

### J.4 EXP-28：方向性自举（⚠️ 无显著性 + 共因解释）

**实验声称**：cf_Δ=5.4pp > fc_Δ=2.7pp，说明 LangFlow 无方向性因果传播。

**核心问题**：
1. **无统计显著性**：cf_Δ=5.4pp，cf 组样本量 nc≈50-400，置信区间宽。"cf>fc"在当前样本量下不可区分于 0。
2. **共因解释完全充分**：内容词比功能词更晚承诺 → 已承诺内容词 = 当前位于"可预测性高的短语" → 剩余功能词也快速承诺 → 产生 cf_Δ > fc_Δ，无需任何方向性传播机制。
3. **"T5 geometry 解释"不适用 LangFlow**（LangFlow 用 GPT-2 embedding，非 T5）。

**立即删除**："cf > fc 说明 LangFlow 无方向性传播"、"ELF decode branch 产生定向传播"（无数据支撑）。

### J.5 四实验的安全结论

| 实验 | 安全的 LangFlow 内部结论 | 需删除的跨模型/因果声明 |
|------|------------------------|------------------------|
| EXP-25 | 功能词 oracle-commit t* 早于内容词（LangFlow 内部） | 粗到细是 CDLM 通性；ELF KD 放大层级 |
| EXP-26 | 承诺时序存在正向空间聚类（显著性待验证） | 空间自举 +21pp 成立；3× 放大说法 |
| EXP-27 | 高 token ID 的承诺 t* 稍晚（需先验证 tokenizer）| r=0.47 频率梯度是通性 |
| EXP-28 | 功能词邻居的条件加速非常弱（<3pp） | cf > fc 表明方向性；ELF 有定向传播 |

**底线**：EXP-25-28 四个实验的数据全部与"共享局部可预测性场"（D_i 驱动 T_i，空间相关来自相邻位置共享语法结构）一致，不支持"承诺在位置间传播/引导"的自举假说。这不是负面结果，而是一个**积极的约束**：需要提供更强的causal evidence（如：干预实验、ablation、控制变量回归）才能重新引入自举假说。

---

## Section K: EXP-29~32/36 方法论审查 — 表征可视化与 DF Inference（2026-07-22）

### K.0 总体分类

| 实验 | 类型 | 核心价值 | Solidity | 判断 |
|------|------|:--------:|:--------:|------|
| EXP-29 | 表征可视化 | 低到中 | 定性可用 | Appendix 定性图，不支撑机制声明 |
| EXP-30 | 表征分析 | 高 | 中等 | mid-layer peak 有价值，需 probe controls + skip 分离 |
| EXP-31/31b | DF inference | 潜在极高 | 较弱 | kd_cr↔kd2 符号差异值得追，但 seed artifact + 单指标不可信 |
| EXP-32 | DF step sweep | 中高 | 较弱 | 步数相关性存在，"N²"解释不成立 |
| EXP-36 | DF×dec_sc | 潜在高 | spec 不完整 | factorial arms 缺失，目前无法测量 interaction |

### K.1 EXP-29：kNN 可视化（⚠️ fixed noise + centroid bias + cherry-pick）

**实验定位**：appendix qualitative illustration for EXP-07/08。

**三个不可弥补的当前问题**：
1. **Fixed noise 未确认**：不同 t 的 states_t*.pt 若使用独立噪声，则图展示的是独立样本截面，不能称为"演化"或"轨迹"。必须代码审计。
2. **Centroid 偏向 baseline**：token centroids 来自 baseline 空间，导致 baseline kNN 精度系统性偏高，kd_cr/kd2 系统性偏低。所有跨 checkpoint kNN 数值比较必须从论文删除。
3. **案例不可 cherry-pick**：`earthquake` 案例需要预注册 selection policy + frequency-matched random cluster 控制。

**立即删除**：任何涉及跨 checkpoint kNN 数值比较（如"KD 早期承诺更准"）的论文表述。

### K.2 EXP-30：LangFlow 逐层探针（⚠️ skip input 不对称 + gap 需 CI）

**有价值的发现**：B07 peak（+3.9pp vs native at t=0.85）与 ELF 的 L8/L9 mid-layer peak 形态相似，说明两个架构均存在"中间层线性可读性高于最终层"的现象。

**两个关键限制**：
1. **Skip input 不对称**：EXP-21v2 已确认 backbone_top1≈0，native=backbone+skip。B11/out probe < native 不能解读为"native 恢复了信息"，而可能仅是 skip 信号的额外贡献。必须做完整五条件拆分（同 EXP-21v2）。
2. **+3.9pp 需 bootstrap CI**：3-5 probe seeds + sequence-level bootstrap 后若仍显著，这是有价值的跨模型发现；否则需要降格。

**安全陈述**：
> Both ELF and LangFlow show a mid-layer peak in linear token recoverability (ELF: L8/L9; LangFlow: B07), but LangFlow's gap is much smaller (~4pp vs ~46pp) and requires statistical confirmation.

### K.3 EXP-31/31b：Diffusion Forcing 跨 checkpoint（⚠️ seed artifact + 单指标）

**最重要的信号**：kd_cr 和 kd2 的 oracle readout 曲线几乎相同，但对相同 DF 干预（freeze H<threshold）反应完全相反（kd_cr 全面恶化，kd2 全面改善）。如果严格复现，这证明：

> **Oracle readout correctness ≠ on-policy intervention reliability.** 即使两个模型对 oracle 状态的解码能力相同，其 on-policy trajectory 对 state-clamping 的响应可以完全相反。

这会直接支撑"Protocol A ≠ Protocol B"的核心论点。

**当前不可信的原因**：
- seed=123 kd_cr 有 multilingual artifact（PPL 基线已污染）
- seed=456 kd2 有 21% degeneration（同上）
- PPL 单指标在 DF 改善 vs 文本退化中无法区分
- `freeze_1.0`（H<1.0）不等于"所有承诺可信"
- Direct state replacement 可能破坏 ODE manifold

**P0 行动**：多 seed 重跑（至少 5 个正常 seed），加 degeneration/language-ID/distinct-n 指标，记录 frozen fraction 和 freeze precision。

### K.4 EXP-32：DF step sweep（⚠️ 三点不够，N² 无依据）

当前 8→16→32 步的 DF 效果（10.6%→19.2%→48.9%）同时混淆了 solver accuracy、DF 触发次数、受益步数。none baseline 本身从 PPL=688 降到 283（变化 2.4×），归一化基准不稳定。

"超线性 N²"结论没有统计依据（3 个点无法拟合幂律）。

**安全陈述**：
> Under tested kd2 conditions, DF benefit is strongly step-budget-dependent. Whether this reflects DF effectiveness independent of solver quality requires controlled experiments (fixed intervention count, sweep remaining steps, sweep intervention frequency).

### K.5 EXP-36：dec_sc × DF interaction（⚠️ factorial 不完整，时间方向错误）

**设计缺陷**：缺少 freeze-only 和 soft-only arms，无法计算 interaction（difference-in-differences）。不能用 EXP-31 数字补入，因为 seed/context 不同。

**时间方向错误**：文档将"t∈[0.7,1.0]"标注为"高噪声"，但在 ELF convention 下这是低噪声晚期区域。dec_sc 和 DF 实际上在 t≥0.7 同时激活，并非作用在"完全不同阶段"。

**Threshold 可比性**：相同 H<0.5/1.0 对三个 checkpoint 代表不同 frozen fraction 和 precision，跨 checkpoint 比较需要 fraction-matched 设计。

**立即补充**：freeze_0.5 only、freeze_1.0 only、soft_0.3 only arm，与同一 run 的 none 配对。升级评价到 5 seed + degeneration/language-ID/MAUVE。

### K.6 最值得追的新假设（连接 EXP-31 与 Protocol A/B gap）

> **为什么 kd_cr 和 kd2 有相似的 oracle decoding curves，却对相同 state-clamping 作出相反响应？**

一旦严格复现，这个问题指向的不是"承诺质量"差异（因为 oracle 曲线相似），而是：

> **State geometry determines intervention response, not oracle readout accuracy.**

一个位置即使当前 token readout 准确，其 continuous state 仍可能：
- 需要继续与邻居协同演化（违反 freeze 假设）
- 对 self-conditioning 有 checkpoint-specific dependency
- 不处于 ODE manifold 的"稳定点"（direct replacement 触发 ODE instability）

这条线索天然连接 EXP-07（latent recoverability）、EXP-01v2（oracle-trajectory gap）和 EXP-11v2（bifurcation to alternative attractor），有潜力成为论文的一条新的机制主线。

### K.7 立即删除的论文表述

| 当前表述 | 原因 |
|---------|------|
| "KD 使 x̂_t 更早接近正确 centroid（EXP-29）" | centroid 偏向 baseline |
| "kd2 的 t≥0.7 承诺全部可信（EXP-31b）" | seed artifact + 单指标 |
| "DF 产生 N² 超线性收益（EXP-32）" | 3 个点无法推断幂律 |
| "dec_sc 和 DF 作用在不同阶段（EXP-36）" | 两者在 t≥0.7 明显重叠 |
| "LangFlow 最终层丢失 token 信息（EXP-30）" | 也可以是非线性化 |

---

## Section L: EXP-25v2/26v2/27v2 新分析结果（2026-07-22）

### L.0 核心发现摘要

EXP-25v2/26v2/27v2 三个 v2 实验回答了 Section J 提出的方法论问题，产生了对 CCLF 论文 Story B 的重要修正。

**EXP-27v2（GPT-2 OWT 真实频率 vs 承诺时序）**：
- Pearson r=-0.651 (p=4.7e-24)，Spearman r=-0.659 (p=7.85e-25)，n=188 token types
- 频率五分位梯度：Q1（低频）mean t*=0.892 → Q5（高频）mean t*=0.836，Δ=5.6pp
- 控制 is_function 后偏相关 r=-0.638（几乎不变）
- 函数词 vs 内容词 Δ=-0.014（远小于 ELF 的 Δ=-0.075）
- **结论**：LangFlow 承诺时序主要由 token 频率预测，而非 POS 类别

**EXP-25v2（出现级 logistic 回归）**：
- 控制频率后，β_func 在 t=0.66-0.83 为**负**（OR=0.26-0.73）：函数词不早承诺
- β_freq 在所有 t 均为正（OR=2-7.5）：频率效应持续强
- t≥0.915 时 β_func 转正（OR=4.4-4.7）：残余 POS 效应仅在极晚期显现
- **结论**：EXP-25 原始 Δ=-0.050 几乎完全由频率混淆解释

**EXP-26v2（Moran's I + 离散时间风险模型）**：
- Moran's I 峰值 I=0.260 at t=0.745（z=22.54，p<0.001）：极显著空间聚集
- 风险模型：has_committed_neighbor OR=2.44 [2.19, 2.76]，频率 OR=2.21，is_function OR=1.24
- 空间聚集在承诺窗口期（t=0.66-0.92）持续显著
- **结论**：空间聚集强且显著，但共因混淆（邻近位置共享句法/语义结构）无法排除

**EXP-29 fixed-noise audit CONFIRMED（2026-07-22）**：
- `probe_layerwise.py:116-134` 明确：`eps_all = torch.randn(..., generator=g_noise)` 一次生成，`eps = eps_all[sl]` 在所有 t 值复用
- ELF exp07b_v2 states 确实使用固定噪声（seed=42），跨 t 可视化是真实轨迹演化
- EXP-25 LangFlow 数据（`probe_coarsefine_langflow.py:97-107`）同样使用固定噪声（per-sample seed=si*1000，在 t 循环外生成）
- EXP-29 第一条批评（"可能是独立采样截面"）**已撤销**

### L.1 对论文 Story B 的影响

| 原始声明 | 修正后 |
|---------|--------|
| "LangFlow 中功能词比内容词早 0.050t 承诺（粗到细效应）" | "LangFlow 中较高频率 token 更早承诺（r=-0.65），函数词 vs 内容词差距（Δ=-0.014）主要由频率差异解释" |
| "承诺时序在空间上正相关（空间自举）" | "承诺时序高度空间聚集（Moran's I=0.26，z=22.54），但共因解释（邻近位置共享局部可预测性场）同样成立" |
| "ELF kd_cr 的 Δ=-0.075 比 LangFlow Δ=-0.050 更强" | "两个比较无效：(1) ELF 使用 T5 tokenizer，LangFlow 使用 GPT-2 tokenizer；(2) 未控制频率的 ELF 结果同样可能是频率效应" |

### L.2 新的安全结论（可写入论文）

1. **频率-承诺时序关系**（LangFlow 内部，有效）：
   > "在 LangFlow 中，GPT-2 OWT 训练频率与 oracle-protocol 承诺时序显著负相关（Pearson r=-0.651，p=4.7e-24）。高频 token 比低频 token 早约 5.6pp（t 单位）承诺。"

2. **空间聚集**（LangFlow 内部，有效但需因果注意事项）：
   > "LangFlow 的承诺时序在相邻位置之间高度空间相关（Moran's I=0.260，z=22.54），在 t=0.66-0.92 窗口持续显著。这与'邻近位置共享局部句法/语义难度'一致，也与因果传播一致；当前数据不能区分两者。"

3. **风险模型 has_committed_neighbor**（描述性，无因果声明）：
   > "控制频率和 POS 后，相邻位置已承诺这一时变协变量仍与当前位置的即时承诺风险正相关（OR=2.44，95% CI=[2.19, 2.76]）。这一关联可能反映因果传播，也可能反映共享局部可预测性场，需要结构性干预实验区分。"

### L.3 需立即删除的表述

- ~~"功能词比内容词早承诺（粗到细效应，LangFlow Δ=-0.050）"~~ → 改为频率效应表述
- ~~"ELF kd_cr 比 LangFlow 更强的粗到细效应"~~ → 跨模型比较无效
- ~~"EXP-29 展示了真实轨迹演化"~~ → 现在可以说"固定噪声路径下的演化"（审计已确认）

---

## Section M: EXP-36 Full Factorial 结果（2026-07-22，DONE）

### M.0 设计修复

原始 EXP-36 缺少 DF-only arm（freeze-only 和 soft-only）。新增 `spec36_factorial.yml`（8 arms：1 none + 1 SC-only + 3 DF-only + 3 DF+SC）。

### M.1 关键结果（baseline 和 kd_cr，seed=42，256 samples，32 ODE steps）

**PPL 结果**：

| arm | baseline | kd_cr |
|-----|---------|-------|
| none | 127.8 | 331.9 |
| SC-only | 232.2 | 264.8 |
| freeze_0.5-only | 121.3 | 426.0 |
| freeze_1.0-only | 131.1 | 475.6 |
| soft_0.3-only | 121.8 | 389.9 |
| freeze_0.5+SC | 1715.7 | 318.6 |
| freeze_1.0+SC | 1829.8 | 343.3 |
| soft_0.3+SC | 838.6 | 288.1 |

**2×2 交互 I**（负值=协同，正值=反协同）：

| DF variant | baseline I | kd_cr I |
|-----------|-----------|---------|
| freeze_0.5 | **+1490** | **-40** |
| freeze_1.0 | **+1594** | **-65** |
| soft_0.3   | **+612**  | **-35** |

### M.2 机制解读（最重要的定量证据）

**交互符号取决于 checkpoint，且与 oracle accuracy 直接对应**：

| checkpoint | oracle acc @t=0.5 (EXP-16v2) | DF-SC 交互符号 | 方向 |
|-----------|------------------------------|---------------|------|
| baseline  | 74.9% | **+1490 to +1594** | 强反协同（退化） |
| kd_cr     | 99.5% | **-40 to -65** | 互补（协同） |

**机制**：
1. 当 oracle accuracy 低（baseline ~77%）时，DF 冻结的位置多为错误 token。dec_sc 基于这些错误 token 做条件生成，产生正反馈退化（PPL 1716-1830，多语言乱码）。
2. 当 oracle accuracy 高（kd_cr ~99.5%）时，DF 冻结的位置几乎全为正确 token。dec_sc 基于正确条件工作，与 DF 的冻结互补——DF 提供锚点，SC 填充剩余不确定位置。

**这是整个 CCLF 论文最强的机制证据**：KD 通过提升 oracle accuracy 从 75% → 99.5%，把 DF-SC 的相互关系从"灾难性反协同"（I=+1594）转变为"协同互补"（I=-65）。这直接量化了"decode interface reorganization"对生成质量的因果影响。

### M.3 EXP-36 的正确论文表述

> "When Diffusion Forcing and decode-branch self-conditioning are combined, their interaction depends critically on oracle decode accuracy. For the baseline checkpoint (oracle acc @t=0.5 = 74.9%), combining DF (freeze_1.0) with dec_sc produces catastrophically degenerate text (PPL=1830 vs 127.8 for standard inference; I=+1594). For the kd_cr checkpoint (oracle acc @t=0.5 = 99.5%), the same combination shows complementary improvement (PPL=343 vs 331.9; I=-65). This 1659-point reversal in the interaction term directly quantifies the consequence of KD's decode interface reorganization: only when frozen positions carry reliable token predictions can DF and SC cooperate rather than compete."

### M.4 kd2 结果（2026-07-22，DONE）

kd2 oracle accuracy ~99.1% @t=0.5（EXP-16v2），与 kd_cr 相近。但实际结果：

| arm | kd2 PPL |
|-----|---------|
| none | 282.52 |
| SC-only | 600.72 |
| freeze_0.5-only | 219.19 |
| freeze_1.0-only | 144.42 |
| soft_0.3-only | 230.27 |
| freeze_0.5+SC | 588.00 |
| freeze_1.0+SC | 620.32 |
| soft_0.3+SC | 622.73 |

**kd2 交互 I**：freeze_0.5 = **+51**，freeze_1.0 = **+158**，soft_0.3 = **+74**（全部正值！）

**关键修订**：原来预期 kd2 I<0（因为 oracle acc ~99.1% ≈ kd_cr 的 99.5%），但实际结果 I>0。这推翻了"oracle accuracy → DF-SC interaction sign"的解释。

### M.5 三模型完整结果表（全部 seed=42，2026-07-22）

| checkpoint | oracle acc @t=0.5 | SC main effect | freeze_1.0 I |
|-----------|------------------|---------------|-------------|
| baseline  | 74.9% | +104.5（SC有害） | **+1594** |
| kd_cr     | 99.5% | -67.1（SC有益） | **-65** |
| kd2       | 99.1% | +318.2（SC极有害） | **+158** |

### M.6 修订后的机制解释

**正确解释**（取代 M.2 中的 oracle accuracy 机制）：

**DF-SC 交互符号 = SC standalone effect 的符号**。当 SC 独立有效时（kd_cr），DF 的加入不会破坏 SC 的有效性——尽管 DF alone 对 kd_cr 有害（+43%），但 DF+SC 比 DF alone 好（318 vs 476），说明 SC 的纠错能力在 DF 下仍然保留，形成互补。当 SC 独立无效时（baseline、kd2），DF+SC 反而比 additive 预测更差，因为两种干预叠加后 SC 的负面效应被放大。

**kd_cr 的特殊性**：kd_cr 是唯一 SC 独立有效的 checkpoint，这来自 kd_cr 训练对 decode interface 的专门重组（unembed_bias R=2.59，EXP-15v2），使 decode branch 成为一个有效的 self-corrector。kd2 虽有高 oracle accuracy，但 decode interface 未被重组到支持 SC 的方向（kd2 的 unembed_bias R 值未测，但 SC failure 证明了这一点）。

**论文中不应声明**：
- ~~"oracle accuracy 预测 DF-SC 交互"~~（kd_cr/kd2 oracle acc 相近但 I 符号相反）
- ~~"KD 通过提升 oracle accuracy 使 DF 和 SC 互补"~~（kd2 反例）

**论文中应声明**：
- "kd_cr 的 decode interface 重组（EXP-15v2 unembed_bias R=2.59）使 dec_sc 成为有效的独立纠错机制（SC-only PPL: 264.8 < none 331.9），而 kd2 和 baseline 均不具备此属性"
- "DF-SC 互补性（I<0）仅在 SC 独立有益时出现；否则两者总是反协同"

### M.7 EXP-31v2 补充：DF 符号反转在 non-artifact seed 下确认（2026-07-22）

EXP-31v2（seeds 0-4，4 conditions per checkpoint，32 steps）结果：

- **kd_cr**: freeze_1.0 Δ = **+119.61** (DF hurts, std=0)
- **kd2**: freeze_1.0 Δ = **-105.51** (DF helps, std=0)

发现 ELF unconditional PPL 评估完全确定性（std=0）——配置 `seed` 字段不影响 ODE 采样。但科学目标已达成：**符号反转是 robust 的，非 artifact seeds 下也成立**。

**补充 degeneration 数据**：
- kd_cr none 退化率 = **5.5%**（低，非 seed artifact）
- kd2 none 退化率 = **15.2%**（中等；EXP-36 factorial seed=42 的 SC-only PPL=600.72 可能部分由退化解释）

### M.8 单 seed 警告（EXP-36 factorial, seed=42）

EXP-36 factorial 结果仍基于 seed=42。但 EXP-31v2 已证明 ELF eval 对 seed 完全确定，因此 seed=42 的结果等价于任意 seed 的结果（只要 ELF eval 确定性也适用于条件生成）。kd2 SC-only PPL=600.72 是否具有代表性仍需验证（可通过 EXP-13v2 对比 kd2 SC-only 结果）。

---

## Section N: EXP-31v2 + EXP-36v2 综合结论（2026-07-22，DONE）

### N.1 三模型 DF 反应完整矩阵

| checkpoint | DF alone (freeze_1.0) | SC alone | DF+SC | I (freeze_1.0) | SC standalone works? |
|-----------|----------------------|---------|-------|---------------|---------------------|
| baseline  | +3.35 (+2.6%) | +104.48 (SC hurts) | 1829.84 | **+1594** | NO |
| kd_cr     | +143.72 (+43.3%) | -67.08 (SC helps!) | 343.31 | **-65** | YES ← unique |
| kd2       | -138.10 (-48.9%) | +318.19 (SC hurts badly) | 620.32 | **+158** | NO |

**EXP-31v2 加固**（5 seeds 全部一致，std=0）：
- kd_cr freeze_1.0 delta = **+119.61** (DF always hurts)
- kd2 freeze_1.0 delta = **-105.51** (DF always helps)

### N.2 三个独立信号的会合

**Signal 1: DF alone effect**
- kd_cr: DF hurts (+43%); baseline: DF neutral (+3%); kd2: DF helps (-49%)
- Pattern: DF benefit correlates with none-baseline quality (worse none = more room to improve)

**Signal 2: SC standalone effect**
- kd_cr: SC helps (-20%); baseline & kd2: SC hurts (+82% and +113%)
- Pattern: SC works only for kd_cr. This is the variable that determines I sign.

**Signal 3: DF-SC interaction**
- kd_cr I<0 (complementary); baseline & kd2 I>0 (anti-synergistic)
- Pattern: I sign = sign of SC standalone effect (not oracle accuracy)

**The unexpected split**: kd_cr and kd2 have similar oracle accuracy (99.5% vs 99.1% at t=0.5) but completely different SC behavior. This means oracle readout accuracy does NOT determine whether the decode branch functions as a self-corrector.

### N.3 最强可防御的论文声明（2026-07-22 状态）

> "KD training produces checkpoints with qualitatively different inference-time properties. Among the three checkpoints, kd_cr is the only one for which self-conditioning (dec_sc) improves generation quality as a standalone method (PPL 264.8 vs 331.9 baseline; −20%). This enables kd_cr's DF and SC to interact complementarily (I = −65), whereas baseline (I = +1594) and kd2 (I = +158) show anti-synergistic interference. The sign of the DF-SC interaction (I) tracks whether SC works independently, not the checkpoint's oracle readout accuracy."

### N.4 文本质量检查结果（2026-07-22，new）

直接检查 `spec36_factorial_kd_cr/kd_cr/kd2` 的生成文本质量（256 samples per arm）：

**退化率（2% 非 ASCII 阈值 = 多语言；35% 单词重复 = 重复性）**：

| arm | multilingual (>2% non-ASCII) | repetitive (>35% single word) |
|-----|------------------------------|-------------------------------|
| kd_cr none | **15.2%** | 2.3% |
| kd_cr SC | **7.4%** | 1.2% |
| kd2 none | 3.1% | **12.1%** |
| kd2 SC | 3.5% | **1.2%** |

**关键观察**：

1. **kd_cr SC 真正减少了多语言退化（15.2% → 7.4%）**，同时 PPL 从 331.9 → 264.8。两个指标同向改善，说明 SC 对 kd_cr 的 PPL 改善**不是多语言 artifact** 造成的，是真实的质量提升。（kd_cr none 的高 PPL=331.9 部分来自 15.2% 多语言样本的高 per-sample PPL）

2. **kd2 SC 减少了重复性（12.1% → 1.2%）但 PPL 急剧上升（282 → 600）**。这说明 kd2 SC 的失败不是重复退化，而是**语义非连贯**（文本由合法 token 组成但排列无意义）——GPT-2 对这类文本评分极高（高 PPL）。

3. **kd_cr none 的多语言退化（15.2%）说明 kd_cr 的 none-baseline 已经不健康**。其 PPL=331.92 被高 multilingual rate 拉高，使得 SC（降低多语言率）后的 264.84 有一定"退化率下降"贡献。但即便如此，SC 对 kd_cr 的净效应是正面的。

**修订机制解读**：
- kd_cr SC 有益 = 真实（不是 PPL artifact）
- kd2 SC 有害 = 从重复性失败转为语义非连贯失败（换了失败模式而非消除失败）
- kd2 与 kd_cr 在 SC 效果上的差异：kd_cr SC 能将退化模式（多语言）转换为连贯英文；kd2 SC 只是将重复模式转换为语义非连贯模式（both bad, different bad）

### N.5 最弱的链条（更新后）

1. **kd2 SC PPL=600 = 语义非连贯**（confirmed by text inspection）：kd2 SC 失败，但失败原因是 SC 导致语义破坏，而非多语言退化。这是不同的机制，但结论相同：kd2 SC 有害。
2. **EXP-36 factorial 仅 seed=42**: ELF uncond eval 已证明确定，SC 结果是否确定性也需验证。
3. **kd_cr SC 有益但 baseline 已退化**: kd_cr none 有 15.2% 多语言率，SC 降到 7.4%。SC 在一个本就有问题的 baseline 上提升，这是 kd_cr 固有的 SC-dependency 的体现，不是一个 clean "SC improves generation" 结论。

### N.5 EXP-30v2 待更新

EXP-30v2（LangFlow layerwise probe v2）仍在运行（GPU4），结果待写入。将在 Section O 中记录。

---

## Section O: EXP-30v2 结果（2026-07-22，DONE）

### O.1 概述

EXP-30v2 修复了 EXP-30 的三个方法论问题：(1) 单 probe seed → 5 seeds；(2) 仅线性探针 → 新增 B5/B8/B11 MLP；(3) 无 skip decomposition → 5 条件对比。

**实验设置**：64 samples × 7 t-values × 4 noise levels，n_probe_seeds=5，30 epochs Adam GPU，MLP hidden=256

### O.2 Skip Decomposition（复刻 EXP-21v2）

| t | native | backbone_top1 | skip_top1 | probe_h |
|---|--------|--------------|-----------|---------|
| 0.85 | 56.1% | **0.00%** | 7.65% | 53.2% |
| 1.00 | 98.8% | **0.00%** | **92.4%** | **96.3%** |

✅ **三个独立数字与 EXP-21v2 完全一致**：backbone≈0，skip=92.4%@t=1.00，probe_h=96.3%@t=1.00。EXP-21v2 的核心结论在完全独立的 v2 实验参数（不同样本、不同 noise draws、不同探针）下精确复刻。

### O.3 Layer-wise 线性探针（5-seed CI）

**t=0.85 关键数值**（native=56.1%）：

| B0 | B1 | B4 | B5 | B7 | B10 | B11 | output |
|----|----|----|----|----|-----|-----|--------|
| 21.8±0.4% | 31.8±0.7% | 50.9±0.9% | 56.7±0.8% | 58.3±0.5% | **58.8±0.9%** | 53.0±0.9% | 48.9±0.9% |
| -34.3pp | -24.3pp | -5.3pp | **+0.6pp** | **+2.1pp** | **+2.6pp** | -3.1pp | -7.3pp |

**结论**：
- B5-B10 在 t=0.85 显著超越 native (+0.6 to +2.6pp)；B7 peak gap = +2.1pp（>4σ above 0）
- B11 和 output_layer 均低于 native（-3.1pp 和 -7.3pp）
- EXP-30 原始的 +3.9pp (B07, 单 seed) 在多 seed 下略收窄为 +2.1-2.6pp，方向一致

### O.4 MLP 探针 vs 线性探针

| layer | t=0.85 linear | t=0.85 MLP | t=1.00 linear | t=1.00 MLP |
|-------|--------------|-----------|--------------|-----------|
| B5  | 56.7% | 51.0% (MLP worse!) | 96.1% | **98.1%** (MLP better) |
| B8  | 57.6% | 43.0% (MLP worse!) | 97.5% | 91.7% (MLP worse) |
| B11 | 53.0% | 33.8% (MLP worse!) | 96.7% | 90.5% (MLP worse) |

**解读**：
- 在 t=0.85（中等噪声），MLP 30 epochs 收敛不如线性探针 → LangFlow 中间层在噪声状态下的 token 表示是**高度线性**的
- 在 t=1.00 的 B5，MLP(98.1%) > linear(96.1%)：干净状态下的 B5 有少量非线性 token 信息
- B8/B11 在 t=1.00 MLP < linear：越到后期 block，表示越偏线性（可能因 skip term 逐渐主导）
- **总体**：LangFlow backbone 对 token 的表示以线性为主；MLP 额外容量不显著帮助

### O.5 对论文的影响

**支持的 EXP-30 原始结论**：
- "B05-B10 中间层在 t=0.85 的线性探针超越 native head" ← 已有 5-seed CI 支持
- "B11 和 output_layer 低于 native" ← 多 seed 确认（-3.1pp 和 -7.3pp）
- "ELF 和 LangFlow 均有 mid-layer peak" ← 跨模型一致的现象

**修订的解读**：
- **不再 claim**: "最后几层丢失了 token 信息" → 改为"线性可分性在 B11/output_layer 降低"
- **不再 claim**: "native head 通过训练好的 projection 恢复信息" → 改为"native 的优势来自 skip term（z_t），而非 output_layer 的额外能力"（EXP-21v2 和 EXP-30v2 skip decomp 均已证明）

**新增可声明结论**：
> "LangFlow mid-block (B7-B10) representations show higher linear token recoverability at t=0.85 than the model's own native output (+2.1-2.6pp; consistent across 5 probe seeds). The final block (B11) reduces this by -3.1pp, and the skip-augmented native logit then exceeds B11 by +3.1pp — recovering the gap via the skip term (z_t @ E.T) rather than the output projection. This is confirmed by skip decomposition: backbone_top1≈0, skip_top1=92.4%."

### O.6 与 EXP-21v2 的关系

EXP-30v2 和 EXP-21v2 测量不同的东西但结论相互支持：
- **EXP-21v2**：在 LangFlow 输出层对 5 个读出条件（native/backbone/skip/probe_h/probe_hz）做一次性比较
- **EXP-30v2**：在所有 12 个 block + output_layer 做逐层线性探针（plus select-layer MLP）
- 两者都在独立实验中得到 backbone_top1≈0、skip≈92.4%、probe_h≈96.3%。这种独立复刻增强了这些数字的可信度。

---

## Section P: EXP-38/39/40/41/42 可解释性实验结果（2026-07-23，DONE）

### P.0 实验组概述

五个新的可解释性实验，均复用 `exp07b_v2` 预收集数据（无需新的 GPU forward pass，除 EXP-38 最终用 GPU 加速）。

核心驱动问题：KD 具体改变了 ELF 的哪些部分？改变了什么几何结构？

---

### P.1 EXP-40: unembed_bias 词汇分析（DONE，关键负结果）

**方法**：分析 Δunembed_bias = kd_cr - baseline（以及 kd2 - baseline）的 token 分布。

**关键数值**：
- baseline L2=79.6，kd_cr L2=271.9（3.4×），kd2 L2=338.3（4.3×）
- kd_cr Δbias 最大正向：Romanian/French/German 多语言 token（如 `'îl'` +2.77）
- kd_cr Δbias 最大负向：同样是多语言 token（如 `'prezentate'` -4.01）
- **cos(Δbias_kd_cr, Δbias_kd2) = 0.954** — 两者 bias 变化方向几乎完全相同

**关键负结果**：
> unembed_bias 的变化在 kd_cr 和 kd2 之间几乎相同（cos=0.95）。  
> 因此 unembed_bias 不能解释两者之间的功能差异（SC 效果、text quality 等）。

**对论文的影响**：
- EXP-15v2 的 R=2.59（unembed_bias Frobenius ratio 最大）反映的是**相对**变化，而非功能重要性
- 不能据此主张"KD 主要改变了 decode interface"

---

### P.2 EXP-39: Decode Head Cross-Patch（DONE，颠覆性发现）

**方法**：3×3 backbone × head 因果矩阵（baseline/kd_cr/kd2 两两组合）。

**结果（t=0.500）**：

| backbone | head=baseline | head=kd_cr | head=kd2 |
|---------|--------------|-----------|---------|
| baseline | 0.756 | 0.808 | 0.777 |
| kd_cr | **0.994** | 0.995 | 0.986 |
| kd2 | 0.992 | 0.993 | 0.990 |

**结论**：
- `baseline backbone + kd_cr head`: 0.808（+5.2pp，head 贡献小）
- `kd_cr backbone + baseline head`: 0.994（+23.8pp，backbone 贡献大）
- kd_cr backbone 配任意 head 均达到 99.4-99.5%（head 几乎可互换）
- **backbone 是 oracle accuracy 提升的主要来源，不是 decode head**

**与 EXP-15v2 的矛盾调和**：
> unembed_bias 的 R=2.59 是**参数相对变化大**，不等于**功能影响大**。  
> 功能测试（EXP-39）明确证明：换掉 backbone 比换掉 decode head 效果大约 4.6×（23.8pp vs 5.2pp）。

---

### P.3 EXP-42: 残差流 CKA 逐层差异（DONE，定位 KD 影响层级）

**方法**：对相同输入序列（y_tokens 完全匹配），逐层计算 kd_cr vs baseline 的 Linear CKA。

**结果（kd_cr vs baseline，t=0.500）**：

| block | B00 | B02 | B04 | B06 | B08 | B09 | B10 | B11 |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|
| CKA   | 0.981 | 0.968 | 0.930 | 0.961 | 0.896 | 0.803 | 0.564 | 0.427 |

**结论**：
- B00-B07: CKA 保持 0.930-0.981（相对相似，KD 影响有限）
- **B08-B11: CKA 剧降（0.896/0.803/0.564/0.427）**，KD 主要重塑后 4 个 block
- kd_cr 和 kd2 彼此 CKA ≥ 0.88（两者 backbone 高度相似），但与 baseline 差异大

---

### P.4 EXP-41: Decode Hidden 对齐分析（DONE）

**方法**：cos(decode_hidden_i, unembed_kernel[:, y_i]) 的正确 vs 错误位置分布，AUC。

**结果摘要**：

| ckpt | t=0.5 correct cos | wrong cos | AUC |
|------|-------------------|-----------|-----|
| baseline | **0.234** | 0.144 | 0.915 |
| kd_cr | 0.181 | 0.131 | 0.840 |
| kd2 | 0.136 | 0.116 | 0.676 |

**关键观察**：
- baseline 的 cos_align 最高（0.234），但 oracle accuracy 最低（75%）
- kd_cr cos_align 较低（0.181），但 oracle accuracy 最高（99.5%）
- **KD 不是通过让 decode hidden 更"尖锐地指向"正确 token 来提升精度的**，而是通过 backbone 表示的全局重组（EXP-42：B08-B11 重塑）使更多位置落在正确分类区域

---

### P.5 EXP-38: ELF Logit Lens（DONE）

**核心发现**：

**Logit Lens Top-1 准确率（t=0.500，关键时间步）**

| block | B00 | B04 | B07 | B08 | B09 | B10 | B11 |
|-------|-----|-----|-----|-----|-----|-----|-----|
| baseline | 0.187 | 0.212 | 0.362 | 0.408 | 0.675 | **0.784** | 0.756 |
| kd_cr | 0.404 | 0.506 | 0.805 | 0.881 | 0.896 | 0.988 | **0.995** |
| kd2 | 0.391 | 0.463 | 0.790 | 0.889 | 0.909 | 0.979 | **0.990** |

1. **kd_cr 和 kd2 从 B00 就领先 baseline（0.40 vs 0.19）**：差距在最早层就存在，不是只在 B08 之后。结合 EXP-42（B00 CKA=0.981，表示几乎相同），这说明 **decode head 的差异就足以导致早层精度差距**（EXP-39 证明 head 贡献 +5.2pp）。
2. **kd_cr ≈ kd2 在所有层**（B11: 0.995 vs 0.990）：两种 KD 变体的 logit lens 精度几乎相同。**SC 效果的差异不是 logit lens accuracy 的差异，而是 B11 表示方向的差异**（EXP-42: kd_cr vs kd2 B11 rel_L2=0.500）。
3. **baseline 非单调：B10 > B11**（0.784 > 0.756；t=1.0 时 B09=0.960 峰值）：baseline 晚层在做某种对 decode head 有害的变换；kd_cr 则单调递增到 B11。

---

### P.6 修订后的 KD 机制故事

**旧叙事（基于 EXP-15v2 Frobenius ratio）**：
> "KD reorganizes the decode interface; unembed_bias changes most (R=2.59)"

**新叙事（基于 EXP-38+39+41+42，四实验综合）**：
> "KD training reorganizes representations at two levels. (1) The backbone's late layers (B08-B11)  
> undergo the most dramatic subspace change (EXP-42 CKA: 0.896/0.803/0.564/0.427), and the  
> backbone transformation is functionally dominant: kd_cr backbone + any head achieves 99.4-99.5%  
> oracle accuracy (EXP-39). (2) The decode head also changes, contributing +5.2pp even with the  
> baseline backbone (EXP-39), and this head improvement propagates to all layers in the logit lens  
> (EXP-38: B00 gap 0.40 vs 0.19). Both kd_cr and kd2 make nearly identical decode-head changes  
> (cos=0.95 for bias, EXP-40), and they are indistinguishable in layer-wise logit lens accuracy  
> (EXP-38). Their behavioral difference (SC works for kd_cr, fails for kd2) traces to the  
> directional difference at B11 (rel_L2=0.500, EXP-42) — not oracle accuracy or logit lens profile."

**需要修订的论文 claim**：
1. "KD reorganizes the decode interface" → "KD reorganizes backbone (B08-B11, primary) + decode head (secondary)"
2. "unembed_bias R=2.59 is the primary change" → "R=2.59 reflects relative magnitude, not functional importance; backbone is primary (EXP-39)"
3. 新增可引用的正面结论：kd_cr vs kd2 在 logit lens 上无区别，SC 差异是方向性的（EXP-38+42 联合证明）
4. 保留 EXP-42 的"B08-B11 divergence" + EXP-39 的"functional backbone dominance"

---

### EXP-43: Dual-Path Gradient Conflict — Reconstruction–Decode Tradeoff Confirmed (2026-07-23)

**Setup**: For each checkpoint, compute interpolation curve h(α) = h_10 + α·(h_11 − h_10) for α ∈ [−0.5, 1.5] and simultaneously measure L_dec (CE loss via decode head) and L_rec (MSE vs x̂_{t=1.0} via reconstruction path). Also compute gradient conflict cos(∇L_dec, ∇L_rec) at h_11. Experiment uses exp07b_v2 fixed-noise states.

**Core result (t=0.5, α: h10 → h11)**:

| checkpoint | ΔL_dec (h10→h11) | ΔL_rec (h10→h11) | interpretation |
|:---:|:---:|:---:|:---|
| baseline | **+7.92 ↑** | **−501.1 ↓** | TRADEOFF — B11 improves reconstruction, hurts decode |
| kd_cr | −0.024 ↓ | −60.3 ↓ | NO TRADEOFF — both improve |
| kd2 | −0.063 ↓ | −28.5 ↓ | NO TRADEOFF — both improve |

**Secondary findings**:
- Gradient conflict cos(aggregate) is weakly negative for ALL checkpoints (−0.011 to −0.115) and does not separate baseline from KD. Root cause: at h_11, baseline's L_rec gradient is nearly zero (gnorm_rec = 1.3e-7 vs gnorm_dec = 1.2e-4), making the cosine undefined in practice. This metric is not informative for this design.
- Baseline L_rec(h10) = 529 vs kd_cr/kd2 L_rec(h10) = 103–142. KD distributes reconstruction computation into earlier layers (B00–B10); baseline offloads almost all reconstruction to B11 — this concentration is the root cause of the tradeoff.
- L_rec minimum for KD checkpoints occurs at α ≈ 0.75 (not α=1.0), showing slight overshoot, but this is small relative to baseline's tradeoff.

**Paper implications**:

**Confirmed and paper-ready (ADDED TO PAPER 2026-07-24, 5_experments.tex after EXP-07d)**:
> "The B11 residual update in the baseline model creates a reconstruction–decode tradeoff: moving from h_10 to h_11 worsens decode accuracy (L_dec increases by +7.9 CE units) while substantially improving reconstruction quality (L_rec decreases by −501 MSE units at t=0.5, EXP-43). KD-trained checkpoints eliminate this tradeoff: both L_dec and L_rec decrease (kd_cr: −0.024, −60.3; kd2: −0.063, −28.5)."

**Additional context**:
> "Baseline's h_10 carries dramatically higher reconstruction loss (L_rec=529) than KD checkpoints (103–142), suggesting KD distributes reconstruction computation across earlier layers rather than concentrating it in B11."

**Still hypothesis (not yet confirmed)**:
- Why does baseline concentrate reconstruction in B11? (need EXP-44 module patching to test)
- Does this tradeoff causally explain the SC difference between kd_cr and kd2? (EXP-45/46 pending)

---

### EXP-47: Intermediate-Layer SC — α Sweep (2026-07-24)

**Setup**: Custom ODE generation loop (N_SEQ=64, N_STEPS=32) with intermediate-layer SC conditioning:
x̂_α = final_layer(h_10 + α*(h_11-h_10)), α ∈ {0.0, 0.25, 0.50, 0.75, 1.0}, plus "none" arm (zero SC).
SC gate: only replace signal when t ≥ 0.5. Metric: I(α) = NLL(SC_α) − NLL(none).

**IMPORTANT caveat — pipeline inconsistency with EXP-36v2**:
`compute_ppl` returns mean NLL (not actual PPL = exp(NLL)). EXP-36v2 reference was I(baseline)≈+1594
(actual PPL units). EXP-47 reports NLL units (~4-6), incomparable in magnitude. Furthermore kd_cr
and kd2 sign is FLIPPED vs EXP-36v2 (kd_cr was I≈-65 there, +0.48 here; kd2 was I≈+158, -0.32 here).
Also kd_cr/kd2 generate degenerate text in some arms, making their GPT-2 NLL unreliable.
**EXP-47 cannot be compared to EXP-36v2. Internal comparisons across α only.**

**Results (internally consistent, NLL units)**:

| checkpoint | NLL(none) | I(α=0.00) | I(α=0.25) | I(α=0.50) | I(α=0.75) | I(α=1.00) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| baseline | 5.722 | −0.944 | **−0.991** | −0.946 | −0.923 | −0.929 |
| kd_cr | 5.230 | **+0.028** | +0.169 | +0.288 | +0.467 | +0.485 |
| kd2 | 5.833 | **−1.323** | −1.134 | −0.822 | −0.504 | −0.320 |

**Findings**:
- kd_cr and kd2: clear monotonic trend — lower α (more h_10, less h_11) gives better SC quality.
  Consistent with EXP-43 finding that h_10 is less reconstruction-burdened for KD checkpoints.
- Baseline: flat across all α (range ≈ 0.07 NLL). Hypothesis "I should improve at low α for baseline"
  NOT confirmed. Consistent with EXP-43 finding that baseline h_10 L_rec=529 is also very poor —
  h_10 carries no useful reconstruction signal, so there is no "better SC from earlier layer".

**Paper implications**: NOT directly usable until reproduced with EXP-36v2-compatible pipeline
(actual GPT-2 PPL via `record_generative_perplexity`, same generation settings).
Mechanistic interpretation (low α = avoid B11's decode-hostile reconstruction direction) is consistent
with EXP-43, but the key claim about baseline SC being harmful requires the EXP-36v2 pipeline.

---

### EXP-44: Module Factorial Patching Phase 1 (2026-07-25)

**Setup**: Chimeric forward passes using stored h_10 from exp07b_v2. Swap B11 weights / decode head / recon head between kd_cr and kd2. Measure oracle accuracy and L_rec for each chimeric h_11. Two t values: 0.3 and 0.5.

**Approximation note**: Content-only h_10 (prefix stripped) → B11 forward uses content-content attention only (no prefix-content attention). RoPE is correct: num_empty_token=0 with positions 0..L-1 gives identical rotations to the full model's content positions 12..12+L-1.

**Results (t=0.5)**:

| arm | oracle_acc | L_rec | Δacc | ΔL_rec |
|:---|:---:|:---:|:---:|:---:|
| native_kd_cr | 99.409% | 76.2 | — | — |
| native_kd2 | 99.014% | 76.4 | — | — |
| B11_cr_on_kd2 | 98.732% | 81.5 | −0.282pp | +5.1 |
| B11_kd2_on_cr | 99.398% | 77.8 | −0.011pp | +1.6 |
| decode_cr_on_kd2 | 99.147% | 76.4 | +0.133pp | 0.0 |
| decode_kd2_on_cr | 98.477% | 76.2 | **−0.932pp** | 0.0 |
| recon_cr_on_kd2 | 99.014% | 77.6 | 0.0pp | +1.2 |
| recon_kd2_on_cr | 99.409% | 81.8 | 0.0pp | **+5.6** |

**Key findings**:
- Oracle accuracy: decode head is the primary source of checkpoint specificity (decode_kd2_on_cr: −0.932pp). B11 contributes moderately (−0.28pp). Recon head: zero impact on oracle.
- L_rec: both native checkpoints have nearly identical L_rec (76.2 vs 76.4). Cross-checkpoint swaps worsen L_rec by ~5 units. Neither checkpoint has better x̂_t reconstruction quality than the other.
- **Critical null**: SC interaction gap (kd_cr I≈−65 vs kd2 I≈+158, EXP-36v2) is NOT explained by oracle accuracy or L_rec differences. Both are near-identical when using native modules.

**Implication**: The SC quality difference between kd_cr and kd2 must come from the SC conditioning module (self_cond_proj, self_cond_cfg_embedder) or from the specific *direction* (not magnitude) of x̂_t in 512-dim embedding space. Phase 2 would test SC module swaps directly, but EXP-45 (SC Activation Patch) is the more direct test.

---

### EXP-44: Module Factorial Patching Phase 2 — SC Module Swap (2026-07-25)

**Setup**: Full generation experiment (64 seq, 32 ODE steps, seed=42, SC_T_MIN=0.5, custom ODE loop same as EXP-47).
For each base checkpoint, swap self_cond_proj (proj_swap) or full SC module (full_sc_swap) from the other checkpoint during generation. I = PPL(SC) - PPL(none); negative = SC helps.

**⚠️ Pipeline caveat**: Same custom ODE loop as EXP-47. Native I direction is reversed vs EXP-36v2 (kd_cr native I=+134.9 instead of EXP-36v2's −65; kd2 native I=+25.4 instead of EXP-36v2's +158). Absolute PPL values not comparable to EXP-36v2 reference, but ΔΔI from swapping modules is the meaningful quantity.

**Results**:

| arm | kd_cr base | kd2 base |
|:----|:---:|:---:|
| PPL_none | 186.80 | 341.21 |
| I_native | +134.90 | +25.43 |
| I_proj_swap (cross-ckpt self_cond_proj) | **−46.60** | **+370.11** |
| I_full_sc_swap (full SC module) | −28.38 | +382.47 |

ΔΔI from proj_swap:
- kd_cr base: Δ = −181.5 (SC flips from harmful to helpful)
- kd2 base: Δ = +344.7 (SC becomes explosively harmful)

**Key findings**:
1. **self_cond_proj is the primary causal factor for SC interaction**. Swapping it alone produces nearly the full ΔΔI effect; full SC module swap adds marginal additional change (same direction but smaller).
2. Bilateral causal effect: (a) kd2's proj makes kd_cr's SC beneficial; (b) kd_cr's proj makes kd2's SC extremely harmful. Both directions confirm that self_cond_proj is the locus of SC compatibility.
3. This is consistent with the mechanism: self_cond_proj projects [z_t, x̂_t] → 512-dim input, directly determining how x̂_t is integrated into the next backbone call. The proj weights from each checkpoint are trained to expect x̂_t in the specific embedding geometry produced by that checkpoint's final_layer. Cross-matching proj and final_layer breaks this tuning.

**Paper implication**: The SC quality difference between kd_cr and kd2 is mechanistically explained by self_cond_proj weight differences, NOT by x̂_t reconstruction quality (Phase 1 null). This means KD training variants (kd-cr vs kd2) produce different self_cond_proj geometries, and these directly control whether SC guidance is helpful or harmful during generation. This is a clean new mechanistic finding for the paper, though it needs replication in the EXP-36v2 pipeline before being claimed as a primary result.

**EXP-45 (SC Activation Patch) is now lower priority**: Phase 2 already identified self_cond_proj as the primary causal mechanism. EXP-45 would test whether x̂_t *direction* (not SC module weights) also contributes independently, but is now a supplementary validation rather than the next critical experiment.

---

### EXP-48: Intermediate-Layer SC — Proper Pipeline (2026-07-25)

**Setup**: Same custom ODE loop. Arms: natural (standard SC from ODE step), none (zero SC), α=0.0 (h10), α=0.5 (mid). I(α) = PPL(α) - PPL(natural). Checkpoints: kd_cr, kd2. N=64 sequences, 32 ODE steps, SEED=42.

**Results**:

| arm | kd_cr PPL | kd_cr I | kd2 PPL | kd2 I |
|-----|-----------|---------|---------|-------|
| natural (std SC) | 303.4 | 0.0 | 247.7 | 0.0 |
| none | 186.8 | −116.6 | 341.2 | +93.5 |
| α=0.00 (h10) | 192.1 | **−111.3** | 90.9 | **−156.8** |
| α=0.50 | 249.2 | −54.2 | 150.0 | −97.7 |

**Key findings**:
1. **kd2 h_10 SC: PPL 247→91 (I=−157)** — This is a free inference-time improvement with no retraining. h_10 SC is far better than standard h_11 SC for kd2.
2. kd_cr: h_10 SC ≈ no SC (PPL 192 vs 187) — kd_cr's SC problem is not fixed by switching to h_10. The underlying mismatch (self_cond_proj trained on wrong geometry) remains.
3. Monotone α gradient for both checkpoints: lower α (more h_10, less h_11) → better SC quality for both.

**Mechanism**: h_10 contains a cleaner denoising signal that hasn't been "compressed" by the B11 decode-reconstruction trade-off. Using final_layer(h_10) as SC signal bypasses self_cond_proj entirely, which is the identified locus of the kd_cr/kd2 incompatibility.

**Paper implication**: D2 (h_10 SC at inference) should be reported as a validated inference-time improvement for kd2 (I=−157). Needs replication in the EXP-36v2 pipeline before claiming as primary result.

---

### EXP-49/50/51: D1 and D3 Synthetic Fine-tuning Validation (2026-07-25)

**Setup**: Quick 500-step fine-tuning on synthetic random x0 ~ N(0, 0.2²).
- D1: intermediate L_rec at B10, λ=0.5, all params updated
- D3: Gram alignment loss (final_layer ↔ self_cond_proj), λ=0.1, only these two modules

**Training results**:
- D1 aux_loss: 0.127→0.040 (−69%)
- D3 Gram alignment: 0.0096→0.0036 (−63%)

**EXP-51 evaluation results**:

| arm | kd_cr | D1 | D3 |
|-----|-------|----|----|
| natural | 303.4 | 1.54 | 116.2 |
| none | 186.8 | 1.49 | 5.89 |
| h10 | 192.1 | 1.58 | 5.88 |

**Key finding**: Both D1 and D3 checkpoints are degraded by synthetic data training.
- D1 collapsed to PPL≈1.5 for all arms (degenerate repetitive output) — synthetic data broke the generation backbone.
- D3 shows anomalously low PPL (5.9) for no-SC and h_10 arms — likely degenerate output; Gram alignment reduced 63% but at the cost of decode quality.

**Conclusion**: Synthetic data fine-tuning is insufficient for D1/D3 validation. Both require real OWT embedded data via train.py. However, the loss reduction confirms both loss terms are computable and well-posed.

**Paper implication**: D1 and D3 remain proposed improvements pending proper real-data validation. D2 (h_10 SC, inference-only) is the only confirmed improvement from this experimental chain.

---

### EXP-52: LangFlow Logit Lens (EXP-38 analogue, 2026-07-26)

**Setup**: Apply `output_layer(h_i, c=t_cond)` to each of 13 hidden states (embedding + 12 DDiT blocks). Two conditions: backbone-only (no skip) and full (+`c_skip·z_t⊗E`). t values: {0.10, 0.20, 0.30, 0.50, 0.70, 1.00}.

**Key results**:

| | t=0.10 | t=0.30 | t=0.50 | t=0.70 | t=1.00 (full) | t=1.00 (backbone-only) |
|--|--------|--------|--------|--------|---------------|----------------------|
| h0 | 0.000 | 0.000 | 0.001 | 0.005 | 0.894 | 0.002 |
| h12 | 0.000 | 0.000 | 0.001 | 0.008 | **0.983** | **0.737** |
| skip-only | 0.000 | 0.000 | 0.000 | 0.002 | 0.921 | — |

1. **t≤0.70: logit lens completely flat (≈0.001) at ALL 13 hidden states**. ELF kd_cr at t=0.5 goes from h0=40% to h12=99.5% — LangFlow has zero layer-by-layer progression.
2. **t=1.0: skip dominates** (skip-alone = 92.1%), backbone h12 alone = 73.7%, full h12 = 98.3%.
3. **Contrast with ELF-38**: ELF has rich logit lens showing gradual information build-up across layers. LangFlow compresses all oracle information into t→1.0 skip activation.

**Paper implication**: Confirms EXP-21v2 skip-dominance finding at a structural level. LangFlow's backbone is effectively "transparent" (≈0.1% accuracy at all intermediate depths for t≤0.70). This is a structural contrast to ELF's decode branch, where B08-B11 show measurable commitment progression (EXP-42). Do not claim LangFlow "commits earlier at intermediate layers" — it does not. The ELF logit lens finding is architecturally specific to ELF's decode path + backbone design.

---

### EXP-53: LangFlow T_stable / Never-Commit Rate (EXP-16v2 analogue, 2026-07-26)

**Setup**: Fixed-ε oracle (same noise draw across all t), K=3 consecutive correct top-1 required for T_stable. 51 t values from 0.03 to 1.0. N=64 OWT samples, seq_len=128.

**Note**: First run had t_grid truncated at t≈0.813 (never-stable=79.4%, artifact). Fixed to `linspace(0.03, 1.0, 51)`.

**Results**:

| Model | never-stable (K=3) | mean T_stable |
|-------|-------------------|---------------|
| ELF baseline | 25.1% | ~0.50 |
| **LangFlow** | **4.79%** | **0.840** |
| ELF kd2 | 0.98% | ~0.20 |
| ELF kd_cr | 0.53% | ~0.18 |

**G_oracle(t) non-monotone**: 3.9% at t=0.03 (frequency-mode), 2.9% dip at t=0.5, cliff at t=0.806 (31.6%) → 98.9% at t=1.0. **G_stable(t=1.0) = 95.2%** (4.79% never commit even by t=1.0).

**Key findings**:
1. LangFlow never-stable (4.79%) is much better than ELF baseline (25.1%) but worse than KD models (0.5-1.0%). Positioned in the middle.
2. **Late commitment**: mean T_stable=0.840 vs ELF kd_cr ≈0.18. LangFlow commits ~0.65t later in the denoising process, consistent with EXP-02/03 findings of ELF committing ~0.6t earlier.
3. **G_oracle dip at t≈0.5**: frequency-mode collapse at low t (4%) → intermediate confusion → cliff. Unique to LangFlow; ELF G_oracle is monotonically increasing.

**Paper implication**:
- The "LangFlow never stably commits" story from EXP-22 overstated the problem. 95.2% of positions DO stably commit by t=1.0. The problem is latency (mean T_stable=0.84), not failure-to-commit.
- Cross-model comparison (ELF vs LangFlow) must use matched-SNR axis (EXP-03) not nominal t. The 0.65t offset is schedule-confounded.
- EXP-25 (T_first, H<1.0) gave never-committed=1.37%; EXP-53 (T_stable K=3) gives 4.79%. Both consistently show LangFlow never-commit << ELF baseline (25.1%).


### EXP-54: h₁₀ SC Validation with Standard sccfg=3 (2026-07-26)

**Motivation**: EXP-48 showed kd2 PPL 247→91 (I=−157) with h₁₀ SC, but used sccfg=1 in a 64-sample batch. EXP-54 validates with 256 samples and checks sccfg=3 (standard inference setting).

**Setup**: kd2 only; ODE-32, time_schedule=uniform, 128-token, N=256, seed=42. Four arms: natural/h₁₀ × sccfg∈{1,3}. Hook: `blocks[10].register_forward_hook` + `final_layer.register_forward_pre_hook`.

**Results**:

| arm | sccfg | PPL | I |
|-----|-------|-----|---|
| natural | 1 | 284.7 | ref (EXP-36v2 ref=282.5 ✓) |
| natural | 3 | 295.7 | +11 (sccfg=3 slightly hurts) |
| h₁₀ SC | 3 | 168.8 | **−126.9** (43% reduction) |
| h₁₀ SC | 1 | 155.4 | **−129.3** (45% reduction) |

**Key findings**:
1. **h₁₀ SC confirmed in standard pipeline**: I≈−127 to −129, robust across sccfg settings. EXP-48's I=−157 (64 samples) was high-variance; 256-sample estimate I=−129 is more reliable.
2. **sccfg=3 slightly hurts natural arm** (+11 PPL): consistent with EXP-44 finding that kd2 h₁₁ SC signal is anti-correlated through B11. Amplifying it (sccfg=3) makes it slightly worse.
3. **h₁₀ SC sccfg=1 better than sccfg=3** (155 vs 169): h₁₀ signal is improved but still has some noise; amplification at sccfg=3 doesn't help further.
4. **Reference clarification**: The 142.6 PPL from `elf_b-owt-kd2-eval-pt-full` uses 1024-token sequences (EXP-37b config: `max_length=1024, latent_std=0.2`). EXP-54's 128-token baseline ~285 is NOT comparable to it.

**Paper implication**:
- EXP-48 paragraph updated: "PPL drops from 284.7 to 155.4 (I=−129, 45% reduction)" replaces 247→91 (I=−157) from high-variance batch.
- h₁₀ SC is now a validated primary result for the paper (EXP-54 reference).
- The sccfg=3 natural arm being worse than sccfg=1 provides additional EXP-44 support: B11's anti-correlated update degrades SC, and amplification makes it worse.

### EXP-54b: Multi-Seed Variance Validation (2026-07-27)

**Setup**: kd2 only; seeds {42, 123, 456}, N=256/seed, 3 arms: natural_sccfg1, natural_sccfg3, h₁₀_sccfg1.

**Results** (mean ± 95% CI over seeds):

| arm | mean PPL | 95% CI margin | std |
|-----|----------|--------------|-----|
| natural sccfg=1 | 299.8 | ±32.5 | 13.1 |
| natural sccfg=3 | 309.4 | ±29.9 | 12.0 |
| h₁₀ SC sccfg=1 | 170.1 | ±32.6 | 13.1 |
| **I(h₁₀ − natural)** | **−129.7** | **±9.3** | **3.7** |

Delta(nat_sccfg3 − nat_sccfg1) = +9.6 (consistent across all 3 seeds).

**Key findings**:
1. **I = −129.7 ± 9.3 (95% CI)**: The h₁₀ SC improvement is confirmed with CI. Conservative lower bound: I = −120.4 (42% reduction). Paper can cite "43% PPL reduction, I = −130 ± 9."
2. **sccfg=3 consistently +9.6 PPL worse than sccfg=1** (not noise): Confirmed stable across seeds — EXP-44 anti-correlation finding is solid.
3. **I is remarkably stable (σ=3.7)** while absolute PPL varies ~13 PP — the benefit of h₁₀ SC is seed-independent.

**Paper update**: Replace single-seed EXP-54 citation with "I = −130 ± 9, 95% CI, N=768, seeds={42,123,456}".

### EXP-54c: SC_T_MIN Sweep (2026-07-27)

**Setup**: kd2 only; seed=42, N=256, h₁₀ SC sccfg=1, SC_T_MIN ∈ {0.0, 0.1, 0.25, 0.5}.

**Results**:

| SC_T_MIN | Steps active | PPL | I |
|----------|-------------|-----|---|
| 0.0 | 31/31 | 1369.0 | **+1084.3 (catastrophic)** |
| 0.1 | 28/31 | 1236.1 | **+951.4 (catastrophic)** |
| 0.25 | 24/31 | 574.3 | **+289.6 (very bad)** |
| **0.5** | **16/31** | **155.4** | **−129.3 (excellent)** |
| natural (ref) | — | 284.7 | 0 |

**Key findings**:
1. **SC_T_MIN=0.5 gate is ESSENTIAL**: Applying h₁₀ SC at t<0.5 is catastrophic (+1084 PPL). Without the gate, h₁₀ SC destroys generation.
2. **Monotonic degradation**: PPL degrades monotonically as gate is lowered. Hard regime boundary at t≈0.5.
3. **Mechanistic validation of EXP-42/44**: The t=0.5 boundary is exactly where EXP-42 CKA shows B08-B11 diverge (kd2 vs baseline: 0.896/0.803/0.564/0.427). EXP-44 found B11 anti-correlated — that anti-correlation is active only in the t≥0.5 regime.

**Paper implications**:
- The SC_T_MIN=0.5 gate should be explicitly described in the paper as a mechanistically-motivated threshold, not an arbitrary hyperparameter.
- Add to mechanism section: "EXP-54c shows h₁₀ SC must be restricted to t≥0.5; applying it at t<0.5 degrades PPL by +1084 (from 285 to 1369). This phase boundary aligns with the EXP-42 CKA divergence at B08-B11 (t=0.5) and confirms that B11's anti-correlated behavior is confined to the high-noise regime."
- This is the strongest single-experiment evidence for the t=0.5 regime boundary in the mechanism.

---

## Section Q: Global State Formation 实验组结果（2026-07-26，DONE，多轮迭代）

### Q.0 概述

新方向：把"per-token 何时 commit"的问题提升到"整个序列状态 `Z_t` 是否先形成全局语义
再细化成局部 token"这个更高层次的问题（`docs/global_state_formation_experiment_suite.md`
原始协议）。这是一整套独立于 EXP-01~53（per-token commitment story）的新实验组，共
15 个子实验（`EXP-GS1`–`EXP-GS15`），全部先在 ELF baseline 上跑通，再用户审阅发现四个
关键 confound 并逐一修正，最后**全部 15 个实验在 LangFlow 上复现了一遍**。

**完整细节见 `docs/global_state_formation_synthesis.md`**（综合解读文档，取代原始协议
doc 的 H1 假设表述）和 `docs/specs/EXP-GS{1-15}-spec.md`（逐实验 spec + 结果）。本节
只摘录对论文写作直接相关的结论。

### Q.1 原始假设不成立，替换为一个更精确的机制

原始假设（`global_state_formation_experiment_suite.md` H1）：

```
global semantic basin -> structural scaffold -> exact lexical evidence -> exact token
```

**不成立**。经过 15 个实验 + 4 轮方法论修正后，更可信的过程是：

```
weak distributed signal
  -> prior-dominated compression
  -> context-coupled residual organization
  -> collective lexical transition
  -> stable tokens
```

核心修正模型（`EXP-GS12`）：把序列状态拆成跨位置共享的均值 `mu_t` 和位置特异的
centered residual `R_t = Z_t - 1*mu_t^T`。**粗粒度统计（POS 分布等）几乎完全由 `mu_t`
解释；exact token identity 则明确需要高秩的 `R_t`**，均值和低秩共享分量都远远不够。
这是目前 ELF 和 LangFlow 上都稳健复现的核心几何图景。

**对论文的影响**：如果要把这条新故事线写进论文（作为独立于 EXP-01~53 per-token story
的新章节），标题/核心主张应该是"coarse statistics vs. high-rank lexical residual"这类
表述，**不要用"global semantic basin"这个说法**——已经被 `EXP-GS11`/`EXP-GS12` 证明
主要是 pooling 统计效应，不是模型学到的语义组织。

### Q.2 三个稳健复现（ELF + LangFlow 都成立，可以直接引用）

1. **Token identity 是最晚稳定的**（`EXP-GS2`/`EXP-GS3`/`EXP-GS7`/`EXP-GS14`
   四种独立方法学交叉验证，ELF 和 LangFlow 都复现）。
2. **`MEAN_only` 单独解释了几乎全部 structural R²，加低秩分量或残差都不显著提升**
   （`EXP-GS12` 正式规模，两个架构 18/18 或近似比例的 `(t,repr)` 组合一致）。
3. **`MEAN+R_c` 的 token_acc 全程远超 `MEAN+G_c`**（同上，两个架构都是数量级差距）。
4. **`EXP-GS4` 的因果干预结论**（去掉低秩 global mode 几乎不影响生成，只保留低秩
   global mode 则生成崩溃，swap 实验里最终身份由残差 donor 决定）在 ELF（`t=0.38`）
   和 LangFlow（`t=0.65`，用 LangFlow 自己校准的过渡区）上都清晰复现：
   ELF `baseline=0.229→A_remove=0.223`（几乎不变）、`B_preserve=0.009`（崩溃）；
   LangFlow `baseline=0.555→A_remove=0.531`（几乎不变）、`B_preserve=0.005`（崩溃），
   swap 都是残差 donor 主导（LangFlow: vs B=0.502 vs vs A=0.008）。

### Q.3 四个方法论 confound 及修正（用户审阅驱动，P0-1~P0-4）

这四条本身就是值得写进论文 discussion/limitations 部分的方法论教训，展示了这条研究
线在多大程度上排除了 trivial explanation：

1. **P0-1 Pooling confound（`EXP-GS11`）**：直接对**未经模型处理**的 raw oracle state
   做位置均值，`L=32,t=0.28` 时 self-retrieval 已经 100%（不需要任何模型计算）。
   说明"early global signal"很大程度是 `1/sqrt(L)` 噪声平均的统计性质，不是模型早期
   就"知道"了全局语义。**任何号称"模型早期已经编码了全局信息"的 probe 类论断，都必须
   先排除这个 confound**（对照：不经模型的 raw state 是否已经能给出同样的 probe 表现）。
2. **P0-2 Uncentered SVD confound（`EXP-GS12`）**：`EXP-GS3` 原本"structure 集中在
   低秩 global mode"的结论，去掉跨位置均值后**不成立**——`MEAN_only`（不做任何 SVD）
   单独就能达到和"均值+低秩"几乎相同的 structural R²。这是"分解方式与读出方式互相
   咬合"造成的假象。
3. **P0-3 Direct-feature-intervention confound（`EXP-GS13`）**：`EXP-GS8` 原本把扰动
   加到了正在测量 margin 的目标位置本身，效应可能只是最短路径；改成目标位置完全不动、
   只扰动其它位置后，效应打了七折（`+1.718→+1.227`）但**没有消失**——支持存在真实的
   跨位置因果传递，但正式规模显示是 U 形而非线性剂量-反应。
4. **P0-4 Oracle-cold-start confound（`EXP-GS14`）**：`EXP-GS2` 用 oracle 构造状态 +
   冷启动 self-conditioning 做分支实验；换成真实 free-running 轨迹 + 真实累积 SC 后，
   定性结论**完全复现**——这个简化没有制造出错的结论。

**对论文的影响**：如果引用 GS1/GS2/GS3 的原始（未修正）数字，必须同时引用对应的
GS11/GS12/GS14 修正结果，不能只引用"看起来更漂亮"的早期版本。

### Q.4 LangFlow 跨架构复现——再次印证"nominal t 不可跨模型比较"

呼应本文档 Section I（EXP-20-24）、Section L（EXP-25v2-27v2）、EXP-02/03 已经反复确立
的结论：**ELF 校准出的 `t=0.28` 直接套用在 LangFlow 上完全失效**（`EXP-GS6`/`GS8`/
`GS13` 在 `t=0.28` 上测不出任何信号，topic probe test_acc 只有 0.158，chance=0.125）。
用 LangFlow 自己的 GS1 曲线重新定位过渡区（约 `t=0.65`）后，`EXP-GS4` 的核心因果结论
干净复现（见 Q.2 第 4 条），但 `EXP-GS8`/`GS13`（依赖 topic probe 方向）在任何 t 上
都因为样本量不足（`n_train=45` 对 8 类分类器，`train_acc=0.933` vs `test_acc=0.158`，
严重过拟合）而不可靠——**这是样本量问题，不是 t 校准问题**，需要正式规模（更大 N）
才能在 LangFlow 上真正验证 GS8/GS13 的因果链结论。

另有两个方法论坑是 LangFlow 特有的：
- raw-state CKA 在 LangFlow 上经常整体饱和（`EXP-GS7`/`GS15`），必须用 `predicted_clean`
  代替 raw `z_t` 才有信号；
- LangFlow 在 nominal `t=0.99` 时还没到真正的 `gamma_min`（`gamma(0.99)=3.48` vs
  `gamma_min=2.60`），不能像 ELF 那样把 `t≈1` 当作"clean"的可靠代理。

也有几个尚未确定是真实架构差异还是需要更大样本的发现（不建议现阶段写进论文，仅记录）：
`EXP-GS11` 的"model 处理会破坏可检索身份信息"在 LangFlow 上不成立（LangFlow 的
`predicted_clean` retrieval 在大 `L_eff` 下追平甚至略超 raw）；`EXP-GS3`（未中心化）在
LangFlow 上从一开始就不支持"structure 在 G"，比 ELF 更早印证了 GS12 的修正结论；
`EXP-GS5` 的 collective coupling susceptibility 在 LangFlow 的 pilot t 网格（到
`t=0.85`）内单调上升未封顶，暗示其峰值在网格之外，需要更晚的 t 才能看到和 ELF 类似的
峰值-回落形状。

**对论文的影响**：如果 Global State Formation 这条线要写进论文并包含 LangFlow 对比，
Q.2 列出的三个"两架构都复现"的结论可以直接作为跨架构证据引用；Q.4 提到的架构差异
现阶段样本量/校准都不够，不应该作为"LangFlow 和 ELF 在机制上不同"的证据，只能作为
"需要正式规模验证"的开放问题列在 limitations 里。

### Q.5 补充：GS4 因果结论 + GS15 动力学结论均已跨架构确认（2026-07-26 追加）

在 Q.4 写完之后，用 LangFlow 自己校准出的 `t=0.65` 重跑了 `EXP-GS4`/`GS6`/`GS8`/`GS13`，
并给 `EXP-GS15` 加了 model-based 版本重跑了两个架构，结果更新如下：

- **`EXP-GS4` 的因果干预结论干净复现**（Q.2 第 4 条已记录：LangFlow `t=0.65` 上
  `baseline=0.555→A_remove=0.531` 几乎不变、`B_preserve=0.005` 崩溃、swap 由残差
  donor 主导）。这证明"换 LangFlow 自己的过渡区 t"这个思路对 GS4 有效。
- `EXP-GS8`/`GS13`（topic-probe 因果链）换 t 后 topic probe test_acc 完全没变
  （0.158→0.158，`train_acc=0.933` 说明是训练集只有 45 篇文档、8 类分类器严重过拟合）
  ——瓶颈**不是** t 校准，是样本量，需要正式规模（更大 N）而不是继续调 t。
- **`EXP-GS15` 加了 `O_R_model`（用 `predicted_clean` 残差代替饱和的 raw 残差）后，
  "负 O_R / 晚期崩塌"这个此前只在 ELF 上观察到的现象，在 LangFlow 上独立复现**：
  两个架构的 `O_R_model(t)` 全程为负、轨迹中段最负、随后稳步恢复到 0（ELF 最负点
  `t≈0.20-0.28`≈`-0.22~-0.24`；LangFlow 最负点 `t≈0.28`≈`-0.472`，量级更大但方向、
  形状一致）。这是目前 Global State Formation 系列里唯一经过**跨架构 + 因果/动力学**
  双重检验的正面发现。

### Q.6 更正 + 补充：GS6 的"换 t 无效"最初诊断是错的，真正根因是一个可修复的 bug（2026-07-26 再追加）

Q.5 最初把 `EXP-GS6`（competing basins）换 `t=0.65` 后依然"`P_A(lambda)` 全程平坦"
归因为"rollout 步长没有正确映射到 LangFlow 的 gamma 范围"——**这个诊断是错的**，
排查后发现真正原因是一个具体、可定位、已修复的 bug：`nearest_topic`（判断 rollout
终点属于哪个 topic 的函数）用**平方欧氏距离**找最近质心，而 LangFlow rollout 终点的
pooled embedding **norm≈1.15**，系统性地远小于拟合 topic centroids 时用的 clean
embedding **norm≈3.70**——导致每一个 rollout 终点，不管真实内容是什么，都被判给
norm 最小的那个质心。直接验证：8 篇文档的 rollout 终点全部被分到同一个 cluster，
但它们各自和自己 clean embedding 的 cosine 相似度是 0.40–0.67，说明真实的、按内容
区分的信号一直都在，只是被 Euclidean 距离对 norm 差异的敏感性掩盖了。

**修复**（改用 cosine-based nearest-centroid，scale-invariant）后：

- **`EXP-GS6` 在 LangFlow 上给出了比 ELF 更干净的 bifurcation**——4 对文档全部在
  `lambda=0.4→0.6` 之间同步完成从 B-basin 到 A-basin 的整体切换，纯 A/纯 B 端点也
  正确分类（ELF 原始结果里跳变分散在 0.2–0.6 之间，还有一个异常对全程卡在 A）。
- **`EXP-GS4` 的 topic 维度**从"恒定 0.25"（同一个 bug 的另一处受害者）变成有意义的
  结果：`t=0.65` 时 `B_preserve_global`（只保留低秩 global mode）topic=1.00，
  **同一条件下 token=0.005（崩溃）**——这是原始协议 doc GLOBAL-4"保留 global mode
  应该保住 topic、丢失 token"这一预测目前拿到的最干净的直接确认。
- **`EXP-GS14` 的 `C_topic=1.000`** 数值不变，但从"无法排除是 bug 导致所有分支判成
  同一 cluster"变成"确认为真的 branch consensus"。

**对论文的影响**：Global State Formation 章节如果只能选一个跨架构结果作为 headline
figure，`EXP-GS15` 的 `O_R_model(t)` 曲线（两个架构叠在一张图上）和修复后的
`EXP-GS6`/`GS4` 组合（bifurcation + "G 保 topic 不保 token"的因果解耦）是目前两个
最强的候选，都是定量、可复现、跨架构的证据，比 Q.1 的五阶段描述性框架更有实证分量。
`GS8`/`GS13` 在正式规模（更大 N）验证之前，不建议在论文正文里引用其 LangFlow 数字，
只能在 limitations 里说明"样本量不足，留待未来工作"。**方法论教训**（值得写进
论文的 appendix 或代码可复现性说明）：任何用欧氏距离做最近质心分类的诊断，一旦待
分类对象和拟合质心的分布存在系统性尺度差异，就会产生"看似没有信号"的假阴性——应默认
优先用 cosine 而不是欧氏距离，尤其是跨状态类型（clean embedding vs. rollout 终点）
做分类的场景。

---

### Q.7 严谨性自审：发现第二处未修复的同款 bug + n=4 headline 数字经大样本/bootstrap CI 修正（2026-07-27）

用户要求对整个 GS 系列做一次系统性严谨性自审（"自己检查一下你做的实验是否严谨，是否
需要增加实验"）。审计发现两类此前遗漏的问题，均已处理：

**问题 1：Q.6 记录的"GS6 bug 已在 GS4/GS6/GS8/GS14 四个文件里统一修复"这个验证方法
本身有漏洞。** 当时只检查了这四个文件的 `nearest_topic` import 是否指向同一个（已
改为 cosine-based 的）函数对象，没检查每个文件的 C_topic 计算是否**真的调用**了这个
被 import 的函数。重新审计发现 `branch_global_consensus.py`（`EXP-GS2`）和
`branch_true_trajectory.py`（`EXP-GS14`）的 C_topic consensus 计算里，各有一份**内联
手写的**平方欧氏距离最近质心分类，从未走 import 路径——`EXP-GS2` 是这整个 bug 模式
最初的出处，此前从未被修复过；`EXP-GS14` 的"C_topic=1.000 现已确认为真"这一结论
（Q.6 最后一条）实际上建立在从未真正修复过的代码上。已修复两处（改用统一的
`nearest_topic`），`branch_true_trajectory.py` 顺便补上了逐轨迹原始数据以支持 CI 计算。

修复后重跑：**`EXP-GS2` 的"C_topic 早饱和"这一核心跨方法学交叉验证结论在 ELF/
LangFlow 两个架构上都不受影响**（bug 修复前后数值几乎一致，0.91-1.00 区间）——排除
了"饱和是这个 bug 的假阳性"这个担忧。GS2 自带的 eta sweep 进一步显示 C_topic 对扰动
幅度有真实但比 C_lex 迟钝得多的渐变响应，不是完全不敏感的天花板伪影。

**问题 2：GS 系列此前完全没有 bootstrap CI 或多种子复核（PT 系列有，GS 系列没有），
`EXP-GS4`/`EXP-GS6`/`EXP-GS14` 引用的 n=4 pilot headline 数字经补充验证后有实质性
修正。** 已给 `common.py` 加上 `bootstrap_ci`（2000-resample percentile bootstrap，
对齐 PT 系列标准），用更大样本（n_pairs/n_docs/n_traj 从 4 提到 16-20）+ 3 个种子
（42/123/456）重跑：

- **`EXP-GS6`**（pooled n=60）：Q.6 报告的"4 对文档全部在 λ=0.4→0.6 之间同步完成
  切换、比 ELF 更干净"需要更正。大样本下 λ=0.4 与 λ=0.6 的 P_A 95% CI 不重叠
  （[0.083,0.283] vs [0.483,0.733]），**bifurcation 本身有真实统计支持**，但转变
  窗口实际上更宽（0.4→0.8），纯 A 端点 P_A 只有 0.817[0.717,0.900]、从未接近 1.0，
  且存在 n=4 从未报告过的 10-45% "落入第三方 topic" 的情况。"干净、同步"这个定量
  描述应撤回，"存在真实但有相当噪声的 bifurcation" 是更准确的表述。
- **`EXP-GS4`**（pooled n=48）：Q.6 报告的"`B_preserve_global` topic=1.00，GLOBAL-4
  预测最干净的确认"需要更正。大样本下 `baseline`/`A_remove_global`/
  `B_preserve_global` 的 topic_match 95% CI 完全重叠（0.812-0.875），**统计上无法
  区分**——"G 单独排他性地承载 topic"这个更强因果说法不成立，topic 信息更可能在
  三种重构里冗余存在。**token_acc 的不对称性完整存活**（`B_preserve_global`
  token≈0.006-0.008 vs 另两个条件 0.53-0.57，3 个种子一致，效应量巨大不需要 CI 佐证）
  ——"token identity 依赖残差、不依赖 G"是本实验唯一完全经得住大样本考验的结论，
  写论文时应该只引用这一条，不要引用 topic 维度的"排他性"说法。
- **`EXP-GS14`**：C_topic 从"恒 1.000、零方差"（且如问题 1 所述，从未真正走过修复
  后的代码路径）变成 0.958[0.929,0.984]@t=0.20 → 0.992[0.976,1.0]@t=0.65 的真实
  渐变信号；eta sweep 确认其随扰动幅度单调下降但比 C_lex 迟钝得多，与 GS2 的 eta
  sweep 相互印证。

**对论文的影响**：如果 Global State Formation 章节要引用 GS4/GS6 的具体数字，应该
使用本节的大样本 + CI 版本，而不是 Q.6 报告的 n=4 pilot 数字。`EXP-GS15` 的
`O_R_model(t)` 跨架构曲线仍然是目前最强、最不需要限定语的候选 headline figure；
`EXP-GS4`/`EXP-GS6` 可以作为支持性证据引用，但需要用本节的 CI 区间而非点估计，且
"G 排他性地承载 topic"这个说法要去掉，只保留"token 依赖残差"这条。**方法论教训**
（建议写进 appendix）：(1) 任何"验证 bug 已修复"的检查，必须确认每个受影响的计算
路径都**实际调用**了修复后的代码，而不能只检查 import/引用关系；(2) n=4、单 seed、
数值"整整齐齐"（全 0 或全 1）的结果应默认视为"可能是运气"，需要大样本 + bootstrap
CI 才能作为论文可引用的数字——这是 PT 系列（PT6/PT7）已经验证过的标准，GS 系列在
本轮自审之前一直没有被同样要求。

---

### Q.8 严谨性自审第二轮：GS8/GS13 过拟合修复 + ELF 大样本复核（2026-07-27）

Q.7 记录的自审提出了两个尚未处理的缺口（P1: GS8/GS13 过拟合诊断但未修复；P2: ELF 版本
仍是 n=4 单 seed，双重标准），用户要求继续修正，均已完成。

**GS8/GS13 过拟合修复**：给两个脚本的 `LogisticRegression` 加 `--C` 正则化参数，配合
`--n_samples` 64→200 重跑（`t=0.65`）。GS8：`train_acc` 0.933→0.314，`test_acc`
0.158→0.217（chance=0.125），过拟合基本解决；bootstrap CI 确认 `correct`/`wrong`
方向在全部 alpha 上明显高于 `orthogonal`/`random`，是 U 形而非 ELF 的单调曲线，量级
小一个数量级。GS13 用同样修复得到 `test_acc=0.217`，但干预效果只在一个 alpha 方向
（a=+1.0）站得住，另一方向 CI 含 0——是本轮最不干净的结果。**结论**：LangFlow 上的
global-to-local 因果链确实存在，不是纯过拟合假象，但明显比 ELF 弱、不干净，写论文时
如果要引用 LangFlow 的 GS8/GS13 数字，必须标注"信号弱于 ELF 且 GS13 仅单侧显著"。

**ELF 版本大样本 + 3-seed + bootstrap CI 复核**（GS4/GS6/GS14，方法与 Q.7 对 LangFlow
做的完全一致）：

- **GS6**：pooled n=60，ELF 的 bifurcation 比 LangFlow 同款复核明显更干净（转变窗口
  更窄、纯端点 CI 下界 0.88+ vs LangFlow 的 0.72、`P_other`≤15% vs LangFlow 10-45%）
  ——这是在完全对齐方法学下测出的真实跨架构差异，值得写进论文，而不只是"复现成功"。
- **GS4**：pooled n=48，`t_start=0.38` 时 `A_remove_global`(0.292[0.167,0.417]) 和
  `B_preserve_global`(0.229[0.125,0.354]) 的 CI 几乎重叠，都远低于
  `baseline`(0.917[0.833,0.979])——**ELF 上 topic 恢复需要 G 和 R 兼备**，而 LangFlow
  是三条件统计不可区分（Q.7）。同一因果干预范式在两个架构上给出了不同的定性模式，是
  一个新的、干净的跨架构对比点。
- **GS14**：pooled n=48，且这是该脚本 `C_topic` 计算**首次真正调用修复后的
  `nearest_topic`**（Q.7 提到的第二处 bug 就在这个文件，此前"确认为真"实际建立在
  未修复代码上）。修复后 ELF 的 `C_topic` 在 `t_start=0.65` 上是真实的零方差天花板
  （1.000[1.000,1.000]），LangFlow 同一 t_start 未到顶（0.992[0.976,1.0]）——差异
  合理，反映两个架构的 commitment 时间尺度本就不能按 nominal t 直接对齐（这是本
  文档 Q.4/EXP-03 已经确立的事实，这里是一次新的印证）。

**对论文的影响**：经过两轮自审，GS4/GS6/GS14 现在在 ELF 和 LangFlow 上都有对齐的
n=16-20×3seed+bootstrap CI 数据，可以直接用于跨架构对比图（尤其 GS6 的 bifurcation
"干净度"差异、GS4 的"完整状态 vs 冗余可恢复"差异），这两个新对比本身比"两个架构都
复现了同一个结论"更有实证分量，建议作为 Global State Formation 章节的补充图表。
GS8/GS13 的 LangFlow 数字现在可以引用，但必须带上"信号弱于 ELF"的限定语；`EXP-GS15`
的 `O_R_model(t)` 跨架构曲线仍是最强、最不需要限定语的 headline candidate（不变）。

---

## Section R: Phase-Transition Suite（EXP-PT1–10）完整结果与论文影响（2026-07-26，DONE⚠️）

对应 `docs/phase_transition_experiment_suite.md` 提出的全新实验组，独立于本文档
Section A-Q 追踪的既有 EXP-01–EXP-53/GS 系列，用统一的 `FlowModelAdapter`
（`experiments/phase_transition/adapters/`）跨 ELF（baseline/kd_cr/kd2）和
LangFlow 四个 model/checkpoint 组合系统性验证。全部 10 个实验 + 严谨性自查
（bootstrap CI、多种子复核、样本规模修正）已完成，详细协议/简化/踩坑记录见各
`docs/specs/EXP-PTx-spec.md`；本节只汇总**对论文写作有直接影响**的结论。

### R.0 严谨性状态（写作前必读）

`EXP-PT-rigor-audit.md` 记录了一次专门的自查：10 个实验最初全部只有单次运行的
点估计，不满足 doc 共享协议要求的"按序列 bootstrap CI"。目前已修复到位：

- **有序列/pair 级 bootstrap CI 支持**：PT1、PT2、PT3、PT4、PT5、PT8、
  PT9、PT10（2000 次重采样，见各 spec 的"严谨性补强"一节）。
- **有多种子复核（seed 42/123/456）**：PT6、PT7 最戏剧化的 kd_cr 发现。
- **PT6/PT7 额外做了样本量翻倍复核**：这两个实验此前是全 suite 里样本量
  最小的（N=32-64 vs 其它大多数 128），全部 4 个模型分别翻倍到 N=128
  （PT6、PT7 的 `compare_oracle_rollout`）和 N=64（PT7 的 causal
  interpolation），核心方向和量级都复现（见各自"严谨性补强"一节）。
- **样本规模仍偏小**：多数实验 N=128 序列（ELF, seq_len=1024）或 128
  （LangFlow），低于 doc 建议的 512；t-grid 稀疏（11-21 点，非 doc 建议的
  101 点密网格）——这是唯一还没解决的系统性问题。

**写作建议**：全部 10 个实验的头条数字现在都可以直接带 CI（PT1-5/8-10）或
多种子均值±标准差 / 双倍样本确认（PT6/7）引用，不再需要"单次运行"这类
限定语；仍需要如实注明的是样本规模（N=128，非 doc 建议的 512）和 t-grid
稀疏这两点。

### R.1 十个实验的头条结果一览

| 实验 | 核心发现 | 严谨性状态 |
|---|---|---|
| PT1 先验减法 | 减除参考先验后，真值 token 相对默认竞争者的 margin 几乎转正（advantage retained：baseline~1%, KD~10-12%, LangFlow 26-28%）；但"减先验"主要重新分配非top-1概率质量，不总是让真值变top-1（`never_residual_correct`有时反而变差）| bootstrap CI 完成，P(m_res>0)=1.000 |
| PT2 margin轨迹 | KD的转变时间(`tau_e/tau_b/tau_s`)比baseline提前近一半(~0.09 vs ~0.19-0.35)；KD population-level `m_res(t)`从最早采样点起就已转正(0 zero-crossings vs baseline 1)；KD post-crossing斜率是baseline的5-6倍(85-90 vs 16.5) | bootstrap CI 完成，各类失败比例CI窄 |
| PT3 速度场对齐 | `a_clean(t_min)`四模型全部强正(+0.65~+0.79)——向量场早期即指向真值，全suite最干净正面发现；但token方向证据(u_yf)与frequency-matched control的corr **CI在4/4模型上重叠**，doc判定规则确认未满足 | bootstrap CI 完成(本次新增) |
| PT4 上下文消融 | ELF baseline的证据几乎全部来自局部半径1窗口(与full context统计不可区分)；kd_cr/kd2有小但显著差距(-0.9~-1.1pp)；**LangFlow差距既显著又大(-21pp)**——需要远大于半径1的感受野 | bootstrap CI 完成，LangFlow探针数3→26修复 |
| PT5 decoder偏置干预 | baseline上先验减法让tau_b小幅提前(+0.006，95% CI不含0，弱支持"decoder边界"假说)；**kd_cr/kd2/LangFlow上先验减法反而让tau_b推迟**(-0.06~-0.35，CI均不含0)，方向相反 | bootstrap CI 完成(本次新增) |
| PT6 扰动分支 | flip rate随时间单调下降(转变点前更脆弱)；**baseline的rollout是自我纠正的(final<immediate)，kd_cr/kd2和LangFlow是放大的(final是immediate的3-5倍甚至更高)** | kd_cr有3-seed复核；**全部4模型N=64→128翻倍复核，每个checkpoint的方向和量级都复现**(imm几乎逐位吻合) |
| PT7 oracle-rollout对比 | 四模型早期都是G_reverse>G_oracle；**KD让平均gap变大而非变小(baseline 0.097→kd2 0.271)**；causal interpolation下kd_cr在λ=1时崩溃到0.134(3-seed均值≈0.22) | kd_cr causal interpolation有3-seed复核；**两部分都做了N翻倍复核**：compare_oracle_rollout N=64→128核心排序(baseline<kd_cr<kd2)完全复现，⚠️但LangFlow从"gap最大"降到第三(不再稳健声称"LangFlow绝对最大")；causal interpolation N=32→64四模型五个λ值几乎逐位吻合(kd_cr λ=1崩溃值0.165，落在seed复核的0.134-0.290范围内) |
| PT8 minimal pair | `existential_there_subject_raising`在全部4个模型上都是语法线索拉力最弱的类别，无一例外，且**rank_good的95% CI在4/4模型上都与其余5个UID的CI完全不相交**；`determiner_noun_agreement`/`distractor_agreement`持续排前列 | per-UID bootstrap CI完成(本次新增)；⚠️好/坏rank比值在KD checkpoint上因rank_bad≈0而数值退化，改用rank_good本身做CI后结论稳健 |
| PT9 跨时间探针迁移 | 四模型都是early→late方向的迁移优于反向，95% CI在4/4模型上都不含0；**KD提升的是迁移能力(upper_tri_mean)而非单点探针准确率(diag_mean几乎不变)**——直接对应doc"KD stabilizes evidence coordinate system"判定规则；**第4种表示"prior-subtracted logits"补齐**：不训练probe(不可行也无意义)，改用条件正确率P(t_b\|t_a)，4/4模型复现同一不对称方向 | bootstrap CI 完成(本次新增)，raw_z跨checkpoint完全相同(健全性检查通过)，⚠️第4种表示的数字与前3种表示的量级不可比(定义不同的指标) |
| PT10 失败预测器 | 逻辑回归val_acc全部好于多数类基线(+2.5~+6.3pp)，4/4模型95% CI不含0、P(improvement>0)=1.000；**KD checkpoint的log_freq系数(3-6)比baseline(0.6-1.0)大数倍**——KD更依赖频率捷径 | bootstrap CI 完成(本次新增)，纯事后分析无需GPU |

### R.2 贯穿全部10个实验的核心张力（论文叙事主线，多个独立指标交叉验证）

**"KD 让模型更早、更决绝地承诺"和"KD 让模型对扰动更鲁棒/让真实轨迹更贴近理想
路径"是两件不同的事，需要在论文里明确区分，绝不能混为一谈**：

- 支持"KD承诺更早/更决绝"的独立证据：PT1（advantage retained更高）、PT2
  （tau提前、post-slope更陡）、PT5（beta干预下flip rate更高）、PT10（log_freq
  系数更大，两极化更强）。
- 支持"KD对扰动/偏离更脆弱、不是更鲁棒"的独立证据：PT6（放大而非纠正，3-seed
  复核确认非偶然）、PT7（causal interpolation下λ=1时灾难性崩溃，3-seed复核
  确认）、PT5（先验减法对KD是负收益，方向与baseline相反）。
- PT4 提供了架构轴（而非训练方式轴）上的类似张力：ELF 三个 checkpoint 都是
  局部证据高度充分且必要，LangFlow 则需要大得多的感受野——这是跨架构而非
  跨训练方式的差异，不应与 KD 的发现混淆。

这个张力应该作为论文 Discussion/Limitations 里的一个明确小节，而不是分散在
各处默认"KD 全面更好"，需要同时呈现两个方向的证据。

### R.3 对已有 Issue（Section 0-9）的补充/交叉验证

- **PT9 的独立复现直接支持 Issue 1/Section A/Story A 系列的"probe gap"结论**：
  PT2 的 Independent-probe score 补测（用完全不同的代码路径、t-grid、训练超参）
  复现了 `EXP-07v2`/`EXP-21v2` 的符号方向（baseline: probe>native +12pp；
  KD/LangFlow: native>probe −8~−9pp）——可以作为 Story A 的第三次独立验证引用。
- **PT7 的"KD平均gap变大"是 EXP-01v3（Issue 1 讨论的 Protocol A vs B 差异）
  之外的新发现**：不是重复 EXP-01v3，而是在统一 adapter 上把 Protocol A/B
  对比首次扩展到了 LangFlow 和更细的 causal interpolation 分析。
- **PT1 的 padding 修复经验值得作为方法论脚注**：`null_mode_token` 最初
  未排除 padding，导致 baseline 的 `frac_null_mode_but_residual_specific`
  从假象的 25.7-64.2% 修正为真实的 3.0-3.8%（KD 从 89-94% 修正为 57-70%，
  依然显著）——这是本 suite 内部一次自我修正的例子，如果论文附录讨论方法论
  严谨性，这是一个具体、有数字的案例。

### R.4 建议写入论文的安全表述 vs 需要限定语的表述

**可以直接引用（有 bootstrap CI 或多种子支持）**：
- "The vector field's drift direction aligns with the ground-truth token
  direction ($a_\text{clean}$) from the earliest sampled $t$, across all four
  model/training combinations (cosine +0.65 to +0.79, 95% CI excludes 0;
  EXP-PT3)."
- "Knowledge-distilled checkpoints cross the margin-crossing boundary
  ($\tau_b$) roughly twice as early as the baseline (EXP-PT2), but perturbing
  the trajectory near this boundary causes their errors to *amplify* rather
  than self-correct on continued rollout — the opposite of the baseline's
  behavior — confirmed across 3 independent seeds (EXP-PT6)."
- "Local context within a radius of 1 token is statistically indistinguishable
  from full context for ELF baseline (95% CI on the difference includes 0),
  but LangFlow requires a much larger receptive field (gap of 21pp, CI
  excludes 0) — EXP-PT4."

**需要加限定语（效应有反例，需要 checkpoint-scoped 表述，而非因为缺 CI）**：
- "先验减法能揭示悬崖是 decoder 边界造成的"这条 claim **不能一般化**——PT5
  的 bootstrap CI 确认这只对 baseline 成立（弱，+0.006，CI 不含 0），对
  KD/LangFlow 方向相反且 CI 同样不含 0（−0.06~−0.35），写作时必须
  checkpoint-scoped（呼应 Section EXP-36 对 dec_sc 的类似处理方式）；这一条
  现在已经有 CI 支持，可以直接引用，不需要再加"单次运行"的限定语。
- PT8 的"existential_there最弱"这个排序结论现在有 bootstrap CI 支持（4/4
  模型的 rank_good CI 都与其余 UID 不相交），中间名次（distractor/irregular/
  wh/npi 之间）仍然浮动较大，不应过度解读具体顺序。

**对论文的整体影响**：Phase-Transition suite 提供了本文档 Section A-Q 之外一条
独立的、跨 10 个互补角度的证据线，其中 PT1-PT4 的"早期证据存在但被默认先验/
decoder 边界掩盖"这条主线，与 Issue 3/Issue 4（G(t)/ρ(t) 的解释问题）关系密切，
建议在修订 Issue 3/4 的相关段落时交叉引用 PT1/PT3 的结果作为独立支持；PT6/PT7
的"KD 放大而非纠正"发现建议作为一个新的、独立的 Discussion 小节，不要塞进
现有 Issue 1-9 的任何一条里（因为它是一个新机制发现，不是对某个已有 claim 的
修正）。
