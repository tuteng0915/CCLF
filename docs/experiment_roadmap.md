# Experiment Roadmap: EXP-00 to EXP-16

**Strategic goal:** Establish the mechanism clearly before optimizing for PPL.
**Implementation stack:** ELF-torch for reverse-trajectory work; JAX ELF for forward-oracle probes
(existing infrastructure is already tested there); LangFlow for corrected comparison.

**Execution order:** Phase 0 → Phase 1 → Phase 2 → Phase 3.
EXP-00 and EXP-01 are the current blockers.

---

## Summary Table

| EXP | Name | Phase | Priority | Est. effort | Blocks |
|-----|------|-------|----------|-------------|--------|
| EXP-00 | Tensor audit doc | 0 | MUST | 1 day | EXP-01, EXP-02, EXP-03 |
| EXP-01 | Forward-oracle vs reverse trajectory | 0 | HIGHEST | 3–5 days | EXP-11, EXP-14 |
| EXP-02 | Correct LangFlow comparison | 0 | HIGH | 2–3 days | abstract claim |
| EXP-03 | Matched-SNR comparison | 0 | HIGH | 2–3 days | LangFlow/ELF timing claim |
| EXP-04 | Decoder geometry null model | 1 | HIGH | 2 days | G(t) interpretation |
| EXP-05 | Learned prior estimation | 1 | MEDIUM | 2 days | ρ(t) overclaim |
| EXP-06 | Prior subtraction (debiased G) | 1 | MEDIUM | 1 day | ρ(t) overclaim |
| EXP-07 | Linear probe on x̂_t | 1 | HIGH | 3–5 days | Story A |
| EXP-08 | Coarse-to-fine ordering | 2 | MEDIUM | 3–5 days | section 4 motivations |
| EXP-09 | Contextual bootstrapping | 2 | MEDIUM | 3–5 days | section 4 motivations |
| EXP-10 | (reserved) | — | — | — | — |
| EXP-11 | Branching stability | 2 | MEDIUM | 3–5 days | EXP-14 |
| EXP-12 | Residual analysis | 2 | MEDIUM | 3–5 days | ρ(t) overclaim |
| EXP-13 | Compute controls for dec_sc | 1 | HIGH | 2–3 days | §4.8 overclaim |
| EXP-14 | Commit–release on actual trajectories | 2 | HIGH | 2 days | §4.14 overclaim |
| EXP-15 | Training t-distribution variants | 3 | DEFER | 1–2 wks | §4.7 overclaim |
| EXP-16 | Minimal training objective | 3 | DEFER | 2–4 wks | Section 4 objectives |

---

## Phase 0 — Blocking correctness

Phase 0 experiments are prerequisites for all others. Do not start Phase 1 until EXP-00 is done
and EXP-01 at least has trajectory-saving code running.

---

### EXP-00 — Tensor and code-path audit (documentation only)

**Hypothesis:** N/A (audit, not a hypothesis-testing experiment).

**Why it's blocking:** Every downstream experiment plots tensors from different models side-by-side.
Without an explicit mapping from variable name → math symbol → shape, cross-model comparisons
contain latent errors. The LangFlow probe measuring G(t) on `z_t` instead of a native posterior
is exactly this kind of error.

**Deliverable:** `CCLF/docs/tensor_audit.md` with:

```
For each model:
  Model: ELF (JAX)
  Script: experiments/probe_elf/probe_geo.py
  Tensors:
    z_t   : (L, d), noisy input, z_t = t·x_clean + (1-t)·ε; NOT the backbone output
    x_hat : (L, d), backbone denoised prediction, represents x̂_t; THIS is what G(t) is computed on
    a_t   : (L, d), vocabulary barycenter E^T p_t; computed as E.T @ softmax(x_hat @ E.T)
    rho_t : scalar, anchor mismatch ratio ||x_hat - a_t|| / ||x_hat||

  Model: ELF (torch)
  Script: models/ELF-torch/src/utils/generation_utils.py
  Tensors:
    z_t (in-loop name "x") : updated noisy state at each ODE/SDE step
    x_pred                 : backbone output = x̂_t (denoised prediction for this step)
    x_pred_prev            : self-conditioning input from previous step (None on first step)
    t_val                  : scalar float, current time, ranges 0→1 (0=noise, 1=clean)

  Model: LangFlow
  Script: experiments/probe_langflow/probe_geo_langflow.py
  BUG: Currently plots G(t) on z_t (noisy input), not on LangFlow's x_t.
  Correct native objects:
    z_t   : (L, d), noisy input — current probe uses this (WRONG for G comparison)
    x_t   : (L, V) logits from vocabulary head; a_t = E^T softmax(x_t) is the correct posterior
    h_t   : hidden state before vocabulary head (if accessible — check LangFlow architecture)
```

**Effort:** 1 day. Read scripts, write table. No code changes.

**Acceptance condition:** Every tensor that appears in any cross-model plot is documented here.
No further experiments may produce cross-model figures without updating this doc first.

---

### EXP-01 — Forward-noise oracle probe vs actual reverse trajectory

**Hypothesis (critical):** The "commitment cliff" and "non-monotonic cosine alignment" patterns
observed in existing probes are artifacts of the forward-noise oracle protocol and may not appear
on actual reverse-generation trajectories. Alternatively: the oracle and reverse-trajectory metrics
coincide in the commitment cliff region but diverge near t≈0.55-0.60.

**Why it's the highest priority:** If the patterns diverge, the entire Section 4 design motivation
(scheduling objectives around the cliff) must be reconsidered. The mechanism story depends on
this comparison.

**Existing code to reuse:**
- `experiments/probe_elf/probe_geo.py` — metric computation functions (top-1 GT, G(t), entropy,
  decode-branch readout); call these on saved reverse-trajectory states
- `experiments/probe_elf/probe_token_trajectories.py` — Type A/B/C/D classification logic;
  apply to reverse-trajectory data as comparison

**New code required (ELF-torch):**

1. **`models/ELF-torch/src/configs/config.py`** — add to `SamplingConfig`:
   ```python
   save_trajectory: bool = False     # if True, accumulate per-step tensors
   trajectory_save_path: str = ""    # where to write trajectory file
   ```

2. **`models/ELF-torch/src/utils/generation_utils.py`** — in `_generate_samples_single_batch()`,
   locate the inner loop (currently a simple for-loop over timesteps) and add:
   ```python
   if cfg.save_trajectory:
       traj_steps.append({
           "t": t_val.item(),
           "z_t": x.cpu().clone(),          # current noisy state BEFORE step
           "x_pred": x_pred.cpu().clone(),  # backbone's denoised prediction x̂_t
           "x_pred_prev": x_pred_prev.cpu().clone() if x_pred_prev is not None else None,
       })
   ```
   After the loop, if enabled, save `traj_steps` as a pickle or torch file.

3. **New script: `experiments/probe_elf/probe_reverse_trajectory.py`**
   ```
   Steps:
   1. Load ELF-torch checkpoint (kd-cr step 703659 = dec_sc trained)
   2. Generate N=64 sequences with save_trajectory=True; save each step's (t, z_t, x̂_t, x̂_t_prev)
   3. For each step in each trajectory:
      - Compute top-1 ground-truth accuracy (using final accepted sequence as reference)
      - Compute G(t): cosine-normalized token readout accuracy
      - Compute entropy of p_t = softmax(x̂_t @ E^T)
      - Compute decode-branch top-1 accuracy (if dec_sc step, use x̂_t_prev → backbone(·, t=1))
      - Compute ρ(t): anchor mismatch ratio
   4. Average across sequences; plot all metrics vs t on same axis as forward-oracle probe (Protocol A)
   ```

**Comparison:** Protocol A (existing oracle, 64 seqs, 20 t-bins) vs Protocol B (reverse trajectory,
64 seqs, steps at their natural t values). Plot each metric as two overlaid curves.

**Effort:** 3–5 days (2 days for code changes, 1–2 days for debugging + running, 1 day for plots).

**Deliverable:**
- `results/exp01/protocol_comparison.png` — 5-panel figure: G(t), entropy, top-1 GT, decode-branch
  top-1, ρ(t), each with Protocol A (blue) and Protocol B (orange)
- `results/exp01/comparison_table.md` — cliff region [0.20,0.35]: mean difference A vs B for each metric

**Decision rule:**
- **B ≈ A (max absolute difference < 5pp in commitment cliff region):** Oracle probes approximately
  represent reverse-trajectory behavior. Add one sentence to Section 5: "We verified that the
  forward-noise oracle probe closely tracks actual reverse-trajectory dynamics (EXP-01, Appendix~X)."
  All trajectory language in the paper can stand with the "forward-noise oracle" qualifier.
- **B diverges from A (>10pp difference in cliff region OR non-monotonic pattern absent in B):**
  The paper's Section 4 motivation needs rewriting. Remove the scheduling motivation based on
  forward-probe cliff. Rewrite as "oracle probe reveals a potential commitment window; whether
  this appears in reverse trajectories requires further study." Section 4 training objectives
  become "motivated by oracle-probe evidence, with the caveat that trajectory equivalence is
  not yet established."

---

### EXP-02 — Correct LangFlow comparison

**Hypothesis:** LangFlow's native token posterior $x_t$ (from the vocabulary projection head)
shows earlier geometric commitment than the noisy input $z_t$ currently used in the paper.
If so, the "ELF commits 60pp earlier" claim is partially a measurement artifact.

**Background:** `probe_geo_langflow.py` currently computes G(t) on `z_t` (the noisy diffusion input),
NOT on LangFlow's internal representation after vocabulary projection. ELF G(t) uses `x̂_t`
(backbone output). These are fundamentally different objects. The comparison in the paper's
abstract ("LangFlow remains below 5%") conflates them.

**What to actually compare (symmetric):**

| Level | ELF | LangFlow |
|-------|-----|----------|
| Noisy input | `z_t` | `z_t` |
| Backbone hidden state | `h_t` (before linear head) | `h_t` (before vocab head) |
| Token posterior embedding | `a_t = E^T p_t`, `p_t = softmax(x̂_t W^T + b)` | `a_t = E^T p_t`, `p_t = softmax(LM_head(h_t))` |
| Denoised prediction | `x̂_t` (ELF backbone output) | N/A (LangFlow outputs logits directly) |

**Code changes to `probe_geo_langflow.py`:**
1. Identify where LangFlow's vocabulary head computes logits: find the `lm_head` or equivalent module
2. Add a forward hook to capture `h_t` (hidden state before head) and `p_t` (softmax of logits)
3. Compute `a_t = E^T p_t` (where E is LangFlow's token embedding matrix)
4. Compute G(t) on `a_t` instead of (or in addition to) `z_t`
5. Also compute G(t) on `h_t` if `h_t` has the same dimensionality as ELF's embedding space

**Effort:** 2–3 days (1 day to understand LangFlow model structure, 1 day to add hooks, 1 day for runs).

**Deliverable:**
- `results/exp02/langflow_comparison.png` — G(t) curves: ELF x̂_t (paper), LangFlow z_t (paper),
  LangFlow a_t (corrected), LangFlow h_t (if available)
- Updated comparison table

**Decision rule:**
- **LangFlow a_t shows early commitment (>30% by t=0.30):** Abstract claim "ELF commits far earlier
  than any other model" is likely a measurement artifact. Replace with: "ELF's backbone outputs
  commit geometrically earlier than LangFlow's vocabulary-posterior embedding, though the gap
  narrows when LangFlow is probed at comparable representational levels."
- **LangFlow a_t still <10% by t=0.30:** The claim survives at the corrected measurement level.
  Update paper to note the comparison is now symmetric: "both measured as cosine-nearest-token
  accuracy on the vocabulary-posterior embedding a_t."

---

### EXP-03 — Matched-SNR comparison

**Hypothesis:** Part of the timing gap between ELF and LangFlow is explained by their different
noise schedules. At matched log-SNR values (not matched nominal t), the gap shrinks.

**Background:** ELF uses flow-matching with $z_t = t \cdot x_\text{clean} + (1-t)\epsilon$.
LangFlow may use a different noise schedule with different log-SNR at the same nominal t.
If ELF's t=0.30 corresponds to LangFlow's t=0.20 in terms of log-SNR, the timing comparison
in the paper is misleading.

**Code:**
1. Read SNR curves from `results/elf/snr_analysis/` (ELF-JAX already computed)
2. Compute LangFlow's log-SNR at each t from its noise schedule formula (read from LangFlow code)
3. Create a t-matching function: for each ELF t, find LangFlow t' such that log-SNR(ELF, t) = log-SNR(LangFlow, t')
4. Re-plot ELF G(t) and LangFlow G(t') on a shared log-SNR axis (instead of nominal t axis)
5. Re-plot with corrected LangFlow posterior from EXP-02 on the SNR-matched axis

**Effort:** 2–3 days (mostly reading LangFlow schedule, computing matching, making plots).

**Deliverable:**
- `results/exp03/snr_matched_comparison.png` — G(t) on shared log-SNR axis
- Number: what fraction of the 60pp timing gap is explained by schedule mismatch vs representation differences

**Decision rule:**
- **Gap shrinks to <30pp at matched SNR:** Add to paper: "controlling for noise schedule, the
  timing gap reduces from ~60pp to ~Xpp; the residual gap reflects representational differences."
- **Gap remains >50pp at matched SNR:** Schedule confound is minor; the current comparison is
  approximately fair. Add a note about matched-SNR analysis in the appendix.

---

## Phase 1 — Mechanism isolation

Phase 1 experiments test the specific hypotheses behind the paper's mechanistic claims.
Start these after EXP-00 is complete and EXP-01 has initial results.

---

### EXP-13 — Compute-matched controls for dec_sc (HIGHEST PRIORITY in Phase 1)

**Hypothesis:** dec_sc improvement comes from token-specific correction information in the decode
branch, not from the extra nonlinear compute per generation step. This is the core claim in
Abstract:8 and Conclusion:8.

**Why this is high priority:** It's the main trained result in the paper. The compute-confound
alternative explanation is simple and plausible. Without this experiment, the claim cannot stand.

**Inference modes to add** (all require same FLOPs as dec_sc, one extra pass per step):

| Mode name | What the extra pass does | Expected behavior if compute hypothesis |
|-----------|--------------------------|----------------------------------------|
| `decode` | Standard dec_sc (decode branch at t=1, fed into self-cond) | — (baseline) |
| `extra_denoise` | Extra denoising-mode pass on current z_t (no decode branch) | ≈ dec_sc if compute matters |
| `decode_shuffled` | dec_sc but positions shuffled within batch before branch | ≈ baseline if correction is position-specific |
| `decode_wrong_t` | Decode branch evaluated at wrong timestep (t-0.2) | ≈ baseline if timing matters |
| `random_residual` | Matched-norm random vector added as self-cond | ≈ baseline if content matters |

**Code changes (ELF-torch only):**

In `generation_utils.py:_generate_samples_single_batch()`, add a `dec_sc_mode: str` parameter.
The current dec_sc code (simplified):
```python
# Step 1: preliminary pass (no self-cond)
x_pred_prev = backbone(z_t, t, sc_input=None)
# Step 2: decode branch
sc_input = decode_branch(x_pred_prev, t=1)
# Step 3: main pass
x_pred = backbone(z_t, t, sc_input=sc_input)
```

Add a `mode` switch:
```python
if dec_sc_mode == "decode":
    sc_input = decode_branch(x_pred_prev, t=1)        # current behavior
elif dec_sc_mode == "extra_denoise":
    sc_input = backbone(z_t, t, sc_input=None)         # denoise-mode extra pass
elif dec_sc_mode == "decode_shuffled":
    idx = torch.randperm(L)
    sc_input = decode_branch(x_pred_prev[:, idx, :], t=1)[:, torch.argsort(idx), :]
elif dec_sc_mode == "decode_wrong_t":
    sc_input = decode_branch(x_pred_prev, t=max(0, t-0.2))
elif dec_sc_mode == "random_residual":
    scale = decode_branch(x_pred_prev, t=1).norm()
    sc_input = torch.randn_like(x_pred_prev) * (scale / x_pred_prev.norm())
```

**Runs:** For each mode × step count {8, 16, 32}:
- 1000 generations each, Gen.PPL + MAUVE
- Report as table with columns: Mode, Steps=8, Steps=16, Steps=32

**Effort:** 2–3 days (1 day code, 1 day runs, 1 day analysis).

**Deliverable:**
- `results/exp13/compute_controls.md` — table of Gen.PPL and MAUVE for all modes × step counts
- Updated Abstract:8 and Conclusion:8 language based on result

**Decision rule:**
- **`extra_denoise` ≈ dec_sc:** Compute hypothesis supported. Abstract/Conclusion must say:
  "dec_sc improvement is primarily explained by additional nonlinear compute per step,
  not by the decode branch's token-specific correction."
- **`extra_denoise` < dec_sc, `shuffle/random` ≈ baseline:** Correction-information hypothesis
  supported. Abstract/Conclusion can say "consistent with the hypothesis that decode branch
  provides token-specific correction information (EXP-13); compute-matched controls show the
  improvement is not explained by additional FLOPs alone."
- **`extra_denoise` < dec_sc, `shuffle` also helps:** Some position-level context, but not
  token-specific. Adjust claims accordingly.

---

### EXP-04 — Decoder geometry null model

**Hypothesis:** G(t)'s early high values are partially explained by output-head geometry
(bias terms, output embedding norms, temperature calibration) rather than meaningful representation
content. A model with isotropic Gaussian x̂_t should show non-trivial G(t) due to output bias.

**Code:** New script `experiments/probe_elf/probe_null_model.py`:
```python
# For each t in [0.05, 0.10, ..., 0.95]:
# 1. Sample z_gauss ~ N(0, I_d), treat as "x_hat" from a random model
# 2. Compute full head prediction: p = softmax(z_gauss @ W^T + b); record top-1 token
# 3. Compute bias-free: p = softmax(z_gauss @ E^T); record top-1 token
# 4. Compute row-normalized (= G(t)): argmax_v (z_gauss @ E_v) / (||z_gauss|| ||E_v||)
# 5. Compare all three against ground truth (using same sequences as main probe)
```

Compare the null model's G(t) curve (constant across t by construction) against ELF's actual curve.
The gap above null-model baseline represents genuine representation quality.

**Existing code:** Use `probe_geo.py` metric functions; only the input (Gaussian vs actual x̂_t) changes.
No ELF model needed — this is a pure vocabulary-matrix geometry test.

**Effort:** 2 days.

**Deliverable:** `results/exp04/null_model_G.png` — G(t) curves: ELF actual, Gaussian null,
bias-free null. Table of null-model G at key t values.

**Decision rule:**
- **Null-model G > 15% at any t:** Output-head geometry provides non-trivial accuracy. Paper must
  say: "G(t) exceeds the output-head geometry baseline by Xpp, where the baseline reflects
  systematic token-prediction biases in the vocabulary head." Adjust the "60.8% by t=0.30" claim
  to report the baseline-corrected number.
- **Null-model G < 5% throughout:** G(t) is not driven by output bias; ELF's actual values
  reflect representation quality. One sentence in paper: "Output-head geometry contributes
  negligibly to G(t) (EXP-04, Appendix)."

---

### EXP-05 — Learned prior estimation

**Hypothesis:** ELF's backbone has learned a prior over common tokens (e.g., "the", "of", "and")
that appears as high G(t) even for context-free inputs. Subtracting this learned prior reveals
when the backbone is responding to the *specific* sequence vs. the corpus-level prior.

**Code:**
```python
# Batch-shuffle z_t across sequences at each t:
# For each t, take the z_t tensors from all N sequences and shuffle them across the batch dimension.
# This preserves the token-embedding geometry but destroys per-sequence information.
# Feed shuffled z_t to backbone → get x_hat_shuffled
# Compute G_prior(t) = G(x_hat_shuffled) — "accuracy" due to corpus-level prior only
# The true "sequence-specific commitment" is G(t) - G_prior(t)
```

**Effort:** 2 days. Uses existing JAX ELF infrastructure; just needs batch shuffling.

**Deliverable:** `results/exp05/prior_estimation.png` — G(t), G_prior(t), G_debiased(t) = G(t) - G_prior(t).

**Decision rule:**
- **G_prior(t) > 20% in cliff region:** The paper's G(t) numbers substantially overcount commitment.
  Report G_debiased(t) as the main metric.
- **G_prior(t) < 5%:** Prior bias is negligible; current G(t) interpretation is fine.

---

### EXP-06 — Prior subtraction (debiased G)

**Hypothesis:** After subtracting the learned prior (from EXP-05), the commitment cliff still
appears at the same t values, confirming that the cliff reflects sequence-specific content formation.

**Code:** Compute G_debiased(t) = G(t) - G_prior(t) from EXP-05 results. Plot and compare
cliff-region behavior. Compare EXP-05 prior curve with EXP-04 null-model curve.

**Effort:** 1 day (analysis only; data from EXP-05).

**Deliverable:** `results/exp06/debiased_cliff.png` — does cliff appear in G_debiased?

**Decision rule:**
- **Cliff still visible in G_debiased:** Cliff is a genuine feature of sequence-specific
  representation formation. Claim stands.
- **Cliff disappears after debiasing:** Cliff is a prior artifact. Remove or heavily qualify
  all cliff-based motivation in Section 4.

---

### EXP-07 — Linear probe on x̂_t (Story A validation)

**Hypothesis (Story A):** ELF's backbone forms token-predictive representations *earlier* than
the linear interface (tied-weight vocabulary head) can expose. A separately-trained linear probe
on x̂_t should exceed the native head's accuracy, especially in the cliff region, even without
decoder bias or temperature calibration.

This is the key experiment for the "Story A" framing recommended by the review:
> "ELF forms token-predictive states earlier than its native linear interface can expose;
> decode branch correction can be fed back into the trajectory."

**Code:** New script `experiments/probe_elf/probe_linear_probe.py`:
```python
# For each t in [0.05, 0.10, ..., 0.95]:
# 1. Run forward-oracle probe to collect x̂_t and ground-truth y for N=500 sequences
# 2. Hold out 20% for test; train a linear probe W_probe on train set:
#    W_probe: (d,) × (V,) → argmax = y_i? (equivalent to learning a new projection)
#    Initialize to E (tied weights) or random; train with CE for 100 steps
# 3. Evaluate on test set: Probe@1(t)
# 4. Compare to: G(t) (row-norm cosine), Rec@1(t) (native head), Dec@1(t) (decode branch)
```

**Key comparison:**
- **Native Rec@1(t):** uses E = W (tied), with bias and temperature
- **G(t):** uses E, bias-free, row-normalized
- **Linear probe Probe@1(t):** learned separately; not constrained to use E
- **Dec@1(t):** decode-branch accuracy (extra nonlinear pass)

If Probe@1 >> Rec@1 throughout: the representation has information the native head cannot extract.
If Probe@1 ≈ G(t): the information is accessible via cosine geometry but not through the biased head.
If Probe@1 ≈ Rec@1: G(t) gap over Rec@1 is from norm/bias, not from hidden structure.

**Effort:** 3–5 days (data collection, probe training loop, multiple t values, comparison).

**Deliverable:**
- `results/exp07/probe_comparison.png` — four-curve figure: G(t), Rec@1(t), Probe@1(t), Dec@1(t)
- Story A verdict: supported / not supported

**Decision rule:**
- **Probe@1 >> Rec@1 by >10pp in cliff region (and Probe@1 approaches Dec@1):**
  Story A confirmed. Paper can claim: "ELF's backbone representations encode token identity
  substantially earlier than the native tied-weight projection can reveal; a separately-trained
  linear probe achieves X% accuracy by t=0.30, vs Y% for the native head."
  This is the core claim to build the revision around.
- **Probe@1 ≈ Rec@1:** The gap between G(t) and Rec@1(t) is entirely explained by bias/norm
  effects in the vocabulary head, not by hidden structure. Story A is not supported.
  The paper must be reframed: "We observe that bias-free cosine readout G(t) exceeds biased
  native readout Rec@1(t); whether this reflects representation quality or head geometry
  is not yet resolved."

---

## Phase 2 — Deeper structure

Run Phase 2 experiments only after Phase 1 results are clear. EXP-14 requires EXP-01;
EXP-11 and EXP-08-09 require strong Phase 1 mechanism results to be worth the effort.

---

### EXP-14 — Validate commit–release–recommit on actual reverse trajectories

**Requires:** EXP-01 data (saved reverse trajectories).

**Hypothesis:** The non-monotonic cosine alignment pattern (peak at t≈0.55-0.60, slight decline)
observed in forward-oracle probes also appears in actual reverse-generation trajectories.

**Code:** Analysis of saved trajectory data from EXP-01. For each saved reverse-trajectory step,
compute G(t) at each t. Plot the resulting curve across sequences. Compare to forward-oracle G(t).

**Effort:** 2 days (analysis; data from EXP-01).

**Deliverable:** `results/exp14/reverse_Gt_nonmonotone.png`

**Decision rule:**
- **Non-monotonic pattern confirmed (>3pp drop after peak) in Protocol B trajectories:**
  "commit–release–recommit" language in Section 4 can be restored with appropriate caveats:
  "The non-monotonic alignment pattern is confirmed in reverse-generation trajectories (EXP-14)."
- **Pattern absent or monotone in Protocol B:** Remove all commit–release–recommit language.
  Section 4.2 becomes "motivated by forward-oracle observation; validation on reverse trajectories
  is EXP-14."

---

### EXP-11 — Branching stability

**Requires:** EXP-01 data (saved reverse trajectories).

**Hypothesis:** The "stuck" phenomenon (wrong positions remain wrong from t≈0.35 onward) that
motivates the cliff-region scheduling occurs at the same t values in actual reverse trajectories.

**Code:**
```python
# From EXP-01 saved trajectories:
# For each saved checkpoint (t, z_t) along the reverse trajectory:
# 1. Branch K=20 continuations from this checkpoint: run reverse ODE from t to t=1
#    using saved z_t as starting state, with different noise seeds for SDE or
#    with different ODE integration seeds via stochastic step size
# 2. Record final token sequence for each continuation
# 3. For each position i, compute: P_branch(t, i) = distribution over final tokens across K branches
# 4. Compute branch entropy H(P_branch(t, i)) → "position irreversibility" at time t
# 5. Flag positions that have committed to wrong token at t=0.35 (argmax ≠ y_i);
#    check whether they self-correct in any of the K continuations
```

**Effort:** 3–5 days (branching infrastructure, K×trajectory generation, analysis).

**Deliverable:** `results/exp11/branch_stability.png` — P(wrong stays wrong | wrong at t)
across t, on actual reverse trajectories.

**Decision rule:**
- **Wrong positions have <5% branch-recovery rate from t=0.35:** "Stuck" phenomenon confirmed
  on actual trajectories. CE supervision in the plateau is not useful for recovery; the cliff
  motivation stands.
- **Wrong positions self-correct in >15% of branches from t=0.35:** Positions are not stuck;
  the stochasticity of the reverse process provides self-correction that the oracle probe misses.
  Section 4.2 cliff motivation must be weakened significantly.

---

### EXP-12 — Residual analysis

**Hypothesis:** The residual $r_t = \hat{x}_t - E^\top p_t$ (with large irreducible norm ρ(t)≈0.82)
actually encodes syntactic and semantic information beyond token identity. If so, the paper's
weaker claim ("large persistent norm") can be upgraded to "encodes contextual structure."

**Code:**
```python
# For N=500 sequences, at each t:
# 1. Compute r_t = x_hat - a_t for each position i
# 2. Train linear probes on r_t to predict:
#    - Syntactic role: POS tag (NOUN, VERB, ADJ, etc.) from spaCy parsing
#    - Next-token coherence: does next token match ground truth?
#    - Semantic cluster: coarse semantic class (entity type, etc.)
#    - Token identity: the actual token y_i (null hypothesis — r_t should not predict this)
# 3. Compare probe accuracy on r_t vs on x_hat vs on a_t
```

**Effort:** 3–5 days (POS tagging pipeline, probe training at each t, comparison).

**Deliverable:** `results/exp12/residual_content.png` — probe accuracy for each content type
across t, comparing r_t vs x̂_t vs a_t probes.

**Decision rule:**
- **r_t probes exceed a_t probes by >5pp for syntactic/semantic tasks:**
  The residual encodes contextual structure. Paper can say: "The residual $r_t$ encodes
  Xpp syntactic role information beyond lexical identity (EXP-12), confirming it is a structural
  feature of contextual embedding."
- **r_t probes ≈ a_t probes or near-chance:** Residual is high-dimensional noise with no
  extractable contextual structure. Must say: "The large anchor mismatch ratio reflects
  geometric properties of contextual embedding spaces; whether the residual encodes
  interpretable structure is not yet established."

---

### EXP-08 — Coarse-to-fine prediction ordering

**Hypothesis:** ELF's backbone first commits short/common tokens (function words), then
uncommon tokens (content words), reflecting a coarse-to-fine prediction order that motivates
the position-adaptive KD mask.

**Code:** Using existing forward-oracle probe data, split positions by:
- Token frequency (log-frequency bucket)
- Token length
- POS tag
- Position in sequence (beginning/middle/end)

Compute G(t) separately for each subgroup. Plot commitment time per group.

**Effort:** 3–5 days (subgroup analysis, frequency bucketing).

**Deliverable:** `results/exp08/coarse_to_fine.png` — G(t) curves stratified by frequency/POS.

**Decision rule:**
- **Significant ordering (>10pp commitment-time gap between high/low frequency):**
  Motivates position-adaptive objectives in Section 4. Paper can claim: "ELF exhibits
  coarse-to-fine ordering: high-frequency positions commit at t=X, content words at t=Y."
- **No ordering:** Remove coarse-to-fine motivation from Section 4. Position-adaptive
  objectives are not well-motivated by this finding.

---

### EXP-09 — Contextual bootstrapping

**Hypothesis:** Early-committed positions (high-frequency, function words, beginning of sequence)
provide context that improves later-committed positions' accuracy — the model bootstraps.

**Code:** 
- Take positions committed by t=0.35 (using oracle probe) and mask them out
- Run oracle probe again without these positions' tokens in the target
- Compare G(t) of remaining positions in full vs masked condition

**Effort:** 3–5 days.

**Deliverable:** `results/exp09/bootstrapping.png` — G(t) of content words with/without
committed function-word context.

**Decision rule:**
- **>5pp G(t) drop when early-committed positions masked:** Bootstrapping is real;
  position-adaptive scheduling is motivated.
- **<2pp difference:** Bootstrapping is not significant; positions commit independently.

---

## Phase 3 — Training intervention

Run Phase 3 only after Phase 1 and Phase 2 establish the mechanism. These experiments are
expensive (1–2 weeks each) and should not run until the mechanism is understood well enough
to know which objectives to test.

**Do not start training experiments to chase PPL.** Start them to test specific mechanism
predictions. The objective of Phase 3 is to answer: "If the mechanism we found in Phase 1-2
is real, does targeting it during training improve the model?"

---

### EXP-15 — Training t-distribution variants

**Requires:** EXP-01 results (to know if cliff matters on actual trajectories)
and EXP-07 results (to know if representation has hidden structure).

**Hypothesis:** Reweighting training toward the commitment cliff region improves final model
quality, because the backbone's gradients are most informative there.

**Three conditions:**
1. **Baseline:** ELF original t-distribution (logit-normal)
2. **q(t) without correction:** q(t) ∝ |dG/dt| from probe; no importance weights (as in paper's method, deliberately reweighted)
3. **q(t) with correction:** Same q(t) sampling but apply importance weights p(t)/q(t) to recover original objective

**Effort:** 1–2 weeks per condition (depends on training infrastructure availability).

**Deliverable:** Gen.PPL comparison at equal steps (not equal wall-clock), MAUVE, repetition rate.

**Decision rule:**
- **Condition 2 beats condition 1 and 3:** Reweighted objective helps (cliff-aware training);
  claim the reweighting is a design choice, not a sampling trick.
- **Condition 3 ≈ condition 1, condition 2 diverges:** No benefit from cliff-focusing when
  properly corrected; the paper's q(t) method only works because it changes the training objective.
- **All three ≈ equal:** t-distribution doesn't matter much; remove from paper's contributions.

---

### EXP-16 — Minimal training objective

**Requires:** EXP-13 (compute controls for dec_sc) and EXP-07 (linear probe Story A).

**Hypothesis:** The simplest version of the training objective (either L_KD alone or L_sc alone)
achieves most of the benefit over the baseline, without needing all five proposed objectives.

**Conditions:**
1. **ELF baseline** (continued training on L_denoise only, equal compute)
2. **L_KD only** — decode-teacher KD into linear head
3. **L_sc only** — decoder self-consistency in embedding space
4. **L_KD + L_sc** — the two geometry-motivated components
5. **Full objective** — all five objectives as in Section 4

**Critical comparison:** Condition 1 (equal-compute baseline) vs conditions 2-5.
If any condition improves over condition 1 by >1 PPL at 16 steps, the method has genuine value.

**Effort:** 2–4 weeks total (5 conditions × training runs).

**Deliverable:** Gen.PPL at {8, 16, 32, 128} ODE steps and {8, 16} SDE steps for all conditions.

**Decision rule:**
- **L_KD alone (or L_sc alone) beats baseline and equals full objective:**
  Paper's main contribution is: "a single distillation objective targeting the decode-branch
  advantage closes most of the gap." Remove the other objectives from the method.
- **Full objective beats single components but all beat baseline:**
  Method is valid; all components contribute. Train with full objective for final results.
- **Nothing beats equal-compute baseline:**
  The inference-time results (dec_sc, SAR) are the paper's main contribution.
  Training-time objectives do not add value beyond continued training.
  Remove Section 4 training claims entirely; focus paper on understanding + inference-time methods.

---

## Execution order (recommended)

```
Week 1-2:
  EXP-00  (tensor audit) — START IMMEDIATELY
  EXP-01  (reverse trajectory) — START IMMEDIATELY in parallel; this is the longest
            code task and should begin while EXP-00 is being written

Week 2-3:
  EXP-02  (correct LangFlow) — start once EXP-00 is done (needs tensor audit to avoid errors)
  EXP-03  (matched-SNR) — start in parallel with EXP-02

  EXP-13  (compute controls dec_sc) — start in parallel; independent of EXP-01 outcome
  EXP-04  (null model) — start in parallel; only needs JAX ELF and vocabulary matrix

Week 3-4:
  EXP-05/06 (prior estimation and debiasing)
  EXP-07  (linear probe) — most important for paper revision direction

Week 4-6:
  Assess Phase 0+1 results. Revise paper based on findings.
  Decide which Phase 2 experiments to run (not all may be necessary).

  EXP-14  (reverse commit–release) — run if EXP-01 shows trajectories work
  EXP-11  (branching stability) — run if phase 1 confirms mechanism

Week 6+:
  Phase 2 remaining (EXP-08, EXP-09, EXP-12) if motivated by Phase 1 results
  Phase 3 (EXP-15, EXP-16) when mechanism is fully established
```

---

## Code locations reference

| File | Role | Experiments |
|------|------|-------------|
| `experiments/probe_elf/probe_geo.py` | G(t), M(t), ρ(t) on JAX ELF | EXP-04, EXP-05, EXP-06, EXP-07 |
| `experiments/probe_elf/probe_token_trajectories.py` | A/B/C/D types; fixed ε | EXP-01 (Protocol A reference) |
| `experiments/probe_elf/probe_transition.py` | w2c/c2w rates | EXP-11 (reference) |
| `experiments/probe_elf/probe_decode_branch.py` | linear vs decode gap | EXP-04, EXP-07 |
| `experiments/probe_langflow/probe_geo_langflow.py` | LangFlow G(t) (**currently wrong**) | EXP-02 (fix here) |
| `models/ELF-torch/src/utils/generation_utils.py` | ELF-torch generation loop | EXP-01 (add trajectory save), EXP-13 (add modes) |
| `models/ELF-torch/src/configs/config.py` | Sampling config | EXP-01 (add save_trajectory), EXP-13 (add dec_sc_mode) |
| `results/elf/snr_analysis/` | SNR curves | EXP-03 |

New scripts to create:
- `experiments/probe_elf/probe_reverse_trajectory.py` (EXP-01)
- `experiments/probe_elf/probe_null_model.py` (EXP-04)
- `experiments/probe_elf/probe_linear_probe.py` (EXP-07)
- `experiments/probe_elf/probe_residual.py` (EXP-12)
