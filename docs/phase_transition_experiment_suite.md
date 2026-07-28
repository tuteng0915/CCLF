# Phase-Transition Experiments for Continuous Language Flows

## 0. Core question

We want to understand how a continuous language model moves from a prior-dominated, apparently meaningless high-frequency-token regime to a sample-specific lexical prediction.

The central hypothesis is:

> The early continuous state already accumulates weak sample-specific evidence. High-frequency tokens dominate only because the native categorical readout contains a strong default prior or geometric bias. A visible token transition occurs when accumulated sample-specific evidence crosses a decoder decision boundary.

The experiments below distinguish four stages:

1. **Prior-dominated stage**: output is mostly explained by state-independent or weakly state-dependent defaults.
2. **Evidence-emergence stage**: the true token becomes distinguishable after prior subtraction, even if it is not top-1.
3. **Boundary-crossing stage**: the true-token logit exceeds the current default/competitor token.
4. **Stabilization stage**: the prediction remains on the final token and is robust to perturbations.

These stages need not all occur. A trajectory may fail by never accumulating correct evidence, accumulating evidence for a wrong token, crossing too early and reversing, or remaining ambiguous until the endpoint.

---

# 1. Operational definitions

For position \(i\), ground-truth token \(y_i\), native logits \(\ell_{i,t}(v)\), and a reference prior \(q_{i,t}(v)\):

\[
e_{i,t}(v)=\ell_{i,t}(v)-\log q_{i,t}(v)
\]

is the prior-subtracted evidence score.

Let \(f_i\) be the early default competitor. Define:

\[
m^{raw}_{i}(t)=\ell_{i,t}(y_i)-\ell_{i,t}(f_i)
\]

\[
m^{res}_{i}(t)=e_{i,t}(y_i)-e_{i,t}(f_i)
\]

Use the following transition times:

### Evidence-emergence time

\[
\tau_i^{e}=\min\{t_k:m_i^{res}(t_j)>\delta_e,\ \forall j=k,\ldots,k+K_e-1\}
\]

Recommended defaults: \(\delta_e=0\), \(K_e=3\).

### Boundary-crossing time

\[
\tau_i^{b}=\min\{t_k:\arg\max_v \ell_{i,t_k}(v)=y_i\}
\]

### Stable-final time

\[
\tau_i^{s}=\min\{t_k:\arg\max_v\ell_{i,t_j}(v)=y_i,\ \forall j\ge k\}
\]

### Robust-stable time

At each \(t_k\), apply normalized perturbations to the full dynamical state and continue sampling. Define \(\tau_i^{r}\) as the earliest time at which the final token remains unchanged with probability at least \(1-\epsilon_r\).

Recommended default: \(1-\epsilon_r=0.9\).

The main phase gaps are:

\[
\Delta_i^{readout}=\tau_i^{b}-\tau_i^{e}
\]

\[
\Delta_i^{stability}=\tau_i^{s}-\tau_i^{b}
\]

These quantify how long evidence exists before it becomes visible, and how long visible correctness exists before it becomes stable.

---

# 2. Shared experimental protocol

## Data

- OpenWebText validation split.
- Primary run: 512 sequences.
- Sequence length: 128 for LangFlow-compatible experiments; optionally 1024 for ELF-only replication.
- Exclude padding.
- Analyze special tokens separately rather than mixing them into lexical results.
- Use document-level train/validation/test splits whenever a learned probe or calibrated reference is involved.

## Time grid

Within each model:

- Dense model-native grid: 101 points.
- Always use the **same epsilon for all time points of the same sequence**.
- For cross-model comparisons, additionally evaluate on 41 matched log-SNR points in the overlapping range.
- Never compare ELF and LangFlow using nominal \(t\) directly.

## Statistics

- Bootstrap confidence intervals by sequence, not by token position.
- Minimum 5 random seeds for interventions.
- Report means, medians, and 10/50/90 percentiles.
- Save per-position results so that first-hit, stable-hit, and failure categories can be recomputed.

## Suggested storage

Use Zarr or chunked `.pt` files rather than one huge tensor.

Per model/checkpoint:

```text
results/phase_transition/<model>/<checkpoint>/
  metadata.json
  tokens.npy
  mask.npy
  t_grid.npy
  logsnr_grid.npy
  logits.zarr
  hidden.zarr
  xhat.zarr
  velocity.zarr
  sc_state.zarr
```

---

# 3. EXP-PT1 — Prior-to-Evidence Decomposition

## Goal

Determine whether early high-frequency predictions are genuine beliefs or default outputs masking already-present sample-specific evidence.

## Reference priors

Estimate three references separately:

### A. Full-model Gaussian reference

Feed matched Gaussian states at each time:

\[
q_t^{gauss}(v)=\mathbb E_{\tilde z\sim \mathcal N(\mu_t,\Sigma_t)}[p_\theta(v\mid \tilde z,t)]
\]

Use mean/covariance matched to real states where possible.

### B. Cross-sequence state-swap reference

For each target sequence, replace its state with another sequence's state while retaining the same time and position index.

This preserves realistic state statistics while removing sample-specific correspondence.

### C. Context-shuffled reference

Keep the target position state fixed but shuffle non-target positions within or across sequences.

This estimates the contribution of generic lexical/contextual priors.

## Metrics

For every position and time:

- Raw true-token rank.
- Residual true-token rank after subtracting each reference.
- Raw and residual true-vs-default margin.
- Fraction where residual rank improves before raw top-1 becomes correct.
- Fraction where raw top-1 is the null-mode token but residual top-1 is sample-specific.
- KL divergence between real posterior and reference posterior.

## Decision rule

Support for “prior masking” requires:

1. Residual true-token rank improves significantly before raw top-1 crossing.
2. The early default token loses most of its advantage after reference subtraction.
3. The effect is present for both ELF and LangFlow, even if its magnitude differs.

## Failure interpretation

- No residual improvement: the state truly lacks sample-specific token evidence.
- Residual evidence exists only in ELF: the mechanism may be architecture-specific.
- Gaussian and shuffled references disagree strongly: the effect is contextual rather than purely geometric.

## Script

```text
experiments/phase_transition/estimate_reference_prior.py
experiments/phase_transition/analyze_prior_subtraction.py
```

## Output

```text
prior_reference_<type>.npz
prior_subtraction_metrics.json
prior_subtraction_curves.pdf
```

---

# 4. EXP-PT2 — True-vs-Default Margin Trajectories

## Goal

Test whether the visible token “cliff” is a discrete boundary crossing produced by a smoothly accumulating margin.

## Default competitor definition

Use three definitions and report all:

1. Earliest-time native top-1.
2. Modal token over the first 10% of the trajectory.
3. Reference-prior top-1 from EXP-PT1.

## Measurements

Track:

\[
m_i^{raw}(t)
\]

\[
m_i^{res}(t)
\]

plus:

- True-token rank.
- Top-1/top-2 margin.
- Entropy.
- Independent-probe score.
- Native top-1 identity.

## Transition analysis

Fit both:

- Monotone isotonic trend.
- Piecewise-linear change-point model with 1–3 breakpoints.

Compute:

- \(\tau^e,\tau^b,\tau^s\).
- Pre-crossing slope.
- Post-crossing slope.
- Number of zero crossings.
- Distance between the detected margin zero and the observed top-1 switch.

## Decision rule

The boundary-crossing account is supported when:

- Residual margin grows before native top-1 changes.
- Native switch occurs close to margin zero.
- Most trajectories have one dominant crossing rather than arbitrary discontinuities.
- KD primarily reduces \(\tau^b-\tau^e\), rather than substantially reducing \(\tau^e\).

## Failure taxonomy

- **No emergence**: residual margin never becomes positive.
- **Wrong-mode accumulation**: a wrong token's residual margin grows instead.
- **Stalled ambiguity**: true-token residual margin improves but never reaches top-1.
- **Premature crossing**: true token becomes top-1 and later leaves.
- **Multiple revision**: more than two stable zero crossings.
- **Endpoint-only correction**: crossing occurs only in the final grid interval.

## Script

```text
experiments/phase_transition/analyze_margin_trajectory.py
experiments/phase_transition/classify_transition_failures.py
```

---

# 5. EXP-PT3 — Velocity Alignment and Integrated Evidence

## Goal

Identify whether the vector field provides a weak correct drift before the native decoder shows a meaningful token.

## Clean-direction alignment

For oracle states:

\[
a_i^{clean}(t)=\frac{\langle v_\theta(z_t,t),x^\star-z_t\rangle}{\|v_\theta(z_t,t)\|\|x^\star-z_t\|}
\]

## Token-discriminative direction

Construct a direction separating the true token from the early default token.

Use two variants:

### A. Centroid direction

\[
u_{y,f}=\frac{c_y-c_f}{\|c_y-c_f\|}
\]

Centroids must be estimated on an independent split and, for cross-checkpoint comparisons, aligned or defined in a shared external space.

### B. Probe direction

For a trained linear probe \(W\):

\[
u_{y,f}=\frac{W_y-W_f}{\|W_y-W_f\|}
\]

## Instantaneous evidence velocity

\[
a_i^{tok}(t)=\langle v_\theta(z_t,t),u_{y,f}\rangle
\]

## Integrated evidence

\[
C_i(t_k)=\sum_{j<k}\Delta t_j\,a_i^{tok}(t_j)
\]

Test whether \(C_i(t)\) predicts the residual-margin trajectory and crossing time.

## Controls

- Random token direction.
- Frequency-matched wrong token.
- Orthogonalized random direction.
- Same-token direction from another sequence.
- Oracle versus free-running states.

## Decision rule

Support for gradual information accumulation requires:

- Positive clean-direction alignment before native top-1 becomes meaningful.
- Token-direction integral predicts later margin growth.
- Random/frequency-matched controls do not show the same alignment.
- Free-running failures correspond to reduced or misdirected integrated evidence.

## Script

```text
experiments/phase_transition/probe_velocity_alignment.py
```

---

# 6. EXP-PT4 — Causal Context-Source Ablation

## Goal

Determine where the earliest sample-specific evidence comes from: the target position's noisy signal, local context, or global context.

## Interventions

At each time, keep the target position state unchanged and intervene only on other positions.

1. **Full context**.
2. **No context**: mask all non-target positions.
3. **Local windows**: radius \(r\in\{1,2,4,8,16\}\).
4. **Global-only**: mask local radius 8 but keep distant context.
5. **Within-sequence shuffle**.
6. **Cross-sequence context swap**.
7. **Oracle-clean context substitution**.
8. **Wrong but grammatically matched context substitution**.

For ELF, preserve tensor shapes and attention masks.  
For LangFlow, also preserve the native self-conditioning pathway.

## Metrics

For target positions:

- Shift in residual margin:
  \[
  \Delta m_i^{res}(t)
  \]
- Shift in evidence-emergence time:
  \[
  \Delta \tau_i^e
  \]
- Shift in boundary-crossing time:
  \[
  \Delta \tau_i^b
  \]
- Change in velocity token alignment.
- Function-word versus content-word effects.
- Frequency and surprisal matched analyses.

## Decision rule

- Local-window recovery comparable to full context: evidence is predominantly local.
- Clean-context substitution advances \(\tau^e\): context causally supplies early evidence.
- Cross-sequence swap destroys early evidence: evidence is sample-specific rather than a global prior.
- No-context condition unchanged: early evidence mainly comes from the target position's noisy state.

## Script

```text
experiments/phase_transition/intervene_context.py
```

---

# 7. EXP-PT5 — Decoder-Bias Intervention

## Goal

Causally test whether the visible top-1 cliff is caused by a decoder boundary rather than sudden representation formation.

## Diagnostic logit intervention

Construct:

\[
\ell'_t(v)=\ell_t(v)-\lambda\log q_t(v)
\]

with:

\[
\lambda\in\{0,0.25,0.5,0.75,1.0\}
\]

Also test direct controlled offsets:

\[
\ell'_t(y)=\ell_t(y)+\beta
\]

\[
\ell'_t(f)=\ell_t(f)-\beta
\]

for small \(\beta\).

Do not feed these logits back into the trajectory in the first version. This is a readout-only diagnostic.

## Measurements

- Shift in \(\tau^b\).
- No change in independent-probe accuracy.
- No change in hidden states or velocity.
- Number of positions whose crossing time changes without a change in residual evidence.

## Decision rule

If small prior/bias corrections strongly move \(\tau^b\) while \(\tau^e\) stays fixed, the apparent cliff is largely a decoder-boundary phenomenon.

If crossing time barely moves, representation formation rather than readout bias is the main bottleneck.

## Script

```text
experiments/phase_transition/intervene_decoder_bias.py
```

---

# 8. EXP-PT6 — Local Stability Around the Crossing

## Goal

Determine whether a top-1 crossing is a robust phase transition or a fragile ranking fluctuation.

## Full-state branching

For each position, select checkpoints around:

\[
\tau^b-\Delta,\quad \tau^b,\quad \tau^b+\Delta,\quad \tau^s
\]

Branch the **complete state**:

\[
S_t=(z_t,s_t)
\]

not \(z_t\) alone.

## Perturbation directions

For normalized relative L2 magnitude:

\[
\eta\in\{10^{-4},3\times10^{-4},10^{-3},3\times10^{-3},10^{-2}\}
\]

apply:

1. Isotropic random direction.
2. True-vs-default token direction.
3. Orthogonal random direction.
4. Empirical rollout-drift direction:
   \[
   z_t^{roll}-z_t^{oracle}
   \]
5. Context-only perturbation.
6. Target-position-only perturbation.

Use:

\[
\delta=\eta\|z\|_2\frac{u}{\|u\|_2}
\]

## Metrics

- Immediate top-1 flip probability.
- Final-token flip probability.
- Pairwise branch agreement.
- Modal outcome probability.
- Branch entropy.
- Local gain:
  \[
  \frac{\|\Phi(z+\delta)-\Phi(z)\|}{\|\delta\|}
  \]

## Decision rule

A robust transition should show a sharp stability increase near or after \(\tau^b\).  
If top-1 becomes correct long before perturbation stability improves, readability precedes commitment.

## Script

```text
experiments/phase_transition/branch_around_transition.py
```

---

# 9. EXP-PT7 — Paired Oracle vs Free-Running Phase Alignment

## Goal

Test whether free-running failure occurs because the sampler fails to enter the same phase progression seen on the oracle corridor.

## Paired construction

For each generated final sample:

1. Save the initial noise \(\epsilon\).
2. Save the complete reverse trajectory \(S_t^{roll}\).
3. Encode the final generated sequence into a clean endpoint \(x^{final}\).
4. Construct a paired oracle path using the same \(\epsilon\):
   \[
   z_t^{oracle}=\alpha_t x^{final}+\sigma_t\epsilon
   \]

This controls the final token target and initial noise.

## Metrics

For both paths:

- \(\tau^e,\tau^b,\tau^s\).
- Raw and residual margin.
- State distance:
  \[
  d_t=\|z_t^{roll}-z_t^{oracle}\|
  \]
- Representation similarity.
- Velocity disagreement.
- Self-conditioning disagreement.
- Distance to the oracle state manifold estimated by nearest neighbors or a discriminator.

## Key analysis

Predict generation failure from:

- Early state distance.
- Delay in \(\tau^e\).
- Delay in \(\tau^b\).
- Wrong-mode accumulation.
- Velocity misalignment.

## Causal interpolation

At selected times:

\[
z_t(\lambda)=(1-\lambda)z_t^{roll}+\lambda z_t^{oracle}
\]

with:

\[
\lambda\in\{0,0.25,0.5,0.75,1\}
\]

Continue the same solver with matched future randomness.

If moving toward the paired oracle state improves final stability or quality, the oracle-rollout gap is causally relevant rather than merely descriptive.

## Script

```text
experiments/phase_transition/compare_oracle_rollout.py
experiments/phase_transition/interpolate_oracle_rollout.py
```

---

# 10. EXP-PT8 — Controlled Minimal-Pair Evidence Sources

## Goal

Create cases where the source of lexical evidence is known, allowing causal attribution of the transition.

## Dataset

Construct or curate natural minimal pairs differing in one cue:

- Subject–verb agreement.
- Named entity location.
- Number agreement.
- Local collocation.
- Negation.
- Semantic role.
- Function-word versus content-word target.

Example form:

```text
The keys to the cabinet [are/is] ...
The key to the cabinets [is/are] ...
```

The target token remains at the same position, while one controlled cue changes.

Prefer naturally occurring sentences with a single edited cue. Filter with an external LM to remove severely unnatural examples.

## Experiments

For each pair:

- Run the same fixed-noise oracle path.
- Compare evidence-emergence and crossing times.
- Swap only the causal cue.
- Keep all other positions identical.
- Measure whether the token-discriminative velocity changes immediately after the cue becomes available.

## Decision rule

A valid evidence mechanism should respond predictably to the controlled cue:

- Correct cue accelerates true-token evidence.
- Incorrect cue redirects evidence toward the alternative token.
- Unrelated edits have much smaller effects.

## Script

```text
experiments/phase_transition/build_minimal_pairs.py
experiments/phase_transition/probe_minimal_pairs.py
```

---

# 11. EXP-PT9 — Cross-Time Evidence-Direction Transfer

## Goal

Determine whether evidence accumulates along a stable token-discriminative direction or is repeatedly re-encoded across time.

## Method

Train a linear probe or direction at time \(t_a\) and evaluate at \(t_b\).

Construct a transfer matrix:

\[
M_{a,b}=\operatorname{Acc}(P_{t_a}(h_{t_b}))
\]

Do the same for:

- Raw states.
- Predicted-clean states.
- Native hidden states.
- Prior-subtracted logits.

## Interpretation

- Strong upper-triangular transfer: early evidence directions persist and strengthen.
- Strong diagonal only: evidence is time-specific and repeatedly reparameterized.
- Late probes work on early states but not vice versa: early states contain weak versions of later evidence.
- KD improving transfer rather than per-time probe accuracy: KD stabilizes the evidence coordinate system.

## Script

```text
experiments/phase_transition/probe_cross_time_transfer.py
```

---

# 12. EXP-PT10 — Transition Failure Predictors

## Goal

Identify why the phase transition sometimes fails.

## Failure categories

Use EXP-PT2 and EXP-PT6 to label:

1. Successful monotonic transition.
2. Wrong-mode accumulation.
3. Stalled ambiguity.
4. Premature crossing and release.
5. Multiple revision.
6. Endpoint-only correction.
7. Free-running drift failure.
8. Perturbation-fragile crossing.

## Predictors

- Training token frequency.
- Contextual surprisal.
- POS/function-content label.
- Token length/subtoken fragmentation.
- Position.
- Prior-mode advantage.
- Initial velocity alignment.
- Local context strength.
- Oracle-rollout state distance.
- Self-conditioning norm.
- Jacobian/local gain estimate.

## Model

Use a simple interpretable multinomial logistic regression first.  
Use sequence-grouped cross-validation and report calibrated probabilities.

## Purpose

This experiment turns the stages into a falsifiable failure taxonomy and tells us which cases a future method must repair.

## Script

```text
experiments/phase_transition/analyze_failure_predictors.py
```

---

# 13. Unified ELF / LangFlow implementation interface

Create a small adapter layer:

```python
class FlowModelAdapter:
    def encode_clean(self, token_ids, attention_mask):
        ...

    def make_oracle_state(self, clean_state, epsilon, t):
        ...

    def forward_state(self, state, sc_state, t, capture_hidden=False):
        """
        Returns:
            logits
            predicted_clean
            velocity
            hidden_states
            next_sc_state
        """
        ...

    def solver_step(self, state, sc_state, t, t_next):
        ...

    def native_logsnr(self, t):
        ...

    def full_state_clone(self, state, sc_state):
        ...
```

Implement:

```text
adapters/elf_adapter.py
adapters/langflow_adapter.py
```

This prevents model-specific code from contaminating the analysis logic and makes the “phase transition” idea genuinely cross-architecture.

---

# 14. Recommended execution order

## P0: establish the mechanism

1. EXP-PT1 Prior-to-Evidence Decomposition.
2. EXP-PT2 Margin Trajectories and Transition Times.
3. EXP-PT3 Velocity Alignment.
4. EXP-PT5 Decoder-Bias Intervention.

These four establish whether evidence exists early, accumulates continuously, and becomes visible through a decision-boundary crossing.

## P1: locate the information source and failure mode

5. EXP-PT4 Context-Source Ablation.
6. EXP-PT6 Local Stability Around Crossing.
7. EXP-PT10 Failure Predictors.

## P1: connect oracle analysis to real generation

8. EXP-PT7 Paired Oracle vs Free-Running Phase Alignment.

## P2: controlled causality and generalization

9. EXP-PT8 Minimal Pairs.
10. EXP-PT9 Cross-Time Transfer.
11. Repeat P0/P1 on LangFlow using matched log-SNR.

---

# 15. Minimum viable implementation

The smallest package that can already support a strong claim is:

### MVP-A

- EXP-PT1 prior subtraction.
- EXP-PT2 margin trajectories.
- EXP-PT5 decoder-bias intervention.

This tests:

> Early high-frequency tokens are default readouts that mask gradually accumulating sample-specific evidence.

### MVP-B

Add EXP-PT3 velocity alignment.

This tests:

> The vector field supplies weak correct drift before native top-1 becomes meaningful.

### MVP-C

Add EXP-PT7 oracle-versus-rollout comparison.

This tests:

> Free-running failure occurs when the trajectory does not maintain the same evidence-accumulation process as the oracle path.

---

# 16. Paper-level hypotheses

## H1: Prior masking

Early high-frequency predictions are primarily caused by default prior or geometric bias rather than the absence of sample-specific evidence.

## H2: Continuous accumulation

True-token residual evidence and token-aligned velocity increase before native top-1 crossing.

## H3: Boundary crossing

The apparent commitment cliff is largely explained by a continuous margin crossing a discrete decoder boundary.

## H4: Non-guaranteed stabilization

Boundary crossing is not necessarily stable; some positions remain perturbation-sensitive or later revise.

## H5: Free-running failure

Actual generation fails when rollout states disrupt or redirect the oracle evidence-accumulation process.

## H6: Cross-architecture generality

ELF and LangFlow share the prior-to-evidence-to-boundary framework, while differing in the coupling between continuous evidence, categorical readout, and state update.
