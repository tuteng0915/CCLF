# Technical Review and Hypothesis-Driven Experimental Roadmap

## For: *Understanding and Controlling Lexical Commitment in Language Flows*

### Scope

This document reviews the current draft as a scientific argument rather than as a writing sample. The main questions are:

1. Which observations are already reliable?
2. Which conclusions currently exceed what the experiments establish?
3. Which alternative hypotheses remain unresolved?
4. What exact experiments should be run next?
5. How should the project be narrowed into a defensible paper?

The review focuses primarily on **ELF** and **LangFlow**, because they expose fundamentally different interfaces between continuous states and tokens. MDLM and DUO are useful controls, but they should not determine the central scientific framing.

---

# 1. Executive diagnosis

The draft contains a genuinely strong empirical signal:

> ELF's intermediate predicted clean representation is much more recoverable by token-oriented readouts than its final-only decoding design would lead one to expect, and its nonlinear decode branch can correct a large fraction of errors made by the linear projection.

The strongest current evidence is not yet the proposed “commit–release–recommit” story. It is the following combination:

1. A large and persistent gap between ELF's linear readout and decode branch.
2. A meaningful interpolation direction from the denoiser output toward the decode-branch output.
3. A zero-training intervention—decode-branch self-conditioning—that substantially improves generation in several sampling regimes.
4. A clear demonstration that representation quality, linear readability, nonlinear decodability, and generation quality are not the same thing.

Those are real contributions.

However, the current paper often interprets readout behavior as trajectory dynamics, and interprets geometric quantities as semantic or causal structure without adequate controls. The most serious issues are:

- Most probes are applied to **forward-corrupted ground-truth states**, not actual reverse-generation trajectories.
- The ELF–LangFlow comparison uses **ELF predicted clean states versus LangFlow noisy inputs**, which is not a valid representational comparison.
- The proposed geometric nearest-token metric is not decoder-independent; in a tied-weight model it is largely a **normalized, bias-free version of the same vocabulary head**.
- The anchor residual is not an orthogonal or identifiable “contextual residual.” Its norm cannot be interpreted as the fraction of information not explained by lexical identity.
- The claim that timing differences are “explained by the SNR schedule” is not established by comparing models at equal nominal \(t\).
- Sampling timesteps from \(q(t)\propto |dG/dt|\) changes the training objective unless importance correction is applied.
- The decode-self-conditioning improvement is confounded by extra computation and nonlinear refinement.
- The draft proposes too many training components before the underlying mechanism has been isolated.

The immediate goal should therefore be:

> Replace the current single story with a controlled set of competing hypotheses about measurement, decoder geometry, learned priors, latent evidence formation, contextual bootstrapping, and self-conditioning feedback.

Only after these hypotheses are separated should the paper commit to a mechanism and training method.

---

# 2. Recommended notation

The current draft uses \(x\) for the clean continuous representation and \(p\) for the vocabulary distribution. For the next experimental phase, use notation that cleanly separates the objects under study.

For position \(i\):

- \(y_i\): clean or reference token.
- \(z_{t,i}\): continuous state at noise level or diffusion time \(t\).
- \(\hat z_{0|t,i}\): model prediction of the clean continuous representation from \(z_t\).
- \(x^{\mathrm{lin}}_{t,i}\in\Delta^{|V|-1}\): vocabulary distribution from the linear/native projection.
- \(x^{\mathrm{dec}}_{t,i}\in\Delta^{|V|-1}\): vocabulary distribution from ELF's nonlinear decode branch.
- \(s_t\): self-conditioning state. A checkpointed dynamical state is therefore generally
  \[
  S_t=(z_t,s_t),
  \]
  not merely \(z_t\).
- \(E\in\mathbb R^{|V|\times d}\): token embedding matrix.
- \(a_t=E^\top x_t\): expected token embedding under a vocabulary distribution.

A critical warning:

\[
a_t=E^\top x_t
\]

is a vocabulary-weighted barycenter. It is **not** an orthogonal projection of \(z_t\) or \(\hat z_{0|t}\) onto a lexical subspace unless an explicit projection operator is defined and justified.

This notation should be used consistently in all future experiment specifications.

---

# 3. What the current draft does well

## 3.1 It distinguishes belief, proposal, and algorithmic commitment

The appendix separates:

\[
z_t\rightarrow x_t\rightarrow \hat y_t\rightarrow m_t,
\]

where a token can be readable without being written back into the generation state. This distinction is conceptually correct and especially important for ELF, whose intermediate lexical readouts are diagnostic rather than explicit generation states.

This framework should remain. The main text, however, must follow it more strictly. Phrases such as “the model committed to the wrong token” should be replaced by “the selected readout produced a stable wrong proposal” unless that proposal actually affected later sampling.

## 3.2 The linear-branch versus decode-branch comparison is valuable

The large ELF gap between direct linear projection and the decode branch is one of the cleanest results in the draft. The reported gap of roughly 17–28 percentage points over a wide interval suggests that:

- the denoiser output contains information not well extracted by the linear head;
- the nonlinear decode branch performs genuine correction or contextual reinterpretation;
- the final interface is not a passive rounding operation.

This is stronger than merely reporting an early nearest-token accuracy curve.

## 3.3 The interpolation experiment is a useful causal precursor

Interpolating

\[
\tilde z(\gamma)=(1-\gamma)\hat z_{0|t}+\gamma z^{\mathrm{dec}}_t
\]

and observing monotonic accuracy improvement is useful. It shows that the decode residual is aligned with a direction that improves the selected token readout.

It does not yet prove that the residual is the model's natural correction direction during reverse dynamics, but it is a legitimate and promising intervention.

## 3.4 Fixed-noise position tracking is better than independent-time aggregation

The draft correctly distinguishes the main aggregate probe from the smaller fixed-noise study. Tracking the same underlying forward noise across times is necessary for any position-level transition analysis.

The limitation is that this still tracks a family of **teacher-forced forward-corruption states**, not necessarily the states visited by a reverse sampler. That distinction must be made explicit.

## 3.5 The paper reports failures rather than hiding them

The non-monotonic effects across ODE/SDE step counts, the instability of the KD checkpoint, and the catastrophic interaction between SAR and decode self-conditioning are all reported. This is scientifically useful.

In particular, the SAR mode-collapse case demonstrates that low Gen.PPL alone can be meaningless. This observation should influence the evaluation protocol throughout the paper.

## 3.6 The zero-training decode-self-conditioning result is potentially strong

A pretrained checkpoint is modified at inference time, and the change materially improves several generation settings. This is a stronger result than an unvalidated collection of proposed losses.

The result should be retained, but its mechanism must be tested against compute-matched and refinement-matched controls.

---

# 4. Critical problems in the current scientific argument

## 4.1 Forward-corruption probes are being described as reverse-trajectory dynamics

The appendix states that the main probing experiments:

- take clean OpenWebText sequences;
- apply noise at a selected \(t\);
- perform a single model forward pass;
- do not execute reverse denoising steps.

This measures:

\[
x_t=f_\theta(z_t,t,s),
\qquad
z_t\sim q(z_t\mid y),
\]

under a teacher-forced corrupted-data distribution.

It does **not** measure:

\[
S_{t_0}\rightarrow S_{t_1}\rightarrow \cdots\rightarrow S_{t_K}
\]

along model-generated reverse trajectories.

Therefore, the following claims are currently too strong when based on the fixed-\(t\) probe:

- “the trajectory commits”;
- “the trajectory releases”;
- “wrong positions are stuck”;
- “the model self-corrects at this step”;
- “the final decoder rescues errors left by the trajectory”;
- “the commitment cliff is traversed too coarsely by the sampler.”

The same underlying token position can be evaluated at several forward noise levels, but the corresponding states are not generated from one another by the learned reverse dynamics.

### Required correction

Use two explicitly named protocols:

1. **Forward-noise denoising probe**
   \[
   z_t\sim q(z_t\mid y).
   \]
   This studies denoising competence under controlled SNR.

2. **Reverse-generation trajectory probe**
   \[
   S_{k+1}=\Phi_\theta(S_k,t_k,t_{k+1}).
   \]
   This studies the actual generation dynamics.

Do not pool or narrate these protocols as if they were equivalent.

---

## 4.2 The current ELF–LangFlow geometric comparison is invalid

The draft computes ELF geometry on the model's predicted clean contextual representation \(\hat z_{0|t}\), but computes LangFlow geometry on the noisy input \(z_t\).

That comparison mixes:

- a denoised model output for ELF;
- a corrupted model input for LangFlow.

The statement that LangFlow has “no analog” of ELF's output is too strong. LangFlow exposes at least:

- a native vocabulary distribution \(x_t\);
- the expected token embedding
  \[
  a_t=E^\top x_t;
  \]
- a model-induced velocity or clean target used to define the flow update.

These are not identical to ELF's contextual clean prediction, but they are more meaningful comparison objects than raw noisy \(z_t\).

### Required correction

Compare models at matched functional levels:

| Functional level | ELF | LangFlow |
|---|---|---|
| Current continuous state | \(z_t\) | \(z_t\) |
| Model-native token belief | linear/decode diagnostic \(x_t\) | native \(x_t\) |
| Token-induced continuous target | \(E^\top x_t\) | \(E^\top x_t\) |
| Model denoising output | \(\hat z_{0|t}\) | velocity/clean target if explicitly available |
| Final prediction | decode branch | native final posterior |

Cross-model conclusions should be based on information recovery, posterior quality, and matched-SNR behavior—not on a raw nearest-centroid score applied to architecturally different tensors.

---

## 4.3 The geometric nearest-token metric is not decoder-independent

The draft defines:

\[
G(t)
=
\Pr\left[
\arg\max_v
\frac{\hat z_{0|t}^{\top}E_v}
{\|\hat z_{0|t}\|\|E_v\|}
=y
\right].
\]

In a tied-weight model, this is closely related to the vocabulary projection:

\[
\ell_v=E_v^\top \hat z_{0|t}+b_v.
\]

The geometric metric mainly changes the head by:

- removing or ignoring the bias term;
- normalizing row norms;
- replacing dot product with cosine similarity.

It is therefore better described as a **cosine-normalized vocabulary readout**, not as a decoder-free measurement of latent token identity.

The observed gap between \(G(t)\) and linear top-1 may be caused by:

- output-row norm;
- output bias;
- latent norm;
- anisotropy;
- temperature;
- contextual-state distribution shift.

### Required correction

Replace the binary “geometric versus distributional commitment” framing with a readout ladder:

1. raw tied linear head;
2. bias-free linear head;
3. row-normalized head;
4. cosine head;
5. timestep-calibrated affine probe;
6. nonlinear decode branch.

The differences between adjacent levels identify which part of the interface causes the accuracy gap.

---

## 4.4 Low entropy at high noise is not commitment

The draft calls a position committed when entropy is below 0.1, even in a regime where almost every prediction is wrong and dominated by frequent tokens.

This is precisely the situation in which entropy is least reliable:

> Low entropy can mean strong evidence for a token, but it can also mean a collapsed or badly calibrated decoder prior.

A model that predicts “the” with probability 0.99 for every position has low entropy and zero token-specific information.

### Required correction

Never define commitment from entropy alone.

At minimum, report:

- entropy;
- sample-specific information above a default prior;
- final-token rank;
- perturbation stability;
- agreement across independent readouts;
- actual downstream influence on future dynamics.

A better early-stage diagnostic is:

\[
r_t(v)=\log x_t(v)-\log q_t(v),
\]

where \(q_t\) is the output after instance information has been destroyed.

---

## 4.5 The anchor residual is not an identifiable contextual residual

The draft defines:

\[
r_t=\hat z_{0|t}-E^\top x_t
\]

and interprets a large normalized residual as contextual information beyond lexical identity.

This interpretation is not established.

Problems:

1. \(E^\top x_t\) is not an orthogonal projection.
2. \(r_t\) need not be orthogonal to the token embedding span.
3. The decomposition is not unique.
4. The two terms can have strongly different scales.
5. A convex combination of token embeddings often has a smaller norm because of cancellation.
6. Contextual encoder states and static input embeddings may live in geometrically different distributions even when weights are tied elsewhere.
7. A residual norm near \(\|\hat z_{0|t}\|\) does not mean “nearly all non-lexical information.”
8. Large residual magnitude does not show that the residual contains syntax or semantics.

The sentence that \(\rho(t)\) “measures the fraction of \(\hat z_t\) not explained by lexical belief” is mathematically incorrect.

### Required correction

Rename it **anchor mismatch ratio** unless an actual projection is defined.

To claim contextual information, run direct tests:

- predict token identity from the residual;
- predict syntax or semantic attributes from the residual;
- remove the residual and test token decoding;
- swap residuals between positions or sequences;
- project onto an explicitly learned lexical subspace;
- compare with norm- and covariance-matched random residuals.

---

## 4.6 The SNR-schedule conclusion is overclaimed

The draft compares models at equal nominal \(t\), observes very different SNR values, and concludes that the timing gap is explained by schedule rather than architecture or noise type.

This establishes that equal \(t\) is not a fair comparison. It does not establish that schedule is the causal explanation.

Architecture, target representation, token supervision, self-conditioning, and decoder design remain different.

### Required correction

Plot all curves against:

- nominal \(t\);
- log-SNR;
- corruption fraction or Bayes-optimal recoverability;
- model-native posterior entropy.

Then perform at least one intervention:

- evaluate the same model under a remapped sampling schedule;
- construct matched-SNR forward-corruption inputs;
- compare outputs at equal empirical clean-signal strength;
- replace only the schedule while holding model weights and state construction fixed where feasible.

The defensible conclusion before intervention is:

> Much of the apparent timing difference at matched \(t\) is confounded by schedule.

Not:

> The timing difference is explained by schedule.

---

## 4.7 Probe-proportional timestep sampling changes the objective

The draft proposes:

\[
q(t)\propto\left|\frac{dG}{dt}\right|
\]

and states that this “does not alter the loss values, only the frequency” with which regions are visited.

Changing the sampling distribution changes the expectation:

\[
\mathbb E_{t\sim p(t)}[L(t)]
\neq
\mathbb E_{t\sim q(t)}[L(t)].
\]

To preserve the original objective, one must use importance weights:

\[
\mathbb E_{t\sim q(t)}
\left[
\frac{p(t)}{q(t)}L(t)
\right].
\]

Without the weight, the method deliberately defines a reweighted training objective. That may be useful, but it must be stated correctly.

There is an additional circularity: \(G(t)\) is derived from the current model and current readout. Oversampling regions where that diagnostic changes fastest does not imply those regions have the most useful gradients.

### Required correction

Treat three variants separately:

1. original sampling \(p(t)\);
2. importance-sampled but unbiased \(q(t)\) with \(p/q\);
3. deliberately reweighted objective under \(q(t)\).

Compare gradient variance, denoising loss, token metrics, and generation quality.

---

## 4.8 Decode-self-conditioning is confounded by extra nonlinear computation

Decode self-conditioning adds an extra pass through a strong nonlinear branch. Improvements may arise because:

- the branch extracts lexical information;
- the branch performs generic representation refinement;
- additional compute improves the estimate;
- temporal smoothing stabilizes self-conditioning;
- the intervention changes effective solver behavior.

The current results show usefulness, but not the claimed mechanism.

### Required correction

Use compute-matched controls:

- one extra denoising-mode pass;
- one extra decode-mode pass with logits detached or shuffled;
- an MLP/refinement block matched in parameter and FLOP count;
- the decode branch evaluated with wrong timestep;
- a random orthogonal residual with matched norm;
- previous-step versus current-step decode estimates;
- a teacher-forced oracle clean estimate as an upper bound.

If only the semantically correct decode branch helps, the correction-information story becomes much stronger.

---

## 4.9 Generation evaluation is not yet statistically reliable

The draft reports very large changes in Gen.PPL when the sample count changes, for example an 8-step baseline shifting from roughly 942 to 872. This is too large to dismiss casually as sample-size variance.

Further, the SAR failure yields Gen.PPL around 2–3 under obvious mode collapse. This directly proves that Gen.PPL alone is not a quality metric.

### Required correction

For every generation table report:

- at least three generation seeds;
- bootstrap confidence intervals;
- MAUVE;
- distinct-1/2/3;
- repetition rate;
- self-BLEU or semantic diversity;
- average length and truncation rate;
- token unigram KL to the reference corpus;
- human evaluation for the final main variants.

A result should not be labeled improved when PPL falls but diversity or distributional coverage collapses.

---

## 4.10 The paper currently contains too many speculative method components

The draft introduces or discusses:

- scheduled CE;
- cosine anchor loss;
- decode-teacher KD;
- embedding-space consistency;
- probe-proportional sampling;
- position-adaptive KD masking;
- decode self-conditioning;
- SAR inference.

Most training-time components are not evaluated. This weakens the paper rather than strengthening it.

The scientific paper should first establish one mechanism and one intervention.

### Recommended narrowing

Keep as the primary method candidate:

- decode self-conditioning, if compute-matched controls support the correction hypothesis.

Keep as a secondary training candidate:

- one distillation or consistency objective derived directly from the validated mechanism.

Move to appendix or future work:

- SAR;
- probe-proportional sampling;
- general anchor schedules;
- position-adaptive masks;
- multi-stage commit–release–recommit training.

---

# 5. Competing hypothesis map

The next experiments should distinguish the following hypotheses.

## H-A: Intermediate token behavior is a readout artifact

The continuous state may not be lexically committed; a final or mismatched decoder may simply output frequent tokens when applied off-distribution.

## H-B: High-frequency domination is caused by static decoder geometry

Output bias, row norms, latent mean direction, or anisotropy may make common tokens win on weakly informative states.

## H-C: The network has learned a time-conditioned default lexical prior

Even after static geometry is controlled, the full model may intentionally emit a frequency-dominated distribution under high noise.

## H-D: Final-token evidence is already present in \(z_t\), but native readout masks it

A calibrated or independent probe may recover \(y\) before the native head does.

## H-E: Token identity is genuinely formed during denoising

Neither independent probes nor debiased readouts can recover \(y\) early; token-specific information appears only later.

## H-F: Information emerges coarse-to-fine

Syntactic or semantic class becomes recoverable before exact lexical identity.

## H-G: Token identity is created through contextual bootstrapping

A position becomes predictable because other positions become informative, not only because its local latent becomes cleaner.

## H-H: Self-conditioning creates a feedback loop

Early predictions affect later states, so a default prior can either stabilize useful structure or reinforce wrong predictions.

## H-I: Readability and dynamical commitment are distinct

A token may be decodable while small perturbations still change the final outcome, or may be dynamically fixed while the native decoder remains uncertain.

## H-J: ELF's decode branch performs token-specific correction

The decode branch is not merely extra compute; it uses contextual and lexical structure to move states toward better token basins.

These hypotheses are compatible in combinations. The purpose of the experiments is to estimate their relative contribution.

---

# 6. Detailed experimental roadmap

Each experiment below is written so it can later be expanded into an implementation spec.

---

## EXP-00 — Probe object and code-path audit

**Priority:** P0, blocking  
**Models:** ELF, LangFlow

### Hypothesis

Some current metrics are being computed on tensors that do not have comparable meanings across models or even across code branches.

### Required audit table

For every stored tensor, document:

- tensor name in code;
- mathematical symbol;
- shape;
- whether it is input, hidden state, predicted clean state, velocity, logits, distribution, or self-conditioning;
- whether it is before or after LayerNorm;
- whether gradients are active;
- whether it is used by the sampler;
- whether it exists during training, generation, or only probing;
- exact decoder applied;
- exact timestep and mode flags.

### Required ELF paths

At minimum identify:

1. noisy state \(z_t\);
2. preliminary prediction;
3. self-conditioned main prediction \(\hat z_{0|t}\);
4. direct linear logits;
5. decode-mode hidden state;
6. decode-mode logits;
7. self-conditioning state actually fed into the next sampler step.

### Required LangFlow paths

At minimum identify:

1. noisy state \(z_t\);
2. transformer hidden state before vocabulary projection;
3. native logits;
4. native distribution \(x_t\);
5. expected embedding \(E^\top x_t\);
6. velocity or clean-target quantity used in the ODE;
7. self-conditioning state.

### Deliverables

- one Markdown table;
- one computational graph per model;
- unit tests checking tensor identity and shape;
- a stored example batch containing all audited tensors.

### Acceptance condition

No cross-model plot may be generated until every plotted quantity has a documented functional meaning.

---

## EXP-01 — Forward-noise probe versus actual reverse trajectory

**Priority:** P0, blocking  
**Models:** ELF first, then LangFlow

### Hypothesis

The current commitment phases measured on forward-corrupted ground-truth states may not occur on generated reverse trajectories.

### Protocol A: Forward-noise denoising probe

For each real sequence \(y\), noise seed \(\epsilon\), and time \(t\):

\[
z_t=\alpha_t z_{\mathrm{clean}}+\sigma_t\epsilon.
\]

Run one model evaluation. Record all readouts.

### Protocol B: Reverse-generation trajectory

Start from the model's sampling prior and run the full sampler. At every solver step save:

\[
S_k=(z_k,s_k)
\]

together with:

- native logits/distributions;
- predicted clean state;
- decode-branch output where applicable;
- solver update;
- current generated proposal;
- final generated sequence.

Do not replace \(s_k\) with zero when resuming or branching.

### Matching analysis

For each metric, plot:

- forward-probe curve;
- reverse-trajectory curve;
- difference curve;
- distribution across examples, not only the mean.

Metrics:

- entropy;
- top-1 and top-\(k\) agreement with the final generated token;
- agreement with ground truth for reconstruction experiments;
- cosine-normalized readout;
- decode-branch accuracy;
- token flip rate;
- branch stability.

### Key interpretations

- Similar curves: forward probes may approximate reverse dynamics.
- Early commitment only in forward probes: current story is teacher-forced denoising behavior.
- Early commitment only in reverse trajectories: self-conditioning or solver feedback creates it.
- Different non-monotonicity: the current “release” phase may be a probing artifact.

### Required wording after this experiment

Use “trajectory” only for Protocol B.

---

## EXP-02 — Correct LangFlow comparison objects

**Priority:** P0  
**Models:** LangFlow, ELF

### Hypothesis

LangFlow appears late only because the draft probes its noisy input rather than its native predicted distribution or induced denoising target.

### LangFlow quantities

At each \(t\), compute:

1. native posterior \(x_t\);
2. posterior top-\(k\) recovery;
3. posterior entropy;
4. prior-corrected posterior;
5. expected embedding
   \[
   a_t=E^\top x_t;
   \]
6. cosine and dot-product recovery from \(a_t\);
7. hidden-state probes before the vocabulary head;
8. any explicit clean-target or velocity-derived clean estimate.

### ELF quantities

Compute comparable functional quantities:

1. direct linear posterior;
2. decode-branch posterior;
3. expected embedding from each posterior;
4. predicted clean contextual state;
5. hidden-state probes.

### Comparison principle

Do not demand identical geometry. Compare:

- token information recoverability;
- calibration;
- entropy;
- final-token rank;
- perturbation stability;
- feedback into the sampler.

### Expected outcome

This experiment may show that LangFlow's native \(x_t\) becomes meaningful far earlier than its raw \(z_t\) aligns with token embeddings. If so, the current cross-model conclusion must be rewritten.

---

## EXP-03 — Matched-SNR and matched-information comparison

**Priority:** P0  
**Models:** ELF, LangFlow; MDLM/DUO optional

### Hypothesis

The apparent timing gap is partly caused by incomparable time parameterizations.

### Protocol

For each model, report curves against:

1. nominal \(t\);
2. log-SNR;
3. clean-signal coefficient \(\alpha_t\);
4. empirical corruption difficulty;
5. Bayes or oracle recoverability where computable.

Construct a shared log-SNR grid. For each grid point, generate model-specific states with matched log-SNR.

### Stronger intervention

Where feasible, evaluate one model under a remapped schedule that visits the same log-SNR values at different nominal times. The network should receive:

- the correct timestep corresponding to the state;
- a counterfactual mismatched timestep in a separate ablation.

This separates state information from the model's time-conditioned behavior.

### Metrics

- native posterior accuracy;
- calibrated probe accuracy;
- entropy;
- default-prior strength;
- branch stability;
- generation quality under schedule remapping.

### Decision rule

Only claim a schedule-caused timing difference if the gap substantially collapses at matched SNR or under a within-model schedule intervention.

---

## EXP-04 — Decoder geometry null model

**Priority:** P1  
**Models:** ELF, LangFlow

### Hypothesis

High-frequency early predictions arise from output-head geometry rather than denoising dynamics.

### Null inputs

At each \(t\), estimate the state distribution and create:

1. isotropic Gaussian states;
2. mean- and variance-matched Gaussian states;
3. covariance-matched Gaussian states;
4. shuffled real states;
5. position-shuffled states;
6. states with instance identity destroyed but global statistics preserved.

### Readout ladder

For every state, apply:

1. full linear head \(Wz+b\);
2. bias-free head \(Wz\);
3. row-normalized head;
4. cosine head;
5. latent-mean-subtracted head;
6. whitened-state head;
7. nonlinear decode branch where applicable.

### Per-token regression

For token \(v\), predict early win rate using:

\[
\log \pi_v,\quad
\|W_v\|,\quad
b_v,\quad
W_v^\top\mu_t,\quad
\text{frequency bucket},\quad
\text{token type}.
\]

Use partial regression or variance decomposition.

### Deliverables

- top-20 early tokens under every null;
- overlap with real early top tokens;
- \(R^2\) explained by geometry;
- frequency effect after controlling for geometry;
- accuracy change across the readout ladder.

### Interpretation

- Gaussian null reproduces the effect: geometry dominates.
- Full network adds substantial frequency bias beyond geometry: learned prior.
- Row normalization removes the effect: output norm is central.
- Bias removal removes the effect: vocabulary bias is central.

---

## EXP-05 — Estimate the learned default prior \(q_t\)

**Priority:** P1  
**Models:** ELF, LangFlow

### Hypothesis

The model has a time-dependent default distribution that is expressed when instance evidence is weak.

### Definition

Estimate:

\[
q_t(v)=
\mathbb E[
x_t(v)\mid \text{instance information destroyed}
].
\]

Do not use corpus unigram frequency as the only estimate.

### Destruction conditions

- batch-shuffled state;
- target position from sample A with context from sample B;
- pure matched Gaussian;
- zeroed local state with real context;
- real local state with shuffled context;
- timestep-only network input if technically possible.

### Analysis

Compare \(q_t\) to:

- corpus unigram distribution;
- decoder-only Gaussian output;
- average normal-model output;
- position-conditional unigram;
- token-frequency buckets.

Compute:

- KL divergence;
- Spearman correlation;
- top-token overlap;
- entropy;
- position dependence;
- timestep dependence.

### Key result

This determines whether “prior-dominated” means:

- corpus prior;
- decoder geometry;
- model-learned time-conditioned prior;
- contextual marginal prior.

---

## EXP-06 — Prior subtraction and frequency-barrier test

**Priority:** P1  
**Models:** ELF, LangFlow

### Hypothesis

Final-token evidence exists early but is hidden beneath a default frequency prior.

### Debiased score

For each position:

\[
r_t(v)=\log x_t(v)-\log q_t(v).
\]

Alternative stable implementation:

\[
r_t(v)
=
\ell_t(v)-b_t(v),
\]

where \(b_t(v)\) is the mean logit under destroyed-instance inputs.

### Measurements

Track final or reference token \(y\):

- raw rank under \(x_t\);
- debiased rank under \(r_t\);
- raw top-\(k\) time;
- debiased top-\(k\) time;
- raw and debiased accuracy;
- evidence against the strongest default competitor \(h_t\):
  \[
  r_t(y)-r_t(h_t).
  \]

### Frequency analysis

Regress emergence time on:

- token frequency;
- autoregressive surprisal from an external LM;
- output-row norm;
- output bias;
- POS;
- position;
- subword length;
- named-entity status.

Repeat after debiasing.

### Interpretation

- Debiasing makes \(y\) visible much earlier: prior masking.
- Debiasing does not help: token evidence is genuinely absent.
- Frequency correlation disappears after debiasing: frequency barrier.
- Frequency correlation remains after controlling for surprisal and geometry: stronger lexical-frequency mechanism.

---

## EXP-07 — Independent probes on the continuous state

**Priority:** P1  
**Models:** ELF, LangFlow

### Hypothesis

Token information may be present in \(z_t\) or \(\hat z_{0|t}\) before the native decoder can use it.

### Probe families

Train separate probes for each \(t\):

1. linear probe;
2. small MLP;
3. low-rank probe;
4. kNN or prototype probe;
5. full-context probe;
6. local-position probe.

Use frequency-balanced sampling and report macro metrics.

### Probe inputs

For ELF:

- current noisy state \(z_t\);
- preliminary prediction;
- self-conditioned predicted clean state;
- decode-branch hidden state;
- residuals only as a separate ablation.

For LangFlow:

- noisy state;
- hidden state before vocabulary projection;
- expected embedding \(E^\top x_t\);
- self-conditioning state.

### Cross-time generalization

Train at time \(t_a\), evaluate at \(t_b\). Include:

- clean-trained probe tested early;
- late-trained probe tested early;
- early-trained probe tested late.

### Interpretation

- A shared probe works across times: stable lexical directions form early.
- Only timestep-specific probes work: representation basis changes over time.
- MLP strongly outperforms linear: information is nonlinear or entangled.
- Full-context strongly outperforms local: token identity is distributed across positions.

### Causal extension

Move \(z_t\) along a probe-derived token direction and continue sampling. Measure whether the final token distribution changes as predicted.

Probe readability without intervention should be described as recoverable information, not model knowledge.

---

## EXP-08 — Coarse-to-fine information emergence

**Priority:** P1  
**Models:** ELF, LangFlow

### Hypothesis

The trajectory first identifies a broad lexical or semantic region and only later resolves the exact token.

### Target hierarchies

Construct:

- function versus content word;
- punctuation class;
- POS;
- morphological family;
- token-frequency bucket;
- semantic embedding clusters;
- subword family;
- nearest-neighbor token neighborhoods.

### Controls

For every hierarchy use:

- equal-size random clusters;
- frequency-matched random clusters;
- equal-difficulty classification baselines;
- several cluster counts.

### Measurements

From both \(z_t\) and \(x_t\), estimate:

- class accuracy;
- exact-token accuracy;
- class mutual information;
- mass assigned to the final token's class:
  \[
  M_t(C_y)=\sum_{v\in C_y}x_t(v);
  \]
- expected embedding similarity:
  \[
  \bar e_t=\sum_v x_t(v)E_v.
  \]

### Interpretation

A coarse-to-fine claim is justified only if real semantic or syntactic classes emerge earlier than difficulty-matched random groupings.

---

## EXP-09 — Contextual bootstrapping

**Priority:** P1  
**Models:** ELF, LangFlow

### Hypothesis

A target position leaves the default prior because other positions become informative.

### Four-way intervention

For target position \(i\):

| Target state | Context state |
|---|---|
| noisy | noisy |
| noisy | clean or low-noise |
| clean or low-noise | noisy |
| noisy | shuffled |

Measure target-token evidence and final outcomes.

### ELF-specific constraint

Do not arbitrarily splice contextual encoder states from incompatible sentences and treat them as in-distribution.

Preferred construction:

1. modify the sequence at token level;
2. re-encode with the frozen contextual encoder;
3. apply matched forward corruption;
4. evaluate the model.

For reverse-trajectory interventions, report distribution-shift diagnostics.

### Influence graph

Clean or improve one context position \(j\) at a time and measure:

\[
A_{j\rightarrow i}(t)
=
\Delta
\log
\frac{x_{t,i}(y_i)}
{q_{t,i}(y_i)}.
\]

Build a directed influence matrix.

### Key questions

- Are early anchor positions punctuation, function words, or content words?
- Does left-to-right influence naturally appear?
- Are the positions that commit early causally useful to later positions?
- Does LangFlow's per-step lexical posterior create stronger bootstrap effects than ELF?

---

## EXP-10 — Self-conditioning causal decomposition

**Priority:** P1  
**Models:** ELF, LangFlow

### Hypothesis

Self-conditioning converts temporary token or representation predictions into persistent dynamical effects.

### Required state

All trajectory experiments must preserve:

\[
S_t=(z_t,s_t).
\]

### Interventions

At selected times:

1. normal self-conditioning;
2. zero self-conditioning;
3. previous-step self-conditioning;
4. shuffled self-conditioning within batch;
5. position-shuffled self-conditioning;
6. decode-branch self-conditioning;
7. prior-only self-conditioning;
8. oracle clean self-conditioning.

### Measurements

- future token evidence;
- token flip rate;
- branch entropy;
- generation quality;
- error amplification;
- correction rate;
- fixed-point behavior.

### Factorization

For each step, isolate changes caused by:

- updating \(z_t\);
- changing timestep;
- changing \(s_t\);
- full update.

This can be done through counterfactual forward calls using the same saved state.

### Interpretation

This experiment determines whether early high-frequency predictions are merely readouts or whether they feed back into the model and shape later generation.

---

## EXP-11 — Dynamical commitment through branching

**Priority:** P1  
**Models:** ELF, LangFlow

### Hypothesis

Token readability and dynamical commitment occur at different times.

### Protocol

From real reverse-trajectory checkpoints \(S_t\):

- add small continuous perturbations to \(z_t\);
- vary future SDE noise;
- vary solver step size;
- vary self-conditioning minimally;
- continue generation \(K\) times.

Estimate:

\[
P_t(v)
=
\Pr(Y_i^{\mathrm{final}}=v\mid S_t+\delta).
\]

Compute:

- branch entropy;
- probability of original final token;
- probability of current top-1 token;
- basin margin;
- sensitivity to perturbation scale.

### Compare with

- native posterior entropy;
- independent probe confidence;
- cosine readout margin;
- decode-branch confidence.

### Interpretation matrix

| Readable? | Branch-stable? | Meaning |
|---|---|---|
| No | No | unresolved |
| Yes | No | information present, not committed |
| No | Yes | dynamically fixed but poorly readable |
| Yes | Yes | robust lexical basin |

This should replace entropy-threshold commitment as the strongest definition.

---

## EXP-12 — What does the ELF anchor residual contain?

**Priority:** P1  
**Models:** ELF

### Hypothesis

The residual after subtracting a lexical expectation contains contextual information that is useful beyond token identity.

### First correction

Rename:

\[
r_t=\hat z_{0|t}-E^\top x_t
\]

as a **barycentric mismatch residual**.

Do not call it a contextual channel before testing.

### Predictive tests

Train probes on:

1. \(\hat z_{0|t}\);
2. anchor \(a_t\);
3. residual \(r_t\);
4. norm-matched random residual;
5. orthogonalized residual if a lexical subspace is defined.

Targets:

- exact token;
- neighboring tokens;
- POS;
- dependency relation;
- sentence position;
- local semantic class;
- sentence-level topic.

### Causal tests

1. Decode from \(a_t\) only.
2. Decode from \(r_t\) only.
3. Replace \(r_t\) with zero.
4. Swap \(r_t\) between positions with the same token.
5. Swap \(r_t\) between different contexts.
6. Preserve norm but randomize residual direction.
7. Continue denoising after each intervention.

### Stronger decomposition

Learn a lexical subspace \(U_{\mathrm{lex}}\) using supervised dimensionality reduction or regression from clean contextual states to token identity.

Then define an actual orthogonal projection:

\[
P_{\mathrm{lex}}=U_{\mathrm{lex}}U_{\mathrm{lex}}^\top.
\]

Analyze:

\[
z^{\mathrm{lex}}=P_{\mathrm{lex}}z,
\qquad
z^{\perp}=(I-P_{\mathrm{lex}})z.
\]

This is far more interpretable than subtracting a token barycenter.

---

## EXP-13 — Decode-branch mechanism and compute-matched controls

**Priority:** P1, directly connected to the strongest current result  
**Models:** ELF

### Hypothesis

The decode branch contains token-specific correction information, not merely generic refinement from extra compute.

### Controls

Compare standard self-conditioning against:

1. current decode-self-conditioning;
2. previous-step decode-self-conditioning;
3. extra denoising-mode pass;
4. decode mode with incorrect timestep;
5. decode output with positions shuffled;
6. decode residual with sign reversed;
7. matched-norm random residual;
8. low-rank approximation to decode residual;
9. direct logits fed back through expected embeddings;
10. a compute-matched learned refinement baseline if training is allowed.

### Correction taxonomy

For each position classify the effect:

- wrong \(\rightarrow\) correct;
- correct \(\rightarrow\) wrong;
- wrong token A \(\rightarrow\) wrong token B;
- unchanged;
- confidence-only change.

Analyze by:

- token frequency;
- surprisal;
- POS;
- position;
- local context ambiguity;
- residual norm;
- linear/decode disagreement.

### Mechanistic criterion

The correction claim is supported if:

- semantically aligned decode residuals outperform matched random or shuffled residuals;
- improvements concentrate on linear-head errors that the decode branch predicts correctly;
- correction remains after compute matching;
- token-specific directionality predicts final changes.

---

## EXP-14 — Validate or reject “commit–release–recommit”

**Priority:** P1  
**Models:** ELF

### Hypothesis

ELF has a genuine non-monotonic dynamical phase in which lexical stability rises, falls, and rises again.

### Required evidence

The pattern must appear on actual reverse trajectories in at least two of:

- calibrated token evidence;
- branch stability;
- final-token agreement;
- token flip rate;
- probe recoverability;
- context sensitivity.

A decrease in cosine nearest-token accuracy alone is insufficient.

### Alternative explanations to test

- output norm drift;
- bias interaction;
- contextual representation moving away from static embeddings;
- solver discretization;
- time-conditioned decoder calibration;
- difference between linear and nonlinear readout;
- over-denoising under the forward probe.

### Decision rule

Use “commit–release–recommit” only if a token-specific, perturbation-robust quantity shows the same non-monotonic pattern.

Otherwise use a narrower phrase such as:

> non-monotonic cosine alignment with static token embeddings.

---

## EXP-15 — Training-time schedule correctness

**Priority:** P2  
**Models:** ELF

### Hypothesis

Training emphasis near an empirically identified transition region improves generation.

### Variants

1. original base time distribution \(p(t)\);
2. importance-sampled \(q(t)\) with weight \(p(t)/q(t)\);
3. deliberately reweighted \(q(t)\);
4. uniform time;
5. hand-designed window;
6. gradient-variance-optimal sampler;
7. probe-derived sampler from reverse trajectories rather than forward probes.

### Controls

Keep fixed:

- total updates;
- number of tokens;
- optimizer;
- batch size;
- effective loss scale;
- checkpoint selection rule.

### Measurements

- denoising loss by time bin;
- token recovery by time bin;
- gradient norm and variance;
- generation quality;
- calibration;
- robustness across step counts.

### Interpretation

If only the deliberately reweighted objective helps, describe it as task-specific loss reweighting—not unbiased importance sampling.

---

## EXP-16 — Minimal training intervention

**Priority:** P2  
**Models:** ELF

Do not train all proposed losses simultaneously.

### Candidate A: Decode-teacher distribution distillation

\[
L_{\mathrm{KD}}
=
\mathrm{KL}
(
\operatorname{sg}(x_t^{\mathrm{dec}})
\|
x_t^{\mathrm{lin}}
).
\]

### Candidate B: Decode-state consistency

\[
L_{\mathrm{SC}}
=
\|
\hat z_{0|t}
-
\operatorname{sg}(z_t^{\mathrm{dec}})
\|^2.
\]

### Candidate C: Ground-truth supervised lexical probe

A carefully weighted CE on an explicitly defined intermediate readout.

### Required sequence

1. train each component alone;
2. compare against an equal-compute continued-training baseline;
3. evaluate at multiple step counts;
4. inspect generation diversity;
5. only then test combinations.

### Selection criterion

Choose the component whose improvement matches the mechanism supported by EXP-13. Do not select based only on the best PPL number.

---

# 7. Recommended priority order

## Phase 0 — Correctness audit

Run first:

1. EXP-00: tensor and code-path audit;
2. EXP-01: forward probe versus actual trajectory;
3. EXP-02: correct LangFlow objects;
4. EXP-03: matched-SNR comparison.

These experiments determine whether the current central observations survive under valid comparisons.

## Phase 1 — Mechanism isolation

Then run:

5. EXP-04: decoder geometry;
6. EXP-05: learned prior;
7. EXP-06: prior subtraction;
8. EXP-07: independent probes;
9. EXP-10: self-conditioning decomposition;
10. EXP-11: branching stability.

These distinguish readout artifact, learned prior, representation formation, and dynamical commitment.

## Phase 2 — Deeper structure

Then run:

11. EXP-08: coarse-to-fine;
12. EXP-09: contextual bootstrapping;
13. EXP-12: residual analysis;
14. EXP-14: validate non-monotonic phases.

## Phase 3 — Intervention

Finally run:

15. EXP-13: compute-matched decode branch;
16. EXP-15: schedule study;
17. EXP-16: one minimal training objective.

---

# 8. How the paper should be reframed depending on results

## Story A — Decoder-interface geometry

Choose this if:

- independent probes recover token identity early;
- decoder normalization and calibration explain much of the early high-frequency behavior;
- the decode branch closes a real readout gap;
- decode-self-conditioning survives compute-matched controls.

Core claim:

> ELF forms token-predictive contextual states earlier than its native linear interface can reliably expose them; nonlinear decode correction can be fed back into the trajectory.

This is currently the most promising story.

## Story B — Evidence formation and frequency barrier

Choose this if:

- debiased posteriors reveal early final-token evidence;
- frequency effects survive basic geometry controls;
- the evidence-versus-prior crossing predicts emergence time;
- LangFlow and ELF differ systematically because of trajectory supervision.

Core claim:

> Continuous language flows transition from a learned default lexical prior to sample-specific token evidence, with different architectures exposing and using that evidence differently.

## Story C — Contextual bootstrapping

Choose this if:

- clean or improved context strongly rescues noisy target positions;
- influence graphs show a consistent cascade;
- self-conditioning carries this structure forward.

Core claim:

> Lexical identity in continuous language flows is jointly constructed across positions rather than independently denoised.

## Story D — Dynamical basin formation

Choose this if:

- branch stability emerges at a distinct time from decoder readability;
- perturbation experiments reveal clear basin transitions;
- actual reverse trajectories show robust non-monotonic commitment.

Core claim:

> Token identity becomes dynamically stable through basin formation, which is not captured by entropy or single-head readability.

Do not combine all four stories unless the evidence genuinely supports a unified mechanism.

---

# 9. Specific recommendations for the current draft

## Keep in the main paper

- The distinction among continuous state, lexical belief, token proposal, and state commitment.
- The ELF linear-versus-decode branch gap.
- The decode residual interpolation experiment.
- Decode-self-conditioning, after compute-matched controls.
- Actual reverse-trajectory diagnostics.
- A careful ELF–LangFlow comparison using native model objects.
- Failure cases and step-count sensitivity.

## Move to appendix temporarily

- MDLM and DUO comparisons.
- SAR.
- Large tables of proposed schedules.
- Position-type taxonomies based only on forward-corruption paths.
- Unvalidated training objectives.

## Remove or weaken now

- “The residual encodes contextual information” based only on \(\rho(t)\).
- “The timing gap is explained by SNR schedule.”
- “Geometric commitment is decoder-independent.”
- “Stable-but-imperfect positions are stuck” unless shown on actual sampler trajectories.
- “Commit–release–recommit” unless validated by branch stability or another robust trajectory measure.
- “Probe-proportional sampling does not change the objective.”
- “Decode self-conditioning directly confirms the commitment mechanism.”

## Rename metrics

- “Geometric nearest-token accuracy”  
  → “Cosine-normalized token readout accuracy.”

- “Normalized anchor residual fraction”  
  → “Anchor mismatch ratio.”

- “Commitment time from entropy threshold”  
  → “Entropy-collapse time.”

- “Wrong-to-correct trajectory rate” on forward-corrupted states  
  → “Cross-noise-level proposal transition rate.”

These names are less rhetorically impressive but scientifically safer.

---

# 10. Minimal set of figures for a revised paper

A strong revised paper may need only the following core figures.

## Figure 1 — Model interfaces

Side-by-side ELF and LangFlow computational graphs showing:

- \(z_t\);
- model output;
- vocabulary distribution;
- expected embedding;
- self-conditioning;
- final decoder.

## Figure 2 — Forward probe versus reverse trajectory

ELF curves for:

- linear posterior;
- decode posterior;
- calibrated probe;
- branch stability.

## Figure 3 — Decoder geometry decomposition

Accuracy or frequency domination under:

- full head;
- no bias;
- normalized rows;
- matched Gaussian;
- full network.

## Figure 4 — Prior versus instance evidence

Raw and debiased final-token ranks over time, split by token frequency.

## Figure 5 — ELF versus LangFlow at matched log-SNR

Native posterior recovery, calibrated probe recovery, and entropy.

## Figure 6 — Decode-self-conditioning controls

Generation quality and correction taxonomy under compute-matched variants.

Additional figures should be added only if they support the selected final story.

---

# 11. Requirements for future Claude-generated implementation specs

Every experiment spec should include the following fields.

## 11.1 Scientific definition

- hypothesis;
- null hypothesis;
- alternative explanations;
- exact claim the experiment can support;
- claims it cannot support.

## 11.2 Model and checkpoint details

- repository commit;
- checkpoint identifier;
- tokenizer;
- sequence length;
- precision;
- device;
- model mode;
- sampler configuration.

## 11.3 Data

- dataset split;
- sample count;
- filtering;
- random seeds;
- padding handling;
- frequency statistics source;
- any external language model used for surprisal.

## 11.4 Tensor capture

- exact module hooks;
- tensor names and shapes;
- pre/post LayerNorm status;
- timestep;
- self-conditioning state;
- gradient status;
- storage format.

## 11.5 Intervention

- exact tensor changed;
- exact mathematical operation;
- whether the changed state remains on-manifold;
- norm matching;
- control conditions;
- whether future noise is shared.

## 11.6 Metrics

- token-level;
- position-level;
- sequence-level;
- generation-level;
- macro and micro versions;
- confidence intervals;
- multiple-comparison correction where needed.

## 11.7 Required plots

Specify axis, grouping, confidence interval, and expected reference lines.

## 11.8 Acceptance criteria

Define in advance what outcome:

- supports the hypothesis;
- rejects it;
- is inconclusive;
- reveals an implementation bug.

## 11.9 Reproducibility

- command line;
- config file;
- output directory;
- cached intermediate tensors;
- deterministic flags;
- runtime and storage estimate.

---

# 12. Final assessment

The current project is not “experiments done at random.” It already discovered a useful fault line:

\[
\text{continuous representation quality}
\neq
\text{linear token readability}
\neq
\text{nonlinear decoder readability}
\neq
\text{dynamical commitment}.
\]

The draft's main weakness is that it compresses these different phenomena into one word—commitment—and then builds a large method stack on top of that compression.

The next version should do the opposite:

1. separate the objects;
2. list competing mechanisms;
3. run discriminative controls;
4. identify one causal bottleneck;
5. design one intervention for that bottleneck.

The highest-value immediate question is:

> In ELF and LangFlow, when early outputs are dominated by frequent tokens, how much is explained by static decoder geometry, how much by a learned time-conditioned prior, how much token-specific information already exists in the continuous state, and when does that information begin to influence the actual reverse dynamics?

Answering that question cleanly would produce a stronger paper than adding more losses to the current draft.
