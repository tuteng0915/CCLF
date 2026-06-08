# ELF Probing & Progressive Decode Correction

---

## 1. Research Question *(~2 min)*

ELF uses *final-only discretization*: the denoising trajectory operates entirely in T5 contextual embedding space, with token projection only at $t=1$. Before designing any new training objective, we need to understand when and how ELF actually makes lexical decisions internally.

**Central question**: Does ELF commit to tokens gradually throughout the trajectory, or all at once at the end? And if it commits early, is that commitment accurate?

---

## 2. Probing Setup *(~2 min)*

**Protocol**: At each fixed $t \in [0,1]$, inject noise to form $z_t = tx + (1-t)\varepsilon$, run one forward pass through the backbone (no denoising), and read the decode head's token posterior $p_\theta(x_0 \mid z_t)$. This gives a snapshot of lexical beliefs at every point along the trajectory.

**Models and setup** (64 seqs, 21 $t$-steps, 4 noise seeds each):

| Model | Noise type | Schedule |
|---|---|---|
| ELF-B (OWT) | Continuous flow | Linear, $\text{SNR}(t) = t^2/(1-t)^2$ |
| LangFlow | Continuous flow | Learnable Gumbel log-SNR |
| MDLM | Absorbing discrete ([MASK]) | Loglinear, $\alpha_\text{min}=10^{-4}$ |
| DUO | Uniform discrete (random token) | Loglinear, $\alpha_\text{min}=10^{-4}$ |

**We ran 8 probe rounds in total** — each motivated by a gap in the previous one. Key iterations for ELF: (1) baseline entropy/top-k; (2) temperature sweep τ ∈ {0.1…5.0} to validate the pattern is temperature-invariant (< 3 pp variation); (3) commitment state decomposition + transition matrix; (4) fate tracking + residual norms + bimodality; (5) two-pass decode branch readout — the pivotal probe. Cross-model: LangFlow (found and fixed a t-axis direction bug), MDLM, DUO (v1 accidentally loaded GPT-2; v2 used the real model).

**Core metrics tracked**: token entropy $H$, top-1/top-5 GT accuracy, per-position commitment state (committed-correct / committed-wrong / uncommitted, threshold $H < 0.1$), transition rates (w→c, c→w), noise agreement across seeds, entropy bimodality, fate tracking, and the decode branch gap $G_\text{dec}(t)$.

---

## 3. Key Findings *(~12 min)*

### Finding 1: ELF has a four-phase commitment structure

| Phase | $t$ range | Key numbers |
|---|---|---|
| Prior-dominated | 0–0.10 | top-1 ≤ 3%; 76.9% of positions "committed" — but all wrong |
| **Commitment cliff** | 0.10–0.35 | Entropy 0.49→0.09; top-1 1.5%→59%; w→c peaks at **20.8%/step** at $t=0.30$ |
| Stable-but-imperfect | 0.35–0.95 | top-1 plateau 77–80%; **13–17% locked to wrong token; w→c ≤ 1.5%** — cannot self-correct |
| Final jump | $t=1.0$ | top-1 81.1%→**97.9% (+16.8 pp)**; 17% of positions flip in one step |

Phase 3 is the core problem: the model has committed early but imperfectly, and cannot revise those errors through continued denoising. The final jump shows the last decode step is doing heavy lifting that the trajectory could not.

---

### Finding 2: The decode branch corrects what the trajectory cannot

We decomposed ELF's readout into two paths:
- **Linear** ($p^\text{lin}_t$): $\hat{x}_t$ projected directly via the unembedding matrix.
- **Decode** ($p^\text{dec}_t$): $\hat{x}_t$ fed back through backbone at $t=1$, then projected — a two-pass readout mimicking what ELF does at the endpoint.

| $t$ | lin top-1 | dec top-1 | $G_\text{dec}$ |
|---|---|---|---|
| 0.30 | 45.5% | 70.0% | **+24.5 pp** |
| 0.35 | 59.2% | 86.8% | **+27.6 pp** |
| 0.50 | 76.8% | 96.3% | +19.6 pp |
| 0.95 | 81.1% | 96.9% | +15.9 pp |
| **1.00** | **97.9%** | 89.9% | **−8.0 pp** (reversal) |

The decode branch leads by **17–28 pp throughout Phase 2–3**. The reversal at $t=1.0$ — where the linear branch overtakes — occurs because the decode head is adapted to $t < 1$ inputs and degrades on clean input. This reversal also explains the "final jump" in Finding 1.

**Interpolation validation**: $\tilde{x} = (1-\gamma)\hat{x}^\text{lin} + \gamma h^\text{dec}$. Top-1 rises monotonically: 81.0% → 88.6% → 94.0% → 96.1% → 96.9% as γ goes 0→1. The decode residual $c_t = h^\text{dec}_t - \hat{x}_t$ is a valid correction direction throughout the plateau — 90.6% of wrong→correct positions have above-median $\|c_t\|$.

---

### Finding 3: Fate tracking of wrong commitments

For positions that are wrong-committed at a given source $t$, where does correction eventually come from?

| Source $t$ | Fixed by trajectory | Fixed only by decode | Permanent error |
|---|---|---|---|
| 0.25 | **73.9%** | 22.2% | 3.9% |
| 0.40 | 38.5% | **56.8%** | 4.7% |
| 0.50 | 24.0% | **71.8%** | 4.3% |

Two patterns stand out. First, there is a ~**4% hard error floor** — positions that stay wrong regardless of source time or correction strategy. Second, the shift from trajectory-correctable to decode-only correctable is dramatic: by $t=0.50$, 72% of fixable errors can only be saved by the decode branch. ELF uses that branch only at $t=1$ — meaning it defers correction of these errors to the last possible moment.

---

### Finding 4: SNR schedule determines commitment timing — not architecture, not noise type

$$\text{SNR}_\text{ELF}(t) = \frac{t^2}{(1-t)^2}, \qquad \text{SNR}_\text{LangFlow}(t) = \exp(-\gamma(t))$$

| $t$ | ELF SNR | LangFlow SNR | Ratio |
|---|---|---|---|
| 0.25 | 0.111 | 0.00106 | **105×** |
| 0.50 | 1.000 | 0.00424 | **236×** |
| 0.75 | 9.000 | 0.01694 | **531×** |

ELF's linear flow provides 100–500× higher SNR at the same nominal $t$. This directly explains why ELF's commitment cliff is at $t \approx 0.25$ while LangFlow's is at $t \approx 0.85$–$0.90$ — a 0.6-unit gap that tracks the SNR difference.

To isolate noise type from schedule effects, we ran MDLM (Absorbing/[MASK]) and DUO (Uniform/random token) with identical loglinear schedules. Their top-1 accuracy differs by only **2–3 pp** at every $t$. Noise type is almost irrelevant; the schedule dominates.

**Bonus: geometric anchoring is model-specific.** For LangFlow (latent space = token embedding space), d_nn monotonically decreases as $t \to 1$ (Δd_nn = −2.76, Δmargin = +0.875) — the state geometrically approaches the token manifold. For ELF (latent space = T5 contextual space), d_nn *increases* because contextual representations encode token + context jointly; a context-independent centroid is the wrong reference. Entropy/top-1 probes, not geometric distance, are the right commitment signal for contextual latent spaces.

---

## 4. Method: Progressive Decode Correction (PDC) *(~3 min)*

**Core insight from Findings 2–3**: The decode branch has corrective capacity (+17–28 pp) throughout the plateau, but ELF applies it only at $t=1$. By $t=0.50$, 72% of fixable errors are decode-only correctable. PDC brings this capacity into the training objective:

$$\mathcal{L}_\text{PDC}(t) = \left\|\hat{x}_t^{den} - \text{sg}(h_t^{dec})\right\|^2 \cdot \mu(t), \qquad \mathcal{L} = \mathcal{L}_\text{ELF} + \lambda\,\mathcal{L}_\text{PDC}$$

- $\text{sg}$: stop-gradient — the decode branch is a fixed target, not a second branch to optimize through.
- $\mu(t) = 0$ for $t < 0.25$ (pre-cliff signal dominated by wrong priors) and $t > 0.95$ (two-pass artifact where linear overtakes decode).
- For $t \in [0.25, 0.95]$: $\mu(t) \propto G_\text{dec}(t)$, fitted as $\mu(t) \approx 0.95\cdot(t-0.25)^{0.031}\cdot(0.95-t)^{0.137}$, peaking at $t \approx 0.35$.

Intuitively, PDC guides the denoising trajectory to internalize the decode residual $c_t$, so the continuous state reaches the Phase 3 plateau already closer to the correct token — without requiring a second pass at inference time.

---

## 5. Broader Picture & Status *(~2 min)*

PDC is one concrete instance of a larger claim:

> **Lexical commitment is not a binary design choice — it is a process that can be scheduled.**

All existing methods sit at fixed points on a coupling-schedule axis $\alpha(t)$:

```
All-step discrete              Final-only discrete
MDLM / DUO / TESS  ←————————————————————→  ELF
   α(t) ≈ 1 constant                  α(t<1) ≈ 0, jump at t=1
                        ↑
              Progressive Anchoring (this work)
              α(t) annealed — find the optimal schedule
```

No prior work treats the coupling schedule itself as the central variable to optimize. That is the gap.

**Current status**:

| | |
|---|---|
| ✅ | All 8 probe rounds complete (ELF ×5, LangFlow ×2, MDLM, DUO); full evidence chain established |
| ✅ | PDC loss and $\mu(t)$ schedule designed and fitted from probe data |
| ✅ | Method draft written; geometric anchoring story resolved |
| ⬜ | **PDC training experiment** — every finding above is probe-supported hypothesis; perplexity / generation quality gains not yet demonstrated |
