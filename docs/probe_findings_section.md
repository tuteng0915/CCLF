# Empirical Analysis: Lexical Commitment Dynamics in Continuous Text Diffusion

> Draft findings section synthesizing all probe experiments.
> Probe data: ELF-B (OWT, step 95085), LangFlow (OWT), MDLM (kuleshov-group/mdlm-owt), DUO (s-sahoo/duo)
> All probes: n=64 sequences, seq_len=128–256, 21 t-steps, 4 noise seeds.

---

## 3.1  ELF Commitment Phases

We probe ELF-B by applying flow noise at each fixed $t\in[0,1]$ and measuring the decode-head token posterior $p_\theta(x_0\mid z_t)$ without performing any denoising steps.  
Four phases emerge (Figure X):

**Phase 1 — Prior-dominated (t = 0–0.10).**  
At maximum noise ($t=0$), 76.9% of positions are "committed" in the sense $H(p_\theta) < 0.1$, yet all of them are committed to the *wrong* token — the model is projecting pure noise onto its nearest token attractor.  
Entropy rises sharply from 0.12 to 0.49 as the weak $t=0.05\to0.10$ signal disrupts this prior, pushing 59% of positions into an uncommitted state.  
Top-1 accuracy is $\le 3\%$.

**Phase 2 — Rapid commitment cliff (t = 0.10–0.35).**  
Over a 0.25-wide window entropy collapses $0.49\to0.09$, top-1 accuracy rises $1.5\%\to59\%$, and the wrong→correct transition rate $w{\to}c$ peaks at $20.8\%$ per step (at $t=0.30$).  
Correct commitments grow from 1.2% to 55.5% of all positions.  
This cliff is the model's primary lexical decision window.

**Phase 3 — Stable-but-imperfect (t = 0.35–0.95).**  
Entropy stabilizes at $0.05\text{–}0.07$, top-1 holds at $77\text{–}80\%$.  
Wrong commitments persist at $13\text{–}17\%$; the wrong→correct rate falls to $1\text{–}1.5\%$ per step — the model cannot self-correct.  
This 0.6-wide plateau represents wasted compute: the model is locked into slightly wrong tokens it can no longer revise.

**Phase 4 — Final refinement at t = 1.0.**  
Top-1 jumps from 81.1% to 97.9% ($+16.8$ pp) and 17.1% of positions revise their top-1 prediction in a single step.  
This shows the final decode step is performing substantial lexical correction that the continuous trajectory could not.

---

## 3.2  Decode Branch Corrects What the Flow Cannot

We compare the *linear branch* (flow output $\hat{x}_t$ projected via unembedding matrix, bypassing the learned decode head) with the *decode branch* (full decode head including nonlinear projection) at each $t$.

| t | Lin top-1 | Dec top-1 | Gap |
|---|---|---|---|
| 0.25 | 26.0% | 34.5% | +8.5 pp |
| 0.30 | 45.5% | 70.0% | **+24.5 pp** |
| 0.35 | 59.2% | 86.8% | **+27.6 pp** |
| 0.40 | 68.3% | 92.4% | +24.1 pp |
| 0.50 | 76.8% | 96.3% | +19.6 pp |
| 0.70 | 79.8% | 97.1% | +17.3 pp |
| 0.90 | 76.9% | 95.5% | +18.6 pp |
| 1.00 | 97.9% | 89.9% | −8.0 pp |

**Key observations:**
1. The decode branch is ahead of the linear branch by $\approx 17\text{–}28$ percentage points throughout $t\in[0.30, 0.95]$.  
   This gap, $G_\text{dec}(t) = \text{top-1}_\text{dec} - \text{top-1}_\text{lin}$, is large precisely in Phase 2–3 where lexical decisions solidify.
2. At $t=1.0$, the gap reverses (−8.0 pp): the final linear projection (tied to the token embedding matrix) outperforms the learned decode head, suggesting the decode head is tuned for $t < 1$ and degrades at clean input.
3. The persistent decode advantage implies that the flow's continuous trajectory already contains the right lexical information, but the linear projection cannot extract it — the decode head bridges this gap by learning a nonlinear correction.

**Interpretation:** The decode branch is performing token-level error correction that the denoising trajectory cannot. The $G_\text{dec}$ curve defines when and where this correction is most needed, motivating methods that internalize this capacity into the flow itself.

---

## 3.3  LangFlow: SNR Schedule Dominates Architecture

LangFlow (Domingo-Enrich et al. 2026) uses a learnable Gumbel proposal for the noise schedule $\gamma(t) = \log\text{SNR}(t)$, resulting in an SNR profile that grows much more slowly than ELF's linear flow.

**Note on t-axis convention:** LangFlow uses the reversed convention $t=0$ = clean, $t=1$ = noisy. After reversal for comparison:

| ELF t (clean=1) | ELF top-1 | LangFlow top-1 (reversed) |
|---|---|---|
| 0.20 | 8.6% | ~35% |
| 0.35 | 59.2% | ~10% |
| 0.50 | 76.8% | ~4% |
| 0.85 | 77.3% | ~4% |
| 0.95 | 81.1% | ~4% |

**SNR comparison at matched t values:**

| t | ELF SNR ($t^2/(1-t)^2$) | LangFlow SNR (approx) | Ratio |
|---|---|---|---|
| 0.25 | 0.111 | 0.00106 | **105×** |
| 0.50 | 1.000 | 0.00424 | **236×** |
| 0.75 | 9.000 | 0.01694 | **531×** |

ELF's linear flow schedule provides 100–500× higher SNR at the same $t$ value. This directly explains the $\approx 0.6$ gap in commitment cliff timing (ELF: $t\approx0.25$; LangFlow: $t\approx0.85$–$0.90$ in reversed convention).

**Conclusion:** The commitment cliff location is primarily determined by the SNR schedule, not the model architecture or noise type.

---

## 3.4  Noise Type vs. Schedule: Three-Model Comparison

To isolate the effect of noise type (AbsorbingState vs. UniformState) from the effect of the noise schedule, we probe MDLM (AbsorbingState: corrupted positions replaced with [MASK]) and DUO (UniformState: corrupted positions replaced with random token) using identical loglinear schedules ($\alpha_\text{min}=10^{-4}$, $\lambda\approx9.21$).

| Model | Noise type | 10% top-1 recovery | $t_\text{commit}$ | Never committed |
|---|---|---|---|---|
| ELF | AbsorbingState flow | $t\approx0.15\text{–}0.20$ | 0.019 | 0.0% |
| MDLM | AbsorbingState diffusion | $t=0.75$ | 0.866 | 85.1% |
| DUO | UniformState diffusion | $t=0.80$ | 0.904 | 86.9% |

**Why MDLM/DUO commit so late:** With loglinear $\alpha_\text{min}=10^{-4}$, the noise level at $t=0.50$ is $1-\alpha_{0.5}=99\%$ (99% of positions corrupted/masked). The model has almost no signal until $t > 0.80$. ELF's flow schedule provides $\text{SNR}=1.0$ at $t=0.50$ — vastly more informative.

**Noise-type effect (MDLM vs. DUO at matched schedule):**

| t | MDLM top-1 (masked) | DUO top-1 (corrupted) |
|---|---|---|
| 0.70 | 8.3% | 6.1% |
| 0.80 | 16.8% | 13.1% |
| 0.90 | 39.1% | 36.4% |
| 0.95 | 55.9% | 55.3% |

MDLM has a modest advantage ($\approx 2\text{–}3$ pp) because AbsorbingState marks corrupted positions explicitly with [MASK], allowing the model to focus inference on known-masked positions using clean context. UniformState (DUO) must simultaneously infer which positions are corrupted and what the correct tokens are, leading to slightly lower recovery.

**Main conclusion:** At matched noise schedules, noise type (Absorbing vs. Uniform) explains only a $2\text{–}3$ pp difference in recovery. The $0.55$–$0.60$ gap in commitment cliff timing between ELF and discrete models is explained almost entirely by the SNR schedule difference.

---

## 3.5  Summary and Implications

| Finding | Implication for method design |
|---|---|
| ELF commits in a sharp cliff at $t\approx0.25\text{–}0.35$ | Intervention target: amplify correct commitments in this window |
| 13–17% positions are wrongly committed and cannot self-correct in Phase 3 | Progressive correction during Phase 3 could recover these without retraining |
| Decode branch outperforms flow by +17–28 pp throughout Phase 2–3 | Decode residual $c_t = x_t^\text{dec} - x_t^\text{lin}$ encodes the correction needed |
| ELF commits 0.55 t-units earlier than LangFlow | ELF's high-SNR schedule is the key structural advantage; schedule design matters more than noise type |
| MDLM≈DUO at matched schedules | Noise type has small effect; "AbsorbingState" is not the primary reason for ELF's early commitment |

**Design prescription for Revisable Lexical Anchoring:**  
Given these findings, the optimal method should:
1. **Not suppress early commitment** — Phase 2 correct commitments are genuine and should be preserved.
2. **Improve commitment quality** at $t\in[0.15, 0.35]$ — reduce the wrong-commitment rate before the cliff locks tokens.
3. **Leverage the decode residual** — $c_t(x) = x_t^\text{dec} - x_t^\text{lin}$ is already computable and correlates with positions needing correction; guiding the flow toward decode-branch predictions during Phase 3 can reduce the stable-but-wrong plateau.
4. **Preserve revision ability** — any anchoring should use soft alignment (e.g., anchor distance regularization) rather than hard discretization, to maintain the model's ability to revise wrong-committed tokens.
