# Spec 03 — Method Section Draft

**Type**: writing  
**Priority**: medium  
**Output**: `docs/method_draft.md`

---

## Background

The probing experiments are complete. The method is now well-defined:
**Progressive Decode Correction (PDC)** — inject decode-branch correction signal into the
denoising trajectory, rather than applying it only at t=1.

This spec asks for a structured draft of the method section, in academic paper style
(can be Markdown; will be converted to LaTeX later).

---

## What to write

File: `docs/method_draft.md`

### Required sections (in order)

#### 3.1 Motivation

One short paragraph: ELF commits at t≈0.25 (cite probe v1/v4 findings), but the decode
branch has corrective capacity throughout the plateau (cite G_dec table from probe_decode_v1).
The gap G_dec(t) ≈ +0.15–+0.25 is wasted by the final-only decode architecture.

#### 3.2 Preliminary: Decode Branch and the Two-Pass Readout

Define notation:
- $z_t = t \cdot x + (1-t) \cdot \varepsilon$ — ELF flow interpolation
- $\hat{x}_t^{den}$ — denoiser output (FinalLayer, T5 contextual space)
- $h_t^{dec}$ — backbone output when $\hat{x}_t^{den}$ is fed back as input at t=1

The decode correction residual: $c_t = h_t^{dec} - \hat{x}_t^{den}$

Two key empirical facts (cite probe results):
1. G_dec(t) = top1(p_t^dec) − top1(p_t^lin) ≈ +0.15–+0.25 throughout t∈[0.30, 0.95]
2. Interpolation probe: top1_gt is monotonically increasing in γ at t=0.95 (cite table)
   → c_t is a valid correction direction throughout the plateau

#### 3.3 Progressive Decode Correction Loss

$$\mathcal{L}_\text{PDC}(t) = \left\| \hat{x}_t^{den} - \text{sg}(h_t^{dec}) \right\|^2 \cdot \mu(t)$$

- $\text{sg}$: stop-gradient on $h_t^{dec}$ (treat it as a fixed target)
- $\mu(t)$: schedule derived from G_dec data (see spec-02):
  - $\mu(t) = 0$ for $t < 0.25$ (pre-cliff, noisy signal)
  - $\mu(t) \propto$ G_dec(t) for $t \in [0.25, 0.95]$
  - $\mu(t) = 0$ for $t > 0.95$ (two-pass artifact)
- Full loss: $\mathcal{L} = \mathcal{L}_\text{ELF} + \lambda \cdot \mathcal{L}_\text{PDC}$

Intuition sentence: "This loss guides the denoising trajectory to internalize the decode
correction residual, so that by the time t reaches the plateau, the model's latent state
is already closer to the token manifold without requiring a second pass."

#### 3.4 Connection to Fate Tracking

Table (from probe v4 `results/elf/probe_v4/anchor_probe_v4.json`):

| source_t | n_wrong | traj_corrected | decode_corrected | stays_wrong |
|---|---|---|---|---|
| 0.25 | 69.7 | 73.9% | 22.2% | 3.9% |
| 0.40 | 39.6 | 38.5% | 56.8% | 4.7% |
| 0.50 | 30.6 | 24.0% | 71.8% | 4.3% |

Interpretation: at t=0.50, 72% of wrong commitments can only be corrected by the decode
branch. PDC brings this correction forward into the trajectory, reducing the hard error
floor beyond the 4% irreducible minimum.

#### 3.5 Training Details (placeholder)

Note: "Training details TBD — this section will be filled after implementation
(see spec-04)."

---

## Data sources to reference

All data is in `results/`; cite exact JSON paths for reproducibility:

| Claim | Source |
|---|---|
| ELF commits at t≈0.25 | `results/elf/probe_v1/anchor_probe.json` |
| G_dec table | `results/elf/probe_decode_v1/probe_decode_branch.json` |
| Interpolation probe (γ sweep) | same file |
| Fate tracking table | `results/elf/probe_v4/anchor_probe_v4.json` |
| μ(t) formula | `results/elf/snr_analysis/analysis.json` |

---

## Success criteria

- All four required sections present and coherent
- Every quantitative claim is backed by a cited result file and specific field
- No unsupported claims (e.g. do not claim PDC improves perplexity — that's untested)
- Length: ~600–900 words excluding tables
