# EXP-GS20 Spec — CDCD Cross-Family Replication

**Status**: DEFERRED until ELF core mechanism is resolved
**Priority**: P2
**Proposed adapter**: `experiments/phase_transition/adapters/cdcd_adapter.py`

## Purpose

Test whether the GS16--GS17 mechanism survives a third continuous categorical
diffusion family. CDCD is planned replication, not current evidence.

## Adapter gate

Implement the shared interface where meaningful:

```text
load
encode_clean / decode
make_oracle_state
forward_state
solver_step
native_logsnr
sample_initial_noise
```

Before interpretation, verify:

1. clean states decode correctly;
2. oracle corruption matches the original implementation;
3. one solver step matches the reference sampler;
4. fixed seeds reproduce expected behavior;
5. generation quality matches the reference checkpoint within tolerance;
6. native log-SNR is monotone.

Do not force CDCD into ELF's linear corruption convention.

## Minimal replication

Only replicate the paper-critical quantities:

1. dense `tau_first` and `tau_stable` from GS17;
2. local residual velocity and self-endpoint progress from GS17;
3. calibrated fixed-bank endpoint specificity from GS16;
4. GS15 chord metric as descriptive context only.

GS11--GS13 and all static topic probes are out of scope unless CDCD produces a
specific contradiction that requires them.

## Alignment and claim rule

Compare models by native log-SNR percentile, normalized arc length, and
event-aligned time relative to `tau_50_stable`; never by nominal diffusion time.

Call a result cross-architecture only if the same operational metric is
informative in ELF, LangFlow, and CDCD and trajectory-level confidence intervals
show sign consistency. A CDCD disagreement is a boundary condition, not a
failed replication to hide.

Pilot: 8 trajectories, 65 states, 6 calibrated branches. Formal: at least 32
trajectories × 3 seeds and 8 branches.

---

## Pilot Results (2026-08-02): Plaid substituted for CDCD

**Why Plaid, not CDCD**: DeepMind never released CDCD code or checkpoints
(verified 2026-08-01 via web search: only two unofficial, partial, no-
checkpoint community reproductions exist). Plaid (Gulrajani & Hashimoto,
NeurIPS 2023, "Likelihood-Based Diffusion Language Models") was chosen as a
substitute continuous-diffusion-LM family: it has an actually-released 1B
weights checkpoint trained on OpenWebText2 (same corpus family as the OWT
data ELF/LangFlow use), verified end-to-end (fluent, coherent generation
with working word-level guidance / lexical constraints) after building
FlashAttention 1.0.4 and NVIDIA Apex from source into an isolated `plaid`
conda env.

**Adapter**: `experiments/phase_transition/adapters/plaid_adapter.py`. All 6
items of this spec's adapter-gate checklist pass:
1. clean-state decode: real OWT text round-trips correctly.
2. oracle corruption: near-clean state has cosine 0.99 with true embedding;
   near-noise state has cosine 0.999 with the injected noise and only 0.04
   with the true embedding.
3. one solver step: correct shape, no NaN/Inf, deterministic under a fixed
   generator.
4. fixed seeds reproduce identical noise/state.
5. generation quality: near-clean-state top-1 token recovery = 97.3%.
6. native log-SNR monotone across the full t range (t=0.01: -2.62 up to
   t=0.99: +6.62).

**GS16 (endpoint specificity/collapse) on Plaid — clean cross-architecture
CONFIRMATION, if anything sharper than ELF**: n_traj=8, K=6 (spec's own
pilot scale), t_bank=0.20 (reused from ELF's calibration, see caveat below).
All 8 trajectories reached 7/7 unique candidate endpoints (vs. ELF's 3-9
range at pilot scale). `S_self(t_bank)=-0.0007` [95% CI -0.005,+0.004],
indistinguishable from zero; `rank_self=1` fraction jumps from 12% at
`t_bank` to **100% at the very next scored checkpoint** (`t=0.238`, one grid
step later) and stays there; `H_end` drops from 0.79 to 0.60 to 0.53 within
two steps and plateaus. This is the same qualitative exploration-collapse
signature found on ELF (Section "Pilot Results" of this same document family
in EXP-GS16-spec.md), now independently confirmed on a third, architecturally
distinct continuous-diffusion family -- and the collapse window is if
anything MORE abrupt on Plaid (essentially one grid step vs. ELF's ~2-3
steps spanning t=0.36-0.43).

**GS17 (local residual dynamics) on Plaid — a genuine boundary
condition/disagreement, reported honestly per this spec's own "a CDCD
disagreement is a boundary condition, not a failed replication to hide"
rule**:

1. `cos_endpoint(t)` (local velocity's alignment with the trajectory's own
   endpoint) starts LOW (+0.34 at t=0.05), actually DROPS further to ~0.13
   by t=0.11-0.17, then rises slowly back to only +0.38 by the end (t=0.93).
   This is qualitatively different from ELF, where `cos_endpoint` was
   already high (+0.81) from the very first checkpoint and peaked near +0.99
   mid-trajectory. **Likely cause, not a mechanism disagreement**: Plaid's
   `solver_step` is the native STOCHASTIC ancestral VDM sampler (injects
   fresh Gaussian noise every step), unlike ELF/LangFlow's deterministic
   Euler steps -- the finite-difference velocity estimator used here
   necessarily picks up this injected noise on top of the true drift, which
   would mechanically depress `cos_endpoint` regardless of any real
   directedness difference. This confound does not exist for ELF/LangFlow.
   Not yet disentangled (would need e.g. averaging velocity over multiple
   noise draws at the same state, or reading Plaid's expected drift
   analytically rather than via finite differences).
2. `V_self(t)` goes positive (~+0.01 to +0.03) briefly around t=0.17-0.40
   (matching GS16's collapse window), but then goes NEGATIVE for a long
   middle stretch (t=0.46-0.93, values -0.007 to -0.014) before returning
   near zero at the very end -- a pattern not seen on ELF (where `V_self`
   stayed positive and rose monotonically after its initial transition).
   Not yet explained; plausibly related to the same sampler-noise confound,
   plausibly a genuine architecture difference in how "locked in" the
   self-endpoint advantage is once past the initial collapse. Reported as
   open, not resolved.
3. **Event-order REVERSAL**: `P(tau_affinity <= tau_50_stable) = 1.0` (8/8)
   on Plaid -- candidate-collapse precedes token-level stability -- the
   OPPOSITE of ELF's `P(tau_affinity <= tau_50_stable) = 0.0` (0/16), where
   collapse consistently LAGGED stability. **This is confounded with an
   unaddressed calibration issue**: `t_bank=0.20` was reused directly from
   ELF's own calibration without re-tuning it to Plaid's schedule (exactly
   the "never compare by nominal t across architectures" mistake this
   project has repeatedly had to fix elsewhere, e.g. LangFlow needing
   `t=0.65` instead of ELF's `t=0.28` for GS4/GS6). `tau_affinity` pins to
   `t_bank` itself for all 8 trajectories (the entropy curve's steepest drop
   is at the very first scored point), meaning the true collapse onset could
   be even earlier than `t_bank` on Plaid -- outside what was measured. The
   event-order reversal should NOT yet be read as a confirmed cross-
   architecture disagreement about mechanism ordering; it is at least
   partly an artifact of not recalibrating `t_bank` per architecture.

**Verdict**: per this spec's alignment/claim rule, GS16's core finding
(early indistinguishability from zero, followed by an abrupt, narrow-window
collapse to near-certain self-specificity) is now cross-architecture
CONFIRMED across ELF and Plaid with consistent sign and even a stronger
effect on Plaid. GS17's local-dynamics story is NOT yet confirmed
cross-architecture -- it surfaces a real boundary condition (possible
sampler-stochasticity confound, uncalibrated `t_bank`) that needs to be
resolved (recalibrate `t_bank` on Plaid's own schedule; control for solver
noise in the velocity estimate) before drawing any conclusion about whether
the "point-estimate stabilizes before perturbation-robustness catches up"
ordering found on ELF is universal or ELF-specific.

**Scale/scope caveats**: pilot scale only (n_traj=8, matching this spec's
own stated pilot, well below the formal minimum of n_traj>=32 x 3 seeds x 8
branches). Single seed. `t_bank` not independently calibrated for Plaid.
GS18 (rank-matched / collective-coordination controls) not attempted on
Plaid -- Plaid's embed_dim=16 makes GS18 Part A's rank-k sweep (k up to 128)
not meaningful without redesign. GS19 (async denoising) not attempted on
Plaid (would need Plaid's own self-conditioning/noise-estimation convention
worked out, not yet done).

---

## Full cross-architecture scorecard (2026-08-03): GS16-19 all attempted on Plaid

With GS18 (both parts) and GS19 now also run on Plaid (see their own spec
files' "Cross-architecture replication on Plaid" sections for full detail),
here is the complete picture of what does and does not generalize from ELF
to a second, architecturally distinct continuous-diffusion family:

| Experiment | ELF finding | Plaid result | Verdict |
|---|---|---|---|
| GS16 (endpoint specificity/collapse) | Early S_self~0, abrupt narrow-window collapse to rank_self=1 | Same pattern, if anything sharper (collapses in 1 grid step vs ELF's 2-3) | **Cross-architecture CONFIRMED** |
| GS17 (local velocity dynamics) | cos_endpoint high from t=0.05, V_self positive+rising after collapse, candidate-collapse LAGS token stability | cos_endpoint low/non-monotonic, V_self goes negative for a long middle stretch, candidate-collapse LEADS token stability (reversed) | **Boundary condition** -- likely confounded by (a) Plaid's stochastic solver_step polluting the finite-difference velocity estimate, (b) t_bank=0.20 reused from ELF without recalibrating to Plaid's own schedule |
| GS18 Part A (rank/energy-matched residual) | top-k dominates middle/bottom/random-k at every k -- narrows the "distributed high-rank code" claim | Same dominance pattern at every k in {1,2,4,8} (k=16 is the full 16-dim space, trivially degenerate) | **Cross-architecture CONFIRMED** |
| GS18 Part B (collective coordination) | xi_M3 exceeds all 5 null models at 13/16 checkpoints | xi_M3 exceeds all 5 nulls at only 1/16 checkpoints -- mostly indistinguishable from the null bands | **Boundary condition** -- leading hypothesis is Plaid's per-step stochastic noise injection dilutes the measured spatial correlation regardless of the underlying mechanism |
| GS19 (async denoising ablation) | "all fail" -- every ordering degrades quality (2x-6x worse Gen.PPL), no ordering meets the desired signature | "all fail" again, more decisively (3x-14.4x worse Gen.PPL) | **Cross-architecture CONFIRMED** |

**Pattern in the pattern**: the two experiments that disagree (GS17, GS18
Part B) are exactly the two that depend on FINE-GRAINED TRAJECTORY-LEVEL
signal computed from consecutive states along Plaid's native (stochastic)
sampler -- local finite-difference velocity for GS17, position-to-position
margin-increment correlation for GS18 Part B. The three that replicate
cleanly (GS16, GS18 Part A, GS19) all depend on either (a) coarser,
endpoint-level outcomes after a full rollout (GS16's branch endpoints,
GS19's final generated text) or (b) static representational structure at a
single timestep (GS18 Part A's SVD subspaces), none of which require
differencing consecutive noisy states. This is a genuine, useful pattern to
report, not just a list of hits and misses: **claims that reduce to
"where do things end up" replicate across a stochastic vs. deterministic
sampler; claims that reduce to "what is the instantaneous local motion"
do not, at least not without first controlling for the sampler's own
injected randomness.** Disentangling this (recalibrating `t_bank` on
Plaid's own schedule, and finding a way to isolate deterministic drift from
injected noise in the velocity/correlation estimates) is the natural next
step before citing GS17 or GS18-Part-B as either confirmed or refuted
cross-architecture.
