# EXP-110 Spec — ELF ODE One-Checkpoint Late-Trigger Selector

**Status:** ACTIVE / STAGE A RUNNING  
**Depends on:** EXP-108 binary `.40`-versus-`.45` oracle passing all gates on
two banks

## Question

Can information available inside a deterministic ELF trajectory decide
whether Unlock-4 should fire at the first late checkpoint or wait exactly one
checkpoint?

This is intentionally not another broad trigger sweep. EXP-108 has already
reduced the quality-safe action space to

\[
a_i\in\{t=.40,\ t=.45\}.
\]

Requested times map to the native ODE-32 grid (`.40625` and `.46875`). Fixed
`.40` remains the fallback and primary baseline.

## Utility target

For trajectory \(i\), define

\[
u_i = \operatorname{NLL}_i(a=.40)-\operatorname{NLL}_i(a=.45).
\]

Positive utility means that waiting one checkpoint is better. GPT-2-large NLL
defines the offline target and evaluation metric only; it is never available
to the deployed selector.

## Stage A — signal screen on the two completed audit banks

Regenerate only the two EXP-108 banks to recover internal states. Require
decoded outputs and per-sequence NLLs to agree with the saved EXP-108 arms
before using any feature.

Test two nested signal families:

1. **Current-state maturity (zero extra branch compute).** At `.40625`, record
   suffix mean/quantiles of posterior entropy, top-1 confidence and margin;
   confidence-`.90` anchor fraction; token stability from the previous grid
   point; and normalized predicted-clean displacement.
2. **Deterministic late-shadow response.** Fork `.40` and `.45` Unlock-4
   branches, compare them at the common post-release checkpoint `.625`, and
   define

   \[
   r_i=\bar H_i^{(.40)}(.625)-\bar H_i^{(.45)}(.625).
   \]

   Positive \(r_i\) says the delayed branch is less ambiguous. Also record the
   corresponding confidence and top-1-disagreement responses. The branches
   share the same initial latent and ODE path; there is no future-noise choice.

For each prespecified scalar, report Spearman correlation with \(u_i\), sign
AUC/pairwise accuracy, and the PPL obtained by its best orientation. A signal
survives only if its direction agrees across both banks and pooled pairwise
accuracy is at least `.60`. Current-state maturity is preferred whenever it
meets the same gate as the shadow response.

Stage A is a signal diagnostic, not a method result. Do not report its selected
PPL as held-out performance.

## Stage B — calibrate a quality-safe fallback

Use a new bank only:

- seed 456 / OWT offset 42000;
- `n=64`, real 64-token prefix plus 64-token suffix;
- ELF baseline, deterministic uniform ODE-32;
- exact paired initial latent for fixed `.40`, fixed `.45`, and selector.

For the single Stage-A signal selected without looking at this bank, choose a
threshold \(\gamma\) from its empirical quantiles:

\[
\pi_i =
\begin{cases}
.45,& s_i\ge\gamma,\\
.40,& \text{otherwise}.
\end{cases}
\]

The orientation may be reversed only if Stage A fixed that orientation. Among
thresholds satisfying the complete quality gate relative to fixed `.40`, take
the one with lowest C-PPL. Require at least 8/64 delayed decisions, C-PPL
improvement of at least 2%, and paired mean-NLL CI upper bound below zero.

## Stage C — untouched final test

Freeze the signal, orientation, threshold, and fallback before opening:

- seed 789 / OWT offset 43000, `n=64`.

Promotion requires:

1. C-PPL improves fixed `.40` by at least 2%;
2. paired NLL bootstrap CI upper bound is below zero;
3. D1 delta is at least `-.005`;
4. Rep-4 delta is at most `+.005`;
5. degeneration delta is at most `+.015`;
6. prompt-gain delta is at least `-.01`;
7. the policy makes at least 8 delayed decisions.

Report fixed `.45`, binary oracle, Standard ODE-32, and the complete quality
panel regardless of outcome.

## Compute accounting

If the current-state signal wins, the selector has the same 32 denoiser calls
as fixed Unlock-4 plus one already-required lexical readout. If the shadow
signal wins, report the exact extra branch calls and include a compute-matched
uniform ODE baseline. A shadow selector may establish causal signal utility,
but it is not called an efficient sampler until distilled.

## Interpretation

- current-state pass: late readiness is readable before intervention;
- shadow-only pass: readiness is causal but not statically readable;
- oracle remains large but both signals fail: timing is real, yet the tested
  online observables cannot deploy it;
- final gate failure: retain fixed `.40`; do not retune on seed 789.

No SDE or Plaid run is required for this ELF method-discovery experiment.

Implementation:
`models/ELF-torch/experiments/probe_elf/late_trigger_signal_screen_exp110.py`.
