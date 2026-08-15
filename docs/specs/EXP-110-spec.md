# EXP-110 Spec — ELF ODE One-Checkpoint Late-Trigger Selector

**Status:** DONE / STAGE-B TRANSFER FAILED; FINAL BANK UNOPENED
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

### Stage-A result

Both arms reproduce the saved EXP-108 outputs exactly (`64/64` texts in both
banks). Every zero-branch current-state score fails: pooled AUC lies between
`.472` and `.539`, and no direction agrees across banks. The two causal
response scores pass:

| signal | seed-42 AUC / Spearman | seed-123 AUC / Spearman | pooled AUC / Spearman | decision |
|---|---:|---:|---:|---|
| shadow entropy response | `.665/.100` | `.649/.280` | `.664/.209` | pass |
| shadow confidence response | `.675/.121` | `.691/.327` | `.682/.229` | **selected** |

Thus waiting readiness is not readable from the tested instantaneous maturity
summaries. It becomes measurable only after comparing the two deterministic
short-horizon responses. Per the frozen preference rule, shadow confidence
response is the sole Stage-B signal.

## Stage B — calibrate a quality-safe fallback

Use a new bank only:

- seed 456 / OWT offset 42000;
- `n=64`, real 64-token prefix plus 64-token suffix;
- ELF baseline, deterministic uniform ODE-32;
- exact paired initial latent for fixed `.40`, fixed `.45`, and selector.

For the single Stage-A signal selected without looking at this bank, choose a
threshold \(\gamma\) from the seven policies that delay the top
`8,12,16,20,24,28,32` response scores:

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

## Stage-B result and stop

The new seed-456 / offset-42000 bank confirms that the decision problem is
real: its binary oracle changes C-PPL `392.06 -> 364.14` (`7.12%`), with paired
NLL `-.0753 [-.1078,-.0448]` and the complete quality gate passing. Its winner
count is `.40/.45 = 35/29`.

However, the frozen signal does not transfer. Shadow-confidence AUC reverses
from `.675/.691` on Stage A to `.438`; shadow-entropy AUC is `.465`. Every
prespecified fallback policy loses to fixed `.40`:

| delayed trajectories | selector C-PPL | change vs fixed `.40` | paired-NLL CI |
|---:|---:|---:|---:|
| 8 | 394.92 | +0.73% worse | `[-.0183,.0318]` |
| 12 | 398.63 | +1.67% worse | `[-.0132,.0511]` |
| 16 | 405.73 | +3.49% worse | `[-.0003,.0733]` |
| 20 | 404.04 | +3.05% worse | `[-.0110,.0717]` |
| 24 | 404.85 | +3.26% worse | `[-.0098,.0728]` |
| 28 | 409.66 | +4.49% worse | `[-.0015,.0894]` |
| 32 | 407.86 | +4.03% worse | `[-.0058,.0877]` |

Stage B therefore fails before threshold selection, and seed 789 remains
unopened for EXP-110. The safe conclusion is that ELF has replicated
trajectory-specific late-trigger headroom, but neither instantaneous maturity
nor the tested deterministic shadow response provides a transferable
sequence-level controller.

Calibration implementation:
`models/ELF-torch/experiments/probe_elf/calibrate_late_trigger_selector_exp110.py`.
