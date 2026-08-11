# EXP-90 Spec — Cross-Architecture Temporary-Anchor Portability

**Status:** IMPLEMENTED / CALIBRATION PILOTS PENDING
**Purpose:** determine whether the EXP-82 coverage-over-confidence signal is a
general continuous-LM intervention or an ELF deterministic-ODE special case.

## Protocol

Run LangFlow and Plaid with each model's native 32-step solver. Do not reuse
ELF nominal `t=.30`; choose the trigger step from each architecture's own
endpoint-collapse calibration, then freeze it before quality evaluation.

Compare:

1. native parallel-32;
2. random 50% position-correct predicted-clean anchors;
3. top-confidence 50% position-correct anchors;
4. the random mask with within-sequence shuffled anchor content.

Anchors remain active for four native solver intervals and are then released.
All arms share initial noise. Plaid additionally shares the exact ancestral
noise at every solver step. Evaluate paired unconditional and fixed-prefix
conditional generation; conditional runs restore the observed clean prompt
after every native step. A duplicate native-parallel batch must achieve exact
token agreement (`1.0`) in both scopes before any arm is scored.

Report U-PPL, prompt-conditioned and shuffled-prompt PPL, prompt gain, ROUGE-L,
D1/D2/Rep-4/degeneration/collapse, exact anchor fraction and confidence,
post-release revision, prompt-clamp error, denoiser/readout calls, and texts.

## Decision

- **Portable coordination clue:** random correct anchors beat native parallel
  and top-confidence anchors on both U/C PPL without a larger diversity or
  degeneration cost, while shuffled content fails.
- **Solver-specific ELF effect:** LangFlow and Plaid show no gain after native
  trigger calibration and paired-noise controls.
- **Architecture boundary:** the signal survives on exactly one alternative
  architecture; scope the mechanism by deterministic versus ancestral solver
  and by embedding parameterization.

Runner:
`experiments/interventions/eval_temporary_anchor_portability_exp90.py`.
