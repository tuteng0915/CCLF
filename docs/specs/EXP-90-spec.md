# EXP-90 Spec — Cross-Architecture Temporary-Anchor Portability

**Status:** DONE / CONDITIONAL PORTABILITY POSITIVE, FULL PARETO PARTIAL
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

## Launch record (2026-08-11)

Both architecture smokes pass U/C duplicate-native agreement `1.0`, exact
anchor density `.50`, and zero latent prompt-clamp error. Plaid shares seeded
ancestral noise at every step. Model-native endpoint calibration freezes:

- LangFlow: trigger step `26/32` (`t_native=.8211`), horizon 4;
- Plaid: trigger step `18/32` (`t_native=.5837`), horizon 4.

Three inference seeds (`42/123/456`) with `n_U=n_C=32` are complete for each
architecture. These are P0 portability panels, not formal method claims.

## Result

Three-seed means are:

| Model | Arm | U-PPL | C-PPL | Gain | C-RL | C-D1 | C-Rep4 | U-Deg. | C-Deg. | Revision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| LangFlow | Standard | 13.33 | 82.01 | .4932 | .0461 | .4471 | .1209 | .3021 | .2604 | -- |
|  | random correct | 14.52 | **73.40** | **.5118** | .0462 | .4316 | .1228 | .3125 | .2708 | .040 |
|  | top-confidence | 16.72 | 87.95 | .4898 | .0457 | .4527 | .1087 | .3229 | .2708 | .000 |
|  | shuffled content | 423.43 | 1180.01 | .3238 | .0392 | .5974 | .0031 | .1562 | .0833 | .036 |
| Plaid | Standard | 136.90 | 97.81 | .5303 | .0964 | .6183 | .0000 | .0417 | .0521 | -- |
|  | random correct | 121.80 | **85.51** | **.5414** | **.1003** | .6096 | .0000 | .0625 | .0417 | .627 |
|  | top-confidence | **114.46** | 90.42 | .5128 | .0998 | .6173 | .0002 | .0417 | .0417 | .389 |
|  | shuffled content | 952.03 | 682.19 | .3082 | .0845 | .6619 | .0002 | .0208 | .0521 | .794 |

Random correct anchors improve C-PPL in `3/3` seeds on both architectures.
They also improve LangFlow prompt gain in `3/3` seeds and Plaid U-PPL in
`3/3`. LangFlow U-PPL instead worsens slightly in `3/3`; top-confidence
anchors worsen both LangFlow scopes, while Plaid top-confidence is strongest
unconditionally but less consistent conditionally. Shuffled content is
catastrophic in every model/seed despite often looking superficially diverse.

The portable result is therefore narrower than a universal method win:
position-correct, broad temporary context reliably helps conditional
generation, and correct content is necessary. Unconditional benefit,
confidence selection, revision rate, and the D1/degeneration trade-off are
architecture- and solver-dependent.
