# EXP-65 Spec — Held-Out Native Hard-Commit Calibration

**Status:** STAGE B COMPLETE / BASELINE NATIVE-SDE CHECK PENDING
**Priority:** P0  
**Script:** `models/ELF-torch/experiments/probe_elf/hard_commit_calibration_exp65.py`

## Question

Hard commitment survived the unified native-recipe panel, but its commit times
were inherited from an older non-native sweep. Can a confidence-gated commit
policy be calibrated under the native recipe without selecting on the final
length-1024 test bank?

## Stage A: held-out calibration

Run each ELF checkpoint on a dedicated length-128 calibration noise bank:

- checkpoints: baseline, Broad-KD (`kd2`), Commit-KD (`kd_cr`);
- uniform ODE-32, noise scale 2, SC-CFG 3;
- 128 paired unconditional samples, calibration seed 31415;
- `t_c in {0.30, 0.40, 0.50, 0.60}`;
- confidence threshold `gamma in {0.60, 0.70, 0.80}`;
- one matched standard arm per checkpoint.

Selection uses the full PPL/diversity/repetition/degeneration panel. A point
with a better PPL but clear unigram collapse or a material D1/D2 loss is not a
clean winner. Retain the Pareto frontier and choose at most one configuration
per checkpoint before opening the formal test bank.

## Stage B: native-length confirmation

Evaluate the frozen selection against standard decoding using paired noise:

- length 1024, uniform ODE-32, 256 samples;
- the complete unconditional and prefix-conditioned quality panel;
- paired bootstrap confidence intervals;
- for surviving configurations, native SDE-32 fidelity and ODE-16/64 solver
  checks.

The Stage-B seed/noise bank must not be inspected during Stage-A selection.

## Decision rule

The method claim survives only if the selected policy:

1. improves generation PPL at length 1024;
2. does not materially worsen diversity, repetition, degeneration, or
   conditioned ROUGE-L;
3. preserves its direction under the native SDE fidelity check.

Stage C will instrument the selected trajectories to distinguish earlier
stable coordination from premature high-frequency locking.

## Stage-A result and frozen selection

The calibration grid completed on 2026-08-08. Selected values prioritize a
clean quality profile rather than the largest PPL decrease:

| Checkpoint | Standard PPL | Frozen `(t_c, gamma)` | Selected PPL | D1 | D2 | Degeneration |
|---|---:|---:|---:|---:|---:|---:|
| ELF base | 292.8 | `(0.40, 0.60)` | 223.4 | .455 | .883 | .023 |
| Broad-KD | 1330.5 | `(0.50, 0.60)` | 1195.7 | .531 | .954 | .055 |
| Commit-KD | 946.6 | `(0.50, 0.60)` | 714.9 | .532 | .956 | .180 |

For comparison, Broad-KD at `(0.30, 0.60)` reaches PPL 682.8 but lowers D1
from .538 to .427 and D2 from .948 to .883. It is therefore rejected as a
calibration winner despite the larger PPL decrease. The frozen Stage-B choices
were made before inspecting the length-1024 test bank.

## Stage-B length-1024 result

The frozen configurations were evaluated on the untouched seed-42 bank with
256 unconditional samples and 128 fixed conditioned continuations:

| Checkpoint | Method | PPL | D1 | D2 | Rep-4 | Deg. | Cond. PPL | Cond. R-L |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| ELF base | standard | 127.8 | .174 | .703 | .001 | .000 | 247.1 | .105 |
| ELF base | hard commit | **118.1** | .177 | .699 | .001 | .000 | **218.4** | .106 |
| Broad-KD | standard | **105.9** | .227 | .748 | .112 | .289 | **456.8** | .098 |
| Broad-KD | hard commit | 439.7 | .223 | .782 | .005 | .020 | 540.9 | .104 |
| Commit-KD | standard | **271.1** | .232 | .787 | .071 | .098 | **15.9** | .053 |
| Commit-KD | hard commit | 433.2 | .236 | .795 | .008 | .027 | 132.5 | .097 |

The baseline result is a clean length-robust ODE signal and proceeds to the
native SDE fidelity check. The KD results do not preserve the short-context
PPL direction. Instead, hard commitment sharply reduces repetition and
degeneration: Broad-KD unigram collapse falls from 32.0% to 1.6%, and
Commit-KD conditioned collapse falls from 82.0% to 34.4%. Sample inspection
confirms that the unusually low standard KD PPL values reward repeated
high-frequency strings (for example long `The`, punctuation, or `with`
loops). Hard commitment therefore behaves as an anti-degeneration
intervention at length 1024, not as a clean KD quality win under the
preregistered rule.

## Stage-B decision

- retain a positive method claim only for ELF baseline, conditional on the
  native SDE-32 check;
- withdraw the claim that KD plus hard commitment robustly lowers PPL across
  sequence lengths;
- keep the KD rows as evidence that PPL and degeneration can move in opposite
  directions, and prioritize the complete metric panel over PPL alone.
