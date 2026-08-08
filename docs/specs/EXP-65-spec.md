# EXP-65 Spec — Held-Out Native Hard-Commit Calibration

**Status:** RUNNING  
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
