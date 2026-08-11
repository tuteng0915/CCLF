# EXP-88 Spec — Shadow-Validated Adaptive Rollback

**Status:** DONE / ROLLBACK LOWERS PPL BUT FAILS DIVERSITY GATE
**Frozen parent cell:** EXP-82 random-position, position-correct content at
`t=.30`, density `.50`, horizon `H=4`.

## Question

Can a counterfactual consistency check release harmful anchors while retaining
the replicated EXP-82 PPL/ROUGE-L gain and reducing its diversity/repetition
cost?

An ordinary lexical rollback is invalid: conditioned positions are restored
to their anchor state inside the denoiser, so their observed identity and
confidence cannot reveal disagreement. At the midpoint of the four-step lock,
perform one zero-step **shadow forward** with those anchors unmasked. Use the
shadow prediction only to decide release; do not advance the sampler with it.

## Arms

1. Standard-32;
2. fixed random-position Unlock (`32` denoiser + `1` readout);
3. shadow-null: extra zero-step forward/readouts but no anchors;
4. shadow-keep: fixed Unlock plus the shadow computation, ignored;
5. identity rollback: release anchors whose shadow token disagrees;
6. confidence rollback: release when shadow confidence falls more than `.10`
   below trigger confidence;
7. combined rollback: union of identity and confidence release.

All anchor arms use the same selected positions and paired U/C noise. Report
release fraction, final anchor revision, denoiser/readout calls, latent prompt
clamp, and the complete EXP-80 quality panel.

## Gate

Promote only a rollback arm that beats the compute-matched shadow-keep arm on
the quality Pareto frontier and keeps the fixed arm's PPL gain while improving
D1 or Rep-4. If shadow-keep alone changes quality, scope the effect as extra
compute. If rollback releases almost nothing, the policy is not informative.

Runner:
`models/ELF-torch/experiments/probe_elf/adaptive_rollback_exp88.py`.

## Result (2026-08-11)

The compute controls behave exactly as required: shadow-null matches
Standard-32 and shadow-keep matches the fixed random-anchor arm. Identity and
combined rollback release `32.5%` and `34.1%` of selected anchors, so the
policy is informative rather than inert.

| Arm | U-PPL | C-PPL | Gain | C-RL | C-D1 | C-Rep4 | release |
|---|---:|---:|---:|---:|---:|---:|---:|
| Standard-32 | 278.7 | 812.8 | .1845 | .0778 | .5945 | .0086 | -- |
| fixed random anchor | 192.1 | 561.3 | .2149 | .0874 | .5754 | .0121 | 0 |
| identity rollback | 183.2 | 515.1 | **.2208** | **.0884** | .5669 | .0111 | .325 |
| confidence rollback | 194.4 | 550.5 | .2121 | .0873 | .5734 | .0125 | .076 |
| combined rollback | **182.6** | **511.1** | .2167 | .0877 | .5680 | .0108 | .341 |

Rollback further improves PPL and slightly reduces Rep-4 relative to keeping
all anchors, but it lowers D1 again. It therefore fails the preregistered
Pareto gate and is not promoted to a multi-seed method result.
