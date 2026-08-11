# EXP-91 Spec — Triggered Subset Flow Forcing

**Status:** DONE / NEGATIVE (200-STEP PAIRED PILOT)
**Launch evidence:** EXP-82 shows random-position, position-correct temporary
anchors outperform top-confidence selection in three formal U/C panels, while
shuffled content is catastrophic.

## Training hypothesis

The useful asymmetry is not a fixed linguistic direction or a local diffusion
clock. It is a heterogeneous state in which a random subset has reached a
predicted-clean representation and the remainder is still noisy. Train the
model to denoise the unresolved subset under that context.

For a real OWT clean latent `x0`, scalar time `t`, and noisy state `z_t`, a
frozen teacher supplies `x_hat_t`. On half of training examples choose a random
50% subset `A` and construct

```text
z_mix[i] = x_hat_t[i] if i in A else z_t[i].
```

The student receives `(z_mix, x_hat_t, t)`. Flow loss is evaluated only on
`U = not A`; the other half of examples use ordinary synchronous states and
all-token loss. This is compared with a matched continued-training control
using the same real OWT bank, optimizer, steps, noise, and time draws.

## Pilot

- ELF base, length 128, real OWT train bank;
- 200 steps, batch 4, seed 42;
- freeze encoder, teacher, and bottom 20 transformer blocks;
- train top four blocks, `final_layer`, and `self_cond_proj`;
- transition examples draw `t` uniformly from `.20--.40`;
- synchronous examples retain the native logit-normal time distribution.

## Gate

Both arms must retain healthy Standard-32 generation. Promotion additionally
requires the anchor-trained checkpoint to improve the random-anchor-minus-
standard interaction over the matched control on paired U/C generation,
without worsening D1, Rep-4, degeneration, or prompt gain. Falling training
loss alone is not a pass.

Runner:
`models/ELF-torch/experiments/probe_elf/train_subset_flow_exp91.py`.

## Training checkpoint (2026-08-11)

Both matched 200-step runs completed. The synchronous control reaches
validation losses `sync=.7133`, `anchor=.7361`; subset-flow reaches
`sync=.7213`, `anchor=.7320`. Thus the targeted anchor-conditioned loss is only
slightly better (`-.0041`) while ordinary validation is slightly worse
(`+.0080`). Paired held-out generation is now complete:

| Checkpoint | Arm | U-PPL | C-PPL | Prompt gain | C-RL | C-D1 | C-Rep4 | C-Deg. |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| matched control | Standard | 296.3 | 599.1 | .2353 | .0762 | .5257 | .0098 | .0156 |
| matched control | Random anchor | 211.7 | **416.9** | **.2943** | .0810 | .5078 | .0131 | .0234 |
| subset-flow | Standard | 301.8 | 610.0 | .2334 | .0745 | .5301 | .0120 | .0234 |
| subset-flow | Random anchor | **208.6** | 419.9 | .2817 | **.0816** | **.5097** | .0134 | .0391 |

The random-minus-Standard interaction improves by about 8 PPL only because
the subset-flow Standard baseline becomes worse. Absolute conditioned PPL and
prompt gain regress, and degeneration rises. The preregistered gate fails; do
not expand this formulation to more seeds.
