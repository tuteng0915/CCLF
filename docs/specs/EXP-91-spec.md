# EXP-91 Spec — Triggered Subset Flow Forcing

**Status:** 200-STEP PAIRED TRAINING DONE / GENERATION EVAL RUNNING
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
(`+.0080`). Per the gate, this is not yet a method result: paired unconditional
and conditional generation on both checkpoints is running.
