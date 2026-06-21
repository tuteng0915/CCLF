# ELF-B CCLF Experiments — Runbook

## Overview

This runbook documents the 4 queued fine-tuning experiments (+ 1 already running) for the CCLF paper.
All experiments fine-tune **ELF-B-owt** (HuggingFace: `embedded-language-flows/ELF-B-owt`)
on OpenWebText for 1 additional epoch, using 4 GPUs.

| Experiment | Key change | Output dir |
|---|---|---|
| kd2 | L_KD, gate [0.25, 0.95], τ=4 | `outputs/elf_b-owt-kd2` |
| spec-05 | Baseline (no modification) | `outputs/elf_b-owt-baseline` |
| ideaC | Cliff importance time sampling | `outputs/elf_b-owt-ideaC` |
| kd-cr | L_KD, narrow gate [0.30, 0.60], τ=4 | `outputs/elf_b-owt-kd-cr` |
| kd-b | L_KD + position-adaptive mask | `outputs/elf_b-owt-kd-b` |

Run them **sequentially** (one 4-GPU job at a time). Each takes ~50 hours.
Checkpoints are saved every 10% of epoch (~5 hours), so evaluation can start early.

---

## 1. Hardware Requirements

- **4× NVIDIA A40 (46 GB)** per experiment (or equivalent ≥ 45 GB VRAM)
- CUDA ≥ 12.4
- NVLink not required; PCIe multi-GPU works

---

## 2. Environment Setup

### 2a. Conda environment

```bash
conda create -n elf python=3.11.15 -y
conda activate elf
```

### 2b. Install packages (exact versions)

```bash
# JAX with CUDA 12 support
pip install "jax[cuda12]==0.10.1" jaxlib==0.10.1

# Core ML
pip install flax==0.12.7 optax==0.2.8

# HuggingFace
pip install transformers==5.9.0 datasets==4.8.5

# Eval
pip install torch==2.6.0 sacrebleu==2.6.0 rouge_score==0.1.2

# Logging
pip install wandb==0.27.2 tqdm

# Other
pip install pyyaml numpy orbax-checkpoint
```

> **Note:** `optax.contrib.muon` is included in optax ≥ 0.2.8 — no separate install needed.

---

## 3. Code Setup

### 3a. Copy the code directory

Copy the entire ELF source directory from this machine:

```
/home/wjzhang/tt_workspace/model/CCLF/CCLF/models/ELF/
```

The working directory for all commands below is:

```bash
cd /path/to/ELF/src
```

### 3b. Critical modified files (already patched in the copied code)

The following files have been modified for these experiments — **do not overwrite with originals**:

| File | Change |
|---|---|
| `train.py` | Fixed `last_save_epoch` to use current `steps_per_epoch` (not checkpoint epoch) |
| `configs/config.py` | Added `kd_gate_k/low/high`, `kd_position_mask`; `save_freq` default = `100.0` (float) |
| `train_step.py` | Added L_KD (Hinton temperature, gate, position mask) |
| `utils/sampling_utils.py` | Added `cliff_importance` time schedule |
| `configs/training_configs/train_owt_ELF-B.yml` | Changed `save_freq: 1` → `save_freq: 1.0` |

---

## 4. Pre-download Models and Data

Run this once to cache everything locally (needs HuggingFace token if repos are private):

```bash
python - <<'EOF'
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download

# Dataset (~20 GB)
print("Downloading openwebtext-t5 ...")
load_dataset("embedded-language-flows/openwebtext-t5", split="train")

# Tokenizer
print("Downloading t5-small tokenizer ...")
AutoTokenizer.from_pretrained("t5-small")

# Model checkpoint + encoder
print("Downloading ELF-B-owt checkpoint ...")
snapshot_download("embedded-language-flows/ELF-B-owt")
snapshot_download("embedded-language-flows/t5_small_encoder_jax")

print("Done.")
EOF
```

If the datasets are already cached on the source machine, copy the HuggingFace cache:

```bash
rsync -av ~/.cache/huggingface/ other-machine:~/.cache/huggingface/
```

---

## 5. WandB Setup

```bash
wandb login   # enter API key (tuteng0915)
# project: "elf", entity: "tuteng0915-national-university-of-singapore"
```

---

## 6. Run Commands

All experiments share these common flags:

```bash
COMMON_ENV="
  XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false'
  XLA_PYTHON_CLIENT_PREALLOCATE=false
  XLA_PYTHON_CLIENT_ALLOCATOR=platform
"

COMMON_ARGS="
  --config configs/training_configs/train_owt_ELF-B.yml
  --config_override resume=embedded-language-flows/ELF-B-owt
  --config_override global_batch_size=32
  --config_override grad_accum_steps=4
  --config_override epochs=6
  --config_override save_freq=0.1
  --config_override online_eval=false
  --config_override use_wandb=true
"
```

> **Effective batch:** 32 samples/step × 4 grad\_accum = 128  
> **LR:** blr=0.001 × (32×4)/256 = 5e-4 peak (warmup over 0.5 epoch ≈ 152k steps)  
> **steps/epoch:** 9,737,184 / 32 = 304,287  
> **Time/epoch:** ~50 hours (4× A40)  
> **Checkpoints:** every 30,429 steps ≈ every 5 hours

### 6a. Experiment 1 — kd2 (L_KD, wide gate) [RUNNING on current machine]

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup env \
  XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false' \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  python train.py \
  --config configs/training_configs/train_owt_ELF-B.yml \
  --config_override resume=embedded-language-flows/ELF-B-owt \
  --config_override output_dir=outputs/elf_b-owt-kd2 \
  --config_override wandb_run_name=elf_b-owt-kd2 \
  --config_override global_batch_size=32 \
  --config_override grad_accum_steps=4 \
  --config_override epochs=6 \
  --config_override save_freq=0.1 \
  --config_override lambda_kd=1.0 \
  --config_override kd_temperature=4.0 \
  --config_override online_eval=false \
  --config_override use_wandb=true \
  > outputs/elf_b-owt-kd2-4gpu.log 2>&1 &
echo "PID=$!"
```

### 6b. Experiment 2 — spec-05 (Baseline)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup env \
  XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false' \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  python train.py \
  --config configs/training_configs/train_owt_ELF-B.yml \
  --config_override resume=embedded-language-flows/ELF-B-owt \
  --config_override output_dir=outputs/elf_b-owt-baseline \
  --config_override wandb_run_name=elf_b-owt-baseline \
  --config_override global_batch_size=32 \
  --config_override grad_accum_steps=4 \
  --config_override epochs=6 \
  --config_override save_freq=0.1 \
  --config_override online_eval=false \
  --config_override use_wandb=true \
  > outputs/elf_b-owt-baseline-4gpu.log 2>&1 &
echo "PID=$!"
```

### 6c. Experiment 3 — ideaC (Cliff Importance Sampling)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup env \
  XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false' \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  python train.py \
  --config configs/training_configs/train_owt_ELF-B.yml \
  --config_override resume=embedded-language-flows/ELF-B-owt \
  --config_override output_dir=outputs/elf_b-owt-ideaC \
  --config_override wandb_run_name=elf_b-owt-ideaC \
  --config_override global_batch_size=32 \
  --config_override grad_accum_steps=4 \
  --config_override epochs=6 \
  --config_override save_freq=0.1 \
  --config_override time_schedule=cliff_importance \
  --config_override online_eval=false \
  --config_override use_wandb=true \
  > outputs/elf_b-owt-ideaC-4gpu.log 2>&1 &
echo "PID=$!"
```

### 6d. Experiment 4 — kd-cr (L_KD, Narrow Gate [0.30, 0.60])

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup env \
  XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false' \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  python train.py \
  --config configs/training_configs/train_owt_ELF-B.yml \
  --config_override resume=embedded-language-flows/ELF-B-owt \
  --config_override output_dir=outputs/elf_b-owt-kd-cr \
  --config_override wandb_run_name=elf_b-owt-kd-cr \
  --config_override global_batch_size=32 \
  --config_override grad_accum_steps=4 \
  --config_override epochs=6 \
  --config_override save_freq=0.1 \
  --config_override lambda_kd=1.0 \
  --config_override kd_temperature=4.0 \
  --config_override kd_gate_low=0.30 \
  --config_override kd_gate_high=0.60 \
  --config_override online_eval=false \
  --config_override use_wandb=true \
  > outputs/elf_b-owt-kd-cr-4gpu.log 2>&1 &
echo "PID=$!"
```

### 6e. Experiment 5 — kd-b (L_KD + Position-Adaptive Mask)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup env \
  XLA_FLAGS='--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false' \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_PYTHON_CLIENT_ALLOCATOR=platform \
  python train.py \
  --config configs/training_configs/train_owt_ELF-B.yml \
  --config_override resume=embedded-language-flows/ELF-B-owt \
  --config_override output_dir=outputs/elf_b-owt-kd-b \
  --config_override wandb_run_name=elf_b-owt-kd-b \
  --config_override global_batch_size=32 \
  --config_override grad_accum_steps=4 \
  --config_override epochs=6 \
  --config_override save_freq=0.1 \
  --config_override lambda_kd=1.0 \
  --config_override kd_temperature=4.0 \
  --config_override kd_position_mask=true \
  --config_override online_eval=false \
  --config_override use_wandb=true \
  > outputs/elf_b-owt-kd-b-4gpu.log 2>&1 &
echo "PID=$!"
```

---

## 7. Monitoring

### 7a. Check GPU utilization

```bash
watch -n5 nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
```

Expected: all 4 GPUs at ~95–100% utilization, ~35–42 GB memory each.

### 7b. Tail training log

```bash
tail -f outputs/elf_b-owt-kd2-4gpu.log | grep -E "Step|Saved|Error"
```

### 7c. Key log lines to watch

```
# Good: training progressing
INFO - engine - Step 125000: loss=..., l2=..., ce=..., kd=..., lr=..., steps/sec=...

# Good: checkpoint saved
INFO - __main__ - Saved checkpoint at epoch 0.41 (step 124755)

# Bad: OOM
ERROR bfc_allocator: GPU_0_bfc ran out of memory
# Fix: ensure XLA_PYTHON_CLIENT_ALLOCATOR=platform is set

# Bad: loss is NaN
loss=nan
# Fix: reduce lr or check batch
```

### 7d. Expected training speed

| Condition | sps | Steps to 1st ckpt | Wall time |
|---|---|---|---|
| 4× A40, warmed up | ~1.7–2.0 | ~30,000 | ~4.5–5 hours |

### 7e. WandB dashboard

Project: `elf` — Entity: `tuteng0915-national-university-of-singapore`

Key metrics to watch:
- `train_l2_loss`: should decrease below 0.70 (baseline reference)
- `train_kd_loss`: (KD experiments only) should decrease from ~100 → 15–20
- `lr`: should warm up over first 0.5 epoch (152k steps) to peak ~5e-4

---

## 8. Checkpoint Evaluation

Run after each checkpoint appears in `outputs/<exp>/checkpoint_*/`.

### 8a. Find the latest checkpoint

```bash
ls -lt outputs/elf_b-owt-kd2/checkpoint_* | head -3
```

### 8b. Generate samples + compute Gen.PPL

```bash
CHECKPOINT=outputs/elf_b-owt-kd2/checkpoint_124755   # adjust step number

CUDA_VISIBLE_DEVICES=0,1,2,3 python eval.py \
  --config configs/training_configs/train_owt_ELF-B.yml \
  --checkpoint_path $CHECKPOINT \
  --config_override "output_dir=outputs/eval_kd2_step124755" \
  --config_override "num_samples=1024" \
  --config_override "online_eval=false" \
  --config_override "sampling_configs_path=configs/sampling_configs/uncond_sampling_configs.yml"

# Then compute Gen.PPL offline (saves GPU for training)
# PPL eval script: specs/idea_a_dec_sc/eval_ppl_offline.py
CUDA_VISIBLE_DEVICES=7 python specs/idea_a_dec_sc/eval_ppl_offline.py \
  outputs/eval_kd2_step124755
```

### 8c. Baseline Gen.PPL reference (ELF-B-owt, no fine-tuning)

| Sampler | Steps | Gen.PPL |
|---|---|---|
| SDE γ=1.5 | 32 | ~24 (paper) |
| ODE uniform | 4 | 175 |
| ODE uniform | 8 | 962 |

A fine-tuned model should ideally **improve** on the 32-step SDE number.

---

## 9. Troubleshooting

### OOM on first step
```bash
# Add this env var (already in commands above)
XLA_PYTHON_CLIENT_ALLOCATOR=platform
```

### save_freq=0.1 not working (no checkpoint after 5+ hours)
Check that `configs/training_configs/train_owt_ELF-B.yml` has `save_freq: 1.0` (float, not `1`).
Also verify `train.py` has the fixed `last_save_epoch` computation (not `= resume_epoch_fractional`).

### Rendezvous timeout (multi-GPU hang)
```
ERROR rendezvous.cc: This thread has been waiting for ... 10 seconds
```
Usually caused by OOM on one GPU blocking the all-reduce. Fix the OOM first.

### Loss immediately NaN
Usually LR too high. The warmup handles this; if it happens at step 1, check the checkpoint loaded correctly.

### Resuming a crashed run
The code auto-resumes: just re-run the same command. It will find the latest checkpoint in `output_dir` automatically.

---

## 10. Experiment Design Summary

### What each experiment tests

**spec-05 (baseline):** Does simple fine-tuning from ELF-B-owt change Gen.PPL at all? Reference point.

**kd2:** Does distilling the decoder's token predictions into the linear branch's intermediate distributions help? Gate [0.25, 0.95] covers the full commitment plateau.

**kd-cr (commit-release):** Same KD but gate [0.30, 0.60] targets only the active commitment zone (where G(t) rises from 60% to 90%). Hypothesis: narrower, more informative signal.

**ideaC (cliff sampling):** Train more on t∈[0.10, 0.35] (the commitment cliff). Hypothesis: the model currently underfits this critical region because logit-normal sampling underweights it.

**kd-b (position mask):** Apply KD only on positions where the linear branch is wrong AND confident. Hypothesis: avoids reinforcing already-correct or already-wrong-and-stuck positions.

### Key scientific question
Does intervening at the commitment cliff (via KD, sampling re-weighting, or masking) improve generation quality (Gen.PPL), and does it close the G(t) gap measured in the probe?
