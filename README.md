# PyTorch ELF

PyTorch version of [ELF: Embedded Language Flows](https://arxiv.org/abs/2605.10938).

## Installation

Create a conda environment named `elf` and install the dependencies:

```bash
conda create -n elf python=3.10 -y
conda activate elf
pip install -r requirements.txt
```

Then log in to WandB to track your experiments if needed:

```bash
wandb login YOUR_WANDB_API_KEY
```

## Converted Checkpoints

We provide PyTorch-converted versions of the official JAX checkpoints on HuggingFace:

| Model | Task | Params | HuggingFace Repo |
| --- | --- | --- | --- |
| ELF-B | OpenWebText (unconditional) | 105M | [embedded-language-flows/ELF-B-owt-torch](https://huggingface.co/embedded-language-flows/ELF-B-owt-torch) |
| ELF-M | OpenWebText (unconditional) | 342M | [embedded-language-flows/ELF-M-owt-torch](https://huggingface.co/embedded-language-flows/ELF-M-owt-torch) |
| ELF-L | OpenWebText (unconditional) | 652M | [embedded-language-flows/ELF-L-owt-torch](https://huggingface.co/embedded-language-flows/ELF-L-owt-torch) |
| ELF-B | XSum (summarization) | 105M | [embedded-language-flows/ELF-B-xsum-torch](https://huggingface.co/embedded-language-flows/ELF-B-xsum-torch) |
| ELF-B | WMT14 De-En (translation) | 105M | [embedded-language-flows/ELF-B-de-en-torch](https://huggingface.co/embedded-language-flows/ELF-B-de-en-torch) |

These are pulled automatically via `--checkpoint_path <hf-repo-id>` — no manual download needed.

## Reference Results

The PyTorch port targets parity with the JAX reference numbers from the
paper. Small differences (≲1 PPL, ≲0.5 BLEU/ROUGE) are expected due to bf16
vs. JAX TPU numerics and sampling stochasticity.

**Unconditional generation (OpenWebText), expected:**

| Model | Sampling | Gen. PPL ↓ | Entropy ↑ |
| --- | --- | --- | --- |
| ELF-B (105M) | 32-step SDE | 24.1 | 5.15 |
| ELF-M (342M) | 64-step SDE | 21.7 | 5.18 |
| ELF-L (652M) | 64-step SDE | 23.3 | 5.28 |

Gen. PPL is computed under a frozen GPT-2 Large; entropy is unigram entropy
over the generated tokens. Default sampling configs
(`src/configs/sampling_configs/uncond_sampling_configs.yml`) use SC-CFG=3 and
γ=1.5 (32-step) or γ=1.0 (64-step).

**Conditional generation (ELF-B), expected on the validation set:**

| Task | Metric | Reference (paper, test) | Validation |
| --- | --- | --- | --- |
| WMT14 De-En | BLEU ↑ | 26.4 | ≈ 26.7 |
| XSum | ROUGE-1 ↑ | 36.0 | ≈ 36.3 |
| XSum | ROUGE-2 ↑ | 12.2 | ≈ 12.5 |
| XSum | ROUGE-L ↑ | 27.8 | ≈ 28.1 |

Default conditional sampling
(`src/configs/sampling_configs/cond_sampling_configs.yml`): 64-step ODE,
CFG=2, SC-CFG=1.

The paper numbers were computed on TPU v5p-64; numbers from this PyTorch port
on 8× L40S / H200 should land within sampling noise (typically <1 PPL or
<0.5 metric points).

## CCLF Fine-tuned Checkpoints

The following local checkpoints were produced by the CCLF project and are available
under `converted/` after running the JAX→PyTorch conversion (see below):

| File | Description | SDE-32 PPL |
| --- | --- | --- |
| `converted/elf_b-owt-baseline_torch.pt` | Official ELF-B-OWT (step 95085), converted from JAX | ~24.1 |
| `converted/elf_b-owt-kd2_torch.pt` | Decode-teacher KD fine-tune (400k steps) | ~24.4 |
| `converted/elf_b-owt-kd-cr_torch.pt` | Commit-release KD fine-tune (700k steps) | ~63.3 |

Load any converted checkpoint by passing its absolute path to `--checkpoint_path`.

## JAX → PyTorch Conversion

To convert a JAX/Orbax ELF checkpoint to a PyTorch `.pt` file:

```bash
# Install JAX dependencies first (separate from main requirements)
pip install -r requirements_convert.txt

# Convert (uses ema_params1 by default — correct for inference)
python convert_jax_to_torch.py \
    --jax_ckpt /path/to/orbax/checkpoint_NNNNN \
    --out       converted/my_checkpoint.pt

# Optionally validate against a reference HF checkpoint
python convert_jax_to_torch.py \
    --jax_ckpt /path/to/checkpoint_95085 \
    --out       converted/baseline.pt \
    --validate  /path/to/ELF-B-owt-torch/checkpoint_95085
```

The validation compares the converted weights against the official HF EMA params;
a correct conversion yields max absolute diff < 1e-6.

## PBS Job Scripts

Ready-to-submit PBS scripts are in `scripts/`. Before using, set two things in each file:

1. `#PBS -P ds_ccds_wei.lu` → replace with your cluster project code
2. `REPO=/path/to/ELF-torch` → absolute path on the cluster

| Script | What it does | Walltime | GPUs |
| --- | --- | --- | --- |
| `scripts/eval_owt_uncond.pbs` | Full ODE+SDE grid eval (12 configs, 1000 samples) | 4 h | 1 |
| `scripts/eval_xsum.pbs` | XSum ROUGE eval (64-step ODE, 1000 samples) | 4 h | 1 |
| `scripts/eval_de_en.pbs` | WMT14 De-En BLEU eval (64-step ODE, 1000 samples) | 2 h | 1 |
| `scripts/train_owt_kd.pbs` | KD fine-tuning on OWT, resumes from any checkpoint | 24 h | 4 |

Override parameters via `-v VAR=value` at submission time, e.g.:

```bash
# Eval our kd2 checkpoint
qsub -v CKPT=/path/to/converted/elf_b-owt-kd2_torch.pt,TAG=kd2 \
    scripts/eval_owt_uncond.pbs

# Eval official XSum baseline
qsub scripts/eval_xsum.pbs

# Start a new KD training run
qsub -v TAG=kd3,EPOCHS=3,WANDB_KEY=your_key \
    scripts/train_owt_kd.pbs
```

## Training

Launch single-GPU training:

```bash
bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
```

Launch multi-GPU (single-host) training:

```bash
NGPU=8 bash scripts/launch.sh train src/configs/training_configs/train_owt_ELF-B.yml
```

Available training configs:

- `src/configs/training_configs/train_owt_ELF-B.yml` — ELF-B on OpenWebText
- `src/configs/training_configs/train_owt_ELF-M.yml` — ELF-M on OpenWebText
- `src/configs/training_configs/train_owt_ELF-L.yml` — ELF-L on OpenWebText
- `src/configs/training_configs/train_de-en_ELF-B.yml` — WMT14 De-En machine translation
- `src/configs/training_configs/train_xsum_ELF-B.yml` — XSum abstractive summarization

**Estimated wall-clock:** ~4 h per epoch on 8× H200 (OpenWebText, ELF-B,
global batch size 512, bf16). The default ELF-B OWT run is 5 epochs.

## Evaluation

Run evaluation against the converted checkpoints on HuggingFace. We recommend
passing `use_bf16=true` (matches the bf16 autocast used at training time) and
`use_compile=true` (wraps the eval model in `torch.compile`) for a ~3–4×
speedup on consumer GPUs:

**Unconditional generation (OpenWebText):**

```bash
# ELF-B (105M)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-owt-torch \
    --config_override use_bf16=true --config_override use_compile=true

# ELF-M (342M)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_owt_ELF-M.yml \
    --checkpoint_path embedded-language-flows/ELF-M-owt-torch \
    --config_override use_bf16=true --config_override use_compile=true

# ELF-L (652M)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_owt_ELF-L.yml \
    --checkpoint_path embedded-language-flows/ELF-L-owt-torch \
    --config_override use_bf16=true --config_override use_compile=true
```

**Conditional generation (XSum / WMT14 De-En):**

```bash
# XSum (ROUGE)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_xsum_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-xsum-torch \
    --config_override use_bf16=true --config_override use_compile=true

# WMT14 De-En (BLEU)
NGPU=8 bash scripts/launch.sh eval src/configs/training_configs/train_de-en_ELF-B.yml \
    --checkpoint_path embedded-language-flows/ELF-B-de-en-torch \
    --config_override use_bf16=true --config_override use_compile=true
```

### Eval config flags

| Flag | Default | What it does |
| --- | --- | --- |
| `use_bf16` | `true` | Wraps the sampling forward in `torch.amp.autocast('cuda', dtype=bfloat16)`. Mirrors the training-time precision; output heads stay fp32. |
| `use_compile` | `false` | Wraps the eval model in `torch.compile`. First batch is slower due to tracing; subsequent batches run materially faster. |

Both flags are also editable in the YAML config under the same names. You can also run the standalone
PPL script afterwards:

```bash
python scripts/eval_ppl.py \
    --input outputs/<run>/<sampling_dir>/all_generated_*.jsonl \
    --batch_size 16
```
