# Spec 05 — ELF-B Baseline Fine-tune

**Type**: training experiment  
**Priority**: high (required comparison point for all KD experiments)  
**Session**: runs in parallel with Idea A (no dependency between them)  
**Output**: `models/ELF/outputs/elf_b-owt-baseline/`

---

## Background

All training experiments (Idea 3, 2, B, 1) need a shared baseline: ELF-B fine-tuned on
OWT for the same number of steps / epochs as the KD run, with `lambda_kd=0`.

Without this, we cannot distinguish "L_KD helped" from "fine-tuning itself helped".

---

## Task

Run ELF-B fine-tune using the existing `train_owt_ELF-B.yml` config, starting from the
pretrained `embedded-language-flows/ELF-B-owt` checkpoint.

```bash
# on new-ncl, from ~/tt_workspace/model/CCLF/CCLF/models/ELF/
CUDA_VISIBLE_DEVICES=0,1,2,3 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
python src/train.py \
    --config src/configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint embedded-language-flows/ELF-B-owt \
    --config_override output_dir=outputs/elf_b-owt-baseline \
                      wandb_run_name=elf_b-owt-baseline \
                      epochs=5
```

---

## Metrics to track (W&B)

| Metric | Expected range | Notes |
|--------|---------------|-------|
| `loss` | ↓ from baseline | total loss |
| `l2_loss` | ↓ | denoiser velocity loss |
| `ce_loss` | ↓ | decoder CE loss |
| `kd_loss` | = 0 | must be zero |
| Gen.PPL | measure | GPT-2-large eval at end of each epoch |

---

## Output

- `outputs/elf_b-owt-baseline/` — checkpoints, metrics.jsonl, generated samples
- Record final Gen.PPL from epoch 5 as the baseline number for all comparisons

---

## Success criteria

- Training completes without NaN loss
- `kd_loss = 0.0` throughout (confirms `lambda_kd=0` code path)
- Gen.PPL at epoch 5 ≤ original ELF-B-owt Gen.PPL (fine-tuning should not regress)
