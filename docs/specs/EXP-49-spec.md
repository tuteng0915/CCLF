# EXP-49 (D1): Intermediate Reconstruction Loss at B10 — Quick Fine-tuning

**Status**: COMPLETE (training phase); evaluation pending (EXP-51)
**Date**: 2026-07-25  
**Related**: EXP-43 (B11 reconstruction analysis), EXP-48 (h_10 SC inference), EXP-51 (evaluation)

---

## Motivation

EXP-43 showed that B11's residual update is reconstruction-helpful but decode-hostile.
KD training (kd_cr) distributes reconstruction to earlier layers (h_10 L_rec: kd_cr=76,
kd2=103–142 vs baseline=529).

**Hypothesis**: Adding an explicit L_rec supervision signal at B10 during fine-tuning will
further push reconstruction burden to earlier layers, improving h_10 SC quality and
potentially fixing kd_cr's broken SC interaction.

**Approach**: synthetic-data fine-tuning (random x0 ~ N(0, 0.2²)) with:
- L_total = λ_l2 × L_x0_pred + λ_d1 × MSE(final_layer(h_10), x0)
- λ_l2=1.0, λ_d1=0.5
- 500 steps, BATCH_SIZE=8, LR=3e-5

Baseline: `kd_cr` checkpoint. Output: `results/finetune_quick/d1.pt`

---

## Training Results

| Step | total_loss | aux_loss (D1 int_rec) | l2_loss |
|------|------------|----------------------|---------|
| 0    | 0.2183     | 0.1269               | 0.1549  |
| 50   | 0.0651     | 0.0460               | 0.0421  |
| 100  | 0.0600     | 0.0402               | 0.0399  |
| 200  | 0.0604     | 0.0403               | 0.0402  |
| 500  | ~0.060     | ~0.040               | ~0.040  |

Auxiliary loss (intermediate reconstruction at B10) dropped from 0.127 → 0.040 (~69% reduction).
Loss plateaued after step ~100 — likely because synthetic random data doesn't provide semantic
gradient signal beyond teaching the geometry of x0 prediction.

**Caveat**: Training on synthetic (random) embeddings limits generalization to real text.
A proper D1 experiment requires OWT-embedded data through train_step.py with
`lambda_intermediate_rec=0.1`.

---

## 下一步

- **EXP-51**: 测试 D1 checkpoint 的 h_10 SC 效果（intermediate SC pipeline）
- **如果 EXP-51 显示 D1 checkpoint 改善了 kd_cr 的 SC 交互**：
  → 用真实 OWT 数据重新训练 D1（src/train_step.py + finetune_owt_ELF-B-intermediate-rec.yml）
- **如果无改善**：synthetic data 不足以真正迁移重建负担；需要真实数据

---

## 结果文件

- 训练日志：`/tmp/exp49_d1.log`
- 训练 loss 曲线：`results/finetune_quick/d1_log.json`
- Fine-tuned checkpoint：`results/finetune_quick/d1.pt`
