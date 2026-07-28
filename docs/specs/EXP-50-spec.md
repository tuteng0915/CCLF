# EXP-50 (D3): Gram-Matrix Alignment Loss — Quick Fine-tuning

**Status**: COMPLETE (training phase); evaluation pending (EXP-51)
**Date**: 2026-07-25  
**Related**: EXP-44 Phase 2 (self_cond_proj causality), EXP-51 (evaluation)

---

## Motivation

EXP-44 Phase 2 identified that `self_cond_proj.weight[:, 512:]` (512×512, the x̂_t consumer)
and `final_layer.linear.weight` (512×768, the x̂_t producer) must be geometrically compatible
to enable effective SC interaction.

When kd_cr's `self_cond_proj` is swapped with kd2's, SC flips from harmful to helpful (ΔΔI≈−182).
This suggests kd_cr's `final_layer` and `self_cond_proj` have mismatched column-space geometry.

**Hypothesis**: Minimizing the Gram-matrix alignment loss between these two weight matrices
will improve SC interaction without retraining the full backbone:

```
G_F = (F @ F.T) / d_F,  F = final_layer.linear.weight [512, 768]
G_P = (P @ P.T) / d_P,  P = self_cond_proj.weight[:, 512:] [512, 512]
L_align = ||(G_F / tr_F) − (G_P / tr_P)||_F²
```

**Approach**: synthetic-data fine-tuning, only training `final_layer` + `self_cond_proj`:
- L_total = λ_l2 × L_x0_pred + λ_d3 × L_align
- λ_l2=1.0, λ_d3=0.1
- 500 steps, BATCH_SIZE=8, LR=3e-5, ~919K trainable params

Baseline: `kd_cr` checkpoint. Output: `results/finetune_quick/d3.pt`

---

## Training Results

| Step | total_loss | aux_loss (align) | l2_loss | Gram align |
|------|------------|-----------------|---------|------------|
| 0    | 0.1733     | 0.0096          | 0.1724  | 0.009555   |
| 50   | 0.1089     | 0.0085          | 0.1080  | —          |
| 100  | 0.0929     | 0.0076          | 0.0921  | —          |
| 300  | 0.0616     | 0.0052          | 0.0611  | —          |
| 500  | ~0.059     | ~0.0039         | ~0.058  | 0.003573   |

**Gram alignment reduced 63%**: 0.009555 → 0.003573 (Δ = −0.005981).

The alignment loss steadily decreases throughout training, suggesting the optimizer
can meaningfully reduce the geometric mismatch between the two matrices.

---

## 下一步

- **EXP-51**: 测试 D3 checkpoint 在 intermediate SC pipeline 和 EXP-44 Phase 2 swap test 中的效果
- **如果 EXP-51 显示 Gram 对齐有效**：
  → 在真实 OWT 数据上使用 src/train_step.py + finetune_owt_ELF-B-align.yml 进行完整微调
  → 测试更强的 lambda_align (0.1, 0.5)
- **如果无效**：Gram 对齐虽然减少了矩阵差异，但不足以修复 SC interaction

---

## 结果文件

- 训练日志：`/tmp/exp50_d3.log`
- 训练 loss 曲线：`results/finetune_quick/d3_log.json`
- Fine-tuned checkpoint：`results/finetune_quick/d3.pt`
- 初始 Gram alignment: 0.009555 → 0.003573
