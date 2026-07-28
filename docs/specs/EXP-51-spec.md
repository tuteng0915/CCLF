# EXP-51: SC Evaluation for D1 and D3 Fine-tuned Checkpoints

**Status**: COMPLETE  
**Date**: 2026-07-25  
**Related**: EXP-48 (h_10 SC, kd_cr/kd2), EXP-49 (D1 fine-tuning), EXP-50 (D3 fine-tuning)

---

## Setup

Uses the same EXP-48 pipeline: natural / none / α=0.0 (h10) SC arms, PPL via GPT-2 Large.
Checkpoints: `kd_cr` (baseline), `d1.pt` (D1 fine-tuned), `d3.pt` (D3 fine-tuned).

Script: `experiments/probe_elf/eval_finetuned_sc_exp51.py`

---

## Results

| arm | kd_cr | D1 | D3 |
|-----|-------|----|----|
| natural (std h_11 SC) | 303.4 | **1.54** | 116.2 |
| none (zero SC) | 186.8 | 1.49 | **5.89** |
| h10 (α=0.0) | 192.1 | 1.58 | **5.88** |
| **I(h10)** | **−111.3** | **+0.04** | **−110.3** |

---

## 分析

### D1 checkpoint: 完全崩溃 (PPL ≈ 1.5)

所有三个 arm 的 PPL 均约为 1.5，说明 D1 微调后的模型生成了退化文本
（单一 token 重复或 EOS 泛滥），GPT-2 Large 对其赋予极高概率。

**根因**：synthetic random x0 ~ N(0, 0.2²) 没有语义结构。模型在 500 步后
学会将任意输入映射到某个单一输出分布（最小化随机 x0 的重建误差→输出接近零），
破坏了正常的生成能力。

**结论**：D1 必须在真实 OWT 嵌入数据上训练（通过 train.py + finetune_owt_ELF-B-intermediate-rec.yml），
不能用 synthetic data 验证。

### D3 checkpoint: 部分崩溃 (none/h10 PPL ≈ 6)

- natural PPL 从 303→116（改善了），但 none/h10 PPL 降到 5.9（过度改善→退化迹象）
- I(h10) = −110.3，与 kd_cr 几乎相同 (−111.3)
- none ≈ h10 ≈ 5.9，说明两个 arm 行为几乎相同，不再有 SC 区分度

**根因**：D3 只更新了 final_layer + self_cond_proj，这两个模块是解码接口的核心。
在 synthetic 随机数据上调整它们改变了 decode 接口，使输出的 token logits 退化
（可能输出极低困惑度的重复文本），掩盖了真实的 SC 效果。

**结论**：D3 同样需要真实数据训练才能有效验证。

---

## 总体结论：Synthetic Data Fine-tuning 不足以验证 D1/D3

这一组实验的核心教训是：合成随机 x0 不能替代真实 OWT 嵌入数据。
实验链 EXP-49→EXP-50→EXP-51 验证了损失下降（D1: aux 0.127→0.040；D3: Gram 0.0096→0.0036），
但 checkpoint 品质无法用于真实评估。

**清楚的结论链（EXP-47→48→51）**：

| 实验 | 结论 |
|------|------|
| EXP-47 | h_10 SC 方向单调改善（bug 版，趋势可信） |
| EXP-48 | kd2 h_10 SC: PPL 247→91（I=−157）；**kd_cr 仍需修复** |
| EXP-51 | D1/D3 synthetic 微调无效；需真实 OWT 数据 |

---

## 下一步

**D2 已完成（无需重训，推荐立即应用于 kd2）**:
- EXP-48 的 h_10 SC 在 kd2 上 I=−157，效果显著
- 在 EXP-36v2 pipeline 中验证 h_10 SC 对 kd2 的效果（消除 pipeline 差异）

**D1/D3 需要真实数据训练**:
- Option A: 修改 train.py 支持 `max_steps` 字段，运行 500–2000 步真实 OWT 训练
- Option B: 直接在 train.py 配置中设置非常小的 dataset subset（如 HuggingFace streaming，
  limit first 1000 batches）
- 优先级：D2 已经给出清晰结论，D1/D3 是 paper 的进一步强化，可在 kd2 h_10 SC 验证后推进

---

## 结果文件

`results/exp51_eval_finetuned_sc/results.json`
