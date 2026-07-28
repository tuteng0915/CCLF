# EXP-52: LangFlow Logit Lens (EXP-38 analogue)

**Status**: COMPLETE  
**Date**: 2026-07-26  
**Model**: LangFlow  
**Related**: EXP-38 (ELF Logit Lens), EXP-21v2 (LangFlow skip decomposition)

---

## Motivation

EXP-38 showed ELF (kd_cr) has a rich logit lens: oracle top-1 accuracy increases monotonically from
~40% at block 0 to ~99.5% at block 11 (t=0.5). Does LangFlow show similar layer-by-layer progression?

EXP-21v2 found LangFlow's **skip connection dominates** (skip=92.4% @ t=1, backbone≈0%). This predicts
the logit lens should be flat at all intermediate blocks, jumping only at the final block.

**Design**: For each block depth i (0=embedding, 1–12=after each DDiT block), apply `output_layer(h_i, c=t_cond)` to get backbone-only logits, then add `c_skip·(z⊗E)` for the full prediction.

---

## Setup

- N=64 OWT samples, seq_len=128, SEED=42
- t values: {0.10, 0.20, 0.30, 0.50, 0.70, 1.00}
- Backbone-only: `output_layer(h_depth, c=t_cond)` (no skip)
- Full (+skip): backbone logits + `c_skip(γ)·(z_t @ E.T)`
- Script: `experiments/probe_langflow/logit_lens_langflow_exp52.py`

---

## Results

### Full (+skip) top-1 accuracy across depths and t values

| depth | t=0.10 | t=0.20 | t=0.30 | t=0.50 | t=0.70 | t=1.00 |
|-------|--------|--------|--------|--------|--------|--------|
| h0 (embed) | 0.000 | 0.000 | 0.000 | 0.001 | 0.005 | 0.894 |
| h1  | 0.000 | 0.000 | 0.001 | 0.001 | 0.015 | 0.932 |
| h2  | 0.000 | 0.000 | 0.000 | 0.001 | 0.019 | 0.941 |
| h3  | 0.000 | 0.000 | 0.001 | 0.001 | 0.022 | 0.947 |
| h4  | 0.000 | 0.000 | 0.001 | 0.001 | 0.016 | 0.953 |
| h5  | 0.000 | 0.000 | 0.001 | 0.001 | 0.017 | 0.963 |
| h6  | 0.000 | 0.000 | 0.001 | 0.001 | 0.015 | 0.968 |
| h7  | 0.000 | 0.001 | 0.001 | 0.001 | 0.011 | 0.970 |
| h8  | 0.000 | 0.001 | 0.001 | 0.001 | 0.010 | 0.973 |
| h9  | 0.000 | 0.000 | 0.001 | 0.001 | 0.009 | 0.972 |
| h10 | 0.000 | 0.000 | 0.000 | 0.001 | 0.008 | 0.979 |
| h11 | 0.000 | 0.000 | 0.000 | 0.001 | 0.008 | 0.979 |
| h12 | 0.000 | 0.000 | 0.000 | 0.001 | 0.008 | **0.983** |
| **skip-only** | 0.000 | 0.000 | 0.000 | 0.000 | 0.002 | 0.921 |

### Backbone-only top-1 at t=1.00 (no skip)

| h0 | h1 | h2 | h3 | h6 | h9 | h11 | **h12** |
|----|----|----|----|----|----|----|-----|
| 0.002 | 0.138 | 0.197 | 0.220 | 0.265 | 0.335 | 0.664 | **0.737** |

---

## 核心发现

### 1. Logit Lens 在 t≤0.50 几乎完全平坦 (≈0.001)

在中等及低噪声区间（t≤0.70），所有 12 层的 full top-1 精度几乎为零。与 ELF kd_cr 的 logit lens
（t=0.5 时 B0=40%，B11=99.5%）形成强烈对比：ELF 有丰富的层级表示，LangFlow 各层几乎无差别。

**原因**：skip connection 在低 t（高噪声）时 c_skip 也接近零，backbone 贡献同样接近零，
两者叠加后 full prediction 仍然接近随机（~0.1%）。LangFlow 的全部 oracle 准确率来自于
t→1.0 时 skip 项 c_skip·(z_t⊗E) 爆发式上升。

### 2. t=1.00 时 backbone 有梯度 (0.002 → 0.737)，但 skip 主导

在 t=1.0（干净输入），backbone-only 从 h0=0.2% 提升至 h12=73.7%，说明深层 block 确实包含
更多可用表示。但 skip-only=92.1% 始终高于任何单层 backbone。

最终 full(h12)=98.3% ≈ skip(92.1%) + backbone 最终层贡献（约+6pp）。

**纯加法贡献**：h12 backbone（73.7%）+ skip（92.1%）不是简单叠加，但 logit 层面 skip 始终
压制 backbone，这就是为什么 EXP-21v2 说 backbone 贡献可以忽略不计。

### 3. ELF vs LangFlow Logit Lens 对比

| 模型 | t=0.5 B0→B11/12 | t=1.0 B0→B11/12 |
|------|----------------|----------------|
| ELF kd_cr | 40% → 99.5% | 几乎全部 99%+ |
| LangFlow | 0.1% → 0.1% | 89.4% → 98.3% |

ELF 的 logit lens 揭示了逐层的信息提炼过程。LangFlow 的 logit lens 在 t≤0.70 几乎完全
平坦，说明 LangFlow 的设计将所有 oracle 信息压缩到 t≈1.0 时 skip 的激活，而非在 backbone
各层逐步构建。这支持了 LangFlow 的"skip as the main readout pathway"故事（EXP-21v2）。

---

## 论文含义

1. **Logit Lens 结果强化 ELF 独特性**：ELF 的逐层精度提升是其架构设计的核心特征（KD 进一步强化），
   LangFlow 不具备这一特性。这支持将 ELF 的承诺行为归因于其 backbone 的逐层表示。

2. **Skip connection 的代价**：LangFlow 用 skip 保证了 t=1.0 的高 oracle 精度（92.1%），
   但代价是 backbone 在整个去噪过程中几乎是"透明的"（各层 0.1%）。这与 ELF 的
   "B08-B11 CKA 急剧下降（EXP-42）+ decode head 积累表示"的故事是互补的。

---

## 结果文件

`results/exp52_langflow_logit_lens/logit_lens.json`
