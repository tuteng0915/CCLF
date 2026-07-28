# EXP-53: LangFlow T_stable / Never-Commit Rate (EXP-16v2 analogue)

**Status**: COMPLETE  
**Date**: 2026-07-26  
**Model**: LangFlow  
**Related**: EXP-16v2 (ELF T_stable), EXP-25 (LangFlow T_first coarse-to-fine), EXP-22 (LangFlow oracle cliff)

---

## Motivation

EXP-16v2 found that ELF baseline has 25.1% never-stably-committed positions while kd_cr has only 0.53%.
This "never-commit" rate is a key paper metric. LangFlow's equivalent has not been measured with K=3
consecutive stability — only T_first (H<1.0 AND top1==gt) was computed in EXP-25 (1.37%).

**Design**: Fixed-ε oracle probe (same noise draw across all t values per position), K=3 consecutive
correct top-1 oracle predictions required for T_stable. Dense t grid from t=0.03 to t=1.0 (51 values).

**Note**: EXP-53 first run had a bug (t_grid cut off at t≈0.813), giving artificially high never-stable
of 79.4%. Fixed by using `np.linspace(0.03, 1.0, 51)` to ensure coverage of LangFlow's cliff at t≈0.83-0.93.

---

## Setup

- N=64 OWT samples, seq_len=128, K=3 consecutive, SEED=42
- t grid: 51 values uniformly from 0.03 to 1.0
- Script: `experiments/probe_langflow/tstable_langflow_exp53.py`

---

## Results

### Never-commit comparison

| Model | never-stable (K=3) | mean T_stable |
|-------|-------------------|---------------|
| ELF baseline | 25.1% | ~0.50 (estimated) |
| **LangFlow** | **4.79%** | **0.840** |
| ELF kd2 | 0.98% | ~0.20 |
| ELF kd_cr | 0.53% | ~0.18 |

LangFlow: 4.79% never-stable — between ELF baseline and kd checkpoints in commitment quality,
but much LATER in the diffusion timeline (mean T_stable=0.840 vs ELF kd_cr ~0.18).

### G_oracle(t) — oracle top-1 accuracy

| t | G_oracle |
|---|---------|
| 0.030 | 3.92% (freq-mode prediction, not genuine signal) |
| 0.127–0.418 | 3.8–4.4% (flat, near-chance) |
| 0.515–0.612 | 2.9% (DIP — intermediate difficulty) |
| 0.709 | 5.6% (pre-cliff onset) |
| **0.806** | **31.6%** ← cliff begins |
| 0.845 | 53.5% |
| 0.884 | 73.6% |
| 0.922 | 87.2% |
| 0.961 | 95.2% |
| **1.000** | **98.9%** |

### G_stable(t) — cumulative T_stable fraction

| t | G_stable |
|---|---------|
| 0.612 | 6.1% (slow early accumulation) |
| 0.806 | 17.1% (cliff begins) |
| 0.884 | 53.7% |
| 0.922 | 73.7% |
| 0.961 | 87.3% |
| 1.000 | **95.2%** |

---

## 核心发现

### 1. LangFlow never-stable = 4.79%（介于 ELF baseline 和 KD checkpoints 之间）

LangFlow 的 never-stable 率（4.79%）比 ELF baseline（25.1%）好得多，但比 ELF kd_cr（0.53%）差。
从"oracle commitment"质量角度，LangFlow 处于中间位置——大多数位置最终会承诺，但不如 ELF KD 模型
那么坚定。

### 2. G_oracle 非单调：低 t 的频率模式预测

G_oracle 在 t=0.03-0.42 约为 4%，然后在 t≈0.5 出现小幅下降（2.9%），最终在 t=0.71-0.81 急剧上升。

- **低 t 的 4%**：高噪声下模型退化为频率先验预测（最常见 token = "the"/"a"/"is"等），
  约 4% 的位置真实 token 就是这些常见 token，导致看似"正确"但实为模式坍塌。
- **t≈0.5 的 dip**：中等噪声时模型脱离频率模式但仍无足够信号，预测更分散 → top-1 准确率低于高噪声。
- **t=0.81 的 cliff**：真实信号开始主导，oracle 准确率快速提升。

这种非单调 G_oracle 是 LangFlow 特有的，与 ELF 的单调递增 G_oracle 完全不同。

### 3. 承诺时序：LangFlow 比 ELF kd_cr 晚约 5× T_stable

ELF kd_cr mean T_stable ≈ 0.18，LangFlow = 0.84，约晚 0.66t。与 EXP-02/03 发现的
"ELF x̂_t 比 LangFlow native posterior 早约 0.6t" 一致。

---

## EXP-25 对比（T_first，H<1.0 entropy threshold）

EXP-25 用 entropy threshold H<1.0 AND top1==gt 定义的 T_first 给出 never-committed=1.37%，
而 EXP-53 的 T_stable（K=3 consecutive, no entropy threshold）给出 4.79%。

差异来自：
1. K=3 consecutive 比单次 H<1.0 更严格（需要持续性）
2. 部分位置在 t≈0.03-0.40 有短暂频率模式正确预测（会被 entropy threshold 过滤，
   但不会被 K=3 consecutive 计为 stable）

两种指标给出同方向结论：LangFlow 的 never-commit 率（1.4%-4.8%）远低于 ELF baseline（25.1%），
略高于 ELF kd 模型（0.5-1.0%）。

---

## 结果文件

- `results/exp53_langflow_tstable/tstable_results.json`
- `results/exp53_langflow_tstable/t_first.npy`
- `results/exp53_langflow_tstable/t_stable.npy`
