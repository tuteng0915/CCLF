# EXP-16 Spec — Per-Position Commitment Timing

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：给出承诺时序的完整分布，而非单一 G(t) 曲线。**

G(t) 曲线描述的是在噪声水平 t 下，**所有位置**中有多少比例可以被正确解码。但这是一个聚合统计，掩盖了个体差异：
- 某些位置在 t=0.10 就已经可以正确解码（"早承诺"）
- 某些位置到 t=0.70 才能正确解码（"晚承诺"）
- 某些位置在我们测试的所有 t 值下都无法正确解码（"永不承诺"）

**要验证的核心假说**：
- KD 训练（kd-cr）大幅减少"永不承诺"的位置（baseline 约 19%，kd-cr 约 0.14%）
- KD 训练将承诺时刻的分布整体前移（earlier peak of the commitment histogram）
- baseline 中约 36.5% 的位置在 t=0.30 以内仍未承诺（= 1 - 0.635 at t=0.30）

**重要性**：
1. 直接支持论文的"早期承诺"主张：不只是"平均 G(t) 更高"，而是"每个位置更早承诺"
2. 量化 KD 的影响：将"永不承诺率"从 19% 降至 0.14%，改善幅度超过 100 倍
3. 为 EXP-20（token 频率分析）提供背景：哪些 token 是"永不承诺"的？

**与其他实验的关系**：
- 使用 EXP-07b 的 layer_states（decode path on L11 隐状态），复用已有数据
- 结果与 EXP-12（rank 分布）互补：EXP-16 问"何时正确"，EXP-12 问"错时差多少"
- 为 EXP-20 提供 baseline 的"永不承诺"位置列表

---

## Implementation

**Script:** `/tmp/exp16_commit_timing.py`

**方法**：对每个位置，找到使 decode_path(layer_feats[-1]) == y_token 成立的**最小 t 值**。

```python
# For each position:
# Find the smallest t in t_grid such that decode_path(h_t) == y_token
# If none found: "never committed" (in our t range)

# Metric: cumulative commitment histogram
# by t=0.1: fraction committed by t=0.1
# by t=0.2: fraction committed by t=0.2
# ...
# by t=never: fraction never committed in [0.1, 0.7]
```

---

## 实验结果（Results）

**状态**: COMPLETED（2026-07-18）

**原始数据文件**：`results/exp16/commit_timing.json`

**日志文件**：`/tmp/exp16.log`

**完整承诺时序直方图**（所有 3 个 checkpoint）：

| 时间节点 | baseline | kd-cr | kd2 |
|---------|---------|-------|-----|
| by t=0.10 | 2.0% | 12.6% | 12.3% |
| by t=0.20 | 36.4% | 59.4% | 58.2% |
| by t=0.30 | 63.5% | 90.2% | 88.8% |
| by t=0.50 | 77.6% | 99.6% | 99.1% |
| by t=0.70 | 81.0% | 99.9% | 99.7% |
| **永不承诺** | **19.0%** | **0.14%** | **0.30%** |

*数据来源：/tmp/exp16.log*

**关键发现**：

1. **KD 将"永不承诺率"从 19% 降至 0.14%**：kd-cr 几乎消除了永不承诺现象（135 倍改善）。

2. **承诺速度提升**：50% 的位置（median commitment time）：
   - baseline：刚过 t=0.20（约 t=0.21）
   - kd-cr：在 t=0.20 之前（约 t=0.17，因为 t=0.20 时已有 59.4%）
   
3. **90% 承诺时刻**：
   - baseline：t=0.70 仍未达到 90%（只有 81.0%）
   - kd-cr：t=0.30 已达到 90.2%

4. **G(t) 与"曾经承诺"的差异**：t=0.70 时，baseline G(t) = 78.7%，但"曾经承诺"率 = 81.0%（ever_correct_so_far）。2.3pp 的差异来自"承诺后又离开"（non-monotonicity）。

**论文使用建议**：
- 在方法/实验部分，用此直方图表作为承诺时序的核心量化证据
- 比 G(t) 曲线更直观：能展示"KD 几乎消除了永不承诺位置"
- 与 LangFlow 对比：EXP-02 中 LangFlow 在 t=0.916 才首次承诺（vs kd-cr 的 t≈0.17）

---

## ⚠️ EXP-16v2 结果（2026-07-22，权威版本）

**脚本**: `experiments/probe_elf/compute_readout_timing.py`
**修复**: 使用 exp07b_v2（fixed noise seed=42）+ correct decode path（GELU(h_L11@proj)@unemb）
**指标**: T_first, T_stable(K=3), T_margin(K=3, margin>5.0)

### 关键数据（T_stable never-commit）

| Checkpoint | never T_stable | never T_first | never T_margin |
|-----------|---------------:|---------------:|---------------:|
| baseline  | **25.1%**       | 8.9%           | **28.9%**       |
| kd_cr     | **0.53%**       | 0.11%          | **1.50%**       |
| kd2       | **0.98%**       | 0.16%          | **2.80%**       |

### G(t)（oracle readout accuracy，fixed noise，T_first）

**注**: 以下数字为 T_first（第一次读出正确），不是 T_stable。T_stable 在 t≤0.70 plateau（见 never-commit 表）

| t | baseline | kd_cr | kd2 |
|---|---------|-------|-----|
| 0.10 | 1.9% | 12.3% | 12.3% |
| 0.20 | **36.1%** | **58.8%** | **57.6%** |
| 0.30 | 63.0% | 89.5% | 88.2% |
| 0.50 | 77.3% | **99.5%** | **99.1%** |
| 0.70 | 80.7% | 99.8% | 99.7% |
| 1.00 | 91.1% | 99.9% | 99.8% |

T_stable (K=3 consecutive correct, plateau):
- baseline: by_t_0.50 = 74.9% (then flat — no more commits after t=0.50!)
- kd_cr: by_t_0.50 = 99.5% (then flat)
- kd2: by_t_0.50 = 99.0% (then flat)

Note: T_stable plateaus because the 6 t-values only go to t=1.0. The plateau is an artifact of the discrete t-value grid.

### 旧 EXP-16 vs EXP-16v2 对比

| 指标 | EXP-16（旧，错误） | EXP-16v2（正确） |
|------|------------------|-----------------|
| baseline never-commit | ~19% | 25.1% |
| kd_cr never-commit | ~0.14% | 0.53% |
| baseline G(t=0.20) | ~9% (non-fixed ε) | **36.1%** (fixed ε, T_first) |
| kd_cr G(t=0.20) | ~59% | **58.8%** (T_first) |

**旧数据"19%"应更新为"25.1%"**（因固定 ε 下 T_stable 标准更严格：K=3 consecutive，而旧版只是 first-hit）

### 与 EXP-09v3 交叉验证

EXP-09v3（稳定 k=3）: baseline never_commit=24.8%，kd_cr=0.67% → 与 EXP-16v2 一致（25.1% vs 0.53%）。两个独立实现给出一致结果。

### 输出文件

- `results/exp16v2/readout_timing.json`
- `results/exp16v2/timing_arrays_{baseline,kd_cr,kd2}.pt`

