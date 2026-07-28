# EXP-09v2 Spec — ELF 非对称自举方向分析（Func→Content vs Content→Func）

## 实验背景与动机

**在整体框架中的地位：区分 ELF 的空间自举是否有语义方向性——是 func→content 的定向传播，还是对称的位置邻近效应。**

EXP-09 证实了 ELF kd_cr 有强空间自举效应（d=5, t=0.2→0.3 时 +45pp，t=0.7→1.0 时 near=0.589 vs far=0.000）；EXP-08 证实了功能词比内容词更早承诺（Δ=-0.073）。EXP-28 则发现 LangFlow 的自举是对称的（fc peak +2.7pp ≈ cf peak +5.4pp）。

**核心问题**：ELF 的自举方向性如何？具体地：
- **fc（func→content）**：已承诺功能词邻居对未承诺内容词的加速效果有多大？
- **cf（content→func）**：已承诺内容词邻居对未承诺功能词的加速效果有多大？

若 fc >> cf，则说明 ELF KD 训练在功能词和内容词之间建立了真正的**时间因果传播链**（功能词先承诺，再作为锚点帮助内容词）。

---

## 实验设计

**脚本**：`models/ELF-torch/experiments/probe_elf/analyze_asymmetric_bootstrap_elf.py`

**数据来源**：EXP-09 生成的 per-position 承诺时间矩阵
- `models/ELF-torch/results/exp09_{baseline,kd_cr,kd2}/commit_times_matrix.npy` — [256, 256]
- `models/ELF-torch/results/exp09_{baseline,kd_cr,kd2}/y_tokens_ref.npy` — [256, 256]

**功能词分类**：使用 GPT-2 tokenizer（ELF 使用 GPT-2 tokenizer），通过 'Ġ' 前缀检测词首 token，
然后与标准英语功能词集合匹配。

**方向定义**（对每个 t_i → t_{i+1} 步骤）：
- `fc_mask`：未承诺的**内容词**位置，且 d=5 邻域内有**已承诺功能词**
- `cf_mask`：未承诺的**功能词**位置，且 d=5 邻域内有**已承诺内容词**
- `base_cont`：所有未承诺内容词位置的基础承诺率（无邻域条件）
- `base_func`：所有未承诺功能词位置的基础承诺率（无邻域条件）
- `fc_delta = fc_rate - base_cont`（func→content boost）
- `cf_delta = cf_rate - base_func`（content→func boost）

**t 步骤**：[0.1→0.2, 0.2→0.3, 0.3→0.5, 0.5→0.7, 0.7→1.0]（EXP-09 t-grid 的 5 个间隔）

---

## 实验结果（Results）

**状态**: COMPLETED（2026-07-20）

**数据输出**：
- `models/ELF-torch/results/exp09v2_kd_cr/asymmetric_bootstrap_d5.json`
- `models/ELF-torch/results/exp09v2_kd2/asymmetric_bootstrap_d5.json`
- `models/ELF-torch/results/exp09v2_baseline/asymmetric_bootstrap_d5.json`

### kd_cr 非对称自举（d=5）

| t_cur→t_next | fc_rate | base_cont | fc_Δ | cf_rate | base_func | cf_Δ | fc_n | cf_n |
|---|---|---|---|---|---|---|---|---|
| 0.1→0.2 | 54.0% | 53.9% | +0.1pp | 74.0% | 72.3% | **+1.8pp** | 4,650 | 1,313 |
| 0.2→0.3 | 75.4% | 75.8% | −0.3pp | 81.6% | 81.7% | −0.0pp | 5,690 | 495 |
| 0.3→0.5 | 96.4% | 94.0% | +2.4pp | 96.7% | 96.7% | +0.0pp | 1,690 | 91 |
| **0.5→0.7** | **80.3%** | 41.6% | **+38.8pp** | 33.3% | 33.3% | +0.0pp | 61 | 3 |
| **0.7→1.0** | **75.0%** | 9.3% | **+65.7pp** | 50.0% | 50.0% | +0.0pp | 12 | 2 |

### kd2 非对称自举（d=5，关键步骤）

| t_cur→t_next | fc_Δ | cf_Δ | fc_n | cf_n |
|---|---|---|---|---|
| 0.1→0.2 | +1.6pp | **+2.7pp** | ~4k | ~1.1k |
| 0.2→0.3 | +1.1pp | +0.0pp | ~5k | ~430 |
| 0.3→0.5 | +3.2pp | +0.0pp | ~1.5k | ~70 |
| 0.5→0.7 | **+20.5pp** | +0.0pp | ~50 | ~2 |
| 0.7→1.0 | **+53.8pp** | +0.0pp | ~10 | ~1 |

### baseline 非对称自举（d=5，关键步骤）

| t_cur→t_next | fc_Δ | cf_Δ |
|---|---|---|
| 0.1→0.2 | **−1.0pp** | −0.3pp |
| 0.2→0.3 | **−5.4pp** | −0.2pp |
| 0.3→0.5 | **−5.5pp** | −0.1pp |
| 0.5→0.7 | −1.9pp | +0.0pp |
| 0.7→1.0 | **+7.8pp** | +0.0pp |

### 四模型对比

| 模型 | fc 峰值 Δ (func→content) | cf 峰值 Δ (content→func) | 方向性偏向 |
|------|--------------------------|--------------------------|----------|
| ELF baseline | +7.8pp (t=0.7→1.0, fc_n~10) | ≤+0pp | 弱，无方向性 |
| ELF kd_cr | **+65.7pp** (t=0.7→1.0, fc_n=12) | ≤+1.8pp (t=0.1→0.2) | **强烈单向 func→content** |
| ELF kd2 | **+53.8pp** (t=0.7→1.0) | ≤+2.7pp (t=0.1→0.2) | 强烈单向 func→content |
| LangFlow (EXP-28) | +2.7pp (t=0.813→0.830) | +5.4pp (t=0.796→0.813) | 对称，弱（甚至偏 cf 方向）|

---

## 关键发现

### 1. ELF KD 具有时间因果方向性

**时间顺序**（kd_cr）：
- t=0.1→0.2：功能词和内容词都有大量未承诺位置（cf_n=1,313, fc_n=4,650）；cf略强（+1.8pp vs +0.1pp）
- t=0.2→0.3：功能词快速收敛（cf_n: 1,313→495→91）；两个方向效应均趋零（大部分功能词已承诺）
- t=0.5→0.7：几乎所有功能词已承诺（cf_n=3）；已承诺功能词大规模帮助内容词（+38.8pp）
- t=0.7→1.0：极少数剩余内容词获得最大加速（+65.7pp，fc_n=12）

**机制**：ELF KD 的功能词在 t=0.1-0.3 建立"承诺锚"（commit anchor），此后 t=0.3-1.0，大量已承诺功能词作为邻域信号加速周围内容词承诺。

### 2. cf_Δ≈0 的解读——不能解读为"content→func 效应不存在"

t=0.5→0.7 时 cf_n=3，t=0.7→1.0 时 cf_n=2，**样本量极小**。

- cf_delta=0 的正确解读：**功能词已经全部承诺，没有未承诺功能词可以测试**
- 这本身就是功能词早期承诺的证据，不是"content→func 传播不存在"的证据

**真正可比的是早期步骤**（t=0.1→0.2，两类词都有足够样本）：
- kd_cr: fc_Δ=+0.1pp, cf_Δ=+1.8pp → **早期 cf 略强**（部分内容词比功能词更早承诺）
- kd2: fc_Δ=+1.6pp, cf_Δ=+2.7pp → **早期 cf 略强**

→ 在承诺悬崖早期（t=0.1），双向效应都较弱，cf 略有优势（因为有些常见内容词与功能词同时承诺）

### 3. baseline 的负 fc_Δ

ELF baseline 在 t=0.1→0.5 的 fc_delta 为负（−1.0 到 −5.5pp）：有承诺功能词邻居反而**轻微抑制**内容词承诺。与 EXP-09 baseline 无正向自举效应一致：baseline 的承诺由词频先验主导，与空间上下文无关。

### 4. 与 LangFlow 的根本区别

| 特征 | ELF kd_cr | LangFlow |
|------|-----------|----------|
| 功能词先承诺 | YES（EXP-08, Δ=-0.073）| YES（EXP-25, Δ=-0.050）|
| 整体空间自举峰值 | +65pp（EXP-09, d=5）| +21pp（EXP-26, d=5）|
| func→content 方向 | **+65.7pp** | +2.7pp |
| content→func 方向 | ≤+1.8pp（早期）, 0pp（晚期）| +5.4pp |
| 方向性机制 | decode branch 明确 token 信号传播 | 隐式 attention 位置邻近效应 |

LangFlow 和 ELF 都有粗到细顺序和空间自举，但 **ELF KD 的 decode branch 创造了真正的时间因果传播**：功能词早期承诺的 token 信号通过 self-conditioning 沿序列明确传播，产生类似 cloze-task 的语境填充效果。LangFlow 的自举仅是对称的位置邻近效应（proximity effect），不具有语义层级的定向性。

---

## 论文意义

EXP-09v2 与 EXP-08/09/28 构成一个统一的三角对比，为 ELF KD 机制提供完整解释：

| 维度 | ELF kd_cr 发现 | ELF baseline | LangFlow 对比 |
|------|---------------|--------------|--------------|
| 粗到细顺序（EXP-08/25）| 功能词提前 Δ=-0.073 | 弱，Δ≈-0.02 | Δ=-0.050 |
| 空间自举强度（EXP-09/26）| +45pp 早期，+65pp 晚期 | 负或弱 | +21pp |
| 方向性（EXP-09v2/28）| **func→content +65pp >> cf ≤+1.8pp** | 无方向 | 对称 fc≈cf≈2-5pp |

**KD 的机制故事**：ELF KD decode branch 通过自条件化（self-conditioning）建立了"已承诺位置 → 邻域"的明确信号通路。功能词因其低熵先验在 t=0.1-0.3 率先承诺，其 token 信号（通过 z_t 中的 SC 分量）沿序列传播，在 t=0.5-1.0 形成 +65pp 的强方向性加速。

---

## 限制与注意事项

1. **t-grid 稀疏**：仅 6 个 t 值（EXP-09 使用 [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]）；Peak Δ 的准确位置无法精确定位（dense re-run 待定）

2. **晚期步骤 fc_n 小**：t=0.7→1.0 时 fc_n=12（kd_cr），fc_delta=+65.7pp 是可信的（n=12 足以检测 +65pp 量级效应），但置信区间宽

3. **cf 方向在晚期不可测**：cf_n=2-3 时 cf_delta=0 不等于"content→func 效应为零"——功能词已经全部承诺，这是功能词早期承诺的证明，不是方向性零效应的证明

4. **Protocol A 局限**：全部基于 oracle probe（独立噪声，非真实 ODE 轨迹）。真实生成轨迹中的方向性效应可能不同（EXP-01 Protocol B 已验证 ELF 轨迹动力学与 Protocol A 存在差异）
