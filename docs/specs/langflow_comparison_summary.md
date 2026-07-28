# LangFlow vs ELF Comparison Summary

## 概述

本文档汇总所有 LangFlow 对比实验（EXP-21~28）的结论，与 ELF 对应实验对比，
用于支撑论文的"连续扩散语言模型对比研究"部分。

---

## 综合对比表

| 实验 | ELF kd_cr | LangFlow | 结论 |
|------|-----------|----------|------|
| **承诺时间（commit cliff）** | t≈0.15-0.30（oracle） | t≈0.83-0.93（oracle） | LangFlow 比 ELF 晚 ~0.60t |
| **早期 G(t)** | ~61% at t=0.20（kd_cr）| ~3.9%（几何偏置 G_null=3.85%）| ELF 早期承诺是真实信号；LangFlow 几乎全是几何偏置 |
| **几何偏置 G_null** | 0.17%（EXP-04/EXP-23）| 3.85% at t=0.10（EXP-23）| LangFlow skip-connection 产生 22× 更强的几何偏置 |
| **ODE 轨迹稳定性** | last-flip step=27/32 (84%)（EXP-14）| last-flip step=8.3/32 (27%)（EXP-24）| LangFlow ODE 早期稳定，ELF 晚期稳定 |
| **Probe Gap（探针 vs native head）** | baseline +46pp，kd_cr −7pp（EXP-07）| PENDING（EXP-21 running）| — |
| **粗到细顺序** | kd_cr Δ=−0.073（EXP-08）| Δ=−0.050（EXP-25）| 两者都有；ELF KD 放大效应 |
| **空间自举效应** | kd_cr peak +65pp（EXP-09）| peak +21.1pp（EXP-26）| 两者都有；ELF ~3× 更强 |

---

## 关键发现与论文含义

### 发现 1：Oracle vs ODE 悖论（EXP-22 vs EXP-24）

**ELF**：oracle probe 显示早承诺（t≈0.15-0.30），实际 ODE 轨迹直到第 27/32 步才稳定。
- 解释：ELF backbone 早已知道答案，但 ODE 探索过程在晚期才稳定（EXP-14）

**LangFlow**：oracle probe 显示晚承诺（t≈0.83-0.93），实际 ODE 轨迹在第 8.3/32 步就稳定了。
- 解释：LangFlow ODE 快速锁定一个 token（8 步后 argmax 稳定），之后主要是提升置信度

**论文含义**：
- ELF："早知道，晚决定"（Early Knowledge, Late Decision）
- LangFlow："早决定，晚置信"（Early Decision, Late Confidence）
- 这对 "commit-release-recommit" 故事的解读：ELF 的晚稳定性来自 ODE 探索；LangFlow 的早稳定性来自 Euler-EDM 的线性性质

### 发现 2：粗到细顺序是 CDLM 通性（EXP-25）

两个模型都先承诺功能词（the, and, of...），后承诺内容词：
- ELF kd_cr: Δ(func - content) = −0.073（在 t=0.182 vs 0.255 之间）
- LangFlow: Δ = −0.050（在 t=0.825 vs 0.875 之间）

**论文含义**：粗到细层级是连续扩散 LM 的**内在属性**，不是 ELF 特有的。
ELF KD 训练使功能词/内容词差距扩大（|Δ| +47% vs LangFlow），说明 KD 强化了这一自然层级。

### 发现 3：空间自举是 CDLM 通性但 ELF 更强（EXP-26）

两个模型都显示已承诺邻居加速周围位置的承诺：
- ELF kd_cr: peak Δ=+65pp（d=5, t=0.5→0.7）
- LangFlow: peak Δ=+21pp（d=5, t=0.93→0.95）

**论文含义**：空间传播是连续扩散 LM 的普遍机制，但 ELF decode branch 放大了约 3×。
这支持了 decode branch 的机制价值：不仅使 ELF 更早承诺，还加强了已承诺位置对邻域的传播能力。

### 发现 4：几何偏置在 LangFlow 更严重（EXP-23）

ELF G_null = 0.17%（可忽略），LangFlow G_null = 3.85%（早期 G(t) 几乎全是偏置）。

**论文含义**：LangFlow 的 bias skip-connection (`logits += c_skip(γ) × z @ E.T`) 在低 t（高噪声）时产生大量几何噪声。LangFlow 在 t≤0.50 时的 native_top1 ≈ G_null ≈ 4%，说明没有真实的 token 信息被利用。ELF 无此问题。

### 发现 5：Probe Gap 对比（EXP-21，PENDING）

**预期情景 A（probe >> native head）**：
- 论文说法："LangFlow 也有表示-读出 gap，说明 gap 是 diffusion LM 通性；ELF KD 针对性地修复了这一通性问题"

**预期情景 B（probe ≈ native head）**：
- 论文说法："LangFlow 的独立 LM head 天然避免了 ELF 的 tied-weight 几何不匹配问题；ELF 的 KD 是对其特定设计缺陷的修复"

---

## 待完成实验

### EXP-21（LangFlow Probe Gap）— RUNNING
- 预计完成：~4 小时后（PID 3143093，GPU 2，~1h48m 已运行）
- 重要性：高（区分 probe gap 是 CDLM 通性 vs ELF 特有问题）

---

## 各实验文件索引

| EXP | 数据文件 | 状态 |
|-----|---------|------|
| EXP-21 | `results/exp21_langflow/probe_gap_results.json` | RUNNING |
| EXP-22 | `results/exp22_langflow/commitment_by_t.json` | DONE |
| EXP-23 | `results/exp22_langflow/null_model.json` | DONE |
| EXP-24 | `results/exp24_langflow/traj_stability.json` | DONE |
| EXP-25 | `results/exp25_langflow/coarse_fine_results.json` | DONE |
| EXP-26 | `results/exp26_langflow/bootstrap_d5.json` | DONE |

### 发现 6：Token 频率-承诺梯度是 CDLM 通性（EXP-27）

LangFlow 和 ELF 都表现出高频 token 比低频 token 更早承诺：
- ELF：rare(21.6%) vs common(13.5%) never_commit_rate，差距 8pp（EXP-20）
- LangFlow：correlation(log10(tok_id), t*) = +0.47；rare never_rate 2.9% vs common 0.3%（EXP-27）

**论文含义**：频率-承诺梯度是 CDLM 的统计自然现象，由训练数据分布决定，不依赖模型架构。
LangFlow 的效应较弱（数据量有限 + T5 多语言 ID 排序不纯粹）。

### 发现 7：LangFlow 空间自举无方向性偏向（EXP-28）

EXP-26 证明了 LangFlow 的空间自举（+21pp），但 EXP-28 发现方向是**对称的**：
- func→content boost 峰值 = +2.7pp（t=0.813→0.830）
- content→func boost 峰值 = +5.4pp（t=0.796→0.813，**反而更强**）

**论文含义**：
- LangFlow 的自举是**位置邻近效应**，不是**语义层级传播**
- ELF KD 的 decode branch 可能创造了真正的方向性传播（功能词 → 内容词），但目前在 ELF 上的 EXP-09 并未分解方向性
- 后续可在 ELF 上做类似的方向分解实验（EXP-09v2），以区分 ELF 的自举是否也有 content→func 的意外方向

---

## 更新的对比表

| 实验 | ELF kd_cr | LangFlow | 结论 |
|------|-----------|----------|------|
| **承诺时间（commit cliff）** | t≈0.15-0.30（oracle） | t≈0.83-0.93（oracle） | LangFlow 比 ELF 晚 ~0.60t |
| **早期 G(t)** | ~61% at t=0.20 | ~3.9%（几何偏置）| ELF 早期承诺是真实信号；LangFlow 是噪声 |
| **几何偏置 G_null** | 0.17%（EXP-23）| 3.85%（EXP-23）| LangFlow skip-connection 22× 偏置 |
| **ODE 轨迹稳定性** | last-flip step=27/32（EXP-14）| step=8.3/32（EXP-24）| 早决定/晚决定正好相反 |
| **Probe Gap** | baseline +46pp, kd_cr −7pp（EXP-07）| PENDING（EXP-21 running）| — |
| **粗到细顺序** | Δ=−0.073（EXP-08）| Δ=−0.050（EXP-25）| 两者都有；ELF KD 放大 |
| **空间自举效应** | peak +65pp, d=5（EXP-09）| peak +21pp（EXP-26）| 两者都有；ELF ~3× 更强 |
| **自举方向性** | 未分解 | content→func 略强（EXP-28）| LangFlow 自举是对称邻近效应 |
| **频率-承诺梯度** | rare vs common 8pp（EXP-20）| r=+0.47，2.6pp（EXP-27）| 两者都有；ELF 更显著 |

---

## 各实验文件索引（更新）

| EXP | 数据文件 | 状态 |
|-----|---------|------|
| EXP-21 | `results/exp21_langflow/probe_gap_results.json` | RUNNING |
| EXP-22 | `results/exp22_langflow/commitment_by_t.json` | DONE |
| EXP-23 | `results/exp22_langflow/null_model.json` | DONE |
| EXP-24 | `results/exp24_langflow/traj_stability.json` | DONE |
| EXP-25 | `results/exp25_langflow/coarse_fine_results.json` | DONE |
| EXP-26 | `results/exp26_langflow/bootstrap_d5.json` | DONE |
| EXP-27 | `results/exp27_langflow/token_freq_analysis.json` | DONE |
| EXP-28 | `results/exp28_langflow/asymmetric_bootstrap_d5.json` | DONE |

---

## 发现 8：ELF KD 具有强烈的 func→content 方向性自举（EXP-09v2）

EXP-09v2 分解了 ELF 三个检查点的自举方向性：

**kd_cr 的时间因果顺序**：
- t=0.1→0.2（早期）：cf_Δ=+1.8pp > fc_Δ=+0.1pp → 早期**双向弱效应，cf 略强**（部分内容词与功能词同步承诺）
- t=0.2→0.5：功能词快速收敛（cf_n: 1,313→91），双向效应趋零
- **t=0.5→0.7**：fc_Δ=**+38.8pp**（fc_n=61），cf_Δ=+0pp（cf_n=3，功能词几乎全部已承诺）
- **t=0.7→1.0**：fc_Δ=**+65.7pp**（fc_n=12），cf_Δ=+0pp（cf_n=2）

**三检查点对比（峰值 fc_Δ）**：

| 检查点 | fc 峰值 Δ | cf 峰值 Δ | 方向偏向 |
|--------|-----------|-----------|---------|
| ELF baseline | +7.8pp（t=0.7→1.0）| ≤+0pp | 几乎无 |
| ELF kd2 | **+53.8pp**（t=0.7→1.0）| ≤+2.7pp | 强 func→content |
| ELF kd_cr | **+65.7pp**（t=0.7→1.0）| ≤+1.8pp | 强 func→content |
| LangFlow（EXP-28）| +2.7pp | +5.4pp | **对称**（偏 cf）|

**论文含义**：
- ELF KD 创造了**时间因果方向性**：功能词在 t=0.1-0.3 率先承诺，之后作为"承诺锚"加速相邻内容词（+65pp）
- LangFlow 的自举对称弱，无语义层级定向性
- 三角对比（粗到细 + 自举强度 + 方向性）为 ELF KD decode branch 的机制提供了统一解释

**注意事项**：cf_Δ=0 在 t>0.5 不意味着 content→func 效应为零——而是功能词已全部承诺，无未承诺功能词可测。cf 方向在早期（t=0.1→0.2，cf_n=1,313）测得 +1.8pp，方向性在早期接近对称。

---

## 更新的综合对比表（含 EXP-09v2）

| 实验 | ELF kd_cr | LangFlow | 结论 |
|------|-----------|----------|------|
| **承诺时间（commit cliff）** | t≈0.15-0.30（oracle）| t≈0.83-0.93（oracle）| LangFlow 比 ELF 晚 ~0.60t |
| **早期 G(t)** | ~61% at t=0.20 | ~3.9%（几何偏置）| ELF 早期承诺是真实信号；LangFlow 是噪声 |
| **几何偏置 G_null** | 0.17%（EXP-23）| 3.85%（EXP-23）| LangFlow skip-connection 22× 偏置 |
| **ODE 轨迹稳定性** | last-flip step=27/32（EXP-14）| step=8.3/32（EXP-24）| 早决定/晚决定正好相反 |
| **Probe Gap** | baseline +46pp, kd_cr −7pp（EXP-07）| PENDING（EXP-21 running）| — |
| **粗到细顺序** | Δ=−0.073（EXP-08）| Δ=−0.050（EXP-25）| 两者都有；ELF KD 放大 |
| **空间自举效应** | peak +65pp, d=5（EXP-09）| peak +21pp（EXP-26）| 两者都有；ELF ~3× 更强 |
| **自举方向性** | **fc=+65.7pp >> cf≤+1.8pp（EXP-09v2）**| fc=+2.7pp ≈ cf=+5.4pp（EXP-28）| **ELF KD 具有 func→content 方向性；LangFlow 对称** |
| **频率-承诺梯度** | rare vs common 8pp（EXP-20）| r=+0.47，2.6pp（EXP-27）| 两者都有；ELF 更显著 |

---

## 各实验文件索引（最终版）

| EXP | 数据文件 | 状态 |
|-----|---------|------|
| EXP-09v2 (ELF) | `models/ELF-torch/results/exp09v2_{baseline,kd_cr,kd2}/asymmetric_bootstrap_d5.json` | DONE |
| EXP-21 | `results/exp21_langflow/probe_gap_results.json` | RUNNING |
| EXP-22 | `results/exp22_langflow/commitment_by_t.json` | DONE |
| EXP-23 | `results/exp22_langflow/null_model.json` | DONE |
| EXP-24 | `results/exp24_langflow/traj_stability.json` | DONE |
| EXP-25 | `results/exp25_langflow/coarse_fine_results.json` | DONE |
| EXP-26 | `results/exp26_langflow/bootstrap_d5.json` | DONE |
| EXP-27 | `results/exp27_langflow/token_freq_analysis.json` | DONE |
| EXP-28 | `results/exp28_langflow/asymmetric_bootstrap_d5.json` | DONE |
