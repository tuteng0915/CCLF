# EXP-26 Spec — LangFlow Contextual Bootstrapping (LangFlow analog of EXP-09)

## 实验背景与动机

**在整体框架中的地位：测试 LangFlow 是否也有"已承诺位置帮助邻近位置更早承诺"的空间自举效应。**

EXP-09 在 ELF-kd_cr 上发现了强烈的空间自举效应：
- 在 t=0.2→0.3 时，已承诺邻居（d=5）旁边的未承诺位置承诺率为 76.8%，而无已承诺邻居的位置为 29.7%（Δ=+47pp）
- 在 t=0.5→0.7 时，近距离 vs 远距离的承诺率差距达 +65pp（800/152，d=5）
- ELF baseline 几乎没有自举效应（甚至是负的）：说明 KD 训练是自举效应的关键驱动

**EXP-26 的核心问题**：LangFlow 的 Euler-EDM ODE 在承诺悬崖（t≈0.83-0.93）中是否有类似的空间传播效应？

两种可能：
1. **H_no_bootstrap**：LangFlow 的承诺悬崖是全序列同步触发的（无空间传播），Δ≈0。说明 LangFlow 的承诺机制与 ELF 不同。
2. **H_bootstrap**：LangFlow 也有空间自举效应（Δ > 5pp），说明这是连续扩散 LM 的通性。

---

## 数据来源

EXP-25 已经保存了每个位置的承诺时序数据：
- `results/exp25_langflow/commit_tidx.npy` — [n_samples, L] 每位置首次正确承诺的 t 索引
- `results/exp25_langflow/gt_tokens.npy` — [n_samples, L] 真实 token ID
- `results/exp25_langflow/t_grid.npy` — 51 个 t 值

**EXP-26 是纯数据分析（CPU only），无需 GPU 重新运行模型。**

---

## 实现

**脚本**：`experiments/probe_langflow/analyze_bootstrap_langflow.py`（新建）

**核心逻辑**：对每个 t_i → t_{i+1} 的转换：
1. 找出在 t_i 已承诺（commit_tidx ≤ i）的位置
2. 对于未承诺位置（commit_tidx > i），判断其是否在 d=5 范围内有已承诺邻居
3. 计算 near_rate（有已承诺邻居的未承诺位置在下一步的承诺率）vs far_rate

**关键优化**：用 `scipy.ndimage.maximum_filter1d` 向量化最近邻距离计算（避免 O(L²) 遍历）。

---

## 期望结果与决策规则

| LangFlow 结果 | 论文意义 |
|--------------|--------|
| 峰值 |Δ| < 5pp（全范围） | "LangFlow 的承诺是全序列同步的，无空间传播；与 ELF kd_cr 的局部传播机制不同" |
| 峰值 Δ > 15pp | "LangFlow 保留了 ELF 的空间自举特征；连续扩散 LM 普遍存在此效应" |

**对比基准（EXP-09 ELF kd_cr）**：
- d=5, t=0.5→0.7: Δ=+65pp（near=80.0%, far=15.2%）

---

## 实验结果（Results）

**状态**: COMPLETE（2026-07-20）

**脚本**: `experiments/probe_langflow/analyze_bootstrap_langflow.py`（CPU only，重用 EXP-25 数据）
**数据**: `results/exp26_langflow/bootstrap_d5.json`
**配置**: d_near=5, n_samples=64, seq_len=128, 51 t 值

### 自举效应（d=5，cliff 区域 t=0.70-1.00）

| t_cur→t_next | near_rate | far_rate | Δ | near_n | far_n |
|---|---|---|---|---|---|
| 0.711→0.728 | 1.2% | 0.7% | +0.6pp | 1,139 | 6,884 |
| 0.779→0.796 | 6.7% | 3.5% | +3.2pp | 2,892 | 4,685 |
| 0.813→0.830 | 12.7% | 9.1% | +3.5pp | 4,402 | 2,159 |
| 0.864→0.881 | 22.4% | 13.8% | +8.6pp | 3,955 | 145 |
| **0.881→0.898** | **26.1%** | **5.5%** | **+20.7pp** | 3,141 | 55 |
| **0.932→0.949** | **37.8%** | **16.7%** | **+21.1pp** | 1,138 | 6 |
| 0.949→1.000 | ~41-50% | n/a | n/a | — | 0 |

（t>0.95 时，near_n/far_n 中 far_n=0，所有未承诺位置均在已承诺邻居附近）

### 对比 ELF EXP-09

| 模型 | 峰值 Δ (d=5) | 峰值发生时间 | 典型 near_rate | 典型 far_rate |
|------|------------|-----------|--------------|--------------|
| **LangFlow** | **+21.1pp** | t=0.932→0.949 | 37.8% | 16.7% |
| ELF kd_cr | +65pp | t=0.5→0.7 | 80.0% | 15.2% |
| ELF baseline | 接近 0pp | — | — | — |

### 关键发现

1. **H_bootstrap 部分成立（Δ=+21.1pp）**：LangFlow 在承诺悬崖后期（t≈0.88-0.95）确实有正向空间自举效应，说明连续扩散 LM 的位置间承诺不是完全同步的。

2. **LangFlow 自举效应比 ELF kd_cr 弱 3×**：峰值 +21.1pp vs ELF 的 +65pp。这与 EXP-25 的 Δ(func-content)=−0.050 vs ELF 的 −0.073 一致：LangFlow 的粗到细层级存在，但较 ELF KD 训练后弱。

3. **效应仅在悬崖后期出现**：t<0.86 时 Δ<9pp，说明在悬崖早期（t=0.70-0.85）LangFlow 的位置间承诺几乎同步。空间传播在悬崖已经启动（near_rate≈13-22%）之后才显现。

4. **原因推测**：LangFlow 缺少 ELF 的 self-conditioning decode branch（决策信号的明确传播通道）。LangFlow 的空间自举完全来自 attention 上下文的隐式效应，比 ELF 的显式 decode branch 弱约 3×。

### 论文意义

- 空间自举效应（contextual bootstrapping）是连续扩散 LM 的**共同特征**，但效应强度取决于模型的解码机制
- ELF KD 训练的 decode branch 将该效应放大了 ~3×（21pp→65pp）
- 这强化了 ELF decode branch 的机制性价值：不仅帮助 ELF 自身更早承诺，也增强了已承诺位置对邻域的"传播"能力

---

## ⚠️ 方法论问题（2026-07-22 审查）

### 1. CRITICAL：晚期峰值（t=0.932）是"risk set collapse"，不是真实自举

数据显示：
- t=0.932→0.949 时：near_rate=37.8%，far_rate=16.7%，**far_n=6**（仅 6 个位置！）
- t=0.949→1.000 时：far_n=0（无远端未承诺位置可比较）

当 far_n=6 时，far_rate=16.7% 的置信区间极宽（95% CI ≈ [0%, 64%]）。峰值 +21.1pp 完全被 far_n=6 的极高方差掩盖，不能视为统计显著的自举效应。

这是经典的 **risk set collapse**（存活分析的基准组在晚期收缩到 0，导致 far_rate 完全不可靠）。

### 2. 共因混淆：同一局部短语的难度

Near 组（有已承诺邻居）和 Far 组（无已承诺邻居）**系统性地处于不同的语言环境**：
- Near 组：局部短语已有部分 token 锁定，说明该短语可预测性高
- Far 组：孤立位置，可能是该局部最难预测的 token

"先提交的邻居促进了后续 commitment"（因果自举）与"可预测局部短语的所有 token 都快速 commit"（共同原因）产生完全相同的条件概率模式，当前数据无法区分。

### 3. 多重比较问题

51 个 t 值 × 多个 d 值的搜索空间，未做任何多重比较校正，"峰值 +21.1pp"是后验选出的最大值，实际 α 水平远高于 0.05。

### 4. 正确分析：生存模型（hazard model）

区分"bootstrap"与"shared local predictability"需要：
1. 用 **hazard model** 以每位置 commit 为事件，以 `has_committed_neighbor`（时间变化协变量）为预测变量
2. **控制混淆变量**：token unigram frequency（真实 OWT 训练频率），context surprisal（-log P(w_i | context)），POS / syntactic role
3. **Moran's I 空间自相关检验**：在随机排列下检验观测到的空间聚类是否显著
4. 若确实需要近远比较，使用 **Fisher exact test** 等适合小样本的检验

### 5. 安全结论（LangFlow 内部）

- LangFlow 的承诺时序存在轻微的正向空间聚类（early-commitment positions tend to cluster）
- 这与"共享局部可预测性场"（nearby positions share syntactic/semantic context）一致
- **不能得出**：
  - ~~"空间自举效应在 LangFlow 中存在"~~（risk-set collapse + 无多重比较校正）
  - ~~"ELF decode branch 将自举效应放大 3×"~~（跨模型 Δ 无法比较，且 EXP-26 vs EXP-09 用不同测量协议）
  - ~~"连续扩散 LM 的共同特征"~~（结论基于无效比较）

---

## EXP-26v2 结果（2026-07-22，DONE）

**脚本**: `experiments/probe_langflow/analyze_hazard_model.py`  
**输出**: `results/exp26v2_langflow/hazard_morans_v2.json`  
**方法**: 离散时间 logistic 风险模型 + Moran's I 空间自相关

### Moran's I（n_perm=300，stride=5）

| t | I_obs | z_score | p_cluster | commit_rate |
|---|-------|---------|-----------|-------------|
| 0.564 | -0.008 | -0.25 | 1.00 | 0.05% |
| 0.660 | 0.115 | 6.61 | 0.000 | 0.71% |
| 0.745 | **0.260** | **22.54** | **0.000** | 3.80% |
| 0.830 | 0.247 | 21.90 | 0.000 | 29.1% |
| 0.915 | 0.106 | 10.15 | 0.000 | 79.3% |
| 1.000 | -0.008 | -0.01 | 0.420 | 98.6% |

- **峰值 I=0.260 at t=0.745**（z=22.54，p<0.001）：承诺高度空间聚集
- 4/6 t 值显著（均在承诺窗口 t=0.66-0.92 内）
- t=1.0 不显著（几乎全部承诺，无区分度）

### 离散时间风险模型

**118,608 person-period observations，2702 events（event rate=2.28%）**

| 预测变量 | β | OR | 95% CI (OR) |
|---------|---|-----|-------------|
| has_committed_neighbor | +0.893 | **2.442** | [2.191, 2.764] |
| log_freq_norm | +0.794 | **2.212** | [2.023, 2.396] |
| is_function | +0.219 | **1.244** | [1.036, 1.482] |
| t_normalized | +17.87 | 5.8×10⁷ | （时间主效应极强） |

### 解读

1. **空间聚集显著**（Moran's I=0.26，z=22.54）：承诺时序在序列内高度空间相关。这与"邻近位置共享句法/语义结构"（common local structure）一致，也可以由因果传播解释，**但 Moran's I 无法区分两者**。

2. **has_committed_neighbor 的风险比 OR=2.44**（CI=[2.19, 2.76]）：邻居已承诺使当前位置的即时承诺风险增加 2.44 倍。效应量大，CI 不包括 1。
   - **但存在共因混淆**：邻近位置共享句法/语义上下文（均在 NP 内、均为 VP 成分），这会使两者同时高承诺，而非因果传播。无结构干预无法区分。

3. **频率效应 OR=2.21**：与 EXP-27v2 r=-0.65 一致。

4. **is_function 小但显著 OR=1.24**：即使控制频率和邻居状态，函数词仍有 24% 更高的即时承诺风险。这是 EXP-25v2 在很高 t 时 β_func 变正的机制：函数词本身在风险模型中有小的残余效应。

### 最终诊断

EXP-26 原始结论（"承诺自举效应"）**仍然无法确立因果关系**，但新分析提供了更可信的描述性证据：
- 空间聚集在承诺窗口期（t=0.66-0.92）极显著（I≈0.25）
- has_committed_neighbor 是强预测因子（OR=2.44）
- 共因解释（"共享局部可预测性场"）仍然无法排除，需要因果干预实验

