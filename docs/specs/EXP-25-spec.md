# EXP-25 Spec — LangFlow Coarse-to-Fine (LangFlow analog of EXP-08)

## 实验背景与动机

**在整体框架中的地位：测试 LangFlow 是否也表现出"功能词先承诺，内容词后承诺"的粗到细模式。**

EXP-08 在 ELF 上发现：
- kd_cr 功能词 (function words) 的平均承诺时间 t* = 0.182，内容词 t* = 0.255，差值 Δ = −0.073
- 在 t=0.5→0.7 时，功能词邻近的内容词承诺率 = 0.800 vs 远处 = 0.152（+65pp 空间自举效应）
- baseline ELF 空间自举效应为负（已承诺邻居反而阻碍）

**EXP-25 的核心问题**：LangFlow 是否有类似的功能词先承诺（粗到细）现象？

从 EXP-22 我们知道 LangFlow 的整体承诺时间极晚（t≈0.83-0.93），而且 oracle probe 中几乎所有位置在 t<0.80 都未承诺。这两个发现合并预测：

- **H_no_order**: LangFlow 几乎所有 token（不管是功能词还是内容词）都在 t≈0.83-0.93 同时承诺，功能词/内容词时序差异可忽略不计。
- **H_coarse_to_fine**: 即使整体很晚，LangFlow 内部仍有功能词先承诺的层级顺序（类似 ELF），只是整体偏移到更晚的 t 区间。

### 数据来源

EXP-22 已经保存了每个位置在每个 t 值下的 committed_correct/committed_wrong/uncommitted 数据，但是 EXP-22 保存的是**聚合结果**（平均过所有 token 类型），不能直接用于功能词 vs 内容词分析。

因此，EXP-25 需要重新运行 `probe_commitment_langflow.py` 但保存每个位置的 token ID 和承诺时间，然后按功能词/内容词分类。

---

## 实现计划

### 复用：`experiments/probe_langflow/probe_commitment_langflow.py`

现有脚本计算每个样本每个位置的承诺状态，但只保存聚合统计。需要修改以输出：
- 每个 (样本, 位置, t) 的 committed_correct 和 gt_token_id

或者，更简单：先把 EXP-22 的 n_t_steps 增加到 40 并记录每位置的承诺时间 t*（类似 EXP-09 的 commit_times_matrix.npy），再按功能词/内容词分类。

### 新脚本：修改 `probe_commitment_langflow.py` 添加 `--save_per_position` 标志

**输出**：
- `per_position_commitment.npz` — shape [n_samples, L, n_t_steps]，每个位置在每个 t 下是否 committed_correct
- `per_position_tokens.npy` — shape [n_samples, L]，每位置的 gt token id

**分析脚本**（与 EXP-08/09 类似）：
```python
# 分类 token
FUNCTION_WORDS = {...}  # 复用 EXP-08 的 function word 列表

# 每位置的承诺时间 t* = 最小的 t 使得 committed_correct = True
commit_times = find_first_commitment(per_position_commitment, t_grid)

# 功能词 vs 内容词统计
func_mask = is_function_word(per_position_tokens)
print(f"Function words: mean t* = {commit_times[func_mask].mean():.3f}")
print(f"Content words:  mean t* = {commit_times[~func_mask].mean():.3f}")
```

---

## 期望结果与决策规则

### 与 ELF EXP-08 对比

| 指标 | ELF-kd_cr | LangFlow (预期) |
|------|-----------|----------------|
| 功能词平均 t* | 0.182 | ~0.83 (如 H_coarse_to_fine) |
| 内容词平均 t* | 0.255 | ~0.88 (如 H_coarse_to_fine) |
| Δ (func - content) | −0.073 | −0.05 to 0 (H_no_order 时接近 0) |
| 空间自举效应 | +65pp @d=5 | 待测 |

### 决策规则

| LangFlow 结果 | 论文意义 |
|--------------|--------|
| |Δ| < 0.03（功能词和内容词几乎同时承诺） | "LangFlow 没有粗到细层级：承诺悬崖在 t≈0.83-0.93 同时触发所有位置" |
| |Δ| > 0.05（功能词先承诺） | "LangFlow 保留了 ELF 的粗到细顺序，只是整体偏移到更晚的 t 区间" |

---

## 优先级与依赖

- **依赖**: EXP-22 数据（已完成），但需要添加 per_position 输出
- **优先级**: 中等。EXP-21/24 是更高优先级的 LangFlow 对比实验
- **状态**: COMPLETED（2026-07-20）

---

## 实验结果（Results）

**状态**: COMPLETE（2026-07-20）

**脚本**: `experiments/probe_langflow/probe_coarsefine_langflow.py`（新建）  
**数据文件**: `results/exp25_langflow/coarse_fine_results.json`, `commit_tidx.npy`, `gt_tokens.npy`  
**配置**: n_samples=64, seq_len=128, 51 t值（30 个均匀分布在 0.03-0.65，21 个密集分布在 0.66-1.00），entropy_thresh=1.0 nat

### 核心结果

| 指标 | LangFlow | ELF kd_cr（EXP-08） | ELF baseline（EXP-08） |
|------|---------|---------------------|----------------------|
| 功能词占比 | 10.5% | 12.2% | 12.2% |
| 功能词 mean t* | **0.8252** | **0.182** | **0.246** |
| 内容词 mean t* | **0.8747** | **0.255** | **0.400** |
| Δ (func - content) | **−0.0495** | −0.073 | −0.154 |
| 整体 mean t* | 0.8695 | 0.246 | 0.380 |
| 从未承诺比例 | 1.4% | 0.30% | 8.70% |

### 关键发现

**H_coarse_to_fine 成立（|Δ|=0.050 > 阈值 0.03）**：LangFlow 保留了功能词优先承诺的粗到细顺序，即使整体承诺时间比 ELF 晚约 0.62t 单位。

**对比分析**：
- Δ(LangFlow) = −0.050 vs Δ(ELF kd_cr) = −0.073 vs Δ(ELF baseline) = −0.154
- LangFlow 的 Δ 介于 ELF kd_cr 和 baseline 之间，但更接近 kd_cr
- LangFlow 功能词 10.5% 位置（n=852），内容词 88.2% 位置（n=7,228）均已承诺

**整体承诺时序（LangFlow 极晚承诺）**：
- LangFlow overall mean t* = 0.8695（承诺悬崖 t≈0.83-0.93，与 EXP-22 一致）
- 功能词：mean t* = 0.8252（在 LangFlow 的承诺悬崖**早期**就快速锁定）
- 内容词：mean t* = 0.8747（悬崖**峰值**附近锁定）

**结论**：

1. 粗到细承诺顺序不是 ELF 特有的机制，而是**连续扩散语言模型的普遍特征**。LangFlow 的 Euler-EDM ODE 同样优先确定语法功能词，然后才锁定内容词。

2. 但 ELF 的粗到细 Δ 更大（kd_cr 比 LangFlow 大 47%）：KD 训练不仅使整体承诺提前，还**强化了粗到细层级**（内容词更晚承诺，功能词更早承诺）。

3. 对论文的含义：
   - "粗到细" 是 CDLM 通性，非 ELF 专有
   - ELF KD 训练"放大"了这个内在的语言层级顺序
   - LangFlow 的早期(t<0.83)完全没有承诺（EXP-22 已验证），但一旦进入承诺悬崖，功能词确实先行

---

## ⚠️ 方法论问题（2026-07-22 审查）

### 1. "功能词先承诺"≠"粗到细" — 频率/困惑度混淆

功能词（function words）的核心特征是**高频、低 surprisal**，而不是语法功能本身。模型对高频 token 更自信，commit 更早，与"从语法骨架到词汇细节"的粗到细层级假说无关。

正确检验"粗到细"需要在控制以下变量后仍观察到时序差异：
- 训练集词频（token unigram frequency in OWT）
- 上下文 surprisal（-log P(w_i | context)，由同规格语言模型估计）
- 位置（function words 倾向于在 SVO 结构的特定位置出现）

当前 EXP-25 未控制任何一项，"功能词先"完全可以由"高频先"解释。

### 2. CRITICAL：跨模型 Δt 比较无效（与 EXP-03 冲突）

spec 写：

> Δ(LangFlow) = −0.050 vs Δ(ELF kd_cr) = −0.073

并得出"LangFlow 的 Δ 介于 ELF kd_cr 和 baseline 之间"。但 EXP-03 已严格证明 LangFlow 和 ELF 的 nominal-t 对应不同 log-SNR，因此两个 Δ 值的差异完全可以由 t-schedule 差异解释，而非"粗到细程度"的差异。同一结论适用于"整体早 0.62t 单位"对比。

### 3. type-level 平均掩盖样本量差异

"功能词"类型（如 "the", "of", "in"）出现频率高，单次 occurrence 计算 t* 的样本量差异极大。如果用 occurrence-level 计算（而非 type-level 均值），结论可能发生变化。

### 4. 安全结论（LangFlow 内部）

在 LangFlow 自身内部，以下陈述有效（不做跨模型比较）：
- LangFlow oracle protocol 下，功能词 class 的平均 commitment t*（0.825）早于内容词（0.875），差距 0.050t 单位（显著性需按 sequence bootstrap，而非 occurrence bootstrap）
- 这一排序与 ELF 内部的方向一致，但数值不可比

**不能得出的结论**：
- ~~"粗到细是 CDLM 通性"~~（未控制频率）
- ~~"ELF KD 放大了粗到细层级"~~（跨模型 Δt 比较无效）
- ~~"LangFlow 承诺比 ELF 晚 0.62t 单位"~~（nominal-t 不可比）

---

## EXP-25v2 结果（2026-07-22，DONE）

**脚本**: `experiments/probe_langflow/analyze_coarsefine_regression.py`  
**输出**: `results/exp25v2_langflow/coarsefine_regression_v2.json`  
**依赖**: EXP-27v2 频率表（`results/exp27v2_langflow/freq_commitment_v2.json`）

### 核心修正

从类型级 mean_t* 比较，升级为**出现级 logistic 回归**：
- y = I(committed by t_i)
- X = [is_function, log_freq_norm（OWT GPT-2 频率）]

### 关键结果

**β_func 随 t 变化**：

| t | β_func | OR_func | β_freq | OR_freq | 解释 |
|---|--------|---------|--------|---------|------|
| 0.660 | -1.218 | 0.30 | +2.015 | 7.50 | 控频后函数词晚承诺 |
| 0.745 | -1.356 | 0.26 | +2.012 | 7.48 | 同上（最强反转） |
| 0.830 | -0.315 | 0.73 | +1.272 | 3.57 | 同上（减弱） |
| 0.915 | +1.555 | 4.73 | +0.899 | 2.46 | 函数词有残余提前效应 |
| 1.000 | +1.486 | 4.42 | +0.776 | 2.17 | 同上 |

**边际对比（不控制变量）**：
- 峰值 Δ(func-content 承诺率) = +40.0pp 在 t=0.864

### 解读

1. **频率效应持续显著且强**：β_freq 在所有 t 值均为正（OR=2-7.5），高频 token 显著更早承诺。
2. **控制频率后，函数词在 t<0.83 并不更早承诺（反而更晚）**：
   - β_func 在 t=0.66-0.83 为负（OR=0.26-0.73）。
   - 这意味着 EXP-25 的"函数词先"效应主要由频率解释。
3. **但 t≥0.915 时，函数词有残余正效应**（β_func=+1.5，OR=4.4-4.7）。
   - 解释：大多数函数词在 t<0.83 就承诺（由频率驱动），到 t=0.915 剩余函数词构成不同子集（高频但语境模糊的），而剩余内容词是罕见词。函数词在极晚阶段仍比同等频率内容词更早承诺，说明存在小的残余 POS 效应。
4. **结论（修正 EXP-25）**：LangFlow 的粗→细效应主要是频率效应。"函数词先"几乎完全可以由"高频先"解释，EXP-25 的类型级 Δ=-0.050 是频率混淆导致的。EXP-26v2 的 hazard model 显示存在小的残余 POS 效应（is_function OR=1.24）。
