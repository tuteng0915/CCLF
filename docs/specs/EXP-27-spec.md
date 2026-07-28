# EXP-27 Spec — LangFlow Token Frequency vs Commitment Timing (LangFlow analog of EXP-20)

## 实验背景与动机

**在整体框架中的地位：验证 token 频率与承诺时间的关系是否在 LangFlow 上也成立（EXP-20 的 LangFlow 对比）。**

EXP-20 在 ELF 上发现：baseline 中 rare token（出现 < 50 次）的 never_commit_rate（21.6%）
高于 common token（≥500 次，13.5%），差距 8pp。这表明 backbone 对低频词汇的表示质量更差。

**EXP-27 核心问题**：
1. LangFlow 是否也有"高频 token 更早承诺，低频 token 更晚承诺"的频率-时间梯度？
2. 承诺时间 t* 与 T5 token ID（频率代理：低 ID = 高频）的相关性如何？

---

## 数据来源

**EXP-25 per-position commitment 数据（CPU 分析，无需 GPU）**：
- `results/exp25_langflow/commit_tidx.npy` — [64, 128] per-position 首次正确承诺的 t 索引
- `results/exp25_langflow/gt_tokens.npy` — [64, 128] 真实 token ID
- `results/exp25_langflow/t_grid.npy` — 51 个 t 值

**T5 SentencePiece 频率代理**：
- ID 0 = `<pad>`, 1 = `</s>`, 2 = `<unk>`（特殊 token）
- ID 3~32099: SentencePiece 词片，近似按训练频率降序排列（低 ID = 更高频）
- 注：T5 训练语料包含多语言内容，"very_common" 层中包含部分非英文高频片段

---

## 频率分组

| 组别 | ID 范围 | 解读 |
|------|---------|------|
| very_common | [3, 100) | 顶端 97 个类型（最高频） |
| common | [100, 500) | 高频词片 |
| medium | [500, 2000) | 中频词片 |
| uncommon | [2000, 10000) | 低频词片 |
| rare | [10000, 32100) | 罕见词片 |

---

## 实验结果（Results）

**状态**: COMPLETED（2026-07-20）

**脚本**: `experiments/probe_langflow/analyze_token_freq_langflow.py`
**数据**: `results/exp27_langflow/token_freq_analysis.json`
**配置**: n_samples=64, L=128, n_t=51, min_occ=10

### 相关性

- **correlation(log10(tok_id), mean_t*) = +0.466**（min_occ≥10，69 token types）
- **correlation(log10(tok_id), mean_t*) = +0.315**（min_occ≥3，483 token types）
- 正相关（r > 0）：高 ID（稀有）token 确实承诺更晚
- 比 ELF 的频率-时间梯度更弱（数据量有限，仅 8192 positions）

### 频率分组统计（min_occ≥3）

| 组别 | ID 范围 | n_types | mean_t* | never_rate |
|------|---------|---------|---------|------------|
| very_common | [3, 100) | 33 | 0.8680 | 1.1% |
| common | [100, 500) | 71 | 0.8613 | 0.3% |
| medium | [500, 2000) | 209 | 0.8791 | 0.9% |
| uncommon | [2000, 10000) | 147 | 0.8953 | 1.1% |
| rare | [10000, 32100) | 23 | 0.8907 | 2.9% |

**趋势**：
- common → uncommon 明显上升（0.8613 → 0.8953, Δ=+0.034）
- very_common 看似较高（0.8680），但 very_common 层包含多语言高频词片（如 '▁și' [Romanian], '▁dans' [French]）会拉低均值
- rare 的 never_rate（2.9%）约是 common（0.3%）的 10×，与 EXP-20 的 ELF 趋势方向一致

### 典型 token 举例（min_occ≥3）

**最早承诺（top-5 by mean t*）**：

| tok_id | piece | mean_t* | n |
|--------|-------|---------|---|
| 198 | `▁și` | 0.776 | 324 |
| 247 | `▁dans` | 0.784 | 65 |
| 447 | `ic` | 0.793 | 90 |
| 82 | `▁my` | 0.800 | 52 |
| 13 | `▁of` | 0.804 | 271 |

→ 功能词（`▁my`, `▁of`）和短词片（`ic`）最早承诺，与 EXP-25 的粗到细发现一致

**最晚承诺（top-5 by mean t*）**：

| tok_id | piece | mean_t* | n |
|--------|-------|---------|---|
| 4041 | `▁partie` | 0.987 | 4 |
| 3397 | `▁Festival` | 0.983 | 3 |
| 7305 | `▁Village` | 0.983 | 3 |
| 3624 | `star` | 0.966 | 3 |
| 17604 | `▁commander` | 0.966 | 3 |

→ 稀有内容词（命名实体类：Festival, Village, commander）承诺最晚，且出现次数极少（n=3-4）

### 对比 ELF EXP-20

| 指标 | ELF baseline | LangFlow |
|------|-------------|----------|
| 整体 never_commit_rate | 19% | 1.4% |
| rare - common never_rate 差距 | ~8pp | ~2.6pp (rare 2.9% vs common 0.3%) |
| correlation(log10(id), t*) | 未直接计算（EXP-20 用 never_rate 分析）| r = +0.47 |
| 最早承诺 token | 高频功能词 | 高频功能词 + multilingual |
| 最晚承诺 token | 稀有词 + 标点 | 稀有内容词 + 命名实体 |

### 局限性

1. **样本量过小**：8192 positions 中每个 token type 的样本量极少（median n ≈ 3）。r=0.47 的相关性估计噪声大，min_occ≥10 时仅 69 token types 通过筛选。

2. **T5 ID 不是纯频率代理**：T5 的 SentencePiece 在多语言语料上训练，low ID 层含大量非英文高频词片，打乱了纯频率假设。'▁și'（罗马尼亚语 'and'，ID=198）的早期承诺可能是因为它在多语言数据中频繁出现，而非因为 ID 低。

3. **LangFlow 数据需更多 samples**：若要做清晰的频率-承诺分析，需要 ≥200 samples（目前只有 64）以覆盖更多 token types。

### 结论

**EXP-27 支持了"频率-承诺梯度"作为 CDLM 通性**：
- LangFlow 和 ELF 都显示高频 token 比低频 token 更早承诺
- LangFlow 的效应较弱（ELF 19% vs 1.4% never_rate，梯度更平）
- 这与粗到细发现（EXP-25）一致：功能词（通常是高频词）更早承诺

**论文建议**：在 EXP-20/EXP-25 小结中加一句：
"LangFlow 上同样观察到频率-承诺梯度（r=0.47, log tok-id vs t*），尽管因数据量限制无法完全对应 ELF 的分析。频率-承诺相关性是 CDLM 通性，而非 ELF 特有。"

---

## ⚠️ 方法论问题（2026-07-22 审查）

### 1. CRITICAL：token ID ≠ OWT 训练频率

EXP-27 用 T5 SentencePiece token ID 作为频率代理，核心假设是"低 ID = 高频"。但：

1. **T5 在多语言 C4 上训练**，而 LangFlow 在 OWT（英文 WebText）上训练。多语言频率排序与 OWT 英文频率排序不一致。`▁și`（罗马尼亚语"and"，ID=198）对 LangFlow 而言是低频词（几乎不出现在 OWT），但 T5 tokenizer 给它一个低 ID。
2. **GPT-2 tokenizer（BPE）vs T5（SentencePiece）**：LangFlow 使用 GPT-2 tokenizer（BPE），token ID 不按频率排序，而是按 merge 操作顺序排列。EXP-27 用 T5 token ID（来自 ELF 的 tokenizer）分析 LangFlow 数据，两者 vocabulary 不同。

> **极可能是 EXP-27 用了错误的 tokenizer 对应关系**。需确认 `gt_tokens.npy` 的 token ID 来自哪个 tokenizer。

### 2. type-level 相关系数的样本量问题

r=0.47 基于 69 个 token types（min_occ≥10），其中许多 types 的样本量极小（n≈3-10 个 occurrence）。每个 type 的 mean_t* 估计误差极大，type-level 相关系数的有效 n=69 会严重高估精度（实际 SE 比名义值大几倍）。

### 3. 频率 vs 困惑度未分离

高频 token 在语言模型中通常也有低 surprisal（-log P(w | context)）。频率效应和困惑度效应完全混淆：
- "模型对高频词更自信" = 频率效应
- "高频词在上下文中更可预测" = 语言结构效应

这两者产生相同的 commitment 模式，当前数据无法区分。

### 4. 安全结论

- LangFlow 内部：高 token ID（代理稀有词）的平均承诺 t*（0.895）稍晚于低 ID（0.861-0.868）
- 相关性方向与 ELF 内部一致
- **不能得出**：
  - ~~"频率-承诺梯度是 CDLM 通性"~~（T5 ID 不是 OWT 频率的有效代理，且跨模型比较无效）
  - ~~"r=0.47 表明显著的频率-时序关系"~~（样本量不足，且 ID 与频率关系存疑）

---

## EXP-27v2 结果（2026-07-22，DONE）

**脚本**: `experiments/probe_langflow/analyze_freq_commitment_v2.py`  
**数据**: 2000 OWT 文档，2,285,886 GPT-2 tokens → 44,320 unique types  
**输出**: `results/exp27v2_langflow/freq_commitment_v2.json`

### 核心修正

使用 GPT-2 OWT unigram 频率（而非 T5 token ID）作为频率代理。

### 关键结果

| 指标 | 数值 |
|------|------|
| 分析类型数 | 188（≥5 次 occurrence） |
| Pearson r(log_ppm, t*) | **-0.651** (p=4.7e-24) |
| Spearman r(log_ppm, t*) | **-0.659** (p=7.85e-25) |
| 偏相关 r 控制 is_func | **-0.638** (p=7.1e-23) |
| 函数词 mean t* | 0.860 |
| 内容词 mean t* | 0.874 |
| Δ(func - content) | **-0.014** |
| Mann-Whitney p | 3.9e-5 |

**频率五分位数（低→高频）**：

| 五分位 | mean log_ppm | mean t* |
|--------|-------------|---------|
| Q1 (lowest) | 2.14 | 0.892 |
| Q2 | 2.63 | 0.879 |
| Q3 | 2.91 | 0.872 |
| Q4 | 3.23 | 0.856 |
| Q5 (highest) | 3.81 | 0.836 |

### 解读

1. **频率-承诺时序关系极强**：r=-0.651，高频 token 显著更早承诺（Q1→Q5：0.892→0.836，Δ=5.6pp）。
2. **控制函数词属性后频率效应仍然强**：偏相关 r=-0.638，说明频率是独立预测因子。
3. **函数词 vs 内容词的差距极小**：Δ=-0.014（EXP-25 报告为 -0.050，ELF kd_cr 为 -0.075）。Man-Whitney 虽然显著但效应量极小。
4. **主要结论**：LangFlow 中"函数词先承诺"的表象很可能主要由**频率效应**解释——函数词本身是高频词。EXP-25v2 regression 验证了这一假设。
