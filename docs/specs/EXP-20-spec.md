# EXP-20 Spec — Token Frequency and Commitment Timing Analysis

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：刻画"永不承诺"位置的 token 特征——是 rare token 还是结构性 token？**

EXP-16 发现 baseline 有 19% 的位置"永不承诺"（在 t∈[0.10, 0.70] 范围内从未正确解码）。但这 19% 是哪些位置？两种可能的解释：

1. **频率假设**：rare token（出现次数少）更难学习，所以 baseline 对它们永远不确定
2. **结构假设**：某些 token 在语义上模糊或高度上下文依赖（如 `"`, `."`），backbone 的 oracle 协议特别难以确定

**要验证的核心假说**：
- 永不承诺的 token 中，rare token（出现 <50 次）的占比 >> common token（≥500 次）
- 某些特殊 token（EOS, 标点符号、引号等）100% 永不承诺，无论频率如何

**重要性**：
- 如果主要是 rare token：说明 baseline 对长尾词汇的表示质量差，KD 通过改善 rare token 的编码提升了承诺率
- 如果主要是结构 token（标点等）：说明 oracle 测量协议本身对这类 token 不适用（T5 编码中标点的 embedding 可能不与 token id 对齐）

**与其他实验的关系**：
- 直接使用 EXP-16 的承诺时序数据（baseline 的永不承诺位置）+ EXP-07b 的 y_tokens
- 为 EXP-01（Protocol B）的分析提供背景：在真实轨迹中，这些 token 是否同样难以承诺？

---

## Implementation

**Script:** `/tmp/exp20_token_analysis.py`

**方法**：
1. 加载 baseline 的 commit_timing（EXP-16 结果）
2. 对每个 token type，计算 never_commit_rate = 永不承诺次数 / 总出现次数
3. 按频率分组：rare (<50)、medium (50-499)、common (≥500)
4. 输出各组的 never_commit_rate 和典型 token 示例

---

## 实验结果（Results）

**状态**: COMPLETED（2026-07-18）

**原始数据文件**：`results/exp20/token_commit_analysis.json`

**日志文件**：`/tmp/exp20.log`

**频率分组统计**：

| 组别 | token 数量 | never_commit_rate |
|------|-----------|-------------------|
| Rare (出现 < 50 次) | 14,667 types | **21.6%** |
| Common (出现 ≥ 500 次) | 51 types | **13.5%** |

**100% 永不承诺的 token（出现 ≥ 10 次）**（部分列表）：

| token id | 描述（估计） | 出现次数 |
|---------|------------|---------|
| 1 | `</s>`（EOS token）| 254 |
| 121 | `"` 或引号类 | 255 |
| 535 | `."` 或类似 | 189 |
| 784 | 某标点 | 94 |
| 1239 | 罕见词 | 347 |

**100% 总是承诺的 token（高频示例）**：

| token id | 描述 | 出现次数 |
|---------|------|---------|
| 66 | `all`（估计）| 387 |
| 97 | `time` | 268 |
| 112 | `his` | 589 |
| 132 | 常见词 | 302 |

**关键发现**：

1. **Rare token 的 never_commit_rate（21.6%）高于 common token（13.5%）**，但差距不如预期大（仅 8pp）。说明 token 频率不是唯一决定因素。

2. **EOS token（tok=1）100% 永不承诺**：EOS 在 T5 编码中的 embedding 可能与词汇 embedding 的距离关系和其他 token 不同，oracle 协议难以捕获。

3. **高频功能词（`all`, `time`, `his`）总是承诺**：这些常见且语义明确的词在任何 t 下都能正确解码，符合直觉。

4. **结构性/标点 token 多次出现在永不承诺列表中**：tok=121（引号）、535（句号+引号）等，可能因为 T5 对这类 token 的上下文编码高度变化，oracle 协议无法在任何 t 下稳定解码。

**论文使用建议**：
- 这个分析可作为脚注或 appendix，说明"永不承诺"不是 KD 与 baseline 的主要区别点（KD 几乎消除了这个现象），而是刻画 baseline 行为的辅助数据
- 可以指出 EOS 和标点的 oracle 协议失效，说明我们的承诺率估计是保守的（真实数字可能更高）

---

## ⚠️ 方法论问题（2026-07-22 审查）

当前 EXP-20 结果作为辅助证据可以保留，但有以下问题，引用时需注意：

### 1. "永不承诺"定义不成立

实验复用 EXP-16 的 oracle grid（只检查 t∈{0.10, 0.30, 0.50, 0.70}等几个离散点）。正确表述是：

> "在所测试的 oracle t 值范围内未被 native decoder 正确读取"（not "never"）

EXP-16v2 显示 baseline 在 t=0.90+ 时接近全量正确，因此这批 21.6% 的位置中大多数并非"真正永不承诺"。

### 2. 频率来源是测试样本，而非训练语料

"rare type 14,667 个，common type 51 个" 反映的是当前 256×256 分析样本中的统计，而非 OpenWebText 训练语料的 token frequency。若要验证训练频率效应，必须用 `freq_train(v)` 从完整语料获取，否则同时反映样本量和主题偏差。

### 3. token ID 描述不能用"估计"

表格中 token 121 "可能是引号"、token 535 "可能是句号+引号" 等估计描述在论文中不可接受。所有 token ID 必须通过 T5 tokenizer 精确 decode（`tokenizer.decode([id])`）。

EOS（token 1）100% 未恢复首先需要 audit：attention mask 是否将其排除、label shift 是否正确、是否是 padding 位置——这是数据/masking 问题，不是语义发现。

### 4. token type-level rate 方差不均匀

一个出现 10 次的 token 全部失败即进入 "100% never" 列表，置信区间极宽。最低 occurrence cutoff 应提高到 ≥50，并报告每种 token 的置信区间。

### 正确分析方向

比较 baseline 和 kd_cr 在相同位置的 T_first（EXP-16v2），按 `freq_train(v)` bucket 分层，回答：**rare token 是 backbone 没有信息（probe 也差），还是 native decoder 特别读不出来（probe 好但 native 差）？** 这能直接支撑 decode interface 故事。
