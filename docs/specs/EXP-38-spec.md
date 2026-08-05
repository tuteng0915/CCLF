# EXP-38: ELF Logit Lens

## 目标

对 ELF 每个 Transformer block 的残差流应用 decode head（logit lens），  
测量在每个层深度处正确 token 的可读出概率。

**核心问题**：KD 改变的是 decode 接口（proj_kernel / unembed_kernel）还是  
backbone 的中间层表示？如果两个 checkpoint 的 logit lens 曲线相似但 L11  
的最终精度不同 → decode 接口是关键变量；如果早层就已分叉 → backbone 也被改变了。

## 方法

**Logit Lens 定义**：在 block_i 输出处 h_i ∈ ℝ^768，直接应用该 checkpoint 的 decode head：

$$\text{logits}_i = \text{GELU}(h_i W_{\text{proj}} + b_{\text{proj}}) W_{\text{unembed}} + b_{\text{unembed}}$$

注意：decode head 原本只在 block_11 输出处使用，因此 logit lens 测量的是  
"如果在层 i 处就停止，decode head 能读出多少正确 token 信息"。

**数据**：复用 `results/exp07b_v2_{baseline,kd_cr,kd2}/layer_states_t*.pt`  
（无需新的 GPU forward pass），每个文件含 layer_feats（12 层 [B,L,768]）和 y_tokens。

**指标**：
- top-1 accuracy（最重要）
- top-5 accuracy
- MRR（mean reciprocal rank）
- 熵（logit 分布的不确定度）

**批量处理**：每次 16 条序列以避免 logit tensor（V=32100）的 OOM。

## 代码

`experiments/probe_elf/logit_lens_exp38.py`  
（无需 GPU，在 CPU 上运行完整分析）

## 预期结果

| 假设 | 如果成立，说明 |
|------|---------------|
| kd_cr 和 baseline 在 B0-B10 几乎相同，仅 B11 分叉 | KD **只改变 decode 接口**，backbone 表示相同 |
| kd_cr 在 B5-B7 就开始领先 baseline | KD **也改变了 backbone 中间层**的表示 |
| baseline 在某层 i < 11 的 top-1 和 kd_cr L11 相当 | kd_cr 的 decode head 能读出更早层的信息 |

## 输出

`results/exp38_logit_lens/logit_lens.json`

结构：`{checkpoint: {t_str: {block_i: {top1, top5, mrr, entropy, n}}}}`

---

## 结果

**状态：DONE**

### Logit Lens Top-1 准确率（t=0.200）

| block | B00 | B02 | B04 | B06 | B07 | B08 | B09 | B10 | B11 |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| baseline | 0.002 | 0.004 | 0.023 | 0.087 | 0.138 | 0.243 | 0.357 | 0.396 | 0.361 |
| kd_cr | 0.019 | 0.030 | 0.082 | 0.188 | 0.297 | 0.422 | 0.493 | 0.544 | **0.585** |
| kd2 | 0.023 | 0.032 | 0.084 | 0.181 | 0.290 | 0.429 | 0.502 | 0.545 | **0.572** |

### Logit Lens Top-1 准确率（t=0.500）

| block | B00 | B02 | B04 | B06 | B07 | B08 | B09 | B10 | B11 |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| baseline | 0.187 | 0.087 | 0.212 | 0.368 | 0.362 | 0.408 | 0.675 | **0.784** | 0.756 |
| kd_cr | 0.404 | 0.274 | 0.506 | 0.706 | 0.805 | 0.881 | 0.896 | 0.988 | **0.995** |
| kd2 | 0.391 | 0.282 | 0.463 | 0.662 | 0.790 | 0.889 | 0.909 | 0.979 | **0.990** |

### Logit Lens Top-1 准确率（t=1.000）

| block | B00 | B02 | B04 | B06 | B07 | B08 | B09 | B10 | B11 |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| baseline | 0.534 | 0.232 | 0.547 | 0.797 | 0.858 | 0.938 | **0.960** | 0.934 | 0.901 |
| kd_cr | 0.782 | 0.554 | 0.815 | 0.926 | 0.949 | 0.964 | 0.990 | 0.998 | **0.999** |
| kd2 | 0.733 | 0.548 | 0.788 | 0.913 | 0.940 | 0.971 | 0.981 | 0.993 | **0.998** |

### 关键发现

**1. kd_cr 和 kd2 从 B00 就已领先 baseline（t=0.5: B00 0.40 vs 0.19）**

两种 KD 变体在最早层就已经与 baseline 分叉。考虑到 EXP-42 在 B00 的 CKA=0.981（表示很相似），这种差距主要来自 **decode head 的不同**（对同样的 B00 表示，kd_cr decode head 读出能力更强）。与 EXP-39 结论一致：baseline backbone + kd_cr head = 80.8% > baseline native 75.6%（+5.2pp 来自 head）。

**2. kd_cr ≈ kd2 在所有层（B11: 0.995 vs 0.990 at t=0.5）**

两种 KD 变体的 logit lens 精度在各层几乎相同。这意味着它们预测正确 token 的能力相近，**差异不在 oracle accuracy 或 logit lens accuracy，而在 B11 表示的方向**（EXP-42: kd_cr vs kd2 B11 rel_L2=0.500）。这解释了为什么 SC 效果天差地别——kd_cr 的 B11 方向支持 SC 利用，kd2 的不支持。

**3. baseline 非单调：B10 > B11（t=0.5: 0.784 > 0.756；t=1.0: B09=0.960 > B10=0.934 > B11=0.901）**

baseline 的最后几个 block 实际上**降低**了 logit lens 准确率。这说明 baseline 的晚层在做某种对 decode head 有害的表示变换（可能是噪声适应变换，提高了 backbone output x̂_t 的质量但降低了 token 可读性）。kd_cr 的晚层则单调提升（B09=0.896 → B10=0.988 → B11=0.995）。

**4. 与其他实验的综合解读**

| 实验 | 发现 | 与 EXP-38 的关系 |
|------|------|-----------------|
| EXP-39 | backbone 是主要因素，head 贡献 +5.2pp | EXP-38 B00 差距主要来自 head |
| EXP-42 | B08-B11 CKA 剧降，B00-B07 相对稳定 | EXP-38 显示即使 B00 就有差距（head 驱动），但 B08-B11 差距最大 |
| EXP-41 | baseline correct cos_align=0.234 最高 | 与 baseline B11 logit lens peak at B10 一致（更"尖锐"但非单调） |
