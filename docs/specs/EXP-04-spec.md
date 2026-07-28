# EXP-04 Spec — Decoder Geometry Null Model

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：量化 G(t) 的方法论偏差，确保指标可信。**

G(t)（cosine-normalized token readout accuracy）是我们测量承诺程度的核心指标。但 G(t) 有一个潜在的系统性偏差：词汇嵌入矩阵 E 的几何结构可能导致"即使给随机噪声，也能正确预测某些 token"。

**具体问题**：ELF 的词汇嵌入矩阵 E（32100 × 512）中，某些 token 的嵌入向量可能在一个很大的"吸引域"中——任何从那个方向来的向量都会被分类为那个 token，哪怕这个向量是纯噪声。

**要验证的核心假说**：
- G_null(t) = 给 backbone 输入纯高斯噪声时，backbone 输出与真实 token 的 cosine 对齐率
- 如果 G_null >> 0%（chance ≈ 0.003%），说明 G(t) 被头部几何偏置膨胀了，需要报告 G_corrected = G_oracle - G_null
- 已知 mode_fraction（backbone 对同一个 token 的预测集中度）= 23.4%，证明几何偏置确实存在

**为什么重要**：如果 G_null ≈ 20%（哪怕随机噪声都能正确预测 20% 的 token），那么我们报告的 G(t) = 62%（baseline, t=0.3）实际上只有 42 个百分点的真实信号。这会影响对"承诺悬崖"幅度的定量解读。

**与其他实验的关系**：EXP-04 是 EXP-07（独立线性探针）的前置动机——如果 G(t) 被证明有 geometry bias，则需要 EXP-07 的独立探针作为更干净的指标。

**当前状态**：COMPLETE。阶段 1（probe_null_model.py，几何偏置确认）和阶段 2（EXP-04b，vs 真实 token）均已完成。结论：G_null ≈ 0.17%，几何偏置可忽略，G(t) 无需修正。

---

**Goal:** Determine whether ELF's cosine-normalized token readout accuracy G(t) reflects genuine
representation quality or is inflated by output head geometry. Feed pure Gaussian noise to the
backbone and compute G_null(t) against true token targets. High G_null → head geometry inflates G.

**Status:** Partially complete. `probe_null_model.py` is implemented and ran, showing
mean mode_fraction = 23.4% from pure Gaussian noise (chance = 0.003%). This confirms strong
geometry bias in the output head. What's missing: G_null against TRUE token targets (needs
validation data + T5 encoder). That determines whether the paper's G(t) claims need correction.

---

## Already done

`experiments/probe_elf/probe_null_model.py` — runs backbone on pure Gaussian z, computes:
- mode_fraction: % of positions predicting the most common token
- top5_coverage: % of positions predicting one of 5 most-predicted tokens
- n_unique_tokens_predicted: vocabulary diversity of null predictions
- mean_max_cosine_sim: average cosine similarity to nearest token centroid

**Result from baseline checkpoint (128 seqs × 1024 positions, 20 t values):**

| t range | mode_fraction | top5_coverage |
|---------|---------------|---------------|
| 0.05    | 48.4%         | 90.8%         |
| 0.10    | 34.8%         | 78.2%         |
| 0.30    | 29.5%         | 59.7%         |
| 0.55    | 18.0%         | 48.9%         |
| 1.00    | 47.1%         | 89.6%         |

**Conclusion from probe_null_model.py:** HIGH geometry bias confirmed.

---

## EXP-04b: Null model G against true tokens (COMPLETED 2026-07-18)

### File to create
`experiments/probe_elf/probe_null_vs_oracle.py`

### What it does
1. Load N=128 validation sequences from `embedded-language-flows/openwebtext-t5`
2. Run T5 encoder → x_clean (B, L, 512)
3. For each t in {0.05, 0.10, ..., 1.00}:
   a. z_null = torch.randn_like(x_clean)  ← pure Gaussian, no real signal
   b. z_oracle = t * x_clean + (1-t) * eps  ← standard oracle protocol
   c. Feed both through ELF backbone → x̂_null, x̂_oracle
   d. Compute G_null(t) = % positions where cosine(x̂_null, E) argmax = y_true
   e. Compute G_oracle(t) = % positions where cosine(x̂_oracle, E) argmax = y_true
4. Plot both curves; compute G_corrected = G_oracle - G_null

### Expected outcomes
- **Outcome A** (G_null < 5%): Null model doesn't align with true tokens → G(t) is genuine.
  Paper needs no correction.
- **Outcome B** (5% < G_null < 15%): Moderate inflation → report G_corrected alongside G.
  The G=60.8% at t=0.35 becomes G_corrected≈48-56%. Claims should say "cosine-readout accuracy,
  baseline-corrected for output head geometry."
- **Outcome C** (G_null > 15%): Strong inflation → major revision needed. The cliff may be
  mostly geometry. Story A claim ("backbone encodes token identity earlier") would require EXP-07
  linear probe as primary evidence instead.

### Code pattern
```python
# Load data
from utils.data_utils import load_dataset_split
from utils.encoder_utils import encode_text
ds = load_dataset_split("embedded-language-flows/openwebtext-t5")
# ds[i] has 'input_ids' (L,) — T5 vocab tokens
# Run T5 encoder on these to get x_clean

# Input IDs → T5 embeddings
from transformers import T5EncoderModel
t5 = T5EncoderModel.from_pretrained("t5-small").to(device).eval()
# ... (get last hidden states)
x_clean = encode_text(input_ids, attn_mask, t5, latent_mean=0.0, latent_std=0.2)

# For each t:
eps = torch.randn_like(x_clean)
z_oracle = t_val * x_clean + (1 - t_val) * eps
z_null = torch.randn_like(x_clean)

# Compare G(z_oracle) vs G(z_null)
```

### Effort: 1 day (mostly data loading boilerplate)

---

## 实验结果（Results）

**阶段 1（probe_null_model.py）**：COMPLETED

原始结果：`experiments/probe_elf/probe_null_model.py` 输出（见上方表格）

关键发现：在纯高斯噪声输入下，mode_fraction = 23.4%（随机概率 = 0.003%），top5_coverage 达 50–90%。这**强烈确认**输出头存在严重几何偏置。

**阶段 2（EXP-04b，vs 真实 token）**：COMPLETED（2026-07-18）

**原始数据文件**：`results/exp04b_baseline/G_null_vs_oracle.json`

**日志文件**：`/tmp/exp04b.log`

**完整结果表**（baseline checkpoint）：

| t | G_null | G_oracle | G_corrected |
|---|--------|---------|------------|
| 0.10 | 0.0027 | 0.0099 | 0.0073 |
| 0.20 | 0.0017 | 0.0994 | 0.0977 |
| 0.30 | 0.0020 | 0.5388 | 0.5368 |
| 0.50 | 0.0017 | 0.7993 | 0.7976 |
| 0.70 | 0.0088 | 0.8535 | 0.8447 |
| 1.00 | 0.0165 | 1.0000 | 0.9835 |

*注：G_oracle 低于 EXP-12 的 36.2%（t=0.20）和 61.9%（t=0.30）——差异来自 self-conditioning 状态不同（EXP-04b 每次独立采样噪声，EXP-12 使用 kd-cr 模型的 self-conditioning）*

**决策：几何偏置 SMALL，G(t) 无需修正**
- Mean G_null in [0.20, 0.40] = **0.0017**（0.17%）
- Mean G_oracle in [0.20, 0.40] = **0.4699**（47%）
- 几何膨胀比例：0.17/47 ≈ **0.4%**（可忽略）

**论文使用建议**：G(t) 指标**不需要**几何修正脚注。原先担心的 EXP-04 问题已排除。论文可以直接报告 G(t) 数值，不需要加 "potentially inflated by geometry" 的保守注释。

*原先的错误预估（预计 G_null ≈ 5-15%）来自将 mode_fraction（23.4%）误解为"G_null vs true tokens"，实际上 mode_fraction 只是几何集中度指标，与真实 token GT 无关。*

---

## Dense G(t) Baseline Curve — probe_geo_v2_dense（2026-07-20）

**状态**: COMPLETED（2026-07-20）

**脚本**: `experiments/probe_elf/probe_geo.py`
**数据**: `results/elf/probe_geo_v2_dense/probe_geo.json`
**配置**: checkpoint=ELF-B-owt（baseline），n_samples=64，seq_len=256，n_t_steps=51，n_noise=4，SC=0

### 关键 G(t) 和 cos(x̂_t, x_clean) 曲线（51 t 值）

**承诺悬崖区（t=0.10-0.40）**：

| t | G(t) = Rec@1 | cos(x̂_t, x_clean) | rho(t) |
|---|---|---|---|
| 0.100 | 1.5% | 0.256 | 174% |
| 0.140 | 2.8% | 0.295 | 159% |
| 0.180 | 5.5% | 0.360 | 152% |
| 0.200 | 8.6% | 0.423 | 148% |
| 0.220 | 14.2% | 0.514 | 140% |
| 0.240 | 21.5% | 0.614 | 128% |
| 0.260 | 30.1% | 0.700 | 116% |
| 0.280 | 38.0% | 0.757 | 108% |
| 0.300 | 45.3% | 0.798 | 101% |
| 0.320 | 51.3% | 0.826 | 97% |
| 0.340 | 57.0% | 0.849 | 93% |
| 0.360 | 61.7% | 0.866 | 90% |
| 0.400 | 68.6% | 0.889 | 86% |

**高 t 区（t=0.40-1.00）**：

| t | G(t) | cos(x̂_t, x_clean) |
|---|---|---|
| 0.500 | 76.6% | 0.911 |
| 0.600 | 79.3% | 0.916 |
| 0.700 | 79.7% | 0.916（peak） |
| 0.800 | 78.4% | 0.910 |
| 0.900 | 76.9% | 0.902 |
| 0.960 | 84.7% | 0.888 |
| 0.980 | 97.9% | 0.357（异常，见注）|
| 1.000 | 97.9% | 0.348（异常，见注）|

**注（t≥0.98 cos_clean 异常）**：在 t=0.98 处，cos_clean 从 0.888 急降至 0.357，这与 probe_geo_v1 的结果（t=1.00: 0.346）一致，是真实现象。

**原因**：ELF 在训练中使用自条件化（SC），SC 来自前一步的预测（非零）。但 oracle 探针（Protocol A）使用 SC=0。在 t=1.0（纯净输入）条件下，模型期望非零 SC 但接收到零 SC，导致 x̂_1 的方向偏离 x_clean（cos≈0.35）。然而 argmax（token 识别）仍然高度准确（G=97.9%），因为最近邻查找对方向的微小偏移不敏感。

这实际上是"表示-读出 gap"的另一种体现：x̂_1 不需要与 x_clean 几何完全对齐，只需在 token 判别子空间中接近正确类别。

---

## ⚠️ 方法论问题 & 待修正 TODO（2026-07-21 审查）

### 问题 1（关键）：当前 "null" 经过了完整 backbone，不是真正的 "head geometry" null

当前 `probe_null_model.py` 将纯高斯噪声输入**整个 backbone**（包括 time embedding、SC embedding、所有 transformer 层），再接 decode head。

这测量的是 **backbone learned prior + output head geometry** 的混合，不是单独的 head geometry。

**真正的 head-only null**：直接对 decode head 输入各向同性高斯（跳过 backbone），测量 head 本身的对 true token 的 coverage。

```python
# 真正的 head-only null（跳过 backbone）:
h_rand = torch.randn(B, L, 768)  # 标准高斯 → 模拟 backbone output
x_hat_null = gelu(h_rand @ proj_kernel + proj_bias) @ unembed_kernel
# 或者用 matched mean/cov:
h_rand = mu_backbone + std_backbone * torch.randn(B, L, 768)
```

### 问题 2：现有结论"几何偏置可忽略"过于强烈

G_null ≈ 0.17% 是针对 true token GT 的 aggregate accuracy，这确实很低。但这**不能**说明：
- backbone geometry 不改变候选排名（rank inflation 可能存在）
- 不存在 margin / entropy 偏置
- 高频 token 的 recall 不被偏置

23.4% 的 mode_fraction 说明 backbone 对纯噪声有很强的输出集中度，这本身是很大的偏置。

### 问题 3：t≥0.98 的 cos 异常（0.89→0.35）是 SC=0 的 OOD artifact

当前解释为"representation-readout disentanglement"，但更直接的解释是：模型在 t=1.0（干净输入）时期望非零 SC，而 oracle 探针用 SC=0，导致 x̂_1 偏向错误方向。在报告这个现象时需要明确注明是 OOD artifact。

### 修正 TODO

- [ ] **P0**：实现真正的 head-only null：`h_rand → proj_kernel → GELU → unembed_kernel`，计算 G_head_only_null vs true tokens
- [ ] **P0**：实现 readout ablations（四条路径：`W*h+b`, `W*h`, `||W_v||^-1 * W_v*h`, `cos(W_v, h)`），看哪个成分贡献最大偏置
- [ ] **P1**：与 empirical unigram baseline 对比（而非 1/32100 chance），计算频率矫正后的 G(t)
- [ ] **P1**：分 token frequency bucket 分析（高频 top-1000 vs 中频 vs rare），看偏置是否集中在高频词
- [ ] **P2**：在论文中将结论修改为"The null backbone output concentrates on a small token set (mode_fraction=23.4%), but this concentration has low overlap with true tokens (G_null=0.17%), leaving open the question of whether rank inflation affects per-position margin."

---

## EXP-04v2 — Head-Only Null vs Backbone Null vs Oracle（2026-07-21 完成）

**背景**：EXP-04 的"修正 TODO P0"——实现真正的 head-only null，将头部几何、backbone 先验、oracle 信号三者分离。

**脚本**：`experiments/probe_elf/probe_head_null.py`  
**输出**：`experiments/probe_elf/results/exp04v2_baseline/probe_head_null_baseline.json`  
**设置**：baseline checkpoint，n_samples=64，seq_len=1024，20 个 t 值  

### 完整结果表

| t | G_head_null | G_backbone_null | G_oracle |
|---|-------------|-----------------|----------|
| 0.05 | 0.0145% | 0.322% | 0.229% |
| 0.10 | 0.0153% | 0.317% | 0.892% |
| 0.15 | 0.0191% | 0.248% | 2.653% |
| 0.20 | 0.0206% | 0.193% | 7.515% |
| 0.25 | 0.0217% | 0.238% | 21.83% |
| 0.30 | 0.0156% | 0.214% | 43.81% |
| 0.35 | 0.0164% | 0.172% | 59.55% |
| 0.40 | 0.0172% | 0.157% | 66.71% |
| 0.45 | 0.0179% | 0.181% | 70.48% |
| 0.50 | 0.0198% | 0.219% | 72.83% |
| 0.55 | 0.0114% | 0.305% | 74.65% |
| 0.60 | 0.0168% | 0.474% | 76.08% |
| 0.65 | 0.0164% | 0.742% | 77.21% |
| 0.70 | 0.0175% | 1.009% | 78.11% |
| 0.75 | 0.0179% | 1.199% | 78.77% |
| 0.80 | 0.0202% | 1.308% | 79.28% |
| 0.85 | 0.0160% | 1.288% | 79.69% |
| 0.90 | 0.0156% | 1.325% | 80.19% |
| 0.95 | 0.0213% | 1.959% | 84.03% |
| 1.00 | 0.0172% | 1.821% | 95.07% |

### 关键发现

**1. Head-only null 基本是随机的**

G_head_null 在所有 t 下约为 **0.014–0.022%**（理论 chance = 1/32100 ≈ 0.003%）。头部本身（proj_kernel → GELU → unembed_kernel）在纯高斯输入下几乎无法猜中真实 token。这**排除了"输出头几何偏置直接膨胀 G(t)"的假设**。

**2. Backbone null 很小但有 t 依赖**

G_backbone_null 在低 t（0.05–0.40）约为 **0.15–0.32%**，在高 t（0.65–0.95）升至 **0.74–1.96%**。这反映了模型的 **learned frequency prior**：即使输入纯噪声，backbone 的 time embedding 也会激活不同的处理路径，高 t 时更多地预测常见词。但这个先验始终很小，最大约 2%。

**3. Oracle 信号完全主导 G(t)**

在承诺悬崖区（t=0.25–0.35），G_oracle 从 22% 跳至 60%，而 G_backbone_null 始终 ≈ 0.2%。oracle signal-to-noise ratio = G_oracle / G_backbone_null 在悬崖区约为 **100:1**。

**4. 结论：三层分离清晰**

| 来源 | 量级 | 含义 |
|------|------|------|
| Head geometry (G_head_null) | ≈ 0.017% | 可忽略，头部不产生 token-specific prediction |
| Backbone prior (G_backbone_null) | 0.15–2% | 很小的频率先验，但非零；高 t 略有放大 |
| Oracle signal (G_oracle - G_backbone_null) | 99.8% of G_oracle | G(t) 的信号几乎完全来自 x_clean 信息 |

**论文影响**：G(t) 的几何偏置（G_head_null ≈ 0.017%）和频率先验偏置（G_backbone_null ≈ 0.2–2%）均不足以影响 G(t) 的定性解读。EXP-05v3 的全局 null prior（基于 G_backbone_null 机制）可作为精确去偏工具。

### 关键发现

1. **承诺悬崖精确定位**：G(t) 在 t=0.20-0.30 之间急跳（8.6%→45.3%，Δ=36.7pp，Δt=0.10）。更精细的采样显示悬崖中点在 t≈0.27（G=38%）。
2. **cos(x̂_t, x_clean) 峰值在 t≈0.65**（0.917），而非 t=1.0 —— 说明模型在中等噪声水平下输出的 x̂_t 最接近 x_clean。
3. **G(t) 非单调**：t=0.66（79.9%）→ t=0.90（76.9%）下降 3pp，然后 t=0.96 回升至 84.7%。这对应原 probe_geo_v1 中同样观察到的非单调性。
4. **G(t) 与 cos(x̂_t, x_clean) 的 disentanglement**：在悬崖区（t=0.20-0.35），两者快速上升且强相关；在 t=0.60-0.90，G(t) 下降而 cos 保持稳定，说明在高 t 区 token 判别和几何恢复是解耦的。
