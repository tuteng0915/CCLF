# EXP-39: Decode Head Cross-Patch（因果检验）

## 目标

**因果隔离**：仅替换 decode head，测量 oracle accuracy 的变化。

如果：baseline backbone + kd_cr decode head → oracle acc ≈ 99%，  
则说明 kd_cr 的 oracle accuracy 提升完全由 decode head 解释，backbone 不变即可。

## 方法

对每个 backbone 来源 × decode head 目标，构成 3×3 组合矩阵：

| backbone (h_L11) | decode head | oracle acc |
|---------|-------------|------------|
| baseline | baseline | ? |
| baseline | kd_cr | ? |
| baseline | kd2 | ? |
| kd_cr | baseline | ? |
| kd_cr | kd_cr | ? (= EXP-07b_v2 结果) |
| kd_cr | kd2 | ? |
| kd2 | baseline | ? |
| kd2 | kd_cr | ? |
| kd2 | kd2 | ? (= EXP-07b_v2 结果) |

**数据**：复用 `results/exp07b_v2_{baseline,kd_cr,kd2}/layer_states_t*.pt`  
的 `layer_feats[-1]`（block_11 输出）。

**指标**：top-1, top-5, MRR（不含 entropy，聚焦精度）。

**批量处理**：16 条序列/批。

## 代码

`experiments/probe_elf/cross_patch_exp39.py`

## 关键预期

**若 decode head 完全决定精度**（支持"decode 接口"假说）：
- `baseline backbone + kd_cr head ≈ 99%` (和 kd_cr native 相同)
- `kd_cr backbone + baseline head ≈ 74%` (和 baseline native 相同)

**若 backbone 也有贡献**：
- `baseline backbone + kd_cr head < 99%` 但 > 74%（部分提升）
- `kd_cr backbone + baseline head > 74%`（backbone 本身也有提升）

## 输出

`results/exp39_cross_patch/cross_patch.json`

结构：`{backbone_src: {head_tgt: {t_str: {top1, top5, mrr}}}}`

---

## 结果

**状态：DONE**

### 3×3 oracle accuracy 矩阵（t=0.500）

| backbone \ head | baseline | kd_cr | kd2 |
|----------------|---------|-------|-----|
| **baseline**   | 0.7555 | 0.8077 | 0.7770 |
| **kd_cr**      | 0.9944 | **0.9948** | 0.9856 |
| **kd2**        | 0.9915 | 0.9928 | **0.9903** |

### 3×3 oracle accuracy 矩阵（t=1.000）

| backbone \ head | baseline | kd_cr | kd2 |
|----------------|---------|-------|-----|
| **baseline**   | 0.9011 | 0.9313 | 0.9012 |
| **kd_cr**      | 0.9987 | **0.9986** | 0.9915 |
| **kd2**        | 0.9984 | 0.9985 | **0.9980** |

### 关键发现（颠覆旧叙事）

**backbone 是决定 oracle accuracy 的主要因素，而非 decode head。**

- `baseline backbone + kd_cr head = 80.8%`（仅 +5.2pp，head 贡献小）
- `kd_cr backbone + baseline head = 99.4%`（+23.8pp，backbone 贡献大）
- kd_cr backbone 使用任意 head 均能达到 99.4-99.5%（head 几乎可互换）
- baseline backbone 使用任意 head 最高仅 80.8%

**与 EXP-15v2 的矛盾调和**：unembed_bias 的 Frobenius ratio R=2.59 反映的是相对变化（原始 bias 很小 L2=79.6，变化看起来大）。但功能上，backbone（尤其 B08-B11，见 EXP-42）的改变才是 oracle accuracy 提升的主要来源。

**对论文叙事的影响**：
- **必须修改**："KD reorganizes the decode interface" → "KD reorganizes the backbone's late-layer representations (B08-B11)"
- decode head 的变化（包括 unembed_bias）是次要的副产品，不是主要机制
