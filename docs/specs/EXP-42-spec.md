# EXP-42: 残差流差异分析（Residual Stream Divergence）

## 目标

由于 exp07b_v2 对所有 checkpoint 使用相同输入（y_tokens 验证完全匹配），  
可以逐层比较不同 checkpoint 的残差流 h_i，定位 KD 对 backbone 的实际影响层级。

**核心问题**：在哪一层，kd_cr 和 baseline 的残差流开始显著分叉？  
- 若从 B0 就开始分叉 → KD 改变了输入嵌入或最早期的表示  
- 若从 B8-B11 才分叉 → KD 主要在网络末端起作用  
- 若全程 CKA ≥ 0.95 仅 x_hat 最终不同 → decode 接口是唯一变量

## 方法

**比较组**：
1. kd_cr vs baseline（最重要：解释 oracle acc 差异来源）
2. kd2 vs baseline（对比）
3. kd_cr vs kd2（解释两者 SC 效果差异来源）

**指标**（每层 i，每个 t 值）：
- **Relative L2**：`||h_A - h_B||_F / ||h_B||_F`（原始幅度差异）
- **Linear CKA**：线性中心核对齐，测量表示子空间相似性（不受线性变换影响），CKA=1 表示完全等价
- **Cosine similarity**：均值中心化后的全局余弦相似度

CKA 计算对 ≤2048 个位置的子采样，固定 seed=42。

**数据**：复用 `results/exp07b_v2_{baseline,kd_cr,kd2}/layer_states_t*.pt`（无需新 forward pass）

## 代码

`experiments/probe_elf/repr_divergence_exp42.py`

## 解读框架

| CKA 模式 | 含义 |
|---------|------|
| 全层 CKA ≈ 1.0 | KD 完全不改变 backbone 表示，只改 decode head |
| 早层 CKA < 0.95，晚层更低 | KD 从第 l 层开始改变 backbone，差异随深度累积 |
| 早层 CKA = 1.0，仅 B10-B11 下降 | KD 在网络最后几层改变 backbone |

## 输出

`results/exp42_repr_divergence/divergence.json`

结构：`{comparison: {t_str: {block_i: {rel_l2, cka, cos_sim}}}}`

---

## 结果

**状态：DONE**

### CKA 逐层结果（t=0.500）

**kd_cr vs baseline**

| block | B00 | B01 | B02 | B03 | B04 | B05 | B06 | B07 | B08 | B09 | B10 | B11 |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| CKA | 0.981 | 0.967 | 0.968 | 0.945 | 0.930 | 0.937 | 0.961 | 0.954 | 0.896 | 0.803 | 0.564 | 0.427 |
| rel_L2 | 0.234 | 0.216 | 0.167 | 0.155 | 0.149 | 0.172 | 0.191 | 0.262 | 0.478 | 0.782 | 0.925 | 0.971 |

**kd_cr vs kd2**

| block | B00 | B01 | B02 | B03 | B04 | B05 | B06 | B07 | B08 | B09 | B10 | B11 |
|-------|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| CKA | 0.985 | 0.977 | 0.977 | 0.972 | 0.974 | 0.970 | 0.975 | 0.976 | 0.958 | 0.916 | 0.882 | 0.892 |
| rel_L2 | 0.139 | 0.114 | 0.094 | 0.087 | 0.082 | 0.084 | 0.094 | 0.107 | 0.146 | 0.124 | 0.133 | 0.500 |

**B11 CKA across t（kd_cr vs baseline）**

| t | 0.10 | 0.20 | 0.30 | 0.50 | 0.70 | 1.00 |
|---|------|------|------|------|------|------|
| CKA | 0.078 | 0.450 | 0.392 | 0.428 | 0.472 | 0.070 |

### 关键发现

1. **KD 的核心影响在 B08-B11**：B00-B07 CKA 保持 0.930-0.981（相对相似），B08-B11 剧烈下降至 0.896/0.803/0.564/0.427。**KD 主要重塑后 4 个 block**（12 层中的后 1/3）。

2. **kd_cr 和 kd2 的 backbone 表示高度相似（CKA ≥ 0.88）**：两种 KD 变体的 backbone 在 t=0.5 时几乎相同（kd_cr vs kd2 CKA 最低 0.882 at B10）。但 B11 rel_L2 = 0.500（明显差异），说明它们的最终 backbone 输出有方向差异，是 SC 行为差异（EXP-36v2）的来源。

3. **B11 CKA 在 t=1.0（最噪声）最低（0.07）**：在最强噪声下，kd_cr backbone 与 baseline 最不同。这与 EXP-39 发现一致——KD 对高噪声（早期 diffusion 步骤）的处理方式有根本差异。

4. **结合 EXP-39**：backbone 是主要影响层（EXP-39 确认），而 CKA 分析定位了影响在 B08-B11。旧叙事"KD reorganizes the decode interface"需修订为"KD reorganizes the late backbone layers (B08-B11)"。

### 对 EXP-15v2（unembed_bias R=2.59）的重新解读

EXP-15v2 的 Frobenius ratio 是**相对**变化（Δ/原始值）。unembed_bias 原始 L2 很小（79.6），任何变化都显得大。但 **EXP-39 的功能测试证明 backbone 才是主要贡献**，EXP-42 定位了 backbone 变化在 B08-B11。两者合并提供了完整的机制故事。
