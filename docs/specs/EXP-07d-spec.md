# EXP-07d Spec — x̂_t Cross-Checkpoint Transfer Analysis

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：量化 final_layer 投影的"重对齐"作用。**

EXP-07c-full 发现，ELF-B transformer 的 L11 隐状态（768-dim）在不同 checkpoint 间的探针迁移率很低（t=0.20 时近乎为零）。然而，L11 通过 `final_layer` 投影到 512-dim，得到 x̂_t（去噪预测，也是下一步的 self-conditioning 信号）。

**核心问题**：这个 768→512 的投影是否部分"重对齐"了不同 checkpoint 的表示？即：
- L11 隐状态（768-dim）的跨 checkpoint 迁移率很低（7.8% at t=0.20）
- x̂_t（512-dim）的跨 checkpoint 迁移率更高（52.5% at t=0.20）

如果是，说明 `final_layer` 不只是维度缩减，而是在功能上承担了"几何对齐"的作用：将各自 checkpoint 特异的 L11 表示投影回一个更通用的空间。

**要验证的核心假说**：
- x̂_t 探针迁移率 >> L11 hidden state 探针迁移率：final_layer 是重对齐机制
- x̂_t 探针迁移的对称性（baseline→kd-cr vs kd-cr→baseline）：比 L11 更对称

**重要性**：
- 如果 x̂_t 重对齐成立：dec_sc 的 self-conditioning 信号在跨 checkpoint 时比 backbone 内部更具可迁移性
- 这解释了为什么 dec_sc 机制（用 x̂_t 作为 self-cond）在推理时能稳健工作：x̂_t 在不同 checkpoint 间共享更通用的几何空间

**与其他实验的关系**：
- EXP-07c 发现 L11 跨 checkpoint 迁移率低 → EXP-07d 解释"但 x̂_t 层面的迁移性更好"
- EXP-07d 与 EXP-12（decode path G(t)）共同解释 ELF 的解码机制：x̂_t 在 512-dim 空间对齐，而 decode head 在 512-dim 空间运作

---

## Implementation

**Script:** `/tmp/exp07d_xhat_transfer.py`（或 `experiments/probe_elf/probe_xhat_transfer.py`）

对 x̂_t（layer_states_t*.pt 中的 `x_hat` 字段，512-dim）跨 3 个 checkpoint 训练线性探针并互相评估。

**方法**：与 EXP-07c-full 相同，但使用 x_hat（512-dim）而非 layer_feats[-1]（768-dim）。

**Usage:**
```bash
CUDA_VISIBLE_DEVICES=6 python /tmp/exp07d_xhat_transfer.py \
  --states_dirs results/exp07b_baseline,results/exp07b_kd_cr,results/exp07b_kd2 \
  --output_dir results/exp07d
```

---

## 实验结果（Results）

**状态**: COMPLETED（2026-07-18）

**原始数据文件**：`results/exp07d/xhat_transfer.json`

**完整 3×3 x̂_t 迁移矩阵**：

| t | baseline→b | baseline→kd_cr | baseline→kd2 | kd_cr→base | kd_cr→kd_cr | kd_cr→kd2 | kd2→base | kd2→kd_cr | kd2→kd2 |
|---|-----------|----------------|--------------|-----------|-------------|----------|---------|----------|---------|
| 0.20 | 0.833 | **0.525** | 0.462 | **0.578** | 0.837 | 0.560 | 0.578 | 0.613 | 0.840 |
| 0.30 | 0.978 | **0.793** | 0.725 | **0.886** | 0.976 | 0.897 | 0.884 | 0.900 | 0.976 |
| 0.50 | 0.994 | **0.935** | 0.889 | **0.978** | 0.990 | 0.981 | 0.975 | 0.963 | 0.992 |
| 0.70 | 0.993 | **0.949** | 0.925 | **0.985** | 0.991 | 0.984 | 0.984 | 0.989 | 0.993 |

**对比 L11 hidden（768-dim）迁移率**（EXP-07c-full 结果）：
- t=0.20：baseline→kd_cr = 12.0%（L11）vs **52.5%（x̂_t）**
- t=0.20：kd_cr→baseline = 5.2%（L11）vs **57.8%（x̂_t）**

**关键发现**：

1. **x̂_t 迁移率大幅高于 L11**：t=0.20 时，L11 跨 checkpoint 迁移 ≈ 8%，x̂_t 跨 checkpoint 迁移 ≈ 52%。final_layer 投影确实是重对齐机制。

2. **x̂_t 迁移近似对称**：t=0.20 时，baseline→kd_cr = 52.5%，kd_cr→baseline = 57.8%，差异不大（而 L11 则严重不对称）。

3. **高 t 时 x̂_t 趋于完全对齐**：t=0.70 时，所有跨 checkpoint 迁移均 > 92%，几近完美。

4. **kd2 处于 baseline 和 kd_cr 之间**：baseline→kd2 = 46.2%（低于→kd_cr = 52.5%），与 kd2 是另一种 KD 变体一致。

**论文启示**：
- final_layer（768→512）不只是维度缩减，而是一个**跨 checkpoint 几何标准化**的隐式机制
- 这解释了为什么 self-conditioning（用 x̂_t）在推理时稳健：x̂_t 比 backbone 内部表示更通用
- t=0.20 时 x̂_t 已有 52% 的跨 checkpoint 迁移，说明在"承诺悬崖"区域 x̂_t 已经有实质性信号
