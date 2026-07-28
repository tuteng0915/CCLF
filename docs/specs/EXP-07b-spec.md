# EXP-07b Spec — Layer-wise Linear Probe on Transformer Blocks

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：承诺现象的空间定位——发生在哪一层？**

EXP-07（独立线性探针）回答"backbone 整体上是否比 native head 更早知道答案"。EXP-07b 进一步追问：**这个"知道"发生在 transformer 的哪一层？**

ELF-B 有 12 个 transformer block（L0 至 L11）。词汇承诺可能是：
- 在早期层（L0-L4）就已经出现：说明这是底层特征，来自输入嵌入的直接投影
- 在中间层（L5-L9）出现：说明需要中等深度的上下文整合
- 在最后几层（L10-L11）才出现：说明承诺是高层抽象的产物，依赖充分的 attention 叠加

**要验证的核心假说**：
- 承诺主要发生在靠后的层（L9-L11），与表示质量的层级提升一致
- L10/L11 是关键层：与参数变化分析（EXP-15）中 L10-L11 的参数改变量最大一致
- KD 训练可能在某些层上改变了表示的组织方式（为 EXP-07c 铺垫）

**方法**：在每个 t 值下，用 forward hook 捕获 12 个 block 的输出（768-dim），分别训练线性探针 → vocab token，得到 12 × |t_grid| 的准确率矩阵。

**与其他实验的关系**：
- EXP-07b → EXP-07c：基于 EXP-07b 发现某些层更有信息量后，用 EXP-07c 分析这些层的跨 checkpoint 可迁移性
- EXP-07b 数据（layer_states_t*.pt）直接被 EXP-07c 和 EXP-07c-full 复用，避免重复运行 backbone

---

## Implementation

**Script:** `experiments/probe_elf/probe_layerwise.py`

**Checkpoints:** baseline / kd-cr / kd2（三个 checkpoint 分别运行）

**Key mechanics:**
1. 用 `model.blocks[i].register_forward_hook()` 捕获每层输出
2. 取 `captured[i][:, prefix:, :]`（跳过 prefix tokens，只取内容 token 位置）
3. 对每层、每个 t 训练 nn.Linear(768, 32100)，用 Adam + 20 epochs
4. 保存结果到 `layer_states_t{t:.3f}.pt`（包含 layer_feats、x_hat、y_tokens、attn_mask）

**Usage:**
```bash
CUDA_VISIBLE_DEVICES=2 python experiments/probe_elf/probe_layerwise.py \
  --checkpoint converted/elf_b-owt-baseline_torch.pt \
  --output_dir results/exp07b_baseline \
  --n_seqs 256 --t_values 0.10,0.20,0.30,0.50,0.70
```

---

## Decision rule

- L0-L4 probe 在 t=0.30 达到 >80%：early commitment，来自输入结构
- L10-L11 probe 在 t=0.30 比 L0-L4 高 >20pp：commitment 是层级的，靠后层更好
- KD vs baseline 在各层的差异图：若 KD 在 L10 改变最大，说明 KD 主要作用于最后两层

---

## 实验结果（Results）

**状态**: COMPLETED（2026-07-18）

**原始数据文件**：
- `results/exp07b_baseline/` — baseline checkpoint 的 layer states
- `results/exp07b_kd_cr/` — kd-cr checkpoint 的 layer states  
- `results/exp07b_kd2/` — kd2 checkpoint 的 layer states
- 每个目录下：`layer_states_t{0.100,0.200,0.300,0.500,0.700}.pt` + `layerwise_probe_accuracies.json`

**关键发现**：

各层线性探针准确率（t=0.30 和 t=0.20）：

**t=0.30（关键承诺区间）**：

| 层 | baseline | kd_cr | kd2 |
|----|---------|-------|-----|
| L0 | 54.7%   | 57.9% | 58.1% |
| L1 | 61.4%   | 62.4% | 62.0% |
| L2 | 63.7%   | 64.3% | 64.1% |
| L3 | 65.5%   | 65.1% | 65.1% |
| L4 | 66.4%   | 66.4% | 66.5% |
| L5 | 68.5%   | 68.3% | 68.5% |
| L6 | 72.7%   | 71.4% | 72.0% |
| L7 | 75.8%   | 73.6% | 74.5% |
| L8 | 78.5%   | 75.7% | 76.6% |
| **L9** | **79.4%** | **76.2%** | **77.5%** |
| L10 | 78.8%  | 74.2% | 77.0% |
| L11 | 76.6%  | 75.6% | 75.6% |
| **x̂_t** | **80.3%** | **77.5%** | **78.4%** |

**t=0.20**：

| 层 | baseline | kd_cr | kd2 |
|----|---------|-------|-----|
| L4 | 40.0%   | 41.7% | 41.2% |
| L8 | 50.1%   | 50.4% | 51.5% |
| L9 | 50.3%   | 50.9% | 49.8% |
| L10 | 50.3%  | 49.8% | 50.8% |
| L11 | 47.3%  | 50.5% | 49.4% |
| **x̂_t** | **53.9%** | **52.9%** | **53.5%** |

**关键现象**：
- baseline: L9 > L11 > L10（信息在 L9 峰值，最后两层有轻微"遗忘"）；x̂_t（GELU 投影后 512-dim）是最优读出
- kd_cr / kd2: 层间差距与 baseline 相似，但 x̂_t 仍然最优
- 三个 checkpoint 的层级准确率几乎相同（对应层差 < 2pp），说明 backbone 中层的 token 编码几何在 KD 训练后基本不变
- L10 的"特殊性"（EXP-07c 跨 checkpoint 迁移差）体现在迁移率上，不在本 checkpoint 内准确率上

**L10 特殊现象**（EXP-07c-full 发现）：L10 的线性探针在本 checkpoint 内准确率正常，但跨 checkpoint 的探针迁移性极差（见 EXP-07c-full spec）。说明 L10 在不同训练目标下学到了不同的编码几何。

**数据复用**：layer_states 文件被以下实验直接复用：
- EXP-07c（cross-checkpoint probe transfer）
- EXP-07c-full（full per-layer 3×3 transfer matrix）
- EXP-12（residual rank analysis，使用 layer_feats[-1]）
- EXP-16（per-position commitment timing）
