# EXP-07c / EXP-07c-full Spec — Cross-Checkpoint Probe Transfer

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：分离 KD 训练对"表示容量"vs"表示几何"的影响。**

EXP-07b 发现，用各自 checkpoint 的数据训练线性探针，kd-cr 和 baseline 在 L11 的探针准确率相近（kd-cr 稍好但差距不大）。这表明两个 checkpoint 的**表示容量**（backbone 能不能编码 token 信息）相似。

但 KD 训练是否改变了编码 token 的**几何方向**（在 768-dim 空间中，token 信息被编码在哪个方向上）？这是不同的问题。

**关键思想**："探针迁移实验"：
- 用 checkpoint A 的数据训练线性探针 → 在 checkpoint B 的数据上测试
- 如果迁移准确率高（≈ checkpoint B 自身的准确率）：A 和 B 使用相同的几何方向
- 如果迁移准确率低（≈ random）：A 和 B 在不同的几何方向上编码 token 信息

**要验证的核心假说**：
- **容量假设**：KD 只增强了信号强度，不改变几何方向 → 探针应该可以跨 checkpoint 迁移
- **几何假设**：KD 改变了编码方向 → 探针在低 t（高噪声）下不能跨 checkpoint 迁移

**重要性**：
- 如果是几何假设（更可能基于 EXP-15 的参数分析）：则 kd-cr 的 decode head 是为 kd-cr 特定几何设计的，不能简单地移植到 baseline 上
- 如果是容量假设：则不同 checkpoint 的 backbone 是可互换的，只有 head 不同

**EXP-07c-full** 是 EXP-07c 的升级版：
- EXP-07c：只训练 baseline 探针，在 3 个 checkpoint 上测试（1×3 矩阵）
- EXP-07c-full：训练所有 3 个 checkpoint 的探针，互相测试（3×3 矩阵），且对 12 个层分别做

**与其他实验的关系**：
- EXP-07b → EXP-07c：EXP-07b 的 layer_states 被 EXP-07c 直接复用
- EXP-07d：在 x_hat（512-dim）空间而非 L11（768-dim）重复同样的迁移分析

---

## EXP-07c Implementation

**Script:** `experiments/probe_elf/probe_cross_checkpoint.py`

训练 baseline 探针 → 测试在 {baseline, kd_cr, kd2} 数据上的准确率

**EXP-07c-full Implementation**

**Script:** `experiments/probe_elf/probe_cross_checkpoint_full.py`

对所有 12 层、4 个 t 值、3×3 checkpoint 组合训练并评估探针

**Usage:**
```bash
CUDA_VISIBLE_DEVICES=5 python experiments/probe_elf/probe_cross_checkpoint_full.py \
  --states_dirs results/exp07b_baseline,results/exp07b_kd_cr,results/exp07b_kd2 \
  --checkpoint_names baseline,kd_cr,kd2 \
  --output_dir results/exp07c_full \
  --t_values 0.20,0.30,0.50,0.70 \
  --n_epochs 8
```

---

## 实验结果（Results）

**状态**: COMPLETED（2026-07-18）

**原始数据文件**：
- `results/exp07c/cross_checkpoint_transfer.json`（EXP-07c，仅最终层，4 个 t 值）
- `results/exp07c_full/cross_checkpoint_full.json`（EXP-07c-full，12 层 × 4 t × 3×3 矩阵）

**关键发现**：

### 1. t 依赖的几何发散（EXP-07c）

在 L11（最终层）：
- t=0.20：baseline→kd_cr = 7.8%，kd_cr→baseline = 5.2%（极低，几何已完全发散）
- t=0.50：baseline→kd_cr = 92.5%，kd_cr→baseline = 62.3%（大幅回升）
- t=0.70：baseline→kd_cr = 94.1%，kd_cr→baseline = 29.4%（出现方向性不对称！）

**结论**：KD 训练在高噪声（低 t）区间对 L11 的几何影响最大；在低噪声（高 t）区间几何趋于一致（因为 token 信号足够强，方向不重要）。

### 2. L10 "发散震中"（EXP-07c-full 新发现）

在 L10：
- t=0.20：baseline→kd_cr = 0.70%（接近随机！），kd_cr→kd2 = 51.7%
- t=0.30：baseline→kd_cr = 4.54%，kd_cr→kd2 = 80.1%
- t=0.70：baseline→kd_cr = 22.4%，kd_cr→kd2 = 95.8%

L10 是 baseline 与 KD 模型之间几何发散最严重的层，在任何 t 值下 baseline 探针均无法迁移到 KD 模型（而 L11 在高 t 时仍可迁移）。

### 3. kd-cr → baseline 的方向性不对称（L11, t=0.70）

- baseline→kd_cr = 94.1%（baseline 探针能迁移到 kd-cr）
- kd_cr→baseline = 29.4%（kd_cr 探针几乎不能迁移到 baseline）

解释：baseline L11 的编码方向是 kd-cr L11 编码方向的"子集"——baseline 探针学到的方向在 kd-cr 的 L11 空间里仍然有效（kd-cr L11 有更多信息），但 kd_cr 探针学到的 kd_cr 特有方向在 baseline 中不存在。

### 4. kd2 vs kd_cr 的相互可迁移性高

kd2 和 kd_cr 之间的探针迁移率在大多数层和 t 值下 > 80%，说明两个 KD 训练变体共享相似的几何空间，与 baseline 的发散方向一致。

**论文启示**：
- 支持"几何假设"而非"容量假设"：KD 改变了 L10 的编码方向
- KD 的 decode head 是为 KD 特定几何设计的，这解释了为什么 dec_sc 对 baseline 模型效果有限
- 高 t 时各 checkpoint 趋于一致：承诺的几何汇聚只在低噪声区间完成
