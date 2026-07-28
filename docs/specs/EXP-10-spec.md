# EXP-10 Spec — Per-Checkpoint Commitment Timing Comparison (baseline vs kd-cr vs kd2)

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：跨 checkpoint 的 Protocol A 承诺时间系统性对比，量化 KD 训练的动力学效果。**

EXP-16 已经对 kd-cr checkpoint 做了 per-position commitment timing 分析（使用 decode_path(L11_hidden) 指标）。EXP-07b/c 已经从层级 probe 和跨 checkpoint 迁移性角度看了几何差异。

**但缺少一个直接对比**：对 baseline、kd2、kd-cr 三个 checkpoint 做相同的 per-position 承诺时间分析，比较：
1. 平均承诺时间分布（CDF 曲线）
2. 早期承诺比例（t < 0.30 时的 G(t)）
3. 位置级别的承诺时间变化：哪些"类型"的位置（内容词、功能词、数字、标点）改变最多

**要验证的核心假说**：
- kd-cr 训练使 **平均承诺时间提前**（更多位置在低 t 时就能解码正确）
- 提前效果在**内容词**（nouns, verbs）上最显著（这些词更难预测，KD 训练特别帮助）
- kd2（两阶段 KD）的效果在 kd-cr 和 baseline 之间，或者类似 kd-cr

**与其他实验的关系**：
- EXP-16 提供了 kd-cr 的完整 t* 分布，本实验用相同方法跑 baseline 和 kd2
- EXP-20 提供了 token 频率与 t* 的关系（kd-cr），本实验验证该关系是否在所有 checkpoint 一致
- EXP-07c 看的是层级几何的跨 checkpoint 差异，本实验看的是宏观承诺时间差异

---

## Implementation Plan

### 脚本

**复用**：`experiments/probe_elf/probe_null_vs_oracle.py`（同 EXP-04b，但用于比较不同 checkpoint）

或新建：`experiments/probe_elf/compare_commitment_timing.py`

```bash
# baseline
CUDA_VISIBLE_DEVICES=0 /home/wjzhang/miniforge3/envs/elf/bin/python \
    experiments/probe_elf/probe_null_vs_oracle.py \
    --checkpoint converted/elf_b-owt-baseline_torch.pt \
    --n_seqs 256 --n_t_steps 20 \
    --output_dir results/exp10_baseline

# kd2
CUDA_VISIBLE_DEVICES=1 /home/wjzhang/miniforge3/envs/elf/bin/python \
    experiments/probe_elf/probe_null_vs_oracle.py \
    --checkpoint converted/elf_b-owt-kd2_torch.pt \
    --n_seqs 256 --n_t_steps 20 \
    --output_dir results/exp10_kd2
```

注：kd-cr 已有数据（来自 EXP-04b 和 EXP-16）

### 分析脚本

```python
# 比较三个 checkpoint 的 G(t) 曲线（已有 kd-cr 和 baseline 的部分数据）
for ckpt in ["baseline", "kd2", "kd_cr"]:
    G_t = load_G_t(f"results/exp10_{ckpt}/G_oracle.json")
    plot_G_t(G_t, label=ckpt)

# CDF of commitment times t*
# （需要 per-sequence per-position 数据，compute-intensive）
```

---

## 实验结果（Results）

**状态**: COMPLETED（2026-07-18）— 见下方完整三方对比表。

**已有数据**：

| checkpoint | t=0.20 G(t) | t=0.30 G(t) | t=0.50 G(t) | 来源 |
|-----------|-------------|-------------|-------------|------|
| baseline | 36.2% | 77.5% | 79.9% | EXP-04b |
| kd-cr | 58.5% | 89.3% | 99.5% | EXP-16 |

**状态**: COMPLETED（2026-07-18）

**数据来源**：`results/exp04b_baseline/`, `results/exp04b_kd_cr/`, `results/exp04b_kd2/`（均来自 probe_null_vs_oracle.py，decode_path(L11_hidden) 指标）

### 完整三方对比表

| t | baseline G(t) | kd2 G(t) | kd-cr G(t) |
|---|---|---|---|
| 0.05 | 0.3% | 5.9% | — |
| 0.10 | 1.0% | 12.6% | 12.8% |
| 0.20 | 9.9% | 60.7% | 61.0% |
| 0.30 | 53.9% | 89.8% | 90.7% |
| 0.50 | 79.9% | 99.3% | ~99.5% |
| 0.70 | 85.4% | 99.8% | 99.9% |
| 0.80 | 85.7% (plateau) | — | — |
| 0.90 | 86.1% (plateau) | — | — |

### 关键发现

**1. kd2 ≈ kd-cr**（最令人惊讶的发现）：
- 两阶段 KD（kd2）与单阶段 KD（kd-cr）在 G(t) 上几乎一致
- t=0.20: 60.7% vs 61.0%；t=0.30: 89.8% vs 90.7%
- 说明 KD 的主要效果在第一阶段就已获得，第二阶段贡献有限

**2. baseline 有 G(t) 上限平台（≈85.6%）**：
- baseline 在 t=0.50-0.90 的 G(t) **几乎不变**（79.9% → 85.4% → 85.7% → 86.1%）
- 即使 z_t 极其干净（85% signal），baseline 仍有约 14% 的位置无法正确解码
- kd-cr/kd2 不存在这个平台，在 t=0.35 就已达到 95%+
- **解读**：baseline 有一批"结构性难解码"的位置（约 14%），KD 训练修复了这些位置的 decode path 对齐

**3. 悬崖位置的移动**：
- baseline: t=0.20→0.30 最陡（9.9%→53.9%，+44pp）
- kd-cr/kd2: t=0.10→0.20 最陡（~12%→61%，+48pp）
- KD 训练不仅使悬崖更陡，而且将其提前约 1 个时间步（Δt ≈ 0.10）

### 论文意义

- kd2 ≈ kd-cr 说明 KD 效果可以用单阶段 kd-cr 完整代表，kd2 在 probe 指标上没有额外优势
- baseline 的 14% "结构性上限"是论文应该报告的发现（KD 训练的质变效果，而非单纯的量变）
- "悬崖提前 Δt ≈ 0.10"是论文的可定量声明

**决策规则**：
- kd-cr > kd2 ≈ kd-cr（确认）→ 可以使用 kd-cr 作为论文的代表 KD checkpoint
- baseline 的平台效应 → 加入论文 §4.x 作为"KD 消除 decode path 结构性错误"的证据

---

## ⚠️ G(t) Provenance 说明（2026-07-22 更新）

EXP-10 中出现两组不一致的 baseline G(t=0.20) 数字：

| 来源 | baseline | kd_cr | 说明 |
|------|---------|-------|------|
| EXP-04b（EXP-10 表1） | **36.2%** | — | fixed ε OR 特定实现 |
| EXP-10 表2（JAX/另一实现） | **9.9%** | 61.0% | 非固定 ε（每 t 独立采样） |
| **EXP-16v2（权威）** | **36.1%** | **58.8%** | fixed ε seed=42，correct decode path，T_first |

**差异来源**：
- **非固定 ε（per-t independent sampling）**：每个 t 用不同的 ε 采样 z_t = t·x_clean + (1-t)·ε。G(t=0.20) 低（baseline ≈9.9%）因为每次随机 ε 都可能让位置更难。
- **固定 ε（seed=42）**：所有 t 共享同一 ε。G(t=0.20) 高（baseline≈36.1%）因为某些位置在这个特定 ε 下恰好容易读出。

**关键新发现（EXP-16v2 揭示）**：
- baseline G(t=0.20) 在非固定 ε 下仅 9.9%，但固定 ε 下 36.1%（差 26pp）
- kd_cr G(t=0.20) 在非固定 ε 下 ~61%，固定 ε 下 **58.8%**（差仅 2.2pp）

**KD 使 decode readout 对噪声采样更鲁棒**：baseline 的 readout 严重依赖噪声实例（某些 ε 好，某些差），而 KD 的 readout 在任何 ε 下都稳定。这是比"更早承诺"更深层的结论。

**正式使用时的标注规则**：
- 引用非固定 ε G(t)（EXP-10 表2）时：标注"averaged over independent ε draws"
- 引用固定 ε G(t)（EXP-16v2）时：标注"fixed ε (seed=42)"
- 不在同一图表中混用两种来源

**权威数字（fixed ε，EXP-16v2，T_first）**：baseline 36.1%，kd_cr **58.8%**，kd2 **57.6%**（at t=0.20）

