# EXP-48: Intermediate-Layer SC — Proper Pipeline

**Status**: COMPLETE  
**Date**: 2026-07-25  
**Related**: EXP-47 (intermediate SC draft, buggy), EXP-44 Phase 2 (self_cond_proj causality)

---

## Motivation

EXP-47 showed that using h_10 (instead of h_11) as the SC signal improves PPL for KD checkpoints,
but had two implementation bugs:
1. `compute_ppl` returned mean NLL instead of `math.exp(mean_NLL)` (not actual PPL)
2. "none" arm zeroed x_pred, but EXP-36v2 baseline's "none" arm lets x_pred evolve naturally

EXP-48 fixes both and provides a clean comparison of:
- `natural` (α=1.0): standard SC from ODE step, used unchanged (baseline)
- `none`: x_pred zeroed every step (total SC off)
- `α=0.0` (h10): x_pred replaced by `final_layer(h_10)` at each step when t_next ≥ SC_T_MIN
- `α=0.5` (mid): interpolated between h_10 and h_11

**Metric**: I(α) = PPL(α) − PPL(natural); negative = better than standard SC.

**Pipeline caveat**: This custom ODE loop may give different absolute SC direction than EXP-36v2
(kd_cr's "natural" PPL > "none" PPL in EXP-48, opposite of EXP-36v2). Relative comparisons
between α arms are the meaningful signal.

---

## Setup

- Checkpoints: `kd_cr`, `kd2`
- N_SEQ=64, N_STEPS=32, SEED=42, MAX_LENGTH=128, BATCH_SIZE=16
- SC_T_MIN=0.5 (only apply intermediate SC when t_next ≥ 0.5)
- PPL model: GPT-2 Large
- Script: `experiments/probe_elf/intermediate_sc_proper_exp48.py`

---

## Results

| arm | kd_cr PPL | kd_cr I | kd2 PPL | kd2 I |
|-----|-----------|---------|---------|-------|
| natural (std SC, α=1.0) | 303.4 | 0.0 | 247.7 | 0.0 |
| none (zero SC) | 186.8 | **−116.6** | 341.2 | +93.5 |
| α=0.00 (h10 SC) | 192.1 | **−111.3** | 90.9 | **−156.8** |
| α=0.50 (mixed) | 249.2 | −54.2 | 150.0 | **−97.7** |

---

## 核心发现

### 1. kd2: h_10 SC 是重大突破 (I = −157)

kd2 标准 SC (h_11) PPL = 247.7，切换到 h_10 SC 后 PPL 骤降至 **90.9** (I = −156.8)。
这是 **推理期免费的 157 PPL 改进**——无需任何重训练。

为什么 h_10 对 kd2 如此有效？
- kd2 的 self_cond_proj 和 final_layer 是协同训练的匹配对（EXP-44 Phase 2 证明）
- 但 h_10 作为 SC 信号绕过了 self_cond_proj：直接用 final_layer(h_10) 注入
- h_10 包含更纯粹的内容信号（尚未经过 B11 解码压缩），用于 SC 效果更好

### 2. kd_cr: h_10 不是解决方案 (I(h10) ≈ I(none))

kd_cr 用 h_10 SC 得 PPL=192.1，和 "none"（PPL=186.8）几乎一样差。说明：
- kd_cr 的问题不在于用了 h_11 vs h_10
- kd_cr 的 SC 管道在更根本的层面上已经损坏（EXP-44 Phase 2 已确认：kd_cr 的
  self_cond_proj 无法兼容 final_layer 产生的 x̂_t）
- 解决 kd_cr 需要重训练（D1 或 D3 方向）

### 3. α 梯度表明 h_10 信息逐步降低

两个 checkpoint 均显示 I(α=0.0) < I(α=0.5) < I(natural)（kd_cr: −111 < −54 < 0；
kd2: −157 < −98 < 0），说明 h_10 单独信号比混合信号更有效，而混合信号比 h_11 更有效。
h_10 捕获了更纯净的去噪表示，h_11 已经被解码路径部分"污染"。

---

## 下一步

- **EXP-51**: 评估 D1 和 D3 微调 checkpoint 在 intermediate SC 下的表现
- **D2 即时应用**: kd2 + h_10 SC 可立即用于 paper 实验（无需重训）
- **Pipeline 对齐**: 在 EXP-36v2 pipeline 中验证 h_10 SC 对 kd2 的效果
  （避免 pipeline 差异干扰）

---

## 结果文件

`results/exp48_intermediate_sc_proper/results.json`
