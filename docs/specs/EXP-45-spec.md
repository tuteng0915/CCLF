# EXP-45: SC Activation Patch（因果干预生成）

## 目标

直接因果测试：用 kd_cr 的 x̂_t 替换 kd2 生成过程中的 x̂_t（SC conditioning input），观察 SC interaction 是否翻转。这把"SC compatible format"从静态几何假说变成可操作的生成干预。

## 核心问题

EXP-43/44 仍在 oracle 层面。真正的 SC compatibility 是动态的：当 x̂_{t,prev} 作为下一步的 conditioning input 时，它是否让 velocity 指向正确方向？

EXP-45 直接在生成 rollout 中干预这个量。

## 方法

### SC 机制回顾（ELF）

在 ELF 生成中，SC（self-conditioning）步骤：
1. 第一次 forward：`x̂_t = model(z_t, t, sc=zeros)`（无 conditioning）
2. 第二次 forward：`x̂_t_new = model(z_t, t, sc=x̂_t)`（用第一次结果作 conditioning）

SC conditioning 路径：
```
x̂_t [512] → self_cond_proj [1024→512] → ... → backbone conditioning tokens
```

（`self_cond_proj.weight (512, 1024)` 可能 concat 了 x̂_t 和某个 cfg 信号后投影）

### Activation Patch 设计

固定 seed，在 kd2 的完整 rollout 中，在每一步的 SC conditioning 位置将 x̂_t 替换为：

```
x̂_t_patched(λ) = (1-λ)·x̂_t_kd2 + λ·x̂_t_kd_cr
```

其中 x̂_t_kd_cr 从 kd_cr 的同步 parallel rollout 获得（同 z_t、同 t）。

**关键 arm**：

| arm | backbone | SC conditioning input |
|-----|:---:|:---:|
| kd2 native | kd2 | kd2 x̂_t |
| kd2 + patch_xhat_cr (λ=1) | kd2 | kd_cr x̂_t |
| kd2 + patch_xhat_mix (λ=0.5) | kd2 | mixed x̂_t |
| kd_cr native | kd_cr | kd_cr x̂_t |

分别 patch 以下节点（分解哪个环节最关键）：
- x̂_t 本身（final_layer 输出，512-dim）
- self_cond_proj 之后的 512-dim 投影
- backbone conditioning tokens（4 tokens × 768-dim）

### 判据

| 结果 | 含义 |
|------|------|
| patch x̂_t(λ=1) 让 kd2 SC interaction 翻转 | x̂_t 的方向直接决定 SC utility；"format"假说成立 |
| patch x̂_t 后 I 在 λ 下平滑变化 | SC compatibility 是 x̂_t 方向的连续属性 |
| patch conditioning tokens 才能翻转 | x̂_t 需要通过 self_cond_proj 变换后才能影响 SC |
| 无论 patch 什么都无法翻转 | SC 差异来自 backbone 本身（EXP-44 候选） |

### 实现细节

需要修改 `src/utils/generation_utils.py`，在 SC conditioning 处加 hook：

```python
if sc_patch_xhat is not None:
    x_hat_prev = (1 - sc_patch_lambda) * x_hat_prev + sc_patch_lambda * sc_patch_xhat
```

评估指标：
- PPL on n=64 sequences（seed=42，32 steps）
- SC interaction I = PPL(SC) - PPL(none)（per checkpoint arm）
- 定性：退化率（非 ASCII、重复词）

## 代码

`experiments/probe_elf/sc_activation_patch_exp45.py`（待实现）

需修改 `generation_utils.py` 支持 activation injection hook。

**成本**：中等（每个 arm 约 5-10 分钟 GPU；全部 arm 约 1 小时）

## 输出

`results/exp45_sc_activation_patch/patch_results.json`

---

## 结果

**状态：PLANNED**
