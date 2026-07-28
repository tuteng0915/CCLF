# EXP-46: SC 响应的 Jacobian 分析

## 目标

把"SC compatibility"从一个关于静态表示的问题，转化为动力学问题：测量 x̂_t 的差异（kd_cr vs kd2）如何通过 SC conditioning pathway 影响 velocity 方向。

## 核心问题

EXP-45 的生成干预回答的是"patch 之后 PPL 是否改变"（全局行为）。EXP-46 回答的是更基础的问题：

> 当 δs_t = x̂_t_kd_cr - x̂_t_kd2 通过 SC conditioning 进入模型时，它产生的 velocity correction J_SC · δs_t 是否指向正确的 recovery direction？

这是 SC compatibility 的直接动力学刻画，不依赖于生成 rollout 的统计。

## 方法

### SC 的 Jacobian

在 ELF 生成中，velocity 模型为：
```
v_θ(z_t, s_t, t)   其中 s_t = x̂_{t,prev}（SC conditioning input）
```

定义 SC Jacobian：
```
J_SC = ∂v_θ(z_t, s_t, t) / ∂s_t    ∈ ℝ^{D_z × D_s}
```

对于两个 checkpoint 的 x̂_t 差异：
```
δs_t = x̂_t_kd_cr - x̂_t_kd2    ∈ ℝ^{D_s=512}
```

计算 velocity 响应：
```
J_SC · δs_t ∈ ℝ^{D_z}    （κ为第二次 forward call 中的 velocity correction）
```

判断这个修正是否朝正确的 recovery direction：
```
alignment = cos(J_SC · δs_t,  x_0 - z_t)
```

其中 `x_0 - z_t` 是"从当前噪声到 clean signal 的目标方向"。

### 高效计算（JVP）

不需要计算完整 Jacobian（O(D_z × D_s) 内存）。使用 Jacobian-vector product（JVP）：

```python
from torch.func import jvp

def model_sc(s):
    return v_theta(z_t, s, t)

_, jvp_val = jvp(model_sc, (s_kd2,), (delta_s,))
# jvp_val = J_SC · delta_s
```

这只需要一次 forward + backward pass，O(D_z) 内存。

### 关键比较

对同一 (z_t, t) 序列（来自 exp07b_v2 数据或 fresh sample）：

1. 用 kd_cr 计算 s_kd_cr = x̂_t_kd_cr（via recon path）
2. 用 kd2 计算 s_kd2 = x̂_t_kd2
3. δs = s_kd_cr - s_kd2
4. 分别对 kd_cr 模型和 kd2 模型计算 J_SC · δs
5. 计算 cos(J_SC · δs, x_0 - z_t)

**预测**：
- kd_cr 模型：J_SC · δs_t 应与 x_0 - z_t 正相关（SC conditioning of kd_cr 在帮助 recovery）
- kd2 模型：J_SC · δs_t 与 x_0 - z_t 相关性低甚至负相关（SC conditioning of kd2 在阻碍 recovery 或无贡献）

### 实现路径

需要运行 ELF-torch 的 forward pass 并对 SC conditioning input 求导。
关键：找到 SC conditioning input 进入模型的位置（self_cond_proj → conditioning tokens）。

从 checkpoint 结构推断 SC 路径：
```
x̂_t [512] 
→ self_cond_proj [1024→512]?（需确认输入维度） 
→ self_cond_cfg_embedder [256→768 → 768→768]
→ self_cond_cfg_tokens [4, 768]（4个 conditioning tokens prepended 或 added to sequence）
→ backbone attention sees these tokens
→ velocity v_θ
```

需要在 ELF-torch 源码中 hook 到 x̂_t 进入 backbone 的位置，并在那里做 JVP。

### 与其他实验的关系

| 实验 | 测量 | 属性 |
|------|------|------|
| EXP-42 | rel_L2(h_11) | 静态 representation 差异 |
| EXP-43 | L_dec, L_rec, gradient conflict | 单步路径几何 |
| EXP-45 | SC interaction I（生成） | 全局 rollout 行为 |
| **EXP-46** | **J_SC · δs vs x_0 - z_t** | **单步动力学响应** |

EXP-46 最直接，因为它问的是"δs_t 通过 SC 机制对 velocity 的贡献是否正确"，而不依赖多步 rollout 的累积效应。

## 代码

`experiments/probe_elf/sc_jacobian_exp46.py`（待实现）

实现前需要：
1. 理解 ELF-torch 中 SC conditioning 的具体代码路径
2. 确认 self_cond_proj 的输入格式（1024-dim 可能是 x̂_t + cfg concatenation）
3. 验证 JVP 在 ELF forward pass 中可以正确传播

**成本**：中等（JVP 约每 batch 几秒；需要 ELF-torch 源码 hook）

## 输出

`results/exp46_sc_jacobian/jacobian.json`

---

## 结果

**状态：PLANNED**
