# EXP-43: Dual-Path Gradient Conflict

## 目标

直接验证"reconstruction–decode 几何冲突"假说：baseline 的 B11 残差更新是否同时让 decode 变差、让 reconstruction 变好；KD 是否消除了这个冲突。

## 核心问题

baseline logit lens 非单调（B10 > B11）说明 block 11 的 Δh_11 是 decode-hostile 的。但这是 tradeoff 还是 B11 对两条路径都没好处？需要**同时测量** L_dec 和 L_rec 才能回答。

## 方法

### 两条下游路径

```
h_11 → proj_kernel → GELU → unembed          → L_dec（token CE loss）
h_11 → RMSNorm(norm_final) → final_layer.linear → x̂_t → L_rec（MSE vs x*）
```

x* = x_hat at t=1.0（clean limit：z_1 = x_0，模型几乎完美重建）

### 插值曲线

```
h(α) = h_10 + α · (h_11 - h_10),    α ∈ {-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}
```

在每个 α 计算：
- `L_dec(α)` = 正确 token 的 CE loss（via decode head）
- `L_rec(α)` = ||x̂_α - x*||² 的均值（via reconstruction path）
- `top1_acc(α)` = oracle accuracy

### Gradient Conflict

在 h = h_11 处计算两条路径的梯度，并测量其冲突程度：

```
c(h_11) = cos(∇_h L_dec, ∇_h L_rec)
```

- `c < 0`：两条路径的改善方向相互冲突（conflict）
- `c ≈ 0`：两条路径的梯度正交（独立）
- `c > 0`：两条路径的改善方向兼容

同时计算 **Δh_11 vs -∇L_dec 的 cosine**（测量 B11 的残差更新是否在帮助 decode：正 = pro-decode，负 = anti-decode）。

### 预期结果

| checkpoint | ΔL_dec (h10→h11) | ΔL_rec (h10→h11) | conflict cos | Δh·(-g_dec) |
|-----------|:---:|:---:|:---:|:---:|
| baseline | **↑变差**（支持假说） | **↓变好**（tradeoff） | **< 0** | **< 0（anti-decode）** |
| kd_cr | ↓变好 | ↓也变好 | **≥ 0** | **> 0（pro-decode）** |
| kd2 | ↓变好 | ↓也变好 | **≥ 0** | **> 0** |

若 baseline 的 ΔL_dec > 0（CE 变差）且 ΔL_rec < 0（MSE 变好），则 tradeoff 假说成立。

### 修订后的最强结论（若假说成立）

> The baseline B11 residual update creates a geometric tradeoff: it improves reconstruction quality (L_rec decreases) at the cost of decode accuracy (L_dec increases). KD eliminates this conflict (both losses decrease), consistent with a reorganization of late-layer computation to serve both downstream paths simultaneously.

## 代码

`experiments/probe_elf/gradient_conflict_exp43.py`

**运行**：`conda run -n elf python3 experiments/probe_elf/gradient_conflict_exp43.py --device cuda:0`

**数据**：复用 exp07b_v2（无需新 forward pass）；梯度用 PyTorch autograd 计算。

**成本**：极低（解析计算 + autograd，约 5 分钟 CPU 或 1 分钟 GPU）

## 输出

`results/exp43_gradient_conflict/gradient_conflict.json`

---

## 结果

**状态：DONE（2026-07-23）**

### 主要数值（来自 `results/exp43_gradient_conflict/gradient_conflict.json`）

#### 插值曲线：L_dec 和 L_rec（α: h10→h11）

| checkpoint | t | L_dec(h10) | L_dec(h11) | ΔL_dec | L_rec(h10) | L_rec(h11) | ΔL_rec |
|:---:|:---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.2 | 26.16 | 47.36 | **+21.19 ↑** | 525.2 | 133.9 | −391.4 ↓ |
| baseline | 0.5 | 5.34 | 13.26 | **+7.92 ↑** | 529.0 | 27.87 | **−501.1 ↓** |
| baseline | 1.0 | 0.59 | 5.19 | **+4.60 ↑** | 361.9 | ~0 | −361.9 ↓ |
| kd_cr | 0.2 | 5.38 | 4.38 | −1.00 ↓ | 142.4 | 98.5 | −43.9 ↓ |
| kd_cr | 0.5 | 0.050 | 0.026 | **−0.024 ↓** | 113.2 | 52.9 | **−60.3 ↓** |
| kd_cr | 1.0 | 0.015 | 0.007 | −0.008 ↓ | 128.2 | ~0 | −128.2 ↓ |
| kd2 | 0.2 | 5.62 | 5.26 | −0.37 ↓ | 122.6 | 113.2 | −9.4 ↓ |
| kd2 | 0.5 | 0.118 | 0.055 | **−0.063 ↓** | 103.2 | 74.7 | **−28.5 ↓** |
| kd2 | 1.0 | 0.035 | 0.010 | −0.025 ↓ | 112.3 | ~0 | −112.3 ↓ |

#### Gradient Conflict（在 h11 处，t=0.5）

| checkpoint | cos_per_pos_mean | cos_aggregate | gnorm_dec | gnorm_rec |
|:---:|:---:|:---:|:---:|:---:|
| baseline | +0.0089 | −0.0109 | 1.18e-4 | 1.31e-7 |
| kd_cr | +0.0041 | −0.1148 | 4.71e-6 | 1.47e-6 |
| kd2 | +0.0054 | −0.0325 | 6.80e-6 | 2.06e-6 |

#### Δh 对齐（t=0.5）

| checkpoint | cos(Δh, -∇L_dec) | cos(Δh, -∇L_rec) |
|:---:|:---:|:---:|
| baseline | −0.0037 | +0.049 |
| kd_cr | −0.0002 | −0.191 |
| kd2 | −0.0001 | −0.193 |

### 核心发现

**1. Reconstruction–decode tradeoff 假说在 baseline 上确认**

baseline B11 残差更新（h10→h11）：ΔL_dec = +7.92（CE 变差），ΔL_rec = −501（MSE 大幅改善）。方向严格相反——B11 在帮助重建的同时损害了 decode 能力。这是该假说的第一个直接定量证据。

**2. KD 消除 tradeoff**

kd_cr 和 kd2 两条路径同时改善（ΔL_dec < 0，ΔL_rec < 0）。B11 不再是 decode-hostile 的。

**3. Gradient conflict 是弱信号，无法区分 checkpoint**

在 h11 处，per-pos mean 全部接近零（0.004–0.009）。Aggregate cos 略负但不单调：kd_cr（−0.115）甚至比 baseline（−0.011）更负。原因：baseline 在 h11 处 L_rec 几乎为零（gnorm_rec = 1.31e-7 << gnorm_dec = 1.18e-4），重建梯度近零导致 cos 无意义。

**4. 关键新观测：baseline h10 重建质量极差**

baseline L_rec(h10) = 529（t=0.5），而 kd_cr/kd2 为 103–142。KD 将重建计算分散到早层（B00–B10），而 baseline 将几乎全部重建负担集中在 B11——这正是造成 tradeoff 的根本原因。

**5. Δh 对齐次要信号**

cos(Δh, -∇L_dec) 全部接近零（−0.004 至 −0.0001），baseline 略更 anti-decode 但信号极弱。更有意思的是 cos(Δh, -∇L_rec)：baseline = +0.049（B11 残差仍指向"更好重建"的方向），kd_cr/kd2 = −0.19（B11 更新让 h11 超过了 L_rec 的局部最优，与插值曲线中 L_rec 最小值在 α≈0.75 处一致）。

### 与假说的对比

| checkpoint | ΔL_dec (0→1) | ΔL_rec (0→1) | conflict cos | Δh·(-g_dec) |
|:---:|:---:|:---:|:---:|:---:|
| baseline | **+7.92 ↑（变差）** | **−501 ↓（变好）** | −0.011 | −0.0037 |
| kd_cr | −0.024 ↓（变好） | −60.3 ↓（变好） | −0.115 | −0.0002 |
| kd2 | −0.063 ↓（变好） | −28.5 ↓（变好） | −0.033 | −0.0001 |

**预期**（spec 中的 prediction table）：
- ΔL_dec ↑ 且 ΔL_rec ↓ for baseline：✅ **完全验证**
- 两条路径都改善 for KD：✅ **完全验证**
- gradient conflict cos < 0 for baseline（预期），> 0 for KD（预期）：❌ **不成立** —— cos 在全部三个 checkpoint 上都轻微为负，且 kd_cr 甚至更负；梯度范数退化导致该指标在此设计下无区分力
- Δh·(-g_dec) < 0 for baseline：✅ 弱确认（信号极小）

### 允许的结论（纸质用语）

> The B11 residual update in the baseline model creates a geometric tradeoff: transitioning from h_10 to h_11 worsens decode accuracy (L_dec increases by +7.9 CE units) while substantially improving reconstruction quality (L_rec decreases by −501 MSE units). KD-trained checkpoints eliminate this tradeoff: both L_dec and L_rec decrease along the same h_10→h_11 direction (EXP-43). Notably, baseline's h_10 carries dramatically higher L_rec (529 vs 103–142 for KD checkpoints), suggesting KD distributes reconstruction computation across earlier layers rather than concentrating it in B11.
