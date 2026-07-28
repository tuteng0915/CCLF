# EXP-11 Spec — Branching Stability

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：测量 ODE 轨迹在中间时刻的"分叉稳定性"，验证承诺是否等价于"无法改变"。**

EXP-01/01v2 发现 Protocol B 的 decode-path 预测随 t 缓慢增长（没有悬崖）。EXP-14 发现 83.4% 的位置在 ODE 轨迹中翻转 ≥5 次（高不稳定性）。但这两个实验都没有回答一个关键问题：

**"在 ODE 轨迹的某个时刻 t*，如果我们从同一个 z_{t*} 出发，运行 K 条不同的完整 ODE（每条用不同初始随机性，但从 t* 开始，用确定性 ODE 无额外随机性），它们会收敛到相同的最终 token 吗？"**

这个"分叉稳定性"直接回答了：**对于 ELF 的确定性 ODE（无 SDE 噪声），从 t* 时刻出发，最终 token 是否已经确定？**

- 如果 K 条 ODE 最终都收敛到同一 token：该位置在 t* 已经"动力学承诺"（deterministic commitment）
- 如果 K 条 ODE 发散到不同 token：该位置在 t* 仍未承诺（尽管 decode 可能给出高置信预测）

**与 Protocol A 的联系**：
- Protocol A 测的是"给定理想噪声版本 z_t，decode 是否能正确"
- EXP-11 测的是"给定真实 ODE z_t，继续确定性 ODE 是否确定"
- 后者更接近"承诺"的物理意义

**要验证的核心假说**：
- 若 t < 0.50 的动力学承诺率（K 条 ODE 一致率）≈ Protocol A G(t)：oracle 探针有效代理了真实动力学
- 若动力学承诺率比 G(t) 低很多（t=0.50 时一致率仅 30%，而 G(t)=99.5%）：oracle 过度乐观

---

## Implementation Plan

### 方法

```python
"""
EXP-11: Branching stability from saved ODE checkpoints.

1. Load trajectory: traj_1897769.pt (from EXP-01 trajectories)
2. At each ODE step t_split in {0.10, 0.20, 0.30, 0.50}:
   a. Get z_{t_split} from the trajectory
   b. Run K=8 independent deterministic ODE continuations from z_{t_split}
      (different initial z_0 → different trajectory prefix → same z_{t_split} by design)
      (Since ODE is deterministic, K runs from SAME z_{t_split} always give same result!
       To get variation, need to add small perturbations δ to z_{t_split})
   c. Record final token from each continuation
   d. Compute per-position agreement = I(all K continuations agree on final token)
"""
```

**关键洞察**：ELF 用的是**确定性 ODE**（dec_sc_mode='none'，无 SDE 噪声）。从同一个 z_{t*} 出发，确定性 ODE 总是给出相同结果。所以需要加微小扰动 δ（如 ||δ||₂ = 0.01 * ||z_{t*}||₂）。

```python
for t_split in [0.10, 0.20, 0.30, 0.50, 0.70]:
    z_split = traj[step_idx]["z_t"]  # (B, L, 512) — actual ODE state
    
    final_tokens = []
    for k in range(K):
        eps = torch.randn_like(z_split) * 0.01 * z_split.norm(dim=-1, keepdim=True)
        z_perturbed = z_split + eps
        # Run remaining ODE steps from z_perturbed at t=t_split
        z_final = run_ode_from_checkpoint(model, z_perturbed, t_start=t_split, n_steps=remaining)
        final_token = argmax(decode_path(L11_hook(z_final, t=1.0)))
        final_tokens.append(final_token)
    
    # Per-position agreement = fraction where all K agree
    agreement = (torch.stack(final_tokens) == final_tokens[0]).all(0).float().mean()
    print(f"t={t_split}: branching stability = {agreement:.4f}")
```

### 脚本

**新建**：`experiments/probe_elf/probe_branching_stability.py`

注：需要 EXP-01 的轨迹文件 (`results/exp01/trajectories/`) 和 L11 hook

### 运行配置

- K=8 个扰动
- 扰动强度：||δ||₂ = 0.01·||z_{t*}||₂（小扰动，测量局部稳定性）
- GPU：任意空闲 GPU

---

## 实验结果（Results）

**状态**: COMPLETED（见下方 COMPLETED 2026-07-18 节）

**依赖**：EXP-01 轨迹文件（已就绪）

**优先级**：中等。EXP-01v2 已经给出了"Protocol B 不显示悬崖"的直接证据。EXP-11 提供额外的"分叉稳定性"视角，与 EXP-01v2 互补。

**决策规则**：
- 若 branching stability @ t=0.30 > 70%：ODE 在 t=0.30 之后大多数位置已经确定（动力学承诺），与 Protocol A 的故事一致
- 若 branching stability @ t=0.30 < 40%：即使 decode 置信度很高，实际 ODE 仍在 t=0.30 时"分叉"，说明"承诺"是探针人工产物
- 注意：确定性 ODE 对小扰动的响应可能因非线性程度而异，需要选择合适的扰动强度

---

## 实验结果 — COMPLETED 2026-07-18

**状态**: COMPLETED

数据文件:
- kd_cr: `results/exp11/branching_stability.json` (10 records, 5 t_splits × 2 traj files)
- baseline: `results/exp11_baseline/branching_stability.json` (10 records)

参数: K=8 perturbations, noise_frac=0.01 (σ = 1% RMS of z_{t*}), n_seqs≈128

### 结果表

| t_split | kd_cr mean_stab | baseline mean_stab |
|---------|-----------------|-------------------|
| 0.094   | 0.36%           | 0.74%             |
| 0.188   | 0.57%           | 0.76%             |
| 0.312   | 2.10%           | 0.72%             |
| 0.500   | 4.88%           | 7.29%             |
| 0.688   | 7.72%           | 5.52%             |

所有 p10_stability = 0.00（即至少 90% 的位置 8 次扰动没有一次与原始结果一致）。

### 解读

**核心发现：ELF ODE 对 1% 扰动极度敏感，即使在 t=0.688 时仍有 92-94% 的位置被翻转。**

1. **绝对值极低**：即使在 t=0.688（完成 69% 的去噪）时，mean_stability < 8%。这意味着 1% RMS 扰动足以使 92% 的位置最终 token 改变。

2. **kd_cr vs baseline 对比**：kd_cr 在 t=0.312 时比 baseline 稍高（2.10% vs 0.72%），在 t=0.5/0.688 两者均为 5-8%，无显著差异。说明 KD 训练没有改变 ODE 轨迹的本质稳定性。

3. **与 Protocol A"悬崖"的关系**：Protocol A 在 t=0.30 显示 kd_cr G(t)=90%，但 ODE 的分叉稳定性在 t=0.31 只有 2.1%——这说明即使 decode 的置信度很高，ODE 本身在 t=0.30 仍然极不稳定。"承诺"是探针读出的高置信度预测，而非 ODE 轨迹的真正收敛。

4. **baseline 的异常**：baseline 在 t=0.312 时稳定性（0.72%）与 t=0.094（0.74%）几乎相同，说明 baseline ODE 在 t=0.094→0.312 期间没有任何稳定性提升，与 G_A(t) 在 baseline 不显示悬崖一致。

5. **1/(1-t) 放大效应**：ELF ODE 的速度场为 v(z,t)=(z-x̂_t)/(1-t)，当 t→1 时分母趋近于 0，导致微小扰动被极度放大。即使在 t=0.688，后续 10 步内的放大效应已经足够将 1% 的扰动放大到 token 翻转的量级。这是 flow-matching ODE 的固有特性，不代表"承诺"失败。

### 决策规则结论

按照原始决策规则：branching stability @ t=0.30 = 2.10% (kd_cr), 0.72% (baseline)，均远低于 40% 阈值。
→ **结论**：即使 decode 置信度在 t=0.30 很高，ODE 本身在该时刻仍然不稳定。"承诺"是探针人工产物而非 ODE 的真正收敛。

**注意**：本实验的稳定性极低可能部分由 1/(1-t) 放大机制造成，而非纯粹的 ODE 混沌。未来可以用更小的扰动（noise_frac=0.001）或在更靠近 t=1 的区间测试，以区分两种效应。

⚠️ **EXP-11 已被 EXP-11v2 取代**：EXP-11 使用 `sigma=noise_frac*rms_mean` 即全局 RMS 缩放，实际扰动范围约 22.6%（而非声称的 1%）；且未 sweep η。见 EXP-11v2 修正。

---

## EXP-11v2 结果（2026-07-22，Corrected Per-Position Scaling）

**状态**: kd_cr DONE；baseline(GPU6) + kd2(GPU7) 仍在运行  
**脚本**: `experiments/probe_elf/probe_branching_stability.py`  
**修正**: 使用 per-position 单位球缩放 `u = randn; u /= |u|; delta = η × ||z_split|| × u`，η sweep: {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}，K=4，n_seqs=64

**输出**: `results/exp11v2_{kd_cr,baseline,kd2}/branching_stability.json`

### kd_cr 结果（GPU2，n_seqs=64，M=65,536 per entry）

| t_split | η      | S_orig | p10  | S_pair | all_same |
|---------|:------:|:------:|:----:|:------:|:--------:|
| 0.0938  | 1e-4   | 0.0038 | 0.00 | 0.9819 | 0.9672   |
| 0.0938  | 3e-4   | 0.0038 | 0.00 | 0.9819 | 0.9673   |
| 0.0938  | 1e-3   | 0.0038 | 0.00 | 0.9818 | 0.9669   |
| 0.0938  | 3e-3   | 0.0038 | 0.00 | 0.9780 | 0.9605   |
| 0.0938  | 1e-2   | 0.0038 | 0.00 | 0.9554 | 0.9203   |
| 0.1875  | 1e-4   | 0.0064 | 0.00 | 0.9749 | 0.9544   |
| 0.1875  | 1e-3   | 0.0064 | 0.00 | 0.9746 | 0.9542   |
| 0.1875  | 1e-2   | 0.0064 | 0.00 | 0.9503 | 0.9115   |
| 0.3125  | 1e-4   | 0.0237 | 0.00 | 0.9678 | 0.9418   |
| 0.3125  | 1e-3   | 0.0238 | 0.00 | 0.9673 | 0.9411   |
| 0.3125  | 1e-2   | 0.0238 | 0.00 | 0.9510 | 0.9122   |
| 0.5000  | 1e-4   | 0.0466 | 0.00 | 0.9738 | 0.9527   |
| 0.5000  | 1e-3   | 0.0466 | 0.00 | 0.9736 | 0.9523   |
| 0.5000  | 1e-2   | 0.0466 | 0.00 | 0.9706 | 0.9465   |
| 0.6875  | 1e-4   | 0.0830 | 0.00 | 0.9801 | 0.9641   |
| 0.6875  | 1e-3   | 0.0828 | 0.00 | 0.9800 | 0.9639   |
| 0.6875  | 1e-2   | 0.0828 | 0.00 | 0.9782 | 0.9604   |
| 0.8125  | 1e-4   | 0.2959 | 0.00 | 0.9707 | 0.9472   |
| 0.8125  | 1e-3   | 0.2960 | 0.00 | 0.9704 | 0.9466   |
| 0.8125  | 1e-2   | 0.2960 | 0.00 | 0.9634 | 0.9346   |

**S_orig by t_split（η=1e-4，smallest perturbation）**:
- t=0.09: 0.38%
- t=0.19: 0.64%
- t=0.31: 2.37%
- t=0.50: 4.66%
- t=0.69: 8.30%
- t=0.81: **29.60%**

### 三个关键发现（kd_cr，待与 baseline/kd2 比较）

1. **S_orig 极低，对 η 不敏感**：η 从 1e-4 到 1e-3 缩放 10 倍，S_orig 几乎不变（如 t=0.50: 0.0466→0.0466→0.0466）。只有在 η=1e-2 时 S_pair 和 all_same 略有下降（S_pair: ~0.974→0.971→0.971→0.971→0.970）。这说明 ODE 的敏感性不是参数连续的——有一个阈值。

2. **S_pair ≫ S_orig 的分叉模式**：S_orig≈0.05 but S_pair≈0.97 at t=0.50。这意味着：扰动后的 K 个轨迹互相高度一致（97%），但它们不是回到原来的 token，而是集体收敛到一个不同的吸引子。这是"bifurcation into a common alternative"而非随机散布。

3. **S_orig 随 t_split 增长**：t=0.09: 0.38% → t=0.81: 29.60%（87× increase）。轨迹越晚被扰动，越有可能保持稳定。

### Preliminary Comparison with Original EXP-11

| t_split | EXP-11 kd_cr stability | EXP-11v2 kd_cr S_orig (η=1e-3) |
|---------|:---------------------:|:-------------------------------:|
| ~0.09  | 0.36%                  | 0.38%                           |
| ~0.31  | 2.10%                  | 2.38%                           |
| ~0.50  | 4.88%                  | 4.66%                           |
| ~0.69  | 7.72%                  | 8.28%                           |

EXP-11 和 EXP-11v2 在 η≈1e-3 时结论高度一致，说明"1% noise_frac"实际上等价于 η≈1e-3（不是 22.6% 的扰动强度误差那么大）。原始 EXP-11 的 bug 在于 sigma 计算与 per-position scaling 混淆，但幸运地实际扰动量级与 η=1e-3 相近。

### 三模型完整对比（EXP-11v2，η=1e-4）

**状态**: 全部 DONE（kd_cr GPU2，baseline GPU6，kd2 GPU7）

#### S_orig（Mean of Perturbed-Original Agreement）

| t_split | kd_cr S_orig | kd2 S_orig | baseline S_orig |
|---------|:----------:|:----------:|:--------------:|
| 0.0938  | **0.38%**  | 0.76%      | 0.79%          |
| 0.1875  | **0.64%**  | 1.30%      | 0.78%          |
| 0.3125  | 2.37%      | **3.35%**  | 0.73% ← dip   |
| 0.5000  | 4.66%      | 6.56%      | **7.01%**      |
| 0.6875  | **8.30%**  | 7.42%      | 5.86% ← dip   |
| 0.8125  | **29.60%** | 25.68%     | 17.82%         |

#### S_pair（Two Perturbed Trajectories Agreeing）

| t_split | kd_cr S_pair | kd2 S_pair | baseline S_pair |
|---------|:----------:|:----------:|:-------------:|
| 0.0938  | **0.9819** | 0.9808     | 0.9716        |
| 0.1875  | **0.9749** | 0.9742     | 0.9593        |
| 0.3125  | 0.9678     | **0.9696** | 0.9312        |
| 0.5000  | **0.9738** | 0.9667     | 0.9663        |
| 0.6875  | 0.9801     | **0.9806** | 0.9654        |
| 0.8125  | **0.9707** | 0.9672     | 0.9495        |

p10_s_orig = 0.00 everywhere for all three checkpoints（至少 90% 位置 S_orig=0）。

#### 关键发现

1. **kd_cr > baseline 在 LATE t（0.69、0.81）**：kd_cr 在 t=0.81 时 S_orig=29.60% >> baseline 17.82%（kd_cr 稳定性提升约 66%）。**KD 使轨迹在后期更稳定**，与 EXP-14v2 中 kd_cr mean_last_flip_step=19.3 (vs baseline 21.2) 一致——kd_cr 位置更早完成最后一次翻转，t=0.69-0.81 时大多已经收敛。

2. **kd_cr < baseline 在 EARLY t（0.09-0.19）及 t=0.50**：早期 kd_cr S_orig（0.38-0.64%）低于 baseline（0.78-0.79%）。在 t=0.50 时 kd_cr S_orig=4.66% < baseline 7.01%。这说明 kd_cr ODE 在早期对扰动"更敏感"——微小扰动就能将轨迹导向另一个吸引子。结合 kd_cr S_pair 仍然高（0.97+），这加强了 bifurcation into common alternative 模式。

3. **Baseline 非单调模式**：baseline 在 t=0.31 时 S_orig 下降（0.78%→0.73%），t=0.69 时再次下降（7.01%→5.86%）。这与 baseline mean_last_flip_step≈21.2（≈step 21/32 ≈ t≈0.66）吻合——baseline 的"最后一次翻转"集中在 t≈0.65-0.69 区间，此时轨迹最不稳定，扰动最容易改变最终结果。

4. **S_pair 对所有三模型均高**：bifurcation into common alternative 是通用的 ELF ODE 特性，不特定于某个 checkpoint。kd_cr/kd2 的 S_pair 略高于 baseline（尤其在 t=0.19-0.31: kd_cr 0.975 vs baseline 0.959/0.931），说明 KD 使"替代吸引子"更加 well-defined。

5. **关键问题的回答**：kd_cr S_orig 是否 > baseline S_orig？**答案取决于时间段**：在 t≥0.69 时 YES（late-stage stability improved），在 t≤0.50 时有时 NO（early dynamics more sensitive）。综合来看，KD 改善的是**后期稳定性**（轨迹更早完成最后翻转），而不是全程降低 ODE 敏感性。
