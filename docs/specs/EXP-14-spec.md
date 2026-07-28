# EXP-14 Spec — Commit-Release-Recommit Validation on Real Trajectories

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：验证"承诺-释放-再承诺"是否是真实动力学现象。**

⚠️ **前提条件**：EXP-01（Protocol B）必须先完成。

论文目前描述的"commit-release-recommit"模式来自 Protocol A（oracle 前向噪声探针）中观察到的**非单调性**：某些位置在 t=0.2 时可以正确解码，但在 t=0.3 时不能，再在 t=0.5 时又可以了（A→B→A 模式）。

**关键问题**：这个非单调性是 Protocol A 特有的现象（因为每个 t 使用独立的随机噪声），还是在真实 ODE/SDE 轨迹中也出现？

**在 Protocol A 中**：非单调性的来源是：z_t 在不同 t 下使用不同的 ε，所以位置的"难度"随 ε 而变化，导致某些位置在某个 t 下的具体噪声实例刚好难，而其他 t 下的实例刚好容易。这是测量噪声，不是真实的动力学。

**在 Protocol B 中**：ODE 轨迹中的 x_pred 序列是连续的（z_{t+Δt} 从 z_t 连续演变）。如果这里出现非单调性（位置在 t=0.2 预测正确 → t=0.3 预测错误 → t=0.5 再次正确），那才是真正的"commit-release-recommit"动力学。

**要验证的核心假说**：
- Protocol B 轨迹中也存在非单调性（commit-release-recommit 是真实动力学）
- OR：Protocol B 轨迹中 x_pred 序列是单调递增的（non-monotonicity 是 Protocol A 的测量噪声）

**决策规则**：
- 若 Protocol B 非单调性率 >5%（即 >5% 的位置在轨迹中出现回退）：论文的 commit-release-recommit 主张有直接证据支持，可以大胆使用这个语言
- 若 Protocol B 非单调性率 <1%：commit-release-recommit 是 Protocol A 的测量伪影，论文必须将其重新表述为"oracle 探针观察到的非单调性"，而非真实动力学特征

**与其他实验的关系**：
- 依赖 EXP-01 Protocol B 轨迹数据（必须先完成）
- 结果决定论文 §4.14（commit-release-recommit 小节）的措辞和重要性

---

## Implementation

**Script:** `experiments/probe_elf/analyze_traj_stability.py`（可能已存在，检查内容）

或编写新脚本：

```python
"""
EXP-14: Analyze non-monotonicity in Protocol B trajectories.

For each position and trajectory:
1. At each step t, check if x_pred decodes to the "final" token (y_proxy = last step argmax)
2. Track the sequence of "commit" (1) and "no commit" (0) states across steps
3. Compute:
   - monotonic rate: fraction of positions where once committed, stay committed
   - non-monotonic rate: fraction where commitment/release pattern appears
"""

import torch, os, json

TRAJ_DIR = "results/exp01/trajectories"
CKPT = "converted/elf_b-owt-kd-cr_torch.pt"

# Load unembed weights for G(t) computation
ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
params = ckpt.get("params", ckpt)
W = params["unembed_kernel"].T  # (V, 512)
bias = params.get("unembed_bias", None)

stats = {"n_positions": 0, "n_monotonic": 0, "n_non_monotonic": 0, 
         "n_never_committed": 0, "n_always_committed": 0}

for fname in sorted(os.listdir(TRAJ_DIR)):
    if not fname.endswith(".pt"):
        continue
    traj = torch.load(os.path.join(TRAJ_DIR, fname), map_location="cpu", weights_only=False)
    
    # Get y_proxy = final step argmax
    last_x = traj[-1]["x_pred"].float()  # (B, L, 512)
    if bias is not None:
        y_proxy = (last_x @ W.float().T + bias.float()).argmax(dim=-1)  # (B, L)
    else:
        y_proxy = (last_x @ W.float().T).argmax(dim=-1)
    
    # For each step, compute commit status
    B, L = y_proxy.shape
    commit_history = []
    for step in traj:
        x_t = step["x_pred"].float()
        if bias is not None:
            pred_t = (x_t @ W.float().T + bias.float()).argmax(dim=-1)
        else:
            pred_t = (x_t @ W.float().T).argmax(dim=-1)
        committed = (pred_t == y_proxy)  # (B, L) bool
        commit_history.append(committed)
    
    # commit_history: list of (B, L) bool tensors, one per step
    commit_matrix = torch.stack(commit_history, dim=0)  # (T, B, L)
    
    # Analyze per position
    for b in range(B):
        for l in range(L):
            seq = commit_matrix[:, b, l].tolist()  # list of bool, length T
            stats["n_positions"] += 1
            
            if all(c == False for c in seq):
                stats["n_never_committed"] += 1
            elif all(c == True for c in seq):
                stats["n_always_committed"] += 1
            else:
                # Check for non-monotonicity (commit then release)
                found_commit = False
                non_monotonic = False
                for c in seq:
                    if c:
                        found_commit = True
                    elif found_commit:  # was committed, now not
                        non_monotonic = True
                        break
                if non_monotonic:
                    stats["n_non_monotonic"] += 1
                else:
                    stats["n_monotonic"] += 1

n = stats["n_positions"]
print(f"Positions: {n}")
print(f"Never committed: {stats['n_never_committed']/n:.2%}")
print(f"Always committed: {stats['n_always_committed']/n:.2%}")  
print(f"Monotonically committed: {stats['n_monotonic']/n:.2%}")
print(f"Non-monotonic (C→R→C): {stats['n_non_monotonic']/n:.2%}")
```

运行（CPU-only，不需要 GPU）：
```bash
cd models/ELF-torch
python experiments/probe_elf/analyze_traj_stability.py
```

---

## 实验结果（Results）

**状态**: COMPLETED（见下方 COMPLETED 2026-07-18 节）

**前置条件已就绪**：
- 轨迹文件：`results/exp01/trajectories/`（3 个文件，每个 64 seqs × 32 steps）
- x_pred 形状：(64, 1024, 512) per file

**可立即运行**：上方脚本可直接在已有轨迹上运行，无需 GPU。

**期望结果**：
- 若 non-monotonic rate < 2%：ELF ODE 轨迹几乎完全单调，承诺后不释放
- 若 non-monotonic rate > 5%：有真实的 commit-release-recommit 动力学

---

## ⚠️ 实验结果（COMPLETED 2026-07-18）

**原始数据文件**：`results/exp14/traj_stability.json`

**日志文件**：`/tmp/exp14.log`

**关键数据**（kd-cr checkpoint, 3批 × 64 sequences × 32步, N=196,608 位置）：

```
0 flips: 0.2%   | 1 flip: 0.7%  | 2-4 flips: 15.6% | 5+ flips: 83.4%
mean flips: 6.91  |  median: 7.0
mean last-flip step: 27.0/32 (t≈0.844)
frac last-flip in first half: 9.4%
```

**每步与最终答案匹配率**（= "已承诺"率）：

| 步骤 | t | 匹配终态 |
|-----|---|---------|
| 1 | 0.000 | 1.66% |
| 9 | 0.250 | 8.88% |
| 17 | 0.500 | 20.10% |
| 25 | 0.750 | 28.40% |
| 29 | 0.875 | 40.10% |
| 32 | 0.969 | 100% |

**关键发现与解读**：

1. **极高的非单调率（83.4%）**：大多数位置在 ODE 轨迹中频繁改变预测，commit-and-stay 模式（单调）只占约 16%。这与 Protocol A 的"承诺悬崖"故事完全相反。

2. **承诺时间极晚**：位置平均在第 27/32 步（t≈0.84）才最终稳定，而 Protocol A 显示 kd-cr 在 t=0.30 就有 90.2% 承诺。差距超过 2× t 单位。

3. **⚠️ 指标差异警告**：EXP-14 使用 `x_hat @ unembed_kernel.T` 作为解码函数（同 EXP-01），这是错误的解码路径（不同于 decode-path G）。高翻转率可能部分来自解码函数在中间步骤的噪声。

4. **EXP-14 vs Protocol A 的不可比性**：
   - Protocol A "committed" = decode_path(L11_hidden) 对真实 token 给出正确答案
   - Protocol B "committed" = argmax(x_hat @ unembed_kernel) 匹配最终步骤的 proxy GT
   - 即使排除指标问题，两者的 GT 定义也不同（真实 token vs 生成的 proxy）

**对论文的影响**：

这一发现需要非常谨慎地处理。两种解读：

**A. 乐观（EXP-01 正确指标可能仍显示悬崖）**：EXP-14 的高翻转率来自 x_hat @ unembed_kernel 在中间步骤的噪声。若用 decode-path（L11 hook）重新计算，翻转率可能低得多，Protocol A 的承诺悬崖故事仍然成立。

**B. 悲观（实际轨迹承诺更晚）**：即使解码函数有噪声，位置在 t=0.85 左右才"锁定"（匹配终态 40%→100%）表明 kd-cr 的真实生成过程中承诺极晚，而非 Protocol A 所示的 t=0.30 早期承诺。

**待做（2026-07-18 已完成）**：EXP-01v2（`experiments/probe_elf/probe_rev_traj_v2.py`）用 L11 forward hook 重新分析了相同的轨迹文件，得到正确的 decode-path G_B(t)。结果（`results/exp01v2/proto_B_decode_G.json`）：

| t | G_B (correct decode path, proxy GT) |
|---|---|
| 0.10 | 11.6% |
| 0.25 | 28.0% |
| 0.50 | 43.4% |
| 0.75 | 51.4% |
| 0.97 | 66.4% |

**EXP-01v2 结论**（支持 B 方向）：即使用正确的 decode path，Protocol B 的 G_B 仍然远低于 Protocol A 的 G_A（t=0.30 时 33% vs 89%），且**没有出现悬崖**，而是线性缓慢增长。这确认了"承诺悬崖"是 oracle 协议的特征，而非真实生成动力学的特征。

EXP-14 的高翻转率（83.4%）和 EXP-01v2 的缓慢 G_B 增长都指向同一方向：kd-cr 的真实 ODE 轨迹中，位置承诺是渐进的，不是突变的。

---

## 三模型对比（2026-07-18 补充）

**数据文件**：
- `results/exp14_baseline/traj_stability.json` — baseline checkpoint
- `results/exp14_kd_cr/traj_stability.json` — kd-cr checkpoint（与上方分析相同）
- `results/exp14_kd2/traj_stability.json` — kd2 checkpoint

**翻转分布对比**（N ≈ 131,072 positions each）：

| 指标 | baseline | kd_cr | kd2 |
|------|---------|-------|-----|
| zero flips | 0.13%  | 0.19%  | 0.046% |
| 1 flip     | 1.60%  | 0.74%  | 0.372% |
| 2–4 flips  | 40.1%  | 15.6%  | 9.74%  |
| **5+ flips** | **58.2%** | **83.4%** | **89.8%** |
| mean flips | 5.17   | 6.91   | 7.93   |
| mean last-flip step (of 32) | **18.3** | **27.0** | **28.5** |
| mean last-flip t  | ≈0.57  | ≈0.844 | ≈0.891 |
| frac last-flip in first half | 47.7% | 9.4%  | 2.9%  |

**与终态匹配率随步骤变化**（即协议B承诺率 G_B_proxy）：

| 步骤 | t | baseline | kd_cr | kd2 |
|-----|---|---------|-------|-----|
| 1  | 0.000 | 0.41% | 1.66% | 0.77% |
| 9  | 0.250 | 12.8% | 8.88% | 3.97% |
| 17 | 0.500 | 54.3% | 20.1% | 7.5% |
| 25 | 0.750 | 75.9% | 28.4% | 15.3% |
| 29 | 0.875 | 82.0% | 40.1% | 26.1% |
| 32 | 0.969 | 100%  | 100%  | 100% |

**关键解读**：

1. **baseline 比 kd_cr 更早稳定**：baseline 到 step 17（t=0.5）时已有 54.3% 的位置锁定终态，而 kd_cr 只有 20.1%，kd2 只有 7.5%。

2. **kd 模型翻转更频繁**：kd_cr 5+ flips = 83.4%，kd2 = 89.8%（高于 baseline 的 58.2%）。KD 训练使模型在 ODE 轨迹中间更频繁地改变预测——这与 Protocol A 的"KD 更早承诺"方向相反。

3. **kd2 最晚稳定**：mean last-flip step = 28.5/32（t≈0.891），是三个模型中最晚的。

4. **解读一致性（与 EXP-01v2）**：EXP-01v2 的 G_B 曲线（kd_cr 在 t=0.50 时 43.4% vs baseline 的 Protocol A 在 t=0.50 时 80%）与此处 baseline 在 step 17 有 54.3% 锁定终态一致——baseline 在真实轨迹上反而更"稳"。

5. **不要混淆 Protocol A 和 Protocol B**：Protocol A 用 oracle 噪声评估模型的"能力"（t=0.30 时 kd_cr 90%），Protocol B（此处）测量实际生成轨迹的稳定性（t=0.30 时 kd_cr 只有约 15% 锁定终态）。

---

## EXP-14v2 结果（2026-07-22，corrected decode path）

**脚本**：`experiments/probe_elf/analyze_traj_stability.py`（重写版，使用 block-11 forward hook）
**正确 decode path**：`GELU(h_L11 @ proj_kernel + proj_bias) @ unembed_kernel + unembed_bias`

### kd_cr（COMPLETE）

**N = 196,608 positions**（192 seqs × 1024 tokens，3 trajectory files）

| 翻转次数 | 比例 |
|----------|------|
| 0 flips  | 2.7% |
| 1 flip   | 10.4% |
| 2–4 flips | 38.5% |
| **5+ flips** | **48.4%** |
| mean flips | **4.66** |
| median flips | 4.0 |
| mean last-flip step | 19.3/32 |
| flip rate per unit t | 4.81 |

**G_B(t)（decode path，proxy GT = decode prediction at final step）**：

| step | t | frac match proxy GT |
|------|---|---------------------|
| 1  | 0.000 | 3.79% |
| 9  | 0.250 | 28.01% |
| 17 | 0.500 | 43.44% |
| 25 | 0.750 | 51.35% |
| 29 | 0.875 | 55.83% |
| 32 | 0.969 | **66.42%** |

**注意**：这些 G_B 数字与旧 EXP-01v2（2026-07-18）完全一致，因为使用相同的轨迹文件和相同的 decode path。

### 与旧 EXP-14（wrong readout）的比较

| 指标 | EXP-14 (wrong: x_hat @ unemb) | EXP-14v2 (correct: decode path) |
|------|-------------------------------|--------------------------------|
| 5+ flips | **83.4%** | **48.4%** |
| mean flips | 6.91 | 4.66 |
| mean last-flip step | 27.0/32 | 19.3/32 |

**正确 decode path 比 raw x_hat 更稳定**：5+ flip 率从 83.4% 降到 48.4%，mean flips 从 6.91 降到 4.66。这合理 — decode path 是经过学习的读出接口，比原始输出更平滑。

但即使如此，**48.4% 的位置仍然翻转 5+ 次，mean flips = 4.66**，on-policy ODE 轨迹仍然极不稳定。

### ⚠️ stable_commit_step 指标在 Protocol B 中无意义

`frac_stably_committed=1.0, mean_step=1.1`：这是假阳性！

原因：t=0 时（步骤 1）模型对所有位置输出相同的默认 token（噪声下的平坦预测），连续 3 步相同即触发 stable commit，但那个 token 并不是最终答案。

**Protocol B 的正确 stable commit 应定义为**：min{step_i: 预测连续 K 步匹配 proxy_GT}。现有实现缺少"匹配 proxy_GT"这一条件，需要在脚本中修复。

### 三模型完整对比（EXP-14v2，correct decode path，2026-07-22）

**N ≈ 196,608 positions per checkpoint**（192 seqs × 1024 tokens，3 trajectory files）

#### 翻转分布

| 指标 | baseline | kd_cr | kd2 |
|------|---------|-------|-----|
| 0 flips | 0.0% | **2.7%** | **2.5%** |
| 1 flip | 2.4% | 10.4% | 11.5% |
| 2–4 flips | 30.0% | 38.5% | 40.2% |
| **5+ flips** | **67.6%** | **48.4%** | **45.8%** |
| mean flips | **6.08** | 4.66 | **4.48** |
| median | 6.0 | 4.0 | 4.0 |
| flip rate / unit t | 6.27 | 4.81 | 4.63 |
| mean last-flip step | 21.2/32 | 19.3/32 | 19.4/32 |

#### G_B(t)（Protocol B，decode path）

| t | baseline | kd_cr | kd2 |
|---|---------|-------|-----|
| 0.000 | 0.4% | **3.8%** | **4.3%** |
| 0.250 | 18.5% | **28.0%** | 26.0% |
| 0.500 | 37.2% | **43.4%** | **43.4%** |
| 0.750 | 46.2% | 51.3% | **53.1%** |
| 0.875 | 52.2% | 55.8% | **60.3%** |
| 0.969 | **73.0%** | 66.4% | **73.0%** |

#### 关键发现（corrected readout）

1. **KD 比 baseline 更稳定（翻转更少）**：kd_cr/kd2 的 5+ flip 率（45-48%）低于 baseline（67.6%）。与旧 EXP-14（错误 readout）结论相反——旧结论是"kd_cr 83.4% > baseline 58.2%"。正确 decode path 下，KD 实际上使 ODE 轨迹更稳定。

2. **KD 在早期步骤 G_B 更高**：kd_cr 在 t=0.25 时 G_B=28%（baseline=18.5%）。说明 KD 使 decode 预测更早对齐 ODE 终态。

3. **kd_cr 最终收敛最差（66.4%）**：baseline 和 kd2 都达到 73%，但 kd_cr 只有 66.4%。kd_cr 的 decode 预测在最后几步（t=0.875→0.969→proxy_GT）仍有较多变化——这与 kd_cr 在中间步骤显示更高 G_B 并不矛盾，说明 kd_cr 的预测在最后几步有特定不稳定性。

4. **与旧 EXP-14 的逆转**：
   - 旧结果（错误 x_hat readout）：kd_cr 最高翻转率（83.4%），baseline 最低（58.2%）
   - 新结果（正确 decode path）：kd_cr 中等翻转率（48.4%），baseline 最高（67.6%）
   - 这个逆转证明**原始 EXP-14 的结论完全无效**。

5. **与 Protocol A 的对比**：
   - Protocol A（EXP-09v3）：kd_cr never_commit = 0.67%（几乎全部稳定提交）
   - Protocol B（EXP-14v2）：kd_cr 5+ flips = 48.4%（接近一半位置频繁翻转）
   - 差距巨大，证明 Protocol A off-policy readout ≠ Protocol B on-policy 动力学

