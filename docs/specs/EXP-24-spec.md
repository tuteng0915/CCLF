# EXP-24 Spec — LangFlow Trajectory Stability (LangFlow analog of EXP-14)

## 实验背景与动机

**在整体框架中的地位：用实际 ODE 轨迹测量 LangFlow 的位置稳定性，与 ELF EXP-14 对比。**

EXP-14 在 ELF 上发现：
- 83.4% 的位置在 32 步 ODE 轨迹中翻转 ≥5 次（argmax 改变）
- 平均最后翻转时间步为第 27/32 步（非常晚）
- 说明 ELF 的 ODE 轨迹是"反复挣扎"的，直到最后几步才稳定

**EXP-24 的核心问题**：LangFlow 的 ODE 轨迹有类似的"晚稳定"现象吗？

EXP-22 的 oracle probe 显示 LangFlow 的承诺时间极晚（t≈0.83-0.93 时 cliff），暗示 LangFlow 轨迹也应该在 t>0.80 之前保持不稳定。但这只是 oracle 预测，实际 ODE 轨迹的行为可能不同：
- LangFlow 的 ODE 步骤是线性插值（Euler-EDM），可能比 ELF 的 SDE 更稳定
- LangFlow 没有 ELF 的 self-conditioning（kd-cr 中的 decode branch），可能翻转更少

与 EXP-14 的区别：
- EXP-14 用 ELF-torch 的 ODE 采样（32 步），测量 argmax(decode_branch_logits) 的翻转
- EXP-24 用 LangFlow 的 Euler-EDM（32 步），测量 argmax(logits) 的翻转

---

## 实现计划

### 新文件：`experiments/probe_langflow/probe_traj_stability_langflow.py`

**核心逻辑**：修改 LangFlow 的 `generate_samples()` 以保存每步的 argmax(logits)，然后分析翻转统计。

```python
def generate_trajectory(model, num_samples, seq_length, num_steps, device):
    """Run LangFlow Euler-EDM, record argmax at each step."""
    embed_dim = model.config.hidden_size
    eps = 1e-5
    z = torch.randn(num_samples, seq_length, embed_dim, device=device)
    t = torch.linspace(1.0 - eps, eps, num_steps, device=device)
    gamma = model.proposal(t)

    x_self_cond = None
    trajectory = []  # list of [N, L] argmax tensors

    for i in range(len(gamma) - 1):
        gamma_t = gamma[i]
        gamma_s = gamma[i + 1]
        gamma_expanded = gamma_t.unsqueeze(0).expand(num_samples)

        with torch.no_grad():
            logits = model(noisy_embeds=z, timesteps=gamma_expanded,
                           x_self_cond=x_self_cond, return_dict=False)

        probs = F.softmax(logits.float(), dim=-1)
        x_pred = model._embed_tokens(probs)

        if model.config.self_conditioning:
            x_self_cond = x_pred

        # Record argmax at this step
        trajectory.append(logits.argmax(dim=-1).cpu())

        z = model._euler_edm_step(z, x_pred, gamma_t, gamma_s)

    # Final step
    gamma_final = gamma[-1]
    with torch.no_grad():
        logits = model(noisy_embeds=z,
                       timesteps=gamma_final.unsqueeze(0).expand(num_samples),
                       x_self_cond=x_self_cond, return_dict=False)
    trajectory.append(logits.argmax(dim=-1).cpu())

    return torch.stack(trajectory, dim=0)  # [num_steps+1, N, L]


def analyze_stability(trajectory):
    """
    trajectory: [T, N, L] int tensor of argmax token ids at each step.
    Returns dict with:
      - flip_count: [N, L] number of steps where argmax changed
      - last_flip_step: [N, L] last step where flip occurred (T if never flipped)
      - n_flips_geq5: fraction of positions with ≥5 flips
      - n_flips_0: fraction of positions with 0 flips (never changed)
      - mean_last_flip: mean last-flip step / T (normalized)
    """
    T, N, L = trajectory.shape
    # Count flips
    flips = (trajectory[1:] != trajectory[:-1])  # [T-1, N, L]
    flip_count = flips.sum(0)  # [N, L]

    # Last flip step (1-indexed step from start)
    # Find last True along time axis
    # trajectory goes t=1.0-eps to t=eps (reverse direction, like ELF)
    last_flip = torch.zeros(N, L, dtype=torch.long)
    for step in range(T - 1):
        mask = flips[step]  # [N, L]
        last_flip[mask] = step + 1  # 1-indexed

    return {
        "n_flips_geq5": float((flip_count >= 5).float().mean()),
        "n_flips_0": float((flip_count == 0).float().mean()),
        "mean_flip_count": float(flip_count.float().mean()),
        "mean_last_flip_frac": float(last_flip.float().mean()) / (T - 1),
        "flip_count_dist": {str(k): float((flip_count == k).float().mean())
                            for k in range(0, min(15, T))},
    }
```

### 运行命令

```bash
CUDA_VISIBLE_DEVICES=X conda run -n elf python \
    experiments/probe_langflow/probe_traj_stability_langflow.py \
    --checkpoint Continuous-Rivals-Discrete/langflow-owt \
    --n_samples 64 --seq_len 128 --num_steps 32 \
    --out_dir results/exp24_langflow
```

---

## 期望结果与决策规则

### 假说

**H_late**: LangFlow 轨迹也显示晚期稳定：大多数位置直到最后 5-10 步才停止翻转。

**H_early**: LangFlow 轨迹比 ELF 更早稳定：一旦在 t≈0.83 发生承诺悬崖，位置就不再翻转。

### ELF EXP-14 基准（用于对比）

| 指标 | ELF-baseline | ELF-kd_cr |
|------|-------------|-----------|
| flip ≥ 5 次 | 83.4% | ~75% |
| flip = 0 次 | ~5% | ~10% |
| 平均最后翻转（步/32） | 27.0 / 32 | ~22 / 32 |

### 决策规则

| LangFlow 结果 | 论文意义 |
|--------------|--------|
| LangFlow flip≥5 > 80%, last_flip > 25/32 | "LangFlow 轨迹同样不稳定；承诺晚是连续扩散 LM 的通性，不特定于 ELF" |
| LangFlow flip≥5 < 40%, last_flip < 15/32 | "LangFlow 轨迹比 ELF 更早稳定；ELF 的晚稳定可能与 self-cond 或 decode branch 有关" |

---

## 实验结果（Results）

**状态**: COMPLETE（2026-07-20）

数据：`results/exp24_langflow/traj_stability.json`

### 核心统计（32 ODE 步，64 样本 × 128 位置 = 8192 个位置）

| 指标 | LangFlow | ELF-baseline（EXP-14）|
|------|---------|----------------------|
| flip ≥ 1 次 | 95.6% | ~95% |
| flip ≥ 5 次 | **38.0%** | **83.4%** |
| flip ≥ 10 次 | 3.3% | — |
| flip = 0 次 | 4.4% | ~5% |
| 平均翻转次数 | **4.07** | ~8-10（估计）|
| 平均最后翻转 | **8.3/32 (26.9%)** | **27.0/32 (84.4%)** |

### 翻转次数分布

```
flips=0:  4.4%
flips=1: 10.3%
flips=2: 14.2%
flips=3: 17.1%
flips=4: 16.0%
flips=5: 12.8%
flips=6:  9.7%
flips=7:  6.3%
flips=8:  3.7%
flips=9:  2.1%
flips=10+:  2.3%
```

### 关键发现与解读

**LangFlow ODE 轨迹早期稳定，与 oracle 探针结论相反：**

- EXP-22（oracle 探针 Protocol A）：LangFlow backbone 在 t≈0.83-0.93 才承诺（oracle 在 t=0.73 时 91.7% 的位置未承诺）
- EXP-24（实际 ODE 轨迹 Protocol B）：LangFlow ODE 轨迹在步骤 8.3/32（t≈0.73）时 argmax 已稳定

这与 ELF 的情况**完全相反**：
- ELF：oracle 早期承诺（t≈0.15-0.30），但实际 ODE 轨迹翻转直到步骤 27/32
- LangFlow：oracle 晚期承诺（t≈0.83-0.93），但实际 ODE 轨迹在步骤 8.3/32 就稳定

**机制解释**：

LangFlow 的 Euler-EDM ODE 早期（步骤 0-8，t=1.0→0.73）快速锁定一个 token 候选，之后步骤 8-32（t=0.73→0.0）主要是提高置信度（降低熵）而不改变 argmax。这解释了 oracle 探针在 t=0.73 看到高熵（H>1 nat）但实际轨迹 argmax 已稳定的矛盾：

- Oracle (EXP-22) 用新鲜 noise ε 在 t=0.73 采样 → 模型看到"未去噪"的输入 → 高熵
- 实际轨迹在 t=0.73 的状态 z_{0.73} 已经被前 8 步"部分去噪" → argmax 已稳定（即使置信度低）

**结论**（与 EXP-14 的 ELF 决策规则对比）：

满足 H_early 而非 H_late：LangFlow 轨迹比 ELF 更早稳定。这证明 ELF 的"晚稳定、反复翻转"现象是 ELF 特有的，可能与 ELF 的 self-conditioning / decode branch 机制有关。

**论文意义（当前版本）**：
- LangFlow 的 ODE 是"早决策、晚置信"模式
- ELF 的 ODE 是"早潜知、晚决策"模式（backbone 知道答案但 ODE 轨迹保持探索）
- 这支持了 ELF 的 decode branch 作为"帮助 ODE 轨迹更快稳定"的机制假说

---

## ⚠️ 方法论问题（2026-07-22 审查）

### 1. CRITICAL：ELF baseline 对比数字使用了错误来源（EXP-14 旧版，非 EXP-14v2）

当前对比表：

> ELF-baseline flip≥5 = 83.4%（来自 EXP-14）

EXP-14 使用了错误的 decode path（`x̂_t @ unembed.T`，而非 `GELU(h_L11 @ proj) @ unembed`），结果已知无效。EXP-14v2 使用正确 decode path 的数字是：

| 指标 | EXP-14v2 baseline | EXP-14v2 kd_cr | LangFlow EXP-24 |
|------|:-----------------:|:--------------:|:---------------:|
| flip ≥ 5 次 | **67.6%** | **48.4%** | **38.0%** |
| mean flips | 6.08 | 4.66 | 4.07 |
| mean_last_flip_frac | ~21.2/32 (66%) | ~19.3/32 (60%) | 8.3/32 (26%) |

因此正确结论是：LangFlow flip≥5（38%）< ELF kd_cr（48.4%）< ELF baseline（67.6%）。"LangFlow 比 ELF 更早稳定"的方向不变，但差距缩小，且 LangFlow 的 mean_last_flip（8.3/32）仍远低于 ELF（约 19-21/32）。

spec 中 ELF baseline 基准（83.4%，决策规则表）必须替换为 EXP-14v2 数字（67.6%）。

### 2. argmax 稳定 ≠ commitment，"早决策"结论过强

LangFlow 在步骤 8/32 时 argmax 稳定，但 top-1 概率可能仍然极低（如 p=0.03 vs p=0.029）。Early argmax stability 不等于早期决策——它可能是：
- 当前最高的 token 仅以极低 margin 领先
- distribution 在剩余步骤中大幅变化（但同一 token 一直排第一）
- 实际 continuous-space ODE trajectory 还在大幅变化

若要支撑"LangFlow 早决策"，必须同时报告：
- top-1 probability 随步骤的变化
- top-1/top-2 margin
- entropy H(p_t) 随步骤的变化
- KL(p_{t+Δ} || p_t)（连续 posterior 变化量）
- expected embedding displacement |E^T p_{t+Δ} − E^T p_t|（这才是真正驱动 ODE 的量）

### 3. self-conditioning 解释已证伪（EXP-24v2 确认）

~~spec 中写"LangFlow 没有 ELF 的 self-conditioning"~~

**已确认：`langflow-owt` 的 `model.config.self_conditioning = True`**（EXP-24v2 脚本输出：`self_conditioning = True ← CONFIRMED`）。

因此"LangFlow 比 ELF 稳定是因为 LangFlow 无 self-conditioning"的机制解释完全无效。两者都有 self-conditioning，LangFlow 还有 skip 连接（`c_skip × z_t @ E.T`），机制差异来源需要 EXP-21v2 的 skip-connection 分离实验来厘清。

### 4. flip count 强烈依赖 step budget

LangFlow 32-step mean_flip=4.07 与 ELF 32-step 数字直接比较，但不同 step count 下 flip 数完全不可比（更多 step 意味着更多观察机会，flip 数必然更高）。若不跑 16/64/128 步进行对比，mean_flip 数字不能跨 solver-setting 比较。

### 5. 样本独立性

64 × 128 = 8,192 个位置，但同一 sequence 内高度相关。CIs 应按 sequence bootstrap（n=64），而非按 position bootstrap（n=8192）。

### 6. "ELF 早潜知、晚决策"与 Protocol B 结论

EXP-01v2 已证明 ELF Protocol B G_B(t) 在早期极低（远低于 oracle G_A(t)），说明实际轨迹 z_t 并未进入 oracle 可读的流形。因此"ELF backbone 知道答案但 ODE 轨迹保持探索"的说法需要限定：backbone 对 oracle states 知道答案，但对实际 ODE states 并非如此，两者是不同的 manifold。

### 安全结论（修正版）

EXP-24 当前结果支持的安全陈述：

> Under 32-step Euler-EDM, LangFlow's argmax proposals stabilize at an average of 8.3/32 steps from the start, compared to ELF's 19-21/32 steps (EXP-14v2). LangFlow shows 38% of positions with ≥5 argmax changes, substantially lower than ELF baseline (67.6%) and kd_cr (48.4%). Whether this reflects early genuine commitment (high-probability stable predictions) or simply low argmax diversity remains to be tested with per-step entropy and margin analysis.

---

## EXP-24v2 结果 — 每步熵分析（2026-07-22）

**数据**：`results/exp24v2_langflow/traj_entropy_stability.json`  
**脚本**：`experiments/probe_langflow/probe_traj_stability_v2.py`

### 翻转统计（修正 ELF 对比使用 EXP-14v2 数字）

| 指标 | LangFlow (EXP-24v2) | ELF baseline (EXP-14v2) | ELF kd_cr (EXP-14v2) |
|------|:-------------------:|:------------------------:|:---------------------:|
| flip ≥ 5 次 | **37.0%** | 67.6% | 48.4% |
| 平均翻转次数 | 3.99 | 6.08 | 4.66 |
| mean_last_flip | 8.4/32 (26.2%) | ~21.2/32 (66%) | ~19.3/32 (60%) |

### 每步熵 & 概率轮廓

| 步骤 | 平均熵 (nats) | 平均 top-1 概率 |
|------|:-------------:|:--------------:|
| step 0 | 7.556 | 0.039 |
| step 8 | 1.695 | 0.653 |
| step 16 | 0.675 | 0.847 |
| step 24 | 0.276 | 0.932 |
| step 31 | 0.016 | 0.994 |

### 首次高置信度阈值

| 阈值 | 平均首次步骤（/32）| frac_never |
|------|:------------------:|:----------:|
| top-1 p > 0.5 | **6.6/32** | 0% |
| top-1 p > 0.9 | **12.6/32** | 1.4% |
| top-1 p > 0.99 | **18.9/32** | 3.6% |

### 关键发现

**"早 argmax 稳定，晚置信建立"**（不是"早决策"）：

- Argmax 在步骤 8.3/32 就稳定（EXP-24 原始数字），但此时 top-1p 仅 **65.3%**
- 真正的 50% 置信度（p>0.5）在步骤 6.6/32，90% 置信度在步骤 12.6/32
- 分布在步骤 8-32 期间继续急剧收窄（熵从 1.69 → 0.016），即使 argmax 不变

这否定了 EXP-24 原始"早决策"叙述：argmax 稳定只是"当前最大 token 未被超越"，
并不意味着分布已经集中。正确表述：

> LangFlow uses an **early argmax lock-in at moderate confidence** (step ~8/32 at p=65%)
> followed by **late probability mass consolidation** (to p=99% by step ~19/32).
> This contrasts with ELF's late argmax stabilization (step ~21/32) despite early oracle-readout accuracy.

### self_conditioning=True 确认

`model.config.self_conditioning = True` 在 EXP-24v2 脚本运行时已直接打印确认。
LangFlow 轨迹稳定性差异**不能**归因于"LangFlow 无 self-conditioning"。
