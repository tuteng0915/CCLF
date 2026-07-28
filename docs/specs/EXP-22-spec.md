# EXP-22 Spec — LangFlow Per-Position Commitment Timing (LangFlow analog of EXP-16)

## 实验背景与动机

**在整体框架中的地位：比较 ELF 与 LangFlow 在位置级别承诺异质性上的差异，验证 ELF 的 CCLF 特征是否独特。**

EXP-16 在 ELF 上发现：
- kd-cr 在 t=0.50 时：73.4% committed_correct，15.1% committed_wrong，11.4% uncommitted
- baseline 在 t=0.50 时：承诺率远低，且 19% 的位置"永不承诺"（在整个去噪过程中从未正确解码）
- 位置承诺的时序高度异质：不同位置在不同 t 值下承诺，而非同步

**核心问题**：LangFlow 是否也表现出类似的 per-position 承诺异质性？

两种可能：
1. **ELF 特有**：LangFlow 的位置承诺更同步（所有位置在相近的 t 处同时转变），这是 ELF 的 embedding-space diffusion 特有的现象。
2. **连续扩散 LM 通性**：LangFlow 也展示类似的异质性，说明这是连续扩散语言建模的普遍现象，而非 ELF 的特例。

如果 LangFlow 异质性更低：ELF 的 CCLF（Contextual Commitment in Language Flows）特性可能源自其独特的 T5-to-GPT2 cross-architecture 设计。

如果 LangFlow 异质性相当：说明 CCLF 是连续扩散 LM 的共同特征。

**与其他实验的关系**：
- EXP-16（ELF per-position timing）：直接对应
- EXP-21（LangFlow probe gap）：互补——EXP-21 问 head 能否提取信息，EXP-22 问信息何时到达

**当前状态**：COMPLETED（2026-07-20）。结果见下方。

---

## 方案设计

### 测量协议

LangFlow 的 native posterior top-1 accuracy（`top1_gt_decoder`）对应 ELF 的 `decoder_rec1`（native decode path）。

用相同的 oracle protocol（固定 ε，扫描 t）来追踪每个位置的"承诺状态"：

```
committed_correct(i, t) = (native_top1(i, t) == gt_id[i]) AND (H(p_{i,t}) < thresh)
committed_wrong(i, t)   = (native_top1(i, t) != gt_id[i]) AND (H(p_{i,t}) < thresh)
uncommitted(i, t)       = H(p_{i,t}) >= thresh
```

熵阈值 thresh = 1.0 nat（与 EXP-16 使用相同，确保可比性）。

**注意**：这里使用 LangFlow 的 native LM head 概率 p_{i,t}（而非 ELF 的 decode branch）作为"读出器"，因为：
1. EXP-21 将测量是否有更好的读出器（probe gap）
2. 这里我们测量 native head 能观测到的承诺异质性
3. 与 EXP-07 baseline ELF 用 native decode path 而非 decode branch 一致

### 固定 ε oracle protocol（与 EXP-16 对应）

```python
# 对每个序列，固定同一个噪声向量 ε：
eps = torch.randn(L, d)  # 固定，不重新采样
for t_val in np.linspace(0.05, 1.00, 40):
    gamma = gamma_from_t(t_val, gamma_min, gamma_max)
    alpha = sqrt(sigmoid(-gamma))
    sigma = sqrt(sigmoid(gamma))
    z_t = alpha * x_clean + sigma * eps  # 同一 ε，不同 t
    logits = model(z_t, gamma)
    p = softmax(logits)
    H = -sum(p * log(p), dim=-1)  # entropy
    top1 = argmax(p, dim=-1)
    # Record commitment state
```

### 指标

1. **committed_correct_rate(t)**：在 t 时刻的 committed_correct 比例
2. **committed_wrong_rate(t)**：在 t 时刻的 committed_wrong 比例
3. **commitment_time(i)**：每个位置首次达到 committed_correct 的 t 值（如果从未达到则记为 NaN）
4. **never_committed_rate**：在整个 t 范围内从未 committed_correct 的位置比例
5. **last_flip_step**：最后一次从 committed_correct 到其他状态（或反向）的 t 值

### 对比表（计划）

| 指标 | ELF baseline | ELF kd_cr | ELF kd2 | LangFlow (EXP-22) |
|------|-------------|-----------|---------|-----------------|
| committed_correct @ t=0.3 | ~54% | ~90% | ~90% | ? |
| committed_correct @ t=0.5 | ~72% | ~73% | ~73% | ? |
| committed_wrong @ t=0.5 | ~15% | ~15% | ~15% | ? |
| never_committed_rate | 19% | ? | ? | ? |
| mean commit_time (committed positions) | ~0.28 | ~0.20 | ~0.20 | ? |

（ELF 数据来自 EXP-16 和 EXP-07 综合）

---

## 实现计划

**文件**：`experiments/probe_langflow/probe_commitment_langflow.py`（新建）

**参考**：
- EXP-16 实现（如有独立脚本，或 EXP-07b 中的承诺时序分析部分）
- `probe_geo_langflow.py`（data loading 和模型调用模式）

**主要区别**：
- 需要固定 ε（不重复采样）来追踪单个序列的时序
- 需要逐 t 扫描（不是独立 t 的批量）

```python
def probe_commitment_langflow(model, sample, t_grid, gamma_grid, seed, thresh=1.0):
    """Track per-position commitment along fixed-eps oracle path."""
    gt_ids, clean_emb, attn_mask = sample
    L, d = clean_emb.shape

    rng = np.random.default_rng(seed)
    eps = rng.standard_normal((L, d)).astype(np.float32)  # fixed!
    eps_t = torch.from_numpy(eps).to(device)
    x_t = torch.from_numpy(clean_emb).to(device, dtype=torch.float32)

    commitment_states = []  # list of (t, committed_correct, committed_wrong)

    for t_val, gamma in zip(t_grid, gamma_grid):
        alpha = math.sqrt(torch.sigmoid(torch.tensor(-gamma)).item())
        sigma = math.sqrt(torch.sigmoid(torch.tensor(gamma)).item())
        z_t = (alpha * x_t + sigma * eps_t)[None]  # [1, L, d]
        gamma_t = torch.full((1,), gamma, device=device)
        sc = torch.zeros_like(z_t) if self_conditioning else None

        with torch.no_grad():
            out = model(noisy_embeds=z_t, timesteps=gamma_t, x_self_cond=sc, return_dict=False)
        logits = (out[0] if isinstance(out, (tuple, list)) else out)[0].cpu().float().numpy()
        p = softmax_np(logits, tau=1.0)  # [L, V]
        H = -((p * np.log(p + 1e-9)).sum(-1))  # [L] entropy in nats
        top1 = np.argmax(p, axis=-1)  # [L]

        committed = H < thresh
        correct = (top1 == gt_ids)
        commitment_states.append({
            "t": float(t_val),
            "committed_correct": float((committed & correct).mean()),
            "committed_wrong": float((committed & ~correct).mean()),
            "uncommitted": float((~committed).mean()),
        })
    return commitment_states
```

**GPU 需求**：1 GPU，约 45 分钟（64 seqs × 40 t values × 1 forward pass）

**运行命令**：
```bash
CUDA_VISIBLE_DEVICES=X conda run -n elf python experiments/probe_langflow/probe_commitment_langflow.py \
    --checkpoint Continuous-Rivals-Discrete/langflow-owt \
    --n_samples 64 --seq_len 128 --n_t_steps 40 --entropy_thresh 1.0 \
    --out_dir results/exp22_langflow/
```

**预计工作量**：1 天（脚本实现 0.5 天 + 运行 + 分析 0.5 天）

---

## 辅助实验：LangFlow Geometry Null Model（EXP-23，附属）

类似 EXP-04 对 ELF 的分析：向 LangFlow 输入纯高斯噪声，测量 mode_fraction 和 G_null（vs 真实 token）。

ELF 的结论：G_null ≈ 0.17%（几何偏置可忽略，G(t) 不需要修正）。

LangFlow 的 token 嵌入矩阵（GPT-2，50257 词汇）几何特性不同，可能有不同的偏置水平。

这个实验可以与 probe_commitment_langflow.py 合并（在同一脚本中的额外 null 分支），工作量增量很小。

```python
# EXP-23: Null model for LangFlow
z_null = torch.randn_like(z_t)  # pure Gaussian
out_null = model(noisy_embeds=z_null, timesteps=gamma_t, x_self_cond=sc, return_dict=False)
logits_null = ...
p_null = softmax_np(logits_null, tau=1.0)
G_null = float((np.argmax(p_null, axis=-1) == gt_ids).mean())
```

---

## 实验结果（Results）— COMPLETED 2026-07-20

**数据文件**：
- `results/exp22_langflow/commitment_by_t.json`（EXP-22）
- `results/exp22_langflow/null_model.json`（EXP-23）

### EXP-22：LangFlow 位置级承诺时序（thresh=1.0 nat）

| t | committed_correct | committed_wrong | uncommitted | native_top1 | entropy |
|---|-------------------|-----------------|-------------|-------------|---------|
| 0.03 | 0.000 | 0.000 | 1.000 | 0.036 | 7.571 |
| 0.12 | 0.000 | 0.000 | 1.000 | 0.037 | 7.582 |
| 0.33 | 0.000 | 0.000 | 1.000 | 0.039 | 7.545 |
| 0.53 | 0.000 | 0.001 | 0.999 | 0.028 | 6.273 |
| 0.62 | 0.003 | 0.034 | 0.964 | 0.033 | 4.779 |
| 0.73 | 0.021 | 0.062 | 0.917 | 0.077 | 3.872 |
| **0.83** | **0.258** | **0.031** | **0.710** | **0.427** | 2.592 |
| **0.93** | **0.835** | **0.042** | **0.123** | **0.883** | 0.326 |

**对比 ELF**（来自 EXP-16/07）：

| t | ELF kd_cr committed_correct | ELF kd_cr native_top1 | LangFlow committed_correct | LangFlow native_top1 |
|---|------------------------------|----------------------|---------------------------|----------------------|
| 0.20 | ~58% | ~60% | 0.0% | 3.9% |
| 0.30 | ~90% | ~90% | 0.0% | 3.9% |
| 0.50 | ~73% | ~99% | 0.0% | 2.8% |
| 0.70 | ~99% | ~99% | 2.1% | 7.7% |
| 0.83 | — | — | 25.8% | 42.7% |
| 0.93 | — | — | 83.5% | 88.3% |

**关键发现**：

1. **承诺时序差异巨大**：LangFlow 直到 t≈0.70 都几乎 0% committed（ELF kd-cr 在 t=0.20 已达 ~58%）。LangFlow 的承诺"悬崖"出现在 t≈0.83，比 ELF 晚约 0.55t 单位。

2. **committed_wrong 极低**：LangFlow 在 t=0.93 时 committed_wrong=4.2%（vs ELF kd-cr 在 t=0.50 时 15.1%）。LangFlow 几乎没有"锁定错误"的问题——但这是因为它不承诺（uncommitted=100%），而非因为它更准确。

3. **native_top1 早期非零之谜（EXP-23 解释）**：LangFlow 在 t=0.10-0.50 的 native_top1=3-4% 实际上来自几何偏置（G_null=3.85%），而非 backbone 对 token 的真实预测。见 EXP-23 结果。

---

### EXP-23：LangFlow 几何零模型（对比 EXP-04）

| t | LangFlow G_null | LangFlow mode_frac | ELF G_null (EXP-04b) |
|---|----------------|--------------------|--------------------|
| 0.10 | **3.85%** | **70.9%** | 0.17% |
| 0.20 | **3.85%** | **69.7%** | 0.17% |
| 0.30 | **3.63%** | **64.6%** | 0.20% |
| 0.50 | **3.03%** | **69.5%** | 0.17% |
| 0.70 | 1.46% | 17.5% | 0.88% |
| 1.00 | 0.07% | 1.6% | 1.65% |

**关键发现**：

1. **LangFlow 早期 G(t) 被 geometry bias 主导**：纯高斯噪声输入，70% 的位置预测同一个 token（mode_frac=70.9%），G_null=3.85%。这是因为 LangFlow 的 bias skip connection：`logits += c_skip(γ) × z @ E.T`，在高噪声（低 t）时，c_skip 较大，随机 z 与 E 的点积产生几何主导的分布。

2. **对比 ELF**：ELF 的 G_null 全范围 < 0.2%，几何偏置可忽略。LangFlow 早期 G_null（22× ELF）说明 LangFlow 的 skip connection 在低 t 时产生大量几何噪声。

3. **修正 EXP-02 的结论**：LangFlow 在 t≤0.50 的 native_top1（3-4%）几乎完全来自几何偏置，而非 backbone 的 token 承诺。真实 G_native_corrected ≈ G_native - G_null ≈ 0%。这使得 ELF 与 LangFlow 的对比更加鲜明：ELF kd-cr 在 t=0.30 的修正 G≈89%（EXP-04b 确认 ELF G_null≈0.17%），LangFlow 修正 G≈0%。

4. **t=0.70：过渡区**：G_null 降至 1.46%，mode_frac 降至 17.5%，与 native_top1=7.7% 接近。此区域混合了几何偏置和真实 backbone 信号。

---

### 综合结论（EXP-22 + EXP-23）

**LangFlow 的 per-position 承诺模式与 ELF 根本不同**：
- 承诺时刻：LangFlow t*≈0.83，ELF kd-cr t*≈0.20（差距 0.63t 单位）
- 几何偏置：LangFlow G_null=3.85% vs ELF 0.17%（22×）
- Wrong-commitment：LangFlow 几乎没有（< 5%），ELF kd-cr 有 15%

**论文使用建议（当前版本的有效部分）**：
1. EXP-22 结果可用于描述 LangFlow **自身**内部的承诺时序：native posterior 在 t≈0.83–0.93 才快速集中（模型内结论有效）
2. EXP-23 发现 LangFlow 早期 G_null=3.85% 高度集中（mode_frac=70%）——这是关于 skip connection 性质的独立发现，可保留
3. **不能在论文中做 nominal-t 的 ELF–LangFlow 跨模型对比**（见⚠️下方）

---

## ⚠️ 方法论问题（2026-07-22 审查）

### 1. CRITICAL：不能使用 nominal-t 比较 ELF 和 LangFlow（EXP-03 已证明）

EXP-03 已经严格证明，ELF 和 LangFlow 的 nominal t 对应完全不同的 log-SNR 范围。

以下表述**必须从论文和 spec 中删除**：
- ~~"LangFlow 比 ELF 晚 0.63t 单位承诺"~~
- ~~"ELF kd-cr 在 t=0.20 已达 ~58%，LangFlow 到 t=0.83"~~
- ~~"这强化了 ELF 的早期承诺优势"~~

这些重新犯了 EXP-03 已经纠正的错误。跨模型比较只能在 empirical log-SNR 轴、matched corruption difficulty、或各自实际轨迹的归一化进度上进行。

### 2. 熵阈值 H < 1.0 nat 不能跨模型比较

ELF 和 LangFlow 的 vocabulary size（ELF T5-base ≈ 32K，LangFlow GPT-2 ≈ 50K）、calibration、temperature scaling 均不同。相同的 1 nat 熵阈值对应不同的置信水平。至少应同时报告 normalized entropy H / log|V|、top-1 probability、top-1/top-2 margin，而非只用绝对熵阈值。

### 3. "committed_wrong 极低"是选择效应——且早期 committed 位置几乎全错

t=0.93 时 LangFlow committed_correct=83.5%，committed_wrong=4.2%。但此时 uncommitted=12.3%，即总共 ~9.4% 的位置满足 H < 1 nat，其中 83.5/(83.5+4.2) ≈ 95.2% 正确。

这不是"LangFlow 几乎没有锁定错误"的证据——而是 LangFlow 在 t<0.83 几乎没有任何位置达到 H < 1 阈值的必然结果。

**实际计算的 P(ŷ ≠ y | H < 1)**（从 `results/exp22_langflow/commitment_by_t.json` 的 `agg` 字段计算）：

| t | committed_correct | committed_wrong | P(wrong \| H<1) |
|---|:-----------------:|:---------------:|:----------------:|
| 0.52 | 0.0% | ~0.2% | **~100%** |
| 0.60 | ~0.8% | ~9.2% | **~91%** |
| 0.75 | ~11.6% | ~17.0% | **59.7%** |
| 0.80 | ~34.1% | ~8.0% | **18.5%** |
| 0.83 | ~51.3% | ~6.3% | **10.9%** |
| 0.93 | 83.5% | 4.2% | **4.8%** |
| 1.00 | 96.6% | 1.0% | **1.0%** |

**结论**：LangFlow 在首次出现 H < 1 的位置（t≈0.52-0.60）中，**91-100% 是错的**。"committed_wrong 极低"的叙述实际上是被高 t 端（t≥0.83）的数字稀释了。真正的早期 commitment 几乎全部错误。

这进一步否定了"LangFlow 的承诺机制健康"的论断：LangFlow 的 oracle protocol 下，early commitment 意味着 wrong commitment。

### 4. G_corrected ≈ G_native − G_null 数学上不成立

Accuracy 不是对数-softmax 的线性可加量，不能直接相减。纯高斯输入下的 G_null=3.85% 不能直接从 G_native 中"扣除"。

此外 G_null 的高斯输入经过完整网络（包括 skip connection、timestep conditioning），混合了 output geometry、learned prior、skip bias 多种来源，不能仅称为 "geometry bias"。

### 5. EXP-22 模型内结论（可保留的部分）

以下描述在 LangFlow 内部有效（不做跨模型比较）：
- 在 LangFlow oracle protocol 下，native posterior entropy 在 t≈0.83 之前始终 > 1 nat
- 高置信错误（committed_wrong）的绝对比例在整个 t 范围内极低
- Gaussian-null 在早期显示 mode_frac=65–71%，说明 LangFlow skip connection 在高噪声时产生高度集中的 default output
