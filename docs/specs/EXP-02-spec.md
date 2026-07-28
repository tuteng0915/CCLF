# EXP-02 Spec — Corrected LangFlow Comparison

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：修正比较基准，避免"苹果与橙子"的对比。**

论文目前的 ELF vs LangFlow 比较存在一个根本性的不对等问题：
- **ELF 侧**：使用 x̂_t（backbone 去噪预测），这是模型对 x_clean 的"猜测"。
- **LangFlow 侧**：使用 z_t（原始含噪输入），这只是输入信号，不是任何预测。

这相当于把 ELF 的"答案"与 LangFlow 的"输入题目"做比较——当然 ELF 的 G(t) 会看起来更高更早。

**要验证的核心假说**：
- 如果用 LangFlow 的 native 词汇后验 a_t = E^T softmax(LM_head(h_t)) 来计算 G(t)（即 LangFlow 真正的"预测"），与 ELF 的 G(x̂_t) 比较，ELF 的早期承诺优势是否仍然成立？
- 如果 LangFlow-a G(t) 远高于 LangFlow-z G(t)，说明原比较大幅低估了 LangFlow 的承诺速度，论文的跨模型比较需要大幅修正。

**与其他实验的关系**：这是 EXP-01 的跨模型扩展。EXP-01 验证 ELF 内部的承诺测量是否真实，EXP-02 验证 ELF vs LangFlow 的比较是否公平。

**当前状态（COMPLETED）**：已完成 LangFlow 的 native posterior 探测。结果见下方。

---

**Goal:** The current LangFlow comparison in the paper probes G(t) on z_t (the noisy input),
not on LangFlow's native token posterior. ELF uses x̂_t (the denoised prediction), which is
a fundamentally different signal. This asymmetry needs to be corrected.

**Model:** LangFlow checkpoint — check `experiments/probe_langflow/` for existing probe scripts.

---

## Problem

Current code (`probe_geo_langflow.py`):
- ELF: computes G(t) on x̂_t (backbone denoised output) — **correct**
- LangFlow: computes G(t) on z_t (raw noisy input) — **wrong**

LangFlow has its own vocabulary head that maps hidden states to token distributions.
The correct comparison is: ELF's G(x̂_t) vs LangFlow's G(a_t) where a_t = E^T softmax(LM_head(h_t)).

---

## Step 1: Understand LangFlow's forward pass

File: `models/LangFlow/langflow/model.py`

Find: the LangFlow forward function. Identify:
- `h_t`: hidden state at time t (the denoised hidden representation)
- `LM_head`: the vocabulary head applied to h_t
- `a_t = E^T softmax(LM_head(h_t) / tau)`: the expected embedding (lexical anchor)

---

## Step 2: Modify probe_geo_langflow.py

Current behavior: feeds z_t (noisy) directly to vocab head.

New behavior: feed z_t through LangFlow backbone to get h_t, then apply LM_head.

```python
# OLD (wrong):
cosine_sims = compute_cosine_sim(z_t, E_norm)  # ← using noisy z_t

# NEW (correct):
h_t = langflow_backbone(z_t, t)  # run backbone
logits = langflow_lm_head(h_t)    # vocabulary head
p_t = softmax(logits / tau)
a_t = E.T @ p_t                   # lexical anchor (expected embedding)
cosine_sims = compute_cosine_sim(a_t, E_norm)  # ← using denoised a_t
```

---

## Step 3: Comparison protocol

Compute at matched t values {0.05, 0.10, ..., 1.00}:
1. **ELF**: G(t) on x̂_t (current approach — correct)
2. **LangFlow-z**: G(t) on z_t (current approach — wrong, keep as reference)
3. **LangFlow-h**: G(t) on h_t (backbone output, no vocab head)
4. **LangFlow-a**: G(t) on a_t (lexical anchor — most comparable to ELF's x̂_t)

---

## Decision rule

- If LangFlow-a G(t) ≈ LangFlow-z G(t): current comparison was OK despite wrong metric; add footnote.
- If LangFlow-a G(t) > LangFlow-z G(t) by >10pp at t<0.5: ELF's early commitment advantage is OVERSTATED.
  Revise paper claim: "ELF's x̂_t shows earlier cosine alignment than LangFlow's noisy z_t;
  the corrected comparison using LangFlow's native token posterior shows a gap of Xpp instead of Ypp."
- If LangFlow-a G(t) < LangFlow-z G(t): earlier noise had fortuitous alignment; ELF advantage is understated.

---

## Effort: 2–3 days (understanding LangFlow architecture + modifying probe)

## Files to modify
- `experiments/probe_langflow/probe_geo_langflow.py`

## Files to reference
- `models/LangFlow/langflow/model.py` — find backbone forward and vocab head
- `experiments/probe_langflow/` — check existing scripts for data loading patterns

---

## 实验结果（Results）

**状态**: COMPLETED（2026-07-18）

**原始结果文件**：`/tmp/exp02_langflow.log`（或参见 5_experiments.tex EXP-02 注释块）

**关键数据**：

| t 值 | LangFlow native posterior G(t) |
|------|-------------------------------|
| 0.10 | 低（未记录确切值）             |
| 0.30 | 3.6%                          |
| 0.70 | 10.9%                         |
| 承诺时刻 | t ≈ 0.916（首次 G > 50% 阈值） |

**对比 ELF（kd-cr）**：
- ELF kd-cr 在 t=0.30 达到 G ≈ 89.3%（EXP-12 decode path），或 by-t=0.30 累积承诺率 ≈ 90.2%（EXP-16）
- LangFlow native posterior 在 t=0.70 仅 10.9%，承诺时刻比 ELF 晚约 0.35t 单位

**结论**：修正后的比较（LangFlow native posterior vs ELF x̂_t）仍然表明 ELF 承诺早得多。但需注意：
1. LangFlow-a 的 G(t) 仍低于 LangFlow-z（说明 native posterior 并未"更早"—实际上更晚）
2. 两个模型的 G(t) 定义不完全等价（ELF 用 decode path，LangFlow 用 softmax posterior）

**论文使用建议**：可在 §4 添加脚注说明比较对称性；主要结论不变（ELF 承诺显著早于 LangFlow）。

---

## ⚠️ 方法论问题 & 待修正 TODO（2026-07-21 审查）

### 问题 1：a_t 路径不必要且可能扭曲结果

当前实现：softmax(ℓ_t) → a_t = E^T x_t → cosine-NN → token

**正确做法**：LangFlow 最直接的 token metric 是 argmax_v x_t(v)（native posterior top-1），不需要回投影到 a_t。a_t 可作为辅助分析但不应作为主要指标。

### 问题 2：使用 kd-cr ELF，应使用 baseline

跨模型比较应用 baseline ELF（非 KD 训练），避免因 KD 本身的影响污染跨模型比较。

### 问题 3：曲线不完整，缺乏置信区间

当前只有 3 个稀疏 t 值，缺少多个噪声 seed 的 CI。

### 问题 4：有效 token mask 处理不一致

两个模型的 padding / special token 处理方式不同，需要统一 mask 以确保指标可比。

### 修正 TODO

- [ ] **P0**：将 LangFlow 主指标改为 native posterior argmax top-1（而非 a_t cosine-NN）
- [ ] **P0**：用 baseline ELF 重跑，报告 ELF native decoder top-1
- [ ] **P1**：补全 t ∈ {0.05,0.10,...,1.00} 完整曲线，N≥8 noise seeds，加 bootstrap CI
- [ ] **P1**：对比报告四条曲线：ELF-probe、ELF-decode、LangFlow-native-posterior、LangFlow-probe
- [ ] **P2**：统一 valid token mask（去掉 padding），确保两模型分母一致
