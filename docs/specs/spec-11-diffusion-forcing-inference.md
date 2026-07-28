# Spec 11 — Diffusion Forcing at Inference (No Training)

**Type**: code modification + inference experiment  
**Priority**: medium (exploratory; validate before deciding to train)  
**Session**: 可与 spec-06 (Idea A) 并行（都是 inference-only，无训练依赖）  
**Output**: `results/elf/df_inference/`

---

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：将"per-position 承诺状态"转化为推理时的位置级 t_i 调度，验证对生成质量的影响。**

EXP-16 表明在 t=0.50 时：73.4% 的位置已 committed_correct（已锁定正确 token），15.1% 的位置 committed_wrong（已锁定错误 token），11.4% uncommitted（仍在探索）。这一分析揭示了一个推理时机会：

**传统 ODE 所有位置共享同一 t**。但如果我们把"已承诺正确"的位置视为"已确定"（它们的 x̂_t 已经接近 x_clean），让它们不再参与去噪（冻结），就能为"还在探索"的位置提供更干净的上下文。更激进的是，对"已承诺错误"的位置重新注入噪声，让它们有机会重新探索。

**这是 Diffusion Forcing（per-position t_i）思想在推理时的直接应用**，无需额外训练。

**要验证的核心假说**：
- **Variant 1（冻结已承诺正确位置）**：Gen.PPL 随步数减少而改善（更少步骤达到同样质量）
- **Variant 2（重新注入噪声到已承诺错误位置）**：committed_wrong 比例随步数加速减少，最终 Gen.PPL 改善
- **Variant 3（软退火）**：平滑的 per-position 插值效果优于硬阈值

**与其他实验的关系**：
- 依赖 EXP-16 的承诺状态分析（已完成）和 EXP-01 的轨迹数据
- EXP-13 验证了 dec_sc 的 token 特异信息价值，spec-11 是 dec_sc 的推理时动力学版本
- 若 Variant 2 有效，则说明 wrong-committed 问题可以在推理时部分修正（无需重训练）

**当前状态**: 代码已实现（2026-07-20）。`generation_utils.py` 中已添加 `_get_df_entropy()` 和 `_apply_df_step()`，支持 freeze/soft 两种 variant。实验正在 GPU 0 运行（baseline checkpoint, 7 conditions × 256 samples）。

---

## 核心思路

Diffusion Forcing 的本质是**序列中不同位置可以有不同的噪声水平 t_i**。  
ELF 当前每步所有位置共享同一个 t，但 probe 表明位置之间存在极大差异：

| 位置类型 | t=0.50 时比例 | 含义 |
|---------|-------------|------|
| committed_correct | 73.4% | 已锁对，可作为其他位置的 clean context |
| committed_wrong | 15.1% | 卡在错误 token，靠自身轨迹几乎无法修正 |
| uncommitted | 11.4% | 仍在探索 |

**关键动机**：如果把已经 committed_correct 的位置"冻结"（t_i → 1），  
它们就变成了更 clean 的上下文，帮助 uncommitted 位置更快做出正确决策；  
如果对 committed_wrong 的位置**重新注入噪声**（t_i → 低值），  
它们就有机会脱离错误锁定，重新探索。

ELF 已有 `cond_seq_mask` 机制（conditioning token 保持 clean），  
这就是 position-level t_i 混合的原型，我们只是把它扩展到 commitment-guided。

---

## 三个 Variant（从简单到复杂）

### Variant 1 — Commitment Freezing（最简单，零额外 forward）

**思路**：每步 ODE 更新后，对 committed_correct 位置，将 z_{t+1,i} 直接替换为 x̂_{t,i}（predicted clean embedding），等效于令该位置的 t_i = 1。

**实现**（在 `_ode_step` / `_sde_step` 返回前插入）：

```python
if sampling_config.df_freeze_committed:
    # Get per-position entropy (need decoder head)
    _, logits = model_apply_fn(
        {"params": model_params}, 
        jnp.concatenate([x_pred, jnp.zeros_like(x_pred)], axis=-1),
        jnp.ones((z.shape[0],)),   # t=1 for decode branch
        deterministic=True, decoder_step_active=jnp.array(True),
    )
    log_p = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    H = -(jnp.exp(log_p) * log_p).sum(-1)   # [B, L]
    committed = H < sampling_config.df_commit_thresh  # [B, L]
    # Freeze: replace z_next with x_pred at committed positions
    z_next = jnp.where(committed[..., None], x_pred, z_next)
```

**额外 cost**：1 次 decoder head forward（projection only，非 backbone）+ 1 次 t=1 backbone forward → 约 +1 backbone pass/step

**SamplingConfig 新字段**：
```python
df_freeze_committed: bool = False
df_commit_thresh: float = 0.1     # entropy threshold for "committed"
```

---

### Variant 2 — Wrong-Position Re-noising（核心 DF 思路）

**思路**：用 decode branch 识别 committed_wrong 位置（committed 但 decode branch 预测不同），对这些位置重新注入噪声，让它们在下一步重新探索。

**实现**：

```python
if sampling_config.df_renoise_wrong:
    # Pass 1b: decode branch to get teacher prediction
    _, logits_dec = model_apply_fn(
        {"params": model_params},
        jnp.concatenate([x_pred, jnp.zeros_like(x_pred)], axis=-1),
        jnp.ones((z.shape[0],)),
        deterministic=True, decoder_step_active=jnp.array(True),
    )
    # Pass 1 (lin): get student prediction from current x_pred via linear head
    _, logits_lin = model_apply_fn(
        {"params": model_params},
        jnp.concatenate([z, x_pred], axis=-1),   # reuse x_pred as SC
        t_batch,
        deterministic=True, decoder_step_active=jnp.array(True),
    )
    H_lin = -(jax.nn.softmax(logits_lin) * jax.nn.log_softmax(logits_lin)).sum(-1)
    committed = H_lin < sampling_config.df_commit_thresh

    # Wrong-committed: committed but decode branch disagrees
    top1_lin = jnp.argmax(logits_lin, axis=-1)
    top1_dec = jnp.argmax(logits_dec, axis=-1)
    wrong = top1_lin != top1_dec
    wrong_committed = committed & wrong   # [B, L]

    # Re-noise: step wrong-committed positions back to t_renoise
    t_renoise = sampling_config.df_renoise_t    # e.g. 0.5 * current t
    eps_fresh = jax.random.normal(rng_renoise, z.shape)
    z_renoised = t_renoise * x_pred + (1.0 - t_renoise) * eps_fresh
    z_next = jnp.where(wrong_committed[..., None], z_renoised, z_next)
```

**额外 cost**：+2 backbone passes/step（1 decode branch + 1 linear with decoder head）  
可与 Idea A 合并：Idea A 已经做了 decode branch pass，直接复用 logits_dec。

**新字段**：
```python
df_renoise_wrong: bool = False
df_renoise_t: float = 0.5     # renoise to t_renoise * current_t (relative)
df_renoise_strength: float = 1.0  # 0=no renoise, 1=full renoise
```

---

### Variant 3 — Soft Annealing（最平滑，无离散阈值）

**思路**：不用二值 committed/uncommitted，而是用 per-position 置信度软插值。

```python
confidence = 1.0 - H / jnp.log(V)   # [B, L], 0=uncertain, 1=committed
# Soft freeze: interpolate z_next toward x_pred
alpha = confidence * sampling_config.df_soft_alpha   # [B, L]
z_next = (1 - alpha[..., None]) * z_next + alpha[..., None] * x_pred
```

**额外 cost**：仅需 decoder head（得到 H），无额外 backbone

**新字段**：
```python
df_soft_anneal: bool = False
df_soft_alpha: float = 0.5    # max interpolation strength
```

---

## 实现位置

**文件**：`src/utils/sampling_utils.py`

在 `_ode_step` 内，`return z + (t_next - t) * v_pred, x_pred` 之前插入：

```python
# Diffusion Forcing: position-level commitment guidance
if getattr(config, 'df_variant', None):
    z_next_raw = z + (t_next - t) * v_pred
    z_next, x_pred = _df_step(
        model_apply_fn, model_params, z_next_raw, x_pred, t_batch,
        config, cond_seq, cond_seq_mask, rng=rng,
    )
else:
    z_next = z + (t_next - t) * v_pred
return z_next, x_pred
```

`_df_step` 是包含上述三个 variant 逻辑的独立函数，通过 `config.df_variant ∈ {None, 'freeze', 'renoise', 'soft'}` 选择。

---

## 实验设计

### Eval script

```bash
# on new-ncl, from ~/tt_workspace/model/CCLF/CCLF/models/ELF/
CUDA_VISIBLE_DEVICES=4 XLA_PYTHON_CLIENT_MEM_FRACTION=0.8 \
python src/eval.py \
    --config src/configs/training_configs/train_owt_ELF-B.yml \
    --checkpoint embedded-language-flows/ELF-B-owt \
    --sampling_configs_path src/configs/sampling_configs/df_eval.yml \
    --output_dir results/elf/df_inference \
    --num_samples 1000
```

### df_eval.yml

```yaml
# Control
- sampling_method: ode
  num_sampling_steps: [4, 8, 16, 50]
  df_variant: null
  time_schedule: uniform

# Variant 1: freeze committed
- sampling_method: ode
  num_sampling_steps: [4, 8, 16, 50]
  df_variant: freeze
  df_commit_thresh: 0.1
  time_schedule: uniform

# Variant 2: renoise wrong-committed
- sampling_method: ode
  num_sampling_steps: [4, 8, 16, 50]
  df_variant: renoise
  df_commit_thresh: 0.1
  df_renoise_t: 0.5
  time_schedule: uniform

# Variant 3: soft annealing
- sampling_method: ode
  num_sampling_steps: [4, 8, 16, 50]
  df_variant: soft
  df_soft_alpha: 0.5
  time_schedule: uniform

# Idea A + DF renoise (组合：复用 decode branch pass)
- sampling_method: ode
  num_sampling_steps: [4, 8, 16, 50]
  use_decode_sc: true
  df_variant: renoise
  df_commit_thresh: 0.1
  time_schedule: uniform
```

### 超参扫描（在最优 variant 上）

| 超参 | 扫描范围 |
|------|---------|
| `df_commit_thresh` | 0.05, 0.10, 0.20 |
| `df_renoise_t` | 0.3, 0.5, 0.7 |
| `df_soft_alpha` | 0.3, 0.5, 0.7 |

---

## 指标

| 指标 | 获取方式 |
|------|---------|
| Gen.PPL (GPT-2-large) | 每个 config 都评 |
| committed_wrong 比例变化 | 每步记录 `(wrong_committed / total).mean()`，画随 step 的曲线 |
| 生成多样性 (MAUVE / BLEU-self) | 可选 |
| 每步用时 | 确认 overhead 在预期范围内 |

---

## 成功标准

| 结果 | 解读 |
|------|------|
| Variant 2 Gen.PPL < control at same step count | wrong re-noising 有效；DF 思路在 ELF 上成立 |
| committed_wrong 比例随步数加速下降 | re-noising 在逃脱错误锁定 |
| 低步数（4-8 步）提升显著，高步数（50 步）提升小 | DF 在受限 budget 下价值更大 |
| 无任何 variant 有效 | wrong-committed 问题根本上是训练问题，只能靠 spec-07 的 L_KD 解决 |

最后一种结果同样有价值：为训练实验提供了强动机。

---

## 与其他 spec 的关系

```
spec-06 (Idea A)          spec-11 (DF inference)
    ↓ decode branch SC         ↓ decode branch oracle + re-noise
    └──────────────────┬───────────────────────────────────┘
                       ↓
            如果两者都有效：组合 (Idea A SC + DF re-noise)
            如果都无效：     证明 gap 是训练问题 → 优先 spec-07
```

如果 spec-11 Variant 2 有效，且 spec-07 L_KD 也有效，那么 DF 的**训练版本**（per-position t_i forward process）就值得作为 spec-12 去实现。否则不必。

---

## 实验结果（Results）

**状态**: 代码实现完成，实验运行中（2026-07-20）

**已完成的前置实验**：
- EXP-16（per-position commitment timing）：kd-cr 在 t=0.50 时 73.4% committed_correct，15.1% committed_wrong ← 给出了 Variant 2 的目标
- EXP-14（trajectory stability）：83.4% 位置在 ODE 中翻转 ≥5 次，mean last-flip 在 step 27/32 ← 证明了冻结/重噪的必要性
- EXP-13v2：tmin=0.5 gate 下 decode branch 基本无害（baseline PPL=226.4），extra_denoise 同等有效（229.2）→ 确认 decode branch 携带的 token 特异信息在 t≥0.5 段不超过额外 forward 的算力效果

**代码实现状态（2026-07-20）**：
- `src/configs/config.py`：新增 `df_variant`, `df_commit_thresh`, `df_soft_alpha`, `df_t_min` 字段
- `src/utils/generation_utils.py`：新增 `_get_df_entropy()`（full decode branch pass 得到熵）、`_apply_df_step()`（freeze/soft 两种 variant）
- ODE 主循环已集成 DF step（EXP-13 dec_sc block 之后）
- `_build_run_name()` 和 `generation.py` 两个调用点已更新
- sampling config: `src/configs/sampling_configs/spec11_df.yml`（7 conditions）
- eval config: `src/configs/training_configs/eval_spec11_baseline.yml`（baseline checkpoint, 256 samples, seed=42）

**实现说明**：
- 实现的是 Variant 1（freeze）和 Variant 3（soft），而非原 spec 中的 Variant 2（re-noise）
- Variant 2（wrong-committed re-noising）复杂度更高（需两次 backbone pass + 随机数管理），推后实现
- `_get_df_entropy()` 通过完整 decode branch pass（`skip_decoder_logits=False`）计算 token 分布熵，而非 unembed_kernel 近似（两者 geometric space 不同）
- Smoke test（`df_smoke_test.py`）通过：entropy H 分布合理（mean=2.39 nats @ t=0.5），freeze/soft 均产生预期修改

**实验设计**（7 conditions × 32 steps × 256 samples）：

| condition | df_variant | 参数 |
|-----------|-----------|------|
| none | none | control |
| freeze_03 | freeze | thresh=0.3 |
| freeze_05 | freeze | thresh=0.5 |
| freeze_10 | freeze | thresh=1.0 |
| soft_03 | soft | alpha=0.3 |
| soft_05 | soft | alpha=0.5 |
| soft_07 | soft | alpha=0.7 |

**结果**：见下方。

---

## 实验结果（Results）— COMPLETED 2026-07-20

### spec-11v1（df_t_min=0.0，无 gate）

**状态**: 已完成，存在严重退化问题。

| condition | df_variant | 参数 | PPL | degen% | mean_entropy |
|-----------|-----------|------|-----|--------|-------------|
| none | — | — | 127.76 | 0.0% | 4.147 |
| freeze_0.3 | freeze | thresh=0.3 | 34.4 | 2.3% | 3.333 |
| freeze_0.5 | freeze | thresh=0.5 | 26.6 | 9.8% | 3.045 |
| freeze_1.0 | freeze | thresh=1.0 | 19.6 | 48.8% | 2.521 |
| soft_0.3 | soft | alpha=0.3 | 13.8 | **100%** | 1.065 |
| soft_0.5 | soft | alpha=0.5 | 14.3 | **100%** | 1.279 |
| soft_0.7 | soft | alpha=0.7 | 12.8 | **100%** | 1.425 |

**根本原因**：与 EXP-13 decode mode 相同机制。在早期 ODE 步骤（t < 0.7），x_pred 还未收敛，decode branch 在 t=1 处对噪声 x_pred 给出"自信"的 token 预测（低熵）。DF 将 z 拉向这些虚假预测 → 正反馈崩溃 → 所有 soft 条件 100% 退化，freeze_1.0 有 48.8% 退化。

**文本示例（soft_0.7，PPL=12.8，100% 退化）**：`"Year 2020202020202020202020...——————————————————..."`
**文本示例（freeze_0.3，PPL=34.4，2.3% 退化）**：`"If the universe is a future, or a future a future, or a real existence?..."` — 部分连贯但重复。

---

### spec-11v2（df_t_min=0.7，只在 t≥0.7 最后 30% 应用 DF）

**状态**: 已完成，无退化，有小幅改善。

| condition | df_variant | 参数 | PPL | degen% | Δ vs none |
|-----------|-----------|------|-----|--------|-----------|
| none | — | — | 127.76 | 0.0% | — |
| freeze_0.3 | freeze | thresh=0.3 | 123.15 | 0.0% | −3.6% |
| **freeze_0.5** | freeze | thresh=0.5 | **121.28** | 0.0% | **−5.1%** |
| freeze_1.0 | freeze | thresh=1.0 | 131.11 | 0.0% | +2.6% |
| **soft_0.3** | soft | alpha=0.3 | **121.83** | 0.0% | **−4.6%** |
| soft_0.5 | soft | alpha=0.5 | 125.63 | 0.0% | −1.7% |
| soft_0.7 | soft | alpha=0.7 | 130.12 | 0.4% | +1.8% |

**结论**：
- tmin=0.7 gate 完全消除退化（0% degen），验证了 v1 的根因分析
- **freeze_0.5 最优**（−5.1% PPL）：在 t≥0.7 阶段，entropy<0.5 nat 的 committed 位置冻结后，有小幅 PPL 改善
- **soft_0.3 次优**（−4.6% PPL）：置信度引导的软插值效果相近
- **高阈值/高 alpha 有害**（freeze_1.0 +2.6%，soft_0.7 +1.8%）：冻结/拉取过度，破坏 ODE 轨迹
- 改善幅度（~5%）较 EXP-13v2 的 decode_shuffled 效果（~12%）小，且依赖额外 decode branch forward

**文本质量**：所有 tmin=0.7 条件均产生连贯英文，与 control 无明显差异。

**数据文件**：
- v1: `outputs/spec11_baseline/ode-steps32-cfg1-ts_uniform-{variant}-uncond/`
- v2: `outputs/spec11v2_baseline/ode-steps32-cfg1-ts_uniform-{variant}-uncond/`

---

## EXP-31/31b — kd_cr 和 kd2 checkpoint（2026-07-21 完成）

**EXP-31 (kd_cr, GPU 4, seed=123)** / **EXP-31b (kd2, GPU 2, seed=456)**

### PPL 完整对比表

| condition | baseline | kd_cr | kd2 |
|-----------|----------|-------|-----|
| none | 127.76 | 331.92 | 282.52 |
| freeze_0.3 | 123.15 (−3.6%) | 422.03 (+27.1%) ✗ | 260.02 (−8.0%) ✓ |
| freeze_0.5 | **121.28 (−5.1%)** | 426.03 (+28.4%) ✗ | 219.19 (−22.4%) ✓ |
| freeze_1.0 | 131.11 (+2.6%) | 475.64 (+43.3%) ✗ | **144.42 (−48.9%)** ✓ |
| soft_0.3 | 121.83 (−4.6%) | 389.89 (+17.5%) ✗ | 230.27 (−18.5%) ✓ |
| soft_0.5 | 125.63 (−1.7%) | 412.85 (+24.4%) ✗ | 160.97 (−43.0%) ✓ |
| soft_0.7 | 130.12 (+1.8%) | 448.33 (+35.1%) ✗ | 167.27 (−40.8%) ✓ |

**关键发现**：
- baseline：温和有益（freeze_0.5 最优，−5.1%）
- kd_cr：**全面恶化**（+17~+43%）——kd_cr 无 dec_sc 时已生成多语言 artifacts
- kd2：**全面大幅改善**（−8~−49%），freeze_1.0 最优（144.42，−48.9%）

---

## EXP-32 — kd2 step-count sweep（2026-07-21 完成）

**步数扩展性（kd2，GPU 4，seed=456）**

| steps | none | freeze_1.0 | Δ |
|-------|------|-----------|---|
| 8 | 688.11 | 615.30 | −10.6% |
| 16 | 602.64 | 486.80 | −19.2% |
| 32 | 282.52 | 144.42 | **−48.9%** |

DF 增益超线性扩展：步数越多，tmin=0.7 门控区间内的冻结步数越多，累积效应越强。

---

## EXP-33/34 — dec_sc × DF 交互（2026-07-21 完成）

**EXP-33 (kd_cr + dec_sc)** / **EXP-34 (kd2 + dec_sc)**

### EXP-33：kd_cr + dec_sc + DF（GPU 6，seed=123）

| condition | PPL | Δ vs none+dec_sc |
|-----------|-----|-----------------|
| none + dec_sc | **98.12** | — |
| freeze_0.3 + dec_sc | 152.84 | +55.8% ✗ |
| freeze_0.5 + dec_sc | 144.54 | +47.3% ✗ |
| freeze_1.0 + dec_sc | 179.22 | +82.7% ✗ |
| soft_0.3 + dec_sc | 127.09 | +29.5% ✗ |
| soft_0.5 + dec_sc | 130.98 | +33.5% ✗ |
| soft_0.7 + dec_sc | 146.88 | +49.7% ✗ |

dec_sc 将 kd_cr PPL 332→98（+）；DF 仍全面有害（kd_cr DF 失败是**根本性的**）。

### EXP-34：kd2 + dec_sc + DF（GPU 2，seed=456）

| condition | PPL | Δ vs none+dec_sc |
|-----------|-----|-----------------|
| none + dec_sc | **72.86** | — |
| freeze_0.3 + dec_sc | 148.88 | +104.3% ✗ |
| freeze_0.5 + dec_sc | 159.17 | +118.5% ✗ |
| freeze_1.0 + dec_sc | 192.01 | +163.5% ✗ |
| soft_0.3 + dec_sc | 126.04 | +73.0% ✗ |
| soft_0.5 + dec_sc | 164.19 | +125.3% ✗ |
| soft_0.7 + dec_sc | 189.72 | +160.5% ✗ |

kd2 + dec_sc alone（72.86）远优于 kd2 + DF alone（144.42）；加 DF 后**完全逆转**，全部变差。

**核心结论**：dec_sc 与 DF **竞争而非互补**。dec_sc 在每步提供位置特异性纠错；DF 冻结固化嵌入，移除了 dec_sc 赖以精炼的动力学。推理时最优策略：单选 dec_sc（对所有 checkpoint 最优）。

### 三维对比矩阵

| checkpoint | 无修改 | +DF 最优 | +dec_sc alone | +dec_sc+DF 最优 |
|-----------|-------|---------|--------------|----------------|
| baseline | 127.76 | 121.28 (−5.1%) | ~81.6* | EXP-35 中 |
| kd_cr | 331.92 | 331.92 (无效) | **98.12** | 127.09 (+29.5%) ✗ |
| kd2 | 282.52 | 144.42 (−48.9%) | **72.86** | 126.04 (+73%) ✗ |

---

### 对论文的影响（更新版）

1. **DF 效果的 checkpoint 依赖性**：相同操作在 kd2 上 −49%，kd_cr 上 +43%——差异来自 kd2 承诺质量高、kd_cr 无 dec_sc 时本就退化
2. **超线性步数扩展**：kd2 DF 在更多步数时有更大收益，是实用的计算-质量 tradeoff 工具
3. **dec_sc 与 DF 互斥**：两者不互补，推理时选择 dec_sc alone 对所有 checkpoint 最优
4. **kd2 dec_sc 最强**：PPL=72.86，超越 baseline+dec_sc 量级（从 282 出发降到 73，跨越 checkpoint 质量差距）
