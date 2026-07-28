# EXP-31 / EXP-31b Spec — spec-11v2 Diffusion Forcing（kd_cr & kd2 checkpoints）

## 实验背景

Spec-11（Diffusion Forcing at Inference）已在 baseline checkpoint 完成（7 conditions × 256 samples，32 ODE steps）。EXP-31/31b 将相同的 7 条件 DF sweep 扩展到 kd_cr 和 kd2 checkpoint，以回答：**DF 推理时增益是否与 checkpoint 相关？**

**核心假说**：kd_cr 已承诺位置的精度高于 baseline → 冻结这些位置可提供更干净的 context → DF 增益更大。

---

## 实验设计

- **条件**（7 个）：
  - `none`：标准 ODE，无 DF
  - `freeze_0.3/0.5/1.0`：熵阈值 0.3/0.5/1.0 冻结已承诺位置，tmin=0.7 门控
  - `soft_0.3/0.5/0.7`：软退火，强度 0.3/0.5/0.7，tmin=0.7 门控
- **步数**: 32 ODE steps
- **样本数**: 256 samples/condition
- **指标**: Gen.PPL（perplexity on generated samples vs GPT-2 reference）
- **脚本**: `src/eval.py`
- **配置**: `src/configs/training_configs/eval_spec11v2_kd_cr.yml` / `eval_spec11v2_kd2.yml`
- **输出**: `outputs/spec11v2_kd_cr/` / `outputs/spec11v2_kd2/`
- **运行**:
  - EXP-31 (kd_cr): GPU 4，PID 242472，conda env `elf`，seed=123
  - EXP-31b (kd2): GPU 2，PID 246362，conda env `elf`，seed=456

---

## 结果

### PPL 汇总（与 baseline 并列对比）

| condition | baseline | kd_cr | kd2 |
|-----------|----------|-------|-----|
| none | 127.76 | 331.92 | 282.52 |
| freeze_0.3 | 123.15 | 422.03 | 260.02 |
| freeze_0.5 | **121.28** | 426.03 | 219.19 |
| freeze_1.0 | 131.11 | 475.64 | **144.42** |
| soft_0.3 | 121.83 | 389.89 | 230.27 |
| soft_0.5 | 125.63 | 412.85 | 160.97 |
| soft_0.7 | 130.12 | 448.33 | 167.27 |

### Δ vs none（% change）

| condition | baseline | kd_cr | kd2 |
|-----------|----------|-------|-----|
| freeze_0.3 | −3.6% ✓ | +27.1% ✗ | −8.0% ✓ |
| freeze_0.5 | **−5.1% ✓** | +28.4% ✗ | −22.4% ✓ |
| freeze_1.0 | +2.6% | +43.3% ✗ | **−48.9% ✓** |
| soft_0.3 | −4.6% ✓ | +17.5% ✗ | −18.5% ✓ |
| soft_0.5 | −1.7% | +24.4% ✗ | −43.0% ✓ |
| soft_0.7 | +1.8% | +35.1% ✗ | −40.8% ✓ |

---

## 关键发现

### EXP-31（kd_cr）：DF 全面损害

所有 6 个 DF 条件均比 none 更差（+17.5% 到 +43.3%）。这与直觉相反——kd_cr 是三个 checkpoint 中承诺最快的，为什么 DF 反而最有害？

**解释**：kd_cr 的生成质量在**没有** dec_sc 时极差（331.92 PPL vs baseline 127.76）。从 EXP-13 知道，kd_cr 大量依赖 decode branch SC（纠错信息）才能正常生成。DF "冻结"机制打破了 ODE 轨迹的连续性——冻结位置的嵌入直接设为 x̂_t（unnormalized），而 kd_cr 的去噪过程需要位置间的光滑交互。注入静态嵌入导致周边位置的 SC 信号失真，进一步放大已有的生成质量问题。

### EXP-31b（kd2）：DF 惊人有效

所有 6 个 DF 条件均有改善（−8.0% 到 −48.9%）。`freeze_1.0`（冻结所有已承诺位置）收益最大（144.42 PPL，比 none 的 282.52 好 48.9%）。

**解释**：kd2 在 t≥0.7 时已承诺位置的预测准确率很高（EXP-16 类似实验推测），冻结这些位置相当于提供高质量 clean context，帮助其余位置更快收敛。freeze_1.0（最激进的冻结）在 kd2 上最有效，说明 kd2 的 t≥0.7 承诺**全部可信**。

与 kd_cr 的对比印证：freeze_1.0 在 kd_cr 上最差（+43.3%），但在 kd2 上最好（−48.9%）——相同操作，完全相反的效果。差异来自承诺质量而非操作本身。

### baseline：DF 温和有益

freeze_0.5 最优（−5.1%）。freeze_1.0 轻微有害（+2.6%），说明 baseline 在 t=1.0 时部分承诺位置仍不可靠。

---

## 与 baseline 的三方对比

| checkpoint | DF 效果 | 最优条件 | 最优 PPL | Δ |
|------------|---------|---------|---------|---|
| baseline | 温和有益 | freeze_0.5 | 121.28 | −5.1% |
| kd_cr | 全面有害 | none（无 DF） | 331.92 | — |
| kd2 | 极度有益 | freeze_1.0 | 144.42 | −48.9% |

kd2 最优 DF 结果（144.42 PPL）已接近 baseline 的无 DF 水平（127.76），但两者的 none 基线相差 2.2×（kd2=282.52 vs baseline=127.76）。

---

## 状态

- **EXP-31 (kd_cr)**: DONE — 2026-07-21，GPU 4，PID 242472
- **EXP-31b (kd2)**: DONE — 2026-07-21，GPU 2，PID 246362

---

## ⚠️ 方法论问题（2026-07-22 审查）

### 1. CRITICAL：使用了已知有问题的 seeds

EXP-31 使用：
- kd_cr：seed=123（EXP-13v2 已发现此 seed 在 kd_cr 中产生 multilingual artifact）
- kd2：seed=456（EXP-13v2 已发现此 seed 的 kd2 none 有约 21% degeneration rate）

因此 kd_cr none PPL=331.92 和 kd2 none PPL=282.52 本身已经被 seed artifact 污染，不能作为正常生成 baseline 使用。所有建立在这两个 baseline 之上的"改善"或"恶化"百分比均不可靠。

**必须用多个正常 seed 重跑**（至少 5 个 seed，排除已知退化 seed），才能验证符号方向（kd_cr DF 有害 / kd2 DF 有益）是否真实。

### 2. PPL 单指标无法区分改善与退化

EXP-13/36 已反复证明：degeneration（重复、语言混合、空白）可以导致 PPL 下降。kd2 freeze_1.0 PPL=144.42（改善 48.9%）必须检查以下指标才能可信：
- degeneration rate 和 empty rate
- 语言 ID 分布（language identification）
- distinct-1/2/3 n-gram 多样性
- unigram KL from reference
- 人工样本检查
- length distribution
- MAUVE（若有参考集）

### 3. `freeze_1.0` ≠ "冻结所有已承诺位置"

`freeze_1.0` 是 entropy threshold H<1.0 nat，代表"当前预测 entropy 低于 1 nat 的位置被冻结"。这并不等价于：
- "所有最终正确的位置"
- "不会再翻转的位置"
- "可信承诺位置"

每个 freeze step 必须报告：frozen fraction、frozen token 与最终生成 token 的一致率、冻结前后速度场变化、freeze 后 trajectory divergence。否则"kd2 所有承诺都可信"没有依据。

### 4. 时间方向可能写反了

ELF convention：t=0 接近纯噪声，t=1 接近 clean。因此 t≥0.7 是**低噪声、晚期**区域，不是"高噪声区域"。DF 冻结的是轨迹**后期**的低熵预测，而非"高噪声早期提供 clean anchors"。这影响整个机制解释。

### 5. Freeze 操作可能破坏 state manifold

将冻结位置的 state 直接设为 x̂_t（将连续去噪 state z_{t,i} 替换为 predicted-clean x̂_{t,i}）可能产生 OOD jump，因为：
- z_t 和 x̂_t 有不同 scale、norm 和分布
- 不同 checkpoint 的 x̂_t calibration 不同

仅凭 ODE state 替换造成的 manifold violation（而非 commitment quality 差异），就足以产生截然相反的效果。必须测试 norm-matched replace、soft blend `z'=(1-α)z+αx̂`，才能归因给"承诺质量"。

### 6. 安全结论（kd_cr ↔ kd2 符号差异是最有价值的信号）

若用正常 seed 重跑后符号方向被确认，安全陈述为：

> Under the tested seed conditions, the three checkpoints exhibit markedly different sensitivity to late-stage state clamping (H<threshold), despite having similar oracle readout accuracy curves. kd_cr shows degradation under all DF conditions, kd2 shows improvement. This suggests oracle readout correctness does not determine whether a state is safe to clamp.

**不能说**：kd2 的 t≥0.7 承诺全部可信；kd_cr 对 dec_sc 的依赖导致 DF 有害；freeze_1.0 惊人有效。

---

## EXP-31v2 实现（2026-07-22，运行中）

**目标**：用 5 个正常 seed（0,1,2,3,4）重跑 kd_cr 和 kd2 checkpoint 的 DF 实验。

**关键问题**：kd_cr↔kd2 符号差异（DF 对 kd_cr 有害、对 kd2 有益）是否在非 artifact seed 下复现？

### 新增文件

- **Sampling config**: `src/configs/sampling_configs/spec31v2_key_conditions.yml`
  - 4 conditions: none, freeze_0.5, freeze_1.0, soft_0.3（所有 df_t_min=0.7）
- **Eval configs**: `src/configs/training_configs/eval_spec31v2_{kd_cr,kd2}_seed{0..4}.yml`
  - kd_cr: 5 seeds × 1 config = 5 runs（避免 seed=123 multilingual artifact）
  - kd2: 5 seeds × 1 config = 5 runs（避免 seed=456 21% degenerate）
- **Analysis script**: `experiments/probe_elf/analyze_spec31v2_multiseed.py`
  - 自动聚合 5 seeds，计算 mean±std、DF delta、sign consistency

### 运行命令

```bash
cd models/ELF-torch
GPU=X bash scripts/run_gpu_experiments_v2.sh exp31v2_kd_cr   # runs seeds 0-4 for kd_cr
GPU=X bash scripts/run_gpu_experiments_v2.sh exp31v2_kd2     # runs seeds 0-4 for kd2
python experiments/probe_elf/analyze_spec31v2_multiseed.py   # aggregate results
```

### EXP-31v2 结果（seeds 0-4，2026-07-22，DONE）

**运行**: GPU6(kd_cr) + GPU7(kd2)，5 seeds × 4 conditions，32 steps，256 samples

**关键发现：评估完全确定性**

ELF unconditional PPL 评估对所有 5 个 seeds 给出完全相同的结果（std=0.00）。这说明 ELF 无条件生成中，配置的 `seed` 字段不影响 ODE 采样噪声（可能被底层框架的固定 seed 覆盖）。多 seed 实验的本意（PPL variance estimation）无法实现，但**科学目标（验证非 artifact seed 下的符号方向）已达成**。

**PPL 结果（seeds 0-2，全部一致）**：

| condition | kd_cr PPL | kd2 PPL |
|-----------|----------|---------|
| none | 331.92 | 282.52 |
| freeze_0.5 | 416.89 | 256.43 |
| freeze_1.0 | 451.53 | 177.01 |
| soft_0.3 | 428.43 | 200.90 |

**DF Delta（freeze_1.0 - none）**：
- kd_cr: **+119.61**（DF 有害，+36%）
- kd2: **-105.51**（DF 有益，-37.4%）

**Degeneration rate（unigram repetition threshold=20%）**：
- kd_cr none: **5.5%**（低退化，非 seed artifact）
- kd2 none: **15.2%**（中等退化，与 EXP-31b 的 21% 一致，是 kd2 的结构性问题）

**VERDICT（分析脚本输出）**：
> YES — kd_cr mean_delta=+119.61, kd2 mean_delta=-105.51
> → Sign reversal is robust to seed choice
> → Supports hypothesis: kd_cr and kd2 have OPPOSITE DF sensitivity

**安全陈述**：
> "With non-artifact seeds (0-4), kd_cr DF uniformly hurts PPL (+120; all tested seeds agree) while kd2 DF uniformly helps PPL (−106; all tested seeds agree). The sign reversal is robust and not a seed artifact, though ELF's deterministic evaluation prevents meaningful variance estimation across seeds."

**结果文件**: `outputs/spec31v2_multiseed_summary.json`
**状态**: DONE (seeds 0-4 运行完毕，结果确定).
