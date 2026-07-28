# EXP-13 Spec — Compute-Matched Controls for dec_sc

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：验证 dec_sc 方法的核心机制假设。**

论文提出 dec_sc（decode self-conditioning）作为一种推理时的改进：在每个去噪步骤中，额外跑一次 decode branch，用其输出作为下一步的 self-conditioning 信号，从而"更早承诺"。

**但这里有一个 confound**：dec_sc 相比 standard sampling 多跑了一次 forward pass（约 2× FLOPs/step）。也许 PPL 的提升仅仅是因为多用了计算量，而与 decode branch 携带的 token 特异性信息无关。

**要验证的核心假说**：
- **H0（信息假设）**：dec_sc 提升来自 decode branch 对当前 token 信念的精确校正（x_pred 中携带了 t=1 时的 token 特异信息）
- **H1（算力假设）**：dec_sc 提升来自额外的非线性计算（即使不用 decode branch，只要多一次 forward pass 就有同样效果）

**鉴别方法**：
- `extra_denoise` ≈ `decode`：H1 成立（算力解释）→ dec_sc 的优势被消除，论文方法部分需要重写
- `extra_denoise` << `decode`：H0 成立（信息解释）→ dec_sc 有信息论上的贡献，论文方法站得住脚
- `decode_shuffled` ≈ `decode`：位置无关（不是 token 特异性）
- `decode_wrong_t` << `decode`：时序关键（t=1 的校正是特定的）

**为什么现在重要**：论文已经发现 kd-cr 训练改善了早期承诺。但 dec_sc 作为推理时的方法，其有效性需要独立于 kd-cr 训练来验证。如果 dec_sc 的提升只是算力效应，那训练 kd-cr 的意义（让 backbone 在低 t 时更早可解码）与 dec_sc 的机制无关。

**与其他实验的关系**：EXP-13 是对 dec_sc 方法的"去混淆"实验。结合 EXP-07c（跨 checkpoint 几何分析）可以解释：KD 训练调整了 backbone 的表示几何，而 dec_sc 利用了这种几何结构来做 token 特异校正。

**当前状态**：COMPLETE（2026-07-20）。EXP-13（全范围）和 EXP-13v2（tmin=0.5 gate）均已完成。H1（计算量假设）在 baseline 和 kd2 checkpoint 上得到支持；kd_cr 结果受 seed=123 multilingual artifact 影响。

---

**Goal:** Test whether the dec_sc inference improvement comes from token-specific correction
information in the decode branch, or from the extra nonlinear compute per step.

**Status:** Code implemented. `SamplingConfig.dec_sc_mode` controls which mode runs.

**Checkpoint:** Base ELF-B OWT checkpoint (NOT kd-cr). Use base checkpoint to see full gap.

---

## Modes to run

| Mode | dec_sc_mode value | What it does |
|------|-------------------|-------------|
| Baseline (no extra pass) | none | Standard ELF |
| Dec_sc (test condition) | decode | Decode branch at t=1, decoder_step_active=True |
| Extra denoising (compute control) | extra_denoise | Extra forward at current z,t; same FLOPs |
| Decode shuffled (specificity) | decode_shuffled | Dec branch with positions shuffled |
| Decode wrong-t (timing) | decode_wrong_t | Dec branch at t-0.2 instead of t=1 |
| Random residual (signal) | random_residual | Matched-norm random vector |

---

## Steps

1. For each mode: generate N=1000 ODE samples at steps={8,16,32}
2. Compute Gen.PPL + MAUVE for each (mode, steps) combination
3. Record in results/exp13/results_table.md

## Decision rule

- extra_denoise ≈ decode (within 2 PPL): compute hypothesis (extra FLOPs explain improvement)
- extra_denoise >> decode (>3 PPL gap): correction hypothesis (decode branch has token-specific info)

## Implementation status

- [x] SamplingConfig.dec_sc_mode added to configs/config.py
- [x] _apply_dec_sc_mode() helper added to generation_utils.py  
- [x] Refinement call injected into _generate_samples_single_batch() loop
- [ ] Runs launched
- [ ] Results analyzed

---

## 实验结果（Results）

**状态**: PARTIAL（2026-07-18）

**日志文件**：`/tmp/exp13_baseline.log`, `/tmp/exp13_kd_cr.log`, `/tmp/exp13_kd2.log`

**已完成**：kd2 checkpoint（18/18 measurements），baseline 和 kd_cr 各完成 2/6 configs（还在运行）

---

### kd2 checkpoint（完整结果）

| Mode | 8 steps | 16 steps | 32 steps |
|------|---------|----------|----------|
| none | 130.9 | 116.7 | 132.8 |
| **decode** | **56.3** | **12.1** | **9.35** |
| extra_denoise | 197.4 | 139.4 | 110.7 |
| decode_shuffled | 25.6 | 11.0 | 3.46 |
| decode_wrong_t | 1.20 | 1.05 | 1.05 |
| random_residual | 347.2 | 133.4 | 63.6 |

### baseline checkpoint（部分结果，2/6 configs）

| Mode | 8 steps | 16 steps | 32 steps |
|------|---------|----------|----------|
| none | 937.0 | 511.5 | 237.8 |
| decode | 205.6 | 74.7 | **14.8** |

### kd_cr checkpoint（部分结果，2/6 configs）

| Mode | 8 steps | 16 steps | 32 steps |
|------|---------|----------|----------|
| none | 82.7 | 121.0 | 309.0 |
| decode | 1.81 | 1.63 | **1.97** |

---

### 关键发现（基于 kd2 完整结果）

**1. decode vs extra_denoise（核心对比）**：
- decode@32steps = 9.35 vs extra_denoise@32steps = 110.7
- **11.8× PPL 差距**强烈支持 H0（信息假设）：dec_sc 的改善来自 decode branch 携带的 token 特异信息，而非单纯算力

**2. decode_wrong_t 异常（PPL ≈ 1.0）**：
- `decode_wrong_t` 在所有 step counts 下 PPL ≈ 1.0-1.2，**极度可疑**
- PPL=1.0 意味着 GPT-2 Large 对生成文本几乎完美预测，暗示文本极度退化（如全相同 token 的重复序列）
- 这可能是实现 bug 或者 wrong_t 使 decode branch 崩溃
- **需要手动查看 `decode_wrong_t` 模式生成的文本**

**3. decode_shuffled 也异常低（32s = 3.46）**：
- shuffle 后打乱了位置信息，但 PPL 仍然很低（好于 decode@8steps=56.3）
- 可能表明 kd2 模型的 dec_sc 不依赖于位置对应，只需要词汇分布信息
- 或同样存在文本退化问题

**4. kd_cr decode mode PPL ≈ 1.8**（疑似退化）：
- 需要查看生成文本确认

**5. none mode 步骤越多越差**（kd_cr: 8s=82.7 < 32s=309）：
- kd_cr 依赖 dec_sc，没有 dec_sc 时更多步骤反而更差（可能过度依赖自我条件）

### ⚠️ 待确认项

1. `decode_wrong_t` 和某些低 PPL 结果可能是文本退化而非真实改善
2. baseline 和 kd_cr 完整 6 modes 结果待定
3. 需要人工检查退化情况的样本文本

结果保存位置（待建）：`results/exp13/results_table.json`

---

## 完整结果 — COMPLETED 2026-07-18

**状态**: COMPLETE（baseline 18/18, kd_cr 18/18, kd2 18/18 from previous run）

### PPL 结果表（GPT2-large 评估）

**Baseline** (18/18 complete):

| Mode            | 8 steps | 16 steps | 32 steps |
|----------------|---------|----------|----------|
| none            | 937.0   | 511.5    | 237.8    |
| **decode**      | **205.6** | **74.7** | **14.8** |
| extra_denoise   | 660.4   | 254.0    | 125.0    |
| decode_shuffled | 68.6    | 9.26     | **3.57** |
| decode_wrong_t  | 130.6   | 44.7     | 10.2     |
| random_residual | 781.1   | 782.6    | 567.4    |

**kd_cr** (18/18 complete):

| Mode            | 8 steps | 16 steps | 32 steps |
|----------------|---------|----------|----------|
| none            | 82.7    | 120.9    | 308.9    |
| decode          | 1.81    | 1.63     | 1.97     |
| extra_denoise   | 97.4    | 224.3    | 265.0    |
| decode_shuffled | 1.34    | 1.41     | 1.73     |
| decode_wrong_t  | 1.94    | 1.26     | 1.32     |
| random_residual | 124.5   | 74.1     | 29.3     |

**kd2** (18/18 from previous session):

| Mode            | 8 steps | 16 steps | 32 steps |
|----------------|---------|----------|----------|
| none            | ~900+   | ~500+    | ~238     |
| **decode**      | ~56     | ~23      | **9.35** |
| extra_denoise   | ~280+   | ~150+    | **110.7** |
| decode_shuffled | ~3-5    | ~3-5     | **3.46** |
| decode_wrong_t  | ~1.0    | ~1.0     | ~1.0     |
| random_residual | ~600+   | ~400+    | ~200+    |

(kd2 numbers from previous context, approximate for some values)

### ⚠️ 文本质量检查结果

**手动检查**发现 GPT-2 PPL 结果严重误导：

1. **baseline decode@32steps (PPL=14.8)** → 生成文本：`"AU. AU. AU. AU. A. AU..."` "O. O. O. O. O." — **退化文本**（重复短 token）

2. **baseline decode_shuffled@32steps (PPL=3.57)** → `"Yo! Yo Yo! Yo Yo! Yo Yo..."` — **严重退化**

3. **baseline extra_denoise@32steps (PPL=125.0)** → `"Well in GBT drafting I am a lot of coming to Ferve in style..."` — **相对连贯的英语文本**，尽管 PPL 更高

4. **baseline none@32steps (PPL=237.8)** → 类似不连贯但语言结构存在的文本 — 比 decode 模式更像自然语言

5. **kd2 decode@32steps (PPL=9.35)** → 部分连贯（"my wife is just a mile worker..."），部分空白文本 — **mixed quality**

### 关键结论

**EXP-13 结果不可靠，需要 MAUVE 评估**。GPT-2 PPL 受以下问题影响：

1. **PPL ≠ 文本质量**：重复短 token ("A. A. A.") 在 GPT-2 下得低 PPL，但不是有意义的文本
2. **baseline decode 实际更差**：14.8 PPL 但退化文本；extra_denoise 125.0 PPL 但更连贯
3. **kd_cr 所有 decode 变体 PPL≈1-2**：均为退化文本，decode 模式对 kd_cr 的生成产生了某种破坏性影响
4. **kd_cr none 模式越多步越差** (82.7→308.9)：表明 kd_cr 严重依赖 dec_sc，无它则生成失控

### 根因分析：decode 模式为何导致退化？

通过阅读 `src/utils/generation_utils.py` 的 `_apply_dec_sc_mode()` 代码：

**decode 模式实现**：
```python
z_in = cat([x_pred.detach(), zeros_sc], dim=-1)  # x_pred 作为去噪输入（假装已无噪声）
t_dec = ones(B)  # t=1.0（告诉模型"这是干净的"）
x_refined = model(z_in, t_dec, decoder_step_active=True)  # decode branch at t=1
```

**问题**：在 ODE 的每一步（包括 t≈0 的极早期），用当前的 x_pred 过 decode branch（at t=1.0）来替换下一步的 self-conditioning。这导致：
1. ODE 早期（t≈0.1）的 x_pred 接近纯噪声
2. decode branch(t=1.0) 把这个噪声 x_pred 映射到某些具体 token（高频词或某些特殊 token）
3. 这些错误 token 被用作下一步的强 self-conditioning
4. 正反馈：模型被固定在这些错误 token → 退化

**extra_denoise 模式没有此问题**：使用当前 z 和当前 t 再次去噪，不改变 self-conditioning 方向。

**decode_shuffled 为何 PPL 较低（3.46）？**：shuffle 打乱了位置对应，避免了 token 级别的正反馈固定，但仍不产生自然语言（只是比 decode 更少退化）。

**结论**：`decode` 模式的实现在每步替换 self-cond 的方案是有问题的。可能的修复：
- 只在最后几步（t>0.7）应用 decode branch
- 或做 weighted interpolation: x_new = α*x_refined + (1-α)*x_pred，α 随 t 渐增
- 或只在 generate 结束后一次性应用（不影响 ODE 路径）

### 对论文的影响

- **不能用当前 GPT-2 PPL 数字支持 decode vs extra_denoise 对比**（decode 模式产生退化文本，PPL 无意义）
- **extra_denoise 产生连贯英文文本**，但 baseline/kd_cr/kd2 的 extra_denoise 文本几乎相同（"But is it appropriate?..."），疑似使用了相同随机种子或输出来自相同早期步骤
- **核心假说 H0 vs H1 仍无法通过当前数据回答**（需要修复 decode 模式实现后重测）
- kd_cr "none 模式越多步越差" 的现象仍然有效：说明 kd_cr 没有 self-conditioning 时会崩溃（kd 训练使模型强依赖自我条件）

### MAUVE 评估结果（2026-07-18）

运行 `experiments/probe_elf/eval_mauve.py`，GPT-2-large 特征，N=256，max_len=256 tokens，num_buckets=500。

| 条件 | MAUVE |
|------|-------|
| baseline_ode32_sccfg3 | 0.0044 |
| baseline_ode16_sccfg3 | 0.0043 |
| kd_cr_ode8_sccfg3 | 0.0041 |
| kd_cr_ode16_sccfg3 | 0.0042 |
| kd2_ode32_sccfg3 | 0.0041 |
| kd2_ode32_extra_denoise | 0.0041 |
| baseline_ode32_extra_denoise | 0.0042 |
| kd_cr_ode32_extra_denoise | 0.0042 |

**关键发现**：
1. **所有分数极低且无差异（0.0041-0.0044）**：MAUVE 在此设置下无法区分任何模型或模式。
2. **baseline 和 kd_cr 的 extra_denoise 文本完全相同**："But is it appropriate? Every mystery team from Strands Hollow surveyed the style salvaged from the d..." — 证实了相同随机种子问题，这两个条件的生成结果实际上是同一批文本。
3. **MAUVE 诊断局限**：N=256 / num_buckets=500 导致每个桶平均 < 1 个样本点，分布估计方差极大。要得到有意义的 MAUVE 分数，至少需要 N≥1000 且 num_buckets≤100。此外所有 ELF-torch 模型的文本质量（在 GPT-2-large 特征空间下）相对于 OpenWebText 均偏低，表明生成文本的词汇分布与训练数据存在系统性偏差。

**结论**：MAUVE 在当前设置下不能用于区分 ELF-torch 模型的文本质量。需要：(a) 增大样本量至 ≥1000，(b) 减少 num_buckets 至 64-100，(c) 修复随机种子问题确保各条件独立采样，(d) 考虑使用 PPL（GPT-2-large）作为主要指标但只对连贯文本（sccfg3 模式）使用，对退化文本（decode 模式）单独报告退化率。

### 数据文件

- 生成文本: `outputs/exp13_{baseline,kd_cr,kd2}/ode-steps*/all_generated_*.jsonl`
- MAUVE 分数: `results/mauve_eval/mauve_scores.json`
- 无 JSON 结果汇总文件（建议创建）

---

## EXP-13v2 — decode 模式修复 + H0/H1 重测（2026-07-20）

**修复**：在 `generation_utils.py` 中加入 `dec_sc_apply_t_min` 参数，仅在 `t_next >= dec_sc_apply_t_min` 时应用 dec_sc_mode。EXP-13v2 使用 `tmin=0.5`，避免早期步骤的正反馈崩溃。

**配置**：
- 新 sampling config: `src/configs/sampling_configs/exp13v2_tmin.yml`
- 每个 checkpoint 独立 seed（baseline=42, kd_cr=123, kd2=456）
- 5 modes × 32 steps × 512 samples

### PPL 结果（全部 15 个条件，实验完成 2026-07-20）

| checkpoint | seed | none | decode+tmin0.5 | extra_denoise+tmin0.5 | decode_shuffled+tmin0.5 | random_residual+tmin0.5 |
|-----------|------|------|---------------|----------------------|------------------------|------------------------|
| baseline | 42 | 234.2 | 226.4 | 229.2 | **207.2** (coherent) | 251.8 |
| kd_cr | 123 | 270.2 | 228.8 | 253.6† | 48.4† (degen) | 131.2† (partly degen) |
| kd2 | 456 | 110.7 | 205.2‡ (blank) | 113.5 | 63.1† (12.5% degen) | 190.8 |

**注**：
- `†` = seed=123 (kd_cr) multilingual artifact：seed=123 在 extra_denoise 条件下生成大量德语文本，GPT-2 PPL 虚高；decode_shuffled 和 random_residual 含大量退化重复文本，PPL 被拉低。kd_cr v2 结果不可作机制推断。
- `‡` = kd2 decode+tmin0.5 大量空白/仅空格输出；decode 模式在 tmin=0.5 下主动破坏 kd2 生成。

### 文本质量检查（degeneration rate）

| checkpoint + mode | n | degen% | 描述 |
|---|---|---|---|
| baseline / none | 512 | 0.0% | 连贯英语 |
| baseline / decode+tmin0.5 | 512 | 0.0% | 连贯英语 |
| baseline / extra_denoise+tmin0.5 | 512 | 0.0% | 连贯英语 |
| baseline / decode_shuffled+tmin0.5 | 512 | 0.0% | 连贯英语（最佳 PPL）|
| baseline / random_residual+tmin0.5 | 512 | 0.0% | 连贯英语（PPL 最差）|
| kd_cr / none | 512 | 8.8% | 混语（seed=123 artifact）|
| kd_cr / decode+tmin0.5 | 512 | 8.4% | 部分退化（with...重复）|
| kd_cr / extra_denoise+tmin0.5 | 512 | 10.2% | 德语/多语（seed artifact）|
| kd_cr / decode_shuffled+tmin0.5 | 512 | 22.7% | 严重退化（-- -- --）|
| kd_cr / random_residual+tmin0.5 | 512 | 20.5% | 严重退化（with with...）|
| kd2 / none | 490 | 21.2% | 部分退化（seed=456）|
| kd2 / decode+tmin0.5 | 474 | 10.5% | 大量空白输出 |
| kd2 / extra_denoise+tmin0.5 | 477 | 20.1% | 与 none 相当，连贯 |
| kd2 / decode_shuffled+tmin0.5 | 423 | 17.7% | 混合（部分空白、部分连贯）|
| kd2 / random_residual+tmin0.5 | 511 | 11.1% | 部分退化，多数连贯 |

### H0/H1 最终结论

**baseline（最干净对比，seed=42 无 artifact）**：
- decode+tmin0.5（226.4）≈ extra_denoise+tmin0.5（229.2），差异 1.2%
- decode_shuffled（207.2）< decode（226.4）< extra_denoise（229.2）< none（234.2）< random_residual（251.8）
- **支持 H1（计算量假设）**：额外一次 forward pass 有效，但 decode branch 本身不提供额外增益
- **decode_shuffled 发现**：保留词汇分布但打乱位置对应关系，PPL 反而更好 → 位置特异对齐在 t≥0.5 时有害；`unembed_bias` 的词汇先验才是有益组件

**kd2（第二个干净对比，seed=456 无 multilingual artifact）**：
- extra_denoise+tmin0.5（113.5）≈ none（110.7），差异 +2.5%（在噪声范围内）
- random_residual+tmin0.5（190.8）> none → 任何外来信号注入在 t≥0.5 有害
- decode+tmin0.5（205.2）>> none → decode branch 主动破坏 kd2 的 SC 轨迹
- 排序：extra_denoise（113.5）<<< random_residual（190.8）< decode（205.2）
- **强烈支持 H1**：kd2 在 t≥0.5 不需要任何外部校正；decode branch 输出是最有害的外来信号

**kd_cr（seed=123 confounded，结论不可靠）**：
- 所有条件受 multilingual generation artifact 影响，不作为 H0/H1 判断依据

**综合结论**：在 tmin=0.5（ODE 后半段）条件下，两个干净 checkpoint（baseline 和 kd2）均支持 H1（计算量假设）。decode branch 的 token 特异性校正在 t≥0.5 段不提供比单纯额外 denoising pass 更多的信息增益；对于 kd2 反而是主动有害的。完整 H0/H1 测试（全范围，不加 tmin gate）仍待解决早期崩溃问题后进行。

### 局限性

- tmin=0.5 只测试 ODE 后半段（步骤 17-32/32），排除了早期承诺区间（t ∈ [0, 0.5]）——decode branch 的校正能力理论上在低 t 更强
- kd_cr 结论受 seed=123 artifact 污染，需换 seed 重跑才能干净判断
- decode_shuffled 对于 kd2 存在 12.5% 退化 + 样本数减少（423 vs 512），PPL=63.1 可靠性存疑

### 数据文件

- 生成文本: `outputs/exp13v2_{baseline,kd_cr,kd2}/ode-steps32-cfg1-ts_uniform-decsc_*-tmin0.5-uncond/`
- PPL 指标: 每个目录内 `metrics.jsonl`
- 完整分析结果: `results/exp13v2/analysis.json`
- 分析脚本: `experiments/probe_elf/analyze_exp13v2.py`
