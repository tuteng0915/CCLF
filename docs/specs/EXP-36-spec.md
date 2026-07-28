# EXP-36 Spec — Valid dec_sc × DF Interaction (Gated SC)

## 背景与动机

EXP-33/34/35 尝试测试 dec_sc 与 DF 的交互，但全部因**无 tmin gate 的 dec_sc 产生退化文本**而作废（"centre centre..."、"eight twenty eight twenty..."、"AS. AS. AS..."）。

根本原因：`dec_sc_apply_t_min=0.0`（默认）→ 高噪声早期步骤（t≥0.7）decode branch 质量极差 → 正反馈循环。

**EXP-36 修复**：使用 `dec_sc_apply_t_min: 0.5`，与 EXP-13v2 一致，保证高噪声步骤（t<0.5）不应用 dec_sc，切断正反馈。

---

## 实验设计

**三个子实验**（同步运行，2026-07-21）：

| 子实验 | checkpoint | GPU | PID | seed | output_dir |
|--------|-----------|-----|-----|------|-----------|
| EXP-36a | baseline | 2 | 1218019 | 42 | exp36_baseline_decsc_gated |
| EXP-36b | kd_cr | 4 | 1221544 | 123 | exp36_kd_cr_decsc_gated |
| EXP-36c | kd2 | 6 | 1224259 | 456 | exp36_kd2_decsc_gated |

**采样配置**：`spec36_df_decsc_gated.yml`（5 条件，32 步，n=256）

| condition | df_variant | df_commit_thresh / df_soft_alpha | df_t_min | dec_sc_mode | dec_sc_apply_t_min |
|-----------|-----------|----------------------------------|---------|------------|-------------------|
| none | none | — | — | none | — |
| none + dec_sc | none | — | — | decode | 0.5 |
| freeze_0.5 + dec_sc | freeze | 0.5 | 0.7 | decode | 0.5 |
| freeze_1.0 + dec_sc | freeze | 1.0 | 0.7 | decode | 0.5 |
| soft_0.3 + dec_sc | soft | 0.3 | 0.7 | decode | 0.5 |

**门控逻辑**：
- t ∈ [0.7, 1.0]：DF + dec_sc 均激活（高噪声，DF 冻结高置信位置，dec_sc 提供校正信号）
- t ∈ [0.5, 0.7)：仅 dec_sc 激活（DF 已结束，dec_sc 仍在低噪声精炼阶段）
- t < 0.5：两者均不激活

**关键科学问题**：当 dec_sc 有正确门控时，dec_sc 与 DF 是**互补**（各自独立贡献，和为正）还是**竞争**（占用同一"纠错角色"，同时使用反而干涉）？

---

## 预期假设

**H1（互补）**：dec_sc 在低噪声阶段（t<0.7）提供 token 级纠错；DF 在高噪声阶段（t≥0.7）提供位置级上下文固定。两个机制作用于不同阶段/不同对象 → 应有叠加收益。

预测（如果 H1 成立）：
- kd2 + DF alone = −48.9%；kd2 + dec_sc alone = ~X%；kd2 + dec_sc + DF ≈ −48.9% - X% - ε（至少与最好单一方法持平）
- baseline + DF alone = −5.1%；baseline + dec_sc alone = ~−36%*；两者结合应接近或超过 dec_sc alone

**H2（竞争）**：DF 冻结位置破坏了 dec_sc 在这些位置依赖的 ODE 动力学 → 干涉。
（这是无效 EXP-33/34/35 表面指向的结论，但数字无效）

*baseline dec_sc PPL 参考：EXP-13v2 提供 decode(tmin=0.5) at 32 steps。

---

## 结果

**完成时间**：2026-07-21（所有三个子实验完成，32步 512-token）

### PPL 汇总表

| 条件 | baseline PPL | kd_cr PPL | kd2 PPL |
|------|-------------|-----------|---------|
| none（无 dec_sc，无 DF） | **127.76** | 331.92 | **282.52** |
| dec_sc (tmin=0.5) | 232.24 | **264.84** | 600.72 |
| dec_sc + DF soft 0.3 | 863.23 | 256.06 | 544.52 |
| dec_sc + DF freeze 0.5 | 1729.28 | 318.42 | 556.38 |
| dec_sc + DF freeze 1.0 | 1857.33 | 337.48 | 571.68 |

### 生成文本质量

**baseline**:
- none（127.76）: 正常英文 — *"The protests that perpetuate across all corners of Europe..."*
- dec_sc（232.24）: 英文但破碎 — *"Think of living in a small place, a new station... Musical. Without arts."*
- dec_sc + soft 0.3（863.23）: 更碎片化 — *"Off this bad night on Saturday 5... Not lane from Stone Ars."*
- dec_sc + freeze 0.5（1729.28）: ⚠️ 法德语混杂 — *"bevor werden desperate erstmal... des mots..."*

**kd_cr**:
- none（331.92）: 轻微不连贯英文 — *"sum as: about keeping priorities when in relation to topic topic..."*
- dec_sc（264.84）: ⚠️ 罗马尼亚语/德语混杂 — *"să Urlaub... pentru... eignen... terapie este multe als..."*
- dec_sc + soft 0.3（256.06）: ⚠️ 德语主导 — *"Des wunderschöne seit 3.03.35 30 mois... arbeitete das film..."*
- dec_sc + freeze 1.0（337.48）: ✓ 英文较正常 — *"So over what power is their own fascination based on..."*

**kd2**:
- none（282.52）: 轻微重复英文 — *"because because because with with with sentiment..."*
- dec_sc（600.72）: ⚠️ 法语混杂 — *"ses fascination wird au haut..."*

### 关键发现

**1. tmin=0.5 gate 不足以防止 kd_cr/kd2 的语言混合退化**

EXP-36 的核心修复目标（防止 dec_sc 正反馈）在 kd_cr 和 kd2 上依然失败。kd_cr + dec_sc(tmin=0.5) 生成罗马尼亚语/德语混合文本（PPL=264.84），比 none(331.92) 的 PPL 更低，但文本质量更差。这与 EXP-33/34/35（无 gate）的退化机制相同，只是程度更轻。

**根本原因**：kd_cr 和 kd2 的 decode head 对欧洲语言 token 有过度自信的预测，即使在 t>0.5 的阶段（信号已较强）也会持续。将这些预测注入 self-conditioning 仍然触发正反馈，只是慢了一些。

**2. PPL 指标在语言退化时失效（再次确认）**

kd_cr + dec_sc 的 PPL=264.84 < kd_cr none 的 331.92，表面上有改善，但生成文本为罗马尼亚语混杂。GPT-2 Large 对欧洲语言文本的评分偏低，导致 PPL 指标错误地报告改善。

**3. baseline + dec_sc(tmin=0.5) 无改善，有损失**

baseline none=127.76 → dec_sc=232.24（+82%），文本变碎片化但仍是英文。这说明即使对 baseline（未受语言混合问题影响），dec_sc 在 tmin=0.5 时仍然干扰了生成过程。

**4. kd_cr + dec_sc + DF freeze 1.0 是唯一"正常英文"的 dec_sc+DF 条件**

kd_cr + freeze_1.0 (337.48) 生成正常英文，但 PPL 与 kd_cr none (331.92) 相近，无实质改善。

**5. H1（互补假说）不成立；H2（竞争假说）部分成立**

所有 dec_sc + DF 组合对 baseline 和 kd2 均严重退化，PPL 大幅上升。kd_cr 只在 freeze_1.0 条件下勉强稳定。dec_sc 和 DF 在这些检查点上不互补，主要产生竞争/干涉。

### 对论文影响

1. **EXP-13v2 的 dec_sc 改善结论需要 checkpoint-specific 注释**：dec_sc 改善仅在 baseline checkpoint 的某些配置下成立；对 kd_cr 和 kd2 均有退化风险。
2. **DF + dec_sc 不能作为通用推理增强**：即使有 tmin gate，三种 checkpoint 中只有 kd_cr + freeze_1.0 勉强稳定。
3. **语言退化机制泛化性**：EXP-33/34/35（无 gate）和 EXP-37c（1024-token DF）的退化机制在 EXP-36 中再次出现，证明这是 kd_cr/kd2 decode head 的结构性问题，不依赖于特定配置。

---

## 状态

- **EXP-36a** (baseline): DONE — 2026-07-21，GPU 2，PID 1218019
- **EXP-36b** (kd_cr): DONE — 2026-07-21，GPU 4，PID 1221544
- **EXP-36c** (kd2): DONE — 2026-07-21，GPU 6，PID 1224259

---

## ⚠️ 方法论问题（2026-07-22 审查）

### 1. CRITICAL：缺少完整 factorial arms，无法测量 interaction

当前五个条件：none、dec_sc、freeze+dec_sc、soft+dec_sc，缺少 **freeze only** 和 **soft only**。

要计算 dec_sc × DF 的 interaction（difference-in-differences），需要完整 2×2 设计：

|          | 无 dec_sc | 有 dec_sc   |
|----------|:--------:|:-----------:|
| 无 DF    | none     | dec_sc      |
| 有 DF    | DF       | DF + dec_sc |

Interaction 计算：`I = Q(DF+SC) − Q(DF) − Q(SC) + Q(none)`

**不能使用 EXP-31 的 DF-only 数字**填入 2×2 表，因为：
- 不同 run，seed/context 可能不同
- EXP-31 baseline 已有 seed artifact（见 EXP-31 ⚠️）
- output filtering 标准可能不同

**建议立即补充**：freeze_0.5 only、freeze_1.0 only、soft_0.3 only，与同一 run 的 none 形成可信对照。

### 2. 时间方向在文档中写反了

ELF convention：t=0 = 纯噪声，t=1 = clean。文档中：
- "t∈[0.7,1.0]：高噪声" ← **错误**，t≥0.7 是低噪声、晚期区域
- "t<0.5：都不激活" ← 这才是高噪声早期区间

准确描述：
- t<0.5：early/high-noise（dec_sc 不激活，DF 不激活）
- 0.5≤t<0.7：mid-to-late，dec_sc 激活（有 tmin=0.5 gate）
- t≥0.7：late/low-noise，dec_sc 和 DF 均激活（两者重叠区间）

因此"dec_sc 和 DF 作用在完全不同阶段"的说法不成立，它们在 t≥0.7 明显重叠，这是 interaction 测量最关键的区间。

### 3. Entropy threshold 跨 checkpoint 不可比

相同 H<0.5/1.0 nat 对三个 checkpoint 对应不同的：
- frozen fraction（baseline/kd_cr/kd2 的 entropy calibration 不同）
- precision（被冻结的位置中有多少最终正确）
- token composition（被冻结的 token 类型不同）

"同一 DF 条件"对三个 checkpoint 实际是不同强度的干预。跨 checkpoint 比较时应使用 frozen fraction matched 或 calibrated error-rate matched，而非相同 entropy threshold。

### 4. tmin=0.5 并不保证 dec_sc 安全

EXP-36 结果已再次确认：tmin=0.5 gate 不足以防止 kd_cr/kd2 的语言混合退化（kd_cr dec_sc PPL 看似下降但文本为多语言混杂）。这说明"tmin 更高、dec_sc 更晚激活"的思路才更合理。建议同时测试 tmin∈{0.5, 0.7, 0.8}。

### 5. 评价指标必须升级

PPL 在语言退化时失效（已在 EXP-33/34/35/36 反复证明）。每个条件至少需要：
- 5 个 generation seeds
- degeneration / empty / repetition rate
- language ID（lang detect）
- distinct-1/2/3 n-gram
- PPL（仅作参考，需结合其他指标解读）
- unigram KL from reference
- MAUVE（若有参考集）
- 少量人工比较

### 6. 安全结论

EXP-36 当前结果支持：

> Adding tmin=0.5 gate reduces dec_sc-induced degeneracy compared to ungated versions, but does not fully prevent multilingual artifact in kd_cr/kd2. dec_sc and DF do not show complementary effects under current tested conditions; interactions require a complete factorial design with matched seeds and multi-metric evaluation.

**不能说**：dec_sc 与 DF 互补/竞争（factorial arms 不完整）；相同 entropy threshold 对不同 checkpoint 等效；PPL 下降 = 质量改善。

---

## EXP-36 Full Factorial 实现（2026-07-22，运行中）

**目标**：添加缺失的 DF-only arms，实现完整 2×2 factorial 设计。

### 新增文件

- **Sampling config**: `src/configs/sampling_configs/spec36_factorial.yml`
  - 8 conditions（完整 2×2）:
    - DF=0, SC=0: `none`
    - DF=0, SC=1: `sc_only`（dec_sc, tmin=0.5）
    - DF=1, SC=0: `freeze_0.5_only`, `freeze_1.0_only`, `soft_0.3_only`（新增！）
    - DF=1, SC=1: `freeze_0.5_sc`, `freeze_1.0_sc`, `soft_0.3_sc`
- **Eval configs**: `src/configs/training_configs/eval_spec36_factorial_{baseline,kd_cr,kd2}.yml`
  - 所有 3 个 checkpoint，seed=42（统一种子）
- **Analysis script**: `experiments/probe_elf/analyze_spec36_factorial.py`
  - 计算 main effect DF、main effect SC、interaction I = (DF+SC) - DF - SC + none

### 2×2 交互量

对每个 DF 变体（freeze_0.5、freeze_1.0、soft_0.3）：

```
I(DF_variant) = PPL(DF+SC) - PPL(DF-only) - PPL(SC-only) + PPL(none)
```
- I < 0: DF 和 SC 互补（协同）
- I > 0: DF 和 SC 竞争（冗余）
- I ≈ 0: 近似加性

### 运行命令

```bash
cd models/ELF-torch
GPU=6 bash scripts/launch.sh eval src/configs/training_configs/eval_spec36_factorial_baseline.yml \
    --checkpoint_path converted/elf_b-owt-baseline_torch.pt  &
GPU=7 bash scripts/launch.sh eval src/configs/training_configs/eval_spec36_factorial_kd_cr.yml \
    --checkpoint_path converted/elf_b-owt-kd-cr_torch.pt  &
GPU=X bash scripts/launch.sh eval src/configs/training_configs/eval_spec36_factorial_kd2.yml \
    --checkpoint_path converted/elf_b-owt-kd2_torch.pt
python experiments/probe_elf/analyze_spec36_factorial.py  # aggregate after all done
```

**状态**: 全部完成 — baseline (GPU6) 19:15，kd_cr (GPU7) 19:15，kd2 (GPU6) 2026-07-22 19:22 启动，~20:00 完成。分析脚本 arm pattern 已修复并运行完毕（`outputs/spec36_factorial_summary.json`）。

### EXP-36 Full Factorial 结果（baseline + kd_cr，2026-07-22）

**PPL 结果（32 ODE steps，256 samples，seed=42）**：

| arm | baseline | kd_cr |
|-----|---------|-------|
| none | **127.8** | 331.9 |
| SC-only | 232.2 | **264.8** |
| freeze_0.5-only | 121.3 | 426.0 | 219.2 |
| freeze_1.0-only | 131.1 | 475.6 | 144.4 |
| soft_0.3-only | 121.8 | 389.9 | 230.3 |
| freeze_0.5+SC | 1715.7 | 318.6 | 588.0 |
| freeze_1.0+SC | 1829.8 | 343.3 | 620.3 |
| soft_0.3+SC | 838.6 | 288.1 | 622.7 |

**2×2 交互 I = (DF+SC) - DF - SC + none（2026-07-22 全部完成）**：

| DF variant | baseline I | kd_cr I | kd2 I |
|-----------|-----------|---------|-------|
| freeze_0.5 | **+1490** | **-40** | **+51** |
| freeze_1.0 | **+1594** | **-65** | **+158** |
| soft_0.3   | **+612**  | **-35** | **+74** |

**SC 主效应（SC-only vs none）**：

| checkpoint | none PPL | SC-only PPL | SC main effect | DF-SC interaction sign |
|-----------|---------|------------|---------------|----------------------|
| baseline  | 127.8 | 232.2 | **+104.5 (SC hurts)** | **+1490 to +1594** |
| kd_cr     | 331.9 | 264.8 | **-67.1 (SC helps)** | **-35 to -65** |
| kd2       | 282.5 | 600.7 | **+318.2 (SC hurts badly)** | **+51 to +158** |

**关键规律**：**SC main effect 的符号 = DF-SC interaction 的符号**。当 SC 独立有益时（kd_cr），DF+SC 互补（I<0）；当 SC 独立有害时（baseline、kd2），DF+SC 反协同（I>0）。

**文本质量量化（256 samples per arm，seed=42）**：

| arm | multilingual (>2% non-ASCII) | repetitive (>35% single word) |
|-----|------------------------------|-------------------------------|
| kd_cr none | **15.2%** | 2.3% |
| kd_cr SC-only | **7.4%** | 1.2% |
| kd2 none | 3.1% | **12.1%** |
| kd2 SC-only | 3.5% | **1.2%** |

kd_cr SC 减少多语言退化（15.2%→7.4%）且 PPL 下降（331.9→264.8）：两指标同向，说明是真实改善。kd2 SC 减少重复率（12.1%→1.2%）但 PPL 急升（282→600）：SC 把重复失败模式转变为语义非连贯失败模式。

**baseline DF+SC 文本**：多语言退化（德文/英文混杂），与 EXP-33/35 相同模式。PPL=1716-1830 由退化解释。

### 关键解读（2026-07-22 三模型完整结果）

1. **交互符号由 SC 独立效果决定，而非 oracle accuracy**：
   - kd_cr oracle acc @t=0.5 = 99.5%，kd2 oracle acc @t=0.5 = 99.1%（相近！）
   - 但 kd_cr I=-65（互补），kd2 I=+158（反协同）
   - **结论：oracle accuracy 不是 DF-SC 交互符号的决定因素**
   - 真正的预测因子：SC 是否作为独立纠错机制有效（kd_cr:YES；baseline/kd2:NO）

2. **kd_cr 的唯一性**：kd_cr 是唯一 SC 独立有益的 checkpoint（PPL 264.8 < 331.9）。这与 kd_cr 的 decode branch 重组（EXP-15v2: unembed_bias R=2.59）直接对应。**kd_cr 的 KD 训练专门重组了 decode interface，使 SC 得以作为纠错机制工作**；kd2 虽有相似的 oracle accuracy，但 decode interface 的重组方向不同，无法支持 SC。

3. **kd2 的独特 profile**：
   - DF 单独极有效（freeze_1.0: 282.5→144.4，-48.9%）
   - SC 单独极有害（282.5→600.7，+118%）
   - DF+SC 仍很差（620.3，比 none 更差），正交互符号（I=+158）
   - **解释**：kd2 DF 机制有效，但 dec_sc 与 kd2 decode branch 不兼容（具体机制待查，可能与 kd2 的训练方式有关）

4. **baseline 的机制已修订**：
   - 原解读：baseline DF 冻结 oracle acc=77% 的错误 token → SC 基于错误条件退化
   - 这是充分条件但非必要条件：kd2 DF 冻结 oracle acc=99.1% 的正确 token，但 DF+SC 仍然反协同
   - 正确的机制描述：**DF+SC 是否有效取决于 SC 本身是否有效（与 DF 无关）**

5. **三模型 DF-only 比较**（不需要 SC）：
   - baseline freeze_1.0: +2.6%（略有害），best = freeze_0.5（-5.1%）
   - kd_cr freeze_1.0: +43.3%（严重有害），DF 全面有害
   - kd2 freeze_1.0: **-48.9%（极有益）**，DF 对 kd2 最有效

   规律：DF 效果与 none 基线的生成质量正相关。kd2 none PPL=282.5（较差）→ DF 改善空间最大；baseline none PPL=127.8（最好）→ DF 改善空间最小；kd_cr none PPL=331.9（依赖 SC，DF 破坏依赖关系）。

### 立即需要删除的旧表述

- ~~"oracle accuracy 预测 DF-SC 交互符号"~~ → **WRONG**：kd_cr/kd2 oracle acc 相近但 I 符号相反
- ~~"dec_sc 与 DF 互补/竞争（factorial arms 不完整）"~~ → 现有答案：**取决于 checkpoint**（kd_cr 互补，baseline/kd2 反协同）
- ~~"两者在 t≥0.7 明显重叠"~~ → 重叠不预测互作符号；SC standalone 效果才是预测因子
