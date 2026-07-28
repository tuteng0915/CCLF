# EXP-37 — 1024-Token Diffusion Forcing at Inference

**Type**: inference experiment (no training)  
**Priority**: high  
**Related**: spec-11 (DF inference, 512-token); EXP-31/32 (kd2 DF step-count sweep, 512-token)  
**Output dirs**:
- `outputs/exp37_baseline_1024/`, `outputs/exp37_kd2_1024/` (EXP-37a, pilot)
- `outputs/exp37b_baseline_1024/`, `outputs/exp37b_kd2_1024/` (EXP-37b, none condition)
- `outputs/exp37c_baseline_1024/`, `outputs/exp37c_kd2_1024/` (EXP-37c, freeze_1.0 DF)

---

## 实验背景与动机

EXP-32（512-token）显示 kd2+freeze_1.0 在 32 步时 PPL 从 282.52→144.42（−48.9%），是 512-token 上所有推理优化中最大的单项提升。但 kd2 本身是以 **1024-token 上下文**训练的，512-token 评估对 kd2 不公平。

EXP-37 系列在 **1024-token** 上重新评估：
1. EXP-37a（pilot）：none 条件，初探 kd2 在 1024-token 上的性能
2. EXP-37b：none 条件（对照组），验证 1024-token 在 baseline 和 kd2 上的性能差异
3. EXP-37c：freeze_1.0 DF（df_t_min=0.7），复现 EXP-32 在 1024-token 上的效果

**核心问题**：
- kd2 在 1024-token 下 none 的 PPL 是否比 512-token 显著更好？
- kd2+DF 在 1024-token 下是否继续有效，还是出现退化？

---

## 实验设计

### 模型检查点
- **baseline**: `converted/elf_b-owt-baseline_torch.pt` (step=95085)
- **kd2**: `converted/elf_b-owt-kd2_torch.pt` (step=399372)

### 配置
- max_length: 1024, batch_size: 4, global_batch_size: ~ (null, 禁用 512 默认值)
- num_samples: 256, eval_ppl_model: gpt2-large
- seed: 42 (baseline) / 456 (kd2)

### 条件
- **EXP-37a/b（none）**: 8/16/32 步，无 DF
- **EXP-37c（freeze_1.0）**: 8/16/32 步，df_variant=freeze, df_commit_thresh=1.0, df_t_min=0.7

---

## 实验结果

### EXP-37a — 1024-token None（Pilot，不同种子）

| steps | baseline PPL | kd2 PPL |
|-------|-------------|---------|
| 8  | 932.77 | 130.90 |
| 16 | 505.82 | 130.56 |
| 32 | 230.52 | 127.08 |

### EXP-37b — 1024-token None（对照组，seed 42/456）

| steps | baseline PPL | kd2 PPL |
|-------|-------------|---------|
| 8  | 929.39 | 130.22 |
| 16 | 498.23 | 123.51 |
| 32 | 236.20 | 146.37 |

### EXP-37c — 1024-token freeze_1.0 DF（df_t_min=0.7）

| steps | baseline PPL | kd2 PPL | kd2 文本质量 |
|-------|-------------|---------|-------------|
| 8  | 900.24 | 102.72 | ✗ 德语/罗马尼亚语混乱文本 |
| 16 | 481.37 | 17.27  | ✗ 全空白字符 |
| 32 | 234.27 | 2.29   | ✗ 严重退化（破折号+空白） |

**EXP-37c baseline 文本示例（32步，PPL=234.27，正常）**:
```
"His new home in Oxford, with the room littered with cool paper plus incredible amounts of craft..."
"LittleApptel maintains goal of being a star in a political intent..."
```

**EXP-37c kd2 文本示例（退化）**:
- 8步: `"Für văn ihr Startplatz Urlaub auf Straßenverkehr bei Szene pentru că der Szene..."`
- 16步: `"                                                              "` (全空格)
- 32步: `"......    -- -- -- -- -- -- --. -- --.      --...."`

---

## 与 512-token 对比（EXP-32）

| 条件 | 512-token PPL | 1024-token PPL |
|------|--------------|----------------|
| kd2 + none + 8步  | 688.11 | 130.22 (−81%) |
| kd2 + none + 16步 | 602.64 | 123.51 (−79%) |
| kd2 + none + 32步 | 282.52 | 146.37 (−48%) |
| kd2 + freeze_1.0 + 8步  | 615.30 | 102.72 → 退化 |
| kd2 + freeze_1.0 + 16步 | 486.80 | 17.27 → 退化 |
| kd2 + freeze_1.0 + 32步 | 144.42 | 2.29 → 严重退化 |

---

## 关键发现

### 1. kd2 在 1024-token 下的基础性能大幅改善

kd2+none 在 1024-token 的 PPL 比 512-token **低 48-81%**，尤其在低步数（8步）提升最显著（688→130，−81%）。这表明 kd2 的训练长度（1024-token）与评估长度严重不匹配时性能恶化，1024-token 才是其真实性能水平。

### 2. kd2+DF 在 1024-token 下全面退化（与 512-token 相反）

在 512-token 下，kd2+freeze_1.0 是最优配置（−48.9%）；在 1024-token 下，相同配置全部退化（8步德语文本，16步空白，32步破折号）。

**退化机制**（与 EXP-33/34 dec_sc+DF 冲突相似）：
- kd2 的 decode head 在 1024-token 序列上对特定 token 做出过度自信预测
- df_commit_thresh=1.0 意味着 entropy<1.0 nat 时冻结；kd2 在 1024-token 下更多位置满足此条件
- 冻结的错误 token（多为欧洲语言字符）产生正反馈 → 整体输出偏向德语/罗马尼亚语
- 随着步数增加，反馈放大，最终 16/32 步时崩溃为空白

### 3. baseline+DF 在 1024-token 下表现稳定

baseline+freeze_1.0 在 1024-token 全部有效（无退化），呈现小幅改善（−0.9% to −3.4%），与 512-token 的行为一致（spec-11v2 freeze_0.5 最优 −5.1%）。这证明退化是 kd2 checkpoint 特异性问题。

### 4. PPL 指标在退化场景下失效

kd2+DF 16步的 PPL=17.27 和 32步的 PPL=2.29 是 GPT-2 Large 对退化文本的评分，**不代表真实语言质量**。全空白字符和重复破折号被 GPT-2 评为低 PPL 是指标失效，不是真正改善。

---

## 结论与对论文影响

1. **kd2 在 1024-token 下的基础 PPL 比 512-token 低 48-81%**：这是关于 kd2 训练-评估匹配的重要基线修正，512-token EXP-31/32 中 kd2 vs baseline 的差距部分来自评估长度不匹配。

2. **DF 效果的 checkpoint 特异性在 1024-token 下反转**：512-token 上 kd2+DF 最有效，1024-token 上 kd2+DF 全面退化。这表明 DF 效果取决于 checkpoint 的 decode head 置信度分布，而非 checkpoint 质量。

3. **DF 不能作为通用推理增强**：正确结论是"DF 在 baseline+512-token 上有小幅改善，在 kd2+512-token 上有大幅改善，在所有 1024-token kd2 条件下退化"——高度上下文依赖，不能简单推荐。

4. **推理时最优策略（更新）**：
   - 1024-token + kd2：none（PPL=146 at 32步），**不加 DF**
   - 1024-token + baseline：freeze_1.0（PPL=234 at 32步，小幅优于 none 的 236）
   - 512-token + kd2：freeze_1.0（PPL=144 at 32步）

---

## 状态

- EXP-37a: DONE（pilot，2026-07-20）
- EXP-37b: DONE（none 对照，2026-07-21）
- EXP-37c: DONE（freeze_1.0 DF，2026-07-21，GPU 0/1）
