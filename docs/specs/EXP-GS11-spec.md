# EXP-GS11 Spec — Pooling/Averaging Confound Check (P0-1)

## 背景与地位

用户对 GS1–GS3 的审阅指出一个此前未被控制的关键 confound：GS1 的"global probe"输入是
`masked_mean_pool(z_t, mask)`，其中 `z_t` 是**原始 oracle state**，不是模型的 hidden
state 或 predicted-clean 输出。对 `L≈1024` 个位置的独立噪声求均值，本身就会把噪声标准差
压低约 `sqrt(L)≈32` 倍：

```
z_t = t*x_clean + (1-t)*eps
mean(z_t) = t*mean(x_clean) + (1-t)*mean(eps)
Std(mean(eps)) ~ Std(eps) / sqrt(L)
```

因此 GS1 观察到的"早期 global 信号"（比如 `t=0.05` 时 sentence cosine 已达 0.953），可能
主要是**对原始输入做平均本身放大了微弱的 clean signal**，和模型是否"先形成全局组织"无关。
这个实验是本轮审阅要求的 P0-1：直接检验这个 confound 是否解释了 GS1/GS2/GS3 的核心发现。

## 1. 设计

**核心比较**：在同一批文档、同一组 t 上，比较

- **raw**：`g_raw = mean_pool(z_t, mask)`——不经过模型，纯粹是 oracle state 的位置均值；
- **model**：`g_model = mean_pool(predicted_clean, mask)`——`predicted_clean` 是
  `adapter.forward_state(z_t, ...)` 的模型输出（denoising 之后的估计）。

在一个**检索式指标**（self-retrieval top-1 accuracy）下比较两者，而不是重新训练回归/分类
probe——用户已经指出 cosine 相似度在这个高维空间容易饱和（GS1 的 `G_sent` 就是先例），
检索式指标有明确的 chance level（`1/N`），不会被"任意两个高维向量本来就有点像"污染。

**长度 sweep 是关键**：`L_eff in {32, 128, 512, 1024}`，通过**真正截断文档**（只保留前
`L_eff` 个真实 token，其余按标准 pad 流程处理，不是事后把 mask 拍成 0——否则模型在自注意力
里仍然"看到"了全部 1024 个位置的真实内容，长度 sweep 就没有意义）来构造真正更短的文档。

若 `retrieval_acc_raw(L_eff)` 和 `retrieval_acc_model(L_eff)` 两条曲线几乎重合、且都随
`L_eff` 单调上升（大致符合 `1/sqrt(L_eff)` 的 SNR 直觉），说明信号主要来自 pooling 平均本身。
若 `model` 曲线显著高于 `raw`（尤其在小 `L_eff` 时，此时 raw averaging 本身很弱），才说明
模型确实做了超出简单平均的处理。

## 2. 指标

对每个 `(L_eff, t)`：

1. 截断 `N` 篇文档到 `L_eff` 个真实 token（不足 `L_eff` 的文档跳过，只保留真实长度
   `>= L_eff` 的文档，避免"文档本来就短"和"人为截断"混淆）。
2. 构造 `z_t`，跑一次 `forward_state` 得到 `predicted_clean`。
3. `g_raw[i]`, `g_model[i]` = 对应的 mean-pool（只在 `L_eff` 个有效位置上取均值）。
4. `clean_ref[i]` = `mean_pool(x_clean[i], mask_Leff[i])`（该文档自己在同样 `L_eff` 截断下
   的 clean pooled embedding，作为检索目标）。
5. **Self-retrieval top-1 accuracy**：对每个 `i`，在 `{clean_ref[0..N-1]}` 里按 cosine
   相似度找 `g_raw[i]`（或 `g_model[i]`）的最近邻，判断是否命中 `clean_ref[i]` 自己。
   报告 `N` 篇文档上的 top-1 准确率（chance level = `1/N`）。

## 3. 数据与规模（pilot）

- ELF baseline，`eval_exp37c_baseline.yml`（1024-token，与 GS1–GS10 一致）。
- `n_samples=48`（需要真实长度 `>=1024` 的文档才能支持全部 4 个 `L_eff`，从更大的候选池里
  过滤，实际可用数可能少于 48，见运行时输出）。
- `t ∈ {0.05, 0.28}`（早期 + GS1/GS8 常用的过渡点）。
- `L_eff ∈ {32, 128, 512, 1000}（观测到 T5-tokenized OWT 在 seq_len=1024 时真实 token 数最多约 1020，故用 1000 而不是 1024）`。

## 4. 已知简化

1. ⚠️ Self-retrieval 的候选池就是当前 batch 内的 `N` 篇文档，`N` 越大 chance level 越低、
   测试越严格；pilot `N=48` 是一个折中，`1/48≈2.1%` 的 chance level 已经足够低，能看出
   有意义的信号，但不是"从全部 OWT 语料里检索"这种更严格的设定。
2. ⚠️ 只比较 `raw` vs `model-predicted_clean` 两种，不测模型中间层 hidden state（如果
   pilot 显示 `predicted_clean` 和 `raw` 差距不大，中间层 hidden state 的进一步对照留给
   后续）。
3. Pilot 规模，数字仅用于判断这个 confound 是否成立、成立到什么程度。

## 5. 脚本与输出

```text
experiments/global_state/analyze_pooling_confound.py
```

```text
results/global_state/<model>/<checkpoint>/pooling_confound_<label>.json
```

## 状态

**Pilot DONE — confound 得到强烈证实，且比预期更严重**（ELF baseline，n=48（真实长度全部
`>=1000`，从 250 篇候选里筛出），`L_eff∈{32,128,512,1000}`，`t∈{0.05,0.28}`，GPU1，
`logs/global_state/gs11_elf_baseline_pilot.log`，输出
`results/global_state/elf/baseline/pooling_confound_pilot.json`）。**这是本轮所有 GS
实验里最重要的一个结果，直接推翻了 GS1/GS3 headline 发现的原有解释。**

## Results（pilot：ELF baseline，n=48，chance level=1/48≈0.021）

| L_eff | t | retrieval_acc (raw) | retrieval_acc (model) | cos (raw) | cos (model) |
|---|---|---|---|---|---|
| 32 | 0.05 | 0.396 | **0.021**（=chance） | 0.130 | 0.417 |
| 32 | 0.28 | **1.000** | **0.021**（=chance） | 0.684 | 0.292 |
| 128 | 0.05 | 0.583 | 0.042 | 0.222 | 0.480 |
| 128 | 0.28 | **1.000** | 0.104 | 0.852 | 0.364 |
| 512 | 0.05 | 0.792 | 0.021（=chance） | 0.402 | 0.486 |
| 512 | 0.28 | **1.000** | 0.188 | 0.953 | 0.533 |
| 1000 | 0.05 | 0.938 | 0.021（=chance） | 0.523 | 0.489 |
| 1000 | 0.28 | **1.000** | 0.771 | 0.975 | 0.731 |

**解读（如实报告，这个结果比原来担心的更严重）**：

1. **confound 完全证实，而且力度远超预期**：不经过模型、只对原始 oracle state 做 mean
   pooling（`raw`），在 `t=0.28` 时**无论 `L_eff` 多小（哪怕只有 32 个 token）都能达到
   100% self-retrieval**；即使在极早的 `t=0.05`，`raw` 的准确率也随 `L_eff` 单调上升
   （0.396→0.583→0.792→0.938），和 `1/sqrt(L_eff)` 噪声平均的直觉完全吻合。**GS1 的
   probe 输入就是这个 `raw` mean-pool**，说明 GS1 报告的"早期 global 信号"（`G_topic`,
   `G_sent`）绝大部分可以被"对原始噪声输入求均值"这一件事解释，不需要假设模型做了任何
   全局组织。
2. **更意外、更重要的发现：模型自己的 `predicted_clean` 输出在几乎所有条件下的
   self-retrieval 表现都明显差于（有时是断崖式差于）不经过模型的 `raw` 均值**——
   `L_eff=32, t=0.28` 时 `raw=1.000` 但 `model` 只有 `0.021`（恰好等于 chance！）；
   `L_eff=1000, t=0.28` 这个信息量最大的条件下，`model` 也只有 `0.771`，仍然低于
   `raw` 的 `1.000`。**模型的去噪处理没有增强可检索的文档身份信息，反而在大多数条件下
   把它抹掉了**——很可能是因为 `predicted_clean` 是模型对"合理续写"的估计，在信号不足
   （小 `L_eff`、低 `t`）时会被拉向某种通用/众数续写，而不是保留每篇文档的具体身份。
3. **这直接推翻了 GS1/GS3 的原有解释框架**：
   - **GS1**：`G_topic(t)`/`G_sent(t)` 的探针输入是 `masked_mean_pool(z_t, mask)`，即
     本实验的 `raw`。GS1 报告的"`t=0.05` 时 sentence cosine 已达 0.953"、"topic 在
     `t=0.28` 已经明显高于随机"，现在看基本就是这个 pooling confound 的直接体现，
     不能再解释为"模型早期已经形成全局语义"。
   - **GS3**：structural probe（`syntax_G`/`syntax_R`）同样是在**未经模型处理的**
     `G_t^{(k)}`/`R_t^{(k)}`（对 raw `z_t` 做 SVD 分解后的 mean-pool）上训练的——
     "structure 集中在低秩 `G`"这个发现，很可能同样主要是"raw 状态的顶奇异方向恰好
     携带更多这类可以被简单线性统计量捕捉的信号"，而不是模型学到的语义-句法解耦。
     这与用户提出的"uncentered SVD + mean-pool 双重构造"批评完全吻合，现在有了
     直接数据支持。
   - **GS2**：情况不同——GS2 测的是完整 multi-step rollout 后的 branch consensus，
     模型的实际去噪计算贯穿全程，不是对 raw 状态做静态 pooling，这个 confound 对 GS2
     的适用性较弱（但 GS2 的 `C_struct`/`C_topic` 饱和问题是另一个独立的指标问题，
     已在 GS2 spec 里记录）。
4. `cos`（余弦相似度）这一列再次印证 GS1 已经发现的"高维 cosine 饱和"问题：即使在
   `model` 的 retrieval accuracy 只有 chance 水平时（`L_eff=32, t=0.05`），
   `cos_model=0.417` 看起来"不算低"——cosine 数值本身具有误导性，这也是本实验优先用
   检索式指标而不是 cosine 数值的原因。

## 结论与后续行动

**GS1 和 GS3 的核心 headline 发现（"structural/topic 信号早于 token"）需要重新解释**：
现有证据更支持"这是对 raw noisy oracle state 做位置平均这一操作本身的统计性质"，而不是
"模型先形成了全局语义组织"。这不代表 GS1/GS3 的数字是错的（它们如实反映了 probe 在给定
输入下的表现），而是**它们测的"early recoverability"主要来自输入构造，不能归因于模型的
early global organization**——这是一个需要在 GS1-spec.md、GS3-spec.md 里补充更正说明、
并在最终写作里如实调整措辞的发现，而不是丢弃这两个实验的数据。

下一步（已经是原计划的 P0-2）：**用中心化（去掉跨位置共享均值）之后的表示重做 GS3 的
SVD 分解**，并且**优先对模型的 hidden state / predicted_clean 做同样的分析而不是 raw
oracle state**，才能把"模型真正做了什么"和"输入本身的统计结构"分开。
