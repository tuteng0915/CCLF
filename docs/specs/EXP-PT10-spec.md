# EXP-PT10 Spec — Transition Failure Predictors

## 背景与地位

P1 阶段实验，但因为它是纯粹的事后分析（不需要 GPU，只需要 EXP-PT1/PT2/PT3 已经
产出的结果），实现成本很低，所以提前插到 P0 收尾阶段一起做了。目标：把 EXP-PT2
的 6 类失败分类变成一个可以被证伪的预测模型——用简单、可解释的多元逻辑回归，
检验哪些位置级特征能预测"哪一种转变失败模式"。

## 复用与新增

**不需要新的前向传播**——纯粹读取已经存在的 npz/中间结果：

- `EXP-PT2` 的 `margin_trajectory_raw_<label>.npz`：取 `t0_ell_gt`/`t0_ell_f1`
  算 `prior_mode_advantage`（即 `m_raw(t_min)`），`t0_rank_raw` 作初始 rank。
- `EXP-PT2` 的失败标签——**原来 `classify_transition_failures.py` 只把汇总
  比例写进 JSON，没有保存逐位置的标签数组**，为了这个实验补了一处改动：现在
  额外存一个 `transition_failure_labels_<label>.npz`（`labels`, `tau_e`,
  `tau_b`, `tau_s`, `n_zero_crossings`, `gt_ids`, `f1`），已经对 EXP-PT1/PT2 里
  跑完的全部 4 个模型重新跑过这一步生成了这个文件。
- `EXP-PT3` 的 `velocity_alignment_raw_<label>.npz`：取 `t0_a_clean`/`t0_a_tok`
  作"初始 velocity alignment"特征，用 `valid_mask` 过滤掉没有可用
  token-discriminative 方向的位置（这些位置直接从回归里丢弃，不是填 0——填 0
  会伪造出一个"没有信号"的观测，比直接丢弃更容易引入偏差）。

## 已实现的 predictor（doc 列表的子集）

| Predictor | 来源 | 备注 |
|---|---|---|
| token 频率 | 当前探测样本内的 `gt_ids` 计数 | ⚠️ 不是训练语料频率，是样本内代理，和 EXP-20 的既有告诫一致 |
| 功能词/内容词 | 固定封闭类词表匹配解码后的 token 字符串 | 不是真正的 POS tagger，和 EXP-08/08v2 的做法一致 |
| token 字符串长度 | 解码字符串长度 | subtoken 碎片化的粗代理 |
| 位置 | `l/L` | 直接可得 |
| prior-mode advantage | EXP-PT2 的 `m_raw(t_min)` | 直接可得 |
| 初始 rank | EXP-PT2 的 `rank_raw(t_min)` | 直接可得 |
| 初始 velocity alignment | EXP-PT3 的 `a_clean(t_min)`、`a_tok(t_min)` | 只在有效方向的位置上可用 |

## 未实现的 predictor（doc 列表里剩下的部分，需要额外基础设施）

- Contextual surprisal：需要外部自回归 LM 打分，没接。
- Local context strength：需要 EXP-PT4（上下文消融）。
- Oracle-rollout state distance：需要 EXP-PT7（paired oracle vs free-running）。
- Self-conditioning norm：在我们目前 Protocol-A-only、self-cond 恒为零的探测
  协议下没有意义（没有真实的 self-cond 状态可以测范数）。
- Jacobian/local gain estimate：需要 EXP-PT6 的扰动分支实验。

## 方法论决策

1. **按序列（sequence-level）分组做 train/val split，不是按位置**——同一条
   序列内的不同位置共享大量上下文，位置级别的随机切分会泄漏信息。这是直接照抄
   `EXP-07`→`EXP-07v2` 那次修复的教训（`docs/specs/EXP-07v2-spec.md` 一类的
   已有记录），这次在写这个新脚本的时候主动应用，而不是等犯错了再修。
2. **稀有类别合并成 "other"**（默认阈值 1%）——否则多元逻辑回归在几十个样本
   的类别上系数会非常不稳定。合并阈值和被合并的类别在输出 JSON 里都有记录。
3. **只报告"比多数类基线好多少"，不只报告绝对 accuracy**——一个 70% 准确率
   在某类占 70% 样本的数据集上毫无意义；每次都同时报告
   `majority_class_val_accuracy` 作为下限参照。

## 状态

**DONE（全部 4 个模型/checkpoint 都跑完）**

### 四模型对比

| model/ckpt | val_acc | majority baseline | 提升 | 最大系数特征 |
|---|---|---|---|---|
| ELF baseline | 0.772 | 0.717 | +5.5pp | log_freq（量级 ~0.6-1.0） |
| ELF kd_cr | **0.894** | 0.831 | +6.3pp | log_freq（量级 **~5-6**） |
| ELF kd2 | **0.887** | 0.832 | +5.5pp | log_freq（量级 **~3-6**） |
| LangFlow | 0.738 | 0.713 | +2.5pp | prior_mode_advantage（量级 ~2-9） |

**新发现**：KD checkpoint 上 `log_freq` 的系数量级（3-6）比 baseline（0.6-1.0）
大好几倍——对 KD 模型，token 是否常见几乎能决定性地预测它会落入哪个失败类别
（罕见→`no_emergence`/`wrong_mode_accumulation`，常见→`successful_monotonic`）；
baseline 上这个关系存在但弱得多、更分散在多个特征上。这和 EXP-PT1/PT2 里已经
看到的"KD 让分布两极化"的猜想是一致的一个新证据：KD 可能让模型更依赖频率这个
捷径（即使是在这套残差/去偏分析框架下）。

### ELF baseline（128 样本，46331/131072 个位置有 velocity 特征可用）

- val_acc=0.772，majority baseline=0.717（**+5.5pp**，比多数类基线好但差距不大——
  这些特征只解释了一部分失败模式，符合预期，不是过拟合的"完美预测"）。
- 系数里最突出的：`wrong_mode_accumulation` 随 `log_freq`（+0.63）、
  `initial_rank`（+0.39，起始排名差）、`position_frac`（+0.37，越靠后越容易）
  变大；`multiple_revision` 随 `log_freq` 变小（-0.97，**罕见词更容易反复翻转**）；
  `successful_monotonic` 随 `is_function_word` 变大（+0.53，功能词更容易一次性
  成功）——这些方向都和直觉、以及本项目已有的 func/content 相关发现
  （EXP-08v2/EXP-27v2）一致，是一个交叉验证。

### LangFlow（128 样本，9081/16384 个位置有 velocity 特征可用）

- val_acc=0.738，majority baseline=0.713（**+2.5pp**，比 ELF 更弱）。
- `prior_mode_advantage` 系数量级远大于其它特征（`no_emergence` 类
  +9.13！），⚠️ 这个"prior_mode_advantage 越高、越容易 no_emergence"的方向
  一开始看起来反直觉（raw margin 已经占优的位置怎么会残差从来不转正？），
  需要进一步检查——一个可能的解释是这类位置本身就是 padding/边缘位置（`m_raw`
  在这些位置的定义可能退化），需要在下一步里排除 padding 之后重新跑才能确认
  这不是又一次"pad token 污染"（见 EXP-PT1-spec.md 里记录的同类问题）。

## 脚本与输出

```text
experiments/phase_transition/analyze_failure_predictors.py
```

```text
results/phase_transition/<model>/<checkpoint>/failure_predictors_<label>.json
```

## 下一步

1. 补 kd_cr/kd2（等 EXP-PT3 跑完）。
2. 排查 LangFlow `prior_mode_advantage` 系数异常大的问题，优先检查是不是
   padding/特殊 token 位置没排除。
3. 如果要更贴近 doc 原文，需要补上 contextual surprisal（比较容易——接一个现成
   的 GPT-2/T5 语言模型打分即可）和真正的 POS tagger（比封闭词表更精细）。

## 严谨性补强：bootstrap CI（rigor audit 后，2026-07-26）

给 `analyze_failure_predictors.py` 加了逐位置 val 集正确性保存
（`failure_predictors_raw_<label>.npz`：`val_seq_idx`, `val_correct`,
`val_is_majority`），重跑全部 4 个模型（数字与修复前完全吻合），新增
`bootstrap_pt10.py` 按 held-out 序列重采样 2000 次（不需要重新拟合分类器，
只重采样哪些 val 序列参与 accuracy 平均）：

| model/ckpt | val_accuracy CI | majority_baseline CI | improvement CI | P(improvement>0) |
|---|---|---|---|---|
| ELF baseline | 0.769 [0.757,0.781] | 0.711 [0.695,0.727] | **+0.057** [+0.044,+0.072] | 1.000 |
| ELF kd_cr | 0.894 [0.889,0.898] | 0.831 [0.819,0.841] | **+0.063** [+0.052,+0.074] | 1.000 |
| ELF kd2 | 0.887 [0.882,0.892] | 0.832 [0.821,0.842] | **+0.055** [+0.045,+0.067] | 1.000 |
| LangFlow | 0.735 [0.717,0.752] | 0.708 [0.686,0.729] | **+0.027** [+0.018,+0.036] | 1.000 |

**结论：全部 4 个模型的"比多数类基线好"这个结论 CI 都不含 0，P(improvement>0)
在 2000 次重采样里全部是 1.000**——这套简单的多元逻辑回归确实提取到了超出
多数类基线的真实预测信号，不是过拟合或噪声撑出来的，即使是量级最小的
LangFlow（+2.7pp）也统计显著。详见
`results/phase_transition/<model>/<checkpoint>/bootstrap_pt10_full.json`。

## 补充：local-context-strength predictor（合并 EXP-PT4 数据，2026-07-26）

`docs/specs/EXP-PT10-spec.md`"未实现的 predictor"清单里列了"Local context
strength：需要 EXP-PT4（上下文消融）"——现在合并进来了。

**实现方式**：`analyze_failure_predictors.py` 新增 `--context_ablation_json`
参数，读取 EXP-PT4 的 `context_ablation_<label>.json`，计算
`local_context_gap = full_context.acc_per_seq - local_window_r1.acc_per_seq`
（对 PT4 自己的 t-grid 取均值），作为新特征列。

**⚠️ 关键限制（必须如实说明）**：PT4 只在稀疏的探针位置（`n_probes`=26，
间隔 `probe_spacing=40`）测量 accuracy，而且只保存到**逐序列**粒度
（`acc_per_seq`：形状 `(T, N)`，已经是该序列全部 26 个探针的平均值），
**没有保存逐个探针位置的准确率**。这意味着 `local_context_gap` 只能是一个
**逐序列**特征（"这条序列整体有多依赖局部上下文"），广播到该序列的**全部**
位置上，不是像其它特征那样的真正逐位置 predictor——同一序列内的所有位置
都拿到完全相同的值。

PT4 和这次分析用的是**同一批**序列（都调用同一个 adapter 的
`load_owt_sequences`，该方法按固定顺序流式读取数据集前 N 条，不受 seq_len
影响，经代码检查确认是确定性的），所以按序列下标对齐是有效的。⚠️
**LangFlow 例外**：PT4 严谨性修复后用的是 `seq_len=1024`（为了凑够 26 个
探针），而这里的 LangFlow 分析用的是 `seq_len=128`——底层文档相同，但
PT4 测量"上下文依赖"用的探针分布在最长到 1024 的范围内，比这里实际分析
的 L=128 窗口长得多，两者不是在测量"同一个窗口内"的上下文强度。ELF 双方
都是 `seq_len=1024`，没有这个问题。

**结果（全部 4 个模型）**：

| model/ckpt | val_acc（含新特征） | val_acc（不含，见前节） | `local_context_gap` 系数量级 |
|---|---|---|---|
| ELF baseline | 0.772 | 0.772 | −0.30 ~ +0.12（"other"类最大） |
| ELF kd_cr | 0.894 | 0.894 | −0.02 ~ +0.02 |
| ELF kd2 | 0.884 | 0.887 | −0.02 ~ +0.03 |
| LangFlow | 0.738 | 0.738 | −0.04 ~ +0.03 |

**一个诚实的负面结果**：加入这个特征后，`val_acc` 在四个模型上**几乎没有
变化**（kd2 甚至略微下降 0.3pp，在噪声范围内），而且 `local_context_gap`
在全部 6 类失败标签、全部 4 个模型上的标准化系数都很小（多数 <0.03，
baseline 上最大也只有 −0.30，远小于 `log_freq`（可达 3-6）或
`prior_mode_advantage`（可达 −2.5~+9）的量级）。**这个特征基本没有提供
额外的预测信号**。最可能的原因就是上面提到的限制：它是一个粗粒度的
逐序列广播值，同一序列内所有位置都相同，而失败模式的大部分变异发生在
**序列内部**（不同位置之间），一个不随位置变化的特征天然无法解释这部分
变异。**如果要让这个 predictor 真正有用，需要回去改
`intervene_context.py`，让它保存每个探针位置各自的准确率（而不是先对
26 个探针取平均），再把 `local_context_gap` 变成一个真正逐位置的特征**——
这是一个具体、可执行的后续工作项，目前的结果不支持"局部上下文强度能预测
失败模式"这个假设，但这更可能是特征粒度太粗导致的，而不是这个假设本身
被证伪。

详见 `results/phase_transition/<model>/<checkpoint>/failure_predictors_with_context.json`。

**加了 bootstrap CI 复核这个负面结果不是巧合**：对 `with_context` 版本重跑
`bootstrap_pt10.py`，四个模型的 `improvement_over_majority` CI 和不含
`local_context_gap` 的版本几乎完全一致（baseline +0.058[+0.044,+0.073] vs
之前 +0.057[+0.044,+0.072]；kd2 +0.052[+0.041,+0.064] vs 之前
+0.055[+0.045,+0.067]；其余两个模型几乎不变），**P(improvement>0)全部仍是
1.000**——确认加入这个特征既没有帮助也没有伤害整体预测能力，"无额外信号"
这个结论本身是稳健的，不是某一次拟合的偶然。
