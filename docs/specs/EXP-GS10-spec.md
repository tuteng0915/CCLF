# EXP-GS10 Spec — Global Failure Predictors

## 背景与地位

原始 doc 第 14 节 GLOBAL-10，P1 最后一项，依赖 GLOBAL-7（`EXP-GS7`）的自由生成基础设施。
原始 doc 要求：8 类 failure label（successful transition / global indecision /
wrong-basin selection / fragmented basin / scaffold failure / lexical crystallization
failure / premature lexical locking / oracle-success-rollout-failure），13 个 predictor
feature，多元 logistic regression + survival analysis + sequence-grouped CV。

## 0. 大幅缩小范围的理由

8 类 failure label 需要**独立于本实验的 ground-truth 标注**（比如人工判断"这条自由生成是不是
选错了 basin"），而本仓库的自由生成没有外部参照文档（GS7 已经说明：clean endpoint 就是
生成自己的终点，不存在"正确答案"）。在没有外部标注的前提下，原始 doc 的 8 类细分标签**无法
被客观地打上**——勉强用模型自己的输出去定义"是否选对了 basin"是循环论证。

**缩小后的可行版本**：只用**不需要外部参照**、可以从生成文本本身客观计算的一个粗粒度代理——
**退化文本检测**（repetition rate，n-gram 重复率），这是语言模型生成质量评估里最标准、最
不需要外部标注的失败信号（本仓库其它实验，如 EXP-33/34/35，就是用类似的"退化文本"判据
把整批结果标记为 INVALID）。用这个信号加上 GS7 已经算出的 `G_token(rollout)` 终值，构造一个
**3 类粗粒度结果标签**（而不是原始 doc 的 8 类）：

1. **healthy**：重复率低、且 `G_token(rollout)@t=0.85` 达到较高水平；
2. **degenerate**：重复率高（陷入重复 n-gram 循环）；
3. **slow/incomplete**：重复率不高，但 `G_token(rollout)@t=0.85` 仍然很低（既没退化，
   也没恢复出稳定内容——更接近原始 doc 的"global indecision"类别，但不做更细的
   sub-classification）。

这三类分别粗略对应原始 doc 8 类里的"successful"、"degenerate 相关的几类"、"indecision/
scaffold failure 相关的几类"，但**不做更细的区分**（第 4 节已知简化）。

## 1. Predictor Features（从原始 doc 13 个里选可以直接算的 5 个）

复用 GS7 每条轨迹在 checkpoint t-grid 上已经算出的量：

1. `early_G_token`：`G_token(rollout)` 在早期 t（`t=0.28`）的值——对应原始 doc 的
   "early global probe confidence"（这里用 token probe 而不是 topic probe，因为 GS1/GS2
   已经确认 topic 指标不可靠）。
2. `min_CKA`：整条轨迹上 `CKA(oracle,rollout)` 的最小值——对应"oracle–rollout global
   distance"。
3. `r_eff_descent_rate`：`r_eff(rollout)` 从 `t=0.05` 到 `t=0.85` 的下降量
   （`r_eff@0.05 - r_eff@0.85`）——对应"effective rank"类特征。
4. `mid_G_token`：`G_token(rollout)@t=0.65`——补充一个中段读数。
5. `repetition_rate`：最终解码文本的 4-gram 重复率（简单启发式：重复 4-gram 数 / 总
   4-gram 数）——不在原始 doc 列表里，是本 pilot 为了打标签而加的必要补充。

## 2. 分析

原始 doc 建议 multinomial logistic regression / survival analysis / sequence-grouped CV /
calibrated probability / feature ablation——**pilot 规模（n=16）完全不支撑**任何一种。
只做**描述性统计**：按 3 类结果分组，报告每组的 4 个 predictor feature 均值/标准差，
不拟合任何分类器，不做显著性检验（延续本项目"先给数字，不自动下结论"的一贯做法）。

## 3. 数据与规模（pilot）

- ELF baseline，`eval_exp37c_baseline.yml`（1024-token）。
- `n_samples=16` 条独立自由生成轨迹（比 GS7 的 8 条多一倍，为分组统计留一点样本，但仍然
  远不足以做真正的预测模型）。
- checkpoint t-grid 与 GS7 相同。

## 4. 已知简化

1. ⚠️ **3 类粗粒度标签**代替原始 doc 的 8 类细分——不做 wrong-basin-selection /
   fragmented-basin / scaffold-failure / premature-locking 的区分，因为这些都需要外部
   ground truth 或更复杂的人工判读，超出本 pilot 范围。
2. ⚠️ `repetition_rate` 是本 pilot 为了打标签新引入的启发式指标，不在原始 doc 的
   feature 列表里。
3. ⚠️ 只做描述性分组统计，不拟合任何预测模型，`n=16` 太小，组内方差可能很大。
4. 只测 ELF baseline。

## 5. 脚本与输出

```text
experiments/global_state/analyze_global_failures.py
```

```text
results/global_state/<model>/<checkpoint>/global_failures_<label>.json
```

## 状态

**Pilot DONE — 零方差，taxonomy 本身在这个设置下不具区分度**（ELF baseline，n=16 条自由生成
轨迹，`n_steps=32`（标准生成质量），GPU1，
`logs/global_state/gs10_elf_baseline_pilot.log`，输出
`results/global_state/elf/baseline/global_failures_pilot.json`）。

## Results

**全部 16 条轨迹都被标记为 `healthy`**（`repetition_rate` 全部 < 0.3，`late_G_token`
全部 > 0.5，范围 0.514–0.829）——`degenerate` 和 `slow_incomplete` 两组 `n=0`。

**解读**：

1. **本次 pilot 没有失败案例，因此 GLOBAL-10 要求的"分组对比 predictor feature"完全做不了**
   （只有一组，没有对照）。这不是代码 bug（smoke test 用 `n_steps=8` 时，全部 4 条轨迹反而
   都被标成 `slow_incomplete`——taxonomy 本身是敏感的，只是在 `n_steps=32` 这个标准
   生成质量设置下，ELF baseline 在 1024-token、`n=16` 的样本量下**没有观测到任何退化或
   长期不收敛的自由生成**）。
2. 这本身是一个有信息量的负结果，和本仓库已有发现吻合：EXP-37a/b 已经证明 ELF baseline
   在 1024-token、None（无 DF）设置下 PPL 表现稳定；EXP-32/EXP-37 系列的 step-count sweep
   显示**步数越少、质量越差**（如 8 步时 PPL 明显更高）。本 pilot 用的是标准 32 步，落在
   "生成质量足够好、失败率天然很低"的区间——要观测到原始 doc 想研究的那种 failure 多样性，
   需要主动制造更容易失败的条件，而不是在默认最优设置下守株待兔。
3. `healthy` 组内部的 feature 分布本身也有一些跨轨迹的变异（`late_G_token` 从 0.514 到
   0.829，`min_CKA` 从 0.808 到 0.950），如果把"healthy"内部按 `late_G_token`/`min_CKA`
   排序，高低两端的轨迹在特征上确实有差异——但因为都被同一个粗糙阈值归为一类，无法做本来
   想要的分组统计。

## 下一步（比原计划更明确、更可执行）

1. **用更容易失败的条件重新采样**，制造真正的多样性，而不是继续在 `n_steps=32` 上加样本量：
   - **降低 `n_steps`**（比如 8，smoke test 已经证实这会让轨迹整体偏向 `slow_incomplete`）；
   - 复用 EXP-33/34/35 已知会产生退化文本的设置（无 tmin gate 的 dec_sc + DF）——但那些
     实验用的是不同的 sampling config 体系，接入本 GS 系列的 `solver_step` 循环需要额外
     适配 dec_sc/DF 的干预点，本 pilot 没做。
   - 对 kd2/kd_cr 也跑一遍（EXP-37c 已经发现 kd2+DF 在 1024-token 下"全面退化"，如果用
     同样条件跑 GS10，应该能拿到货真价实的 `degenerate` 样本）。
2. 拿到跨越三个标签的样本后，再回头做原计划的分组统计，甚至尝试 doc 建议的
   multinomial logistic regression（需要 `n` 明显大于 16）。
3. 保留当前的粗粒度 3 类 taxonomy，不必强行细分成原始 doc 的 8 类——除非能找到不依赖
   外部标注、可以客观计算的方式来区分"wrong-basin"和"fragmented-basin"这类更细的失败模式。
