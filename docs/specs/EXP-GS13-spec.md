# EXP-GS13 Spec — Context-Only Global-to-Local Intervention (P0-3)

## 背景与地位

用户审阅指出 `EXP-GS8` 的根本问题：干预 `Z_t' = Z_t + delta*u` 是把同一个方向向量加到
**每一个位置，包括正在测量 margin 的目标位置本身**。因此 GS8 观察到的"往正确 topic 方向推
margin 就上升"，完全可以用最短路径解释：

```
u_topic -> z_i (目标位置自己的状态被直接改了) -> ell_i(y_i)
```

不需要任何"全局/上下文信息经过其它位置、再影响目标位置"的因果链。这不是 global-to-local
mediation，是 direct feature intervention。

本实验（P0-3）做真正的因果链检验：**目标位置 `i` 的状态完全不动，只扰动其它位置**，
再看目标位置的 margin 是否仍然变化。只有这样，margin 变化才必须经过"其它位置的表示改变
→ 通过 self-attention 传给位置 `i`→ 改变位置 `i` 的 logit"这条路径，才配得上"global/context
→ local"这个说法。

## 1. 设计

沿用 `EXP-GS8` 已经验证的基础设施（topic probe 方向构造、`f_i` 默认竞争者定义、
`eta`/alpha 扰动范数约定），只改一件事：**扰动 mask 排除目标位置**。

对每个 `(文档, 目标位置 i)`：

```
perturb_mask[j] = 1   if j is valid and j != i
perturb_mask[i] = 0   (目标位置完全不动)

Z_t'[j] = Z_t[j] + alpha*eta*||Z_t||_F/sqrt(n_valid-1) * u * perturb_mask[j]   for all j
Z_t'[i] = Z_t[i]   (逐位置显式保证，不依赖 perturb_mask 数值精度)
```

`u` 的构造（correct/wrong/orthogonal/random 四个方向）和范数缩放公式完全复用
`EXP-GS8`（包括 GS8 已经修复过的"广播到 L 个位置需要除以 sqrt(n_valid)"的教训，这里
自然是 `sqrt(n_valid - 1)`，因为少了一个位置）。

## 2. 目标位置采样

对**成本**的现实考虑：如果对每个文档的**所有**有效位置逐一做"留一个不动"的扰动，
`N` 篇文档 × `L≈1000` 个位置 × 4 方向 × 2 alpha 的前向次数完全不可行。改为：每篇文档
**随机采样 8 个有效位置**作为目标位置（而不是全部 ~1000 个），把"文档 × 采样位置"当作
新的批维度，一次性 batch 前向（同一 `(direction, alpha)` 组合下，不同目标位置的扰动
mask 不同，但可以放进同一个 batch 里一起过模型）。

## 3. 数据与规模（pilot）

- 复用 `EXP-GS8` 同款 topic probe 拟合流程（`n_samples=64`，`t=0.28`，document-level
  70/30 split，用完整训练集拟合 8 类 topic probe，和 GS8 保持可比）。
- 但**只对测试集里的一个子集（12 篇文档）**做本实验的目标位置扰动（每篇 8 个采样位置，
  共 96 个"文档×位置"组合），控制算力（12×8=96，比 GS8 原本 19 篇文档×所有位置的隐式
  全量参与更聚焦，但每个方向×alpha组合的前向调用数更省，因为一次 batch 完成 96 个）。
- `eta=0.03`（沿用 GS2/GS8 校准值），`alpha ∈ {-1.0, 1.0}`（pilot 先用两端点，控制算力，
  如果出现有意义的信号再补 `±0.5`）。
- 四个方向：`correct`/`wrong`/`orthogonal`/`random`，定义完全复用 GS8。

## 4. 判定

和 GS8 相同的判定框架，但现在因果链更严格：

- `correct` 方向、`alpha>0`：目标位置（完全没被直接扰动）的 `delta_margin` 是否依然
  显著为正？如果是，说明改变**其它位置**的 topic 相关表示，确实通过某种机制（大概率是
  self-attention）影响了目标位置的证据——这才是真正的 global-to-local 因果链。
- `orthogonal`/`random` 对照：是否依然明显弱于 `correct`/`wrong`。
- 如果本实验里 `correct`/`wrong` 的效应相对 GS8（direct intervention）大幅减弱甚至消失，
  说明 GS8 原来测到的效应主要就是"直接改了目标位置自己"这个平凡路径，本实验才是对
  "全局→局部"因果链更诚实的检验。

## 5. 已知简化

1. ⚠️ 每篇文档只采样 8 个目标位置（不是全部），采样是否有代表性未知。
2. ⚠️ `alpha` 只测两端点（±1.0），比 GS8 原来的四点更稀疏。
3. ⚠️ `n_docs=12`（做扰动的子集），比 GS8 原来的 19 篇测试集更小。
4. Pilot 规模，数字仅用于判断方向。

## 6. 脚本与输出

```text
experiments/global_state/intervene_context_only.py
```

```text
results/global_state/<model>/<checkpoint>/intervene_context_only_<label>.json
```

## 状态

**Pilot DONE — GS8 的核心发现在更严格的检验下依然成立**（ELF baseline，n=64（probe 拟合）
+ 12 篇测试文档子集 × 8 个采样位置，`t=0.28`，`eta=0.03`，4 方向 × 4 alpha，GPU1，
`logs/global_state/gs13_elf_baseline_pilot.log`，输出
`results/global_state/elf/baseline/intervene_context_only_pilot.json`）。

## Results（pilot：12 docs × 8 positions = 96 个"文档×目标位置"组合，目标位置完全未被扰动）

| direction | a=−1.0 | a=−0.5 | a=+0.5 | a=+1.0 |
|---|---|---|---|---|
| correct | +0.095 | −0.068 | +0.471 | **+1.227** |
| wrong | +1.227 | +0.471 | −0.068 | +0.095 |
| orthogonal | +0.244 | +0.088 | −0.041 | −0.036 |
| random | +0.523 | +0.217 | −0.527 | −0.326 |

**解读**：

1. **`correct` 方向的效应基本存活**：`alpha=+1.0` 时 `delta_margin=+1.227`，是 GS8
   原始（直接扰动目标位置）实验同一 alpha 下 `+1.718` 的约 71%——**在目标位置自身状态
   完全没有被触碰的前提下，只扰动其它位置，目标位置的 margin 依然明显上升**。这是本轮
   审阅要求的关键检验，结果支持"存在从上下文位置到目标位置的因果传递"（大概率通过
   self-attention），而不只是 GS8 里"直接改了目标位置自己"这个平凡路径的产物。
   `wrong` 方向依然精确镜像 `correct`（`wrong@a=-1.0 = +1.227 = correct@a=+1.0`），
   内部一致性检验通过。
2. **`orthogonal` 对照依然明显更弱**（最大量级 `0.244`，约为 `correct` 峰值 `1.227` 的
   20%，比例上甚至比 GS8 原始实验里的 29% 更干净）——支持"效应是方向特异的，不是任意
   扰动都会引起类似幅度的 margin 变化"。
3. **`random` 对照比 `orthogonal` 更强、也更不干净**（`-0.527` 到 `+0.523`，量级约为
   `correct` 峰值的 43%，而且方向和 `correct` 相反——`random` 在 alpha 为负时反而是正的，
   在 alpha 为正时是负的）。⚠️ 因为 `random` 方向对每篇文档是固定采样一次、跨所有 alpha
   复用（不是每个 alpha 单独重新采样），这一个特定随机方向恰好和某种 margin-相关的几何
   有一定程度的（负）关联，这是单次随机抽样的噪声，不代表"随机方向系统性地有负效应"。
4. **相比 GS8，`correct` 的峰值效应打了七折**（1.227 vs 1.718），符合预期方向——去掉了
   "直接命中目标位置"这个最短路径后，纯粹通过上下文中介的效应确实应该比直接扰动更弱一些，
   但没有弱到消失，说明 GS8 的原始效应**不是**完全由这个 confound 解释的，其中确实包含
   一个真实的、通过上下文传递的因果分量。

## 结论

**用户对 GS8 的批评是对的（原始设计确实混入了 direct feature intervention），但修正后
再测，核心因果链主张本身站得住**：往正确 topic 方向推上下文位置，确实会经过某种传递机制
（大概率是 self-attention）改变一个完全没被直接触碰的目标位置的 true-token margin，且
这个效应明显强于对照组。这是本轮 P0 系列返工里少数"修正后依然支持原始结论"的例子（另一个
是 GS3 的"token 集中在残差"部分），值得在论文里作为一个经过因果检验的发现来写，同时注明
效应量级从直接扰动的 `+1.718` 降到上下文中介的 `+1.227`（约 71%）。

## 正式规模复现（n=128 probe 拟合，24 docs × 12 位置 = 288 组合，`alpha∈{-1,-0.5,0.5,1}`，
`random`/`orthogonal` 已修复为逐 `(doc,alpha)` 重新采样，
`logs/global_state/gs13_elf_baseline_formal.log`，
`results/global_state/elf/baseline/intervene_context_only_formal.json`）

| direction | a=−1.0 | a=−0.5 | a=+0.5 | a=+1.0 |
|---|---|---|---|---|
| correct | +0.505 | +0.234 | +0.180 | +0.817 |
| wrong | +0.817 | +0.180 | +0.234 | +0.505 |
| orthogonal | +0.129 | −0.106 | +0.126 | +0.038 |
| random | −0.125 | +0.055 | −0.116 | +0.032 |

**解读（如实报告，比 pilot 更复杂）**：

1. **`correct`/`wrong` vs `orthogonal`/`random` 的量级对比依然成立**：`correct`
   的四个 alpha 点数值范围 `[0.18, 0.82]`，`orthogonal`/`random` 的范围只有
   `[-0.13, +0.13]`——前者始终明显大于后者，支持"沿着 topic 方向的扰动比随便一个方向
   的扰动效应更大"这个核心判据。
2. **`random` 修复后确实变得干净了**：不再像 pilot 里那样系统性地和 `correct` 反向，
   而是在 0 附近小幅波动（`-0.125` 到 `+0.032`），符合一个正常空 baseline 该有的样子——
   证实了 pilot 观察到的"random 方向异常"确实是"每篇文档固定复用同一次随机抽样、跨全部
   alpha"这个实现问题造成的，不是真实效应。
3. **⚠️ 但 pilot 里"随 alpha 单调"的干净剂量-反应曲线在正式规模下没有复现**：
   `correct` 在 `alpha∈{-1.0,-0.5,0.5,1.0}` 上的值是 `[+0.505, +0.234, +0.180,
   +0.817]`——两端都是正的，中间反而更低，是一个 U 形而不是单调曲线。`wrong` 精确
   镜像（`wrong(a) = correct(-a)`，内部一致性检验依然通过），所以这不是代码 bug，
   是真实的、更复杂的剂量-反应关系。可能的解释：沿着"topic probe 方向"做**大幅**扰动
   （`|alpha|=1.0`）时，无论往哪个方向推，都可能把上下文状态推出某种"模糊/不确定"区域、
   进入某个更"决断"的区域，从而提升目标位置的证据——而不是"正确方向增证据、错误方向减
   证据"这种线性可加的因果故事。这比 pilot 暗示的更弱，需要如实调整措辞。

## 结论（修正后）

**"correct/wrong 方向 vs 正交/随机对照"这个较弱的因果主张在正式规模下站得住**（上下文
中介的效应确实存在，且方向轴特异）；**"沿 topic 方向的扰动幅度和 margin 变化方向线性
可加"这个更强的主张不成立**（U 形而非单调）。这是本轮返工里"部分存活、部分需要弱化"的
又一个例子，和 GS3/GS12 的"两条结论拆开验证"是同一种模式。

## 下一步（第 1 点已修复，见上方"正式规模"）

1. ✅ **已修复**：`orthogonal`/`random` 方向现在对每个 `(doc, alpha)` 组合独立重新
   采样（原 pilot 是每篇文档固定一个、跨全部 alpha 复用），排除"单次抽样恰好有系统性
   关联"这个噪声来源。`correct`/`wrong` 保持确定性（直接由 probe 权重决定，不需要
   随机采样，重新采样也不会改变它们）。
2. 扩大 `n_docs_subset`（当前 12）和 `n_positions_per_doc`（当前 8），提高统计效力。
3. 补充 `alpha=±0.25`，确认 `correct` 方向在小 alpha 时的单调性（pilot 里
   `alpha=-0.5` 出现了小幅负值 `-0.068`，和 GS8 类似的非严格单调现象，可能需要更大样本
   才能看清）。

## LangFlow 过拟合修复（2026-07-27，严谨性自审驱动）

和 GS8 完全相同的根因（见 EXP-GS8-spec.md 对应章节）：LangFlow 换 `t=0.65` 后 topic
probe `test_acc` 仍是 0.158，是 45 篇训练文档对 8 类分类器过拟合。用同样的修复
（`--C 0.05 --n_samples 200 --t 0.65`）重跑：`test_acc=0.217`（chance=0.125）。

bootstrap CI（context-only intervention，n=160 per (direction, alpha)）：

| direction | a=−1.0 | a=+1.0 |
|---|---|---|
| correct | +0.061 [−0.043,+0.171] | +0.150 [+0.044,+0.265] |
| orthogonal | +0.023 [−0.021,+0.072] | +0.010 [−0.029,+0.053] |
| random | +0.010 [−0.036,+0.054] | +0.037 [−0.007,+0.080] |

**这是本轮修复里最不干净的一个结果**：a=+1.0 时 `correct` 的 CI 明显高于 `orthogonal`/
`random`（两者 CI 都含 0），支持有信号；但 a=−1.0 时 `correct` 自己的 CI 就含 0，和
对照组统计不可区分。信号只在一个 alpha 方向上站得住，比 ELF 正式规模的结果（`correct`
四个 alpha 点全部明显高于对照组，见本文件此前章节）弱得多、也不对称得多。引用这个
LangFlow 结果时必须注明"仅单侧显著，样本量修复后信号依然偏弱"，不能类比 ELF 的表述。

详见 `logs/global_state/gs13_langflow_bigN_reg.log` 和
`results/global_state/langflow/baseline/intervene_context_only_bigN_reg.json`。
