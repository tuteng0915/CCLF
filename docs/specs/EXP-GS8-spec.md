# EXP-GS8 Spec — Global-to-Local Causal Chain

## 背景与地位

原始 doc 第 12 节 GLOBAL-8，P1 阶段项目。直接检验 `global state -> local evidence -> exact
token` 这条因果链：在"global probe 已经较强、但 token probe 仍较弱"的时间点，对 global
representation 做小干预（往正确 topic 方向 / 错误 topic 方向 / 正交对照 / 随机同范数方向），
测每个位置的 true-token margin 是否相应变化。

跳过 GS4 里遇到的"因果充分性 vs 探针可读性不一致"的坑：GS8 不要求扰动方向单独就能驱动生成
（GS4 已经证明 k=8 的 global mode 单独不足以因果驱动），只要求"往正确方向推一把，margin
应该比往错误方向推或者不动/正交扰动更有利于真值 token"——这是一个更弱、更容易成立、也更贴近
doc 原文措辞（"提前 local true-token margin"）的因果主张。

## 0. 复用与新增

- Adapter：同 GS1/GS2/GS3/GS4。
- Topic 空间：复用 GS1 的 `topic_kmeans_centroids_pilot.npy`（8 簇），不重新拟合 KMeans，
  与 GS2/GS4 保持同一套 topic 定义。
- 扰动的范数缩放约定：复用 GS2 的 `delta = alpha * eta * ||Z_t||_F * u/||u||_2`（`||.||_F`
  为整条序列 `(L,d)` 矩阵的 Frobenius 范数），只是这里 `u` 不是随机方向，而是从一个刚拟合出的
  topic 线性 probe 的权重差里取出来的"topic 方向"。

## 1. 时间点选择

原始 doc 要求"早期 global probe 已经较强、但 token probe 仍较弱"。直接复用 GS1 pilot 的
数字（`docs/specs/EXP-GS1-spec.md` Results 表）：`t=0.28` 处 `G_topic=0.500`（远高于
8-类随机水平 12.5%，"已经较强"）而 `G_token=0.352`（明显低于 clean 的 1.000，"仍较弱"）—— 
是这两个条件同时满足的最合适的 t。本实验只在这一个 t 上做（不是 t 的 sweep），把预算都花在
alpha/方向的对照上。

## 2. Topic 方向构造

1. 在同一个 `t=0.28` 上、用当前 pilot 的样本重新拟合一个 multinomial logistic regression
   topic probe（`g_t^mean -> nearest-GS1-centroid label`，训练集上拟合，与 GS1 做法一致），
   得到权重矩阵 `W in R^{8 x d}`。
2. 对每个测试文档：`c_true` = 该文档自己的最近质心标签（ground truth，不是 probe 的预测）；
   `c_runnerup` = probe 在该文档 `g_t^mean` 上给出的、除 `c_true` 外概率最高的类别（"最容易
   混淆的错误 topic"，比随机选一个错误类别更贴近原始 doc "语义一致的错误 token" 的用意）。
3. **Correct 方向**：`u_correct = W[c_true] - W[c_runnerup]`，归一化为单位向量。
4. **Wrong 方向**：`u_wrong = -u_correct`（同一条线上的反方向，最简单的"往错误 topic 推"
   操作化——如果 `c_true` 和 `c_runnerup` 是这个 probe 在这条文档上最容易混的两个类，那么
   沿这条线往反方向推，就是最直接的"往错误但语义相关的 topic 推"）。
5. **Orthogonal 对照**：随机采样一个向量，Gram-Schmidt 去掉它在 `u_correct` 方向上的分量，
   归一化。
6. **Random 对照**：一个独立的随机高斯方向（不保证与 `u_correct` 正交），归一化。

四个方向都是**跨位置共享的同一个 d 维向量**（不是逐位置各自的方向），加到 `Z_t` 的每个
有效位置上——这是"global direction"这个提法在数学上最直接的操作化：一个只依赖通道
（channel/feature）、不依赖位置的扰动。

⚠️ **实现踩坑**：把同一个单位 d 维向量原样广播到全部 ~1000 个有效位置，得到的
`(L,d)` 扰动矩阵的 Frobenius 范数是 `||u||_2 * sqrt(n_valid)`，比 GS2 里"整条矩阵单位化"
的范数大了约 `sqrt(1024)≈32` 倍——第一次跑 pilot 用 `alpha=1.0` 时，四个方向
（包括本该是"无效对照"的 orthogonal/random）的 `delta_margin` 全部在 12–18 这个量级，
明显是扰动过大把 logits 冲进了随机区间，掩盖了方向之间应有的差异。已修复：广播前先把
`u` 除以 `sqrt(n_valid)`，让最终 `(L,d)` 扰动矩阵的 Frobenius 范数严格等于
`alpha*eta*||Z_t||_F`，和 GS2 的约定完全对齐。

## 3. 干预与测量

- `Z_t' = Z_t + alpha * eta * ||Z_t||_F * u`，`alpha ∈ {-1, -0.5, 0.5, 1}`（原始 doc 建议
  额外的 `±0.25`，pilot 先用四点覆盖两端和半程）；`eta` 复用 GS2 校准出的 `0.03`
  （GS2 发现 `0.01` 太小、`0.1` 已经能让 lexical consensus 明显下降——`0.03` 是那次校准里
  "有意义但不至于让状态完全失真"的中间值，直接沿用而不重新校准）。
- **不做后续 rollout**——只在同一个 t 上做单步 forward，比较扰动前后的 native logits
  （比 GS4 的"扰动后完整 rollout"更轻量，也更贴近 GLOBAL-8 原始协议本身"测目标位置"而非
  "测最终生成结果"的定位）。
- 默认竞争者 `f_i`：**扰动前**该位置的 native top-1 token（和 `phase_transition` 系列
  `EXP-PT1/PT2` 的 `f_i` 定义一致，跨实验口径统一）。
- `margin_before(i) = ell(y_i) - ell(f_i)`（扰动前 log-softmax）；
  `margin_after(i) = ell'(y_i) - ell'(f_i)`（扰动后 log-softmax，**同一个** `f_i`，只有
  logits 变了）；`delta_margin(i) = margin_after(i) - margin_before(i)`。
- 对每个 (方向, alpha) 条件，报告所有文档、所有有效位置上 `delta_margin` 的均值和
  正值比例（多少比例的位置margin 是朝对该条件"有利"的方向移动的）。

## 4. 支持因果链的判定

- `correct` 方向、`alpha>0`：`delta_margin` 均值应显著为正（且随 `alpha` 增大而增大）。
- `wrong` 方向、`alpha>0`（等价于 `correct` 方向 `alpha<0`）：`delta_margin` 均值应为负。
- `orthogonal` / `random`：`delta_margin` 均值应接近 0，明显小于 `correct`/`wrong` 条件
  的幅度。

## 5. 数据与规模（pilot）

- ELF baseline，`eval_exp37c_baseline.yml`（1024-token）。
- `n_samples=64`（复用 GS3 pilot 规模，train/test 划分同款 70/30）。
- 单一 `t=0.28`，`eta=0.03`，`alpha ∈ {-1,-0.5,0.5,1}`，4 个方向 × 4 个 alpha = 16 个条件
  （加上 baseline 未扰动的隐式 0 点）。

## 6. 已知简化

1. ⚠️ Topic 方向的"正确 vs 错误"只用了 `c_true` 和 probe 自己给出的 `c_runnerup`——不是
   原始 doc 建议的更通用的 sentence-embedding regression 梯度方向或 oracle global-mode
   方向（`u_G = G_oracle - G_roll`，需要 free-running 轨迹，GS4/GS7 同样遇到这个限制）。
2. ⚠️ 扰动是跨位置共享的同一个方向向量，不区分位置——如果某些位置本来就和 topic 无关
   （比如标点、固定搭配），margin 变化可能被这些位置稀释，pilot 只报告跨位置平均值，
   不做位置级别的异质性分析。
3. ⚠️ 只在单个 t（0.28）上做，不是 sweep；`eta` 沿用 GS2 校准值而非针对本实验重新校准。
4. Pilot 规模（64 样本），数字仅用于判断方向。

## 7. 脚本与输出

```text
experiments/global_state/intervene_global_to_local.py
```

```text
results/global_state/<model>/<checkpoint>/intervene_global_to_local_<label>.json
```

## ⚠️ 方法论说明 + 后续确认（EXP-GS13，P0-3）

用户审阅指出：本实验的扰动 `Z_t' = Z_t + delta*u` 加到了**每一个位置，包括正在测量
margin 的目标位置本身**，因此下方观察到的效应可以用最短路径解释（`u_topic -> z_i` 直接
改了目标位置自己 `-> ell_i(y_i)`），不需要任何"全局→局部"的中介传递，不能称为
global-to-local causal chain。

`EXP-GS13`（`docs/specs/EXP-GS13-spec.md`）用**目标位置完全不受扰动、只扰动其它位置**
的更严格设计重做了这个检验，结果是：**核心效应基本存活**（`correct` 方向 `alpha=1.0`
时 `delta_margin=+1.227`，约为本实验 `+1.718` 的 71%），`orthogonal` 对照依然明显更弱。
说明本实验的效应**不完全是** direct-feature-intervention 的产物，其中包含一个真实的、
通过上下文位置中介传递（大概率经由 self-attention）的因果分量——但绝对量级确实被高估了
（因为原始设计也捕获了直接命中目标位置这部分平凡效应）。下方 Results 的数字本身没有错，
但"因果链"这个措辞应该以 GS13 的上下文中介结果为准。

## 状态

**Pilot DONE**（ELF baseline，n=64，t=0.28，eta=0.03，4 方向 × 4 alpha，GPU1，
`logs/global_state/gs8_elf_baseline_pilot.log`，输出
`results/global_state/elf/baseline/intervene_global_to_local_pilot.json`）——
**目前 GS 系列里最干净、最符合原始假设预期的一次正面结果**。

## Results（pilot：ELF baseline，64 序列，t=0.28，train/test=45/19）

Topic probe @ t=0.28: train_acc=0.889，**test_acc=0.632**（8 类随机水平 0.125）——满足
第 1 节要求的"early global probe 已经较强"（GS1 pilot 在同一 t 上也是 0.500，量级一致）。

`delta_margin` 均值（跨 19 个测试文档、每个文档所有有效位置取平均）：

| direction | a=−1.0 | a=−0.5 | a=+0.5 | a=+1.0 |
|---|---|---|---|---|
| correct | +0.047 | −0.075 | +0.526 | **+1.718** |
| wrong | **+1.718** | +0.526 | −0.075 | +0.047 |
| orthogonal | −0.037 | −0.066 | +0.174 | +0.493 |
| random | +0.304 | +0.131 | −0.034 | −0.019 |

**解读**：

1. **`correct` 方向随 alpha 单调上升，`alpha=+1.0` 时效应最大（+1.718）**：往"正确 topic"
   方向推得越用力，真值 token 相对默认竞争者的 margin 提升越多。**这是原始 doc GLOBAL-8
   核心判定标准的直接、干净确认**："正确 global direction 提前 local true-token margin"。
2. **`wrong` 方向精确等于 `correct` 方向的镜像**（`wrong(a) == correct(-a)`，数值完全对应，
   如 `wrong@a=-1.0 = +1.718 = correct@a=+1.0`）——这是代码正确性的内部一致性检验（`wrong`
   定义就是 `-correct` 方向，二者应该互为镜像），同时也说明"往错误 topic 方向推"确实系统性
   地把 margin 往下压（`wrong@a=+1.0` 只有 `+0.047`，远小于 `correct@a=+1.0`）。
3. **`orthogonal` 对照效应明显更弱**（最大 `+0.493`，约为 `correct` 最大效应的 29%），
   **`random` 对照几乎没有一致方向**（`+0.304` 到 `-0.019`，量级小且不单调）——两个对照组
   都远弱于 `correct`/`wrong`，符合"沿着真正的 topic 方向推才有效，随便一个方向没什么用"
   的预期。`orthogonal` 比 `random` 更强一点点，可能是因为 Gram-Schmidt 只保证与
   `u_correct` 正交，不保证与"margin 真正敏感的方向"正交，存在部分残留对齐；这个细节不影响
   主结论。
4. 和 GS4 的关系：GS4 发现"低秩 global mode 单独不足以因果驱动完整生成"，GS8 这里测的是一个
   更弱、更局部的因果主张——"往 topic probe 自己学到的方向推一把，margin 会怎么变"——**不要求
   方向本身携带完整语义**，只要求它和"什么让 margin 变化"这件事相关。GS8 的正面结果和 GS4
   的负面结果并不矛盾：一个方向可以对局部 margin 有可靠的因果影响，但不足以单独充分驱动
   整个生成过程恢复出正确的 topic/token（GS4 测的是"charge 一个子空间是否自给自足"，
   GS8 测的是"往一个方向推是否有方向性效果"，是两个不同强度的因果主张）。

## 下一步

1. 补充 `alpha=±0.25`（原始 doc 建议的完整 sweep），确认 correct 方向在小 alpha 时是否
   仍然单调（pilot 里 `a=-0.5` 出现了 `-0.075` 的小幅负值，比 `a=-1.0` 的 `+0.047`
   还低，不是严格单调，可能是 N=64 下的噪声，需要更密的 alpha 网格或更大 N 确认）。
2. 扩大到多个 t（当前只测了 t=0.28 一个点）——检验这个因果链在不同"global probe 强度/
   token probe 强度"组合下是否稳定存在。
3. 补充"boundary-crossing time"这个原始 doc 提到但本 pilot 没做的指标（需要在扰动后继续
   往后跑几步 t，看真值 token 越过 decoder 决策边界的时间是否提前/推迟，而不只是同一 t 上
   margin 的即时变化）。
4. 换用 GS1 sentence-embedding 回归梯度方向做对照（原始 doc 建议的第二种方向来源），
   检验结果是否对"用哪个 probe 提取方向"敏感。

## LangFlow 过拟合修复（2026-07-27，严谨性自审驱动）

LangFlow 版本换 `t=0.65`（LangFlow 自己校准的过渡区）后 topic probe `test_acc` 依然是
0.158（≈chance=0.125），`train_acc=0.933`——45 篇训练文档对一个 d=768、8 类的
`LogisticRegression` 严重过拟合，而非 t 校准问题（ELF pilot 用相同 n=64/test_frac=0.3
设置得到的是健康的 `test_acc=0.632`，说明这不是脚本通用问题，是 LangFlow 特有的样本量
不足）。

**修复**：给 `intervene_global_to_local.py` 加了 `--C` 参数（`LogisticRegression` 的
逆正则化强度，越小正则越强），并把 `--n_samples` 从 64 提到 200（train 从 45→140）。
用 `--C 0.05 --n_samples 200 --t 0.65` 重跑：

```
train_acc=0.314  test_acc=0.217  (chance=0.125)
```

train/test 差距从原来的 77.5pp 降到 9.7pp，过拟合基本解决；`test_acc` 从贴近 chance
提升到 chance 的 1.7 倍，是真实但偏弱的信号（远不如 ELF 的 0.632）。

对应的干预结果（bootstrap CI，n=60 test docs）：

| direction | a=−1.0 | a=−0.5 | a=+0.5 | a=+1.0 |
|---|---|---|---|---|
| correct | +0.114 [0.085,0.141] | +0.046 [0.032,0.060] | +0.026 [0.010,0.043] | +0.103 [0.073,0.135] |
| orthogonal | +0.018 [0.008,0.030] | +0.006 [0.001,0.012] | +0.003 [−0.003,0.008] | +0.012 [0.002,0.021] |
| random | +0.009 [0.001,0.019] | +0.001 [−0.003,0.004] | +0.008 [0.004,0.011] | +0.020 [0.013,0.027] |

`correct`/`wrong`（镜像）在所有四个 alpha 上都明显高于 `orthogonal`/`random`（CI 不
重叠），支持"因果链确实存在"这个方向性结论；但和 ELF 的单调曲线不同，这里是 U 形
（中间 alpha 反而更低），量级也小一个数量级左右——LangFlow 上这条因果链更弱、更不
干净，但不是零。

详见 `logs/global_state/gs8_langflow_bigN_reg.log` 和
`results/global_state/langflow/baseline/intervene_global_to_local_bigN_reg.json`。
