# EXP-GS14 Spec — True-Trajectory Hierarchical Branching (P0-4)

## 背景与地位

用户审阅指出 `EXP-GS2` 的一个限制：branch 起点是**直接构造的 oracle state**
（`Z_t = t*x_clean + (1-t)*eps`），self-conditioning **冷启动为全零**，不是"从一条真实
free-running 轨迹里保存下来的中间状态"。这意味着 GS2 测的是"oracle 状态的可扰动性"，不是
"真实生成过程中 basin 的形成"——两者可能给出不同的答案，因为真实轨迹在到达某个 t 之前，
self-conditioning 已经积累了一路的历史信息，而 GS2 的每个起点都是从零开始的。

本实验（P0-4）用 `EXP-GS7` 已经建好的自由生成基础设施，先跑出**真正的 free-running
轨迹**，在几个 checkpoint t 上同时保存 `Z_t` **和** `SC_t`（真实积累的自条件状态，不是
零），再从这些真实状态做 `EXP-GS2` 同款的 branch-and-measure-consensus 分析，比较结果
是否和 GS2（oracle 起点、冷启动 SC）一致。

## 1. 设计

- **第一阶段**：对 `n_traj` 条独立初始噪声做真正的 free-running rollout（复用
  `EXP-GS7` 的 `rollout_with_checkpoints` 逻辑，但额外保存每个 checkpoint 的
  `sc_state`，不只是 `z`——`EXP-GS7`/`EXP-GS10` 已经在用的 `rollout_with_checkpoints`
  不保存 `sc`，为了不破坏这两个已经跑通的脚本，本实验在自己的脚本里重新实现一个
  同时保存 `(z, sc)` 的版本，不改动 `compare_oracle_rollout_global.py`。
- **第二阶段**：在每个 checkpoint `(Z_t, SC_t)` 上，做 `EXP-GS2` 同款的 K-branch
  consensus 分析——**唯一的区别是**：（a）起点是真实轨迹中间态，不是 oracle 构造；
  （b）branch 的 rollout 从**真实 `SC_t`** 初始化，不是冷启动全零（需要一个新的
  "从给定 sc 继续 rollout"函数，`EXP-GS2` 的 `rollout_branches` 内部固定
  `sc=torch.zeros_like(z)`，这里不能直接复用，另写一个接受 `sc_start` 参数的版本）。
  其它（扰动构造、consensus 指标：`C_topic`/`C_struct`/`C_lex`/`C_sent`）完全复用
  `EXP-GS2` 的定义。

## 2. 判定

对比本实验（真实轨迹起点 + 真实 SC）和 `EXP-GS2`（oracle 起点 + 冷启动 SC）在同一批
checkpoint t 上的 consensus 曲线：

- 如果两者的定性模式一致（比如都是 `C_struct` 早早饱和、`C_lex` 随 t 单调上升），说明
  GS2 的"冷启动 oracle 起点"简化不影响主结论，可以放心引用 GS2 的数字。
- 如果不一致（比如真实轨迹上 `C_topic`/`C_struct` 表现出明显不同的时序），说明 SC 的
  真实累积历史确实重要，GS2 的数字需要标注为"仅在冷启动 oracle 协议下成立"。

## 3. 数据与规模（pilot）

- ELF baseline，`eval_exp37c_baseline.yml`（1024-token）。
- `n_traj=4`（独立自由生成轨迹，和 GS4/GS6 同一量级——生成本身比 GS2 的 oracle 起点贵，
  必须控制规模）。
- checkpoint t：`[0.20, 0.38, 0.65]`（复用 GS2 pilot t_start 网格的一个子集，覆盖已知
  cliff 附近到 cliff 之后）。
- `K=6` branch（比 GS2 pilot 的 8 略少，控制算力——生成 + branch 两段 rollout 叠加，
  比 GS2 单纯 branch 更贵）。
- `eta=0.03`（GS2 校准出的中间值，不重新做 eta sweep）。
- topic centroids 复用 GS1 的 `topic_kmeans_centroids_pilot.npy`。

## 4. 已知简化

1. ⚠️ `n_traj=4` 很小，consensus 数字的样本量比 GS2 pilot（4 docs）相当但比正式规模
   小得多。
2. ⚠️ 只测一个 `eta` 值，不做 sweep。
3. ⚠️ Free-running 生成本身也依赖 `EXP-GS7` 已经标注过的简化（`t_eps` 起点、
   `n_steps=32` 标准生成质量）。
4. Pilot 规模，数字仅用于判断"真实 SC 是否改变 GS2 的定性结论"这一方向性问题。

## 5. 脚本与输出

```text
experiments/global_state/branch_true_trajectory.py
```

```text
results/global_state/<model>/<checkpoint>/branch_true_trajectory_<label>.json
```

## 状态

**Pilot DONE — GS2 的定性结论在真实轨迹上依然成立**（ELF baseline，4 条真实自由生成轨迹，
`K=6` branch，`checkpoint_ts=[0.20,0.38,0.65]`，`eta=0.03`，GPU1，
`logs/global_state/gs14_elf_baseline_pilot.log`，输出
`results/global_state/elf/baseline/branch_true_trajectory_pilot.json`）。

## Results（真实轨迹 + 真实 SC vs GS2 oracle + 冷启动 SC，同一 eta=0.03）

| t_start | C_topic (GS14真实) | C_topic (GS2 oracle) | C_struct (GS14) | C_struct (GS2) | C_lex (GS14) | C_lex (GS2) |
|---|---|---|---|---|---|---|
| 0.20 | 0.937 | 1.000 | 0.999 | 1.000 | **0.851** | 0.959 |
| 0.38 | 1.000 | 1.000 | 1.000 | 1.000 | **0.969** | 0.996 |
| 0.65 | 1.000 | 1.000 | 1.000 | 1.000 | **0.992** | 0.999 |

**解读**：

1. **模式完全一致**：无论是真实轨迹（本实验）还是 oracle 冷启动（GS2），`C_struct`
   和 `C_topic` 都从很早的 checkpoint 起就接近饱和（≥0.937），只有 `C_lex` 表现出
   有意义的动态范围、并随 `t_start` 单调上升——这个 GS2 的核心定性发现在换成真实
   free-running 轨迹 + 真实累积 SC 之后**完全复现**。
2. **数值上真实轨迹版本略低一点，尤其是 `C_lex` 和早期的 `C_topic`**（`t=0.20` 时
   `C_lex`: 0.851 vs 0.959；`C_topic`: 0.937 vs 1.000）——方向上说得通：真实轨迹的
   branch 起点携带了一路真实积累的路径依赖，比"凭空构造的 oracle 状态+冷启动 SC"更
   接近生成过程的实际分布，理论上应该比人为构造的起点表现出**略多**、而不是更少的
   分支多样性（因为 oracle 起点本身可能因为直接从 `t*x_clean+(1-t)*eps` 构造、缺乏
   真实路径历史而"更容易"收敛回同一个 basin）。这个方向上的小幅差异和这个推理一致，
   但 `n_traj=4` 太小，不能确认这是真实效应还是噪声。
3. **结论**：`EXP-GS2` 的"冷启动 oracle 起点"简化**没有改变其定性结论**——至少在
   `n_traj=4` 的 pilot 规模下，"structural/topic consensus 早饱和、lexical consensus
   是唯一有动态范围的指标"这个发现在真实轨迹上依然成立，可以放心引用 GS2 的定性结论
   （数值精度层面，真实轨迹版本的 `C_lex`/`C_topic` 略低，如果要给出精确数字，应该
   优先引用本实验而不是 GS2）。

## 下一步

1. 扩大 `n_traj`（当前 4），确认 `t=0.20` 处 `C_topic`/`C_lex` 偏低是否是真实的
   "真实路径历史增加分支多样性"效应，还是纯粹的小样本噪声。
2. 补充更多 checkpoint t（尤其是更早的、接近已知 cliff 起点的 `t=0.05`），看真实轨迹
   在信息量最少的阶段是否比 oracle 冷启动表现出更明显的差异。
3. 这个"真实轨迹 vs oracle 冷启动"的对比框架（`rollout_with_checkpoints_and_sc` +
   `rollout_branches_from_state`）也可以用来重新检验 `EXP-GS4`（同样是从 oracle 状态
   + 冷启动 SC 做因果干预），看 GS4 的"低秩 global mode 不足以因果驱动"这个结论是否
   在真实轨迹上同样成立。

## 严谨性自审：发现并修复了未走 import 路径的 nearest_topic 副本，重新评估 C_topic=1.000（2026-07-27）

之前"GS6 修复后 C_topic=1.000 现已确认为真"这个结论（见本文件此前版本、EXP-INDEX.md）
建立在一个错误的前提上：GS6 debug 时只验证了 `intervene_global_modes.py`（GS4）、
`probe_competing_basins.py`（GS6）、`intervene_global_to_local.py`（GS8-mechanism）、
`branch_true_trajectory.py`（本文件，GS14）这四个文件的 `nearest_topic` import 都
指向同一个（已修复为 cosine-based 的）函数对象——但从未检查每个文件的 C_topic 计算
**是否真的调用了这个被 import 的函数**。

用户要求做严谨性自审后，重新审查本文件发现：第 179-181 行（`C_topic` consensus 的
计算部分）是一份**内联手写的**平方欧氏距离最近质心分类（`dists = ((emb[:,None,:] -
centroids[None,:,:])**2).sum(-1); topic_ids = dists.argmin(-1)`），和被 import 进来的
`nearest_topic` 完全无关——GS6 那次修复实际上从未触及这里，`C_topic=1.000` 从头到尾
都是用未修复的欧氏距离算出来的。已改为直接调用 `nearest_topic(emb, centroids)`，同时
补上了逐轨迹原始数据（`{metric}_per_doc` 字段，原来 JSON 里只存了跨轨迹均值，没法做
bootstrap CI）。

**修复 + `n_traj=4→16`、3 个种子（42/123/456）重跑，pooled n=48 的 bootstrap CI：**

| t_start | C_topic [95% CI] | C_lex [95% CI] |
|---|---|---|
| 0.20 | 0.958 [0.929, 0.984] | 0.765 [0.752, 0.778] |
| 0.38 | 0.985 [0.963, 1.000] | 0.843 [0.836, 0.851] |
| 0.65 | 0.992 [0.976, 1.000] | 0.941 [0.938, 0.944] |

C_topic 不再是 n=4 时"恒为 1.000、零方差"的样子，而是一个**真实的、随 t_start 单调
上升、但绝对值一直很高**的信号——修复本身让数字略微降低了一点（从"永远 1.000"变成
"0.96-0.99 之间"），但方向和"branch 之间在 topic 层面高度一致"这个定性结论没有变化。

**eta sweep（n_traj=16, seed=42, t_start=0.65 切片，检验"C_topic 是否只是扰动幅度太
小导致的天花板伪影"）：**

| eta | C_topic | C_lex |
|---|---|---|
| 0.01 | 1.000 | 0.977 |
| 0.03（默认） | 0.976 | 0.938 |
| 0.10 | 0.962 | 0.837 |
| 0.30 | 0.854 | 0.626 |

C_topic 随 eta 增大单调、明确地下降，排除了"该指标对扰动完全不敏感"这个担忧——它是
真实的、渐变的信号，只是比 C_lex 迟钝得多（eta 从 0.01 升到 0.3，C_lex 掉了 0.35，
C_topic 只掉了 0.15），这个鲁棒性差异本身与 GS2 eta sweep 的结论（见 EXP-GS2-spec.md）
完全一致，是跨两个不同 branch-consensus 实验、两种扰动方式共同支持的稳健发现。

详见 `logs/global_state/gs14_langflow_bigN_seed{42,123,456}.log`、
`gs14_langflow_etasweep_{0.01,0.1,0.3}.log` 和对应
`results/global_state/langflow/baseline/branch_true_trajectory_{bigN_seed*,etasweep_eta*}.json`。

## 严谨性自审：ELF 版本补做同样的大样本 + bootstrap CI 复核（2026-07-27）

之前的 n=16×3 seed + eta sweep 复核（见上一节）只做了 LangFlow。补做 ELF 版本：
`--n_traj 16 --k_branches 6 --seed {42,123,456}`（`t_start` 用原生默认
`{0.20,0.38,0.65}`），用的是本轮修复过的代码（这也是本文件 C_topic 计算首次真正
调用修复后的 `nearest_topic`，此前 n=4 pilot 和"确认为真"那次都是用未修复的内联
欧氏距离算出来的，见上方"发现并修复了未走 import 路径的 nearest_topic 副本"一节）。

Pooled n=48（3 seed × 16 traj）bootstrap CI：

| t_start | C_topic [95% CI] | C_lex [95% CI] |
|---|---|---|
| 0.20 | 0.974 [0.946, 0.995] | 0.858 [0.852, 0.864] |
| 0.38 | 0.982 [0.960, 1.000] | 0.966 [0.962, 0.970] |
| 0.65 | **1.000 [1.000, 1.000]** | 0.992 [0.990, 0.993] |

t_start=0.65 时 C_topic 是真正的零方差天花板（48 个观测全部=1.0）。这不是 bug 的
残留——本次用的是修复后代码，且 LangFlow 同款复核（0.992[0.976,1.0]）在同一 t_start
上没有到达硬天花板，说明这个差异是真实的架构/时间点效应：ELF 在 t_start=0.65 时已经
非常接近其自身的 commitment cliff 之后（cliff≈0.2-0.3），branch 内部的 topic 高度
一致是合理的；LangFlow 的等效"晚"点相对更靠前（LangFlow 整体 commitment 更晚），所以
在同一 nominal t_start=0.65 上还没有完全到顶。两个架构上 C_topic 都随 t_start 单调
上升、C_lex 保持更大动态范围，GS2 的核心定性结论完整存活。

详见 `logs/global_state/gs14_elf_bigN_seed{42,123,456}.log` 和对应
`results/global_state/elf/baseline/branch_true_trajectory_bigN_seed{42,123,456}.json`。
