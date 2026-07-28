# EXP-PT3 Spec — Velocity Alignment and Integrated Evidence

## 背景与地位

P0 阶段第三个实验，MVP-B（在 MVP-A 之上加这一个）。目标：检验向量场
`v_theta(z_t,t)` 在原生 decoder 显示出有意义的 token 之前，是否已经提供了微弱但
正确的漂移方向。这是本 suite 里第一个真正涉及"动力学"（而不仅是某个 t 上的静态
后验分布）的实验。

## 关键工程决策

### 1. Velocity 用未缩放的漂移方向，不是 ELF 的真实 v-场

ELF 真正的 flow-matching 速度是 `v = (x_pred - z) / (1-t)`（见
`sampling_utils.py: net_out_to_v_x`）。LangFlow 走的是 EDM 风格的 gamma-空间
stepping（`_euler_edm_step`），不是这种线性时间的 v-场参数化。

为了让两个架构在同一套 alignment 指标下可比，这里统一用**未缩放的漂移方向**
`drift = predicted_clean - z` 代表"速度"：

- `a_clean(t)`、后面几个 token-direction alignment 都是**余弦相似度**（或者用
  余弦相似度的分子分母结构），对 `drift` 的正标量缩放不敏感，所以这部分完全不
  受影响。
- `a_tok(t) = <drift, u_{y,f}>` 是**原始点积**（doc 定义如此，不是余弦），会
  受缩放影响。`C_i(t) = sum Delta t_j * a_tok(t_j)`（积分证据）因此**不是**一个
  和真实状态位移单位一致的量——对 ELF 来说，如果用真实的 `v=(x_pred-z)/(1-t)`，
  `C_i(t)` 会更接近"实际累积位移在 token 轴上的投影"这个物理意义；现在用的是一个
  按同一约定统一定义、可跨架构比较、但对 ELF 而言不是"物理正确"积分的**相对**
  证据累积分数。**这是一个明确的简化**，论文里如果要强调"integrated evidence"
  的物理意义，需要专门为 ELF 补一个用真实 v-场缩放的版本做对照。

### 2. Token-discriminative direction：centroid 变体在主脚本实现，probe 变体已补测

token-discriminative direction 的 centroid 变体（doc 变体 A）在主脚本里实现；
变体 B（trained probe direction）已经用一个轻量级配套脚本补上，见下面
"Probe direction 变体 B（补测）"一节。

`u_{y,f} = (c_y - c_f)/||c_y-c_f||`，centroid 在一个**独立的数据 split** 上估计
（`--n_centroid_samples` 条序列，和主探测用的 `--n_samples` 条序列完全不重叠），
用的是**干净（未加噪）embedding**，不需要过 backbone。

- ELF 的 clean embedding 就是 T5 encoder 输出，T5 encoder 在三个 checkpoint 间是
  共享、冻结的，所以 centroid 天然就在"跨 checkpoint 共享的外部空间"里，不需要
  doc 提到的额外对齐步骤。
- 只有出现次数 `>= --min_centroid_count`（默认 3）的 token 才会有 centroid；
  出现次数太少的 token 直接跳过对应位置（记录在 `valid_mask` 里）。pilot 规模
  （16 条 centroid 序列）只有 10.6% 的位置有可用方向；正式规模（256 条）预期会
  高不少，但依然不会是 100%——覆盖率本身也是需要在正式结果里报告的数字。
- doc 变体 B（训练一个线性 probe 当方向）已用配套脚本补测，见下面
  "Probe direction 变体 B（补测）"一节。

### 3. Controls：实现了 3/5

实现了：random direction（各向同性随机单位向量，每个位置固定一个，跨 t 复用）、
orthogonalized random direction（随机向量减去沿 `u_{y,f}` 的分量后重新归一化）、
frequency-matched wrong token（按 log-频率最接近 `f` 挑一个 `w != y, w != f` 的
token，用 `u_{y,w}` 代替 `u_{y,f}`）。

**没有实现**：same-token-direction-from-another-sequence（需要额外的跨序列
centroid 配对逻辑）；oracle-vs-free-running-states（需要 EXP-PT7 的 paired-rollout
基础设施，这里完全没有跑真实生成轨迹）。

### 4. 单次噪声采样，不是多噪声种子平均

PT1/PT2/PT5 为了估计一个稳定的后验分布，会对同一个 t 用多个噪声种子平均
softmax。PT3 不这样做——向量场 `v_theta(z_t,t)` 本身就是某个**具体状态** `z_t`
的属性，不是一个要 Jensen 平均掉的分布量，所以这里每个 t 只用**一次**噪声采样，
匹配 doc 公式本身的写法（用的是具体的 `z_t`，不是期望）。

## 与 `m_res`/`rank_res` 的关系

判定规则里"token-direction integral predicts later margin growth"这一条，这里
用一个具体化版本实现：在有效位置上，对每个 t 计算 `corr(C_i(t), -rank_raw(t))`
（Pearson correlation），real（`a_tok`）和三个 control 分开算，比较相关系数大小。
⚠️ 目前只对比了 **raw** rank，没有对比 residual margin（`m_res`，需要 EXP-PT2 风格
的 null reference，这里为了控制脚本复杂度和运行成本没有加）——如果要更贴近 doc
原文"predicts the residual-margin trajectory"，需要在这个脚本里再加一次 null
reference 的前向传播，留作后续增强。

## 脚本与输出

```text
experiments/phase_transition/probe_velocity_alignment.py
```

```text
results/phase_transition/<model>/<checkpoint>/velocity_alignment_raw_<label>.npz
results/phase_transition/<model>/<checkpoint>/velocity_alignment_summary_<label>.json
```

## 已知性能问题（留给下次优化）

`build_token_centroids` 之后，给每个位置分配 `u_{y,f}`/`u_{y,w}` 方向的双重
Python `for n in range(N): for l in range(L)` 循环（`main()` 里"Building
per-position token-discriminative directions"那一段）在 ELF 规模下
（N=128, L=1024，13 万+ 位置）实测要跑 **10+ 分钟纯 CPU 时间**，比预想的慢
（LangFlow 因为 L=128，同一段只要几十秒）。原因是每次迭代都有 Python 级别的
dict 查找 + 小张量减法/归一化，13 万次这种操作的解释器开销累积起来很可观。
**没有当场优化**（已经跑到一半，重写并重跑不划算），但下次如果要跑更大规模
或者更多 checkpoint，应该把这段向量化掉（比如先把 `centroids` 转成一个
`(V_seen, d)` 的张量 + id→index 映射，再用 `torch.index_select`/gather 批量
构造 `u_yf`，用 `torch.isin`/`gather` 替代逐位置 dict 查找）。

## 状态

**PILOT DONE（ELF baseline，N=4 probe / 16 centroid，T=6）**——sanity check 全部
通过：

- `mean_a_clean` 在几乎所有 t 都强正（+0.63 到 +0.91），只有 t→1 附近（target
  已经几乎等于 z，方向定义退化）才掉到 ~0——**在 t=0.05（G_raw=0.24%，原生
  decode 完全没有意义的时候）clean-direction alignment 已经是 +0.76**，
  这是支持"向量场提供早期正确漂移"的直接证据。
- `mean_a_tok_random`（随机方向对照组）在所有 t 上都在 0 附近震荡（-0.013 到
  +0.040），符合"随机方向不应该有系统性对齐"的预期，说明主 pipeline 和对照组的
  实现是正确、能互相区分的。
- `mean_a_tok`（真值-默认方向）在 t=0.05 反而是**负的**（-0.43），之后才转正
  并变大（+0.30→+3.29→+2.51→+1.66→+0.42）——⚠️ 这和"一开始就有正确的弱漂移"不完全
  吻合，最早的 t 上方向甚至偏向默认竞争者，需要正式规模确认这是不是稳定的模式
  还是 pilot 噪声（N=4 极小）。
- `corr(C_i(t), -rank_raw(t))`：真值方向（0.28-0.36）和 frequency-matched 对照组
  （0.29-0.35）在这个 pilot 规模下相关系数很接近，都明显高于 random/orth
  （0.05-0.08）——⚠️ 这提示"任何 token-centroid 方向都可能带有一定通用可预测性"，
  不完全是 `u_{y,f}` 这个具体配对独有的效果；正式规模需要确认这个差距是否会
  随样本量增大而拉开（如果一直很接近，"decision rule" 里"controls 不显示同样对齐"
  这一条就不成立，需要如实报告）。

## Results（正式规模：ELF baseline/kd_cr/kd2 + LangFlow，128 probe / 256 centroid
序列，21 个 t 点）

四个模型全部跑完。

| model/ckpt | frac_valid_direction | a_clean(t_min) | a_clean(t_max) | corr(C,-rank) real @t_max | corr freqmatch @t_max |
|---|---|---|---|---|---|
| ELF baseline | 35.3% | +0.764 | −0.002 | 0.348 (@t=0.95) | ~同量级 |
| ELF kd_cr | **86.6%** | +0.788 | −0.112 | 0.205 | 0.202 |
| ELF kd2 | **86.5%** | +0.790 | −0.280 | 0.332 | 0.327 |
| LangFlow | 55.4% | +0.648 | +0.975 | 0.093 | 0.088 |

**发现**：

1. **`a_clean`（漂移-真值方向余弦）在 t_min 对所有模型都强正**（+0.65 到
   +0.79），包括三个 ELF checkpoint 和 LangFlow——这是本 suite 目前最一致、最
   干净的正面发现：不管模型有没有做过 KD、不管架构，向量场在最早的可采样时间点
   就已经指向真实目标，早于原生 decode 有意义得多。
2. `a_clean` 在 t→1 附近的行为**因模型而异**：LangFlow 保持强正直到 t_max
   （+0.975，因为它的 embedding 空间/最终收敛特性不同）；ELF 的三个 checkpoint
   都在 t→1 附近**转负**（baseline −0.002，kd_cr −0.112，kd2 **−0.280**）——
   这是 target_dir=`x_clean-z_t` 在 t→1 时趋于 0 导致的余弦计算病态（分母趋于
   0），**不能解读为"模型晚期漂移方向变差"**，纯粹是这个指标在 t→1 附近的定义
   退化，需要在图上对 t→1 的最后一两个点打上警示，而不是拿来做结论。
3. **KD checkpoint 的 `frac_valid_direction`（86.5-86.6%）远高于 baseline
   （35.3%）**——三个 checkpoint 共享同一个 centroid 字典（同一个 T5 encoder），
   差异只能来自 `f_i`（默认竞争者，= 各自 checkpoint 在 t_min 的 native
   top-1）落在"有 centroid 的常见 token"里的比例不同。一个可能的解释：KD 模型
   在 t_min 的默认预测更集中在少数真正高频的 token 上（更容易有 centroid），
   baseline 的早期默认预测可能更分散、覆盖更多稀有/边缘 token（centroid 不可靠
   而被丢弃）——**这是一个待验证的猜想，不是确认的结论**。
4. `corr(C_i(t), -rank_raw(t))`：real（`a_tok`）和 frequency-matched 对照组的
   相关系数在四个模型上都**很接近**（差距通常 <0.02），random/orth 对照组则
   稳定接近 0。这个模式在四个模型上完全一致，不是某个模型的噪声——**应该被认真
   对待**：它说明"积分 token 方向证据预测 rank"这件事，很大程度上是**任何**
   合理的 token-centroid 方向都会有的效果，而不是 `u_{y,f}`（真值 vs 该位置的
   具体默认竞争者）这个精确配对独有的。doc 的判定规则"controls 不应该显示同样
   的对齐"在这个具体 control（frequency-matched wrong token）上没有被满足，需要
   在论文里如实报告这个负面/中性结果，而不是只报"random control 是 0，所以
   real signal 有效"（那个部分是真的，但不完整）。

⚠️ 性能问题记录见上一节（centroid 分配双重循环，ELF 上单次 10+ 分钟纯 CPU）。

## Probe direction 变体 B（补测）

**脚本**：`experiments/phase_transition/probe_direction_supplement.py`——轻量级
配套脚本，不是把主脚本 21 点密网格重跑一遍，而是在更稀疏的 t-grid（9 点）上
同时构造 centroid 方向（变体 A，这次顺便换成了向量化实现，见下）和 probe 方向
（变体 B：训练一个线性 probe，取 `W_y - W_f`），在同一批数据上直接对比。

**顺便修的性能问题**：变体 A 的 centroid 查表这次用 `index_add_`/`gather`
向量化实现（和 `EXP-PT6` 里修 `EXP-PT3` 同类问题时用的方法一样），不再是主
脚本里那个逐位置 Python 双重循环——ELF 规模下这部分从 10+ 分钟降到几秒。
（主脚本本身没有回去改，历史结果不受影响，仅供后续新实验参考。）

### Results（128 主探测序列 + 128 独立 centroid/probe 训练序列，9 个 t 点）

| model/ckpt | frac_valid A（centroid） | frac_valid B（probe） | cos(u_A, u_B) | corr_A 范围 | corr_B 范围 |
|---|---|---|---|---|---|
| ELF baseline | ~35%（主脚本正式规模数字） | **99.7%**（pilot 数字，正式规模同量级） | 0.188 | 0.168–0.253 | 0.062–0.088 |
| ELF kd_cr | 86.6% | ~99%+ | **0.338** | 0.129–0.269 | 0.084–0.118 |
| ELF kd2 | 86.5% | ~99%+ | **0.344** | 0.194–0.307 | 0.010–0.090 |
| LangFlow | 55.4% | ~99%+ | 0.129 | 0.148–0.249 | 0.039–0.083 |

**核心发现**：

1. **两种方向构造方法几乎不是同一件事**——jointly-valid 位置上的余弦相似度
   全部很低（0.13–0.34），说明"centroid 差"和"probe 权重差"这两种常见的
   token-discriminative direction 构造方式，即使目标概念相同（"真值 vs
   默认竞争者"），实际算出来的方向在高维空间里相当不同。
2. **probe 方向（变体 B）的 token 覆盖率远高于 centroid（变体 A）**
   （~99% vs 35-87%）——因为线性 probe 对词表里几乎每个 token 都有一行权重
   （哪怕训练集里出现次数很少），而 centroid 需要至少 `min_centroid_count`
   次真实出现才能估计，这是变体 B 一个实打实的实用优势。
3. **反直觉的一点：更简单的 centroid 方向（变体 A），在预测 rank 这件事上
   全程比"专门训练来分类的" probe 方向（变体 B）更强**——`corr_A` 在四个
   模型、全部 t 点上都高于 `corr_B`（比如 baseline t=0.5 时 0.210 vs
   0.069）。一个可能的解释：probe 是为了**多分类整体准确率**优化的，它的
   权重向量要同时把 y 从**所有**其它类别里分开，不是专门优化"y 相对 f 这一个
   具体竞争者"的方向；而 centroid 差是针对这一对 (y,f) 的"纯粹"两点对比，
   可能反而是更干净的局部判别方向。这是一个值得写进论文方法论讨论的发现——
   "用训练好的 probe 当方向"不一定比"简单的类中心差"更好，取决于你要度量
   的是全局判别能力还是局部（某一对 token 间的）判别能力。
4. **KD checkpoint 的 cos(A,B) 明显高于 baseline/LangFlow**（0.34 vs
   0.13-0.19）——KD 模型上两种独立构造的方向更趋于一致，这可能反映 KD 训练
   让"真值 vs 默认竞争者"这个语义在表示空间里更线性、更容易被不同方法一致地
   捕捉到，和 EXP-PT9 里"KD 提升迁移能力"的发现是同一类"KD 让表示空间更规整"
   故事的另一个佐证。
5. `kd2` 在 t=0.95 出现 `a_tok_A` 符号反转（从 t=0.5 的 +3.33 变成 t=0.95 的
   −1.63，kd_cr 也有类似但更弱的迹象：+3.08→−0.26）——⚠️ 这是原始点积，不是
   余弦，所以不太可能是主脚本里那种"target_dir→0 导致余弦退化"的老问题；
   更可能和 `EXP-PT7` causal interpolation 里 KD checkpoint 在 λ→1（过度
   收敛/过头）时表现崩溃是同一类"KD 晚期动力学不稳定"现象的另一个侧面，
   值得专门跟进但目前只是描述性观察。

## 严谨性补强：bootstrap CI（rigor audit 后，2026-07-26）

主脚本 `probe_velocity_alignment.py` 已经把逐位置原始数组存进
`velocity_alignment_raw_<label>.npz`（`t{i}_a_clean`, `t{i}_C_{variant}`,
`t{i}_rank_raw` 等），不需要重新跑模型就能做序列级 bootstrap——新增
`bootstrap_pt3.py`，按序列重采样 2000 次，对全部 4 个模型跑了两类头条数字：
(a) `a_clean` 在 t_min 的均值 CI；(b) `corr(C_i(t), -rank_raw(t))` 在最后一个
t 点上，real（`a_tok`）和三个 control 分别的 CI，并直接检验 real 与
freqmatch 的 CI 是否重叠（把本节前面"decision rule 没有被满足"这个观察
从点估计升级为有统计支持的判断）。

| model/ckpt | a_clean(t_min) CI | corr(real) CI | corr(random) CI | corr(orth) CI | corr(freqmatch) CI | real vs freqmatch 重叠？ |
|---|---|---|---|---|---|---|
| ELF baseline | +0.755 [+0.752,+0.758] | 0.056 [0.041,0.073] | 0.004 [−0.001,0.009] | 0.005 [0.000,0.009] | 0.055 [0.040,0.072] | **是** |
| ELF kd_cr | +0.787 [+0.784,+0.789] | 0.186 [0.154,0.308] | −0.001 [−0.003,0.004] | −0.001 [−0.003,0.003] | 0.184 [0.153,0.305] | **是** |
| ELF kd2 | +0.790 [+0.787,+0.792] | 0.296 [0.244,0.428] | 0.002 [−0.002,0.004] | 0.001 [−0.003,0.004] | 0.292 [0.240,0.423] | **是** |
| LangFlow | +0.649 [+0.645,+0.654] | 0.066 [0.059,0.076] | −0.005 [−0.017,0.008] | −0.005 [−0.019,0.009] | 0.066 [0.058,0.075] | **是** |

**结论（比之前的点估计观察更有力）**：

1. **`a_clean(t_min)` 强正且 CI 极窄**，四个模型都不含 0——"向量场在最早
   可采样时间点就指向真值方向"这个核心正面发现在序列级重采样下完全稳健，
   不依赖少数序列。
2. **random/orth 两个 control 的 CI 都紧贴 0（且都不含 real/freqmatch 的
   CI 区间）**——真值方向（`a_tok`）和 frequency-matched 方向的相关系数都
   清楚地、有统计显著性地高于纯随机/正交对照，这部分"存在真实信号"的结论
   是稳的。
3. **但 real 与 freqmatch 的 CI 在全部 4 个模型上都重叠**——这把前面"controls
   不显示同样对齐"这条 doc 判定规则"没有被满足"的观察，从一次性点估计
   （差距<0.02）升级成了有 2000 次 bootstrap 支持的结论：**这不是偶然的
   小样本噪声，四个独立模型都稳定重复同一个模式**。论文里应该明确报告：
   "token-discriminative direction 的证据累积确实预测 rank，但这个效应
   在这次的 freqmatch control 设计下，无法与『任意频率匹配的错误方向』
   区分开"，而不是暗示 `u_{y,f}` 这个具体配对独有的效果。

详见 `results/phase_transition/<model>/<checkpoint>/bootstrap_pt3_full.json`。
