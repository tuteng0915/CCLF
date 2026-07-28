# EXP-PT5 Spec — Decoder-Bias Intervention（readout-only 诊断）

## 背景与地位

P0 阶段第四个实验，MVP-A 的最后一块。目标：**因果性地**检验可见的 top-1 悬崖是不是
主要由 decoder 边界造成，而不是表征本身突然形成。做法是只改读出层的 logits（绝不
回灌到轨迹里），看小幅度的先验修正/logit 偏移能不能大幅度移动 `tau_b`（boundary
crossing time）而 `tau_e`（evidence emergence time，代表层面证据）几乎不动。

## 实现

`experiments/phase_transition/intervene_decoder_bias.py` 直接复用了 EXP-PT2
(`analyze_margin_trajectory.py`) 里的 `oracle_probs`/`null_probs` 函数（同一份
EXP-05v3 风格 null reference q_t，理由同 PT2 spec），在同一次 dense t-grid 前向
的基础上做两组干预（都是纯 post-hoc 的 logit 变换，没有额外前向传播成本）：

### A. 先验减法 sweep

```
ell'_t(v) = ell_t(v) - lambda * log q_t(v),  lambda ∈ {0, 0.25, 0.5, 0.75, 1.0}
```

`lambda=0` 就是原始 raw decode；`lambda=1` 正好是 PT1/PT2 里定义 `tau_e` 用的同一个
`e_t(v)`。这让 `tau_b(lambda)` 从 `tau_b(0)`(标准 boundary time) 平滑过渡到
`tau_b(1)`(全量去偏之后的 boundary time，理论上应该更接近 `tau_e`)。

### B. 加性偏移 sweep

```
ell'_t(y) = ell_t(y) + beta
ell'_t(f) = ell_t(f) - beta,  beta ∈ {0.5, 1, 2, 4, 8}
```

`f` = `f1`（earliest-time native top-1，同 PT1/PT2 的默认竞争者定义）。只修改这两个
token 的 logit，其余全部不变；判断真值 token 是否因此变成新的 argmax
（用 `max(ell(f)-beta, max_v∉{y,f} ell(v))` 做比较）。报告的是"多大的 beta 能让多少
本来错误的位置翻转成正确"——如果很小的 beta 就能翻转大量位置，说明模型在 log 空间
里其实已经"很接近"正确答案，只是没跨过读出边界。

## 已知简化 / 未做的部分

1. Doc 里 PT5 的另外两个测量项——"no change in independent-probe accuracy"和
   "no change in hidden states or velocity"——**这里没有单独验证**，因为脚本本身
   除了对已经算出来的 logits 做后处理变换之外，没有跑任何新的前向传播或采样，
   所以这两条在设计上是平凡成立的（trivially true），没有必要另外记录数字。
   但如果以后有人把这个诊断也用在会影响 hidden states 的场景（比如真的把
   `ell'` 喂回轨迹），这个"平凡成立"的假设就不再有效，需要重新检查。
2. `q_t` 复用 PT2 的 cheap null reference，不是 PT1 的三种 reference 之一——见
   `EXP-PT2-spec.md` 里同样的讨论和待办。
3. "positions whose crossing time changes without a change in residual evidence"
   这条 doc 定义比较宽泛，这里用一个具体操作化版本代替：
   `frac_positions_boundary_explained` = `tau_b(0) > tau_e` 但
   `tau_b(1) <= tau_e` 的位置比例——即"如果去掉先验，这个位置的 boundary time
   就会追上 evidence time"，作为"悬崖主要是读出边界造成"的直接证据。
   ⚠️ 这不是 doc 字面定义的直接翻译，是一个近似操作化，解读时需要说明。

## 脚本与输出

```text
experiments/phase_transition/intervene_decoder_bias.py
```

```text
results/phase_transition/<model>/<checkpoint>/decoder_bias_intervention_<label>.json
```

## 状态

**PILOT DONE** — ELF baseline，N=4，T=6 跑通。关键 sanity check：

- `beta` sweep 的 flip rate 随 beta 单调递增（0.5→0.023, 1→0.031, 2→0.043,
  4→0.066, 8→0.110）——这正是应该看到的方向（扰动越大，翻转越多），说明干预逻辑
  实现正确。
- `lambda` sweep 在这个极小 pilot 上效果很弱且方向不完全单调（G(lambda) 从
  lambda=0 到 1 略微下降而不是上升）——⚠️ pilot 规模下 null reference 本身噪声很大
  （只有 n_null=2 个采样），不能就此下结论说"先验减法没用"；需要正式规模复核。
  这和 EXP-PT1 pilot 里"m_res 几乎抹平 raw margin 但 top-1 覆盖率没怎么变"的发现
  方向是一致的——都指向"去偏"更多是重新分配非 top-1 的概率质量，而不是直接把
  真值顶到 top-1。

## Results（正式规模：LangFlow，128 序列，seq_len=128，21 个 t 点）

这是一个**和 EXP-PT1 的正面发现方向相反的重要反例**，需要如实报告：

- `tau_e` 均值 = 0.181，`tau_b(lambda=0)` 均值 = 0.479，但 **`tau_b(lambda=1)`
  均值 = 0.787——比 lambda=0 更晚，不是更早**！`mean_shift_vs_lambda0` 在
  lambda=1 时是 **−0.347**（负号 = 变晚），而且随 lambda 增大单调变晚
  （lambda=0.25: −0.028 → 0.5: −0.154 → 0.75: −0.273 → 1.0: −0.347）。
- `frac_positions_boundary_explained`（"去掉先验后 tau_b 追上 tau_e"的位置比例）
  只有 **0.018%**，基本等于 0。
- `frac_tau_b_finite` 随 lambda 增大而**下降**（97.5%→71.3%），即完全去偏后，
  更多位置永远达不到 top-1 正确——和 EXP-PT1 里 LangFlow 的
  `never_residual_correct(28-42%) >> never_raw_correct(3.9%)` 完全吻合，是同一个
  现象在两个不同实验里的交叉验证。
- `beta`（加性偏移）sweep 的 flip rate 随 beta 单调递增（0.5→8.6%, 1→16.0%,
  2→29.1%, 4→51.8%, 8→83.4%）——sanity check 通过，机制实现正确；但达到"多数位置
  翻转正确"需要 beta=4-8 这种量级的偏移，不是一个很小的量，说明原始 raw logits
  在 log 空间里离正确答案并不总是"近在咫尺"。

**解读**：LangFlow 上，这个特定的 readout-only 干预（用便宜的 EXP-05v3 风格 null
reference 去偏）**不支持"悬崖主要是 decoder 边界"的说法**——如果是，去偏应该让
`tau_b` 提前、`frac_positions_boundary_explained` 应该显著大于 0。实际观察到的是
相反方向：去偏反而经常让"覆盖率"变差（更多位置永远拿不到正确 top-1）。这和
EXP-PT1 的核心张力（"减先验能让 margin 转正，但不一定让真值变成新的 top-1，
概率质量有时候被分给了另一个错误 token"）是同一个机制在不同度量下的表现。
⚠️ 这不代表"decoder boundary 假说"整体被推翻——这里用的先验参考（EXP-05v3 风格
零信号 null）本身可能不是一个好的、能干净分离"通用先验 vs. 样本证据"的参考；
用 EXP-PT1 里更贵、更贴合真实状态统计量的 Gaussian reference（Reference A）重跑
一次这个诊断，是判断这个结论是否稳健的下一步。

## Results（正式规模：ELF baseline/kd_cr/kd2，128 序列，seq_len=1024，21 个 t 点）

| checkpoint | tau_e | tau_b(λ=0) | tau_b(λ=1) | shift(λ=1) | frac_boundary_explained | beta=1 flip rate | beta=8 flip rate |
|---|---|---|---|---|---|---|---|
| baseline | 0.194 | 0.353 | 0.343 | **+0.006**（提前） | **4.06%** | 5.5% | 16.5% |
| kd_cr | 0.097 | 0.203 | 0.263 | **−0.060**（推迟） | 0.85% | 12.4% | 46.9% |
| kd2 | 0.093 | 0.205 | 0.263 | **−0.058**（推迟） | 1.29% | 12.4% | 45.8% |

**核心发现：baseline 和 KD checkpoint 在这个诊断上方向相反**：

- baseline 上，先验减法让 `tau_b` **略微提前**（+0.006，很小但方向和"decoder
  boundary"假说一致），`frac_boundary_explained`（4.06%）在三者里最高。
- kd_cr / kd2 上，先验减法让 `tau_b` **明显推迟**（−0.06，量级是 baseline 的
  10 倍，方向和 LangFlow 的发现完全一致——见上面 LangFlow 部分），
  `frac_boundary_explained` 反而最低（0.85-1.29%）。
- beta（加性偏移）sweep：KD checkpoint 在同样的 beta 下 flip rate 明显更高
  （beta=1 时 kd 12.4% vs baseline 5.5%；beta=8 时 kd 46-47% vs baseline
  16.5%）——即 KD 模型的 raw logits 在 log 空间里离正确答案"更近"，一个小扰动
  就能翻转很多位置，这和 KD 模型 tau_e/tau_b 本来就更早、覆盖率更高的其它发现
  是一致的。

**综合解读**：这个"用便宜 null reference 做先验减法"的诊断，对 baseline 给出
弱的正面信号（支持"部分是 decoder 边界"），对 KD checkpoint 和 LangFlow 给出
负面信号（先验减法反而伤害 tau_b）。结合 EXP-PT1 的发现（KD 的 `advantage
retained` 远高于 baseline，且去偏对 KD 是净收益、对 baseline 接近无收益），
一个连贯的故事是：**KD 训练已经把"怎么从 backbone 表示里正确提取样本证据"这件
事学到了 native readout 里**，所以这里用的这种通用、非学习的事后先验修正对 KD
模型反而是噪声或者干扰（把概率质量送错方向的概率变大了），而对 baseline（没有
这种学习到的 readout 能力）这个简单修正还能带来一点点净正效果。这和 EXP-10
"KD 改变的是 decode interface"的结论是同一个机制在不同实验下的体现。

## 严谨性补强：bootstrap CI（rigor audit 后，2026-07-26）

给 `intervene_decoder_bias.py` 加了逐位置原始数组保存（`decoder_bias_raw_<label>.npz`：
`tau_e`, `tau_b_lambda0`, `tau_b_lambda1`, `correct_beta`, `wrong_raw`），重跑全部
4 个模型（数字与修复前逐位吻合，是一次免费的一致性检查），新增 `bootstrap_pt5.py`
按序列重采样 2000 次：

| model/ckpt | shift(λ=1) 95% CI | frac_boundary_explained CI | beta=1 flip CI | beta=8 flip CI |
|---|---|---|---|---|
| baseline | +0.0061 [+0.0056,+0.0066] | 4.06% [3.87%,4.24%] | 5.50% [5.36%,5.62%] | 16.47% [15.93%,17.24%] |
| kd_cr | **−0.0602** [−0.0612,−0.0591] | 0.85% [0.78%,0.93%] | 12.41% [11.76%,12.98%] | 46.88% [46.32%,47.38%] |
| kd2 | **−0.0582** [−0.0590,−0.0572] | 1.29% [1.14%,1.46%] | 12.42% [11.78%,12.96%] | 45.81% [45.40%,46.20%] |
| LangFlow | **−0.3470** [−0.3523,−0.3416] | 0.02% [0.00%,0.04%] | 15.97% [15.63%,16.29%] | 83.44% [82.90%,83.95%] |

**结论：全部 CI 都很窄且都不跨 0，把前面基于单次运行的"baseline 与 KD/LangFlow
方向相反"这个观察，升级为有统计支持、可以直接引用的结论**——baseline 的
+0.006 虽然量级很小，但 95% CI 稳定地落在正侧；kd_cr/kd2/LangFlow 的负向
shift 同样 CI 稳定不跨 0，且量级（kd≈−0.06，LangFlow≈−0.35）明显大于 baseline
的正向量级，不是同一噪声水平下的对称波动。这次 bootstrap 没有改变任何方向性
结论，只是把"看起来是这样"确认为"不是单次噪声画出来的"。详见
`results/phase_transition/<model>/<checkpoint>/bootstrap_pt5_full.json`。
