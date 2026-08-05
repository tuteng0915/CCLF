# Global State Formation：综合解读文档

## 状态

本文档整合 `EXP-GS1`–`EXP-GS15`（`docs/specs/EXP-GS{1-15}-spec.md`）的全部结果，
包括对 GS1–GS10 的用户审阅、P0-1–P0-4 方法论复核（GS11–GS14）、GS12/GS13 的正式规模
复现、GS15（residual organization trajectory）、以及**全部 15 个实验在 LangFlow 上的
pilot 规模复现**（第 10 节）。它取代 `docs/global_state_formation_experiment_suite.md`
（原始协议 doc）里第 1、18、19 节提出的 H1 假设表述，是当前对"连续语言模型如何从不确定
走向确定"这个问题**最新、最可信的解读**。后续如果要写论文的 method/discussion 部分，
应该以本文档为准，不要直接引用原始 doc 的 H1 表述。

---

## 0. 结论先行

原始假设：

```
global semantic basin -> structural scaffold -> exact lexical evidence -> exact token
```

**不成立**。不是模型先主动形成一个清晰的全局语义 basin，再逐层细化成 syntax 和
token。全部十四个实验交叉验证后，更可信的过程是：

> **原始连续状态一直含有微弱的 clean signal；模型早期主要产生一个通用、低分辨率的
> 共享表示，随后依靠跨位置交互逐渐组织高秩、位置特异的 residual，最终发生集体性的
> lexical crystallization。**

压缩成一句话：

```
weak distributed signal
  -> prior-dominated compression
  -> context-coupled residual organization
  -> collective lexical transition
  -> stable tokens
```

---

## 1. 最大的更新：早期 global signal 不等于模型已经形成 global meaning

`EXP-GS11`（`docs/specs/EXP-GS11-spec.md`）基本推翻了 `EXP-GS1` 最初的 headline。

直接对原始 oracle state `z_t = t*x + (1-t)*eps` 做位置平均 `mean(z_t) = t*mean(x) +
(1-t)*mean(eps)`，就能极早恢复文档身份：`L=32, t=0.28` 时 raw mean-pooling 的
self-retrieval 已经是 100%；`t=0.05` 时准确率随长度从 32 到 1000 单调从 39.6% 升到
93.8%，完全符合噪声被 `1/sqrt(L)` 平均掉的规律。

更关键的是，**模型自己的 `predicted_clean` 在相同条件下表现反而差得多**：
`L=32, t=0.28` 时 raw retrieval=1.00，但 model retrieval 只有 0.021（等于 chance）。

必须区分：

```
信息统计上存在  !=  模型已经组织或使用了信息
```

早期 oracle state 里确实有 clean information，但这部分信号首先只是 corruption
construction 里的线性泄露；模型自己的输出甚至可能把文档特异信息压回一个更通用的
prior。这回答了"不确定状态是什么"：**不确定不等于完全没有信息，而是信息以微弱、
分布式、尚未被模型组织利用的形式存在。**

---

## 2. 最有解释力的分解：共享均值与位置残差

`EXP-GS12`（正式规模，`docs/specs/EXP-GS12-spec.md`）给出了目前最干净的修正模型：

```
Z_t = 1 * mu_t^T + R_t
```

其中 `mu_t` 是跨位置共享的通道均值，`R_t` 是位置特异的 centered residual。

**1）粗粒度统计主要在均值里**：只用 `MEAN-only` 就已经能预测 POS histogram；加入
centered low-rank component 或 centered residual，几乎没有额外帮助。正式规模里，
全部 18 个 `(t, representation)` 条件，`MEAN-only` 无一例外是最高或并列最高的
POS R²。因此不能再说"结构信息先形成在低秩 global mode"，更准确的是：**POS
composition 这种低分辨率文档统计量，几乎完全可以由跨位置均值解释**——但它不是
dependency scaffold，更不是完整句法结构。

**2）Exact token identity 主要在高秩 residual 里**，这一半结论非常稳：`t=0.65` 的
model representation 上，`Acc(MEAN+R_c)=0.814` vs `Acc(MEAN+G_c)=0.091`；raw 和
model representation 上都如此，完整 t 网格复现。

目前最可信的几何图景：

```
coarse shared statistics live in mu_t
lexical identity requires high-rank R_t
```

"从不确定到确定"的真正主角，很可能不是低秩 global mode，而是**高秩、位置特异
residual 如何从无组织状态逐渐变成能够区分具体 token 的结构**。

---

## 3. 上下文确实推动局部 evidence，但不是简单线性的

`EXP-GS13`（正式规模，`docs/specs/EXP-GS13-spec.md`）修正了 `EXP-GS8` 最大的问题：
目标位置自身完全不受扰动，只改变其它位置。结果：沿 topic-probe axis 扰动其它位置后，
未被触碰的目标位置 true-token margin 仍明显变化。正式规模下 correct/wrong axis 的
效应范围约 `[0.18, 0.82]`，random/orthogonal 基本只在 `[-0.13, 0.13]` 内波动。

说明确实存在跨位置因果传递（`Z_{-i} -> ell_i(y_i)`），不是所有位置各自独立
denoise——residual 的形成是 contextual、coupled 的：`R_{i,t+dt} = F_i(R_{1,t}, ...,
R_{L,t})`。

但正式结果是 **U 形**，不是干净的 signed dose-response：沿"正确"方向和沿相反方向做
**大幅**干预都可能提高 margin，小幅干预反而更弱。因此不能说"正确 topic evidence 被
逐步传播到每个 token"，更安全的解释是：**模型对某条语义判别轴上的上下文组织非常
敏感；沿该轴推动上下文会改变局部 lexical certainty，但作用是非线性的**——更像是改变
了上下文的"决断程度"或 basin sharpness，而不是简单加减一个语义 logit。

---

## 4. 从独立模糊到集体确定

`EXP-GS5`（`docs/specs/EXP-GS5-spec.md`）发现不同位置的 margin increment
`delta_m_i(t) = m_i(t+dt) - m_i(t)` 不是独立的：真实位置间相关长度始终高于
position-shuffle 对照；susceptibility 和 excess correlation 在 `t≈0.28-0.39`
附近同时达峰，恰好略晚于 population-mean margin 在 `t≈0.22-0.28` 的过零点。

这给出一个有意思的时序：(1) 一部分位置首先跨越自己的 token boundary；(2) 随后整个
序列进入最大分歧、最大相互依赖的重组阶段；(3) 再往后各位置的 lexical outcome 一起
稳定。commitment cliff 可能不是一个瞬间，而包含两个相邻过程：

```
local boundary crossing -> collective coordination -> global stabilization
```

**⚠️ 待补的关键控制**（尚未执行，见第 9 节实验建议）：GS5 目前没有排除"同一条序列内
位置共享同一个 sequence-level common factor"这个混淆——应先对每条序列去掉平均
margin increment（`delta_m_i - mean_seq(delta_m)`）再算空间相关性，如果峰值依然
存在，才能更有信心叫 collective lexical reorganization，而不是全序列 logit scale
一起上升的假象。

---

## 5. 真实 trajectory 的分支实验支持"词汇最后稳定"

`EXP-GS14`（`docs/specs/EXP-GS14-spec.md`）用真实 free-running trajectory 和真实
累计 SC 重做 GS2 的 branching，定性模式继续成立：`C_topic`、`C_struct` 很早接近饱和，
`C_lex` 有明显动态范围（0.851 → 0.992）。说明 GS2 的 oracle/冷启动 SC 简化没有制造
lexical-stability 曲线。

但需要谨慎：topic 是饱和的 KMeans/mean-pooling 指标，structure 是高度饱和的 POS
histogram cosine（都是第 2 节已经证明"主要反映均值"的那类指标）。因此 GS14 真正
证明的是：**同一个真实中间状态附近的小扰动，早期仍可产生不同的 exact lexical
realization；这种 lexical branch diversity 随时间持续收缩**——它没有可靠证明"主题
已经确定"，但证明了 lexical uncertainty 是真实动态收缩的过程，不是一步到位。

---

## 6. Oracle–rollout gap 应该放进这套过程里理解

`EXP-GS7`（`docs/specs/EXP-GS7-spec.md`）：`G_token(oracle, t=0.28)=0.589` vs
`G_token(rollout, t=0.28)=0.038`；直到 `t≈0.65`，rollout 才追上 oracle 在 `t≈0.28`
的 lexical recoverability。

结合第 2 节的 residual 结论，最合理的解释是：**oracle path 从一开始就在线性地注入
正确的、位置特异的 clean residual；free-running path 则必须依靠模型自己通过多步
上下文交互，把一个尚未组织的高秩状态逐渐塑造成正确的 lexical residual**。teacher-
forcing gap 因此可以更具体地写成：

```
oracle directly supplies target-aligned residual signal
free-running must construct that residual through closed-loop dynamics
```

这比泛泛地说"distribution shift"精确得多。

---

## 7. 五阶段过程（当前最可信的版本）

**Stage 0 — Weak signal presence**：`z_t = t*x + (1-t)*eps` 里已经包含微弱 clean
component，跨位置平均甚至能很早恢复文档身份，但这只是统计 recoverability，不是模型
commitment（GS11）。

**Stage 1 — Prior-dominated compression**：模型自己的 `predicted_clean` 在早期反而
丢失文档身份，可能被拉向某种通用、平均化的语言状态；top-1 是高频默认 token；
document-specific retrieval 很弱；高秩 residual 尚未被组织（GS11）。

**Stage 2 — Context-coupled residual formation**：其它位置的状态开始因果性地影响
目标位置 token margin，不是各位置独立恢复，而是上下文组织调制局部 evidence，且这个
过程非线性（GS13）。

**Stage 3 — Collective lexical reorganization**：一部分 token margin 先过零，随后
位置间 margin increment 的相关性和 susceptibility 达到峰值——高秩 residual 从许多
互不协调的局部方向，重新组织成一个整体相容的 lexical configuration（GS5）。

**Stage 4 — Lexical crystallization**：exact token identity 主要依赖高秩
residual；branch lexical entropy 持续下降，最终每个位置落入稳定 token basin；
free-running 比 oracle 更晚完成这一步，因此出现 error propagation / exposure
gap（GS3/GS12、GS2/GS14、GS7）。

---

## 8. 核心 hypothesis（建议写进论文的版本）

不是：

> The model first commits to global semantics, then syntax, then tokens.

而是：

> **Continuous language generation begins with weak but statistically
> recoverable clean signal. The model initially compresses this signal into a
> prior-dominated representation, then uses cross-position interactions to
> progressively organize high-rank, position-specific residuals. Lexical
> certainty emerges through a collective reorganization of these residuals
> and finally crystallizes into stable discrete tokens.**

压缩形式：

```
weak distributed signal
  -> prior-dominated compression
  -> context-coupled residual organization
  -> collective lexical transition
  -> stable tokens
```

---

## 9. 下一步实验建议：不要再测 topic probe，直接追踪 R_t

下一步最应该直接追踪 `R_t`（高秩位置特异残差），而不是再找一个新的 global metric
（topic/sentence 类指标已经被证明主要反映均值，参见第 1、2 节）。

> **⚠️ 9.1 节已经跑了 pilot（`EXP-GS15`），结果和下面写的预期方向相反**，见本节末尾
> "9.1 pilot 更新"——`O_R(t)` 全程为负（free-running rollout 比一条纯几何直线插值
> 更慢组织出终点 residual），不是"存在加速组织窗口"。9.1 原始文字保留作为最初的
> 假设记录，但结论以更新段落为准。

### 9.1 Residual alignment trajectory

在真实 free-running trajectory 上定义 `R_t = Z_t - 1*mu_t^T`，最终 residual
`R_star`。测 `A_R(t) = CKA(R_t, R_star)`（或 Procrustes/whitened alignment），
同时测 native token accuracy、独立 token probe、branch lexical entropy、residual
effective rank，看 GS5 的 collective peak 是否正好对应 `R_t` 向 `R_star` 快速
对齐。

**9.1 pilot 更新（`EXP-GS15`，`docs/specs/EXP-GS15-spec.md`）**：n=4 条真实自由生成
轨迹，8 个 checkpoint，加了一个关键对照——`R_{t0}` 和 `R_star` 之间的纯直线插值
`A_linear(t)`（不含任何模型动力学，排除"越接近终点当然越像"的同义反复）。核心判定量
`O_R(t) = A_rollout(t) - A_linear(t)` **在全部中间 checkpoint 上都是负的**，
`t=0.38-0.50` 附近最负（约 -0.22 到 -0.23）——**free-running rollout 组织 residual
的速度比一条朴素直线插值还慢**，不是"存在加速组织窗口"。同时发现 paired oracle 路径
（同一初始噪声+同一终点）的 residual 对齐进度全程接近甚至优于线性插值，rollout 明显
落后——这把 `EXP-GS7` 的"oracle-rollout token gap"第一次量化到了 residual alignment
层面，而且落后幅度比"落后于一条几何直线"还大。**这是一个需要修正预期的负结果，不是
"residual 加速组织"这个机制假设的确认**，详见 GS15 spec 的完整解读和"下一步"（怀疑
可能和 residual 路径本身的高维几何形状有关，而不是纯粹的组织速度问题，需要换成
"随机配对轨迹"这类不依赖线性假设的对照来进一步排查）。

**9.1 跨架构确认（v2，`O_R_model`）**：`EXP-GS7`/`GS15` 的 LangFlow 复现发现 raw-state
CKA 在 LangFlow 上整体饱和（第 10.1 节），于是给 GS15 加了一套完全平行的
**model-based** 指标（`predicted_clean` 残差的线性插值对照）。**在 LangFlow 上独立
重跑后，`O_R_model(t)` 同样全程为负，且同样在轨迹中段最负、随后稳步恢复到 0**
（ELF 最负点 `t≈0.20-0.28`，约 `-0.22~-0.24`；LangFlow 最负点 `t≈0.28`，约
`-0.47`，量级更大但方向、形状一致）。**"晚期崩塌而非渐进组织"这个假说因此从单一
架构的孤立观察，升级为两个训练方式、backbone、noise schedule 都不同的模型共享的
动力学性质**——这是当前 GS 系列里少有的、真正经过跨架构因果/动力学检验的正面发现，
建议作为 synthesis 第 7 节 Stage 2-3（context-coupled residual formation → collective
lexical reorganization）的核心支持证据。详见 `EXP-GS15-spec.md` "正式跨架构验证"一节。

### 9.2 Residual velocity decomposition

把 velocity 也中心化：`V_t = 1*v_bar_t^T + V_t^R`，测
`<V_t^R, R_star - R_t>`，直接回答"每一步 vector field 是否真的在把高秩 residual
推向最终 lexical configuration"。

### 9.3 Oracle vs rollout residual gap

分别比较 `R_t^oracle` 和 `R_t^rollout`。如果 oracle-rollout 的巨大 token gap
（GS7）主要对应 residual alignment gap、而均值 component 差异很小，整个机制会更
完整、更可信。

### 9.4 修正版 collective coupling（GS5 补控制）

对 GS5 的 margin increment 做：sequence demeaning（见第 4 节"待补的关键控制"）、
token-frequency residualization、logit-scale normalization、多个 epsilon 种子；
再对比 kd_cr / kd2——如果 KD 让 residual-alignment peak 和 collective peak 一起
提前，就能解释 KD 到底改变了什么（呼应 EXP-01v3/EXP-10 等已知的"KD 大幅提前
commitment cliff"发现）。

---

## 10. 最终判断

新一轮实验（GS11–GS14 + 正式规模复现）最大的贡献，不是重新证明"global earlier
than local"，而是把这个过于宽泛的说法拆掉了：

- 早期的"全局信息"大量来自 raw pooling，不是模型计算的产物；
- POS 这类粗粒度统计主要来自跨位置均值；
- exact lexical identity 需要高秩 residual，均值和低秩共享分量都远远不够；
- 上下文确实会因果地影响未被直接触碰位置的 lexical evidence，但作用非线性；
- 多位置在 lexical transition 附近出现真实的协调重组（有待第 9.4 节补控制确认）；
- free-running 在组织这种 residual 上明显落后于 oracle。

因此目前最值得追的机制问题是：

> **How does a context-coupled high-rank residual become organized enough to
> support stable token identities?**

这比"模型何时知道 topic"更接近"不确定如何逐渐变成确定"这个真正想回答的问题。

---

## 10. LangFlow 跨架构复现

全部 15 个 GS 实验（GS1–GS15）都在 LangFlow baseline 上按相同的 pilot 规模重跑了一遍
（脚本层面：新建 `common.py` 的 `load_adapter`/`load_owt_docs` 两个共享 helper，统一
处理 ELF/LangFlow 的加载差异，取代 14 个脚本里原来的 `assert args.model == "elf"` 硬
拦截；顺带修复了 GS9 的一个真实 bug——`encode_pair_variant` 原来的目标位置定位逻辑是
T5-tokenizer 专用的"编码 `context+' '` 后丢弃最后一个 token（假设是自动加的 `</s>`）"，
对 GPT-2/LangFlow 的 tokenizer 不成立，因为 GPT-2 不自动加 EOS，而是把尾随空格和目标词
合并成同一个 BPE token；已经改成 tokenizer-agnostic 的写法：编码 `context`（不加尾随
空格），如果最后一个 token 恰好是 EOS 就丢掉，取长度作为目标位置，用两种 tokenizer
分别验证过 ✅）。

### 10.1 三个跨架构方法论教训

1. **涉及 raw oracle state 的 CKA 类指标在 LangFlow 上经常整体饱和，不提供信息**。
   `EXP-GS7` 的 `CKA(oracle,rollout)` 在 LangFlow 上全程 ≈1.000（ELF 有明显的 U 形，
   最低到 0.894）；`EXP-GS15` 的 `A_rollout(raw)`/`A_oracle`/`A_linear` 三条曲线全部
   从 `t=0.05` 起就 ≈0.99+（ELF 是从 0.13 平滑升到 1.0）。但**同一批实验里的
   `A_model`（用 `predicted_clean` 而不是 raw `z_t`）在 LangFlow 上反而表现出干净的
   动态范围**（GS15 的 `A_model`: 0.181→1.000）。说明这不是"LangFlow 没有可测的动态
   过程"，而是"raw-state CKA 这个具体操作化对 LangFlow 的几何/schedule 不敏感，必须换成
   model 输出"。后续任何要跨架构比较的 residual/CKA 类分析，都应该默认用 `predicted_clean`
   而不是 raw `z_t`。
2. **沿用 ELF 校准出的 `t=0.28` 作为"过渡区代表点"对 LangFlow 完全无效**。这是
   `EXP-02`/`EXP-03` 早就确立的"nominal t 不可跨模型比较"结论的又一次直接印证：
   `EXP-GS8` 在 `t=0.28` 上 LangFlow 的 topic probe test_acc 只有 0.158（chance=0.125，
   几乎没有信号），导致 `EXP-GS13` 的因果检验完全测不出东西；`EXP-GS6` 在同一个 t 上
   topic 分类噪声大到连"纯 A"起点都会被误判成"other"。**任何需要"early-but-informative
   t"的 GS 实验（GS4/GS6/GS8/GS13），跨到 LangFlow 时都必须先用 LangFlow 自己的 GS1
   曲线重新定位过渡区**（从 LangFlow 的 GS1 pilot 看，大致要到 `t≈0.5-0.85` 才有和 ELF
   `t=0.28` 相当的 `G_topic`/`G_token` 水平），不能直接複用 ELF 的数值。
3. **LangFlow 在名义 `t=0.99` 时依然有真实的残余不确定性，不像 ELF 那样接近完全 clean**。
   `native_logsnr(0.99)=3.48`，而 LangFlow 自己的 `gamma_min=2.60`（真正最干净的点）——
   `t=0.99` 并不是 LangFlow 调度下的渐近终点。这至少解释了两件事：`EXP-GS9` 在 LangFlow
   上没有出现 ELF 那种"高 t 时 delta 精确为 0.000"的自我一致性饱和问题（双向 denoiser
   在真正 clean 的输入上才会"读出自己写好的答案"，LangFlow 的 `t=0.99` 还没到那个程度）；
   `EXP-GS7` 的 `G_token(oracle)` 出现"先升后降"的反直觉曲线，很可能是因为脚本里用
   `t=0.99` 的 rollout 终点冒充"clean state"来构造 paired oracle，而这个代理本身对
   LangFlow 来说还不够干净，导致更高 t 的 oracle 状态注入了比预期更多的噪声。

### 10.2 两条核心结论在架构间稳健复现

尽管有上面三个方法论坑，**`EXP-GS12` 的两条核心结论、以及 `EXP-GS2`/`EXP-GS14` 的
"C_lex 是唯一有动态范围的 consensus 指标"，在 LangFlow 上都干净复现**，这是目前
横跨两个架构都成立、最值得写进论文的发现：

- **Structure ≈ mean**：LangFlow 的 `MEAN_only` 在几乎全部 `(t,representation)` 组合下
  依然是三者中最高或并列最高的 structural R²（如 `t=0.99,raw`: `MEAN_only=0.379 >
  MEAN+G_c=0.297 > MEAN+R_c=0.205`）。
- **Token ∈ residual**：`MEAN+R_c` 的 token_acc 全程远超 `MEAN+G_c`（`t=0.99,raw`:
  `0.895` vs `0.146`），且比 ELF 更早在较低 t 就出现差距（`t=0.50,raw`: `0.364` vs
  `0.027`）。
- **Lexical consensus 是唯一有意义的动态范围**：`EXP-GS2`/`EXP-GS14` 在 LangFlow 上
  `C_topic`/`C_struct` 同样从很早就饱和在 ~1.0，只有 `C_lex` 随 `t_start` 单调上升
  （GS14: `0.719→0.818→0.942`）。

### 10.3 有意思的架构差异（不确定是 confound 还是真实差异，需要更多样本）

- **GS3（raw、未中心化）在 LangFlow 上从一开始就不支持"structure 在 G"**：`A_G` 全程
  低于 `A_R`，`syntax_G` 几乎全程为负——即使不做 GS12 的中心化修正，LangFlow 的原始
  数据也没有表现出 ELF raw GS3 那种"看起来支持 H1"的假象。这可能说明 ELF 的"structure
  在低秩 G"假象和它自己的均值结构/checkpoint 有关，不是所有连续语言模型的通用几何性质。
- **GS11 的"model 处理会抹除可检索身份信息"这个 ELF 发现，在 LangFlow 上不成立**：
  LangFlow 的 `predicted_clean` retrieval accuracy 在大 `L_eff` 下追平甚至略超 raw
  （`0.958` vs `0.917`，`L_eff=1000, t=0.28`），不像 ELF 那样持续更差。这意味着
  synthesis 第 1 节"Stage 1: prior-dominated compression"的具体机制（模型早期输出把
  文档身份压回通用 prior）可能是 ELF（或至少是 ELF baseline 这个 checkpoint）特有的，
  不能不加验证地当成所有连续语言模型的普遍阶段——写论文时需要把这一步的表述限定在
  "至少在 ELF baseline 上观察到"，而不是笼统地说"continuous LMs"。
- **GS5 的 collective coupling 时间形状不同**：LangFlow 的 `chi`（susceptibility）在
  整个 pilot t 网格（到 `t=0.85`）内单调上升、没有像 ELF 那样出现明确的峰值和回落，
  `xi` 的 excess-over-shuffle 反而是早高晚低。结合 10.1 第 2 条（LangFlow 的"过渡区"
  比 ELF 晚得多），最合理的解释是**LangFlow 的 collective coupling 峰值落在这个 pilot
  t 网格之外（`t>0.85`）**，需要把网格往后延伸才能看到和 ELF 类似的峰值-回落形状，
  不能直接下"LangFlow 没有 collective 峰值"这个结论。

### 10.4 下一步（第 1、3 条已完成，见下方"已完成的后续更新"）

1. ✅ 对 GS4/GS6/GS8/GS13 用 LangFlow 自己校准出的过渡区 t（`t=0.65`）重跑——**GS4 干净
   复现，GS6 在修好一个真正的 bug（不是 t 选点问题，见下）之后比 ELF 更干净，GS8/GS13
   换 t 不解决问题（根因是训练样本量太小、8 类分类器过拟合，不是 t 校准）**。
2. 把 GS7/GS15 的"clean 代理"改成更严格地贴近 LangFlow 自己的 `gamma_min`（比如用
   `t=0.999` 或直接在 gamma 空间里取 `gamma_min` 对应的 t，而不是固定用 nominal
   `t=0.99`）。
3. ✅ 把 GS15 的核心指标从 `A_rollout(raw)` 换成 `A_rollout(model)`——已完成，**LangFlow
   上 `O_R_model(t)` 同样全程为负、中段最负（`t≈0.28`，`-0.472`）、随后恢复到 0，和 ELF
   （最负点 `t≈0.20-0.28`，`-0.22~-0.24`）方向、形状一致**，"负 O_R / 晚期 collapse"
   假说得到跨架构确认。
4. 把 GS3/GS11 观察到的架构差异（"structure 在 G"是否只是 ELF 现象、"model 处理是否
   破坏身份信息"）在 kd_cr/kd2 checkpoint 上也测一遍，看这些差异是 architecture-level
   的还是 training-procedure-level 的（比如和 KD 有关）。

### 已完成的后续更新（2026-07-26）

- **GS4/GS15 见上（第 1、3 条）**。
- **GS6 的"P_A(lambda) 全程平坦"根因被找到并修复，不是 t 校准或 rollout 步长问题**：
  `nearest_topic` 原来用平方欧氏距离找最近质心，而 LangFlow rollout 终点的 pooled
  embedding norm（≈1.15）系统性地远小于拟合 topic centroids 时用的 clean embedding
  norm（≈3.70）——每一个 rollout 终点都被判给 norm 最小的那个质心，和真实内容无关
  （诊断：8 篇文档的 rollout 终点全部被分到同一个 cluster，但它们各自和自己 clean
  embedding 的 cosine 相似度是 0.40–0.67，真实信号一直都在，只是被 norm 差异掩盖）。
  改成 cosine-based nearest-centroid（合并进 `common.py`，一并修复 GS4/GS8-mechanism/
  GS14 用到的同一函数）后，**GS6 在 LangFlow 上给出了比 ELF 更干净的 bifurcation**
  （4 对文档全部在 `lambda=0.4→0.6` 同步完成切换，纯 A/纯 B 端点也正确分类），GS4 的
  topic 维度也从"恒 0.25"变成有意义的结果（`B_preserve_global` 时 topic=1.00、
  token=0.005——GLOBAL-4"保留 global mode 应该保住 topic 丢失 token"这一预测目前最
  干净的确认），GS14 的 `C_topic=1.000` 从"无法排除是 bug"变成"确认为真"。详见
  `docs/specs/EXP-GS6-spec.md`"LangFlow 复现"一节。
  **通用教训**：任何用欧氏距离做最近质心分类的诊断，一旦待分类对象和拟合质心的分布
  存在系统性尺度差异，就会产生"看似没有信号"的假阴性——应默认优先用 cosine 而不是
  欧氏距离。

### 已完成的后续更新（2026-07-27，严谨性自审 + 大样本/bootstrap CI 复核）

用户要求对整个 GS 系列做一次严谨性自审。发现并处理了两类问题：

**1. 上面记录的"GS6 bug 修复"验证方法本身有漏洞，遗漏了第二处同款 bug。**
之前只验证了 GS4/GS6/GS8-mechanism/GS14 四个文件的 `nearest_topic` import 是否指向
同一个函数对象，没检查每个文件的 C_topic 计算是否**真的调用**了这个函数。重新审计
发现 `branch_global_consensus.py`（GS2）和 `branch_true_trajectory.py`（GS14）里各有
一份内联手写的平方欧氏距离最近质心分类，从未走 import 路径——GS2 是这个 bug 模式
最早的出处（其它文件应是从这里 copy-paste 出去的），此前从未被修复过。已修复两处，
GS14 顺便补上了逐轨迹原始数据（原来只存跨轨迹均值，无法做 CI）。

修复后重跑：**GS2 的"C_topic 早饱和"这一核心交叉验证结论在 ELF/LangFlow 两个架构上
都不受影响**（bug 修复前后数值几乎一样，0.91-1.00 区间），排除了"饱和=这个 bug 的
假阳性"这个担忧；GS2 自带的 eta sweep（0.01/0.03/0.1）确认 C_topic 对扰动幅度有真实
但比 C_lex 迟钝得多的渐变响应，不是完全不敏感的天花板伪影。

**2. GS4/GS6/GS14 此前引用的 n=4 pilot headline 数字，在补上大样本 + bootstrap CI
之后有实质性修正**（`common.py` 新增 `bootstrap_ci`，对齐 PT 系列 2000-resample 标准
——GS 系列此前完全没有任何 CI，这本身就是本轮自审发现的最大缺口）：

- **GS6**（n_pairs 4→20，3 个种子，pooled n=60）：λ=0.4 与 λ=0.6 的 P_A 95% CI 不重叠
  （[0.083,0.283] vs [0.483,0.733]），bifurcation 本身有真实统计支持；但转变窗口比
  n=4 显示的更宽（0.4→0.8 而非单一 0.4→0.6 区间），纯 A 端点（λ=1.0）P_A 只有 0.817
  [0.717,0.900]、从未接近 1.0，且存在 n=4 从未报告过的 10-45% "P_other"（落入第三方
  topic）。"比 ELF 更干净、全部同步切换"的表述已撤回，改为"存在真实但有相当噪声的
  bifurcation"。
- **GS4**（n_docs 4→16，3 个种子，pooled n=48）：`baseline`/`A_remove_global`/
  `B_preserve_global` 的 topic_match 95% CI 完全重叠（0.812-0.875 区间内互相包含），
  统计上无法区分——"G 单独排他性地承载 topic"这个更强说法不成立，topic 信息更可能是
  在三种重构里冗余存在。**token_acc 的不对称性完整存活**（`B_preserve_global`
  token≈0.006-0.008 vs 另两个条件 0.53-0.57，3 个种子一致，无需 CI）：token identity
  依赖残差、topic 不明显依赖 G 这条不对称结论是本轮复核里唯一完全经得住考验的。
- **GS14**（n_traj 4→16，3 个种子，pooled n=48，且用的是刚修复的 bug-free 代码）：
  C_topic 从"恒为 1.000、零方差"变成 0.958[0.929,0.984]@t=0.20 → 0.992[0.976,1.0]
  @t=0.65 的真实渐变信号；配合 eta sweep（0.01→0.3）确认 C_topic 随扰动幅度单调下降
  但比 C_lex 迟钝得多，和 GS2 的 eta sweep 结论相互印证。

**总体教训**：n=4、单 seed、看起来"整整齐齐"（全 0 或全 1）的结果，本身就该被当作
"可能是运气"对待，而不能因为数字漂亮就默认可信——这是 PT 系列（PT6/PT7）已经验证过
的教训，但 GS 系列直到这次自审之前一直没有被同样的标准要求过。详见
`docs/specs/EXP-GS2/GS4/GS6/GS14-spec.md` 对应章节。

### 已完成的后续更新（2026-07-27 第二轮：GS8/GS13 过拟合修复 + ELF 大样本复核）

在上一轮自审（GS2/GS4/GS6/GS14 的 bug 修复 + LangFlow 大样本复核）之后，用户要求继续
修正剩下已知的两个缺口：

**1. GS8/GS13 的 LangFlow 过拟合问题（此前只诊断，未修复）。** 给
`intervene_global_to_local.py`/`intervene_context_only.py` 的 `LogisticRegression`
加了 `--C` 正则化参数，配合 `--n_samples` 从 64 提到 200 重跑（`t=0.65`）：

- **GS8**：`train_acc` 0.933→0.314，`test_acc` 0.158→0.217（chance=0.125）——过拟合
  基本解决；bootstrap CI 确认 `correct`/`wrong` 方向在全部 4 个 alpha 上都明显高于
  `orthogonal`/`random`，但曲线是 U 形而非 ELF 那样的单调剂量-反应，量级也小一个数量级。
- **GS13**：同样的修复，`test_acc` 0.158→0.217；但干预效果只在一个 alpha 方向上
  （a=+1.0）的 CI 不含 0，另一方向（a=−1.0）`correct` 自己的 CI 就含 0——是本轮所有
  修复里最不干净的一个，引用时需要如实注明"仅单侧显著"。

两者共同的结论：LangFlow 上的 global-to-local 因果链**确实存在**（不是 t 校准或纯粹的
过拟合假象），但比 ELF 弱得多、也不如 ELF 干净，样本量修复只能部分改善，不能让它变得
和 ELF 一样干净。

**2. ELF 版本补做和 LangFlow 同款的大样本（n=16-20）+ 3 seed + bootstrap CI 复核**
（消除"只对 LangFlow 较真、ELF 还停留在 n=4 双重标准"这个问题）：

- **GS6**：n=60 pooled，ELF 的 bifurcation 在大样本下依然非常干净（λ=0.4→0.6 几乎
  一步到位，纯端点 CI 下界 0.88+，`P_other`≤15%），**明显比 LangFlow 同款复核更干净**
  （LangFlow 转变窗口更宽、纯端点只到 0.82、`P_other` 高达 45%）——这不是 n=4 pilot
  的运气，是在完全对齐方法学下测出的真实架构差异。
- **GS4**：n=48 pooled，`t_start=0.38` 时 `A_remove_global`（0.292）和
  `B_preserve_global`（0.229）的 CI 几乎重叠，都远低于 `baseline`（0.917）——**ELF 上
  G 和 R 单独都不足以恢复 topic，必须两者兼备**，比"G 不够"这个 n=4 结论更完整；这和
  LangFlow（三条件统计不可区分）形成一个有意思的架构对比：ELF 需要完整状态，LangFlow
  从任一部分重构都够。
- **GS14**：n=48 pooled，且这是本文件 `C_topic` 计算**首次真正走过修复后的
  `nearest_topic`**（此前"确认为真"其实是用未修复代码算出来的，见下面第 3 点）。
  `C_topic` 在 `t_start=0.65` 上是真正的零方差天花板（1.000[1.000,1.000]），LangFlow
  同一 t_start 上没有到顶（0.992[0.976,1.0]）——差异合理（LangFlow 整体 commitment
  更晚，同一 nominal t 对两个架构不是同一个相对位置）。

**3.（延续上一轮）`branch_global_consensus.py`/`branch_true_trajectory.py` 里发现的
第二处未修复 bug**：细节见上一轮"已完成的后续更新（2026-07-27）"小节，这里只强调一点
后果——GS14 之前"C_topic=1.000 现已确认为真"这个结论，实际上从未真正验证过（因为
C_topic 的计算从未走过被 import 的 `nearest_topic`），本轮是它第一次在修复后代码上
被真正测过。

详见 `docs/specs/EXP-GS4/GS6/GS8/GS13/GS14-spec.md` 对应新增章节和
`docs/paper_revision_notes.md` Q.8。

### 已完成的后续更新（2026-07-31，GS16/GS17：机制链第二、三环，精化五阶段叙事的 Stage 3-4）

用户提供了更严格的 GS16-GS20 spec（自带校准协议、多重对照、决策规则），要求依次执行。
GS16（Calibrated Endpoint Bank, Specificity, and Affinity Collapse）和 GS17（Local
Residual Dynamics and Unified Transition Timing）是其中的 P0，已在 ELF baseline 上
以 n_traj=16 跑完 pilot（两个实验复用同一批 16 条真实轨迹，可以做逐轨迹级别的交叉
验证，细节见 EXP-GS16-spec.md、EXP-GS17-spec.md 各自的"Pilot Results"章节）。

**这两个实验合起来，把第 7 节 Stage 3（"collective lexical reorganization"）和
Stage 4（"lexical crystallization"）之间原来比较笼统的描述，替换成一个可以被证伪、
已经跑出干净数字的三段式机制**：

1. **早期（t 从 0.05 起）：方向性已经存在，但不针对某个具体终点（"common-manifold
   transport"）**。GS17 的 `cos_endpoint(t)`（局部速度是否朝自身真实终点方向）从
   `t=0.05` 起就高达 `+0.81`，说明模型的局部更新从很早就不是随机游走；但同一时刻
   GS16/GS17 的相对指标（`S_self`/`V_self`——自身终点相对于"用校准扰动重新走一遍"
   产生的其它候选终点的进度优势）统计上和 0 没有区别，说明这个早期方向性对多个
   candidate 完成几乎一视同仁，还没有偏向某一个具体答案。
2. **中段（`t≈0.23-0.29` 起）：候选端点开始分化，多个候选的"可行性"开始塌缩**。
   `V_self` 从这个窗口起转正并持续上升；GS16 用更贵的"真的扰动一下、重新走完整条
   轨迹看落到哪"的方法确认，`rank_self=1`（自身端点是全部候选里最相似的）的比例
   在 `t=0.36→0.43` 一步之内从 12-25% 跳到 100%（16/16 条轨迹全部），亲和度熵同步
   从 0.87 骤降到 0.54 并保持平台。两个独立方法（局部速度分解 vs 真实扰动重跑）
   在同一个窗口给出一致的坍缩信号。
3. **"看起来已经决定"和"真的对扰动免疫"是两个略微错开的事件**：GS17 的事件顺序
   统计显示，GS16 测出的候选坍缩窗口（`t=0.30-0.36`）在全部 16 条轨迹上都**晚于**
   该轨迹自己逐位置 top-1 解码稳定下来的中位时间（`tau_50_stable≈0.24-0.30`）——
   `P(tau_affinity<=tau_50_stable)=0`，无一例外。也就是说，模型的"当前最优猜测"先
   看起来定型，然后（隔了一小段但一致的距离）底层的吸引域才真正收窄到"扰动也逃不
   出去"的程度。这是原来 Stage 3/4 的描述里没有的时序细节。

**这个精化后的三段式，比第 8 节现在的压缩形式更精确**：

```
weak distributed signal
  -> prior-dominated compression
  -> common-manifold transport（局部速度已有方向性，但对多个候选终点一视同仁）
  -> narrow-window candidate collapse（候选端点可行性坍缩，双方法交叉验证同一窗口）
  -> point-estimate stabilizes slightly before perturbation-robustness catches up
  -> stable tokens
```

**已知局限（如实记录，避免重蹈 GS15 的覆辙）**：
- GS15 的 `O_R(t)` 负结果需要打折扣看待：用户指出 `A_linear` 这个对照本身"偷看"了
  `R_star`（终点是它的一个插值锚点，`t=t_end` 时 `A_linear` 被定义强行等于 1），
  不是一个公平的、不知道未来的零假设，所以"rollout 全程比朴素插值还慢"这个结论
  被削弱为"局限"，不再是干净的正面发现——已在 EXP-GS15-spec.md 反映。
- GS16 只实现了两种扰动校准协议里的一种（"一步匹配冲击"），"终点线性化匹配冲击"
  未实现，Control 6（两种协议的稳健性对照）暂缺。
- GS17 的 `tau_velocity`（简单"取导数最大值"检测器）在 4/16 条轨迹上明显被轨迹
  末端的数值不稳定污染，需要更稳健的检测方法才能放心引用。
- 目前只有 ELF baseline、n_traj=16、单一 t_bank=0.20、无完整的层级 bootstrap——
  spec 要求的正式规模（n_traj>=48、多 seed、ELF+LangFlow）还没有做。

GS18（Conditional Reviewer Controls）是否要跑，取决于 GS12 的"高秩残差承载 lexical
信息"和 GS5 的"collective coordination"这两条论点目前是否仍是论文核心主张——按 spec
自带的 stop rule，这个判断留给用户决定，不自动执行。

### 已完成的后续更新（2026-08-01，GS18：Conditional Reviewer Controls，Part A 收窄一条核心结论，Part B 加固另一条）

用户要求执行 GS18（此前因为是 conditional/stop-rule 门控的实验，一直没跑）。GS12 的
"高秩残差承载 lexical 信息"和 GS5 的"transition 是集体的"这两条都被确认目前仍是论文
核心论点，所以 Part A、Part B 都跑了。结果一正一反，都已写入
`docs/specs/EXP-GS18-spec.md`：

**Part A（rank/energy-matched 残差对照）：收窄了 GS3/GS12 的结论，不是推翻。**
固定 k、比较 top-k vs middle-k vs bottom-k vs random-k 四种同维度子空间后发现，
**top-k 在几乎所有 k 上都全面碾压其它三种**（token 恢复、结构 R²、保留能量全部更高）
——这意味着 GS12"残差（MEAN+R_c）远好于低秩 global mode（MEAN+G_c）"这个此前被
当作稳健跨架构结论的发现，**更可能主要是维度/能量效应**（完整残差比一个 k=8 的
截断多出几百个维度），而不是"lexical 信息特意编码在非主方向、和 topic/structure
信息分开存放"这个更强的说法。第 7-8 节现在的表述（"高秩、位置特异的 residual"）
本身没错——token identity 确实需要很多维度才能恢复——但如果论文里暗示或明说这是
一种"分布式编码、刻意避开主成分方向"的机制，需要根据这个结果去掉，改成更朴素的
"需要足够多的维度，不是某几个特殊方向的专利"。

**Part B（common-factor-controlled collective dynamics）：加固了 GS5 的方向，但
重塑了时间形状。** 用 GS17 的真实 free-running rollout（而非 GS5 原来用的 oracle
state），逐步剔除位置、当前 margin、序列级 logit 范数/熵/margin 等混杂因素后，
剩余的空间相关长度在 16 个 checkpoint 里有 13 个都超过 5 种不同 null 模型（位置
shuffle、序列 shuffle、循环移位、符号翻转、方差匹配高斯噪声）的 95 分位——**"存在
真实的、不能被这些混杂因素解释掉的集体协调"这个方向性结论站得住，且是在比 GS5
更严格的对照下站住的**。但具体的时间形状变了：不是 GS5 报告的"cliff 之后一个尖峰"，
而是**早期（t<0.35，涵盖并早于 GS17 测出的 tau_50_stable 中位数~0.24-0.30）持续
偏高、中段（0.4-0.7）明显回落、晚期（0.75-0.93）又有一次目前还解释不了的回升**。
"崩溃点后单一尖峰"这个具体画面需要修正为"预承诺期到承诺期整体偏高、形状更复杂"。

**两条結果合起来对第 7-8 节的影响**：Stage 3（collective reorganization）的存在性
证据变得更强了（双重对照都支持"不是混杂因素"），但其"紧跟在 cliff 后面的单一窗口"
这个时间细节需要放宽；Stage 4（lexical crystallization 依赖高秩 residual）的存在性
本身不受影响，但如果之前的表述暗示了"分布式编码/避开主成分"这类更强机制含义，需要
弱化为"需要足够多维度"这个更朴素、但更站得住的版本。

**已知局限**：两部分都是 pilot 规模（Part A n=64、单一 t=0.28；Part B n_traj=32、
17 checkpoint、200 次 null 置换），均低于 spec 要求的正式规模（≥128、≥33
checkpoint、1000 次置换），也都还没做 LangFlow 复现。Part A 修复了一个实现效率问题
（SVD 被在 k/kind 循环里重复计算了约 8 倍，已改成每条序列只算一次并缓存复用，数值
验证前后一致）。

### 已完成的后续更新（2026-08-01，GS19：异步去噪消融，干净负结果）

用户要求继续执行 GS19（P2，spec 标注"AFTER P0"，此时 GS16-18 均已完成，门槛满足）。
这个实验和前面 GS15-18 不同——不是被动分析已有轨迹，而是一次真正的采样器干预：
给每个位置分配不同的局部去噪进度（而不是全局统一的标量 t），测试这种"打破同步"
是否能缓解 GS16/17/18 揭示的"协调坍缩"瓶颈。结果是**干净的负结果**：LTR、RTL、
fixed_random、confidence_adaptive 四种异步顺序，在 spec 定义的三个"期望信号"
（`tau_stable` 应该更早、`tau_first` 应该基本不变、生成质量不应变差）上**全部
失败**——`tau_stable` 反而更晚、revision 次数几乎翻倍、生成 PPL 普遍上升 2-6 倍，
RTL 甚至让 75% 的样本退化。四种顺序、三个指标同时一致失败，不是某个方向或某个
指标的边缘信号。按 spec 自带的决策表，这精确对应"all fail → 不建议训练 Wavefront
Flow Forcing"这一档。

**如何理解这个负结果**：GS16-18 揭示的"早期方向性存在但不针对具体终点→候选端点
窄窗口坍缩→集体协调"这套机制，是在一个**只用全局标量时间训练**的模型上观察到的。
把"给每个位置局部时间"这个想法，作为一个事后（未经训练）的采样时刻推理技巧
（用已训练模型在标量时间下产出的 xhat/eps_hat，反推出每个位置在自己局部进度下
"应该"是什么样子）强行嫁接上去，效果是负的——这更可能说明"该模型的内部计算方式
已经和'所有位置在同一个时间点被联合处理'这个训练时的假设深度绑定"，而不是说明
"给位置解耦局部时间"这个思路本身没有价值。要真正检验后者，需要**从头训练**一个
原生支持逐位置局部时间的模型（真正的 Wavefront Flow Forcing），而不是在已训练
模型上做事后干预——但 GS19 这个负结果本身，已经是"不建议现在就投入训练 WFF"
这个具体建议的直接证据。

详见 `docs/specs/EXP-GS19-spec.md`。实现过程中发现并修复了一个真实的正确性 bug
（初版用 `forward_state` 而非 `solver_step` 的自条件机制算 `xhat`，导致 delta=0
的"同步"对照臂本身都对不上一个普通 rollout，修复后验证到 100% token 一致）。

### 已完成的后续更新（2026-08-02，GS20：第三个架构 Plaid 的跨架构复现，一正一反）

CDCD（spec 原定的第三个架构）确认官方无代码无 checkpoint 后，改用 Plaid（Gulrajani &
Hashimoto, NeurIPS 2023，训练语料 OpenWebText2，有真实发布的 1B checkpoint）替代，搭了
独立环境、写了完整 adapter（GS20 spec 自带的 6 项 gate 检查全部通过），在 Plaid 上跑了
GS16/GS17 的 pilot 规模复现。

**GS16 的核心发现——干净的第三次独立确认**：早期 `S_self` 与 0 无区别、随后在极窄窗口内
坍缩到"自身端点几乎必然排第一"，这个模式在 Plaid 上不仅复现，坍缩窗口比 ELF 还要窄
（一个 checkpoint 步长内完成，ELF 需要 2-3 步）。三个架构（ELF、LangFlow 层面的相关
证据、现在加上 Plaid）目前唯一一个"exploration-collapse"式坍缩得到跨架构确认的机制。

**GS17 的核心发现——如实报告一个真实的边界条件，不是简单复现失败**：局部速度对齐
（`cos_endpoint`）的曲线形状和 ELF 定性不同，`V_self` 在中段出现一段持续为负、ELF 没有
的模式，"候选坍缩 vs 逐位置稳定"这两个事件的先后顺序在 Plaid 上和 ELF 相反。三者最可能
的共同原因：(1) Plaid 的原生采样器是随机 ancestral 过程（每步注入噪声），有限差分速度
估计会混入这个采样噪声，这是 ELF/LangFlow 的确定性 Euler step 没有的混淆因素；(2) 这次
复现直接沿用了 ELF 校准出的 `t_bank=0.20`，没有针对 Plaid 自己的 schedule 重新校准——
这正是本项目自己反复强调过的"不能跨架构直接比 nominal t"这条教训，这次自己又踩了一次。
按 GS20 spec 自带的判定规则（"CDCD 分歧是需要报告的边界条件，不是要藏起来的复现失败"），
这个事件顺序反转目前**不能**当作确认的跨架构机制分歧，只能如实记录为"尚未排除测量混淆"。

详见 `docs/specs/EXP-GS20-spec.md`"Pilot Results"节。GS18（rank-matched/collective
coordination）和 GS19（异步去噪）尚未在 Plaid 上尝试——GS18 Part A 尤其需要重新设计，
因为 Plaid 的 embedding 维度只有 16，"k 从 1 扫到 128"这个 ELF 尺度的协议在这么低维的
空间里不成立。

### 已完成的后续更新（2026-08-03，GS20 完整记分卡：GS16-19 全部在 Plaid 上跑完）

在上一轮只跑了 GS16/17 的基础上，GS18（两部分）和 GS19 也在 Plaid 上跑完了。完整的
5 项记分卡：

| 实验 | ELF | Plaid | 判定 |
|---|---|---|---|
| GS16（端点专属性坍缩） | 早期无区别→窄窗口坍缩 | 同样模式，坍缩窗口更窄 | **跨架构确认** |
| GS17（局部速度动力学） | cos_endpoint 早期就高，坍缩晚于 token 稳定 | cos_endpoint 早期低、非单调，坍缩早于 token 稳定（反转） | **边界条件** |
| GS18-A（rank/energy 对照） | top-k 全面碾压同维度非 top-k | 同样碾压 | **跨架构确认** |
| GS18-B（集体协调） | 13/16 个 checkpoint 打赢全部 null | 只有 1/16 | **边界条件** |
| GS19（异步去噪消融） | "all fail"，2-6x PPL 变差 | "all fail"，3-14.4x PPL 变差，更彻底 | **跨架构确认** |

**一个可能比任何单条结论都重要的元发现**：会跨架构复现的三个（GS16、GS18-A、GS19）
共同特点是都在看"最终落点"——GS16 的分支端点、GS19 的最终生成文本、GS18-A 单个时刻的
静态子空间结构，都不需要对连续的噪声状态做逐步差分。不复现的两个（GS17、GS18-B）
共同特点是都依赖对 Plaid 原生随机 ancestral 采样器产出的**相邻状态做差分**——GS17 的
有限差分局部速度、GS18-B 的位置间 margin 增量空间相关性。Plaid 的原生采样器每一步都
给每个位置独立注入高斯噪声（ELF/LangFlow 是确定性 Euler/EDM step，没有这个噪声源），
这个噪声几乎必然会污染任何"逐步差分"类型的度量，不管背后真实机制在两个架构上是否
一致。这不是"GS17/GS18-B 的机制只在 ELF 上成立"的证据，而是"这套方法论对确定性 vs
随机采样器的敏感度"这个更一般问题的证据——在把结论写进论文之前，需要先把这个混杂
因素和真实的架构差异分开（比如把 t_bank 按 Plaid 自己的 log-SNR schedule 重新校准，
以及想办法把随机采样器的注入噪声从速度/相关性估计里剥离出去）。

详见 `docs/specs/EXP-GS18/19/20-spec.md` 各自新增的"Plaid 复现"章节。过程中还修了
两个 `plaid` conda 环境的依赖缺口（nltk 触发的 libstdc++ ABI 冲突、缺 scikit-learn）
和一个真实的 `PlaidAdapter` bug（`make_oracle_state` 漏掉 `@torch.no_grad()`，调用
Plaid 自己学出来的 gamma_bounds/noise_schedule 模块时保留了梯度图——GS16/17/19 从
纯噪声起步、从来不调用这个方法，所以直到 GS18 Part A 才第一次触发）。
