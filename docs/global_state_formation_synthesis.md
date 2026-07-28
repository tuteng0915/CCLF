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
