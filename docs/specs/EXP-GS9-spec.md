# EXP-GS9 Spec — Minimal Global Contrast Sets

## 背景与地位

原始 doc 第 13 节 GLOBAL-9，P1 阶段最后一项。和 GS1–GS8 的关键区别：不用自然 OWT 文档，而是
**人工构造的最小对照句对**——固定局部词形/句法环境，只改变全局语义/discourse frame，检验全局
frame 是否在因果意义上塑造局部 lexical evidence。这是本系列里第一个、也是唯一一个需要
手工编写数据的实验（GS1–GS8 全部自动从 OWT 采样），因此规模天然更小，且存在
"实验者选择偏差"的风险（人工挑选的对照句对可能无意中偏向支持假设的方向）。

## 0. 数据构造

不追求原始 doc 建议的严格 token-length-matched pair（要求 A/B 两句在 T5 tokenizer 下逐 token
对齐，实际操作中很难在不影响语义的前提下保证），改用**动态定位目标位置**的操作化：

1. 每对 `(A, B)` 共享一个"frame 设定短语"位置（早期，决定全局语义），在后段有一个共同的
   句法槽位（比如"declared a state/period of ___"），槽位后面接一个**目标补全词**。
2. 对每对，定义 `target_A`（frame A 下语义合理的补全）和 `target_B`（frame B 下语义合理的
   补全）。
3. 构造 4 个 `(context, target)` 组合：`(A,target_A)`、`(A,target_B)`、`(B,target_A)`、
   `(B,target_B)`——即两个 context 分别接上两个 target。
4. 目标 token 位置**动态定位**：先编码 `context + " "`（不含 target），得到长度 `p`，
   target 的第一个 token 落在位置 `p`（T5 tokenizer 在这个位置分词是稳定的，不要求 A/B
   两个变体整体等长）。

## 1. 数据集

12 对手工构造的最小对照句（`build_global_minimal_pairs.py` 里的 `PAIRS` 列表），覆盖
disaster/election/sports/weather/economy/health 等 6 个语义域，每域 2 对，风格模仿原始 doc
第 13 节给出的例子（"After the tremor.../After the election..."）。⚠️ 人工构造，见第 4 节
已知简化。

## 2. 指标

**不用 default-competitor margin**（GS2/GS8 的口径）——ELF 是双向 denoiser，不是自回归
语言模型，"context-only、无 target"这个中间状态没有一个干净的、能独立编码的表示（没有显式
mask token，`z_t` 本身就是连续噪声）。改用更直接、更贴近 minimal-pair 对照实验标准做法
（类似 BLiMP 风格的最小对比）的操作化：**同一个 target token 在两个不同 context 下的
log-probability 直接对比**。

对每对、每个 `t`，构造 4 个完整编码 `(context, target)`：`(A,target_A)`、`(A,target_B)`、
`(B,target_A)`、`(B,target_B)`，在动态定位的目标位置 `p` 读出 `ell(target token)`。

- `Delta_A = ell(target_A | A, t) - ell(target_A | B, t)`（target_A 在自己的 frame A
  里 vs 被错误地放进 frame B 里，log-prob 差）。
- `Delta_B = ell(target_B | B, t) - ell(target_B | A, t)`（对称地）。

若全局 frame 因果性地塑造局部证据，`Delta_A > 0` 且 `Delta_B > 0`（目标词在自己的 frame
里 log-prob 应该系统性高于放在错误 frame 里）。
- 在多个 `t` 上重复（`t ∈ {0.05, 0.28, 0.65, 0.99}`，覆盖"早期 oracle 状态""GS1/GS8 用过的
  过渡点""接近 cliff 之后""clean"四个阶段），看 `Delta_A`/`Delta_B` 是否随 t 变化、
  是否在 exact-lexical-evidence 尚不明显的早期 t 就已经为正（原始 doc 的判定标准："如果
  global frame 在 lexical evidence 出现前已可恢复，并因果改变 target margin，则支持
  global-to-local conditioning"）。

## 3. 已知简化

1. ⚠️ **12 对全部人工构造**，不是从语料库自动采样——存在实验者选择偏差风险：写这些句子的人
   （本实验里是执行任务的 AI）同时也设计了判定标准，容易无意识挑选"看起来应该成立"的例子。
   缓解措施有限，只能如实标注，正式规模应该找独立标注者或用模板+词表自动生成更大规模的
   对照集，减少单个例子的权重。
2. ⚠️ 目标位置是"动态定位"（编码 `context+" "` 的长度），依赖 T5 tokenizer 对 context 和
   `context+target` 编码的前缀保持一致——大多数情况下成立（BPE-like tokenizer 通常前缀稳定），
   但没有逐对显式验证，个别对可能因为 target 首词和前一个词合并成不同 token 边界而错位。
3. ⚠️ 不含原始 doc 建议的 "global-context swap" 和 "local-window-only control" 两个额外
   对照（本 pilot 只做核心的 `Delta_A`/`Delta_B` 判定），留给后续正式规模。
4. n=12 对，统计效力低，只看方向和一致性（多少对满足 `Delta>0`），不做显著性检验。

## 4. 脚本与输出

```text
experiments/global_state/build_global_minimal_pairs.py
experiments/global_state/probe_global_minimal_pairs.py
```

```text
results/global_state/<model>/<checkpoint>/minimal_pairs_<label>.json
```

## 状态

**Pilot DONE — 结果不干净，作为负面/存疑结果如实报告**（ELF baseline，12 对，
t∈{0.05,0.28,0.65,0.99}，GPU1，`logs/global_state/gs9_elf_baseline_pilot.log`，输出
`results/global_state/elf/baseline/minimal_pairs_pilot.json`）。

## Results

| t | mean(Delta_A) | frac_A>0 | mean(Delta_B) | frac_B>0 | median(Delta_A) | median(Delta_B) |
|---|---|---|---|---|---|---|
| 0.05 | +2.835 | 0.75 | −1.794 | 0.33 | +0.986 | −0.443 |
| 0.28 | −6.643 | 0.58 | +5.614 | 0.67 | +0.654 | +0.429 |
| 0.65 | −1.302 | 0.50 | +1.214 | 0.42 | +0.000 | +0.000 |
| 0.99 | +0.001 | 0.42 | +0.003 | 0.58 | +0.000 | +0.000 |

**解读（如实报告，不做正面 spin）**：

1. **`t=0.65`/`0.99` 的 delta 几乎全部literal 0.000**（逐对数值见脚本输出，大多数对
   在这两个 t 上 `delta_A`/`delta_B` 精确为 0 或 1e-3 量级）。诊断后发现原因：
   **在高 t（接近 clean）时，目标词本身的 token 已经几乎原样出现在待去噪的输入序列里
   （因为构造方式就是把 target 直接编码进 context+target 的完整文本里再加极少量噪声）**，
   模型作为一个双向 denoiser，此时主要是在"读出自己输入里已经写好的词"，而不是"根据左侧
   context 语义推断该填什么词"——不管 context 是 A 还是 B，只要 target 已经原样躺在
   对应位置里，模型都会以接近 1 的置信度预测出它自己，context 的调制效应被这种"自我一致性"
   完全淹没。**这是本次 pilot 最重要的方法论发现**：GLOBAL-9 原始设计隐含假设"高 t/clean
   状态下 context 效应最强"，但对这类双向 denoiser 而言恰恰相反——**高 t 下这个 minimal-pair
   设计几乎测不出东西**，真正有信息量的区间只可能在低/中 t。
2. **`t=0.05`/`0.28` 出现极端离群值**（例如 `crime_tech_2` 在 `t=0.28` 时
   `delta_A=-81.55`，`delta_B=+107.38`；`crime_tech_1` 在 `t=0.05` 时 `delta_A=+19.02`）——
   均值因此被少数几对严重拖动，`t=0.05` 的 `mean(Delta_A)=+2.835` 和
   `mean(Delta_B)=-1.794` 方向不对称（按对称设计，二者符号应该一致），改看 **median**
   更稳健：`t=0.28` 时 `median(Delta_A)=+0.654`、`median(Delta_B)=+0.429`，两个都是正的，
   **方向上弱支持** GLOBAL-9 的判定标准；但 `t=0.05` 时 `median(Delta_A)=+0.986`（正）、
   `median(Delta_B)=-0.443`（负）——不对称，不支持。
3. **12 对、单一噪声种子（每对只用一个共享 epsilon，不做多种子平均）在低 t 下的方差太大**，
   离群值很可能就是"那一个特定噪声实现"造成的随机大幅摆动，而不是真实的语义效应——这是本
   pilot 明确的可修复限制（多个既有实验，如 PT2 的 `n_oracle=4`，都对这种低 t 下的高方差
   做了多噪声种子平均，GS9 pilot 没有做，是复用别处已知教训失败的一个例子）。
4. **总体结论**：GS9 pilot **不能确认也不能证伪** GLOBAL-9 的核心判定（"global frame
   在 lexical evidence 出现前已可恢复，并因果改变 target margin"）——`t=0.28` 的中位数
   结果方向上弱支持，但样本量、噪声方差、以及高 t 区间的方法论缺陷都还没有被控制住，
   不能像 GS8 那样作为一个可信的正面/负面结论引用。

## 下一步

1. **优先修复高-t confound**：不要在高 t 测试 GLOBAL-9 的判定标准（这是原始 doc 隐含假设
   但对双向 denoiser 不成立的地方），把预算集中在低/中 t（比如 0.05–0.4 之间加密）。
2. **每对增加多个噪声种子取平均**（比如 n_oracle=4，复用 `phase_transition` 系列已经验证
   过的做法），压低离群值造成的方差，再看 median/mean 是否稳定。
3. **扩大 n（当前 12 对太少）**，且理想情况下应由不同的人/独立于假设设计者来构造对照句，
   缓解第 3 节标注的实验者选择偏差风险；或者改用模板 + 词表自动生成更大规模（几十到上百对）
   的对照集，用统计检验代替"看方向"。
4. 排查个别离群对（如 `crime_tech_2`）是否有可识别的具体原因（比如某个 target 词在训练语料
   里极其罕见，导致低 t 下 logit 本身就不稳定），必要时替换掉这些对。
