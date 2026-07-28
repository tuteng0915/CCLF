# EXP-29 Spec: kNN Word Visualization per Position per Timestep

**状态**: DONE (2026-07-21)  
**目的**: 定性可视化——在每个位置和每个噪声水平 t 下，ELF backbone 的 x̂_t 最接近哪些 token？  
**关键问题**: kd_cr 模型是否比 baseline 更早地让 x̂_t[pos] 接近正确 token 的 centroid？这一差异在哪个 t 区间最明显？

---

## 背景

EXP-07 已证明：probe_acc（线性探针精度）和 decoder_rec1（native decode）之间存在显著 gap。  
但这是聚合统计量。EXP-29 提供逐位置、逐 t 的定性视角：  
- 每个位置在每个 t 下"在想哪个词"（cosine 最近邻 centroid）  
- 这个词是否是真实词（绿色），还是近似（橙色）或错误（红色）  
- baseline vs kd_cr vs kd2 的对比

---

## 设计

### 数据来源
- 直接读取 EXP-07 64-step 已收集的状态文件  
  `results/exp07_{baseline,kd_cr,kd2}_64/states/states_t*.pt`  
- 每个文件含 `x_hat: (512, 1024, 512)` — 512 个序列，1024 个位置，512 维 x̂_t

### Token Centroid 匹配
- 使用 `results/data/token_centroids.npz`（(32100, 512) float32）  
- 余弦相似度：`cos(x̂_t[pos], centroid[v])` → 取 top-3 最近 token  
- 正确性判断：top-1 == gt_id → correct；gt_id in top-3 → near；否则 wrong

### Oracle 协议
- 固定 ε（各 state 文件独立随机），sweep t  
- 与 EXP-07 完全一致（同一批 state 文件）

### 参数
```
N_SEQS = 5     # 取前 5 个样本序列
N_POS  = 36    # 每序列取前 36 个位置
K      = 3     # top-K 近邻
t_grid = np.linspace(0.05, 1.00, 64)  # 64 个 t 值
```

### 样本序列（来自 OpenWebText-T5，前 5 个）
- seq0: "Port-au-Prince, Haiti (CNN) -- Earthquake victims, w..."
- seq1: "A 7.0 magnitude earthquake flattened Haiti's capital city..."
- seq2: "But that path looked much more difficult, if not impossible..."
- seq3: "In Youngstown, Ohio, Dave Williams, 52, cast a ballot..."
- seq4: "Let me give you another example. Remember Obama's depiction..."

---

## 脚本

```
experiments/probe_elf/probe_knn_words.py
```

用法（从 ELF-torch 根目录）：
```bash
CUDA_VISIBLE_DEVICES=2 conda run -n elf python experiments/probe_elf/probe_knn_words.py \
    --output_dir results/exp29
```

输出：
- `results/exp29/knn_words.json` — 所有数据（~30MB）
- `results/exp29/knn_viz.html` — 交互式可视化

---

## 可视化设计

交互式 HTML：
- 顶部：Checkpoint 选择器（baseline / kd_cr / kd2）
- 左侧：Sequence 选择器（5 个样本）
- 主体：Grid（行=位置，列=t 值），每格显示 top-1 最近 token
  - 绿色：top-1 == 正确词
  - 橙色：正确词在 top-3 内
  - 红色：top-3 内无正确词
- Hover tooltip：显示 top-3 词及相似度

---

## 预期结果

**kd_cr**: 在 t=0.20–0.30 就能看到大量绿色（多数位置 top-1 已正确）  
**baseline**: 绿色主要出现在 t=0.50+ 区域，前期以橙/红为主  
**kd2**: 介于两者之间

这与 EXP-07 的 probe_acc 结果（kd_cr 在 t=0.20 已达 90%+ probe accuracy）一致，  
但 EXP-29 的贡献在于提供**定性视角**：在每个具体位置上，承诺的确是逐步且有方向的。

---

## 结果

**完成时间**: 2026-07-21 06:03 UTC  
**输出文件**: 
- `results/exp29/knn_words.json` (3.3MB)
- `results/exp29/knn_viz.html` (3.0MB)
- **交互式可视化**: https://claude.ai/code/artifact/4f181198-25a4-4c19-9037-c7027845527c

### 关键发现（以 seq1 "…7.0 magnitude earthquake…" pos=4 为例）

| t | baseline top-3 | kd_cr top-3 | kd2 top-3 |
|---|---------------|------------|----------|
| 0.050 | s, is, are | ribbon, fashionable, submitting | fashionable, ribbon, submitting |
| 0.110 | in, are, s | storm, crisis, humidity | surfacing, humidity, flare |
| **0.201** | storm, **earthquake**, storm | storm, **earthquake**, hurricane | **earthquake**, storm, hurricane |
| 0.502 | **earthquake**, tsunami, hurricane | **earthquake**, hurricane, tsunami | **earthquake**, hurricane, storm |
| 1.000 | **earthquake**, tsunami, disaster | **earthquake**, hurricane, tsunami | **earthquake**, tsunami, eruption |

**解读**：
1. **t=0.05**: 三个模型全部错误（baseline 预测功能词 s/is/are，kd_cr/kd2 预测随机内容词）
2. **t=0.20**: kd2 已经把 "earthquake" 排到 top-1；kd_cr 排 top-2；baseline 排 top-2（与 storm 并列）
3. **t=0.50+**: 全部模型稳定预测 "earthquake" 为 top-1
4. **top-3 的语义一致性**: 即使 top-1 错，top-3 中已出现 earthquake/tsunami/hurricane/storm 等语义相关词——模型很早就知道"这是灾害相关词语"

**与 EXP-07 的一致性 & 重要局限**：

EXP-29 使用的 token centroids 是从 **baseline 模型**输出计算的。因此：
- 对 baseline 模型，kNN 精度高（与 centroid 空间匹配）
- 对 kd_cr/kd2，kNN 精度略低——不是因为表示更差，而是因为它们的 x̂_t 空间与 baseline centroids 不完全对齐

**跨 checkpoint 的 kNN 精度（180 个位置汇总）**：
| t | baseline | kd_cr | kd2 |
|---|---------|-------|-----|
| 0.05 | 5.0% | 3.9% | 1.7% |
| **0.20** | **53.9%** | **45.6%** | **46.1%** |
| 0.50 | 91.1% | 90.6% | 88.9% |
| 1.00 | 94.4% | 92.8% | 93.9% |

- **t=0.20 cliff 清晰可见**（baseline: 10.6%→53.9%, +43pp）
- **跨 checkpoint 比较不公平**（centroid 偏向 baseline）——use EXP-07 probe_acc for that
- **kNN "commits earlier" 统计**：baseline=30 positions, kd_cr=7（偏差确认）

**EXP-29 的正确用途**：
- ✅ 单个 checkpoint 内随 t 变化的语义邻居演化（定性图表）
- ✅ 展示模型"在想哪些词"的质性描述（正确词及其语义近邻）
- ❌ 跨 checkpoint 早期承诺时间比较（需 checkpoint-specific centroids）
- ❌ 替代 EXP-07 的定量分析

**纸面结论**：EXP-29 作为 EXP-07 的定性补充，展示承诺过程；跨 checkpoint 的时序比较应引用 EXP-07。

---

## 与 CCLF Paper 的关联

EXP-29 作为 EXP-07 的**定性补充**，可作为附录图表或论文 Figure：  
- 直观展示 "KD training 使 x̂_t 更早锁定正确 token" 的具体表现  
- 可选展示 func vs content word 的承诺时序差异（EXP-08 的 qualitative evidence）

---

## ⚠️ 方法论问题（2026-07-22 审查）

### 1. ✅ Fixed noise 已审计确认（2026-07-22）

**代码审计结果**：`probe_layerwise.py:116-134`（exp07b_v2 收集脚本）明确实现固定噪声：

```python
# Generate fixed noise once and reuse across all t values so that
# commit_times comparisons across t use the same ε draw per position.
g_noise = torch.Generator(device='cpu').manual_seed(fixed_noise_seed)
eps_all = torch.randn(x_clean_all.shape, generator=g_noise)  # (N, L, d)
...
eps = eps_all[sl].to(device)  # same ε for all t values ← 明确复用
z_t = t_val * x_c + (1.0 - t_val) * eps
```

`exp07b_v2_*` 的所有 t 值使用**完全相同的 ε（seed=42）**，因此 EXP-29 的跨 t 可视化展示的是同一固定噪声路径下的演化，不是独立采样截面。此批评**已撤销**。

kNN 可视化可以合理称为"固定噪声轨迹上的近邻演化"。

### 2. CRITICAL：baseline centroid 偏置令跨 checkpoint 比较无效

token centroids 来自 baseline checkpoint，因此：
- baseline 的 x̂_t 与 centroid 空间匹配；
- kd_cr/kd2 的 x̂_t 可能发生 rotation，导致跨 checkpoint kNN accuracy 低估。

当前"baseline kNN 高于 kd_cr/kd2"完全可以由几何偏置解释，而非"baseline 早期承诺更准确"。spec 已标注"跨 checkpoint 比较不公平"，但这个结论不够充分——**所有涉及跨 checkpoint kNN 数值比较的论文表述必须删除**。

checkpoint-specific centroids 可改善 within-checkpoint 比较，但跨 checkpoint 数值仍不可比（几何度量不同）。真正公平的方案是：Procrustes 对齐后比较，或使用外部 lexical reference space。

### 3. Centroid 可能有数据泄漏

必须明确 token centroids 在哪些文本上计算。若 centroid 使用了当前展示的 5 条 sequence 或其所在 validation set，则 nearest-centroid 结果被自身样本污染。Centroid 必须仅由独立训练 split 计算，展示与评估使用 held-out documents。

### 4. 案例分析无法作为系统证据

"earthquake 周围出现 storm/hurricane/tsunami"是案例级证据，容易被 cherry-pick。"语义邻居先出现"的系统化验证需要：
- 自动计算 M_t(C_y) = Σ_{v∈C_y} p_t(v)（真实 token 语义 cluster 的概率质量）
- 控制组：frequency-matched random clusters、same-size random clusters
- 预注册的 selection policy（非后验选择最好的案例）

### 5. 安全定位

EXP-29 **只适合**作为 EXP-07/08 的 appendix qualitative illustration，用单个 checkpoint 内随 t 变化的近邻演化做定性展示。

**不能支撑**：KD 优于 baseline（centroid 偏置）；语义承诺（未系统化）；粗到细机制（单点证据）；真实轨迹演化（noise 独立性未确认）。
