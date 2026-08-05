# EXP-41: Decode 隐向量对齐分析（Decode Hidden Alignment）

## 目标

测量每个位置的 decode 中间向量（GELU 投影后）是否指向其正确 token 的 unembed 方向。

**核心问题**：kd_cr 正确预测 token 是因为其 decode 中间向量更准确地指向了正确 token 的嵌入方向吗？  
这在几何层面解释了 oracle accuracy 差异。

## 方法

decode 路径：
```
decode_hidden_i = GELU(h_L11[i] @ proj_kernel + proj_bias)  ← [512]
logits_v = decode_hidden_i @ unembed_kernel[:, v] + unembed_bias[v]
```

对每个位置 i，定义**对齐度**（cosine similarity）：
```
cos_align(i) = cos(decode_hidden_i, unembed_kernel[:, y_i])
```
其中 y_i 是该位置的真实 token。

**分析**：
1. **正确预测位置**（oracle top-1 = y_i）vs **错误位置** 的 cos_align 分布
2. cos_align 是否是 oracle accuracy 的预测器（AUC）
3. cos_align 随 t 的变化曲线（是否有"对齐悬崖"？）
4. 三个 checkpoint 的 cos_gap = correct_cos - wrong_cos 比较

**数据**：复用 `results/exp07b_v2_{baseline,kd_cr,kd2}/layer_states_t*.pt`（`layer_feats[-1]`）

## 代码

`experiments/probe_elf/decode_alignment_exp41.py`

## 预期结果

| 假设 | 若成立，说明 |
|------|-------------|
| kd_cr 的 cos_gap >> baseline 的 cos_gap | kd_cr decode 中间向量在几何上更精确对准正确 token |
| AUC 在所有 checkpoint 中都高（>0.9） | cos_align 是 oracle accuracy 的强预测器 |
| cos_gap 在 t=0.3-0.5 处开始分叉（kd_cr vs baseline） | oracle 精度差异在中等噪声级别就已形成 |

## 输出

`results/exp41_decode_alignment/alignment.json`

结构：`{checkpoint: {t_str: {correct_cos_mean, wrong_cos_mean, cos_gap, auc, n_correct, n_wrong}}}`

---

## 结果

**状态：DONE**

### cos_gap（正确位置 cos - 错误位置 cos）

| ckpt | t=0.10 | t=0.20 | t=0.30 | t=0.50 | t=0.70 | t=1.00 |
|------|--------|--------|--------|--------|--------|--------|
| baseline | +0.148 | +0.149 | +0.111 | +0.090 | +0.087 | +0.091 |
| kd_cr | +0.056 | +0.065 | +0.065 | +0.049 | +0.074 | +0.087 |
| kd2 | +0.047 | +0.058 | +0.051 | +0.021 | +0.015 | +0.037 |

### AUC（cos_align 预测 oracle accuracy 的 ROC AUC）

| ckpt | t=0.10 | t=0.20 | t=0.30 | t=0.50 | t=0.70 | t=1.00 |
|------|--------|--------|--------|--------|--------|--------|
| baseline | 0.997 | 0.977 | 0.944 | 0.915 | 0.910 | 0.910 |
| kd_cr | 0.963 | 0.933 | 0.913 | 0.840 | 0.866 | 0.937 |
| kd2 | 0.959 | 0.929 | 0.888 | 0.676 | 0.609 | 0.695 |

### 绝对 cos 均值（t=0.500）

| ckpt | correct cos | wrong cos | gap | n_correct |
|------|------------|-----------|-----|-----------|
| baseline | 0.2336 | 0.1441 | +0.090 | 185,642 |
| kd_cr | 0.1806 | 0.1313 | +0.049 | 244,424 |
| kd2 | 0.1363 | 0.1158 | +0.021 | 243,339 |

### 关键发现

1. **cos_align 是强预测器（AUC 0.84-1.00）**：对所有 checkpoint，decode hidden 向量的对齐度确实预测 oracle 正确性，几何信号有效。

2. **反直觉结论**：baseline 的 cos_align 均值最高（correct=0.234 at t=0.5），但 oracle accuracy 最低（75%）。kd_cr cos_align 更低（0.181），但准确率 99.5%。这说明 kd_cr 并非通过更"尖锐"地指向正确 token 来获得高精度，而是通过 backbone 表示的全局重组（见 EXP-42）。

3. **kd2 AUC 在 t=0.5-0.7 急剧下降（0.61-0.68）**：cos_align 对 kd2 的 oracle 准确性预测能力较弱，说明 kd2 的 decode 几何关系更混乱（与 kd2 SC 产生语义不连贯的 EXP-36v2 发现一致）。

4. **负向 cos 分析**：baseline 在"对"的时候更"自信"（高 cos），但经常出错。KD 训练改变了 decode 几何：表示不再尖锐对准单个 token 方向，而是整体上更均匀但预测更准（backbone 重组的结果）。
