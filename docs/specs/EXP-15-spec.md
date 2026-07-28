# EXP-15 Spec — Parameter-Space Layer Analysis

## 实验背景与动机（为什么要做这个实验）

**在整体框架中的地位：从参数变化量定位 KD 训练的"作用层"。**

EXP-07c-full 从**激活空间**（hidden states）发现 L10 是跨 checkpoint 迁移率最低的层。但这只是间接证据。直接的问题是：**KD 训练最多地改变了哪些层的参数？**

如果参数变化最大的层 = 激活迁移性最低的层（L10），则两个分析相互印证，说明 L10 的几何发散确实由 KD 对该层参数的直接修改导致。

**要验证的核心假说**：
- KD 训练（从 baseline 到 kd-cr）在 L10/L11 的参数变化相对 L2 范数最大
- 参数变化最大的层 ≈ 表示迁移性最低的层（来自 EXP-07c-full）

**方法**：计算 kd-cr 与 baseline 之间，每个 transformer block 的参数相对 L2 范数变化量：
```
relative_change[i] = ||θ_kd-cr[block_i] - θ_baseline[block_i]||_F / ||θ_baseline[block_i]||_F
```

**与其他实验的关系**：
- 作为 EXP-07c/07c-full 的参数空间对应分析
- 两个层面（参数空间 + 激活空间）的一致性增强了 L10 作为"KD 作用层"的结论可信度

---

## Implementation

简单脚本，不需要 GPU：

```python
import torch

baseline = torch.load("converted/elf_b-owt-baseline_torch.pt", map_location="cpu", weights_only=False)
kd_cr = torch.load("converted/elf_b-owt-kd-cr_torch.pt", map_location="cpu", weights_only=False)

base_params = baseline.get("params", baseline)
kd_params = kd_cr.get("params", kd_cr)

# Group params by block index
for i in range(12):
    # Find params matching blocks[i]
    block_keys = [k for k in base_params if f"blocks.{i}." in k]
    
    total_diff_sq = 0.0
    total_base_sq = 0.0
    for k in block_keys:
        diff = (kd_params[k].float() - base_params[k].float())
        total_diff_sq += diff.norm().item() ** 2
        total_base_sq += base_params[k].float().norm().item() ** 2
    
    rel_change = (total_diff_sq ** 0.5) / (total_base_sq ** 0.5)
    print(f"Block {i:2d}: relative L2 change = {rel_change:.4f}")
```

---

## 实验结果（Results）

**状态**: ⚠️ SUPERSEDED by EXP-15v2 — 原始数值有误，见 v2 结果

---

### EXP-15v2 结果（2026-07-22，analyze_param_distance.py，module-level 分解）

**Block-level 相对 L2 变化（baseline → kd-cr / kd2）**：

| 层 | kd_cr R_l | kd2 R_l | cos(Δkd_cr, Δkd2) |
|----|----------:|--------:|------------------:|
| L0-L3  | 0.215–0.238 | 0.219–0.242 | 0.808–0.833 |
| L4-L8  | 0.211–0.237 | 0.212–0.241 | 0.815–0.860 |
| L9     | 0.280 | 0.288 | 0.877 |
| L10    | 0.338 | 0.344 | 0.891 |
| L11    | 0.338 | 0.338 | 0.902 |

**⚠️ 原 spec 数据（L0-L3: 0.08-0.12）错误**：v2 用 module-level 分解后早期层实际变化 ~0.21-0.24，不是原来声称的 0.08-0.12。原始测量可能使用了不同的 normalization 或 parameter grouping。

**最重要的新发现：Decode head 参数变化远大于 transformer blocks**：

| 参数 | kd_cr R |
|------|--------:|
| unembed_bias | **2.59** |
| final_layer.linear.* | 0.58–1.23 |
| proj_kernel | 0.36 |
| unembed_kernel | 0.32 |
| proj_bias | 0.29 |

**Module-level 分解（blocks 内部）**：
- LayerNorm 参数变化最小（R ≈ 0.04–0.09）
- attn_qkv / attn_out / mlp_up / mlp_down 变化相似（R ≈ 0.21–0.38，随层深增加）
- Late blocks 各模块绝对量均高于 early blocks

**Update direction similarity（kd_cr vs kd2 更新方向一致性）**：
- L0: cos = 0.83 → L11: cos = 0.90
- 两种 KD 变种在 late blocks 的更新方向更加对齐，说明 late block 改变是 KD objective 驱动的系统性重组

---

### 修订后的安全结论（替代原 spec 结论）

> Both KD variants show largest relative parameter changes in the **decode head** (unembed_bias R≈2.6, proj_kernel R≈0.36) and late transformer blocks (L10-L11 R≈0.34 vs L0-L8 R≈0.21). Update directions in late blocks are more correlated between kd_cr and kd2 (cos≈0.90 at L11 vs 0.83 at L0), suggesting KD-driven systematic reorganization concentrated in the native decode interface.

### ~~原结论（已废弃）~~

~~"L10 是 KD 训练的核心作用层"~~ — 早期层实际也有 ~0.22 变化，且 decode head 变化更大（2.59x）。更准确的表述是：KD 对 decode head 的修改比例最大，其次是 late blocks。

~~"KD 主要通过微调高层来调整表示几何，不重新训练低层"~~ — 所有层都有 ~0.22 变化，差异是程度（late：0.34 vs early：0.22），不是有/无。

**结果保存位置**：`results/exp15v2/param_distance.json`
