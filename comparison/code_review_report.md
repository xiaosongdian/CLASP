# Comparison 基线方法代码审查报告

## 1. 方法支持情况

✅ **所有6种方法都有完整的代码支持**

| 方法 | 代码位置 | 状态 |
|------|---------|------|
| static_s0 | line 205-206 | ✅ 正常 |
| prefix_refresh | line 208-213 | ✅ 正常 |
| clasp_online | line 167-179, 229-267 | ⚠️ 有问题 |
| incremental_persona | line 215-227 | ✅ 正常 |
| s0_sliding_history | line 141-147, 205-206 | ✅ 正常 |
| user_full_history | line 148-158, 205-206 | ✅ 正常 |

## 2. 发现的问题

### 🔴 严重问题：clasp_online 候选画像评估逻辑错误

**位置：** `window_chain_eval.py:255-262`

**问题描述：**

```python
# 当前代码（有问题）
for cand in candidates:
    if not (cand or "").strip():
        continue
    fc, lc, qc = evaluate_profile_on_window(
        cand,
        hist,      # ❌ W_t：当前步的历史窗口
        targets,   # ❌ W_{t+1}：当前步的目标窗口（刚刚预测过的）
        action_model,
        action_tokenizer,
        semantic_scorer,
    )
    if qc > best_q:
        best_q = qc
        best_profile = cand
```

**问题分析：**

1. **时序混乱：** 画像精炼发生在预测 W_{t+1} **之后**，新画像应该用于预测 W_{t+2}
2. **过拟合风险：** 用刚刚预测过的同一个窗口对（W_t → W_{t+1}）来评估新画像
3. **评估偏差：** 新画像已经针对 W_t → W_{t+1} 的误差进行了优化，在同一数据上评估会高估性能

**影响：**

- 候选画像的选择可能过度拟合当前窗口
- 无法真实反映新画像在未来窗口上的泛化能力
- 可能导致选择了在当前窗口表现好但在后续窗口表现差的画像

**修复方案：**

#### 方案A（推荐）：使用下一步窗口评估

```python
# 在 clasp_online 的候选评估中
if method == "clasp_online":
    discrepancies = build_behavior_discrepancies(
        preds_for_refine, targets, hist
    )
    n_var = max(1, int(refinement_variants))
    candidates = generate_candidate_profiles(
        profile_model,
        profile_tokenizer,
        profile,
        discrepancies,
        n=n_var,
        workers=min(workers, n_var),
    )
    if always_accept_refinement:
        new_p = profile
        for cand in candidates:
            if (cand or "").strip():
                new_p = cand
                break
        profile = new_p
    else:
        best_profile = profile
        best_q = q_s
        
        # ✅ 修复：使用下一步窗口评估
        # 下一步的历史窗口 = 当前步的目标窗口
        next_hist = targets  # W_{t+1}
        # 下一步的目标窗口 = W_{t+2}
        next_targets = windows[keys[step_idx + 2]]
        
        for cand in candidates:
            if not (cand or "").strip():
                continue
            fc, lc, qc = evaluate_profile_on_window(
                cand,
                next_hist,      # ✅ W_{t+1}
                next_targets,   # ✅ W_{t+2}
                action_model,
                action_tokenizer,
                semantic_scorer,
            )
            if qc > best_q:
                best_q = qc
                best_profile = cand
        profile = best_profile
    continue
```

**优点：**
- 真实评估新画像在未来窗口上的性能
- 避免过拟合当前窗口
- 符合在线学习的验证逻辑

**缺点：**
- 增加一次额外的预测调用（每个候选画像）
- 计算成本略高

#### 方案B：保持现状但添加文档说明

如果当前逻辑是有意设计（贪心选择），需要：

1. 在代码注释中明确说明这是"在训练集上选择"
2. 在论文/文档中说明这个设计选择及其影响
3. 考虑添加一个参数让用户选择评估策略

## 3. 其他观察

### ✅ 正确的设计

1. **static_s0：** 正确地保持初始画像不变
2. **prefix_refresh：** 正确地使用 W0..W_{t+1}（包含刚预测的目标窗口）重算画像
3. **incremental_persona：** 正确地只使用当前窗口真实行为（无误差信号）
4. **s0_sliding_history：** 正确地只附加历史窗口 W_t，不包含目标窗口 W_{t+1}
5. **user_full_history：** 正确地使用 W0..W_t（不包含目标窗口）

### ⚠️ 需要注意的边界情况

1. **最后一步处理：** line 201-203 正确地在最后一步跳出，避免越界
2. **空画像处理：** line 225-226, 244-247 正确地处理了空画像的情况
3. **截断逻辑：** line 160-163 正确地截断过长的 profile_suffix

## 4. 修复状态

### ✅ 已修复
- **clasp_online 候选画像评估逻辑** (2026-05-06)
  - 修改位置：`window_chain_eval.py:252-270`
  - 修改内容：使用下一步窗口（W_{t+1} → W_{t+2}）评估候选画像
  - 边界安全：已验证不会越界（is_last 检查确保有足够的窗口）

## 5. 建议

### 后续优化
- 添加单元测试验证每种方法的逻辑正确性
- 添加边界情况测试（只有2个窗口、空画像等）
- 考虑添加日志记录每步的画像更新决策

### 文档改进
- 在 README.md 中明确说明每种方法的评估策略
- 添加时序图说明窗口链的执行流程
- 说明 `always_accept_refinement` 参数的影响
