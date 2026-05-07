# 统一历史窗口使用机制 - 分析与方案

## 一、当前各方法的历史使用情况

### 1.1 现状分析

| 方法 | 画像来源 | 历史窗口使用 | profile_suffix |
|------|---------|-------------|---------------|
| **static_s0** | S0（固定） | ✅ hist（当前窗口） | ❌ None |
| **clasp_online** | S_t（更新） | ✅ hist（当前窗口） | ❌ None |
| **prefix_refresh** | 重算 | ✅ hist（当前窗口） | ❌ None |
| **incremental_persona** | S_{t-1}+当前窗 | ✅ hist（当前窗口） | ❌ None |
| **s0_sliding_history** | S0（固定） | ✅ hist（当前窗口） | ✅ **显式附加当前窗行为** |
| **user_full_history** | 无画像 | ✅ hist（当前窗口） | ✅ **显式附加累积行为** |

### 1.2 关键发现

**所有方法都使用了 `hist`（当前历史窗口）！**

但是：
- **static_s0, clasp_online, prefix_refresh, incremental_persona**: 
  - 只通过 `predict_actions_for_window` 的 `history_actions` 参数传入
  - 使用**滑动历史窗口**（最近 5 条动作）

- **s0_sliding_history, user_full_history**:
  - 除了 `history_actions`，还通过 `profile_suffix` **显式附加**完整的历史行为文本
  - 相当于**双重历史**：滑动窗口 + 显式文本

### 1.3 你的担心

你担心的是：**static_s0 没有使用历史**

**实际情况**：static_s0 **有使用历史**，只是没有像 s0_sliding_history 那样显式附加。

---

## 二、问题的本质

### 2.1 两种历史使用方式

#### 方式 1: 隐式历史（通过 history_actions）
```python
predict_actions_for_window(
    profile=S0,
    history_actions=hist,  # ← 滑动历史（最近5条）
    target_actions=targets,
    profile_suffix=None
)
```

#### 方式 2: 显式历史（通过 profile_suffix）
```python
predict_actions_for_window(
    profile=S0,
    history_actions=hist,  # ← 滑动历史（最近5条）
    target_actions=targets,
    profile_suffix="### Recent behaviors:\n" + format_behavior_data(hist)  # ← 显式附加
)
```

### 2.2 差异的影响

**隐式历史**（方式 1）：
- 历史动作在 prompt 中以**结构化格式**呈现
- 模型需要从历史中**推断**用户偏好
- 历史长度受限（默认最近 5 条）

**显式历史**（方式 2）：
- 历史动作在 prompt 中以**完整文本**呈现
- 模型可以**直接看到**所有历史细节
- 历史长度更长（受 `ACTION_PROMPT_HISTORY_MAX_CHARS` 限制）

---

## 三、你的建议：统一历史使用机制

### 3.1 目标

**让所有方法都使用相同的历史输入机制**，消除"历史使用方式"这个变量，从而：
- 更公平地对比**画像更新策略**的效果
- 避免"性能差异是因为历史使用方式不同"的混淆

### 3.2 两种统一方案

#### 方案 A: 全部使用隐式历史（当前 static_s0 的方式）

**修改**：
- 移除 s0_sliding_history 和 user_full_history 的 `profile_suffix`
- 所有方法都只通过 `history_actions` 使用历史

**优点**：
- 简单，不需要修改代码
- 所有方法完全一致

**缺点**：
- 失去了"显式历史"这个对比维度
- s0_sliding_history 和 user_full_history 失去了原本的设计意图

#### 方案 B: 全部使用显式历史（推荐）✨

**修改**：
- 为所有方法添加 `profile_suffix`，显式附加当前窗口的历史行为
- 保持画像更新策略的差异

**优点**：
- 所有方法使用相同的历史输入
- 更充分地利用历史信息
- 可能提升所有方法的性能

**缺点**：
- 需要修改代码
- 增加 prompt 长度（但在限制范围内）

---

## 四、推荐方案：方案 B（统一使用显式历史）

### 4.1 修改方案

为所有方法添加统一的 `profile_suffix`：

```python
# 在 window_chain_eval.py 中

for step_idx in range(n_keys - 1):
    hist = windows[keys[step_idx]]
    targets = windows[keys[step_idx + 1]]
    wkey = keys[step_idx]
    
    # === 统一：所有方法都使用显式历史 ===
    profile_suffix = (
        f"### Recent behaviors (observed window {wkey})\n"
        + format_behavior_data(hist)
    )
    
    # 截断
    if profile_suffix and int(ACTION_PROMPT_HISTORY_MAX_CHARS) > 0:
        profile_suffix = truncate_behavior_plaintext(
            profile_suffix, int(ACTION_PROMPT_HISTORY_MAX_CHARS)
        )
    
    # 根据方法设置画像
    if method == "static_s0":
        eval_profile = s0_fixed
    elif method == "clasp_online":
        eval_profile = profile
    elif method == "prefix_refresh":
        eval_profile = profile
    elif method == "incremental_persona":
        eval_profile = profile
    elif method == "s0_sliding_history":
        eval_profile = s0_fixed
    elif method == "user_full_history":
        # 特殊：无独立画像
        eval_profile = "No separate long-term persona."
        # 可选：累积历史 vs 当前窗口历史
        # 如果要完全统一，这里也只用当前窗口
    
    # 预测（所有方法都使用 profile_suffix）
    f_s, l_s, q_s = evaluate_profile_on_window(
        eval_profile,
        hist,
        targets,
        action_model,
        action_tokenizer,
        semantic_scorer,
        profile_suffix=profile_suffix,  # ← 所有方法都用
    )
```

### 4.2 修改后的对比

| 方法 | 画像 | 历史（隐式） | 历史（显式） | 对比维度 |
|------|------|------------|------------|---------|
| static_s0 | S0（固定） | ✅ hist | ✅ profile_suffix | 画像不更新 |
| clasp_online | S_t（误差驱动） | ✅ hist | ✅ profile_suffix | 误差驱动更新 |
| prefix_refresh | 重算 | ✅ hist | ✅ profile_suffix | 全量重算 |
| incremental_persona | S_{t-1}+当前窗 | ✅ hist | ✅ profile_suffix | 无误差信号更新 |

**现在所有方法的唯一差异是：画像更新策略**

---

## 五、关于删除方法的建议

### 5.1 统一历史后，可以删除的方法

#### 删除 s0_sliding_history
**理由**：
- 统一历史后，s0_sliding_history = static_s0
- 两者完全相同，保留一个即可

#### 删除 user_full_history（可选）
**理由**：
- 如果统一使用当前窗口历史，user_full_history 失去了"累积历史"的特点
- 但如果保留"累积历史 vs 当前窗口历史"的对比，可以保留

### 5.2 保留的核心方法

统一历史后，保留这 4 个方法：

1. **static_s0**: 画像不更新（基线）
2. **clasp_online**: 误差驱动更新（主方法）
3. **prefix_refresh**: 全量重算更新
4. **incremental_persona**: 无误差信号更新

**这 4 个方法清晰地对比了不同的画像更新策略。**

---

## 六、实施步骤

### Step 1: 修改 window_chain_eval.py

我可以帮你修改代码，为所有方法添加统一的 `profile_suffix`。

### Step 2: 删除 s0_sliding_history

因为统一历史后，它与 static_s0 完全相同。

### Step 3: 决定是否保留 user_full_history

- 如果保留：使用累积历史（W0..W_t）
- 如果删除：只保留 4 个核心方法

---

## 七、总结

### 你的想法非常正确！

✅ **统一历史使用机制**可以：
- 消除"历史使用方式"这个混淆变量
- 更公平地对比画像更新策略
- 简化实验设计

### 我的建议

1. ✅ **采用方案 B**：所有方法都使用显式历史（profile_suffix）
2. ✅ **删除 s0_sliding_history**：统一后与 static_s0 相同
3. ⚠️ **user_full_history**：决定是保留（累积历史）还是删除

### 下一步

你希望我：
1. 立即修改代码实现统一历史？
2. 还是先讨论 user_full_history 的处理？
3. 或者你有其他想法？
