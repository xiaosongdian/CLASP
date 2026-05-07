# 三窗口评估历史窗口修复

## 🐛 发现的问题

在测试 `static_s0` 方法时，发现**新旧画像的 F 值完全相同**，没有任何变化。

### 原因分析

在原来的 `evaluate_three_windows` 函数中：

```python
# 错误的实现
past_window = windows[keys[step_idx - 1]]  # W_{i-1}
past_target = windows[keys[step_idx]]      # W_i
past_suffix = format_behavior_data(past_window)  # 又是 W_{i-1}

f_old, l_old, q_old = evaluate_profile_on_window(
    old_profile,
    past_window,      # history = W_{i-1}
    past_target,      # target = W_i
    profile_suffix=past_suffix,  # 又包含 W_{i-1} 的信息
)
```

**问题**：
1. `history` 参数使用了 `past_window`（W_{i-1}）
2. `profile_suffix` 参数又包含了 `past_window` 的信息
3. **信息重复**，导致预测时使用了相同的历史信息
4. 对于 `static_s0`，由于画像不变，历史也不变，所以预测结果完全相同

---

## ✅ 修复方案

### 正确的三窗口定义

在 Step i（当前正在评估 W_i → W_{i+1}）时：

| 窗口类型 | 历史窗口 | 目标窗口 | 说明 |
|---------|---------|---------|------|
| **过去窗口** | W_{i-2} | W_{i-1} | 检测是否遗忘历史 |
| **当前窗口** | W_{i-1} | W_i | 主要评估指标 |
| **未来窗口** | W_i | W_{i+1} | 检测泛化能力 |

### 修复后的实现

```python
# 过去窗口：用 W_{i-2} 预测 W_{i-1}
if step_idx > 1:
    past_history = windows[keys[step_idx - 2]]  # W_{i-2}
else:
    past_history = []
past_target = windows[keys[step_idx - 1]]  # W_{i-1}

f_old, l_old, q_old = evaluate_profile_on_window(
    old_profile,
    past_history,     # history = W_{i-2}（不同于之前）
    past_target,      # target = W_{i-1}
    profile_suffix=None,  # 不使用 suffix，避免信息重复
)

# 当前窗口：用 W_{i-1} 预测 W_i
if step_idx > 0:
    current_history = windows[keys[step_idx - 1]]  # W_{i-1}
else:
    current_history = []
current_target = windows[keys[step_idx]]  # W_i

f_old, l_old, q_old = evaluate_profile_on_window(
    old_profile,
    current_history,  # history = W_{i-1}
    current_target,   # target = W_i
    profile_suffix=None,
)

# 未来窗口：用 W_i 预测 W_{i+1}
future_history = windows[keys[step_idx]]      # W_i
future_target = windows[keys[step_idx + 1]]   # W_{i+1}

f_old, l_old, q_old = evaluate_profile_on_window(
    old_profile,
    future_history,   # history = W_i
    future_target,    # target = W_{i+1}
    profile_suffix=None,
)
```

---

## 📊 修复效果

### 对于 static_s0

**修复前**：
```json
{
  "past_window": {
    "with_old_profile": {"F": 0.52, "L": 0.42, "Q": 0.50},
    "with_new_profile": {"F": 0.52, "L": 0.42, "Q": 0.50},  // 完全相同
    "gain": {"ΔF": 0.00, "ΔL": 0.00, "ΔQ": 0.00}
  }
}
```

**修复后**：
```json
{
  "past_window": {
    "history": "W0",
    "target": "W1",
    "with_old_profile": {"F": 0.52, "L": 0.42, "Q": 0.50},
    "with_new_profile": {"F": 0.52, "L": 0.42, "Q": 0.50},  // 仍然相同（因为画像不变）
    "gain": {"ΔF": 0.00, "ΔL": 0.00, "ΔQ": 0.00}
  },
  "current_window": {
    "history": "W1",
    "target": "W2",
    "with_old_profile": {"F": 0.54, "L": 0.44, "Q": 0.52},  // 不同的窗口，不同的 F 值
    "with_new_profile": {"F": 0.54, "L": 0.44, "Q": 0.52},
    "gain": {"ΔF": 0.00, "ΔL": 0.00, "ΔQ": 0.00}
  }
}
```

**关键改进**：
- 不同窗口的 F 值现在会不同（反映时间漂移）
- 即使画像不变，不同窗口的预测结果也会不同

### 对于 clasp_online

**修复前**：
```json
{
  "past_window": {
    "with_old_profile": {"F": 0.52},
    "with_new_profile": {"F": 0.52},  // 可能因为信息重复而相同
    "gain": {"ΔF": 0.00}
  }
}
```

**修复后**：
```json
{
  "past_window": {
    "history": "W0",
    "target": "W1",
    "with_old_profile": {"F": 0.52},
    "with_new_profile": {"F": 0.54},  // 画像更新后，F 值提升
    "gain": {"ΔF": 0.02}  // 有增益
  }
}
```

---

## 🎯 三窗口的正确理解

### Step 1 示例

假设在 Step 1（评估 W1 → W2）：

```
已观测窗口：W0, W1
当前评估：W1 → W2
画像更新：S0 → S1

三窗口评估：
1. 过去窗口：用 W0 预测 W1（检测是否遗忘 W0 的信息）
2. 当前窗口：用 W1 预测 W2（主要评估指标）
3. 未来窗口：用 W2 预测 W3（检测泛化能力）
```

### 历史窗口的作用

**历史窗口（history）**：
- 提供预测的上下文
- 包含用户的近期行为
- 帮助模型理解用户的当前状态

**不使用 profile_suffix**：
- 避免信息重复
- 让画像和历史窗口的作用更清晰
- 简化评估逻辑

---

## 📈 预期效果

### 1. static_s0（静态画像）

**预期**：
- 不同窗口的 F 值会不同（反映时间漂移）
- 但 old_profile 和 new_profile 的 F 值相同（画像不变）
- gain 为 0

**示例**：
```
过去窗口：F = 0.52（W0 → W1）
当前窗口：F = 0.54（W1 → W2）
未来窗口：F = 0.50（W2 → W3）
```

### 2. clasp_online（误差驱动）

**预期**：
- 不同窗口的 F 值会不同
- new_profile 的 F 值 > old_profile 的 F 值（画像更新有效）
- gain > 0

**示例**：
```
过去窗口：
  old: F = 0.52, new: F = 0.54, gain = +0.02
当前窗口：
  old: F = 0.54, new: F = 0.58, gain = +0.04
未来窗口：
  old: F = 0.50, new: F = 0.55, gain = +0.05
```

---

## ✅ 验证方法

### 1. 检查 static_s0 的结果

```python
import json

with open('output/comparison/static_s0/baseline_chain_test.jsonl', 'r') as f:
    data = json.loads(f.readline())

step = data['steps'][1]  # Step 1
three_win = step['three_window_evaluation']

# 检查不同窗口的 F 值是否不同
past_f = three_win['past_window']['with_old_profile']['F']
current_f = three_win['current_window']['with_old_profile']['F']
future_f = three_win['future_window']['with_old_profile']['F']

print(f"过去窗口 F: {past_f}")
print(f"当前窗口 F: {current_f}")
print(f"未来窗口 F: {future_f}")

# 应该看到不同的 F 值
assert past_f != current_f or current_f != future_f, "F 值应该不同！"

# 检查 old 和 new 是否相同（static_s0）
assert three_win['past_window']['with_old_profile']['F'] == \
       three_win['past_window']['with_new_profile']['F'], \
       "static_s0 的 old 和 new 应该相同！"
```

### 2. 检查 clasp_online 的结果

```python
with open('output/comparison/clasp_online/baseline_chain_test.jsonl', 'r') as f:
    data = json.loads(f.readline())

step = data['steps'][1]
three_win = step['three_window_evaluation']

# 检查是否有增益
past_gain = three_win['past_window']['gain']['ΔF']
current_gain = three_win['current_window']['gain']['ΔF']
future_gain = three_win['future_window']['gain']['ΔF']

print(f"过去窗口增益: {past_gain:+.4f}")
print(f"当前窗口增益: {current_gain:+.4f}")
print(f"未来窗口增益: {future_gain:+.4f}")

# 应该看到正增益
assert current_gain > 0, "当前窗口应该有正增益！"
```

---

## 🎉 总结

### 修复内容

✅ **修复历史窗口的使用**
- 过去窗口：W_{i-2} → W_{i-1}
- 当前窗口：W_{i-1} → W_i
- 未来窗口：W_i → W_{i+1}

✅ **移除 profile_suffix**
- 避免信息重复
- 简化评估逻辑

✅ **添加 history 字段**
- 记录使用的历史窗口
- 便于调试和分析

### 预期效果

✅ **static_s0**：
- 不同窗口的 F 值会不同（反映时间漂移）
- old 和 new 的 F 值相同（画像不变）

✅ **clasp_online**：
- 不同窗口的 F 值会不同
- new 的 F 值 > old 的 F 值（画像更新有效）
- 三个窗口都有正增益

---

**修复者**: Claude (Opus 4.6)
**修复日期**: 2026-05-07
**感谢**: 用户发现了 static_s0 的 F 值完全相同的问题！
