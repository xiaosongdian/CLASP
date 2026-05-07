# 三窗口评估优化：只统计最后一次更新

## 🎯 优化目标

降低三窗口评估的计算开销，只在**最后一次画像更新**时进行三窗口评估。

---

## 📊 优化前后对比

### 优化前

**所有步骤都进行三窗口评估**：

```
Step 0: 生成 S0，三窗口评估 ✓
Step 1: S0 → S1，三窗口评估 ✓
Step 2: S1 → S2，三窗口评估 ✓
Step 3: S2 → S3，三窗口评估 ✓
Step 4: S3 → S4，三窗口评估 ✓
```

**计算成本**：
- 4 个方法 × 1,801 个用户 × 5 步 × 6 次预测 = **216,120 次**
- 预计时间：**6-9 小时**

### 优化后

**只在最后一步进行三窗口评估**：

```
Step 0: 生成 S0
Step 1: S0 → S1
Step 2: S1 → S2
Step 3: S2 → S3
Step 4: S3 → S4，三窗口评估 ✓（只有这一次）
```

**计算成本**：
- 4 个方法 × 1,801 个用户 × 1 步 × 6 次预测 = **43,224 次**
- 预计时间：**1-2 小时**（额外）

**节省**：
- 减少 **80%** 的三窗口评估次数
- 节省 **4-7 小时**

---

## 🔧 实现逻辑

### 判断条件

```python
is_last_step = (step_idx >= n_keys - 2)

if method == "static_s0":
    # static_s0: 只在最后一步记录一次（作为基线）
    should_evaluate = is_last_step
else:
    # 其他方法: 只在最后一次画像更新时记录
    should_evaluate = is_last_step and (old_profile != profile)
```

### 各方法的行为

| 方法 | 三窗口评估时机 | 说明 |
|------|--------------|------|
| **static_s0** | 最后一步 | 画像不变，只记录一次作为基线 |
| **prefix_refresh** | 最后一步（如果有更新） | 记录最终画像的三窗口表现 |
| **incremental_persona** | 最后一步（如果有更新） | 记录最终画像的三窗口表现 |
| **clasp_online** | 最后一步（如果有更新） | 记录最终画像的三窗口表现 |

---

## 📈 科学价值

### 为什么只统计最后一次就够了？

#### 1. 最终画像最重要

**目标**：评估各方法的最终画像质量

- 中间步骤的画像是过渡状态
- 最终画像代表了方法的最终效果
- 只需要对比各方法的最终画像即可

#### 2. 三窗口评估的目的

**回答的问题**：
- ✅ 最终画像是否遗忘历史？（past_window）
- ✅ 最终画像在当前窗口的表现？（current_window）
- ✅ 最终画像的泛化能力？（future_window）

**不需要回答的问题**：
- ❌ 中间步骤的画像表现如何？（不重要）
- ❌ 画像更新的轨迹如何？（可以通过 mean_Q 看出）

#### 3. 避免窗口索引问题

**问题**：
- 早期步骤可能没有足够的历史窗口（step_idx < 2）
- 后期步骤可能没有足够的未来窗口（step_idx + 2 >= len(keys)）

**解决**：
- 只在最后一步评估，确保有足够的窗口数据
- 避免边界情况的处理

---

## 📊 输出数据结构

### 示例输出

```json
{
  "user_id": 539166,
  "method": "clasp_online",
  "steps": [
    {
      "step_index": 0,
      "F": 0.62,
      "L": 0.45,
      "Q": 0.55
      // 没有 three_window_evaluation
    },
    {
      "step_index": 1,
      "F": 0.64,
      "L": 0.47,
      "Q": 0.57
      // 没有 three_window_evaluation
    },
    {
      "step_index": 2,
      "F": 0.66,
      "L": 0.48,
      "Q": 0.59
      // 没有 three_window_evaluation
    },
    {
      "step_index": 3,
      "F": 0.67,
      "L": 0.49,
      "Q": 0.60
      // 没有 three_window_evaluation
    },
    {
      "step_index": 4,
      "F": 0.68,
      "L": 0.50,
      "Q": 0.61,
      // 只有最后一步有 three_window_evaluation
      "three_window_evaluation": {
        "past_window": {
          "history": "W2",
          "target": "W3",
          "with_old_profile": {"F": 0.66, "L": 0.48, "Q": 0.59},
          "with_new_profile": {"F": 0.67, "L": 0.49, "Q": 0.60},
          "gain": {"ΔF": 0.01, "ΔL": 0.01, "ΔQ": 0.01}
        },
        "current_window": {
          "history": "W3",
          "target": "W4",
          "with_old_profile": {"F": 0.67, "L": 0.49, "Q": 0.60},
          "with_new_profile": {"F": 0.68, "L": 0.50, "Q": 0.61},
          "gain": {"ΔF": 0.01, "ΔL": 0.01, "ΔQ": 0.01}
        },
        "future_window": {
          "history": "W4",
          "target": "W5",
          "with_old_profile": {"F": 0.65, "L": 0.48, "Q": 0.58},
          "with_new_profile": {"F": 0.67, "L": 0.50, "Q": 0.60},
          "gain": {"ΔF": 0.02, "ΔL": 0.02, "ΔQ": 0.02}
        }
      },
      "profile_changed": true
    }
  ]
}
```

---

## 📈 分析方法

### 提取最终画像的三窗口统计

```python
import json
import numpy as np

# 读取结果
with open('output/comparison/clasp_online/baseline_chain_test.jsonl', 'r') as f:
    results = [json.loads(line) for line in f]

# 提取最终画像的三窗口统计
final_three_window_stats = []

for user in results:
    # 找到最后一步
    last_step = user['steps'][-1]
    
    # 检查是否有三窗口评估
    if 'three_window_evaluation' in last_step:
        three_win = last_step['three_window_evaluation']
        
        # 提取增益
        past_gain = three_win.get('past_window', {}).get('gain', {}).get('ΔQ', 0)
        current_gain = three_win.get('current_window', {}).get('gain', {}).get('ΔQ', 0)
        future_gain = three_win.get('future_window', {}).get('gain', {}).get('ΔQ', 0)
        
        final_three_window_stats.append({
            'user_id': user['user_id'],
            'past_gain': past_gain,
            'current_gain': current_gain,
            'future_gain': future_gain,
        })

# 统计
past_gains = [s['past_gain'] for s in final_three_window_stats]
current_gains = [s['current_gain'] for s in final_three_window_stats]
future_gains = [s['future_gain'] for s in final_three_window_stats]

print(f"最终画像的三窗口统计（{len(final_three_window_stats)} 个用户）：")
print(f"  过去窗口平均增益: {np.mean(past_gains):.4f} ± {np.std(past_gains):.4f}")
print(f"  当前窗口平均增益: {np.mean(current_gains):.4f} ± {np.std(current_gains):.4f}")
print(f"  未来窗口平均增益: {np.mean(future_gains):.4f} ± {np.std(future_gains):.4f}")
print(f"  遗忘率: {sum(1 for g in past_gains if g < 0) / len(past_gains):.2%}")
print(f"  泛化比: {np.mean(future_gains) / np.mean(current_gains):.2f}")
```

### 对比各方法

```python
methods = ['static_s0', 'clasp_online', 'prefix_refresh', 'incremental_persona']

for method in methods:
    with open(f'output/comparison/{method}/baseline_chain_test.jsonl', 'r') as f:
        results = [json.loads(line) for line in f]
    
    # 提取最终画像的三窗口统计
    gains = []
    for user in results:
        last_step = user['steps'][-1]
        if 'three_window_evaluation' in last_step:
            three_win = last_step['three_window_evaluation']
            current_gain = three_win.get('current_window', {}).get('gain', {}).get('ΔQ', 0)
            gains.append(current_gain)
    
    print(f"{method}: {np.mean(gains):.4f} ± {np.std(gains):.4f}")
```

---

## 💰 计算成本对比

### 完整评估（1,801 个用户）

| 方案 | 三窗口评估次数 | 额外预测次数 | 额外时间 | 总时间 |
|------|--------------|-------------|---------|--------|
| **优化前**（每步都评估） | 36,020 | 216,120 | 6-9 小时 | 8-12 小时 |
| **优化后**（只评估最后一步） | 7,204 | 43,224 | 1-2 小时 | 3-5 小时 ⚡ |
| **节省** | **80%** | **80%** | **5-7 小时** | **5-7 小时** |

### 小规模测试（5 个用户）

| 方案 | 额外预测次数 | 额外时间 |
|------|-------------|---------|
| **优化前** | 600 | 5-10 分钟 |
| **优化后** | 120 | 1-2 分钟 ⚡ |
| **节省** | **80%** | **4-8 分钟** |

---

## ✅ 优势总结

### 1. 大幅降低计算成本

✅ 减少 **80%** 的三窗口评估次数
✅ 节省 **5-7 小时**的计算时间
✅ 降低 API 调用成本

### 2. 避免窗口索引问题

✅ 只在最后一步评估，确保有足够的窗口数据
✅ 避免早期步骤缺少历史窗口的问题
✅ 避免后期步骤缺少未来窗口的问题

### 3. 科学价值不减

✅ 最终画像最重要，中间步骤不重要
✅ 仍然可以回答所有关键问题：
  - 是否遗忘历史？
  - 当前窗口表现如何？
  - 泛化能力如何？

### 4. 简化分析

✅ 每个用户只有一个三窗口统计
✅ 更容易提取和分析
✅ 更容易可视化

---

## 🎯 使用建议

### 推荐配置

```bash
python -m comparison.run_baseline_comparison \
  --split test \
  --windowed-root output/windowed \
  --methods static_s0,clasp_online,prefix_refresh,incremental_persona \
  --comparison-root output/comparison \
  --scorer-device cpu \
  --skip-window-split \
  --user-processes 8 \
  --workers 10
```

**预计时间**：3-5 小时（vs 优化前 8-12 小时）⚡

---

## 🎉 总结

### 优化内容

✅ **只在最后一步进行三窗口评估**
- static_s0: 最后一步记录一次
- 其他方法: 最后一次画像更新时记录

### 优化效果

✅ **减少 80% 的计算成本**
- 从 216,120 次预测降到 43,224 次
- 从 8-12 小时降到 3-5 小时

✅ **科学价值不减**
- 最终画像最重要
- 仍然可以回答所有关键问题

✅ **避免窗口索引问题**
- 确保有足够的窗口数据
- 简化边界情况处理

---

**优化者**: Claude (Opus 4.6)
**优化日期**: 2026-05-07
**感谢**: 用户提出了只统计最后一次更新的优化建议！
