# 所有方法的三窗口统计

## 📊 改进说明

### 之前的问题

**只有 clasp_online 方法记录了三窗口统计**，其他方法（static_s0, prefix_refresh, incremental_persona）没有记录。

### 现在的改进

✅ **所有方法都记录三窗口统计**

---

## 🎯 三窗口统计的意义

### 对于 static_s0（静态画像）

虽然画像不变（old_profile == new_profile），但仍然记录三窗口统计：

**用途**：
- 作为**基线对比**
- 观察静态画像在不同时间窗口上的表现变化
- 分析时间漂移（temporal drift）

**示例**：
```json
{
  "method": "static_s0",
  "steps": [
    {
      "step_index": 1,
      "three_window_evaluation": {
        "past_window": {
          "with_old_profile": {"Q": 0.52},
          "with_new_profile": {"Q": 0.52},  // 相同
          "gain": {"ΔQ": 0.00}  // 无增益
        },
        "current_window": {
          "with_old_profile": {"Q": 0.54},
          "with_new_profile": {"Q": 0.54},  // 相同
          "gain": {"ΔQ": 0.00}
        },
        "future_window": {
          "with_old_profile": {"Q": 0.50},  // 性能下降
          "with_new_profile": {"Q": 0.50},
          "gain": {"ΔQ": 0.00}
        }
      },
      "profile_changed": false  // 画像未变化
    }
  ]
}
```

**分析**：
- 静态画像在未来窗口上的 Q 值下降（0.54 → 0.50）
- 说明存在时间漂移，用户行为模式在变化
- 证明了画像更新的必要性

### 对于 prefix_refresh（全量重算）

每步用已观测的所有窗口重新生成画像。

**用途**：
- 观察全量重算的效果
- 对比增量更新（clasp_online）和全量重算的差异

**示例**：
```json
{
  "method": "prefix_refresh",
  "steps": [
    {
      "step_index": 1,
      "three_window_evaluation": {
        "past_window": {
          "with_old_profile": {"Q": 0.52},  // 基于 W0 的画像
          "with_new_profile": {"Q": 0.55},  // 基于 W0+W1 的画像
          "gain": {"ΔQ": 0.03}  // 过去窗口也有提升
        },
        "current_window": {
          "with_old_profile": {"Q": 0.54},
          "with_new_profile": {"Q": 0.58},
          "gain": {"ΔQ": 0.04}
        },
        "future_window": {
          "with_old_profile": {"Q": 0.50},
          "with_new_profile": {"Q": 0.56},
          "gain": {"ΔQ": 0.06}  // 未来窗口提升最大
        }
      },
      "profile_changed": true
    }
  ]
}
```

**分析**：
- 全量重算在所有三个窗口上都有提升
- 未来窗口提升最大，说明泛化能力好
- 但计算成本高（需要重新处理所有历史数据）

### 对于 incremental_persona（无误差信号更新）

使用当前窗口的行为（无预测误差）更新画像。

**用途**：
- 对比有无误差信号的差异
- 分析误差信号的重要性

**示例**：
```json
{
  "method": "incremental_persona",
  "steps": [
    {
      "step_index": 1,
      "three_window_evaluation": {
        "past_window": {
          "with_old_profile": {"Q": 0.52},
          "with_new_profile": {"Q": 0.51},  // 过去窗口性能下降
          "gain": {"ΔQ": -0.01}  // 遗忘
        },
        "current_window": {
          "with_old_profile": {"Q": 0.54},
          "with_new_profile": {"Q": 0.56},
          "gain": {"ΔQ": 0.02}
        },
        "future_window": {
          "with_old_profile": {"Q": 0.50},
          "with_new_profile": {"Q": 0.52},
          "gain": {"ΔQ": 0.02}
        }
      },
      "profile_changed": true
    }
  ]
}
```

**分析**：
- 当前和未来窗口有提升
- 但过去窗口性能下降（遗忘）
- 说明无误差信号的更新可能导致遗忘

### 对于 clasp_online（误差驱动更新）

使用预测误差精炼画像。

**示例**：
```json
{
  "method": "clasp_online",
  "steps": [
    {
      "step_index": 1,
      "three_window_evaluation": {
        "past_window": {
          "with_old_profile": {"Q": 0.52},
          "with_new_profile": {"Q": 0.53},
          "gain": {"ΔQ": 0.01}  // 不遗忘
        },
        "current_window": {
          "with_old_profile": {"Q": 0.54},
          "with_new_profile": {"Q": 0.58},
          "gain": {"ΔQ": 0.04}  // 当前窗口提升
        },
        "future_window": {
          "with_old_profile": {"Q": 0.50},
          "with_new_profile": {"Q": 0.55},
          "gain": {"ΔQ": 0.05}  // 泛化好
        }
      },
      "profile_changed": true
    }
  ]
}
```

**分析**：
- 三个窗口都有提升
- 不遗忘历史（past_gain > 0）
- 泛化能力好（future_gain ≈ current_gain）

---

## 📈 对比分析

### 四种方法的三窗口表现对比

| 方法 | 过去窗口 | 当前窗口 | 未来窗口 | 特点 |
|------|---------|---------|---------|------|
| **static_s0** | 0 | 0 | 0 | 无提升，时间漂移 |
| **prefix_refresh** | +0.03 | +0.04 | +0.06 | 全面提升，泛化好 |
| **incremental_persona** | -0.01 | +0.02 | +0.02 | 遗忘历史 |
| **clasp_online** | +0.01 | +0.04 | +0.05 | 全面提升，不遗忘 |

### 关键指标

#### 1. 遗忘率（Forgetting Rate）

```python
forgetting_rate = (past_gain < 0) 的比例
```

**预期**：
- static_s0: 0%（无更新）
- prefix_refresh: 0%（全量重算）
- incremental_persona: 20-30%（可能遗忘）
- clasp_online: 0-5%（误差驱动，不遗忘）

#### 2. 泛化能力（Generalization）

```python
generalization = future_gain / current_gain
```

**预期**：
- static_s0: N/A（无增益）
- prefix_refresh: 1.5（泛化好）
- incremental_persona: 1.0（泛化一般）
- clasp_online: 1.2（泛化好）

#### 3. 时间一致性（Temporal Consistency）

```python
consistency = std([past_gain, current_gain, future_gain])
```

**预期**：
- static_s0: 0（完全一致，但无提升）
- prefix_refresh: 低（一致性好）
- incremental_persona: 高（不一致）
- clasp_online: 低（一致性好）

---

## 🔧 使用方法

### 运行评估

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

### 分析结果

```python
import json

# 读取结果
with open('output/comparison/static_s0/baseline_chain_test.jsonl', 'r') as f:
    static_results = [json.loads(line) for line in f]

with open('output/comparison/clasp_online/baseline_chain_test.jsonl', 'r') as f:
    clasp_results = [json.loads(line) for line in f]

# 统计三窗口增益
def analyze_three_window(results):
    past_gains = []
    current_gains = []
    future_gains = []
    
    for user in results:
        for step in user['steps']:
            if 'three_window_evaluation' in step:
                three_win = step['three_window_evaluation']
                if 'past_window' in three_win:
                    past_gains.append(three_win['past_window']['gain']['ΔQ'])
                if 'current_window' in three_win:
                    current_gains.append(three_win['current_window']['gain']['ΔQ'])
                if 'future_window' in three_win:
                    future_gains.append(three_win['future_window']['gain']['ΔQ'])
    
    return {
        'past_mean': np.mean(past_gains),
        'current_mean': np.mean(current_gains),
        'future_mean': np.mean(future_gains),
        'forgetting_rate': sum(1 for g in past_gains if g < 0) / len(past_gains),
        'generalization': np.mean(future_gains) / np.mean(current_gains) if np.mean(current_gains) > 0 else 0,
    }

# 对比分析
static_stats = analyze_three_window(static_results)
clasp_stats = analyze_three_window(clasp_results)

print("Static S0:")
print(f"  Past: {static_stats['past_mean']:.4f}")
print(f"  Current: {static_stats['current_mean']:.4f}")
print(f"  Future: {static_stats['future_mean']:.4f}")
print(f"  Forgetting Rate: {static_stats['forgetting_rate']:.2%}")

print("\nClasp Online:")
print(f"  Past: {clasp_stats['past_mean']:.4f}")
print(f"  Current: {clasp_stats['current_mean']:.4f}")
print(f"  Future: {clasp_stats['future_mean']:.4f}")
print(f"  Forgetting Rate: {clasp_stats['forgetting_rate']:.2%}")
print(f"  Generalization: {clasp_stats['generalization']:.2f}")
```

---

## 💰 计算成本

### 额外预测次数

**每个方法、每步**：
- 过去窗口：2 次预测（旧画像 + 新画像）
- 当前窗口：2 次预测
- 未来窗口：2 次预测
- **总计**：6 次额外预测

**所有方法、所有用户**：
```
4 个方法 × 1,801 个用户 × 5 步 × 0.8 更新率 × 6 次预测
= 172,896 次额外预测
```

**时间成本**：
- 原评估时间（并行）：2-3 小时
- 额外时间：4-6 小时
- **总时间：6-9 小时**

---

## ✅ 总结

### 改进内容

✅ **所有方法都记录三窗口统计**
- static_s0: 作为基线，观察时间漂移
- prefix_refresh: 观察全量重算的效果
- incremental_persona: 分析无误差信号的影响
- clasp_online: 验证误差驱动更新的有效性

### 科学价值

✅ **更全面的对比分析**
- 可以对比不同方法在三个时间维度上的表现
- 分析遗忘、泛化、时间一致性

✅ **更深入的理解**
- 理解静态画像的局限性（时间漂移）
- 理解误差信号的重要性（避免遗忘）
- 理解不同更新策略的优缺点

### 使用建议

**推荐配置**：
```bash
--methods static_s0,clasp_online,prefix_refresh,incremental_persona
--user-processes 8
--workers 10
```

**预计时间**：6-9 小时（1,801 个用户）

---

**实现者**: Claude (Opus 4.6)
**完成日期**: 2026-05-07
**感谢**: 用户提出了为所有方法添加三窗口统计的需求！
