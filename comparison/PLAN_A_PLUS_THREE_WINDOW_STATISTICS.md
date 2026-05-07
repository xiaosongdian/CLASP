# 方案 A+ 增强版：完整的三窗口画像更新统计

## 核心改进

原方案 A 只记录了"下一步"的对比，**没有完整覆盖过去、当前、未来三个窗口**。

新方案 A+ 将**完整记录画像在三个时间维度上的表现**。

---

## 一、三窗口评估协议

### 1.1 Clasp_online 的评估流程回顾

以 Step 1 为例（S1 → W2）：

```
时间线：
  W0 (过去) → W1 (当前) → W2 (未来)
  
Step 0: 用 S0 预测 W1 → 得到误差 → 精炼得到候选画像
        在 W2 上评估候选 → 选出 S1

Step 1: 用 S1 预测 W2 ← 我们在这一步
        需要评估 S1 在三个窗口上的表现
```

### 1.2 三窗口定义

对于 Step i（使用画像 S_i 预测 W_{i+1}）：

| 窗口类型 | 窗口索引 | 作用 | 说明 |
|---------|---------|------|------|
| **过去窗口** | W_i | 历史上下文 | 用作预测的历史输入 |
| **当前窗口** | W_{i+1} | 预测目标 | 当前步骤要预测的窗口 |
| **未来窗口** | W_{i+2} | 验证泛化 | 用于验证画像的未来表现 |

### 1.3 为什么需要三窗口统计？

1. **过去窗口（W_i）**: 
   - 检验画像是否"记住"了历史
   - 如果在过去窗口上表现差，说明画像丢失了历史信息

2. **当前窗口（W_{i+1}）**:
   - 检验画像在当前任务上的表现
   - 这是主要的评估指标

3. **未来窗口（W_{i+2}）**:
   - 检验画像的泛化能力
   - 避免过拟合当前窗口

---

## 二、完整的数据记录结构

### 2.1 每个 Step 的完整记录

```python
{
  "step_index": 1,
  "history_window": "W1",
  "target_window": "W2",
  
  # === 当前步骤的预测结果（主要指标）===
  "current_prediction": {
    "F": 0.6234,
    "L": 0.4521,
    "Q": 0.5547
  },
  
  # === 画像更新信息 ===
  "profile_info": {
    "profile_id": "S1",
    "profile_length": 1234,
    "profile_updated": true,
    "num_candidates": 10,
    "best_candidate_index": 3
  },
  
  # === 候选画像评估（在未来窗口 W2 上）===
  "candidate_evaluation": {
    "evaluation_window": "W2",
    "old_profile_score": {
      "profile_id": "S0",
      "F": 0.6100,
      "L": 0.4400,
      "Q": 0.5420
    },
    "candidates": [
      {"index": 0, "Q": 0.5234, "F": 0.6012, "L": 0.4123},
      {"index": 1, "Q": 0.5456, "F": 0.6234, "L": 0.4321},
      ...
    ],
    "selected_profile_score": {
      "profile_id": "S1",
      "index": 3,
      "F": 0.6234,
      "L": 0.4521,
      "Q": 0.5547
    }
  },
  
  # === 新增：三窗口完整评估 ===
  "three_window_evaluation": {
    // 过去窗口：用新画像 S1 在过去窗口 W0 上的表现
    "past_window": {
      "window": "W0",
      "with_old_profile": {
        "profile_id": "S0",
        "F": 0.5800,
        "L": 0.4200,
        "Q": 0.5200
      },
      "with_new_profile": {
        "profile_id": "S1",
        "F": 0.5850,
        "L": 0.4250,
        "Q": 0.5250
      },
      "gain": {
        "ΔF": 0.0050,
        "ΔL": 0.0050,
        "ΔQ": 0.0050
      }
    },
    
    // 当前窗口：用新画像 S1 在当前窗口 W1 上的表现
    "current_window": {
      "window": "W1",
      "with_old_profile": {
        "profile_id": "S0",
        "F": 0.6100,
        "L": 0.4400,
        "Q": 0.5420
      },
      "with_new_profile": {
        "profile_id": "S1",
        "F": 0.6234,
        "L": 0.4521,
        "Q": 0.5547
      },
      "gain": {
        "ΔF": 0.0134,
        "ΔL": 0.0121,
        "ΔQ": 0.0127
      }
    },
    
    // 未来窗口：用新画像 S1 在未来窗口 W2 上的表现
    "future_window": {
      "window": "W2",
      "with_old_profile": {
        "profile_id": "S0",
        "F": 0.6050,
        "L": 0.4350,
        "Q": 0.5380
      },
      "with_new_profile": {
        "profile_id": "S1",
        "F": 0.6180,
        "L": 0.4480,
        "Q": 0.5508
      },
      "gain": {
        "ΔF": 0.0130,
        "ΔL": 0.0130,
        "ΔQ": 0.0128
      }
    }
  }
}
```

### 2.2 实现说明

**关键点**：在每个 step 结束后，需要用**旧画像**和**新画像**分别在三个窗口上进行预测：

```python
# Step 1 结束后（已选出 S1）
old_profile = S0
new_profile = S1

# 在过去窗口 W0 上评估
past_old = evaluate_profile_on_window(old_profile, history=[], target=W0)
past_new = evaluate_profile_on_window(new_profile, history=[], target=W0)

# 在当前窗口 W1 上评估
current_old = evaluate_profile_on_window(old_profile, history=W0, target=W1)
current_new = evaluate_profile_on_window(new_profile, history=W0, target=W1)

# 在未来窗口 W2 上评估
future_old = evaluate_profile_on_window(old_profile, history=W1, target=W2)
future_new = evaluate_profile_on_window(new_profile, history=W1, target=W2)
```

**计算成本**：
- 每个 step 需要额外 6 次预测（3 个窗口 × 2 个画像）
- 对于 1,801 个用户 × 5 步 = 9,005 步
- 总共需要 54,030 次额外预测
- **预计增加 2-3 倍的运行时间**

---

## 三、统计指标设计

### 3.1 三窗口增益分析

#### 指标 1: 分窗口平均增益
```python
{
  "past_window_gain": {
    "mean_ΔQ": 0.0050,
    "mean_ΔF": 0.0045,
    "mean_ΔL": 0.0055
  },
  "current_window_gain": {
    "mean_ΔQ": 0.0127,
    "mean_ΔF": 0.0134,
    "mean_ΔL": 0.0121
  },
  "future_window_gain": {
    "mean_ΔQ": 0.0128,
    "mean_ΔF": 0.0130,
    "mean_ΔL": 0.0130
  }
}
```

**解读**：
- 如果 `current_window_gain` 最大：说明画像针对当前任务优化
- 如果 `future_window_gain` 也很大：说明画像泛化能力强
- 如果 `past_window_gain` 很小或负：说明画像可能"遗忘"了历史

#### 指标 2: 三窗口一致性
```python
consistency = std([past_gain, current_gain, future_gain])
```
- 低一致性（高 std）：画像在不同时间窗口上表现不稳定
- 高一致性（低 std）：画像在所有窗口上均衡提升

#### 指标 3: 时间衰减率
```python
decay_rate = (past_gain - future_gain) / past_gain
```
- 正值：画像在未来窗口上表现下降（过拟合历史）
- 负值：画像在未来窗口上表现更好（泛化能力强）

### 3.2 更新有效性判断

**判断标准**：
```python
def is_update_effective(three_window_eval):
    # 标准 1: 当前窗口必须提升
    current_improved = three_window_eval["current_window"]["gain"]["ΔQ"] > 0
    
    # 标准 2: 未来窗口也应该提升（泛化）
    future_improved = three_window_eval["future_window"]["gain"]["ΔQ"] > 0
    
    # 标准 3: 过去窗口不应该显著下降（不遗忘）
    past_not_degraded = three_window_eval["past_window"]["gain"]["ΔQ"] > -0.01
    
    return current_improved and future_improved and past_not_degraded
```

---

## 四、可视化方案

### 4.1 三窗口增益对比图

**图 1: 三窗口增益箱线图**
```
X 轴: 窗口类型（过去、当前、未来）
Y 轴: ΔQ
每个窗口一个箱线图，显示所有更新的增益分布
```

**图 2: 三窗口增益热力图**
```
行: Step Index (0, 1, 2, 3, 4)
列: 窗口类型（过去、当前、未来）
颜色: 平均 ΔQ
```

### 4.2 时间演化图

**图 3: 画像性能随时间演化**
```
X 轴: Step Index
Y 轴: Q 值
线条:
  - 过去窗口（蓝色）
  - 当前窗口（绿色）
  - 未来窗口（红色）
每条线显示该窗口上的平均 Q 值
```

### 4.3 增益散点图

**图 4: 当前 vs 未来增益**
```
X 轴: 当前窗口增益（ΔQ_current）
Y 轴: 未来窗口增益（ΔQ_future）
散点: 每次画像更新
对角线: y=x（一致性基线）
```

**解读**：
- 点在对角线上：当前和未来增益一致
- 点在对角线上方：未来增益更大（泛化好）
- 点在对角线下方：当前增益更大（可能过拟合）

---

## 五、实施步骤

### Step 1: 修改评估代码

**文件**: `comparison/window_chain_eval.py`

**修改位置**: `clasp_online` 分支（line 229-271）

**伪代码**：
```python
if method == "clasp_online":
    # 现有代码：生成候选画像
    candidates = generate_candidate_profiles(...)
    
    # 现有代码：选择最佳候选
    best_profile = select_best_candidate(...)
    
    # === 新增：三窗口评估 ===
    if step_idx > 0:  # 有过去窗口
        three_window_eval = evaluate_three_windows(
            old_profile=profile,
            new_profile=best_profile,
            past_window=windows[keys[step_idx - 1]],
            current_window=windows[keys[step_idx]],
            future_window=windows[keys[step_idx + 1]],
            action_model=action_model,
            action_tokenizer=action_tokenizer,
            semantic_scorer=semantic_scorer
        )
        steps_out[-1]["three_window_evaluation"] = three_window_eval
    
    profile = best_profile
```

### Step 2: 实现三窗口评估函数

**新函数**: `evaluate_three_windows()`

```python
def evaluate_three_windows(
    old_profile: str,
    new_profile: str,
    past_window: List[Dict],
    current_window: List[Dict],
    future_window: List[Dict],
    action_model,
    action_tokenizer,
    semantic_scorer
) -> Dict:
    """
    在三个窗口上评估旧画像和新画像的性能
    """
    result = {}
    
    # 过去窗口（history=空，target=past）
    past_old = evaluate_profile_on_window(
        old_profile, [], past_window, 
        action_model, action_tokenizer, semantic_scorer
    )
    past_new = evaluate_profile_on_window(
        new_profile, [], past_window,
        action_model, action_tokenizer, semantic_scorer
    )
    result["past_window"] = {
        "window": "W_past",
        "with_old_profile": {"F": past_old[0], "L": past_old[1], "Q": past_old[2]},
        "with_new_profile": {"F": past_new[0], "L": past_new[1], "Q": past_new[2]},
        "gain": {
            "ΔF": past_new[0] - past_old[0],
            "ΔL": past_new[1] - past_old[1],
            "ΔQ": past_new[2] - past_old[2]
        }
    }
    
    # 当前窗口（history=past，target=current）
    current_old = evaluate_profile_on_window(
        old_profile, past_window, current_window,
        action_model, action_tokenizer, semantic_scorer
    )
    current_new = evaluate_profile_on_window(
        new_profile, past_window, current_window,
        action_model, action_tokenizer, semantic_scorer
    )
    result["current_window"] = {
        "window": "W_current",
        "with_old_profile": {"F": current_old[0], "L": current_old[1], "Q": current_old[2]},
        "with_new_profile": {"F": current_new[0], "L": current_new[1], "Q": current_new[2]},
        "gain": {
            "ΔF": current_new[0] - current_old[0],
            "ΔL": current_new[1] - current_old[1],
            "ΔQ": current_new[2] - current_old[2]
        }
    }
    
    # 未来窗口（history=current，target=future）
    future_old = evaluate_profile_on_window(
        old_profile, current_window, future_window,
        action_model, action_tokenizer, semantic_scorer
    )
    future_new = evaluate_profile_on_window(
        new_profile, current_window, future_window,
        action_model, action_tokenizer, semantic_scorer
    )
    result["future_window"] = {
        "window": "W_future",
        "with_old_profile": {"F": future_old[0], "L": future_old[1], "Q": future_old[2]},
        "with_new_profile": {"F": future_new[0], "L": future_new[1], "Q": future_new[2]},
        "gain": {
            "ΔF": future_new[0] - future_old[0],
            "ΔL": future_new[1] - future_old[1],
            "ΔQ": future_new[2] - future_old[2]
        }
    }
    
    return result
```

### Step 3: 编写分析脚本

**新文件**: `comparison/analyze_three_window_updates.py`

功能：
- 读取评估结果 JSONL
- 提取三窗口评估数据
- 计算统计指标
- 生成可视化图表

---

## 六、成本与收益分析

### 6.1 计算成本

**额外预测次数**：
- 每个 step：6 次预测（3 窗口 × 2 画像）
- 总 steps：1,801 用户 × 5 步 = 9,005 步
- 总额外预测：54,030 次

**时间成本**：
- 原评估时间：8-12 小时
- 额外时间：16-24 小时（2-3 倍）
- **总时间：24-36 小时**

### 6.2 收益

**数据完整性**：
- ✅ 完整的三窗口对比
- ✅ 画像泛化能力评估
- ✅ 历史遗忘检测
- ✅ 过拟合检测

**论文价值**：
- ✅ 可以发表高质量论文
- ✅ 深入分析画像更新机制
- ✅ 证明方法的全面有效性

---

## 七、简化版本（如果时间紧迫）

### 简化方案 1: 只评估当前和未来窗口

**省略过去窗口**，只记录：
- 当前窗口（主要指标）
- 未来窗口（泛化能力）

**节省**：1/3 的计算时间

### 简化方案 2: 采样评估

只对 **20% 的用户**进行三窗口评估：
- 随机采样 360 个用户
- 其他用户只记录当前窗口

**节省**：80% 的额外计算时间

---

## 八、推荐决策

### 如果你的目标是：

**1. 发表顶会论文** → 采用完整方案 A+
- 数据完整，分析深入
- 可以回答审稿人的所有问题

**2. 快速验证方法** → 采用简化方案 2（采样）
- 保留核心分析能力
- 大幅减少计算时间

**3. 平衡方案** → 采用简化方案 1（当前+未来）
- 保留最重要的两个窗口
- 节省 1/3 时间

---

## 九、总结

**方案 A+ 的核心价值**：

✅ **完整性**: 覆盖过去、当前、未来三个时间维度
✅ **深度**: 可以分析泛化能力、遗忘问题、过拟合
✅ **可信度**: 数据充分，结论可靠
❌ **成本**: 需要 24-36 小时运行时间

**你现在需要决定**：
1. 采用完整方案 A+（24-36 小时）
2. 采用简化方案 1：当前+未来（16-24 小时）
3. 采用简化方案 2：采样评估（10-15 小时）

我可以立即帮你实现任何一个方案！

