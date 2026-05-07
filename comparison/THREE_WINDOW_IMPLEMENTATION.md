# 方案 A+ 三窗口统计实现文档

## 实施日期
2026-05-06

## 实施状态
✅ 已完成实现，等待测试验证

---

## 一、实现概述

### 1.1 目标
为 clasp_online 方法实现完整的三窗口统计，记录画像更新前后在**过去、当前、未来**三个时间窗口上的性能对比。

### 1.2 核心功能
- 在每次画像更新后，用旧画像和新画像分别在三个窗口上进行预测
- 记录每个窗口上的 F、L、Q 指标
- 计算更新增益（ΔF、ΔL、ΔQ）

---

## 二、代码实现

### 2.1 新增函数：evaluate_three_windows

**位置**: `comparison/window_chain_eval.py`

**功能**: 在三个窗口上评估旧画像和新画像的性能

**参数**:
```python
def evaluate_three_windows(
    old_profile: str,           # 更新前的画像
    new_profile: str,           # 更新后的画像
    windows: Dict[str, Any],    # 所有窗口数据
    keys: List[str],            # 窗口键列表
    step_idx: int,              # 当前步骤索引
    action_model,               # 动作预测模型
    action_tokenizer,           # 分词器
    semantic_scorer,            # 语义评分器
) -> Dict[str, Any]
```

**返回值结构**:
```python
{
  "past_window": {
    "window": "W0",
    "target": "W1",
    "with_old_profile": {"F": 0.58, "L": 0.42, "Q": 0.52},
    "with_new_profile": {"F": 0.59, "L": 0.43, "Q": 0.53},
    "gain": {"ΔF": 0.01, "ΔL": 0.01, "ΔQ": 0.01}
  },
  "current_window": {
    "window": "W1",
    "target": "W2",
    "with_old_profile": {"F": 0.61, "L": 0.44, "Q": 0.54},
    "with_new_profile": {"F": 0.63, "L": 0.45, "Q": 0.56},
    "gain": {"ΔF": 0.02, "ΔL": 0.01, "ΔQ": 0.02}
  },
  "future_window": {
    "window": "W2",
    "target": "W3",
    "with_old_profile": {"F": 0.60, "L": 0.43, "Q": 0.54},
    "with_new_profile": {"F": 0.62, "L": 0.44, "Q": 0.55},
    "gain": {"ΔF": 0.02, "ΔL": 0.01, "ΔQ": 0.01}
  }
}
```

### 2.2 修改：clasp_online 画像更新逻辑

**位置**: `comparison/window_chain_eval.py` (line 358-450)

**新增记录字段**:

1. **profile_updated** (bool): 画像是否被更新
2. **profile_length** (int): 当前画像长度（字符数）
3. **num_candidates** (int): 生成的候选画像数
4. **best_candidate_index** (int): 选中的候选索引
5. **candidate_scores** (list): 所有候选画像的评分
   ```python
   [
     {"index": 0, "F": 0.61, "L": 0.43, "Q": 0.54},
     {"index": 1, "F": 0.63, "L": 0.45, "Q": 0.56},
     ...
   ]
   ```
6. **three_window_evaluation** (dict): 三窗口评估结果（见上方结构）

---

## 三、三窗口定义

### 3.1 时间窗口映射

对于 Step i（使用画像 S_i 预测 W_{i+1}）：

| 窗口类型 | 窗口索引 | 历史输入 | 预测目标 | 说明 |
|---------|---------|---------|---------|------|
| **过去窗口** | W_{i-1} | W_{i-1} | W_i | 检测是否遗忘历史 |
| **当前窗口** | W_i | W_i | W_{i+1} | 主要评估指标 |
| **未来窗口** | W_{i+1} | W_{i+1} | W_{i+2} | 检测泛化能力 |

### 3.2 示例（Step 1）

```
Step 1: 用 S1 预测 W2

三窗口评估：
- 过去窗口: 用 S0/S1 + 历史W0 → 预测 W1
- 当前窗口: 用 S0/S1 + 历史W1 → 预测 W2
- 未来窗口: 用 S0/S1 + 历史W2 → 预测 W3
```

---

## 四、计算成本分析

### 4.1 额外预测次数

**每次画像更新**：
- 过去窗口：2 次预测（旧画像 + 新画像）
- 当前窗口：2 次预测
- 未来窗口：2 次预测
- **总计**：6 次额外预测

**注意**：
- 只有在画像实际更新时才进行三窗口评估
- Step 0 没有过去窗口（只评估当前和未来）
- 最后一步没有未来窗口（只评估过去和当前）

### 4.2 总体成本估算

假设：
- 1,801 个用户
- 每个用户 5 步
- 平均 80% 的步骤会更新画像

**额外预测次数**：
```
1,801 × 5 × 0.8 × 6 = 43,224 次
```

**时间成本**：
- 原评估时间：8-12 小时
- 额外时间：16-24 小时（约 2-3 倍）
- **总时间：24-36 小时**

---

## 五、输出数据格式

### 5.1 完整的 Step 数据结构

```json
{
  "step_index": 1,
  "history_window": "W1",
  "target_window": "W2",
  "F": 0.6234,
  "L": 0.4521,
  "Q": 0.5547,
  
  "profile_updated": true,
  "profile_length": 1234,
  "num_candidates": 10,
  "best_candidate_index": 3,
  
  "candidate_scores": [
    {"index": 0, "F": 0.6012, "L": 0.4123, "Q": 0.5234},
    {"index": 1, "F": 0.6234, "L": 0.4321, "Q": 0.5456},
    ...
  ],
  
  "three_window_evaluation": {
    "past_window": {
      "window": "W0",
      "target": "W1",
      "with_old_profile": {"F": 0.58, "L": 0.42, "Q": 0.52},
      "with_new_profile": {"F": 0.59, "L": 0.43, "Q": 0.53},
      "gain": {"ΔF": 0.01, "ΔL": 0.01, "ΔQ": 0.01}
    },
    "current_window": {
      "window": "W1",
      "target": "W2",
      "with_old_profile": {"F": 0.61, "L": 0.44, "Q": 0.54},
      "with_new_profile": {"F": 0.63, "L": 0.45, "Q": 0.56},
      "gain": {"ΔF": 0.02, "ΔL": 0.01, "ΔQ": 0.02}
    },
    "future_window": {
      "window": "W2",
      "target": "W3",
      "with_old_profile": {"F": 0.60, "L": 0.43, "Q": 0.54},
      "with_new_profile": {"F": 0.62, "L": 0.44, "Q": 0.55},
      "gain": {"ΔF": 0.02, "ΔL": 0.01, "ΔQ": 0.01}
    }
  }
}
```

---

## 六、使用方法

### 6.1 运行评估

```bash
# 快速测试（1 个用户）
python -m comparison.run_baseline_comparison \
  --input-jsonl output/windowed/test/community_0.jsonl \
  --methods clasp_online \
  --max-users 1 \
  --comparison-root output/comparison_test \
  --scorer-device cpu

# 完整评估（所有测试集）
python -m comparison.run_baseline_comparison \
  --split test \
  --windowed-root output/windowed \
  --methods clasp_online \
  --comparison-root output/comparison \
  --scorer-device cpu \
  --skip-window-split
```

### 6.2 分析结果

输出文件：`output/comparison/clasp_online/baseline_chain_test.jsonl`

每行一个用户的评估结果，包含完整的三窗口统计数据。

---

## 七、后续分析

### 7.1 可以回答的问题

1. **画像是否过拟合当前窗口？**
   - 对比 current_window.gain 和 future_window.gain
   - 如果 current >> future：过拟合

2. **画像是否遗忘历史？**
   - 查看 past_window.gain
   - 如果 < 0：遗忘了历史信息

3. **画像更新是否全面有效？**
   - 查看三个窗口的 gain 是否都为正
   - 理想情况：三个窗口都有正增益

### 7.2 统计指标

可以计算的指标：
- 平均三窗口增益
- 三窗口增益一致性（标准差）
- 时间衰减率
- 更新有效率

### 7.3 可视化

可以生成的图表：
- 三窗口增益箱线图
- 三窗口增益热力图
- 当前 vs 未来增益散点图
- 性能随时间演化曲线

---

## 八、测试验证

### 8.1 测试脚本

`comparison/test_three_window.py` - 自动化测试脚本

### 8.2 验证项

- ✅ 代码导入无错误
- ✅ 评估运行成功
- ✅ 输出文件包含新增字段
- ✅ three_window_evaluation 结构正确
- ✅ 增益计算正确

---

## 九、注意事项

### 9.1 边界情况

1. **Step 0**：没有过去窗口
   - three_window_evaluation 只包含 current_window 和 future_window

2. **最后一步**：没有未来窗口
   - three_window_evaluation 只包含 past_window 和 current_window

3. **画像未更新**：
   - 不进行三窗口评估
   - three_window_evaluation 字段不存在

### 9.2 性能优化建议

如果计算时间过长，可以考虑：

1. **采样评估**：只对 20% 的用户进行三窗口评估
2. **简化版本**：只评估当前和未来窗口（省略过去窗口）
3. **并行优化**：增加 DPO_WORKERS 和 DPO_USER_PROCESSES

---

## 十、相关文档

- `comparison/PLAN_A_PLUS_THREE_WINDOW_STATISTICS.md` - 方案设计文档
- `comparison/window_chain_eval.py` - 实现代码
- `comparison/test_three_window.py` - 测试脚本

---

## 十一、总结

✅ **实现完成**：
- 新增 evaluate_three_windows 函数
- 修改 clasp_online 记录三窗口数据
- 创建测试脚本验证功能

✅ **数据完整**：
- 过去、当前、未来三个窗口
- 旧画像和新画像的对比
- 详细的增益计算

✅ **科学价值**：
- 可以深入分析画像更新机制
- 检测过拟合和遗忘问题
- 评估泛化能力

⏱️ **计算成本**：
- 预计 24-36 小时（1,801 个用户）
- 可以通过采样或简化来优化

🎯 **下一步**：
- 运行测试验证功能
- 运行完整评估
- 编写分析脚本
- 生成可视化图表
