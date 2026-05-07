# 统一历史机制修改总结

## 修改日期
2026-05-06

## 修改目标
统一所有基线方法的历史输入机制，消除"历史使用方式"这个混淆变量，确保公平对比**画像更新策略本身**的效果。

---

## 一、修改内容

### 1.1 精简方法列表

**删除的方法**：
- ❌ `s0_sliding_history` - 统一历史后与 static_s0 完全相同
- ❌ `user_full_history` - 不符合统一历史机制的设计

**保留的核心 4 方法**：
- ✅ `static_s0` - 画像不更新（基线）
- ✅ `clasp_online` - 误差驱动更新（主方法）
- ✅ `prefix_refresh` - 全量重算更新
- ✅ `incremental_persona` - 无误差信号更新

### 1.2 统一历史输入机制

**修改前**：
- static_s0, clasp_online, prefix_refresh, incremental_persona：只使用隐式历史（history_actions）
- s0_sliding_history, user_full_history：使用显式历史（profile_suffix）

**修改后**：
- **所有 4 个方法都使用统一的 profile_suffix**：
  ```python
  profile_suffix = (
      f"### Recent behaviors (observed window {wkey})\n"
      + format_behavior_data(hist)
  )
  ```

### 1.3 代码修改位置

**文件 1**: `comparison/window_chain_eval.py`

1. **更新 VALID_METHODS**（line 37-42）：
   ```python
   VALID_METHODS = frozenset({
       "static_s0",
       "prefix_refresh",
       "clasp_online",
       "incremental_persona",
   })
   ```

2. **更新函数文档**（line 90-102）：
   - 删除 s0_sliding_history 和 user_full_history 的说明
   - 添加统一历史机制的说明

3. **统一历史输入**（line 133-163）：
   - 删除针对 s0_sliding_history 和 user_full_history 的特殊处理
   - 为所有方法添加统一的 profile_suffix
   - clasp_online 也使用 profile_suffix（之前是 None）

4. **简化画像更新判断**（line 205-206）：
   ```python
   # 修改前
   if method in ("static_s0", "s0_sliding_history", "user_full_history"):
       continue
   
   # 修改后
   if method == "static_s0":
       continue
   ```

**文件 2**: `comparison/README.md`

- 更新方法列表，只保留 4 个核心方法
- 添加统一历史机制的说明

**文件 3**: `comparison/run_baseline_comparison.py`

- 更新默认方法说明
- 添加统一历史机制的注释

---

## 二、修改后的对比维度

### 2.1 统一的输入

| 组件 | 所有方法 |
|------|---------|
| **隐式历史** | ✅ history_actions（滑动窗口，最近 5 条） |
| **显式历史** | ✅ profile_suffix（当前窗口完整行为） |
| **历史窗口** | ✅ W_t（当前观测窗口） |

### 2.2 唯一的差异：画像更新策略

| 方法 | 画像来源 | 更新机制 | 对比维度 |
|------|---------|---------|---------|
| **static_s0** | S0（固定） | 不更新 | 基线 |
| **clasp_online** | S_t | 误差驱动精炼 | 主方法 |
| **prefix_refresh** | 重算 | 全量重算 | 对比：重算 vs 精炼 |
| **incremental_persona** | S_{t-1}+当前窗 | 无误差信号精炼 | 对比：有无误差信号 |

---

## 三、修改的科学意义

### 3.1 消除混淆变量

**修改前**：
- 性能差异可能来自：画像更新策略 + 历史使用方式
- 无法确定哪个因素更重要

**修改后**：
- 性能差异**只来自**：画像更新策略
- 可以明确证明画像更新的有效性

### 3.2 更公平的对比

**修改前**：
- s0_sliding_history 使用显式历史，可能比 static_s0 表现更好
- 但这不是因为画像策略，而是因为历史输入更丰富

**修改后**：
- 所有方法使用相同的历史输入
- 性能差异完全归因于画像更新策略

### 3.3 更清晰的结论

**修改前**：
- "clasp_online 比 static_s0 好" → 可能是因为历史使用方式不同

**修改后**：
- "clasp_online 比 static_s0 好" → **确定是因为误差驱动的画像更新有效**

---

## 四、向后兼容性

### 4.1 不兼容的变化

1. **方法名称**：
   - ❌ `s0_sliding_history` 不再支持
   - ❌ `user_full_history` 不再支持

2. **输出格式**：
   - 所有方法的输出中，`method` 字段只会是 4 个核心方法之一

### 4.2 迁移指南

**如果之前使用 s0_sliding_history**：
```bash
# 修改前
--methods s0_sliding_history

# 修改后（功能相同）
--methods static_s0
```

**如果之前使用 user_full_history**：
- 该方法已删除，无直接替代
- 如果需要类似功能，可以使用 static_s0（画像不更新）

---

## 五、测试建议

### 5.1 验证修改

```bash
# 快速验证（5 个用户）
python -m comparison.run_baseline_comparison \
  --input-jsonl output/windowed/test/community_0.jsonl \
  --methods static_s0,clasp_online,prefix_refresh,incremental_persona \
  --max-users 5 \
  --comparison-root output/comparison_test
```

### 5.2 完整测试

```bash
# 在测试集上运行所有 4 个方法
python -m comparison.run_baseline_comparison \
  --split test \
  --windowed-root output/windowed \
  --methods static_s0,clasp_online,prefix_refresh,incremental_persona \
  --comparison-root output/comparison \
  --scorer-device cpu \
  --skip-window-split
```

---

## 六、预期结果

### 6.1 如果 clasp_online 有效

应该观察到：
- ✅ clasp_online > static_s0（显著）
- ✅ clasp_online ≥ prefix_refresh
- ✅ clasp_online > incremental_persona

### 6.2 如果误差信号重要

应该观察到：
- ✅ clasp_online > incremental_persona（显著）
- 说明误差信号对画像精炼至关重要

### 6.3 如果精炼优于重算

应该观察到：
- ✅ clasp_online > prefix_refresh
- 说明增量精炼比全量重算更有效

---

## 七、相关文档

- `comparison/UNIFIED_HISTORY_MECHANISM.md` - 统一历史机制的详细分析
- `comparison/ANALYSIS_STATIC_S0_DELETION.md` - 关于 static_s0 的讨论
- `comparison/README.md` - 更新后的使用说明
- `comparison/window_chain_eval.py` - 核心实现代码

---

## 八、总结

✅ **修改完成**：
- 精简为 4 个核心方法
- 统一历史输入机制
- 确保公平对比画像更新策略

✅ **科学价值**：
- 消除混淆变量
- 更清晰的因果关系
- 更可信的实验结论

✅ **下一步**：
- 运行完整评估
- 分析结果
- 撰写论文
