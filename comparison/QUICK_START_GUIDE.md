# Clasp 三窗口统计评估 - 快速开始指南

## 🎯 目标

在测试集上运行 clasp_online 方法，记录完整的三窗口统计数据，证明画像更新的有效性。

---

## 📋 前置条件

### 1. 数据准备

✅ 已完成：测试数据已窗口化
- 位置: `output/windowed/test/`
- 包含社区: 0, 1, 3, 4, 6, 7
- 总用户数: 1,801

### 2. 模型服务

需要启动两个 vLLM 服务：

#### 画像生成模型（端口 8000）
```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/LLM_models/Meta-Llama-3-8B-Instruct \
  --served-model-name Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45
```

#### 动作预测模型（端口 8001）
```bash
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft-289 \
  --served-model-name Meta-Llama-3-8B-Instruct-bluesky-sft-289 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45
```

#### 验证服务
```bash
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models
```

---

## 🚀 运行评估

### 阶段 1: 快速验证（10-20 分钟）

测试 1 个用户，验证功能正常：

```bash
python -m comparison.run_baseline_comparison \
  --input-jsonl output/windowed/test/community_0.jsonl \
  --methods clasp_online \
  --max-users 1 \
  --comparison-root output/comparison_test \
  --scorer-device cpu \
  --skip-window-split
```

**检查输出**：
```bash
# 查看结果文件
cat output/comparison_test/clasp_online/baseline_chain_community_0.jsonl | python -m json.tool | head -100

# 检查是否包含三窗口数据
cat output/comparison_test/clasp_online/baseline_chain_community_0.jsonl | grep -o "three_window_evaluation"
```

### 阶段 2: 小规模测试（1-2 小时）

测试 50 个用户：

```bash
python -m comparison.run_baseline_comparison \
  --input-jsonl output/windowed/test/community_0.jsonl \
  --methods clasp_online \
  --max-users 50 \
  --comparison-root output/comparison_pilot \
  --scorer-device cpu \
  --skip-window-split
```

### 阶段 3: 完整评估（24-36 小时）

评估所有 1,801 个用户：

```bash
python -m comparison.run_baseline_comparison \
  --split test \
  --windowed-root output/windowed \
  --methods clasp_online \
  --comparison-root output/comparison \
  --scorer-device cpu \
  --skip-window-split
```

**输出文件**：
- `output/comparison/clasp_online/baseline_chain_test.jsonl`

---

## 📊 输出数据结构

每行一个用户，包含：

```json
{
  "user_id": 539166,
  "community_id": 0,
  "method": "clasp_online",
  "mean_Q": 0.5623,
  "mean_F": 0.6145,
  "steps": [
    {
      "step_index": 0,
      "F": 0.6234,
      "L": 0.4521,
      "Q": 0.5547,
      
      "profile_updated": true,
      "profile_length": 1234,
      "num_candidates": 10,
      "best_candidate_index": 3,
      
      "candidate_scores": [
        {"index": 0, "F": 0.60, "L": 0.41, "Q": 0.52},
        {"index": 1, "F": 0.62, "L": 0.43, "Q": 0.55}
      ],
      
      "three_window_evaluation": {
        "past_window": {
          "window": "W0",
          "target": "W1",
          "with_old_profile": {"F": 0.58, "L": 0.42, "Q": 0.52},
          "with_new_profile": {"F": 0.59, "L": 0.43, "Q": 0.53},
          "gain": {"ΔF": 0.01, "ΔL": 0.01, "ΔQ": 0.01}
        },
        "current_window": { ... },
        "future_window": { ... }
      }
    }
  ]
}
```

---

## 📈 数据分析

### 基本统计

```python
import json

# 读取结果
results = []
with open('output/comparison/clasp_online/baseline_chain_test.jsonl', 'r') as f:
    for line in f:
        results.append(json.loads(line))

# 统计画像更新率
total_steps = 0
updated_steps = 0
for user in results:
    for step in user['steps']:
        total_steps += 1
        if step.get('profile_updated', False):
            updated_steps += 1

print(f"画像更新率: {updated_steps / total_steps:.2%}")

# 统计三窗口平均增益
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

import numpy as np
print(f"过去窗口平均增益: {np.mean(past_gains):.4f}")
print(f"当前窗口平均增益: {np.mean(current_gains):.4f}")
print(f"未来窗口平均增益: {np.mean(future_gains):.4f}")
```

---

## 🎯 预期结果

如果 clasp_online 有效，应该观察到：

1. ✅ **画像更新率**: 60-90%
2. ✅ **三窗口增益都为正**: 
   - past_gain > 0（不遗忘）
   - current_gain > 0（当前有效）
   - future_gain > 0（泛化好）
3. ✅ **泛化能力**: current_gain ≈ future_gain
4. ✅ **持续改进**: Q 值随步骤递增

---

## 🔧 性能优化

### 如果运行太慢

1. **增加并行度**（修改 `src/config.py`）：
   ```python
   DPO_WORKERS = 15              # 候选评估线程数（默认 10）
   DPO_USER_PROCESSES = 8        # 多用户并行进程数（默认 5）
   ```

2. **使用 GPU 加速语义分**：
   ```bash
   --scorer-device cuda  # 默认 cpu
   ```

3. **减少候选数**（修改 `src/config.py`）：
   ```python
   NUM_CANDIDATE_PROFILES = 5    # 默认 10
   ```

### 如果时间太长

采用**采样方案**：只对 20% 用户做三窗口评估

修改 `comparison/window_chain_eval.py`：
```python
# 在 clasp_online 部分
import random
if profile != old_profile and random.random() < 0.2:  # 20% 采样
    three_window_eval = evaluate_three_windows(...)
```

---

## 🐛 常见问题

### Q1: 模型服务连接失败
```
openai.APIConnectionError: Connection error.
```

**解决**：检查 vLLM 服务是否启动，端口是否正确

### Q2: 显存不足
```
CUDA out of memory
```

**解决**：降低 `--gpu-memory-utilization` 或使用更小的 `--max-model-len`

### Q3: 评估速度太慢

**解决**：
1. 增加 `DPO_WORKERS` 和 `DPO_USER_PROCESSES`
2. 使用 `--scorer-device cuda`
3. 采用采样方案

---

## 📚 相关文档

- `comparison/THREE_WINDOW_IMPLEMENTATION.md` - 实现文档
- `comparison/THREE_WINDOW_COMPLETION_SUMMARY.md` - 完成总结
- `comparison/UNIFIED_HISTORY_MODIFICATION_SUMMARY.md` - 历史统一说明
- `comparison/README.md` - 对比评估说明

---

## ✅ 检查清单

启动评估前，确认：

- [ ] vLLM 服务已启动（端口 8000, 8001）
- [ ] 测试数据已窗口化（`output/windowed/test/`）
- [ ] 配置文件正确（`src/config.py`）
- [ ] 有足够的磁盘空间（至少 10GB）
- [ ] 有足够的时间（24-36 小时）

---

## 🎉 开始评估

```bash
# 1. 启动模型服务（两个终端）
# 终端 1: 画像生成模型
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server ...

# 终端 2: 动作预测模型
CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server ...

# 2. 验证服务
curl http://localhost:8000/v1/models
curl http://localhost:8001/v1/models

# 3. 运行快速测试
python -m comparison.run_baseline_comparison \
  --input-jsonl output/windowed/test/community_0.jsonl \
  --methods clasp_online \
  --max-users 1 \
  --comparison-root output/comparison_test \
  --scorer-device cpu \
  --skip-window-split

# 4. 检查输出
cat output/comparison_test/clasp_online/baseline_chain_community_0.jsonl | python -m json.tool

# 5. 运行完整评估
python -m comparison.run_baseline_comparison \
  --split test \
  --windowed-root output/windowed \
  --methods clasp_online \
  --comparison-root output/comparison \
  --scorer-device cpu \
  --skip-window-split
```

祝评估顺利！🚀
