# 启动基线评估（5个用户测试）- 分步指南

## 📋 准备工作

### 检查配置
当前配置（`src/config.py`）：
- 画像 API: `http://localhost:8000/v1` (需要启动)
- 动作 API: `http://localhost:8002/v1` (需要启动)
- 注意：配置中的远程地址 `http://175.6.27.230:8001/v1` 需要改为本地

---

## 🚀 启动步骤

### Step 1: 修改配置文件

编辑 `src/config.py`，修改第 20 行和第 31 行：

```python
# 修改前
PROFILE_API_BASE = "http://175.6.27.230:8001/v1"
ACTION_API_BASE = "http://localhost:8002/v1"

# 修改后
PROFILE_API_BASE = "http://localhost:8000/v1"
ACTION_API_BASE = "http://localhost:8002/v1"
```

### Step 2: 启动画像生成模型（终端 1）

打开**第一个终端**，运行：

```bash
cd /home/xiaosong/personality/Clasp

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/LLM_models/Meta-Llama-3-8B-Instruct \
  --served-model-name Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45
```

**等待看到**：
```
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Step 3: 启动动作预测模型（终端 2）

打开**第二个终端**，运行：

```bash
cd /home/xiaosong/personality/Clasp

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft-289 \
  --served-model-name Meta-Llama-3-8B-Instruct-bluesky-sft-289 \
  --port 8002 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.45
```

**等待看到**：
```
INFO:     Uvicorn running on http://0.0.0.0:8002
```

### Step 4: 验证服务（终端 3）

打开**第三个终端**，验证服务：

```bash
# 检查画像模型
curl http://localhost:8000/v1/models

# 检查动作模型
curl http://localhost:8002/v1/models
```

**应该看到**：JSON 格式的模型信息

### Step 5: 运行评估（终端 3）

在第三个终端运行评估：

```bash
cd /home/xiaosong/personality/Clasp

python -m comparison.run_baseline_comparison \
  --input-jsonl output/windowed/test/community_0.jsonl \
  --methods static_s0,clasp_online,prefix_refresh,incremental_persona \
  --max-users 5 \
  --comparison-root output/comparison_5users \
  --scorer-device cpu \
  --skip-window-split
```

**预计时间**: 10-20 分钟

---

## 📊 查看结果

### 检查输出文件

```bash
# 查看生成的文件
ls -lh output/comparison_5users/*/baseline_chain_community_0.jsonl

# 应该看到 4 个文件（4 个方法）
# output/comparison_5users/static_s0/baseline_chain_community_0.jsonl
# output/comparison_5users/clasp_online/baseline_chain_community_0.jsonl
# output/comparison_5users/prefix_refresh/baseline_chain_community_0.jsonl
# output/comparison_5users/incremental_persona/baseline_chain_community_0.jsonl
```

### 查看 clasp_online 的详细数据

```bash
# 查看第一个用户的完整数据
head -1 output/comparison_5users/clasp_online/baseline_chain_community_0.jsonl | python -m json.tool | less

# 检查是否包含三窗口数据
head -1 output/comparison_5users/clasp_online/baseline_chain_community_0.jsonl | grep -o "three_window_evaluation"
```

### 快速统计

```bash
# 统计每个方法的平均 Q 值
for method in static_s0 clasp_online prefix_refresh incremental_persona; do
  echo "=== $method ==="
  cat output/comparison_5users/$method/baseline_chain_community_0.jsonl | \
    python -c "import sys, json; data=[json.loads(l) for l in sys.stdin]; print(f'平均 Q: {sum(d[\"mean_Q\"] for d in data)/len(data):.4f}')"
done
```

---

## 🐛 常见问题

### Q1: 端口被占用
```
Address already in use
```

**解决**：
```bash
# 查看占用端口的进程
lsof -i :8000
lsof -i :8002

# 杀死进程
kill -9 <PID>
```

### Q2: 显存不足
```
CUDA out of memory
```

**解决**：降低 `--gpu-memory-utilization`
```bash
--gpu-memory-utilization 0.35  # 从 0.45 降到 0.35
```

### Q3: 模型加载失败
```
FileNotFoundError: /data/LLM_models/...
```

**解决**：检查模型路径是否正确
```bash
ls -la /data/LLM_models/Meta-Llama-3-8B-Instruct
ls -la /data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft-289
```

---

## ✅ 成功标志

评估成功完成后，你应该看到：

1. ✅ 4 个方法的输出文件都存在
2. ✅ 每个文件包含 5 行（5 个用户）
3. ✅ clasp_online 的输出包含 `three_window_evaluation` 字段
4. ✅ 终端显示 "✅ 评估完成"

---

## 📈 下一步

如果 5 个用户测试成功：

1. **运行更大规模测试**（50 个用户）
2. **运行完整评估**（1,801 个用户）
3. **分析结果并生成图表**

---

需要帮助？查看：
- `comparison/QUICK_START_GUIDE.md` - 完整指南
- `comparison/THREE_WINDOW_IMPLEMENTATION.md` - 三窗口统计说明
