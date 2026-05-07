# 🚀 启动基线评估（5个用户）- 最终指南

## ✅ 配置已完成

- ✅ 配置文件已修改（`src/config.py`）
- ✅ 画像 API: `http://localhost:8000/v1`
- ✅ 动作 API: `http://localhost:8002/v1`
- ✅ 启动脚本已创建

---

## 🎯 两种启动方式

### 方式 1: 一键启动（推荐）⭐

使用 tmux 自动启动所有服务：

```bash
cd /home/xiaosong/personality/Clasp
bash scripts/start_all.sh
```

**优点**：
- 自动启动两个模型服务
- 自动验证服务状态
- 自动运行评估
- 所有服务在后台运行

**查看进度**：
```bash
# 进入 tmux 会话
tmux attach -t clasp_eval

# 切换窗口
Ctrl+B 然后按 0  # 画像模型日志
Ctrl+B 然后按 1  # 动作模型日志
Ctrl+B 然后按 2  # 评估进度

# 退出 tmux（不关闭服务）
Ctrl+B 然后按 D
```

**停止所有服务**：
```bash
tmux kill-session -t clasp_eval
```

---

### 方式 2: 手动启动（更灵活）

#### 终端 1: 启动画像生成模型

```bash
cd /home/xiaosong/personality/Clasp

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/LLM_models/Meta-Llama-3-8B-Instruct \
  --served-model-name Meta-Llama-3-8B-Instruct \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.30
```

**说明**: 画像生成压力小，显存占用 30%

**等待看到**: `INFO:     Uvicorn running on http://0.0.0.0:8000`

#### 终端 2: 启动动作预测模型

```bash
cd /home/xiaosong/personality/Clasp

CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \
  --model /data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft \
  --served-model-name Meta-Llama-3-8B-Instruct-bluesky-sft \
  --port 8002 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.60
```

**说明**: 动作预测压力大，显存占用 60%

**等待看到**: `INFO:     Uvicorn running on http://0.0.0.0:8002`

#### 终端 3: 验证服务

```bash
# 检查画像模型
curl http://localhost:8000/v1/models

# 检查动作模型
curl http://localhost:8002/v1/models
```

**应该看到**: JSON 格式的模型信息

#### 终端 3: 运行评估

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

### 1. 检查输出文件

```bash
ls -lh output/comparison_5users/*/baseline_chain_community_0.jsonl
```

**应该看到 4 个文件**：
```
output/comparison_5users/static_s0/baseline_chain_community_0.jsonl
output/comparison_5users/clasp_online/baseline_chain_community_0.jsonl
output/comparison_5users/prefix_refresh/baseline_chain_community_0.jsonl
output/comparison_5users/incremental_persona/baseline_chain_community_0.jsonl
```

### 2. 查看 clasp_online 的三窗口数据

```bash
# 查看第一个用户的完整数据
head -1 output/comparison_5users/clasp_online/baseline_chain_community_0.jsonl | python -m json.tool | less

# 检查是否包含三窗口评估
head -1 output/comparison_5users/clasp_online/baseline_chain_community_0.jsonl | grep -o "three_window_evaluation"
```

### 3. 快速对比各方法

```bash
echo "=== 各方法平均 Q 值对比 ==="
for method in static_s0 clasp_online prefix_refresh incremental_persona; do
  avg_q=$(cat output/comparison_5users/$method/baseline_chain_community_0.jsonl | \
    python -c "import sys, json; data=[json.loads(l) for l in sys.stdin]; print(f'{sum(d[\"mean_Q\"] for d in data)/len(data):.4f}')")
  echo "$method: $avg_q"
done
```

---

## 🎯 预期结果

如果一切正常，你应该看到：

1. ✅ 4 个方法的输出文件都存在
2. ✅ 每个文件包含 5 行（5 个用户）
3. ✅ clasp_online 的输出包含 `three_window_evaluation` 字段
4. ✅ clasp_online 的 Q 值 > static_s0 的 Q 值

---

## 🐛 常见问题

### Q1: 端口被占用
```bash
# 查看占用端口的进程
lsof -i :8000
lsof -i :8002

# 杀死进程
kill -9 <PID>
```

### Q2: 显存不足
降低 `--gpu-memory-utilization` 从 0.45 到 0.35

### Q3: 模型加载失败
```bash
# 检查模型路径
ls -la /data/LLM_models/Meta-Llama-3-8B-Instruct
ls -la /data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft-289
```

---

## 📈 下一步

如果 5 个用户测试成功：

### 1. 运行更大规模测试（50 个用户）

```bash
python -m comparison.run_baseline_comparison \
  --input-jsonl output/windowed/test/community_0.jsonl \
  --methods static_s0,clasp_online,prefix_refresh,incremental_persona \
  --max-users 50 \
  --comparison-root output/comparison_50users \
  --scorer-device cpu \
  --skip-window-split
```

### 2. 运行完整评估（1,801 个用户）

```bash
python -m comparison.run_baseline_comparison \
  --split test \
  --windowed-root output/windowed \
  --methods static_s0,clasp_online,prefix_refresh,incremental_persona \
  --comparison-root output/comparison \
  --scorer-device cpu \
  --skip-window-split
```

**预计时间**: 24-36 小时

---

## 🎉 快速开始

**推荐命令**（一键启动）：

```bash
cd /home/xiaosong/personality/Clasp
bash scripts/start_all.sh
```

然后等待 10-20 分钟，查看结果！

---

**需要帮助？** 查看详细文档：
- `START_BASELINE_5USERS.md` - 本文档
- `comparison/QUICK_START_GUIDE.md` - 完整指南
- `comparison/THREE_WINDOW_IMPLEMENTATION.md` - 三窗口统计说明
