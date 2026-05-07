#!/bin/bash
# 一键启动脚本 - 在 tmux 中启动所有服务并运行评估

set -e

echo "=========================================="
echo "Clasp 基线评估 - 一键启动"
echo "=========================================="
echo ""

# 检查 tmux
if ! command -v tmux &> /dev/null; then
    echo "❌ 需要安装 tmux"
    echo "   sudo apt-get install tmux"
    exit 1
fi

# 创建 tmux 会话
SESSION="clasp_eval"

echo "1. 创建 tmux 会话: $SESSION"
tmux new-session -d -s $SESSION

# 窗口 0: 画像生成模型（显存占用小）
echo "2. 启动画像生成模型（端口 8001）..."
tmux rename-window -t $SESSION:0 'profile_model'
tmux send-keys -t $SESSION:0 "cd /home/xiaosong/personality/Clasp" C-m
tmux send-keys -t $SESSION:0 "CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model /data/LLM_models/Meta-Llama-3-8B-Instruct --served-model-name Meta-Llama-3-8B-Instruct --port 8001 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.30" C-m

# 窗口 1: 动作预测模型（显存占用大）
echo "3. 启动动作预测模型（端口 8002）..."
tmux new-window -t $SESSION:1 -n 'action_model'
tmux send-keys -t $SESSION:1 "cd /home/xiaosong/personality/Clasp" C-m
tmux send-keys -t $SESSION:1 "CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server --model /data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft --served-model-name Meta-Llama-3-8B-Instruct-bluesky-sft --port 8002 --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.60" C-m

# 窗口 2: 评估脚本
echo "4. 准备评估窗口..."
tmux new-window -t $SESSION:2 -n 'evaluation'
tmux send-keys -t $SESSION:2 "cd /home/xiaosong/personality/Clasp" C-m

echo ""
echo "等待模型启动（60秒）..."
sleep 30

# 验证服务
echo ""
echo "5. 验证服务..."
MAX_RETRIES=10
RETRY=0

while [ $RETRY -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:8001/v1/models > /dev/null 2>&1 && \
       curl -s http://localhost:8002/v1/models > /dev/null 2>&1; then
        echo "   ✅ 所有服务已就绪"
        break
    fi
    RETRY=$((RETRY+1))
    echo "   等待服务启动... ($RETRY/$MAX_RETRIES)"
    sleep 10
done

if [ $RETRY -eq $MAX_RETRIES ]; then
    echo "   ❌ 服务启动超时"
    echo ""
    echo "请手动检查："
    echo "   tmux attach -t $SESSION"
    exit 1
fi

# 运行评估
echo ""
echo "6. 运行评估（5个用户）..."
tmux send-keys -t $SESSION:2 "python -m comparison.run_baseline_comparison --input-jsonl output/windowed/test/community_0.jsonl --methods static_s0,clasp_online,prefix_refresh,incremental_persona --max-users 5 --comparison-root output/comparison_5users --scorer-device cpu --skip-window-split" C-m

echo ""
echo "=========================================="
echo "✅ 启动完成！"
echo "=========================================="
echo ""
echo "查看运行状态："
echo "   tmux attach -t $SESSION"
echo ""
echo "切换窗口："
echo "   Ctrl+B 然后按 0 - 查看画像模型"
echo "   Ctrl+B 然后按 1 - 查看动作模型"
echo "   Ctrl+B 然后按 2 - 查看评估进度"
echo ""
echo "退出 tmux（不关闭会话）："
echo "   Ctrl+B 然后按 D"
echo ""
echo "关闭所有服务："
echo "   tmux kill-session -t $SESSION"
echo ""
