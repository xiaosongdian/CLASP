#!/bin/bash
# 启动模型服务并运行基线评估（5个用户测试）

set -e  # 遇到错误立即退出

echo "=========================================="
echo "Clasp 基线评估启动脚本"
echo "=========================================="
echo ""

# 检查 GPU
echo "1. 检查 GPU 状态..."
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader
echo ""

# 启动画像生成模型（端口 8000）
echo "2. 启动画像生成模型（端口 8000）..."
echo "   模型: Meta-Llama-3-8B-Instruct"
echo "   命令: 在新终端运行以下命令"
echo ""
echo "CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \\"
echo "  --model /data/LLM_models/Meta-Llama-3-8B-Instruct \\"
echo "  --served-model-name Meta-Llama-3-8B-Instruct \\"
echo "  --port 8000 \\"
echo "  --dtype bfloat16 \\"
echo "  --max-model-len 4096 \\"
echo "  --gpu-memory-utilization 0.45"
echo ""
read -p "按回车键继续（确认已在新终端启动画像模型）..."
echo ""

# 启动动作预测模型（端口 8002）
echo "3. 启动动作预测模型（端口 8002）..."
echo "   模型: Meta-Llama-3-8B-Instruct-bluesky-sft-289"
echo "   命令: 在新终端运行以下命令"
echo ""
echo "CUDA_VISIBLE_DEVICES=0 python -m vllm.entrypoints.openai.api_server \\"
echo "  --model /data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft-289 \\"
echo "  --served-model-name Meta-Llama-3-8B-Instruct-bluesky-sft-289 \\"
echo "  --port 8002 \\"
echo "  --dtype bfloat16 \\"
echo "  --max-model-len 4096 \\"
echo "  --gpu-memory-utilization 0.45"
echo ""
read -p "按回车键继续（确认已在新终端启动动作模型）..."
echo ""

# 验证服务
echo "4. 验证模型服务..."
echo "   检查画像生成模型（端口 8000）..."
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "   ✅ 画像生成模型服务正常"
else
    echo "   ❌ 画像生成模型服务未响应"
    exit 1
fi

echo "   检查动作预测模型（端口 8002）..."
if curl -s http://localhost:8002/v1/models > /dev/null 2>&1; then
    echo "   ✅ 动作预测模型服务正常"
else
    echo "   ❌ 动作预测模型服务未响应"
    exit 1
fi
echo ""

# 检查配置
echo "5. 检查配置文件..."
echo "   画像 API: http://localhost:8000/v1"
echo "   动作 API: http://localhost:8002/v1"
echo ""

# 运行评估
echo "6. 运行基线评估（5个用户）..."
echo "   方法: static_s0, clasp_online, prefix_refresh, incremental_persona"
echo "   用户数: 5"
echo "   输出: output/comparison_5users/"
echo ""

python -m comparison.run_baseline_comparison \
  --input-jsonl output/windowed/test/community_0.jsonl \
  --methods static_s0,clasp_online,prefix_refresh,incremental_persona \
  --max-users 5 \
  --comparison-root output/comparison_5users \
  --scorer-device cpu \
  --skip-window-split

echo ""
echo "=========================================="
echo "✅ 评估完成！"
echo "=========================================="
echo ""
echo "查看结果："
echo "  ls -lh output/comparison_5users/*/baseline_chain_community_0.jsonl"
echo ""
echo "查看详细数据："
echo "  cat output/comparison_5users/clasp_online/baseline_chain_community_0.jsonl | python -m json.tool | less"
echo ""
