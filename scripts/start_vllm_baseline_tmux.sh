#!/usr/bin/env bash
#
# 基线对比：用 vLLM 启动两个相同的 Llama-3-8B-Instruct「基础」checkpoint。
# - 端口 8001：画像（对应 PROFILE_API_BASE）
# - 端口 8002：动作（对应 ACTION_API_BASE）
#
# served-model-name 须与 src.config.COMPARISON_BASELINE_VLLM_MODEL 一致（对比脚本里按路径请求）。
#
# 先启动第一个 vLLM，等 /v1/models 可访问后再启动第二个，避免双进程同时占显存导致失败。
#
# 用法：
#   ./scripts/start_vllm_baseline_tmux.sh
#   tmux attach -t vllm_baseline
#
# 环境变量（可选）：
#   VLLM_START_WAIT_MAX_SEC=900   等待第一个服务就绪的最长时间（秒）
#   VLLM_START_POLL_INTERVAL=5    轮询间隔（秒）
#   GPU_PROFILE=0 GPU_ACTION=1    双卡时指定 GPU
#

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 与 src/config.py 中 COMPARISON_BASELINE_VLLM_MODEL 对齐（OpenAI API 的 model 字段）
BASE_MODEL_PATH="/data/LLM_models/Meta-Llama-3-8B-Instruct"
SERVED_MODEL_NAME="Meta-Llama-3-8B-Instruct"

PROFILE_PORT=8001
ACTION_PORT=8002

GPU_PROFILE="${GPU_PROFILE:-0}"
GPU_ACTION="${GPU_ACTION:-0}"

WAIT_MAX_SEC="${VLLM_START_WAIT_MAX_SEC:-120}"
POLL_SEC="${VLLM_START_POLL_INTERVAL:-5}"

SESSION="vllm_baseline"

if ! command -v tmux &>/dev/null; then
  echo "需要安装 tmux，例如: sudo apt-get install tmux"
  exit 1
fi

if ! command -v curl &>/dev/null; then
  echo "需要 curl 用于检测 vLLM 是否就绪"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux 会话已存在: $SESSION"
  echo "请先: tmux kill-session -t $SESSION   或   tmux attach -t $SESSION"
  exit 1
fi

echo "=========================================="
echo "vLLM 基线（双端口同基础模型，顺序启动）"
echo "  模型目录: $BASE_MODEL_PATH"
echo "  served-model-name: $SERVED_MODEL_NAME"
echo "  先 :${PROFILE_PORT}，就绪后再 :${ACTION_PORT}"
echo "=========================================="

tmux new-session -d -s "$SESSION"

tmux rename-window -t "${SESSION}:0" 'profile_8001'
tmux send-keys -t "${SESSION}:0" "cd '$ROOT'" C-m
tmux send-keys -t "${SESSION}:0" "CUDA_VISIBLE_DEVICES=${GPU_PROFILE} python -m vllm.entrypoints.openai.api_server --model '${BASE_MODEL_PATH}' --served-model-name '${SERVED_MODEL_NAME}' --port ${PROFILE_PORT} --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.3" C-m

echo ""
echo "[1/2] 已启动画像侧 vLLM（端口 ${PROFILE_PORT}），等待 API 就绪（最长 ${WAIT_MAX_SEC}s，每 ${POLL_SEC}s 检查）..."

elapsed=0
ready=0
while [ "$elapsed" -lt "$WAIT_MAX_SEC" ]; do
  if code=$(curl -s -S -o /dev/null -w "%{http_code}" "http://127.0.0.1:${PROFILE_PORT}/v1/models" 2>/dev/null) && [ "$code" = "200" ]; then
    ready=1
    break
  fi
  sleep "$POLL_SEC"
  elapsed=$((elapsed + POLL_SEC))
  if [ $((elapsed % 30)) -eq 0 ] || [ "$elapsed" -eq "$POLL_SEC" ]; then
    echo "  ... 已等待 ${elapsed}s"
  fi
done

if [ "$ready" != "1" ]; then
  echo "[✗] 第一个 vLLM 在 ${WAIT_MAX_SEC}s 内未就绪。请 tmux attach -t ${SESSION} 查看窗口 0 日志。"
  exit 1
fi

echo "[✓] 端口 ${PROFILE_PORT} 已就绪（/v1/models HTTP 200）"

echo ""
echo "[2/2] 启动动作侧 vLLM（端口 ${ACTION_PORT}）..."
tmux new-window -t "${SESSION}:1" -n 'action_8002'
tmux send-keys -t "${SESSION}:1" "cd '$ROOT'" C-m
tmux send-keys -t "${SESSION}:1" "CUDA_VISIBLE_DEVICES=${GPU_ACTION} python -m vllm.entrypoints.openai.api_server --model '${BASE_MODEL_PATH}' --served-model-name '${SERVED_MODEL_NAME}' --port ${ACTION_PORT} --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.5" C-m

echo ""
echo "两个进程均已下发启动命令。"
echo "  接入: tmux attach -t ${SESSION}"
echo "  窗口 0: 画像（8001）  窗口 1: 动作（8002）"
echo "  分离: Ctrl+B 然后按 D"
echo "  结束: tmux kill-session -t ${SESSION}"
echo ""
