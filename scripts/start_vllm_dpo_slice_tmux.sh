#!/usr/bin/env bash
#
# DPO 画像切片实验（comparison.run_dpo_profile_slice_eval）用 vLLM：双端口、顺序启动。
#
# - 端口 8001：画像 → 须与 src.config.COMPARISON_BASELINE_VLLM_MODEL 的 served-model-name 一致
#               （baseline / 部分流程的 vLLM 画像；gpt-4o-mini 走 OpenAI 兼容 API，不经本端口）
# - 端口 8002：动作 → 须与 src.config.COMPARISON_CLASP_ACTION_VLLM_MODEL 一致
#               （Meta-Llama-3-8B-Instruct-bluesky-sft）
#
# 先启 8001，等 /v1/models 返回 200 后再启 8002，避免双进程同时占显存失败。
#
# 用法：
#   ./scripts/start_vllm_dpo_slice_tmux.sh
#   tmux attach -t vllm_dpo_slice
#
# 环境变量（可选，默认与 src/config.py 中路径、模型名对齐）：
#   PROFILE_MODEL_PATH   画像权重目录（默认：Meta-Llama-3-8B-Instruct 基座）
#   PROFILE_SERVED_NAME  与 COMPARISON_BASELINE_VLLM_MODEL 一致
#   ACTION_MODEL_PATH    动作权重目录（默认见下方；本机若无该目录须显式指定）
#   ACTION_SERVED_NAME   与 COMPARISON_CLASP_ACTION_VLLM_MODEL 一致
#
# 常见报错：OSError: Can't load the configuration of '.../Meta-Llama-3-8B-Instruct-bluesky'
#   → 该路径下没有 HuggingFace 格式的 config.json（目录不存在、或只是 LoRA 适配器未合并）。
#   → 请把 ACTION_MODEL_PATH 指到「已合并/完整导出」的 SFT 权重目录。
#
#   VLLM_START_WAIT_MAX_SEC=900
#   VLLM_START_POLL_INTERVAL=5
#   GPU_PROFILE=0 GPU_ACTION=1
#
# 若你还要在同一台机跑 clasp_dpo 变体（DPO 画像），可另开会话或改 PROFILE_* 指向
# COMPARISON_CLASP_PROFILE_VLLM_MODEL 对应权重，并保持 config 里字符串与 --served-model-name 一致。
#

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 默认与 src/config.py 对齐
PROFILE_MODEL_PATH="${PROFILE_MODEL_PATH:-/data/LLM_models/Meta-Llama-3-8B-Instruct}"
PROFILE_SERVED_NAME="${PROFILE_SERVED_NAME:-Meta-Llama-3-8B-Instruct}"

ACTION_MODEL_PATH="${ACTION_MODEL_PATH:-/data/LLM_models/Meta-Llama-3-8B-Instruct-bluesky-sft}"
ACTION_SERVED_NAME="${ACTION_SERVED_NAME:-Meta-Llama-3-8B-Instruct-bluesky-sft}"

PROFILE_PORT=8001
ACTION_PORT=8002

GPU_PROFILE="${GPU_PROFILE:-0}"
GPU_ACTION="${GPU_ACTION:-0}"

WAIT_MAX_SEC="${VLLM_START_WAIT_MAX_SEC:-120}"
POLL_SEC="${VLLM_START_POLL_INTERVAL:-5}"

SESSION="vllm_dpo_slice"

require_hf_model_dir() {
  local label="$1"
  local path="$2"
  if [[ ! -d "$path" ]]; then
    echo "[✗] ${label}：目录不存在"
    echo "    $path"
    return 1
  fi
  if [[ ! -f "$path/config.json" ]]; then
    echo "[✗] ${label}：不是有效的 HuggingFace 模型目录（缺少 config.json）"
    echo "    $path"
    echo "    vLLM 的 --model 需要指向含 config.json 的完整权重目录（合并后的 checkpoint），"
    echo "    不能是仅含 adapter 的 LoRA 目录。"
    return 1
  fi
  return 0
}

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

if ! require_hf_model_dir "画像模型" "$PROFILE_MODEL_PATH"; then
  echo ""
  echo "请设置 PROFILE_MODEL_PATH 为本地 Llama-3-8B-Instruct 基座目录，例如："
  echo "  PROFILE_MODEL_PATH=/data/LLM_models/Meta-Llama-3-8B-Instruct ./scripts/start_vllm_dpo_slice_tmux.sh"
  exit 1
fi

if ! require_hf_model_dir "动作模型" "$ACTION_MODEL_PATH"; then
  echo ""
  echo "与终端报错一致：默认路径在本机无效。请指定你训练/合并后的 Bluesky-SFT 完整权重，例如："
  echo "  ACTION_MODEL_PATH=/path/to/Meta-Llama-3-8B-Instruct-bluesky-sft \\"
  echo "  ACTION_SERVED_NAME=Meta-Llama-3-8B-Instruct-bluesky-sft \\"
  echo "  ./scripts/start_vllm_dpo_slice_tmux.sh"
  echo ""
  echo "（须与 src/config.py 里 COMPARISON_CLASP_ACTION_VLLM_MODEL 的 model 名一致。）"
  exit 1
fi

echo "=========================================="
echo "vLLM · DPO 切片实验（画像基座 + 动作 bluesky-sft，顺序启动）"
echo "  [8001 画像] path=$PROFILE_MODEL_PATH"
echo "              served-model-name=$PROFILE_SERVED_NAME"
echo "  [8002 动作] path=$ACTION_MODEL_PATH"
echo "              served-model-name=$ACTION_SERVED_NAME"
echo "  先 :${PROFILE_PORT}，就绪后再 :${ACTION_PORT}"
echo "=========================================="

tmux new-session -d -s "$SESSION"

tmux rename-window -t "${SESSION}:0" 'profile_8001'
tmux send-keys -t "${SESSION}:0" "cd '$ROOT'" C-m
tmux send-keys -t "${SESSION}:0" "CUDA_VISIBLE_DEVICES=${GPU_PROFILE} python -m vllm.entrypoints.openai.api_server --model '${PROFILE_MODEL_PATH}' --served-model-name '${PROFILE_SERVED_NAME}' --port ${PROFILE_PORT} --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.3" C-m

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
  echo "[✗] 画像 vLLM 在 ${WAIT_MAX_SEC}s 内未就绪。请 tmux attach -t ${SESSION} 查看窗口 0 日志。"
  exit 1
fi

echo "[✓] 端口 ${PROFILE_PORT} 已就绪（/v1/models HTTP 200）"

echo ""
echo "[2/2] 启动动作侧 vLLM（端口 ${ACTION_PORT}）..."
tmux new-window -t "${SESSION}:1" -n 'action_8002'
tmux send-keys -t "${SESSION}:1" "cd '$ROOT'" C-m
tmux send-keys -t "${SESSION}:1" "CUDA_VISIBLE_DEVICES=${GPU_ACTION} python -m vllm.entrypoints.openai.api_server --model '${ACTION_MODEL_PATH}' --served-model-name '${ACTION_SERVED_NAME}' --port ${ACTION_PORT} --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.5" C-m

echo ""
echo "两个进程均已下发启动命令。"
echo "  请确认 src/config.py 中："
echo "    PROFILE_API_BASE=http://localhost:8001/v1"
echo "    ACTION_API_BASE=http://localhost:8002/v1"
echo "  接入: tmux attach -t ${SESSION}"
echo "  窗口 0: 画像（8001）  窗口 1: 动作（8002）"
echo "  分离: Ctrl+B 然后按 D"
echo "  结束: tmux kill-session -t ${SESSION}"
echo ""
