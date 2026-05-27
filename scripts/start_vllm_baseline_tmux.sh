#!/usr/bin/env bash
#
# Start vLLM services for baseline comparison (same base model on two ports)
# - Port 8001: Persona generation (PROFILE_API_BASE)
# - Port 8002: Action prediction (ACTION_API_BASE)
#
# Both use the same base Llama-3.1-8B-Instruct checkpoint.
# The served-model-name must match COMPARISON_BASELINE_VLLM_MODEL in src/config.py.
#
# The first vLLM service is started and waits until ready before starting the second,
# to avoid GPU memory conflicts.
#
# Usage:
#   ./scripts/start_vllm_baseline_tmux.sh
#   tmux attach -t vllm_baseline
#
# Environment variables (optional):
#   BASE_MODEL_PATH          Path to base model checkpoint
#   SERVED_MODEL_NAME        Model name for API
#   VLLM_START_WAIT_MAX_SEC  Max wait time for first service (default: 120s)
#   VLLM_START_POLL_INTERVAL Polling interval (default: 5s)
#   GPU_PROFILE              GPU ID for persona model (default: 0)
#   GPU_ACTION               GPU ID for action model (default: 0)
#

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Default paths - override with environment variables
BASE_MODEL_PATH="${BASE_MODEL_PATH:-models/Meta-Llama-3.1-8B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Meta-Llama-3.1-8B-Instruct}"

PROFILE_PORT=8001
ACTION_PORT=8002

GPU_PROFILE="${GPU_PROFILE:-0}"
GPU_ACTION="${GPU_ACTION:-0}"

WAIT_MAX_SEC="${VLLM_START_WAIT_MAX_SEC:-120}"
POLL_SEC="${VLLM_START_POLL_INTERVAL:-5}"

SESSION="vllm_baseline"

if ! command -v tmux &>/dev/null; then
  echo "Error: tmux is required. Install with: sudo apt-get install tmux"
  exit 1
fi

if ! command -v curl &>/dev/null; then
  echo "Error: curl is required for health checks"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Error: tmux session already exists: $SESSION"
  echo "Please run: tmux kill-session -t $SESSION   or   tmux attach -t $SESSION"
  exit 1
fi

echo "=========================================="
echo "Starting vLLM Baseline (dual ports, same base model)"
echo "  Model: $BASE_MODEL_PATH"
echo "  Served name: $SERVED_MODEL_NAME"
echo "  Starting :${PROFILE_PORT} first, then :${ACTION_PORT}"
echo "=========================================="

tmux new-session -d -s "$SESSION"

tmux rename-window -t "${SESSION}:0" 'profile_8001'
tmux send-keys -t "${SESSION}:0" "cd '$ROOT'" C-m
tmux send-keys -t "${SESSION}:0" "CUDA_VISIBLE_DEVICES=${GPU_PROFILE} python -m vllm.entrypoints.openai.api_server --model '${BASE_MODEL_PATH}' --served-model-name '${SERVED_MODEL_NAME}' --port ${PROFILE_PORT} --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.5" C-m

echo ""
echo "[1/2] Started persona vLLM (port ${PROFILE_PORT}), waiting for API ready (max ${WAIT_MAX_SEC}s, polling every ${POLL_SEC}s)..."

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
    echo "  ... waited ${elapsed}s"
  fi
done

if [ "$ready" != "1" ]; then
  echo "[✗] First vLLM not ready within ${WAIT_MAX_SEC}s. Check logs: tmux attach -t ${SESSION}"
  exit 1
fi

echo "[✓] Port ${PROFILE_PORT} ready (/v1/models HTTP 200)"

echo ""
echo "[2/2] Starting action vLLM (port ${ACTION_PORT})..."
tmux new-window -t "${SESSION}:1" -n 'action_8002'
tmux send-keys -t "${SESSION}:1" "cd '$ROOT'" C-m
tmux send-keys -t "${SESSION}:1" "CUDA_VISIBLE_DEVICES=${GPU_ACTION} python -m vllm.entrypoints.openai.api_server --model '${BASE_MODEL_PATH}' --served-model-name '${SERVED_MODEL_NAME}' --port ${ACTION_PORT} --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.3" C-m

echo ""
echo "Both processes started."
echo "  Attach: tmux attach -t ${SESSION}"
echo "  Window 0: Persona (8001)  Window 1: Action (8002)"
echo "  Detach: Ctrl+B then D"
echo "  Kill:   tmux kill-session -t ${SESSION}"
echo ""
