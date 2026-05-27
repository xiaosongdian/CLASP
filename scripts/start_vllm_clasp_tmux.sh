#!/usr/bin/env bash
#
# Clasp approach: persona and action use different fine-tuned checkpoints (consistent with clasp_online in comparison).
# - Port 8001: persona DPO stage2
# - Port 8002: action bluesky SFT
#
# Start first vLLM, wait for /v1/models to be accessible before starting second, avoid dual processes competing for GPU memory/resources causing failure.
#
# Usage:
#   ./scripts/start_vllm_clasp_tmux.sh
#   tmux attach -t vllm_clasp
#
# Environment variables (optional):
#   VLLM_START_WAIT_MAX_SEC=900   max wait time for first service ready (seconds)
#   VLLM_START_POLL_INTERVAL=5    polling interval (seconds)
#   GPU_PROFILE=0 GPU_ACTION=1    specify GPU when dual-card
#

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Aligned with COMPARISON_CLASP_* in src/config.py
PROFILE_MODEL_PATH="/data/LLM_models/Meta-Llama-3.1-8B-Instruct-clasp-dpo-stage2"
ACTION_MODEL_PATH="/data/LLM_models/Meta-Llama-3.1-8B-Instruct-bluesky-sft"
SERVED_PROFILE_NAME="Meta-Llama-3.1-8B-Instruct-clasp-dpo-stage2"
SERVED_ACTION_NAME="Meta-Llama-3.1-8B-Instruct-bluesky-sft"

PROFILE_PORT=8001
ACTION_PORT=8002

GPU_PROFILE="${GPU_PROFILE:-0}"
GPU_ACTION="${GPU_ACTION:-0}"

WAIT_MAX_SEC="${VLLM_START_WAIT_MAX_SEC:-120}"
POLL_SEC="${VLLM_START_POLL_INTERVAL:-5}"

SESSION="vllm_clasp"

if ! command -v tmux &>/dev/null; then
  echo "tmux required, e.g.: sudo apt-get install tmux"
  exit 1
fi

if ! command -v curl &>/dev/null; then
  echo "curl required to detect vLLM readiness"
  exit 1
fi

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session already exists: $SESSION"
  echo "First: tmux kill-session -t $SESSION   or   tmux attach -t $SESSION"
  exit 1
fi

echo "=========================================="
echo "vLLM Clasp (persona DPO + action SFT, sequential startup)"
echo "  Persona: $PROFILE_MODEL_PATH"
echo "  Action: $ACTION_MODEL_PATH"
echo "  First :${PROFILE_PORT}, then :${ACTION_PORT} after ready"
echo "=========================================="

# --- Start only first model (window 0) ---
tmux new-session -d -s "$SESSION"
tmux rename-window -t "${SESSION}:0" 'profile_8001'
tmux send-keys -t "${SESSION}:0" "cd '$ROOT'" C-m
tmux send-keys -t "${SESSION}:0" "CUDA_VISIBLE_DEVICES=${GPU_PROFILE} python -m vllm.entrypoints.openai.api_server --model '${PROFILE_MODEL_PATH}' --served-model-name '${SERVED_PROFILE_NAME}' --port ${PROFILE_PORT} --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.3" C-m

echo ""
echo "[1/2] Started persona vLLM (port ${PROFILE_PORT}), waiting for API ready (max ${WAIT_MAX_SEC}s, check every ${POLL_SEC}s)..."

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
  echo "[✗] Persona vLLM not ready within ${WAIT_MAX_SEC}s. Run: tmux attach -t ${SESSION} to check window 0 logs."
  exit 1
fi

echo "[✓] Persona vLLM ready (${PROFILE_PORT}/v1/models HTTP 200)"

# --- Start second model (new window) ---
echo ""
echo "[2/2] Starting action vLLM (port ${ACTION_PORT})..."
tmux new-window -t "${SESSION}:1" -n 'action_8002'
tmux send-keys -t "${SESSION}:1" "cd '$ROOT'" C-m
tmux send-keys -t "${SESSION}:1" "CUDA_VISIBLE_DEVICES=${GPU_ACTION} python -m vllm.entrypoints.openai.api_server --model '${ACTION_MODEL_PATH}' --served-model-name '${SERVED_ACTION_NAME}' --port ${ACTION_PORT} --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.5" C-m

echo ""
echo "Both processes startup commands sent."
echo "  Attach: tmux attach -t ${SESSION}"
echo "  Window 0: persona (8001)  Window 1: action (8002)"
echo "  Detach: Ctrl+B then D"
echo "  End: tmux kill-session -t ${SESSION}"
echo ""
