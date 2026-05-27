#!/usr/bin/env bash
#
# Start vLLM services for DPO profile slice evaluation
# - Port 8001: Persona model (base or DPO checkpoint)
# - Port 8002: Action model (Bluesky SFT)
#
# The first vLLM service is started and waits until ready before starting the second,
# to avoid GPU memory conflicts.
#
# Usage:
#   ./scripts/start_vllm_dpo_slice_tmux.sh
#   tmux attach -t vllm_dpo_slice
#
# Environment variables (optional):
#   PROFILE_MODEL_PATH       Path to persona model checkpoint
#   PROFILE_SERVED_NAME      Model name for persona API (must match config.py)
#   ACTION_MODEL_PATH        Path to action model checkpoint
#   ACTION_SERVED_NAME       Model name for action API (must match config.py)
#   VLLM_START_WAIT_MAX_SEC  Max wait time for first service (default: 120s)
#   VLLM_START_POLL_INTERVAL Polling interval (default: 5s)
#   GPU_PROFILE              GPU ID for persona model (default: 0)
#   GPU_ACTION               GPU ID for action model (default: 0)
#
# Note: The model paths must point to complete HuggingFace model directories
#       (with config.json), not LoRA adapter-only directories.
#

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Default paths - override with environment variables
PROFILE_MODEL_PATH="${PROFILE_MODEL_PATH:-models/Meta-Llama-3.1-8B-Instruct}"
PROFILE_SERVED_NAME="${PROFILE_SERVED_NAME:-Meta-Llama-3.1-8B-Instruct}"

ACTION_MODEL_PATH="${ACTION_MODEL_PATH:-models/Meta-Llama-3.1-8B-Instruct-bluesky-sft}"
ACTION_SERVED_NAME="${ACTION_SERVED_NAME:-Meta-Llama-3.1-8B-Instruct-bluesky-sft}"

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
    echo "[✗] ${label}: Directory not found"
    echo "    $path"
    return 1
  fi
  if [[ ! -f "$path/config.json" ]]; then
    echo "[✗] ${label}: Not a valid HuggingFace model directory (missing config.json)"
    echo "    $path"
    echo "    vLLM requires a complete model directory with config.json,"
    echo "    not a LoRA adapter-only directory."
    return 1
  fi
  return 0
}

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

if ! require_hf_model_dir "Persona model" "$PROFILE_MODEL_PATH"; then
  echo ""
  echo "Please set PROFILE_MODEL_PATH to your local Llama-3-8B-Instruct directory, e.g.:"
  echo "  PROFILE_MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct ./scripts/start_vllm_dpo_slice_tmux.sh"
  exit 1
fi

if ! require_hf_model_dir "Action model" "$ACTION_MODEL_PATH"; then
  echo ""
  echo "Please set ACTION_MODEL_PATH to your trained/merged Bluesky-SFT checkpoint, e.g.:"
  echo "  ACTION_MODEL_PATH=/path/to/Meta-Llama-3.1-8B-Instruct-bluesky-sft \\"
  echo "  ACTION_SERVED_NAME=Meta-Llama-3.1-8B-Instruct-bluesky-sft \\"
  echo "  ./scripts/start_vllm_dpo_slice_tmux.sh"
  echo ""
  echo "(Must match COMPARISON_CLASP_ACTION_VLLM_MODEL in src/config.py)"
  exit 1
fi

echo "=========================================="
echo "Starting vLLM DPO Slice Evaluation"
echo "  [8001 Persona] path=$PROFILE_MODEL_PATH"
echo "                 served-model-name=$PROFILE_SERVED_NAME"
echo "  [8002 Action]  path=$ACTION_MODEL_PATH"
echo "                 served-model-name=$ACTION_SERVED_NAME"
echo "  Starting :${PROFILE_PORT} first, then :${ACTION_PORT}"
echo "=========================================="

tmux new-session -d -s "$SESSION"

tmux rename-window -t "${SESSION}:0" 'profile_8001'
tmux send-keys -t "${SESSION}:0" "cd '$ROOT'" C-m
tmux send-keys -t "${SESSION}:0" "CUDA_VISIBLE_DEVICES=${GPU_PROFILE} python -m vllm.entrypoints.openai.api_server --model '${PROFILE_MODEL_PATH}' --served-model-name '${PROFILE_SERVED_NAME}' --port ${PROFILE_PORT} --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.3" C-m

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
  echo "[✗] Persona vLLM not ready within ${WAIT_MAX_SEC}s. Check logs: tmux attach -t ${SESSION}"
  exit 1
fi

echo "[✓] Port ${PROFILE_PORT} ready (/v1/models HTTP 200)"

echo ""
echo "[2/2] Starting action vLLM (port ${ACTION_PORT})..."
tmux new-window -t "${SESSION}:1" -n 'action_8002'
tmux send-keys -t "${SESSION}:1" "cd '$ROOT'" C-m
tmux send-keys -t "${SESSION}:1" "CUDA_VISIBLE_DEVICES=${GPU_ACTION} python -m vllm.entrypoints.openai.api_server --model '${ACTION_MODEL_PATH}' --served-model-name '${ACTION_SERVED_NAME}' --port ${ACTION_PORT} --dtype bfloat16 --max-model-len 4096 --gpu-memory-utilization 0.5" C-m

echo ""
echo "Both processes started."
echo "  Verify src/config.py settings:"
echo "    PROFILE_API_BASE=http://localhost:8001/v1"
echo "    ACTION_API_BASE=http://localhost:8002/v1"
echo "  Attach: tmux attach -t ${SESSION}"
echo "  Window 0: Persona (8001)  Window 1: Action (8002)"
echo "  Detach: Ctrl+B then D"
echo "  Kill:   tmux kill-session -t ${SESSION}"
echo ""
