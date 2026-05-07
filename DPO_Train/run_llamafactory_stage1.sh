#!/usr/bin/env bash
# 仅训练阶段 1（请事先已导出 profile_dpo_stage1.jsonl，或先运行 export_llamafactory_data.sh）
set -euo pipefail

DPO_TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPER_ROOT="${DEEPER_ROOT:-/home/xiaosong/personality/DEEPER}"
LF_ROOT="${LF_ROOT:-$DEEPER_ROOT/LLaMA-Factory}"
CFG="$DPO_TRAIN_DIR/config/clasp_profile_dpo_stage1_llama3_lora.yaml"

cd "$LF_ROOT"
exec llamafactory-cli train "$CFG"
