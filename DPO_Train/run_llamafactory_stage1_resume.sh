#!/usr/bin/env bash
# 从 checkpoint-500 续跑阶段 1（见 config 内 resume_from_checkpoint）
set -euo pipefail

DPO_TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPER_ROOT="${DEEPER_ROOT:-/home/xiaosong/personality/DEEPER}"
LF_ROOT="${LF_ROOT:-$DEEPER_ROOT/LLaMA-Factory}"
CFG="$DPO_TRAIN_DIR/config/clasp_profile_dpo_stage1_resume_checkpoint500.yaml"

cd "$LF_ROOT"
exec llamafactory-cli train "$CFG"
