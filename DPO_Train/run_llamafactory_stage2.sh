#!/usr/bin/env bash
# 仅训练阶段 2：默认从阶段 1 输出目录自动选最新的 checkpoint-* 作为 adapter_name_or_path
# 可覆盖环境变量： STAGE1_OUT、STAGE2_OUT、CFG、HANDOFF_CKPT、CLASP_STAGE2_RESUME
# 断点续训（阶段 2 已跑过一段）：任选其一
#   HANDOFF_CKPT=/path/to/clasp_profile_dpo_stage2_commercial/checkpoint-900 bash ...
#   CLASP_STAGE2_RESUME=1 bash ...   （自动选 STAGE2_OUT 下最新 checkpoint-*）
set -euo pipefail

DPO_TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPER_ROOT="${DEEPER_ROOT:-/home/xiaosong/personality/DEEPER}"
LF_ROOT="${LF_ROOT:-$DEEPER_ROOT/LLaMA-Factory}"
CFG="$DPO_TRAIN_DIR/config/clasp_profile_dpo_stage2_llama3_lora.yaml"
STAGE1_OUT="${STAGE1_OUT:-$DEEPER_ROOT/saves/clasp_profile_dpo_stage1_base}"
STAGE2_OUT="${STAGE2_OUT:-$DEEPER_ROOT/saves/clasp_profile_dpo_stage2_commercial}"

find_latest_checkpoint_dir() {
  local root="$1" best="" best_num=-1 d base num
  shopt -s nullglob
  for d in "$root"/checkpoint-*; do
    [[ -d "$d" ]] || continue
    base=$(basename "$d")
    num=${base#checkpoint-}
    if [[ "$num" =~ ^[0-9]+$ ]] && [[ "$num" -gt "$best_num" ]]; then
      best_num=$num
      best=$d
    fi
  done
  printf '%s\n' "${best}"
}

requires_adapter_files() {
  local dir="$1"
  [[ -d "$dir" ]] && [[ -f "$dir/adapter_config.json" ]] &&
    [[ -f "$dir/adapter_model.safetensors" || -f "$dir/adapter_model.bin" ]]
}

if [[ -n "${HANDOFF_CKPT:-}" ]]; then
  ADAPTER="$HANDOFF_CKPT"
elif [[ "${CLASP_STAGE2_RESUME:-0}" == "1" ]]; then
  ADAPTER="$(find_latest_checkpoint_dir "$STAGE2_OUT")"
else
  ADAPTER="$(find_latest_checkpoint_dir "$STAGE1_OUT")"
fi

if [[ -z "$ADAPTER" ]] || ! requires_adapter_files "$ADAPTER"; then
  if [[ "${CLASP_STAGE2_RESUME:-0}" == "1" ]]; then
    echo "ERROR: 阶段 2 续训未找到 checkpoint（需含 adapter_model.*），路径: $STAGE2_OUT/checkpoint-*" >&2
  else
    echo "ERROR: 未找到可用的阶段 1 checkpoint（需含 adapter_model.*），路径: ${HANDOFF_CKPT:-$STAGE1_OUT/checkpoint-*}" >&2
  fi
  exit 2
fi

echo "== [stage2] adapter_name_or_path=$ADAPTER"

cd "$LF_ROOT"
exec llamafactory-cli train "$CFG" adapter_name_or_path="$ADAPTER"
