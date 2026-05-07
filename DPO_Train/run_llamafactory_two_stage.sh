#!/usr/bin/env bash
# Clasp 画像 DPO：导出 jsonl → 阶段 1（若有 checkpoint 则续训）→ 阶段 2（加载阶段 1 最新 checkpoint-* 目录）
#
# 环境变量（可选）：
#   FORCE_FRESH_STAGE1=1   忽略已有 checkpoint，按 yaml 从零训练阶段 1（注意 yaml 里 overwrite_output_dir:true 仍会生效）
#   SKIP_STAGE2=1        只跑到阶段 1 结束，不启动阶段 2
#   CLASP_STAGE2_RESUME=1 阶段 2 若已有 checkpoint-*，则用其最新档续训（与 run_llamafactory_stage2.sh 一致）
#   DEEPER_ROOT / LF_ROOT  同上
#
# 说明：阶段 2 必须通过 adapter_name_or_path 指向包含 adapter_model.safetensors 的目录，故脚本取「最新」checkpoint-*；
#       若你希望手动指定档位，可先跑阶段 1，再在阶段 2 yaml 改路径后单独调用 run_llamafactory_stage2.sh。
#
set -euo pipefail

DPO_TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASP_ROOT="$(cd "$DPO_TRAIN_DIR/.." && pwd)"
DEEPER_ROOT="${DEEPER_ROOT:-/home/xiaosong/personality/DEEPER}"
LF_ROOT="${LF_ROOT:-$DEEPER_ROOT/LLaMA-Factory}"

CFG1="$DPO_TRAIN_DIR/config/clasp_profile_dpo_stage1_llama3_lora.yaml"
CFG2="$DPO_TRAIN_DIR/config/clasp_profile_dpo_stage2_llama3_lora.yaml"
STAGE1_OUT="${STAGE1_OUT:-$DEEPER_ROOT/saves/clasp_profile_dpo_stage1_base}"
STAGE2_OUT="${STAGE2_OUT:-$DEEPER_ROOT/saves/clasp_profile_dpo_stage2_commercial}"

export_st1="$DEEPER_ROOT/data/DEEPER_train_data/clasp/profile_dpo_stage1.jsonl"
export_st2="$DEEPER_ROOT/data/DEEPER_train_data/clasp/profile_dpo_stage2.jsonl"

# 选出 $1 目录下 checkpoint-<数字> 中步数最大的完整路径（无则输出空）
find_latest_checkpoint_dir() {
  local root="$1"
  local best="" best_num=-1 d base num
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
  [[ -d "$dir" ]] && [[ -f "$dir/adapter_config.json" ]] && [[ -f "$dir/adapter_model.safetensors" || -f "$dir/adapter_model.bin" ]]
}

mkdir -p "$(dirname "$export_st1")"

echo "== [export] stage1 → $export_st1"
python "$CLASP_ROOT/scripts/export_clasp_dpo_for_llamafactory.py" \
  --input "$CLASP_ROOT/output/dpo/train/dpo_pairs_stage1_base_only.jsonl" \
  --output "$export_st1"

echo "== [export] stage2 → $export_st2"
python "$CLASP_ROOT/scripts/export_clasp_dpo_for_llamafactory.py" \
  --input "$CLASP_ROOT/output/dpo/train/dpo_pairs_stage2_commercial_involved.jsonl" \
  --output "$export_st2"

STAGE1_CKPT="$(find_latest_checkpoint_dir "$STAGE1_OUT")"
cd "$LF_ROOT"

if [[ "${FORCE_FRESH_STAGE1:-0}" == "1" ]]; then
  echo "== [train] stage1: FORCE_FRESH_STAGE1=1 → 不从 checkpoint 续训"
  llamafactory-cli train "$CFG1"
else
  if [[ -n "$STAGE1_CKPT" ]] && requires_adapter_files "$STAGE1_CKPT"; then
    echo "== [train] stage1: 续训 ← $STAGE1_CKPT"
    llamafactory-cli train "$CFG1" \
      overwrite_output_dir=false \
      resume_from_checkpoint="$STAGE1_CKPT"
  else
    echo "== [train] stage1: 未发现可用 checkpoint-*，从零开始（沿用 yaml）"
    if [[ -n "$STAGE1_CKPT" ]]; then
      echo "    (warning: $STAGE1_CKPT 缺少 adapter 文件，已忽略)"
    fi
    llamafactory-cli train "$CFG1"
  fi
fi

STAGE1_CKPT_AFTER="$(find_latest_checkpoint_dir "$STAGE1_OUT")"
if [[ -z "$STAGE1_CKPT_AFTER" ]] || ! requires_adapter_files "$STAGE1_CKPT_AFTER"; then
  echo "ERROR: 阶段 1 结束后未找到含 adapter_model 的 checkpoint-*，目录: $STAGE1_OUT" >&2
  exit 2
fi

if [[ "${SKIP_STAGE2:-0}" == "1" ]]; then
  echo "== [skip] SKIP_STAGE2=1，不启动阶段 2。"
  echo "Done. stage1 最新适配器目录: $STAGE1_CKPT_AFTER"
  exit 0
fi

STAGE2_CKPT="$(find_latest_checkpoint_dir "$STAGE2_OUT")"
if [[ "${CLASP_STAGE2_RESUME:-0}" == "1" ]] && [[ -n "$STAGE2_CKPT" ]] && requires_adapter_files "$STAGE2_CKPT"; then
  echo "== [train] stage2: 续训 ← adapter_name_or_path=$STAGE2_CKPT"
  llamafactory-cli train "$CFG2" adapter_name_or_path="$STAGE2_CKPT"
else
  echo "== [train] stage2: adapter_name_or_path=$STAGE1_CKPT_AFTER"
  llamafactory-cli train "$CFG2" adapter_name_or_path="$STAGE1_CKPT_AFTER"
fi

echo "Done. stage1 → $STAGE1_OUT (最新适配器用于阶段 2 的 checkpoint: $STAGE1_CKPT_AFTER)"
echo "      stage2 → $DEEPER_ROOT/saves/clasp_profile_dpo_stage2_commercial"
