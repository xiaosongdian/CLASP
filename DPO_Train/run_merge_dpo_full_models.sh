#!/usr/bin/env bash
# 将阶段 1 / 阶段 2 训练目录中的 LoRA 与 Meta-Llama-3-8B-Instruct 合并，
# 在 /data/LLM_models/ 下生成两个完整画像模型目录（与原始基座同级）。
set -euo pipefail

DPO_TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEEPER_ROOT="${DEEPER_ROOT:-/home/xiaosong/personality/DEEPER}"
LF_ROOT="${LF_ROOT:-$DEEPER_ROOT/LLaMA-Factory}"

CFG1="$DPO_TRAIN_DIR/config/merge_clasp_profile_dpo_stage1_full.yaml"
CFG2="$DPO_TRAIN_DIR/config/merge_clasp_profile_dpo_stage2_full.yaml"

for p in "$CFG1" "$CFG2"; do
  [[ -f "$p" ]] || { echo "缺少配置文件: $p" >&2; exit 2; }
done

cd "$LF_ROOT"
echo "== [merge] stage1 -> $(grep '^export_dir:' "$CFG1" | awk '{print $2}')"
llamafactory-cli export "$CFG1"
echo "== [merge] stage2 -> $(grep '^export_dir:' "$CFG2" | awk '{print $2}')"
llamafactory-cli export "$CFG2"
echo "== 完成。"
