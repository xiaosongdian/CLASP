#!/usr/bin/env bash
# 仅导出两阶段 jsonl 到 DEEPER/data（不写盘训练）
set -euo pipefail

DPO_TRAIN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLASP_ROOT="$(cd "$DPO_TRAIN_DIR/.." && pwd)"
DEEPER_ROOT="${DEEPER_ROOT:-/home/xiaosong/personality/DEEPER}"

export_st1="$DEEPER_ROOT/data/DEEPER_train_data/clasp/profile_dpo_stage1.jsonl"
export_st2="$DEEPER_ROOT/data/DEEPER_train_data/clasp/profile_dpo_stage2.jsonl"
mkdir -p "$(dirname "$export_st1")"

python "$CLASP_ROOT/scripts/export_clasp_dpo_for_llamafactory.py" \
  --input "$CLASP_ROOT/output/dpo/train/dpo_pairs_stage1_base_only.jsonl" \
  --output "$export_st1"
python "$CLASP_ROOT/scripts/export_clasp_dpo_for_llamafactory.py" \
  --input "$CLASP_ROOT/output/dpo/train/dpo_pairs_stage2_commercial_involved.jsonl" \
  --output "$export_st2"

echo "Exported: $export_st1"
echo "           $export_st2"
