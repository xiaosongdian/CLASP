#!/usr/bin/env python3
"""将 Clasp 导出的 dpo_pairs jsonl 转为 LLaMA-Factory Alpaca + ranking 可用的 jsonl。

每行字段（与 DEEPER/data/dataset_info 中 `clasp_profile_dpo_stage*` 对齐）：
  - system: 画像精炼 system 指令（与线上一致）
  - prompt: user 侧「旧画像 + 行为误差」块
  - chosen / rejected: 两个待选画像全文

用法（在 Clasp 仓库根目录执行）::

    python scripts/export_clasp_dpo_for_llamafactory.py \\
        --input output/dpo/train/dpo_pairs_stage1_base_only.jsonl \\
        --output /path/to/DEEPER/data/DEEPER_train_data/clasp/profile_dpo_stage1.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import (  # noqa: E402
    PROFILE_REFINEMENT_DISCREPANCY_MAX_CHARS,
    PROFILE_REFINEMENT_OLD_PERSONA_MAX_CHARS,
)
from src.profile_generator import truncate_behavior_plaintext  # noqa: E402
from src.prompts import build_profile_refinement_prompt_messages  # noqa: E402


def _preference_ok(chosen: Dict[str, Any], rejected: Dict[str, Any]) -> bool:
    if "r_all" not in chosen or "r_all" not in rejected:
        return True
    try:
        return float(chosen["r_all"]) > float(rejected["r_all"])
    except (TypeError, ValueError):
        return True


def export_file(inp: Path, out: Path, *, skip_bad_preference: bool) -> tuple[int, int, int]:
    out.parent.mkdir(parents=True, exist_ok=True)
    n_in = n_out = n_skip = 0
    with inp.open("r", encoding="utf-8") as fi, out.open("w", encoding="utf-8") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            n_in += 1
            obj = json.loads(line)
            chosen = obj.get("chosen")
            rejected = obj.get("rejected")
            if not isinstance(chosen, dict) or not isinstance(rejected, dict):
                n_skip += 1
                continue
            if "profile" not in chosen or "profile" not in rejected:
                n_skip += 1
                continue
            if skip_bad_preference and not _preference_ok(chosen, rejected):
                n_skip += 1
                continue
            baseline = obj.get("baseline_profile")
            if baseline is None or not str(baseline).strip():
                n_skip += 1
                continue
            disc_raw = obj.get("discrepancies")
            disc = "" if disc_raw is None else str(disc_raw)
            old_t = truncate_behavior_plaintext(
                str(baseline), int(PROFILE_REFINEMENT_OLD_PERSONA_MAX_CHARS)
            )
            disc_t = truncate_behavior_plaintext(disc, int(PROFILE_REFINEMENT_DISCREPANCY_MAX_CHARS))
            messages = build_profile_refinement_prompt_messages(old_t, disc_t)
            if len(messages) < 2:
                n_skip += 1
                continue
            system = str(messages[0].get("content", ""))
            prompt = str(messages[1].get("content", ""))
            row = {
                "system": system,
                "prompt": prompt,
                "chosen": str(chosen["profile"]),
                "rejected": str(rejected["profile"]),
            }
            fo.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1
    return n_in, n_out, n_skip


def main() -> None:
    p = argparse.ArgumentParser(description="Clasp dpo_pairs -> LLaMA-Factory jsonl")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--allow-inverted-pairs",
        action="store_true",
        help="保留 chosen.r_all<=rejected.r_all 的样本（默认跳过）",
    )
    args = p.parse_args()
    n_in, n_out, n_skip = export_file(
        args.input,
        args.output,
        skip_bad_preference=not args.allow_inverted_pairs,
    )
    print(f"[export] 读入行: {n_in}, 写出: {n_out}, 跳过: {n_skip}")
    print(f"[export] 输出: {args.output}")


if __name__ == "__main__":
    main()
