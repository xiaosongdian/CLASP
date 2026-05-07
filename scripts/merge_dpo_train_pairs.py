#!/usr/bin/env python3
"""将各社区 dpo_pairs_community_*.jsonl 合并，并切分为两阶段 DPO 训练集。

- **阶段 1**：chosen / rejected 的 ``profile_source`` 均为 ``base``（纯本地基座画像对比）。
  先在该集上 DPO，得到「base 课程」后的模型。
- **阶段 2**：至少一侧为 ``commercial``（与阶段 1 **互斥**；阶段 1 ∪ 阶段 2 = 全量）。
  在阶段 1 模型上继续 DPO，引入含商用候选画像的偏好对。
- **全量**：所有社区合并后的完整 ``jsonl``（可选，便于一次性训练或抽查）。

用法示例::

    python scripts/merge_dpo_train_pairs.py \\
        --input-dir output/dpo/train \\
        --merged output/dpo/train/dpo_pairs_merged_all.jsonl \\
        --stage1-base output/dpo/train/dpo_pairs_stage1_base_only.jsonl \\
        --stage2-commercial output/dpo/train/dpo_pairs_stage2_commercial_involved.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple


def iter_pair_records(paths: List[str]) -> Iterator[Tuple[str, Dict[str, Any]]]:
    for fp in paths:
        with open(fp, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield fp, json.loads(line)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{fp}:{line_num} JSON 解析失败: {e}") from e


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合并多社区 DPO jsonl，并导出阶段1(base-only)与阶段2(含commercial)划分"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="output/dpo/train",
        help="含 dpo_pairs_community_*.jsonl 的目录",
    )
    parser.add_argument(
        "--glob",
        type=str,
        default="dpo_pairs_community_*.jsonl",
        help="相对 input-dir 的 glob",
    )
    parser.add_argument(
        "--merged",
        type=str,
        default="output/dpo/train/dpo_pairs_merged_all.jsonl",
        help="全量合并输出路径",
    )
    parser.add_argument(
        "--stage1-base",
        type=str,
        default="output/dpo/train/dpo_pairs_stage1_base_only.jsonl",
        help="阶段1：双侧 profile_source=base",
    )
    parser.add_argument(
        "--stage2-commercial",
        type=str,
        default="output/dpo/train/dpo_pairs_stage2_commercial_involved.jsonl",
        help="阶段2：至少一侧为 commercial（与阶段1互斥）",
    )
    parser.add_argument(
        "--legacy-base-only",
        type=str,
        default="",
        help="可选：额外写一份与阶段1相同内容的文件（兼容旧路径，如 dpo_pairs_merged_base_only.jsonl）",
    )
    args = parser.parse_args()

    in_dir = Path(args.input_dir)
    pattern = str(in_dir / args.glob)
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"未找到匹配文件: {pattern}")

    merged_path = Path(args.merged)
    stage1_path = Path(args.stage1_base)
    stage2_path = Path(args.stage2_commercial)
    merged_path.parent.mkdir(parents=True, exist_ok=True)

    n_all = n_s1 = n_s2 = 0
    legacy_path = Path(args.legacy_base_only) if args.legacy_base_only.strip() else None
    if legacy_path:
        legacy_path.parent.mkdir(parents=True, exist_ok=True)

    with open(merged_path, "w", encoding="utf-8") as out_all, open(
        stage1_path, "w", encoding="utf-8"
    ) as out_s1, open(stage2_path, "w", encoding="utf-8") as out_s2:
        legacy_f = open(legacy_path, "w", encoding="utf-8") if legacy_path else None
        try:
            for _src, rec in iter_pair_records(files):
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                out_all.write(line)
                n_all += 1
                ch = (rec.get("chosen") or {}).get("profile_source")
                rj = (rec.get("rejected") or {}).get("profile_source")
                if ch == "base" and rj == "base":
                    out_s1.write(line)
                    n_s1 += 1
                    if legacy_f:
                        legacy_f.write(line)
                else:
                    out_s2.write(line)
                    n_s2 += 1
        finally:
            if legacy_f:
                legacy_f.close()

    if n_s1 + n_s2 != n_all:
        raise RuntimeError(f"内部错误: 阶段1({n_s1})+阶段2({n_s2}) != 全量({n_all})")

    def rel(p: str) -> str:
        cwd = os.getcwd()
        return os.path.relpath(p, start=cwd) if p.startswith(cwd) else p

    print(f"输入文件 ({len(files)}):")
    for fp in files:
        print(f"  {fp}")
    print(f"全量合并:     {n_all:5d} 条 -> {rel(str(merged_path))}")
    print(f"阶段1(base):  {n_s1:5d} 条 -> {rel(str(stage1_path))}")
    print(f"阶段2(商用):  {n_s2:5d} 条 -> {rel(str(stage2_path))}")
    if legacy_path:
        print(f"兼容副本:     {n_s1:5d} 条 -> {rel(str(legacy_path))}")


if __name__ == "__main__":
    main()
