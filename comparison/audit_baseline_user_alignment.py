#!/usr/bin/env python3
"""
检查各 method 的 baseline_chain jsonl 中，成功评估行的 user_id+community_id 集合是否一致。

用法（仓库根目录）：
  python3 -m comparison.audit_baseline_user_alignment \\
    --comparison-root output/comparison \\
    --stem baseline_chain_test_contiguous.jsonl \\
    --methods static_s0,prefix_refresh,incremental_persona,clasp_online,clasp_online_no_hist,history_only

默认跳过含 error 的行（与 resume 逻辑一致）；用 --include-error-rows 则按行数统计全部行。
退出码：0=各方法成功集合一致；1=不一致或缺少文件。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple


def serialize_user_key(rec: dict) -> str:
    return f"{rec.get('user_id')}\t{rec.get('community_id')}"


def load_keys(
    path: Path, *, skip_errors: bool
) -> Tuple[Set[str], Set[str], int, bool]:
    """返回 (用于比对的主键集合, error 键集合, json 坏行数, 文件存在)."""
    if not path.is_file():
        return set(), set(), 0, False
    ok: Set[str] = set()
    err: Set[str] = set()
    bad = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            k = serialize_user_key(o)
            if o.get("error"):
                err.add(k)
            else:
                ok.add(k)
    if skip_errors:
        return ok, err, bad, True
    primary = ok | err
    return primary, err, bad, True


def main() -> int:
    p = argparse.ArgumentParser(description="比对各 baseline method 输出中的用户键是否一致")
    p.add_argument(
        "--comparison-root",
        type=Path,
        default=Path("output/comparison"),
        help="run_baseline_comparison 的 --comparison-root",
    )
    p.add_argument(
        "--stem",
        required=True,
        help="各 method 子目录下同名 jsonl，如 baseline_chain_test_contiguous.jsonl",
    )
    p.add_argument(
        "--methods",
        required=True,
        help="逗号分隔 method 名，与输出子目录名一致",
    )
    p.add_argument(
        "--include-error-rows",
        action="store_true",
        help="不把 error 行排除在「成功集合」外（默认排除，与 resume 一致）",
    )
    args = p.parse_args()

    root: Path = args.comparison_root.resolve()
    stem: str = args.stem
    methods: List[str] = [m.strip() for m in args.methods.split(",") if m.strip()]
    skip_errors = not args.include_error_rows

    by_path: Dict[str, Path] = {m: root / m / stem for m in methods}
    data: Dict[str, Tuple[Set[str], Set[str], int, bool]] = {}
    for m in methods:
        path = by_path[m]
        data[m] = load_keys(path, skip_errors=skip_errors)

    print(f"[audit] comparison-root: {root}", flush=True)
    print(f"[audit] stem: {stem}", flush=True)
    print(f"[audit] skip_error_rows={skip_errors}", flush=True)

    missing = [m for m in methods if not data[m][3]]
    if missing:
        print(f"[audit] 缺失文件的方法: {missing}", flush=True)

    present = [m for m in methods if data[m][3]]
    if len(present) < 2:
        print("[audit] 至少两个已存在的结果文件才能比对。", flush=True)
        return 1

    ok_sets = {m: data[m][0] for m in present}
    ref = present[0]
    ref_set = ok_sets[ref]
    union = set().union(*ok_sets.values())
    inter = set.intersection(*ok_sets.values())

    print(f"\n[audit] 各方法成功用户数: " + ", ".join(f"{m}={len(ok_sets[m])}" for m in present), flush=True)
    print(f"[audit] 成功用户并集 |U|={len(union)}  交集 |∩|={len(inter)}", flush=True)

    mismatch = False
    for m in present:
        only_m = ok_sets[m] - ref_set
        miss_m = ref_set - ok_sets[m]
        if only_m or miss_m:
            mismatch = True
        err_n = len(data[m][1])
        bad = data[m][2]
        print(f"\n--- {m} --- path={by_path[m]}", flush=True)
        print(f"    ok={len(ok_sets[m])}  err_keys={err_n}  bad_json={bad}", flush=True)
        if only_m:
            ex = sorted(only_m)[:20]
            print(f"    相对 {ref} 多出 ({len(only_m)}), 示例: {ex}{' ...' if len(only_m)>20 else ''}", flush=True)
        if miss_m:
            ex = sorted(miss_m)[:20]
            print(f"    相对 {ref} 缺少 ({len(miss_m)}), 示例: {ex}{' ...' if len(miss_m)>20 else ''}", flush=True)

    if mismatch:
        print("\n[audit] 结论: **不一致** — 不同方法结果中的用户集合有差异，对比 F/L/Q 前请先对齐输入与续跑参数。", flush=True)
        return 1
    print("\n[audit] 结论: **一致** — 所列方法在成功行上的 user_id+community_id 集合相同。", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
