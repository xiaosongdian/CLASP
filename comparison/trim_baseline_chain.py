#!/usr/bin/env python3
"""
对 baseline_chain *.jsonl 按 mean_Q（或其它 key）做分位裁剪：

- **最小侧**去掉最低的 tail_fraction（默认 5%）：仅保留 mean_Q ≥ 第 5 百分位；
- **最大侧**去掉最高的 tail_fraction（默认 5%）：仅保留 mean_Q ≤ 第 95 百分位；
- 两侧合计约去掉 **2×tail_fraction** 的样本（有得分且参与分位的行）。

无 mean_Q、或含 error 的行不参与分位数，且原样保留写入输出。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _percentile(sorted_vals: List[float], p: float) -> float:
    """p ∈ [0, 100]，线性插值。"""
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    p = max(0.0, min(100.0, p))
    k = (len(sorted_vals) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    if f == c:
        return sorted_vals[f]
    return sorted_vals[f] + (k - f) * (sorted_vals[c] - sorted_vals[f])


def trim_rows(
    rows: List[Dict[str, Any]],
    *,
    tail_fraction: float = 0.05,
    key: str = "mean_Q",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    返回 (保留的行, 统计摘要)。
    tail_fraction: **每一侧**去掉的比例（默认 0.05 → 低侧 5%% + 高侧 5%%，合计约 10%%）。
    """
    t = max(0.0, min(0.45, float(tail_fraction)))
    low_p = t * 100.0
    high_p = (1.0 - t) * 100.0

    scored: List[Tuple[int, float]] = []
    for i, r in enumerate(rows):
        if r.get("error"):
            continue
        v = r.get(key)
        if v is None:
            continue
        try:
            scored.append((i, float(v)))
        except (TypeError, ValueError):
            continue

    if len(scored) < 3:
        return list(rows), {
            "trimmed": 0,
            "kept": len(rows),
            "reason": "too_few_scored_rows_skip_trim",
            "low_bound": None,
            "high_bound": None,
        }

    vals = sorted(x[1] for x in scored)
    low_b = _percentile(vals, low_p)
    high_b = _percentile(vals, high_p)

    drop_idx = set()
    for i, v in scored:
        if v < low_b or v > high_b:
            drop_idx.add(i)

    kept = [r for j, r in enumerate(rows) if j not in drop_idx]
    meta = {
        "tail_fraction_each_side": t,
        "approx_total_removed_fraction": 2.0 * t,
        "key": key,
        "low_percentile": low_p,
        "high_percentile": high_p,
        "low_bound": low_b,
        "high_bound": high_b,
        "scored_rows": len(scored),
        "dropped": len(drop_idx),
        "kept": len(kept),
        "total_in": len(rows),
    }
    return kept, meta


def main() -> None:
    p = argparse.ArgumentParser(description="baseline_chain JSONL 按 mean_Q 去极端（分位裁剪）")
    p.add_argument("input", type=Path, help="输入 jsonl")
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出 jsonl；默认 <stem>_trimmed.jsonl",
    )
    p.add_argument(
        "--trim-fraction",
        type=float,
        default=0.05,
        help="两端合计去掉的比例（默认 0.05：低 2.5%% + 高 2.5%%）",
    )
    p.add_argument(
        "--key",
        default="mean_Q",
        help="用于分位的字段（默认 mean_Q）",
    )
    args = p.parse_args()

    inp = args.input.resolve()
    if not inp.is_file():
        print(f"不存在: {inp}", file=sys.stderr)
        sys.exit(1)

    out = args.output
    if out is None:
        out = inp.parent / f"{inp.stem}_trimmed{inp.suffix}"
    else:
        out = out.resolve()

    rows: List[Dict[str, Any]] = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    kept, meta = trim_rows(rows, trim_fraction=args.trim_fraction, key=args.key)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_path = out.parent / f"{out.stem}_trim_meta.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({**meta, "input": str(inp), "output": str(out)}, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)
    print(f"已写入: {out}", flush=True)
    print(f"摘要: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
