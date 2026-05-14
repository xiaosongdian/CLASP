#!/usr/bin/env python3
"""
从与 ``visualize_baseline_chain`` 网格模式相同的 jsonl 输入，导出链上聚合 F/L/Q 的 CSV（与图一致），不依赖 matplotlib。

默认写出两份 UTF-8 BOM 表：
- **宽表**：每行 (community, method, 窗口步)，列含 F、L、Q（与 ``--export-stats-csv`` 一致）；
- **长表**：每行一个指标值，列含 ``metric`` ∈ {F,L,Q}，便于透视/筛选（约 方法数×社区数×窗口数×3 行）。

用法（与生成 ``grid_contiguous.png`` 时相同的三项网格参数）：

  python3 -m comparison.plot.export_grid_chain_csv \\
    --comparison-root output/comparison \\
    --results-stem baseline_chain_test_contiguous.jsonl \\
    --methods static_s0,prefix_refresh,incremental_persona,clasp_online,clasp_online_no_hist \\
    --out-wide output/comparison/grid_contiguous_flq_wide.csv \\
    --out-long output/comparison/grid_contiguous_flq_long.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparison.plot.visualize_baseline_chain import (  # noqa: E402
    _baseline_flq_at_target,
    _comm_slug,
    _community_sort_key,
    _csv_float_cell,
    _window_labels_from_rows,
    _write_chain_step_stats_csv,
    filter_rows_for_plot_tails,
    load_rows,
)
from comparison.window_chain_plot import aggregate_flq_by_step  # noqa: E402


def _write_long_csv(path: Path, wide_rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "community_id",
        "method",
        "step_index",
        "step_order",
        "target_window_label",
        "metric",
        "value",
        "n_users",
        "aggregate",
        "baseline_window",
        "baseline_F",
        "baseline_L",
        "baseline_Q",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in wide_rows:
            base = {k: r.get(k, "") for k in fieldnames if k != "metric" and k != "value"}
            for metric in ("F", "L", "Q"):
                row = dict(base)
                row["metric"] = metric
                v = r.get(metric)
                row["value"] = _csv_float_cell(v) if v != "" and v is not None else ""
                for bk in ("baseline_F", "baseline_L", "baseline_Q"):
                    if row.get(bk) not in ("", None):
                        row[bk] = _csv_float_cell(row[bk])
                w.writerow(row)


def export_grid_chain_stats(
    method_paths: Dict[str, Path],
    *,
    method: Optional[str] = None,
    baseline_window: str = "W1",
    disable_baseline: bool = False,
    plot_trim_each_tail: float = 0.0,
    trim_key: str = "mean_Q",
    trim_sides: str = "both",
    aggregate: str = "mean",
    plot_trim_scope: str = "user",
    step_trim_basis: str = "deviation",
) -> List[Dict[str, Any]]:
    """返回与 ``run_multi_method_grid`` 中 ``step_csv_rows`` 相同结构的宽表行列表。"""
    agg = aggregate.lower().strip()
    if agg not in ("mean", "median"):
        agg = "mean"
    scope = str(plot_trim_scope).lower().strip()
    if scope not in ("user", "step"):
        scope = "user"
    is_step_trim = scope == "step" and float(plot_trim_each_tail) > 0
    is_user_trim = scope == "user" and float(plot_trim_each_tail) > 0

    by_method_plot: Dict[str, List[Dict[str, Any]]] = {}
    for meth, path in method_paths.items():
        rows = load_rows(path, method=method)
        valid = [r for r in rows if not r.get("error")]
        if is_user_trim:
            plot_rows, _ = filter_rows_for_plot_tails(
                valid,
                tail_fraction=float(plot_trim_each_tail),
                key=str(trim_key),
                trim_sides=str(trim_sides),
            )
        else:
            plot_rows = valid
        by_method_plot[meth] = plot_rows

    comm_ids: set = set()
    for rows in by_method_plot.values():
        for r in rows:
            comm_ids.add(r.get("community_id"))
    comm_ids_sorted = sorted(comm_ids, key=_community_sort_key)
    methods_order = list(method_paths.keys())

    _step_trim_kwargs = dict(
        step_trim_each_tail=float(plot_trim_each_tail) if is_step_trim else 0.0,
        step_trim_sides=str(trim_sides),
        step_trim_basis=str(step_trim_basis),
    )

    step_csv_rows: List[Dict[str, Any]] = []
    for cid in comm_ids_sorted:
        for meth in methods_order:
            sub = [r for r in by_method_plot[meth] if r.get("community_id") == cid]
            if not sub:
                continue
            means, _n_chain = aggregate_flq_by_step(
                sub, method=None, stat=agg, **_step_trim_kwargs
            )
            steps_sorted = sorted(means.keys())
            labels = _window_labels_from_rows(sub, steps_sorted)
            baseline_flq: Optional[Tuple[float, float, float]] = None
            if not disable_baseline and steps_sorted:
                baseline_flq = _baseline_flq_at_target(
                    means, steps_sorted, labels, baseline_window
                )
            if disable_baseline or baseline_flq is None:
                bw_s = ""
                bfp = blp = bqp = ""
            else:
                bFa, bLa, bQa = baseline_flq
                bw_s = str(baseline_window)
                bfp, blp, bqp = bFa, bLa, bQa
            for idx, skey in enumerate(steps_sorted):
                wl = labels[idx] if idx < len(labels) else f"step{skey}"
                mm = means[skey]
                step_csv_rows.append(
                    {
                        "community_id": _comm_slug(cid),
                        "method": meth,
                        "step_index": idx,
                        "step_order": int(skey),
                        "target_window_label": wl,
                        "n_users": int(mm.get("n", 0) or 0),
                        "F": float(mm["F"]),
                        "L": float(mm["L"]),
                        "Q": float(mm["Q"]),
                        "baseline_window": bw_s,
                        "baseline_F": bfp,
                        "baseline_L": blp,
                        "baseline_Q": bqp,
                        "aggregate": agg,
                    }
                )
    return step_csv_rows


def _parse_method_jsonl(spec: str) -> Tuple[str, Path]:
    s = str(spec).strip()
    if "=" not in s:
        raise argparse.ArgumentTypeError(
            "期望 NAME=PATH，例如 static_s0=output/comparison/static_s0/baseline_chain_test.jsonl"
        )
    name, path = s.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("NAME 不能为空")
    return name, Path(path.strip())


def main() -> None:
    p = argparse.ArgumentParser(description="网格 baseline_chain：导出与 grid 图一致的 F/L/Q CSV（无作图）")
    p.add_argument(
        "--comparison-root",
        type=Path,
        required=True,
        help="comparison 根目录",
    )
    p.add_argument(
        "--results-stem",
        type=str,
        required=True,
        help="各方法子目录下 jsonl 文件名，如 baseline_chain_test_contiguous.jsonl",
    )
    p.add_argument(
        "--methods",
        type=str,
        required=True,
        help="逗号分隔方法子目录名，列顺序与此一致",
    )
    p.add_argument(
        "--method-jsonl",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="可替代 --comparison-root 组合：显式 NAME=PATH，可重复",
    )
    p.add_argument(
        "--out-wide",
        type=Path,
        default=None,
        help="宽表输出路径（默认 output/comparison/<stem>_flq_wide.csv）",
    )
    p.add_argument(
        "--out-long",
        type=Path,
        default=None,
        help="长表输出路径（默认 output/comparison/<stem>_flq_long.csv）",
    )
    p.add_argument("--method", default=None, help="只读 jsonl 中该 method 字段的行（默认不过滤）")
    p.add_argument(
        "--baseline-window",
        default="W1",
        help="基线对应的 target 窗口标签（与作图一致）",
    )
    p.add_argument("--no-baseline", action="store_true", help="不填基线列")
    p.add_argument("--plot-trim-each-tail", type=float, default=0.0, metavar="P")
    p.add_argument("--trim-key", default="mean_Q", choices=("mean_Q", "mean_F"))
    p.add_argument("--plot-trim-sides", choices=("both", "lower", "upper"), default="both")
    p.add_argument("--plot-trim-scope", choices=("user", "step"), default="user")
    p.add_argument("--step-trim-basis", choices=("deviation", "value"), default="deviation")
    p.add_argument("--aggregate", choices=("mean", "median"), default="mean")
    args = p.parse_args()

    grid_pairs: List[Tuple[str, Path]] = []
    for item in args.method_jsonl or []:
        grid_pairs.append(_parse_method_jsonl(item))

    root = args.comparison_root.resolve()
    stem = args.results_stem
    if not grid_pairs:
        for m in str(args.methods).split(","):
            m = m.strip()
            if not m:
                continue
            grid_pairs.append((m, root / m / stem))

    method_paths: Dict[str, Path] = {}
    for name, path in grid_pairs:
        if name in method_paths:
            print(f"重复的方法名: {name}", file=sys.stderr)
            sys.exit(2)
        method_paths[name] = path.resolve()

    for name, jp in method_paths.items():
        if not jp.is_file():
            print(f"输入文件不存在 [{name}]: {jp}", file=sys.stderr)
            sys.exit(1)

    stem_hint = Path(stem).stem
    out_wide = args.out_wide
    out_long = args.out_long
    if out_wide is None:
        out_wide = root / f"{stem_hint}_flq_wide.csv"
    if out_long is None:
        out_long = root / f"{stem_hint}_flq_long.csv"

    rows = export_grid_chain_stats(
        method_paths,
        method=args.method,
        baseline_window=str(args.baseline_window).strip() or "W1",
        disable_baseline=bool(args.no_baseline),
        plot_trim_each_tail=float(args.plot_trim_each_tail),
        trim_key=str(args.trim_key),
        trim_sides=str(args.plot_trim_sides),
        aggregate=str(args.aggregate),
        plot_trim_scope=str(args.plot_trim_scope),
        step_trim_basis=str(args.step_trim_basis),
    )
    if not rows:
        print("无聚合行（检查 jsonl 是否含 steps 与 community_id）", file=sys.stderr)
        sys.exit(1)

    _write_chain_step_stats_csv(Path(out_wide), rows)
    _write_long_csv(Path(out_long), rows)
    n_long = len(rows) * 3
    print(f"[export] 宽表 {len(rows)} 行 -> {out_wide}", flush=True)
    print(f"[export] 长表 {n_long} 行（F/L/Q 各一行）-> {out_long}", flush=True)


if __name__ == "__main__":
    main()
