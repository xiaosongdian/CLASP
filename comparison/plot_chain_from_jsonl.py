#!/usr/bin/env python3
"""从 run_baseline_comparison 输出的 JSONL 汇总各步 F/L/Q 并画折线图（无需再调 LLM）。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import PLOT_TRIM_EACH_TAIL as CFG_PLOT_TRIM

from comparison.window_chain_plot import (
    aggregate_flq_by_step,
    filter_rows_for_plot_tails,
    plot_flq_separate_figures,
    print_step_table,
)


def load_rows(path: Path, method: str | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if method and r.get("method") != method:
                continue
            rows.append(r)
    return rows


def main() -> None:
    p = argparse.ArgumentParser(description="从 baseline_chain JSONL 绘制窗口链 F/L/Q 折线")
    p.add_argument("jsonl", type=Path, help="baseline_chain_*.jsonl")
    p.add_argument("--method", default=None, help="只画某一种 method（默认 jsonl 内全部分别输出）")
    p.add_argument(
        "--plot",
        type=Path,
        required=True,
        help="基底路径，如 out/x.png -> 生成 out/x_F.png, x_L.png, x_Q.png",
    )
    p.add_argument("--title-prefix", default="", help="图标题前缀")
    p.add_argument(
        "--plot-trim-each-tail",
        type=float,
        default=None,
        metavar="P",
        help=(
            "作图聚合前按 mean_Q 裁剪尾部该比例；0=关闭；"
            "双侧默认最低/最高各 P；单侧见 --plot-trim-sides；"
            f"省略则用 config.PLOT_TRIM_EACH_TAIL（当前 {CFG_PLOT_TRIM}）；不改 jsonl"
        ),
    )
    p.add_argument(
        "--plot-trim-sides",
        choices=("both", "lower", "upper"),
        default="both",
        help=(
            "与 --plot-trim-each-tail 配合：both=最低与最高各去掉该比例；"
            "lower=只去掉最低比例；upper=只去掉最高比例"
        ),
    )
    p.add_argument(
        "--plot-trim-scope",
        choices=("user", "step"),
        default="user",
        help="user=按 mean_Q 整行裁剪；step=每步内裁剪后再聚合",
    )
    p.add_argument(
        "--step-trim-basis",
        choices=("deviation", "value"),
        default="deviation",
        help="plot-trim-scope=step 时：deviation=Q−当步均值；value=当步 Q 分位",
    )
    args = p.parse_args()
    trim_tail = CFG_PLOT_TRIM if args.plot_trim_each_tail is None else float(args.plot_trim_each_tail)

    path = args.jsonl.resolve()
    if not path.is_file():
        print(f"文件不存在: {path}", file=sys.stderr)
        sys.exit(1)

    all_rows = load_rows(path, method=None)
    methods = sorted({r.get("method") for r in all_rows if r.get("method")})
    if args.method:
        methods = [args.method] if args.method in methods else []
    if not methods:
        print("无匹配 method 的记录", file=sys.stderr)
        sys.exit(1)

    plot_base = args.plot
    for m in methods:
        sub = [r for r in all_rows if r.get("method") == m and not r.get("error")]
        if not sub:
            print(f"[skip] {m}: 无有效行", flush=True)
            continue
        sub_plot = sub
        if trim_tail and trim_tail > 0 and str(args.plot_trim_scope).lower() == "user":
            sub_plot, tmeta = filter_rows_for_plot_tails(
                sub,
                tail_fraction=trim_tail,
                key="mean_Q",
                trim_sides=str(args.plot_trim_sides),
            )
            print(
                f"=== {m} | 作图用户(去极值后)={len(sub_plot)}/{len(sub)} | "
                f"trim={tmeta} ===",
                flush=True,
            )
            if not sub_plot:
                sub_plot = sub
        elif trim_tail and trim_tail > 0 and str(args.plot_trim_scope).lower() == "step":
            print(
                f"=== {m} | per-step trim P={trim_tail} sides={args.plot_trim_sides} "
                f"basis={args.step_trim_basis} | users={len(sub)} ===",
                flush=True,
            )
        else:
            print(f"=== {m} | 作图用户={len(sub)}（未去极值）===", flush=True)
        st_tail = (
            float(trim_tail)
            if trim_tail and trim_tail > 0 and str(args.plot_trim_scope).lower() == "step"
            else 0.0
        )
        means, n_users = aggregate_flq_by_step(
            sub_plot,
            method=None,
            step_trim_each_tail=st_tail,
            step_trim_sides=str(args.plot_trim_sides),
            step_trim_basis=str(args.step_trim_basis),
        )
        print_step_table(means, label=m)
        steps_sorted = sorted(means.keys())
        labels: list[str] = []
        for si in steps_sorted:
            tw = None
            for st in sub_plot[0].get("steps") or []:
                if int(st.get("step_index", -1)) == si:
                    tw = st.get("target_window")
                    break
            labels.append(str(tw) if tw else f"step{si}")
        if len(methods) == 1:
            out = Path(plot_base)
        else:
            out = plot_base.parent / f"{plot_base.stem}_{m}{plot_base.suffix}"
        prefix = args.title_prefix.strip() if args.title_prefix else ""
        paths = plot_flq_separate_figures(
            means,
            out,
            title_prefix=f"{prefix + ' ' if prefix else ''}{m}".strip(),
            window_labels=labels,
            n_users=len(sub_plot),
        )
        for p in paths:
            print(f"已保存: {p}", flush=True)


if __name__ == "__main__":
    main()
