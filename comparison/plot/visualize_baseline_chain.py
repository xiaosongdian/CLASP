#!/usr/bin/env python3
"""
从 baseline_chain_*.jsonl 可视化：

1. 各步（按 target 窗口）跨用户的平均 F、L、Q（链上主指标）
2. 三窗口评估中 past / current / future 的 gain（ΔF、ΔL、ΔQ）的跨用户均值

支持 --watch 轮询 jsonl，便于实验跑分过程中「准实时」刷新图（写同一输出文件或带时间戳）。

用法（仓库根目录）：

  python -m comparison.plot.visualize_baseline_chain \\
    output/comparison/clasp_online/baseline_chain_test.jsonl \\
    --out output/comparison/clasp_online/baseline_chain_viz.png

  python -m comparison.plot.visualize_baseline_chain \\
    output/comparison/clasp_online/baseline_chain_test.jsonl \\
    --out output/comparison/clasp_online/viz.png --watch 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparison.window_chain_plot import aggregate_flq_by_step


def load_rows(path: Path, *, method: Optional[str]) -> List[Dict[str, Any]]:
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


def _window_labels_from_rows(rows: List[Dict[str, Any]], step_indices: List[int]) -> List[str]:
    labels: List[str] = []
    ref = next((r for r in rows if not r.get("error")), None)
    if not ref:
        return [f"step{s}" for s in step_indices]
    steps_ref = ref.get("steps") or []
    for si in step_indices:
        tw = None
        for st in steps_ref:
            if int(st.get("step_index", -1)) == si:
                tw = st.get("target_window")
                break
        labels.append(str(tw) if tw else f"step{si}")
    return labels


def _extract_three_window(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """每条用户记录通常仅在最后一步含 three_window_evaluation；取 step_index 最大的一条。"""
    best_si = -1
    best_tw: Optional[Dict[str, Any]] = None
    for st in record.get("steps") or []:
        tw = st.get("three_window_evaluation")
        if not tw:
            continue
        si = int(st.get("step_index", -1))
        if si >= best_si:
            best_si = si
            best_tw = tw
    return best_tw


def _gain_tuple(gain: Dict[str, Any]) -> Tuple[float, float, float]:
    if not gain:
        return (0.0, 0.0, 0.0)
    return (
        float(gain.get("ΔF", gain.get("dF", 0.0))),
        float(gain.get("ΔL", gain.get("dL", 0.0))),
        float(gain.get("ΔQ", gain.get("dQ", 0.0))),
    )


def aggregate_three_window_means(
    rows: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, float]], int]:
    """
    对每个用户取一条三窗口评估，对 past/current/future 的 gain 做跨用户平均。

    返回 ({"past": {"F","L","Q"}, ...}, n_users_used)
    """
    keys_long = ("past_window", "current_window", "future_window")
    keys_short = ("past", "current", "future")
    acc: Dict[str, Dict[str, List[float]]] = {
        k: {"F": [], "L": [], "Q": []} for k in keys_short
    }

    n = 0
    for r in rows:
        if r.get("error"):
            continue
        tw = _extract_three_window(r)
        if not tw:
            continue
        if not all(tw.get(k) for k in keys_long):
            continue
        for long_k, short_k in zip(keys_long, keys_short):
            block = tw[long_k]
            d_f, d_l, d_q = _gain_tuple((block or {}).get("gain") or {})
            acc[short_k]["F"].append(d_f)
            acc[short_k]["L"].append(d_l)
            acc[short_k]["Q"].append(d_q)
        n += 1

    out: Dict[str, Dict[str, float]] = {}
    for sk in keys_short:
        out[sk] = {}
        for m in ("F", "L", "Q"):
            xs = acc[sk][m]
            out[sk][m] = sum(xs) / len(xs) if xs else float("nan")

    return out, n


def render_figure(
    *,
    means: Dict[int, Dict[str, float]],
    window_labels: List[str],
    n_chain_users: int,
    tw_means: Dict[str, Dict[str, float]],
    n_tw_users: int,
    title_suffix: str,
) -> "Any":
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))

    # -------- 上图：链上各窗口平均 F/L/Q --------
    ax0 = axes[0]
    steps = sorted(means.keys())
    if steps:
        xs = list(range(len(steps)))
        f_y = [means[s]["F"] for s in steps]
        l_y = [means[s]["L"] for s in steps]
        q_y = [means[s]["Q"] for s in steps]
        labs = [
            window_labels[i] if i < len(window_labels) else f"step{steps[i]}"
            for i in range(len(steps))
        ]
        ax0.plot(xs, f_y, marker="o", label="F")
        ax0.plot(xs, l_y, marker="s", label="L")
        ax0.plot(xs, q_y, marker="^", label="Q")
        ax0.set_xticks(xs)
        ax0.set_xticklabels(labs)
        ax0.set_xlabel("Target window (chain step)")
        ax0.set_ylabel("Score (mean over users)")
        ax0.set_title(
            f"Chain: mean F / L / Q per target window (n_users={n_chain_users}){title_suffix}"
        )
        ax0.legend()
        ax0.grid(True, alpha=0.3)
    else:
        ax0.text(0.5, 0.5, "No chain step data", ha="center", va="center")

    # -------- 下图：三窗口 ΔF / ΔL / ΔQ（各时段一条曲线）--------
    ax1 = axes[1]
    period_labels = ("Past", "Current", "Future")
    period_keys = ("past", "current", "future")
    xs_b = list(range(3))
    has_tw = n_tw_users > 0

    if has_tw:
        d_f = [tw_means[k]["F"] for k in period_keys]
        d_l = [tw_means[k]["L"] for k in period_keys]
        d_q = [tw_means[k]["Q"] for k in period_keys]
        ax1.plot(xs_b, d_f, marker="o", label="mean ΔF")
        ax1.plot(xs_b, d_l, marker="s", label="mean ΔL")
        ax1.plot(xs_b, d_q, marker="^", label="mean ΔQ")
        ax1.set_xticks(xs_b)
        ax1.set_xticklabels(period_labels)
        ax1.axhline(0.0, color="gray", linestyle="--", linewidth=0.8)
        ax1.set_xlabel("Three-window evaluation period (gain: new profile vs old)")
        ax1.set_ylabel("Mean Δ over users")
        ax1.set_title(
            f"Three-window mean ΔF, ΔL, ΔQ (n_users={n_tw_users}){title_suffix}"
        )
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(
            0.5,
            0.5,
            "No three_window_evaluation\n(last chain step with profile change, etc.)",
            ha="center",
            va="center",
            transform=ax1.transAxes,
        )

    fig.tight_layout()
    return fig


def run_once(
    jsonl: Path,
    out: Path,
    *,
    method: Optional[str],
    title_suffix: str,
) -> None:
    import matplotlib.pyplot as plt

    rows = load_rows(jsonl, method=method)
    valid = [r for r in rows if not r.get("error")]
    means, n_chain = aggregate_flq_by_step(valid, method=None)
    steps_sorted = sorted(means.keys())
    labels = _window_labels_from_rows(valid, steps_sorted)
    tw_means, n_tw = aggregate_three_window_means(valid)

    fig = render_figure(
        means=means,
        window_labels=labels,
        n_chain_users=n_chain,
        tw_means=tw_means,
        n_tw_users=n_tw,
        title_suffix=title_suffix,
    )
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)

    print(
        f"[viz] 用户行={len(valid)} | 链上聚合 n={n_chain} | 三窗口聚合 n={n_tw} -> {out}",
        flush=True,
    )
    if tw_means and n_tw > 0:
        for pk in ("past", "current", "future"):
            m = tw_means[pk]
            print(
                f"  {pk}: ΔF={m['F']:.4f} ΔL={m['L']:.4f} ΔQ={m['Q']:.4f}",
                flush=True,
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description="baseline_chain jsonl：链上 F/L/Q + 三窗口 Δ 可视化"
    )
    p.add_argument(
        "jsonl",
        type=Path,
        help="如 output/comparison/clasp_online/baseline_chain_test.jsonl",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 PNG 路径（默认：与 jsonl 同目录，文件名加 _viz.png）",
    )
    p.add_argument("--method", default=None, help="只统计某种 method（默认不过滤）")
    p.add_argument(
        "--watch",
        type=float,
        default=0.0,
        metavar="SEC",
        help="每 SEC 秒重新读取 jsonl 并覆盖保存 --out（准实时监控）",
    )
    args = p.parse_args()

    jsonl = args.jsonl.resolve()
    if not jsonl.is_file():
        print(f"文件不存在: {jsonl}", file=sys.stderr)
        sys.exit(1)

    out = args.out
    if out is None:
        out = jsonl.parent / f"{jsonl.stem}_viz.png"
    else:
        out = Path(out).resolve()

    title_suffix = ""
    if args.method:
        title_suffix = f" [{args.method}]"

    watch = float(args.watch or 0.0)
    if watch <= 0:
        run_once(jsonl, out, method=args.method, title_suffix=title_suffix)
        return

    print(f"[viz] watch={watch}s -> {out}", flush=True)
    try:
        while True:
            try:
                run_once(jsonl, out, method=args.method, title_suffix=title_suffix)
            except Exception as e:
                print(f"[viz] 本轮绘图失败: {e}", flush=True)
            time.sleep(watch)
    except KeyboardInterrupt:
        print("[viz] 已停止 watch", flush=True)


if __name__ == "__main__":
    main()
