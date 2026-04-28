#!/usr/bin/env python3
"""窗口链评估结果：按步聚合 F/L/Q 并绘制折线图。"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _percentile_linear(sorted_vals: List[float], p: float) -> float:
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


def filter_rows_for_plot_tails(
    rows: List[Dict[str, Any]],
    *,
    tail_fraction: float = 0.05,
    key: str = "mean_Q",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    作图专用：去掉 key 最低与最高各 tail_fraction 比例的用户行（默认各 5%，不删改磁盘上的 jsonl）。
    无 key 或带 error 的行保留（不参与分位，照常参与后续聚合若仍在列表中——通常 error 行已在外部过滤）。
    """
    t = max(0.0, min(0.45, float(tail_fraction)))
    if t <= 0:
        return list(rows), {"trim_disabled": True, "kept": len(rows), "total_in": len(rows)}

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
            "trim_skipped": True,
            "reason": "too_few_scored",
            "kept": len(rows),
        }

    vals = sorted(x[1] for x in scored)
    low_b = _percentile_linear(vals, t * 100.0)
    high_b = _percentile_linear(vals, (1.0 - t) * 100.0)
    drop = {i for i, v in scored if v < low_b or v > high_b}
    kept = [r for j, r in enumerate(rows) if j not in drop]
    meta = {
        "tail_fraction_each_side": t,
        "trim_key": key,
        "low_bound": low_b,
        "high_bound": high_b,
        "dropped": len(drop),
        "kept_plot_users": len(kept),
        "total_in": len(rows),
    }
    return kept, meta


def aggregate_flq_by_step(
    rows: List[Dict[str, Any]],
    *,
    method: Optional[str] = None,
) -> Tuple[Dict[int, Dict[str, float]], int]:
    """
    返回 (step_index -> {"F","L","Q","n"}, 参与用户数)。
    method 为 None 时不按 method 过滤（应保证 rows 已只有一种 method）。
    """
    step_vals: Dict[int, Dict[str, List[float]]] = defaultdict(
        lambda: {"F": [], "L": [], "Q": []}
    )
    users = 0
    for r in rows:
        if r.get("error"):
            continue
        if method is not None and r.get("method") != method:
            continue
        users += 1
        for st in r.get("steps") or []:
            si = int(st.get("step_index", -1))
            if si < 0:
                continue
            step_vals[si]["F"].append(float(st["F"]))
            step_vals[si]["L"].append(float(st["L"]))
            step_vals[si]["Q"].append(float(st["Q"]))

    means: Dict[int, Dict[str, float]] = {}
    for si in sorted(step_vals.keys()):
        bucket = step_vals[si]
        n = len(bucket["Q"])
        if n == 0:
            continue
        means[si] = {
            "F": sum(bucket["F"]) / n,
            "L": sum(bucket["L"]) / n,
            "Q": sum(bucket["Q"]) / n,
            "n": float(n),
        }
    return means, users


def print_step_table(means: Dict[int, Dict[str, float]], *, label: str = "") -> None:
    pre = f"[{label}] " if label else ""
    print(f"{pre}各轮窗口（预测 Wk）平均 F / L / Q：", flush=True)
    for si in sorted(means.keys()):
        m = means[si]
        print(
            f"  step {si}: F={m['F']:.4f} L={m['L']:.4f} Q={m['Q']:.4f} (n={int(m['n'])})",
            flush=True,
        )


def plot_flq_lines(
    means: Dict[int, Dict[str, float]],
    out_path: Path,
    *,
    title: str = "Window chain: mean F / L / Q per step",
    window_labels: Optional[List[str]] = None,
) -> None:
    """将 aggregate_flq_by_step 的 means 画成折线图并保存。"""
    import matplotlib.pyplot as plt

    steps = sorted(means.keys())
    if not steps:
        raise ValueError("无有效 step 数据，无法作图")

    if window_labels is None:
        xtick = [f"step{s}" for s in steps]
    else:
        xtick = [window_labels[i] if i < len(window_labels) else f"step{s}" for i, s in enumerate(steps)]

    xs = list(range(len(steps)))
    f_y = [means[s]["F"] for s in steps]
    l_y = [means[s]["L"] for s in steps]
    q_y = [means[s]["Q"] for s in steps]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, f_y, marker="o", label="F (weighted F1)")
    ax.plot(xs, l_y, marker="s", label="L (semantic)")
    ax.plot(xs, q_y, marker="^", label="Q (combined)")
    ax.set_xticks(xs)
    ax.set_xticklabels(xtick)
    ax.set_xlabel("Target window")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_single_metric_line(
    means: Dict[int, Dict[str, float]],
    out_path: Path,
    metric: str,
    *,
    title: str,
    window_labels: Optional[List[str]] = None,
    ylabel: Optional[str] = None,
) -> None:
    """单指标折线图，metric 为 F / L / Q。"""
    import matplotlib.pyplot as plt

    if metric not in ("F", "L", "Q"):
        raise ValueError("metric must be F, L, or Q")
    steps = sorted(means.keys())
    if not steps:
        raise ValueError("无有效 step 数据，无法作图")

    if window_labels is None:
        xtick = [f"step{s}" for s in steps]
    else:
        xtick = [
            window_labels[i] if i < len(window_labels) else f"step{s}"
            for i, s in enumerate(steps)
        ]

    xs = list(range(len(steps)))
    y = [means[s][metric] for s in steps]

    default_ylabel = {
        "F": "F (weighted F1)",
        "L": "L (semantic)",
        "Q": "Q (combined)",
    }[metric]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(xs, y, marker="o", color="#1f77b4" if metric == "F" else ("#2ca02c" if metric == "L" else "#d62728"))
    ax.set_xticks(xs)
    ax.set_xticklabels(xtick)
    ax.set_xlabel("Target window")
    ax.set_ylabel(ylabel or default_ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_flq_separate_figures(
    means: Dict[int, Dict[str, float]],
    base_path: Path,
    *,
    title_prefix: str = "",
    window_labels: Optional[List[str]] = None,
    n_users: int = 0,
) -> List[Path]:
    """
    将 F、L、Q 各保存一张图。
    base_path 例如 output/clasp.png -> 写出 clasp_F.png, clasp_L.png, clasp_Q.png
    """
    base = Path(base_path)
    stem = base.stem
    parent = base.parent
    suffix = base.suffix if base.suffix else ".png"
    saved: List[Path] = []
    for metric in ("F", "L", "Q"):
        outp = parent / f"{stem}_{metric}{suffix}"
        sub = {
            "F": "Mean F (weighted F1)",
            "L": "Mean L (semantic cosine)",
            "Q": "Mean Q (combined)",
        }[metric]
        tp = (title_prefix or "").strip()
        title = f"{tp + ': ' if tp else ''}{sub} per step (n_users={n_users})"
        plot_single_metric_line(
            means,
            outp,
            metric,
            title=title,
            window_labels=window_labels,
        )
        saved.append(outp.resolve())
    return saved
