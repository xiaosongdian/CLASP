#!/usr/bin/env python3
"""
从 baseline_chain_*.jsonl 可视化：

1. 按 **community_id** 分组，对每个社区分别绘制链上各步（target 窗口）跨用户的聚合 F、L、Q。
2. **不再**绘制三窗口 past/current/future Δ（已移除；专注 chain-step）。

**降噪（曲线更平滑、减轻极端用户拉扯）**：
- `--plot-trim-scope user`（默认）：按整条记录的 `mean_Q`（或 `--trim-key`）**整行**裁剪用户后再逐步聚合。
- `--plot-trim-scope step`：**每个链上窗口内**按当步 Q 的 `--step-trim-basis`（默认 `deviation` = Q−该步均值）做分位去尾，**同一批用户**在当步的 F/L/Q 一并剔除后再算 mean/median；不按全链 mean_Q 删行。
- `--plot-trim-each-tail P` 与 `--plot-trim-sides`：在 user 模式下裁剪用户行；在 step 模式下作用于**每步**分布。
- `--aggregate median`：每一步对各用户的 F/L/Q 取**中位数**而非算术平均。

建议（按步去极值）：`--plot-trim-scope step --plot-trim-each-tail 0.05 --plot-trim-sides lower --aggregate median`

支持 --watch 轮询 jsonl（单文件模式；网格模式会轮询所涉全部 jsonl）。

输出文件：**始终一个 PNG**（`--out`）；多个社区时在同一图中 **纵向子图**（每社区一行）。

**多方法对比（列为方法、行为社区）**：使用 `--comparison-root` + `--results-stem` + `--methods`，或重复传入 `--method-jsonl NAME=PATH`；**最右一列**为同社区下 **各方法链上 Q 随窗口变化** 的折线叠图（仅 Q，图例为方法名）。可加入 **`clasp_online_no_hist`** 与 **`clasp_online`** 并列对比（需先分别跑出对应子目录下的 jsonl）。

用法（仓库根目录）：

  python -m comparison.plot.visualize_baseline_chain \\
    output/comparison/clasp_online/baseline_chain_test.jsonl \\
    --out output/comparison/clasp_online/baseline_chain_viz.png

  python -m comparison.plot.visualize_baseline_chain \\
    output/comparison/clasp_online/baseline_chain_test.jsonl \\
    --out output/comparison/clasp_online/viz.png --watch 10

  # 网格：多基线 + clasp_online 与无观测历史消融并列，行=各 community
  python -m comparison.plot.visualize_baseline_chain \\
    --comparison-root output/comparison \\
    --results-stem baseline_chain_test_contiguous.jsonl \\
    --methods static_s0,prefix_refresh,incremental_persona,clasp_online,clasp_online_no_hist \\
    --out output/comparison/grid_contiguous.png
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

from comparison.window_chain_plot import aggregate_flq_by_step, filter_rows_for_plot_tails


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


def _group_by_community(rows: List[Dict[str, Any]]) -> Dict[Any, List[Dict[str, Any]]]:
    """community_id 缺失时归为 None。"""
    out: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        cid = r.get("community_id")
        out.setdefault(cid, []).append(r)
    return out


def _community_sort_key(cid: Any) -> Tuple:
    """稳定排序：None 最后。"""
    if cid is None:
        return (1, "")
    return (0, str(cid))


def _comm_slug(cid: Any) -> str:
    if cid is None:
        return "unknown"
    return str(cid).replace("/", "_").replace("\\", "_")


def _trim_plot_caption_from_meta(
    trim_meta: Dict[str, Any],
    trim_key: str,
) -> str:
    """子图脚注：trim 说明（与 filter_rows_for_plot_tails 的 meta 一致）。"""
    if trim_meta.get("trim_skipped"):
        return "trim_skipped(too_few_users)"
    if trim_meta.get("trim_disabled"):
        return ""
    side = trim_meta.get("trim_sides", "both")
    p = float(trim_meta.get("tail_fraction_each_side", 0)) * 100
    tk = trim_meta.get("trim_key", trim_key)
    kept = trim_meta.get("kept_plot_users")
    tin = trim_meta.get("total_in")
    if side == "lower":
        return f"trim lower {p:.0f}% by {tk} (kept {kept}/{tin} rows)"
    if side == "upper":
        return f"trim upper {p:.0f}% by {tk} (kept {kept}/{tin} rows)"
    return f"trim±{p:.0f}% by {tk} (kept {kept}/{tin} rows)"


def _trim_caption_short_per_method(
    trim_sides: str,
    tail_frac: float,
    trim_key: str,
) -> str:
    """网格图共用脚注（各 method 单独 trim，不写 kept）。"""
    p = float(tail_frac) * 100
    ts = str(trim_sides).lower().strip()
    if ts == "lower":
        return f"trim lower {p:.0f}% by {trim_key} (per-method)"
    if ts == "upper":
        return f"trim upper {p:.0f}% by {trim_key} (per-method)"
    return f"trim±{p:.0f}% by {trim_key} (per-method)"


def _trim_caption_step_per_method(
    trim_sides: str,
    tail_frac: float,
    basis: str,
) -> str:
    """网格图脚注：按步裁剪（每格内社区×方法各自聚合时生效）。"""
    p = float(tail_frac) * 100
    ts = str(trim_sides).lower().strip()
    b = str(basis).lower().strip()
    score = "Q−stepMean" if b == "deviation" else "Q"
    if ts == "lower":
        return f"per-step trim lower {p:.0f}% by {score} (per cell)"
    if ts == "upper":
        return f"per-step trim upper {p:.0f}% by {score} (per cell)"
    return f"per-step trim ±{p:.0f}% by {score} (per cell)"


def _baseline_flq_at_target(
    means: Dict[int, Dict[str, float]],
    steps_sorted: List[int],
    labels: List[str],
    target: str,
) -> Optional[Tuple[float, float, float]]:
    t = str(target).strip()
    for i, si in enumerate(steps_sorted):
        if i < len(labels) and str(labels[i]) == t:
            m = means[si]
            return (float(m["F"]), float(m["L"]), float(m["Q"]))
    return None


def _plot_chain_on_ax(
    ax: Any,
    *,
    means: Dict[int, Dict[str, float]],
    window_labels: List[str],
    n_chain_users: int,
    title_suffix: str,
    community_label: str,
    baseline_flq: Optional[Tuple[float, float, float]] = None,
    baseline_metric: str = "q",
    baseline_window_label: str = "W1",
    plot_caption: str = "",
    aggregate_label: str = "mean",
    show_xlabel: bool = True,
    ylabel_override: Optional[str] = None,
    skip_title: bool = False,
    short_title: bool = False,
    legend_fontsize: Optional[float] = None,
) -> None:
    """在单个 Axes 上画链上 F/L/Q（多社区大图中每行一个；网格模式中每格一个）。"""
    steps = sorted(means.keys())
    if not steps:
        ax.text(0.5, 0.5, "No chain step data", ha="center", va="center", transform=ax.transAxes)
        return
    xs = list(range(len(steps)))
    f_y = [means[s]["F"] for s in steps]
    l_y = [means[s]["L"] for s in steps]
    q_y = [means[s]["Q"] for s in steps]
    labs = [
        window_labels[i] if i < len(window_labels) else f"step{steps[i]}"
        for i in range(len(steps))
    ]
    ax.plot(xs, f_y, marker="o", color="C0", label="F")
    ax.plot(xs, l_y, marker="s", color="C1", label="L")
    ax.plot(xs, q_y, marker="^", color="C2", label="Q")
    if baseline_flq is not None:
        bF, bL, bQ = baseline_flq
        bm = baseline_metric.lower().strip()
        if bm in ("all", "f"):
            ax.axhline(
                bF,
                color="C0",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label=f"F baseline ({baseline_window_label})",
            )
        if bm in ("all", "l"):
            ax.axhline(
                bL,
                color="C1",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label=f"L baseline ({baseline_window_label})",
            )
        if bm in ("all", "q"):
            ax.axhline(
                bQ,
                color="C2",
                linestyle="--",
                linewidth=1.2,
                alpha=0.85,
                label=f"Q baseline ({baseline_window_label})",
            )
    ax.set_xticks(xs)
    ax.set_xticklabels(labs)
    if show_xlabel:
        ax.set_xlabel("Target window (chain step)")
    ylab_base = f"Score ({aggregate_label} over users)"
    if plot_caption:
        ylab_base += f" [{plot_caption}]"
    if ylabel_override is not None:
        ax.set_ylabel(ylabel_override)
    else:
        ax.set_ylabel(ylab_base)
    if skip_title:
        pass
    elif short_title:
        ax.set_title(f"n={n_chain_users}{title_suffix}", fontsize=9)
    else:
        ax.set_title(
            f"community={community_label} | Chain: {aggregate_label} F/L/Q "
            f"(n_users={n_chain_users}){title_suffix}"
        )
    ax.legend(loc="best", fontsize=float(legend_fontsize) if legend_fontsize is not None else 8.0)
    ax.grid(True, alpha=0.3)


def render_figure_by_community(
    *,
    panel_args: List[Dict[str, Any]],
    aggregate_label: str,
    title_suffix: str,
) -> Any:
    """
    单张图内多个子图：每个社区一行（纵向 stack）。
    panel_args: 每元素为传给 _plot_chain_on_ax 的关键字参数字典（已含 means 等）。
    """
    import matplotlib.pyplot as plt

    n = max(1, len(panel_args))
    row_h = 3.5
    fig, axes = plt.subplots(
        n,
        1,
        figsize=(9, max(4.0, row_h * n + 0.8)),
        squeeze=False,
    )
    ax_list = [axes[i, 0] for i in range(n)]
    for i, p in enumerate(panel_args):
        _plot_chain_on_ax(
            ax_list[i],
            show_xlabel=(i == n - 1),
            **p,
        )
    if not panel_args:
        ax_list[0].text(0.5, 0.5, "No data", ha="center", va="center", transform=ax_list[0].transAxes)
    supt = f"Chain F/L/Q by community ({aggregate_label}){title_suffix}"
    fig.suptitle(supt, fontsize=12, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    return fig


def run_multi_method_grid(
    method_paths: Dict[str, Path],
    out: Path,
    *,
    method: Optional[str] = None,
    title_suffix: str,
    baseline_window: str = "W1",
    baseline_metric: str = "q",
    disable_baseline: bool = False,
    plot_trim_each_tail: float = 0.0,
    trim_key: str = "mean_Q",
    trim_sides: str = "both",
    aggregate: str = "mean",
    plot_trim_scope: str = "user",
    step_trim_basis: str = "deviation",
) -> None:
    """
    多方法 × 多社区：行为社区、方法为列；每格画该 (community, method) 的链上 F/L/Q。
    最右侧一列：同一社区内所有方法的 **Q 随链上窗口** 折线（与其它列相同的横轴步长与标签）。
    method_paths: 有序字典，键顺序即列顺序。
    """
    import matplotlib.pyplot as plt

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
            plot_rows, trim_meta = filter_rows_for_plot_tails(
                valid,
                tail_fraction=float(plot_trim_each_tail),
                key=str(trim_key),
                trim_sides=str(trim_sides),
            )
            print(f"[viz] trim [{meth}]: {trim_meta}", flush=True)
        else:
            plot_rows = valid
            if is_step_trim:
                print(
                    f"[viz] per-step trim [{meth}]: P={plot_trim_each_tail} "
                    f"sides={trim_sides} basis={step_trim_basis} | 用户行={len(valid)}",
                    flush=True,
                )
        by_method_plot[meth] = plot_rows
        print(
            f"[viz] {meth} <- {path.name}: 有效={len(valid)} | 作图行={len(plot_rows)}",
            flush=True,
        )

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

    plot_caption = ""
    if is_user_trim:
        plot_caption = _trim_caption_short_per_method(
            trim_sides, float(plot_trim_each_tail), str(trim_key)
        )
    elif is_step_trim:
        plot_caption = _trim_caption_step_per_method(
            trim_sides, float(plot_trim_each_tail), str(step_trim_basis)
        )

    n_r = len(comm_ids_sorted)
    n_m = len(methods_order)
    if n_r == 0 or n_m == 0:
        print("[viz] 无社区或无法网格作图", flush=True)
        return

    n_c_grid = n_m + 1
    fig_w = max(14, 4.0 * n_m + 3.6)
    fig_h = max(8, 2.85 * n_r)
    fig, axes = plt.subplots(n_r, n_c_grid, figsize=(fig_w, fig_h), squeeze=False)
    for ii in range(1, n_r):
        axes[ii, n_m].sharey(axes[0, n_m])

    for i, cid in enumerate(comm_ids_sorted):
        ylab_left = f"community {_comm_slug(cid)}\n({agg})"
        for j, meth in enumerate(methods_order):
            ax = axes[i, j]
            sub = [r for r in by_method_plot[meth] if r.get("community_id") == cid]
            if not sub:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=9,
                )
                ax.set_xticks([])
                ax.set_yticks([])
                if i == 0:
                    ax.set_title(f"{meth}", fontsize=10)
                continue

            means, n_chain = aggregate_flq_by_step(
                sub, method=None, stat=agg, **_step_trim_kwargs
            )
            steps_sorted = sorted(means.keys())
            labels = _window_labels_from_rows(sub, steps_sorted)

            baseline_flq: Optional[Tuple[float, float, float]] = None
            if not disable_baseline and steps_sorted:
                baseline_flq = _baseline_flq_at_target(
                    means, steps_sorted, labels, baseline_window
                )

            ylb = ylab_left if j == 0 else ""
            _plot_chain_on_ax(
                ax,
                means=means,
                window_labels=labels,
                n_chain_users=n_chain,
                title_suffix=title_suffix,
                community_label=_comm_slug(cid),
                baseline_flq=None if disable_baseline else baseline_flq,
                baseline_metric=baseline_metric,
                baseline_window_label=baseline_window,
                plot_caption=plot_caption,
                aggregate_label=agg,
                show_xlabel=(i == n_r - 1),
                ylabel_override=ylb,
                skip_title=True,
                legend_fontsize=6,
            )
            if i == 0:
                ax.set_title(f"{meth}\n(n={n_chain})", fontsize=10)
            else:
                ax.set_title(f"n={n_chain}", fontsize=9)

        ax_q = axes[i, n_m]
        ref_steps: Optional[List[int]] = None
        win_labels: List[str] = []
        for meth in methods_order:
            sub_ref = [r for r in by_method_plot[meth] if r.get("community_id") == cid]
            if not sub_ref:
                continue
            m_ref, _ = aggregate_flq_by_step(
                sub_ref, method=None, stat=agg, **_step_trim_kwargs
            )
            st = sorted(m_ref.keys())
            if st:
                ref_steps = st
                win_labels = _window_labels_from_rows(sub_ref, st)
                break

        if not ref_steps:
            ax_q.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                transform=ax_q.transAxes,
                fontsize=9,
            )
            ax_q.set_xticks([])
            ax_q.set_yticks([])
            if i == 0:
                ax_q.set_title(f"Q only ({agg})", fontsize=10)
        else:
            xs = list(range(len(ref_steps)))
            for j, meth in enumerate(methods_order):
                sub_m = [r for r in by_method_plot[meth] if r.get("community_id") == cid]
                if not sub_m:
                    continue
                means_m, _ = aggregate_flq_by_step(
                    sub_m, method=None, stat=agg, **_step_trim_kwargs
                )
                q_y: List[float] = []
                for si in ref_steps:
                    if si in means_m:
                        q_y.append(float(means_m[si]["Q"]))
                    else:
                        q_y.append(float("nan"))
                ax_q.plot(
                    xs,
                    q_y,
                    marker="o",
                    color=f"C{j % 10}",
                    label=meth,
                    linewidth=1.6,
                    markersize=4,
                )
            ax_q.set_xticks(xs)
            lab_show = [
                win_labels[k] if k < len(win_labels) else f"step{ref_steps[k]}"
                for k in range(len(xs))
            ]
            ax_q.set_xticklabels(lab_show, fontsize=8)
            ax_q.grid(True, alpha=0.3)
            ax_q.tick_params(axis="y", labelsize=8)
            if i == n_r - 1:
                ax_q.set_xlabel("Target window (chain step)", fontsize=8)
            if i == 0:
                ax_q.set_title(f"Q ({agg}, all methods)", fontsize=10)
            if i > 0:
                ax_q.tick_params(labelleft=False)
            ax_q.legend(loc="best", fontsize=6)

    axes[0, n_m].set_ylabel(f"Q ({agg} over users)", fontsize=9)

    supt = f"Chain F/L/Q — methods × communities ({agg}){title_suffix}"
    fig.suptitle(supt, fontsize=12, y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=150)
    plt.close(fig)

    print(
        f"[viz] 已写入网格图 {n_r} 社区 × {n_m} 方法 + Q 对比列 -> {out_p}",
        flush=True,
    )


def run_once(
    jsonl: Path,
    out: Path,
    *,
    method: Optional[str],
    title_suffix: str,
    baseline_window: str = "W1",
    baseline_metric: str = "q",
    disable_baseline: bool = False,
    plot_trim_each_tail: float = 0.0,
    trim_key: str = "mean_Q",
    trim_sides: str = "both",
    aggregate: str = "mean",
    plot_trim_scope: str = "user",
    step_trim_basis: str = "deviation",
) -> None:
    import matplotlib.pyplot as plt

    rows = load_rows(jsonl, method=method)
    valid = [r for r in rows if not r.get("error")]
    n_before = len(valid)

    scope = str(plot_trim_scope).lower().strip()
    if scope not in ("user", "step"):
        scope = "user"
    is_step_trim = scope == "step" and float(plot_trim_each_tail) > 0
    is_user_trim = scope == "user" and float(plot_trim_each_tail) > 0

    plot_rows = valid
    trim_meta: Dict[str, Any] = {}
    if is_user_trim:
        plot_rows, trim_meta = filter_rows_for_plot_tails(
            valid,
            tail_fraction=float(plot_trim_each_tail),
            key=str(trim_key),
            trim_sides=str(trim_sides),
        )
        print(f"[viz] 降噪 trim: {trim_meta}", flush=True)
    elif is_step_trim:
        print(
            f"[viz] per-step trim: P={plot_trim_each_tail} sides={trim_sides} basis={step_trim_basis}",
            flush=True,
        )

    agg = aggregate.lower().strip()
    if agg not in ("mean", "median"):
        agg = "mean"

    _step_trim_kwargs = dict(
        step_trim_each_tail=float(plot_trim_each_tail) if is_step_trim else 0.0,
        step_trim_sides=str(trim_sides),
        step_trim_basis=str(step_trim_basis),
    )

    by_comm = _group_by_community(plot_rows)
    comm_ids = sorted(by_comm.keys(), key=_community_sort_key)
    if not comm_ids:
        print("[viz] 无有效用户，跳过作图", flush=True)
        return

    plot_caption = ""
    if is_user_trim:
        plot_caption = _trim_plot_caption_from_meta(trim_meta, trim_key)
    elif is_step_trim:
        plot_caption = _trim_caption_step_per_method(
            trim_sides, float(plot_trim_each_tail), str(step_trim_basis)
        )

    panel_args: List[Dict[str, Any]] = []
    for cid in comm_ids:
        sub = by_comm[cid]
        means, n_chain = aggregate_flq_by_step(
            sub, method=None, stat=agg, **_step_trim_kwargs
        )
        steps_sorted = sorted(means.keys())
        labels = _window_labels_from_rows(sub, steps_sorted)

        baseline_flq: Optional[Tuple[float, float, float]] = None
        if not disable_baseline and steps_sorted:
            baseline_flq = _baseline_flq_at_target(
                means, steps_sorted, labels, baseline_window
            )
            if baseline_flq is None:
                print(
                    f"[viz] community={cid}: 未找到 target_window={baseline_window}，跳过基线虚线",
                    flush=True,
                )

        comm_label = _comm_slug(cid)
        panel_args.append(
            {
                "means": means,
                "window_labels": labels,
                "n_chain_users": n_chain,
                "title_suffix": title_suffix,
                "community_label": comm_label,
                "baseline_flq": None if disable_baseline else baseline_flq,
                "baseline_metric": baseline_metric,
                "baseline_window_label": baseline_window,
                "plot_caption": plot_caption,
                "aggregate_label": agg,
            }
        )
        print(
            f"[viz] panel community={cid} | 用户={len(sub)} | 链上 n={n_chain}",
            flush=True,
        )
        if baseline_flq is not None:
            bF, bL, bQ = baseline_flq
            print(
                f"[viz]   Baseline {baseline_window}: F={bF:.4f} L={bL:.4f} Q={bQ:.4f}",
                flush=True,
            )

    fig = render_figure_by_community(
        panel_args=panel_args,
        aggregate_label=agg,
        title_suffix=title_suffix,
    )
    out_p = Path(out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=150)
    plt.close(fig)

    print(
        f"[viz] 已写入单图（{len(comm_ids)} 个社区子图）-> {out_p}",
        flush=True,
    )
    tail_note = (
        f"per-step trim P={plot_trim_each_tail}"
        if is_step_trim
        else f"trim 后作图用户={len(plot_rows)}"
    )
    print(
        f"[viz] 全局：读入有效用户={n_before} | {tail_note} | "
        f"社区数={len(comm_ids)}",
        flush=True,
    )


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
    p = argparse.ArgumentParser(
        description=(
            "baseline_chain jsonl：链上 F/L/Q；多社区则单 PNG 内纵向子图；"
            "或使用 --comparison-root / --method-jsonl 生成「社区×方法」网格图"
        )
    )
    p.add_argument(
        "jsonl",
        nargs="?",
        type=Path,
        default=None,
        help="单个 jsonl（与网格参数二选一），如 output/comparison/clasp_online/baseline_chain_test.jsonl",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 PNG（单文件；多社区时一张图内纵向多子图）",
    )
    p.add_argument(
        "--plot-trim-each-tail",
        type=float,
        default=0.0,
        metavar="P",
        help=(
            "作图前裁剪比例 P（0=关闭）。scope=user：按 trim-key 整行删用户；"
            "scope=step：每步内去尾后再聚合；双侧默认各 P 见 --plot-trim-sides"
        ),
    )
    p.add_argument(
        "--trim-key",
        default="mean_Q",
        choices=("mean_Q", "mean_F"),
        help="仅 plot-trim-scope=user：整行裁剪时排序依据的全局字段（默认 mean_Q）",
    )
    p.add_argument(
        "--plot-trim-sides",
        choices=("both", "lower", "upper"),
        default="both",
        help=(
            "与 --plot-trim-each-tail 配合：both=去掉最低与最高各 P；"
            "lower=只去掉最低 P（常用于去掉极差噪声，保留高分侧）；upper=只去掉最高 P"
        ),
    )
    p.add_argument(
        "--plot-trim-scope",
        choices=("user", "step"),
        default="user",
        help=(
            "user（默认）：按整条链 --trim-key 整行删用户后再逐步聚合；"
            "step：保留全部用户，在每个链上窗口内按 Q 与 --step-trim-basis 去尾后再聚合"
        ),
    )
    p.add_argument(
        "--step-trim-basis",
        choices=("deviation", "value"),
        default="deviation",
        help=(
            "仅 plot-trim-scope=step：deviation=按 Q−当步算术均值 的分位去尾；"
            "value=按当步 Q 数值本身的分位去尾"
        ),
    )
    p.add_argument(
        "--aggregate",
        choices=("mean", "median"),
        default="mean",
        help="逐步聚合方式：mean 算术平均；median 逐步中位数，抗离群值更强（噪声大可试 median）",
    )
    p.add_argument("--method", default=None, help="只统计某种 method（默认不过滤）")
    p.add_argument(
        "--baseline-window",
        default="W1",
        help="链上子图中作为水平虚线基准的 target 窗口（默认 W1，即预测 W1 步的跨用户均值）",
    )
    p.add_argument(
        "--baseline-metric",
        choices=("q", "f", "l", "all"),
        default="q",
        help="基线虚线数量：默认 q 仅一条（与 Q 曲线同色）；all 为 F/L/Q 各一条虚线",
    )
    p.add_argument(
        "--no-baseline",
        action="store_true",
        help="不绘制 W1（或其它 baseline-window）水平基线",
    )
    p.add_argument(
        "--watch",
        type=float,
        default=0.0,
        metavar="SEC",
        help="每 SEC 秒重新读取 jsonl 并覆盖保存 --out（网格模式会轮询全部输入文件）",
    )
    p.add_argument(
        "--comparison-root",
        type=Path,
        default=None,
        help="与 --results-stem、--methods 联用：每列为 comparison-root/<method>/<results-stem>",
    )
    p.add_argument(
        "--results-stem",
        type=str,
        default=None,
        help="相对于各方法子目录的文件名，如 baseline_chain_test_contiguous.jsonl",
    )
    p.add_argument(
        "--methods",
        type=str,
        default=None,
        help="逗号分隔的方法子目录名（与 run_baseline_comparison 输出目录名一致），如 static_s0,prefix_refresh,clasp_online,clasp_online_no_hist（列顺序与此一致）",
    )
    p.add_argument(
        "--method-jsonl",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="显式指定列：方法显示名与 jsonl 路径，可重复多次，顺序即列顺序",
    )
    args = p.parse_args()

    grid_pairs: List[Tuple[str, Path]] = []
    for item in args.method_jsonl or []:
        try:
            grid_pairs.append(_parse_method_jsonl(item))
        except argparse.ArgumentTypeError as e:
            print(f"--method-jsonl 无效: {item} ({e})", file=sys.stderr)
            sys.exit(2)

    cr = args.comparison_root
    stem = args.results_stem
    meth_csv = args.methods
    if (cr is not None) or stem or meth_csv:
        if cr is None or not stem or not meth_csv:
            print(
                "--comparison-root、--results-stem、--methods 必须三项同时给出",
                file=sys.stderr,
            )
            sys.exit(2)
        root = cr.resolve()
        for m in meth_csv.split(","):
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

    use_grid = len(method_paths) > 0

    if use_grid and args.jsonl is not None:
        print("请勿同时指定位置参数 jsonl 与网格选项（--method-jsonl / --comparison-root）", file=sys.stderr)
        sys.exit(2)

    if not use_grid:
        if args.jsonl is None:
            print("请提供 jsonl 路径，或使用网格模式（--method-jsonl 或 --comparison-root）", file=sys.stderr)
            sys.exit(2)
        jsonl = args.jsonl.resolve()
        if not jsonl.is_file():
            print(f"文件不存在: {jsonl}", file=sys.stderr)
            sys.exit(1)
    else:
        jsonl = None  # type: ignore
        for name, jp in method_paths.items():
            if not jp.is_file():
                print(f"网格输入缺失 [{name}]: {jp}", file=sys.stderr)
                sys.exit(1)

    out = args.out
    if out is None:
        if use_grid:
            first = next(iter(method_paths.values()))
            stem_hint = stem or first.name
            out = first.parent.parent / f"grid_{Path(stem_hint).stem}_viz.png"
        else:
            out = jsonl.parent / f"{jsonl.stem}_viz.png"  # type: ignore
    out = Path(out).resolve()

    title_suffix = ""
    if args.method:
        title_suffix = f" [{args.method}]"

    watch = float(args.watch or 0.0)
    kw = dict(
        method=args.method,
        title_suffix=title_suffix,
        baseline_window=str(args.baseline_window).strip() or "W1",
        baseline_metric=args.baseline_metric,
        disable_baseline=bool(args.no_baseline),
        plot_trim_each_tail=float(args.plot_trim_each_tail),
        trim_key=str(args.trim_key),
        trim_sides=str(args.plot_trim_sides),
        aggregate=str(args.aggregate),
        plot_trim_scope=str(args.plot_trim_scope),
        step_trim_basis=str(args.step_trim_basis),
    )

    def run_grid() -> None:
        run_multi_method_grid(method_paths, out, **kw)

    def run_single() -> None:
        run_once(jsonl, out, **kw)  # type: ignore

    if watch <= 0:
        if use_grid:
            run_grid()
        else:
            run_single()
        return

    print(f"[viz] watch={watch}s -> {out}", flush=True)
    try:
        while True:
            try:
                if use_grid:
                    missing = [n for n, jp in method_paths.items() if not jp.is_file()]
                    if missing:
                        print(f"[viz] 跳过本轮：缺失文件 {missing}", flush=True)
                    else:
                        run_grid()
                else:
                    run_single()
            except Exception as e:
                print(f"[viz] 本轮绘图失败: {e}", flush=True)
            time.sleep(watch)
    except KeyboardInterrupt:
        print("[viz] 已停止 watch", flush=True)


if __name__ == "__main__":
    main()
