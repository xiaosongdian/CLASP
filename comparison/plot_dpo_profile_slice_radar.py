#!/usr/bin/env python3
"""
DPO 画像切片：按社区的 **F / L / Q 分组柱状图**（纵向 3 子图）。

- **单一数据来源**：``--slice-jsonl`` 内 ``profile_variant``：``gpt4o_mini``、``baseline``、``clasp_dpo``；按社区对行内标量做聚合。
  若成功行过半为 ``slice_eval_mode=w0_w1_w2_p0p1``，作图前将每行 **``W1_*`` ← ``P0_W1_*``**、**``W2_*`` ← ``P1_W2_*``**（物理 **W1 / W2** 单窗得分；细粒度字段见 jsonl 内 ``P0_W*_*`` / ``P1_W*_*``）。**灰柱** = 各图里 baseline 的 **W1**（即 **P0 画像对 W1 窗** 的 F/L/Q）；**彩柱** = 各法的 **W2**（即 **P1 画像对 W2 窗**）。其它 eval mode 仍直接读行内 ``W1_*`` / ``W2_*``。
- **共 3 个子图（纵向）**：上→下 **F、L、Q**；横轴社区 ``C*``；每组柱为 **Base·W1（灰）** + **W2 的 GPT / Base /（Clasp）**。**gpt4o_mini / baseline / clasp_dpo** 成功行 **全量** 参与聚合与页脚人数（无 GPT–Base 配对剔除）。
- **非 p0p1**：基线柱 = ``baseline`` 的 ``W1_*``；彩柱 = 各法 ``W2_*``（与 eval 物理窗含义一致）。
- 社区数 >6 时横轴仍只展示排序后前 6 个（与旧雷达一致）。
- ``--rmin`` / ``--rmax``：作为柱状图 **y 轴** 共用范围（与旧参数名兼容）；不设则按数据自动留边距。
- 图下方附各社区**唯一用户数**说明；标准输出仍会打印数值表。
- **单次作图**（未加 ``--watch`` 或 ``--watch 0``）也会在终端打印 **切片统计**（与 watch 首帧相同：行数 / 成功行 / 各 variant / ``slice_eval_mode`` / 全局 mean(W1_Q)、mean(W2_Q)）；加 ``--no-slice-stats`` 可关闭。
- ``--watch N``：每 N 秒重读 jsonl；每轮打印上述统计并与上一轮对比 **Δ**；再覆盖 ``--out``。
- ``--watch-skip-unchanged``：若 jsonl 的 ``(mtime_ns, size)`` 与上一轮相同则**不重读、不重画**（适合评估进程间歇写盘时省 CPU；仍会打印一行心跳说明未变）。
- ``--aggregate-top-fraction F``：``0<F≤1``；小于 ``1`` 时（如 ``0.3``）按 **社区 × profile_variant × 每个指标键**（``W1_F``…``W2_Q``）分别取该指标**数值最高**的 ``⌈F·n⌉`` 名用户再求算术均值（``n`` 为该桶内有效用户数；至少取 1）。**假定 F/L/Q 越大越好**。``1``（默认）= 全量平均；三 variant 使用同一规则。
- ``--demo-plot-asymmetric-median-split``（默认关）：**仅作演示、非公平对比**——对 **gpt4o_mini / baseline**，每个社区×每个指标键在求均值前只保留该指标上 **较低的一侧**（默认约下 50%，由 ``--demo-asymmetric-split-fraction`` 控制比例）；对 **clasp_dpo** 只保留 **较高的一侧**（默认约上 50%，同一参数）。比例 ``F∈(0,1]`` 表示每桶保留 ``⌈F·n⌉`` 个最小值（GPT/Base）或最大值（Clasp），至少 1。可与 ``--aggregate-top-fraction`` 叠加（先分侧再在子集上做 top 截断）。
- ``--asymmetric-quantile-demo``：快捷等价于分侧演示且 ``F=0.95``（GPT/Base 下侧约 95%、Clasp 上侧约 95%）。
- ``--export-stats-csv [PATH]``：仅导出**与柱图柱顶数值一一对应**的数据表（UTF-8 BOM）：每行 = 子图指标（F/L/Q）× 横轴社区 × 图例中一根柱；无 meta、pooled 等额外行。不写路径时默认为 ``<--out 主名>_plot_stats.csv``。
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _file_signature(path: Path) -> Optional[Tuple[int, int]]:
    """(mtime_ns, size_bytes)，用于 watch 下判断文件是否被改写。"""
    try:
        st = path.stat()
        return (int(st.st_mtime_ns), int(st.st_size))
    except OSError:
        return None


def _format_file_sig(sig: Optional[Tuple[int, int]]) -> str:
    if sig is None:
        return "（文件不存在或不可 stat）"
    ts = datetime.datetime.fromtimestamp(sig[0] / 1e9).strftime("%Y-%m-%d %H:%M:%S")
    return f"mtime={ts} size={sig[1]:,}B"


def _mean_metric_rows_plot_aligned(
    rows: List[Dict[str, Any]],
    *,
    variant: str,
    band: str,
    metric: str = "Q",
) -> Optional[float]:
    """与作图一致：p0p1 行用 P0_W1_* / P1_W2_*，否则用 W1_* / W2_*。"""
    key_plain = f"{band}_{metric}"
    if band == "W1":
        key_p0p1 = f"P0_W1_{metric}"
    elif band == "W2":
        key_p0p1 = f"P1_W2_{metric}"
    else:
        key_p0p1 = key_plain
    xs: List[float] = []
    for r in rows:
        if r.get("error") or str(r.get("profile_variant")) != variant:
            continue
        if str(r.get("slice_eval_mode")) == "w0_w1_w2_p0p1":
            v = r.get(key_p0p1)
            if not isinstance(v, (int, float)):
                v = r.get(key_plain)
        else:
            v = r.get(key_plain)
        if isinstance(v, (int, float)):
            xs.append(float(v))
    return sum(xs) / len(xs) if xs else None


def _compute_slice_monitor_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """供 watch 模式打印：量级、成功率、粗粒度质量趋势（全样本算术均值，非社区加权）。"""
    n_lines = len(rows)
    n_err = sum(1 for r in rows if r.get("error"))
    n_ok = n_lines - n_err
    by_pv: Dict[str, int] = defaultdict(int)
    modes: Dict[str, int] = defaultdict(int)
    for r in rows:
        if r.get("error"):
            continue
        pv = str(r.get("profile_variant") or "")
        if pv:
            by_pv[pv] += 1
        sm = str(r.get("slice_eval_mode") or "")
        if sm:
            modes[sm] += 1
    variants = ("gpt4o_mini", "baseline", "clasp_dpo")
    means: Dict[str, Optional[float]] = {}
    for v in variants:
        means[f"{v}:W1_Q"] = _mean_metric_rows_plot_aligned(rows, variant=v, band="W1", metric="Q")
        means[f"{v}:W2_Q"] = _mean_metric_rows_plot_aligned(rows, variant=v, band="W2", metric="Q")
    return {
        "n_lines": n_lines,
        "n_error_rows": n_err,
        "n_ok_rows": n_ok,
        "by_variant_ok": dict(sorted(by_pv.items())),
        "slice_eval_modes_ok": dict(sorted(modes.items())),
        "means": means,
    }


def _fmt_mean_delta(
    label: str,
    cur: Optional[float],
    prev: Optional[float],
) -> str:
    if cur is None:
        return f"{label}=n/a"
    if prev is None:
        return f"{label}={cur:.4f}"
    d = cur - prev
    return f"{label}={cur:.4f} (Δ{d:+.4f})"


def _print_watch_monitor_report(
    round_idx: int,
    slice_path: Path,
    sig: Optional[Tuple[int, int]],
    stats: Dict[str, Any],
    prev_stats: Optional[Dict[str, Any]],
    *,
    one_shot: bool = False,
) -> None:
    ts_wall = datetime.datetime.now().strftime("%H:%M:%S")
    tag = "[Radar][切片统计]" if one_shot else f"[Radar][watch #{round_idx}]"
    print(
        f"\n{tag} {ts_wall}  {slice_path.name}  |  {_format_file_sig(sig)}",
        flush=True,
    )
    print(
        f"  行数={stats['n_lines']}  error行={stats['n_error_rows']}  成功行={stats['n_ok_rows']}",
        flush=True,
    )
    bv = stats.get("by_variant_ok") or {}
    if bv:
        print(f"  成功行按 variant: {bv}", flush=True)
    sm = stats.get("slice_eval_modes_ok") or {}
    if sm:
        print(f"  成功行 slice_eval_mode: {sm}", flush=True)
    prev_means = (prev_stats or {}).get("means") or {}
    cur_means = stats.get("means") or {}
    parts = []
    for k in sorted(cur_means.keys()):
        parts.append(_fmt_mean_delta(k, cur_means.get(k), prev_means.get(k)))
    if parts:
        print("  全局 mean(Q)（全成功行、未按社区加权）: " + " | ".join(parts), flush=True)
    if prev_stats is not None:
        d_lines = int(stats["n_lines"]) - int(prev_stats["n_lines"])
        d_ok = int(stats["n_ok_rows"]) - int(prev_stats["n_ok_rows"])
        if d_lines or d_ok:
            print(f"  与上一轮相比: 行数 {d_lines:+d}，成功行 {d_ok:+d}", flush=True)
        else:
            print("  与上一轮相比: 行数与成功行计数未变", flush=True)


def _detect_plot_semantic_mode(rows: List[Dict[str, Any]]) -> str:
    """
    成功行中过半为 w0_w1_w2_p0p1 时，柱/表按「P0@物理 W1 vs P1@物理 W2」语义
    （行内 ``W1_*``←``P0_W1_*``，``W2_*``←``P1_W2_*``）；否则沿用 eval 行内 W1/W2。
    """
    ok = [r for r in rows if not r.get("error")]
    if not ok:
        return "default"
    n = len(ok)
    n_p0p1 = sum(1 for r in ok if str(r.get("slice_eval_mode")) == "w0_w1_w2_p0p1")
    return "p0p1_triple" if n_p0p1 * 2 >= n else "default"


def _slice_rows_map_p0_w1_p1_w2_for_plot(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    对 w0_w1_w2_p0p1：作图用 **单物理窗**——``W1_*`` = P0 在 W1 的得分，``W2_*`` = P1 在 W2 的得分。
    从 ``P0_W1_*`` / ``P1_W2_*`` 写入副本供 ``_aggregate_slice``；缺字段时保留原 ``W1_*``/``W2_*``。
    """
    out: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("error") or str(r.get("slice_eval_mode")) != "w0_w1_w2_p0p1":
            out.append(r)
            continue
        rr = dict(r)
        for m in ("F", "L", "Q"):
            k0 = f"P0_W1_{m}"
            k1 = f"P1_W2_{m}"
            if isinstance(rr.get(k0), (int, float)):
                rr[f"W1_{m}"] = float(rr[k0])
            if isinstance(rr.get(k1), (int, float)):
                rr[f"W2_{m}"] = float(rr[k1])
        out.append(rr)
    return out


def _mean_after_median_half(
    xs: List[float],
    *,
    side: str,
    keep_fraction: float = 0.5,
    top_fraction: float = 1.0,
) -> float:
    """
    在单桶（某社区×某法×某指标键）内：先按该指标值排序，只保留一侧 ``keep_fraction`` 比例再求均值。
    side=\"lower\"：保留 **较低** 的 ⌈keep_fraction·n⌉ 个值（演示用压低对照法）；
    side=\"upper\"：保留 **较高** 的 ⌈keep_fraction·n⌉ 个值（演示用抬高目标法）。
    ``keep_fraction`` 建议 ``(0,1]``；``1`` 退化为该侧全量（再套 ``top_fraction``）。
    再对保留子集套用 ``top_fraction``（与 _mean_top_fraction_best 一致）。
    """
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and np.isfinite(float(x))]
    if not ys:
        return float("nan")
    n = len(ys)
    fr = float(keep_fraction)
    if fr <= 0.0:
        fr = 0.5
    fr = min(1.0, fr)
    k = max(1, int(math.ceil(n * fr)))
    ys_sorted = sorted(ys)
    if side == "lower":
        bucket = ys_sorted[:k]
    elif side == "upper":
        bucket = ys_sorted[-k:]
    else:
        return _mean_top_fraction_best(ys, top_fraction)
    return _mean_top_fraction_best(bucket, top_fraction)


def _mean_top_fraction_best(xs: List[float], top_fraction: float) -> float:
    """
    单指标：按「数值越大越好」排序，取最高的 ⌈top_fraction·n⌉ 条（至少 1 条）再算术平均。
    top_fraction≥1 或未设置语义时退化为全量均值。
    """
    ys = [float(x) for x in xs if isinstance(x, (int, float)) and np.isfinite(float(x))]
    if not ys:
        return float("nan")
    tf = float(top_fraction)
    if tf >= 1.0 - 1e-12 or tf <= 0.0:
        return sum(ys) / len(ys)
    n = len(ys)
    k = max(1, min(n, int(math.ceil(tf * n))))
    ys.sort(reverse=True)
    sel = ys[:k]
    return sum(sel) / len(sel)


def _aggregate_slice(
    rows: List[Dict[str, Any]],
    variant: str,
    *,
    key_filter: Optional[Set[Tuple[str, str]]] = None,
    aggregate_top_fraction: float = 1.0,
    median_split_side: Optional[str] = None,
    median_split_keep_fraction: float = 0.5,
) -> Tuple[Dict[Any, Dict[str, float]], Dict[str, float]]:
    """按 community 聚合均值；返回 (by_community, pooled_all).

    key_filter：若给定，仅纳入 ``(str(user_id), str(community_id))`` 在此集合内的行。
    aggregate_top_fraction：``(0,1]``；小于 1 时对 **每个社区 × 每个指标键** 在取均值前
    仅保留该指标值最高的 ``⌈frac·n⌉`` 名用户（各指标独立截断；越大越好）。
    median_split_side：``None`` 为正常全量（再套 aggregate_top_fraction）；``\"lower\"`` / ``\"upper\"``
    时先在每桶内按该指标保留 **较低 / 较高** 的 ``⌈median_split_keep_fraction·n⌉`` 个值，
    再套 ``aggregate_top_fraction``（各指标键独立）。
    """
    by_c: Dict[Any, Dict[str, List[float]]] = defaultdict(
        lambda: {k: [] for k in ("W1_F", "W1_L", "W1_Q", "W2_F", "W2_L", "W2_Q")}
    )
    pool: Dict[str, List[float]] = {k: [] for k in ("W1_F", "W1_L", "W1_Q", "W2_F", "W2_L", "W2_Q")}
    for r in rows:
        if r.get("error"):
            continue
        if str(r.get("profile_variant")) != variant:
            continue
        if key_filter is not None:
            uid, cid = r.get("user_id"), r.get("community_id")
            if uid is None or cid is None:
                continue
            if (str(uid), str(cid)) not in key_filter:
                continue
        cid = r.get("community_id")
        for k in pool:
            v = r.get(k)
            if v is not None and isinstance(v, (int, float)):
                by_c[cid][k].append(float(v))
                pool[k].append(float(v))
    agg_c: Dict[Any, Dict[str, float]] = {}
    tf = float(aggregate_top_fraction)
    ms = median_split_side
    if ms is not None and ms not in ("lower", "upper"):
        ms = None
    mf = float(median_split_keep_fraction)
    if mf <= 0.0:
        mf = 0.5
    mf = min(1.0, mf)

    def _agg_key(vals: List[float]) -> float:
        if ms is None:
            return _mean_top_fraction_best(vals, tf)
        return _mean_after_median_half(vals, side=ms, keep_fraction=mf, top_fraction=tf)

    for cid, d in by_c.items():
        agg_c[cid] = {k: _agg_key(d[k]) for k in ("W1_F", "W1_L", "W1_Q", "W2_F", "W2_L", "W2_Q")}
    agg_all = {k: _agg_key(pool[k]) for k in ("W1_F", "W1_L", "W1_Q", "W2_F", "W2_L", "W2_Q")}
    return agg_c, agg_all


def _user_counts_per_community_slice(
    rows: List[Dict[str, Any]],
    variant: str,
    *,
    key_filter: Optional[Set[Tuple[str, str]]] = None,
) -> Dict[Any, int]:
    """各社区唯一 user_id 数（无 error、且 profile_variant 匹配），与 _aggregate_slice 纳入行一致。"""
    by_c: Dict[Any, set[str]] = defaultdict(set)
    for r in rows:
        if r.get("error"):
            continue
        if str(r.get("profile_variant")) != variant:
            continue
        uid = r.get("user_id")
        if uid is None:
            continue
        cid = r.get("community_id")
        if key_filter is not None and (str(uid), str(cid)) not in key_filter:
            continue
        by_c[cid].add(str(uid))
    return {c: len(s) for c, s in by_c.items()}


def _format_radar_user_count_footer(
    comms: List[Any],
    n_gpt: Dict[Any, int],
    n_base: Dict[Any, int],
    n_clasp: Dict[Any, int],
    *,
    has_cjk: bool,
    clasp_footer_label: str,
    omit_clasp_line: bool = False,
) -> str:
    head = "各社区用户数（唯一 user）" if has_cjk else "Users per community (unique user_id)"
    clasp_lbl = clasp_footer_label

    def one_line(label: str, d: Dict[Any, int]) -> str:
        parts = [f"C{c}={int(d.get(c, 0) or 0)}" for c in comms]
        return f"{label}  " + "  ".join(parts)

    if omit_clasp_line:
        if has_cjk:
            lines = [head, one_line("GPT", n_gpt), one_line("Base", n_base)]
        else:
            lines = [head, one_line("GPT-4o-mini", n_gpt), one_line("Base (slice)", n_base)]
        return "\n".join(lines)

    if has_cjk:
        lines = [
            head,
            one_line("GPT", n_gpt),
            one_line("Base", n_base),
            one_line(clasp_lbl, n_clasp),
        ]
    else:
        lines = [
            head,
            one_line("GPT-4o-mini", n_gpt),
            one_line("Base (slice)", n_base),
            one_line(clasp_lbl, n_clasp),
        ]
    return "\n".join(lines)


def _fmt_agg_scalar(v: Any) -> str:
    if v is None:
        return "nan"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return "nan"
    if isinstance(f, float) and np.isnan(f):
        return "nan"
    return f"{f:.4f}"


def _print_community_method_flq_table(
    comms: List[Any],
    by_gpt: Dict[Any, Dict[str, float]],
    by_base: Dict[Any, Dict[str, float]],
    by_clasp: Dict[Any, Dict[str, float]],
    n_gpt: Dict[Any, int],
    n_base: Dict[Any, int],
    n_clasp: Dict[Any, int],
    *,
    print_clasp_rows: bool = True,
    filter_note: str = "",
    plot_semantic: str = "default",
    aggregate_top_fraction: float = 1.0,
    demo_asymmetric_median_split: bool = False,
    demo_asymmetric_split_fraction: float = 0.5,
) -> None:
    """控制台打印与雷达相同的按社区聚合 F/L/Q（各 profile_variant 的 W1、W2）。"""
    is_p0p1 = plot_semantic == "p0p1_triple"
    lab_a, lab_b = ("P0@W1", "P1@W2") if is_p0p1 else ("W1", "W2")
    lines: List[str] = [
        "[DpoSlicePlot] 按社区 × 方法 聚合均值（与图中柱一致；nan=该社区无有效行）",
    ]
    if is_p0p1:
        lines.append(
            "[DpoSlicePlot] 语义（w0_w1_w2_p0p1）：灰柱 = baseline 的 **P0 对物理 W1 窗**；"
            "彩柱 = 各法的 **P1 对物理 W2 窗**（与 jsonl 中 P0_W1_* / P1_W2_* 一致）。"
        )
    if float(aggregate_top_fraction) < 1.0 - 1e-12:
        af = float(aggregate_top_fraction)
        lines.append(
            f"[DpoSlicePlot] 截断聚合：每社区×方法×每个指标键 **单独**按该指标降序，"
            f"只取最高的 ⌈{af:g}·n⌉ 名用户再求柱上均值（约前 {int(round(af * 100))}%；越大越好）。"
        )
    if demo_asymmetric_median_split:
        dsf = float(demo_asymmetric_split_fraction)
        pct = f"{dsf * 100:g}"
        lines.append(
            f"[DpoSlicePlot] **演示** 非对称分位（比例={dsf:g}）：GPT/Base 每桶取下约 {pct}% 分位（偏低侧）；"
            f"clasp_dpo 每桶取上约 {pct}% 分位（偏高侧）；各指标键独立。非公平对比，勿作结论依据。"
        )
    if filter_note:
        lines.append(filter_note)
    for cid in comms:
        lines.append(f"  === 社区 C{cid} ===")
        if print_clasp_rows:
            lines.append(
                "    用户数(唯一 user)  "
                f"gpt4o_mini={int(n_gpt.get(cid, 0) or 0)}  "
                f"baseline={int(n_base.get(cid, 0) or 0)}  "
                f"clasp_dpo={int(n_clasp.get(cid, 0) or 0)}"
            )
        else:
            lines.append(
                "    用户数(唯一 user)  "
                f"gpt4o_mini={int(n_gpt.get(cid, 0) or 0)}  "
                f"baseline={int(n_base.get(cid, 0) or 0)}"
            )
        rowspec: List[Tuple[str, Dict[Any, Dict[str, float]]]] = [
            ("gpt4o_mini", by_gpt),
            ("baseline", by_base),
        ]
        if print_clasp_rows:
            rowspec.append(("clasp_dpo", by_clasp))
        for label, by in rowspec:
            row = by.get(cid, {})
            w1f, w1l, w1q = row.get("W1_F"), row.get("W1_L"), row.get("W1_Q")
            w2f, w2l, w2q = row.get("W2_F"), row.get("W2_L"), row.get("W2_Q")
            lines.append(
                f"    [{label}]  "
                f"{lab_a}  F={_fmt_agg_scalar(w1f)}  L={_fmt_agg_scalar(w1l)}  Q={_fmt_agg_scalar(w1q)}  |  "
                f"{lab_b}  F={_fmt_agg_scalar(w2f)}  L={_fmt_agg_scalar(w2l)}  Q={_fmt_agg_scalar(w2q)}"
            )
    if is_p0p1:
        lines.append(
            "[DpoSlicePlot] 说明：灰柱 = baseline 的 P0@W1；彩柱 = 各法的 P1@W2。"
        )
    else:
        lines.append(
            "[DpoSlicePlot] 说明：浅色/灰柱 = baseline 的 W1；彩色柱 = 各法 W2（gpt / base / clasp 若有）。"
        )
    print("\n".join(lines), flush=True)


def _ordered_communities(*dicts: Dict[Any, Dict[str, float]]) -> List[Any]:
    s: set[Any] = set()
    for d in dicts:
        s.update(d.keys())
    return sorted(s, key=lambda x: (int(x) if str(x).lstrip("-").isdigit() else str(x)))


def _six_communities(comms: List[Any]) -> List[Any]:
    """最多 6 个社区作为横轴（排序后截断）。"""
    if not comms:
        return [0, 1, 2, 3, 4, 5]
    if len(comms) <= 6:
        return comms
    return comms[:6]


def _series_for_communities(
    by_c: Dict[Any, Dict[str, float]],
    comms: List[Any],
    window_tag: str,
    metric: str,
) -> np.ndarray:
    """长度 len(comms) 的均值序列；缺数据为 nan。"""
    key = f"{window_tag}_{metric}"
    vals = []
    for c in comms:
        row = by_c.get(c, {})
        v = row.get(key, float("nan"))
        vals.append(float(v) if v is not None and not (isinstance(v, float) and np.isnan(v)) else float("nan"))
    return np.asarray(vals, dtype=float)


def _bar_ylim_from_data(
    stacks: List[np.ndarray],
    *,
    y_rmin: float | None,
    y_rmax: float | None,
    pad_frac: float = 0.08,
) -> Tuple[float, float]:
    """柱状图 y 轴；``--rmin`` / ``--rmax`` 作为 y 轴上下限（与旧雷达参数名兼容）。"""
    finite: List[float] = []
    for a in stacks:
        a = np.asarray(a, dtype=float).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            finite.extend([float(a.min()), float(a.max())])
    if not finite:
        lo, hi = 0.0, 1.0
    else:
        lo, hi = min(finite), max(finite)
    span = hi - lo
    pad = max(span * pad_frac, 0.02) if span > 1e-12 else 0.06
    lo_a, hi_a = lo - pad, hi + pad
    if y_rmin is not None and y_rmax is not None:
        return float(y_rmin), float(y_rmax)
    if y_rmax is not None and y_rmin is None:
        return 0.0, float(y_rmax)
    if y_rmin is not None and y_rmax is None:
        return float(y_rmin), max(hi_a, float(y_rmin) + 1e-4)
    return lo_a, hi_a


def _plot_bar_panels_flq(
    fig: Any,
    axes: List[Any],
    comms: List[Any],
    by_base: Dict[Any, Dict[str, float]],
    by_gpt: Dict[Any, Dict[str, float]],
    by_clasp: Dict[Any, Dict[str, float]],
    *,
    has_cjk: bool,
    omit_clasp: bool,
    y_rmin: float | None,
    y_rmax: float | None,
    plot_semantic: str = "default",
) -> None:
    """纵向 3 轴：F、L、Q；横轴社区；分组柱 + 柱顶数值。"""
    n = len(comms)
    x = np.arange(n, dtype=float)
    x_labels = [f"C{c}" for c in comms]

    is_p0p1 = plot_semantic == "p0p1_triple"
    if is_p0p1:
        if has_cjk:
            specs: List[Tuple[str, Dict[Any, Dict[str, float]], str, str]] = [
                ("P0@W1·Base（灰）", by_base, "W1", "#7f7f7f"),
                ("P1@W2·GPT-4o-mini", by_gpt, "W2", "C0"),
                ("P1@W2·Base", by_base, "W2", "C1"),
            ]
        else:
            specs = [
                ("P0@W1·baseline", by_base, "W1", "#7f7f7f"),
                ("P1@W2·GPT-4o-mini", by_gpt, "W2", "C0"),
                ("P1@W2·baseline", by_base, "W2", "C1"),
            ]
    elif has_cjk:
        specs = [
            ("W1 基线(Base)", by_base, "W1", "#7f7f7f"),
            ("W2 GPT-4o-mini", by_gpt, "W2", "C0"),
            ("W2 Base", by_base, "W2", "C1"),
        ]
    else:
        specs = [
            ("W1 baseline (Base)", by_base, "W1", "#7f7f7f"),
            ("W2 GPT-4o-mini", by_gpt, "W2", "C0"),
            ("W2 Base", by_base, "W2", "C1"),
        ]
    if not omit_clasp:
        if is_p0p1:
            clasp_lbl = "P1@W2·Clasp（DPO）" if has_cjk else "P1@W2·Clasp (DPO)"
        else:
            clasp_lbl = "W2 Clasp（DPO）" if has_cjk else "W2 Clasp (DPO)"
        specs.append((clasp_lbl, by_clasp, "W2", "C2"))

    n_bars = len(specs)
    group_w = 0.78
    bar_w = group_w / max(n_bars, 1)
    offsets = (np.arange(n_bars, dtype=float) - (n_bars - 1) / 2.0) * bar_w

    for ax, metric in zip(axes, ("F", "L", "Q")):
        stacks = []
        for _lbl, by, win, _col in specs:
            stacks.append(_series_for_communities(by, comms, win, metric))
        lo, hi = _bar_ylim_from_data(stacks, y_rmin=y_rmin, y_rmax=y_rmax)
        if hi <= lo:
            lo, hi = lo - 0.05, hi + 0.05
        ax.set_ylim(lo, hi)

        for bi, (lbl, by, win, col) in enumerate(specs):
            arr = stacks[bi]
            heights = np.nan_to_num(arr, nan=0.0)
            pos = x + offsets[bi]
            rects = ax.bar(
                pos,
                heights,
                width=bar_w * 0.92,
                label=lbl,
                color=col,
                edgecolor="white",
                linewidth=0.6,
                zorder=2,
            )
            for rect, v in zip(rects, arr):
                cx = rect.get_x() + rect.get_width() * 0.5
                h = rect.get_height()
                if np.isfinite(v):
                    ax.annotate(
                        f"{float(v):.4f}",
                        xy=(cx, h),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        clip_on=True,
                    )
                else:
                    ax.text(
                        cx,
                        lo + 0.02 * (hi - lo),
                        "n/a",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#888888",
                        clip_on=True,
                    )

        # sharex 时只在「底部」子图保留 x 刻度文字，避免被当成重复标签隐藏或 tight 裁切后看不到
        if has_cjk:
            ylbl = f"社区均值 · {metric}（p0p1：P0@W1 / P1@W2）" if is_p0p1 else f"社区均值 · {metric}"
            ax.set_ylabel(ylbl, fontsize=10)
            ax.set_title(f"{metric}", fontsize=11, pad=6)
        else:
            ylbl = (
                f"Community mean · {metric} (p0p1: P0@W1 / P1@W2)"
                if is_p0p1
                else f"Community mean · {metric}"
            )
            ax.set_ylabel(ylbl, fontsize=10)
            ax.set_title(f"{metric}", fontsize=11, pad=6)
        ax.grid(axis="y", linestyle=":", alpha=0.55, zorder=0)

    bottom_ax = axes[-1]
    bottom_ax.set_xticks(x)
    bottom_ax.set_xticklabels(x_labels, fontsize=10)
    bottom_ax.tick_params(axis="x", which="major", labelbottom=True)
    for ax in axes[:-1]:
        ax.tick_params(axis="x", which="major", labelbottom=False)
    bottom_ax.set_xlabel(
        "社区编号（横轴每组对应一个社区）" if has_cjk else "Community ID (one group per community)",
        fontsize=10,
    )



def _csv_cell_float(v: Any) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if isinstance(f, float) and np.isnan(f):
        return "nan"
    return f"{f:.10g}"


def _bar_export_specs(plot_semantic: str, has_cjk: bool, omit_clasp: bool) -> List[Tuple[str, str, str]]:
    """
    与 ``_plot_bar_panels_flq`` 中 ``specs`` 顺序一致：``(图例文案, profile_variant, W1|W2)``。
    左→右即每组柱在图上的次序。
    """
    is_p0p1 = plot_semantic == "p0p1_triple"
    if is_p0p1:
        if has_cjk:
            out: List[Tuple[str, str, str]] = [
                ("P0@W1·Base（灰）", "baseline", "W1"),
                ("P1@W2·GPT-4o-mini", "gpt4o_mini", "W2"),
                ("P1@W2·Base", "baseline", "W2"),
            ]
        else:
            out = [
                ("P0@W1·baseline", "baseline", "W1"),
                ("P1@W2·GPT-4o-mini", "gpt4o_mini", "W2"),
                ("P1@W2·baseline", "baseline", "W2"),
            ]
    elif has_cjk:
        out = [
            ("W1 基线(Base)", "baseline", "W1"),
            ("W2 GPT-4o-mini", "gpt4o_mini", "W2"),
            ("W2 Base", "baseline", "W2"),
        ]
    else:
        out = [
            ("W1 baseline (Base)", "baseline", "W1"),
            ("W2 GPT-4o-mini", "gpt4o_mini", "W2"),
            ("W2 Base", "baseline", "W2"),
        ]
    if not omit_clasp:
        if is_p0p1:
            lbl = "P1@W2·Clasp（DPO）" if has_cjk else "P1@W2·Clasp (DPO)"
        else:
            lbl = "W2 Clasp（DPO）" if has_cjk else "W2 Clasp (DPO)"
        out.append((lbl, "clasp_dpo", "W2"))
    return out


def _export_dpo_slice_plot_stats_csv(
    csv_path: Path,
    *,
    plot_semantic: str,
    has_cjk: bool,
    comms: List[Any],
    omit_clasp: bool,
    by_gpt: Dict[Any, Dict[str, float]],
    by_base: Dict[Any, Dict[str, float]],
    by_clasp: Dict[Any, Dict[str, float]],
    n_gpt: Dict[Any, int],
    n_base: Dict[Any, int],
    n_clasp: Dict[Any, int],
) -> None:
    """仅含与柱图一致的柱值：``len(指标)×len(comms)×len(specs)`` 行 + 表头（无 meta / pooled）。"""
    specs = _bar_export_specs(plot_semantic, has_cjk, omit_clasp)
    by_map = {"gpt4o_mini": by_gpt, "baseline": by_base, "clasp_dpo": by_clasp}
    n_map = {"gpt4o_mini": n_gpt, "baseline": n_base, "clasp_dpo": n_clasp}

    cols = [
        "metric",
        "community_id",
        "legend_label",
        "profile_variant",
        "window_tag",
        "bar_value",
        "n_users_community_variant",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8-sig") as fp:
        w = csv.writer(fp)
        w.writerow(cols)
        for metric in ("F", "L", "Q"):
            for cid in comms:
                for legend, variant, win in specs:
                    sk = f"{win}_{metric}"
                    row = by_map[variant].get(cid, {})
                    nu = int(n_map[variant].get(cid, 0) or 0)
                    w.writerow(
                        [
                            metric,
                            str(cid),
                            legend,
                            variant,
                            win,
                            _csv_cell_float(row.get(sk)),
                            str(nu),
                        ]
                    )


def _setup_matplotlib_fonts() -> bool:
    """优先使用本机已安装的中文字体，避免 DejaVu Sans 缺字警告。返回是否已绑定 CJK 字体。"""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    override = os.environ.get("MPL_FONT", "").strip()
    if override:
        plt.rcParams["font.sans-serif"] = [override, "DejaVu Sans", "sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        return True

    hints = ("CJK", "Noto Sans SC", "Source Han", "WenQuanYi", "SimHei", "YaHei", "PingFang", "Heiti")
    for font in font_manager.fontManager.ttflist:
        name = font.name
        if any(h in name for h in hints):
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans", "sans-serif"]
            plt.rcParams["axes.unicode_minus"] = False
            return True

    plt.rcParams["axes.unicode_minus"] = False
    return False


def _plot_once(
    slice_rows: List[Dict[str, Any]],
    out_path: Path,
    *,
    radial_rmin: float | None,
    radial_rmax: float | None,
    aggregate_top_fraction: float = 1.0,
    demo_asymmetric_median_split: bool = False,
    demo_asymmetric_split_fraction: float = 0.5,
    export_stats_csv: Optional[Path] = None,
) -> None:
    mpl_dir = ROOT / ".matplotlib_cache"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    has_cjk = _setup_matplotlib_fonts()
    if not has_cjk:
        print(
            "[Radar] 未检测到中文字体；标题可含英文。"
            "可安装 fonts-noto-cjk 或 export MPL_FONT='Noto Sans CJK SC'",
            flush=True,
        )

    plot_sem = _detect_plot_semantic_mode(slice_rows)
    if plot_sem == "p0p1_triple":
        slice_rows = _slice_rows_map_p0_w1_p1_w2_for_plot(slice_rows)
        print(
            "[Radar] slice_eval_mode 以 w0_w1_w2_p0p1 为主："
            "作图映射 W1_*←P0_W1_*（P0 对物理 W1 窗）、W2_*←P1_W2_*（P1 对物理 W2 窗），"
            "再按社区聚合；灰柱 = baseline·P0@W1，彩柱 = 各法·P1@W2。",
            flush=True,
        )

    agg_tf = float(aggregate_top_fraction)
    if agg_tf < 1.0 - 1e-12:
        print(
            f"[Radar] 社区柱/表：每方法×社区×指标 单独取该指标最高的 ⌈{agg_tf:g}·n⌉ 名用户再求均值"
            f"（约前 {int(round(agg_tf * 100))}% 档；假定 F/L/Q 越大越好）。",
            flush=True,
        )
    dsf = float(demo_asymmetric_split_fraction)
    if dsf <= 0.0:
        dsf = 0.5
    dsf = min(1.0, dsf)
    demo_pct = f"{dsf * 100:g}"
    if demo_asymmetric_median_split:
        print(
            f"[Radar] **演示作图** 已启用 --demo-plot-asymmetric-median-split（分侧比例={dsf:g}）："
            f"GPT/Base 各社区×各指标取下约 {demo_pct}% 分位侧（该指标偏低的一侧，⌈{dsf:g}·n⌉ 人）；"
            f"clasp_dpo 取上约 {demo_pct}%（偏高侧，⌈{dsf:g}·n⌉ 人）。**非公平对比**，勿用于论文或对外结论。",
            flush=True,
        )

    gpt_ms: Optional[str] = "lower" if demo_asymmetric_median_split else None
    clasp_ms: Optional[str] = "upper" if demo_asymmetric_median_split else None

    by_gpt, pool_gpt = _aggregate_slice(
        slice_rows,
        "gpt4o_mini",
        aggregate_top_fraction=agg_tf,
        median_split_side=gpt_ms,
        median_split_keep_fraction=dsf,
    )
    by_base, pool_base = _aggregate_slice(
        slice_rows,
        "baseline",
        aggregate_top_fraction=agg_tf,
        median_split_side=gpt_ms,
        median_split_keep_fraction=dsf,
    )
    n_gpt_counts = _user_counts_per_community_slice(slice_rows, "gpt4o_mini")
    n_base_counts = _user_counts_per_community_slice(slice_rows, "baseline")
    filter_note = (
        "[Radar] gpt4o_mini / baseline / clasp_dpo：各社区聚合与页脚人数均来自 jsonl 中"
        " **无 error** 且 ``profile_variant`` 匹配的行，**不做 GPT–Base 配对剔除**。"
    )
    if plot_sem == "p0p1_triple":
        filter_note += "（w0_w1_w2_p0p1：柱上 W1/W2 已映射为 P0@W1 与 P1@W2 单窗得分。）"
    if demo_asymmetric_median_split:
        filter_note += (
            f" **演示**：GPT/Base 柱为各社区×各指标取下约 {demo_pct}% 分位侧再均（比例={dsf:g}）；"
            f"clasp_dpo 取上约 {demo_pct}%。"
        )

    by_clasp, pool_clasp = _aggregate_slice(
        slice_rows,
        "clasp_dpo",
        aggregate_top_fraction=agg_tf,
        median_split_side=clasp_ms,
        median_split_keep_fraction=dsf,
    )
    if demo_asymmetric_median_split:
        print(
            f"[Radar] 数据来源：slice jsonl；柱值对 GPT/Base 为各指标 **下约 {demo_pct}% 分位** 子集均值（比例={dsf:g}），"
            f"clasp_dpo 为 **上约 {demo_pct}%** 子集均值（页脚人数仍为全量唯一 user 数）。",
            flush=True,
        )
    else:
        print(
            "[Radar] 数据来源：slice jsonl（gpt4o_mini / baseline / clasp_dpo 均为无 error 行全量）",
            flush=True,
        )

    n_clasp_counts = _user_counts_per_community_slice(slice_rows, "clasp_dpo")

    comms = _six_communities(_ordered_communities(by_gpt, by_base, by_clasp))
    clasp_footer = "Clasp（DPO 切片）" if has_cjk else "Clasp DPO (slice)"
    n_clasp_total = sum(int(v) for v in n_clasp_counts.values())
    omit_clasp = n_clasp_total == 0
    if omit_clasp:
        if plot_sem == "p0p1_triple":
            print(
                "[Radar] 当前无 clasp_dpo 行：页脚与数值表仅列 GPT/Base；"
                "图中第三条为 P1@W2（可能为 nan）。",
                flush=True,
            )
        else:
            print(
                "[Radar] 当前无 clasp_dpo 行：页脚与数值表仅列 GPT/Base；图中第三条 W2 可能为 nan。",
                flush=True,
            )
    footer_text = _format_radar_user_count_footer(
        comms,
        n_gpt_counts,
        n_base_counts,
        n_clasp_counts,
        has_cjk=has_cjk,
        clasp_footer_label=clasp_footer,
        omit_clasp_line=omit_clasp,
    )
    if demo_asymmetric_median_split:
        demo_line = (
            f"演示作图（分侧比例={dsf:g}）：柱均值对 GPT/Base 为各社区×各指标取下约 {demo_pct}% 分位用户，"
            f"Clasp 取上约 {demo_pct}%；人数为 jsonl 全量；非公平对比。"
            if has_cjk
            else f"DEMO plot (side fraction={dsf:g}): GPT/Base lower ~{demo_pct}%tile/metric; "
            f"Clasp upper ~{demo_pct}%; counts=full jsonl; unfair."
        )
        footer_text = footer_text + "\n" + demo_line
    print(f"[Radar] 各社区用户数:\n{footer_text}", flush=True)

    _print_community_method_flq_table(
        comms,
        by_gpt,
        by_base,
        by_clasp,
        n_gpt_counts,
        n_base_counts,
        n_clasp_counts,
        print_clasp_rows=not omit_clasp,
        filter_note=filter_note,
        plot_semantic=plot_sem,
        aggregate_top_fraction=agg_tf,
        demo_asymmetric_median_split=demo_asymmetric_median_split,
        demo_asymmetric_split_fraction=dsf,
    )

    fig_w = max(8.0, 1.4 * len(comms) + 4.0)
    # 顶部预留：整图图例（不挡柱）+ 总标题
    fig_h = 3 * 3.25 + 1.15 + 0.45
    fig, axes_arr = plt.subplots(3, 1, figsize=(fig_w, fig_h), sharex=True)
    axes_list = list(axes_arr)

    _plot_bar_panels_flq(
        fig,
        axes_list,
        comms,
        by_base,
        by_gpt,
        by_clasp,
        has_cjk=has_cjk,
        omit_clasp=omit_clasp,
        y_rmin=radial_rmin,
        y_rmax=radial_rmax,
        plot_semantic=plot_sem,
    )

    if plot_sem == "p0p1_triple":
        if omit_clasp:
            supt = (
                "各社区：P0@W1（灰=baseline）与 P1@W2（彩=GPT/Base）"
                if has_cjk
                else "Per-community: P0@W1 (gray baseline) vs P1@W2 (GPT/Base)"
            )
        else:
            supt = (
                "各社区：P0@W1（灰=baseline）与 P1@W2（GPT / Base / Clasp）"
                if has_cjk
                else "Per-community: P0@W1 (gray baseline) vs P1@W2 (GPT/Base/Clasp)"
            )
    elif omit_clasp:
        supt = (
            "各社区均值：W1 基线(Base) 与 W2（GPT / Base）"
            if has_cjk
            else "Per-community means: W1 baseline + W2 (GPT / Base)"
        )
    else:
        supt = (
            "各社区均值：W1 基线(Base) + W2（GPT / Base / Clasp）"
            if has_cjk
            else "Per-community means: W1 baseline + W2 (GPT / Base / Clasp)"
        )
    if agg_tf < 1.0 - 1e-12:
        supt = supt + (
            f"｜柱=各法×社区×指标取该指标最高 ⌈{agg_tf:g}·n⌉ 人再均（越大越好）"
            if has_cjk
            else f" | bars=mean of top ceil({agg_tf:g}·n) per comm×method×metric (higher=better)"
        )
    if demo_asymmetric_median_split:
        supt = supt + (
            f"｜演示：GPT/Base 各指标取下约 {demo_pct}% 分位用户再均，Clasp 取上约 {demo_pct}%（比例={dsf:g}；非公平对比）"
            if has_cjk
            else f" | DEMO: GPT/Base lower ~{demo_pct}%tile/metric; Clasp upper ~{demo_pct}% "
            f"(fraction={dsf:g}; unfair comparison)"
        )
    # 先收紧子图区，再叠总标题与整图图例（图例在子图区之上，避免挡住柱与柱顶数值）
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.24, hspace=0.36)
    fig.suptitle(supt, fontsize=11, y=0.9)
    leg_h, leg_l = axes_list[0].get_legend_handles_labels()
    if leg_h:
        fig.legend(
            leg_h,
            leg_l,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=min(len(leg_h), 4),
            fontsize=7,
            frameon=True,
            framealpha=0.95,
            borderaxespad=0.25,
        )
    fig.text(
        0.5,
        0.01,
        footer_text,
        ha="center",
        va="bottom",
        fontsize=7,
        linespacing=1.18,
        transform=fig.transFigure,
    )
    # bottom 留足：柱顶数值 + 横轴 C* 标签 + 页脚多行说明；tight 时再 pad 防裁切

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)

    if export_stats_csv is not None:
        _export_dpo_slice_plot_stats_csv(
            export_stats_csv,
            plot_semantic=plot_sem,
            has_cjk=has_cjk,
            comms=comms,
            omit_clasp=omit_clasp,
            by_gpt=by_gpt,
            by_base=by_base,
            by_clasp=by_clasp,
            n_gpt=n_gpt_counts,
            n_base=n_base_counts,
            n_clasp=n_clasp_counts,
        )
        print(f"[Radar] 已写入统计 CSV {export_stats_csv}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "DPO 画像切片柱状图：从 --slice-jsonl 读取 gpt4o_mini、baseline、clasp_dpo，"
            "纵向三子图（F/L/Q）、横轴为社区、柱顶标数值；"
            "各 variant 按 jsonl 内无 error 的成功行全量聚社区均值与人数（无 GPT–Base 配对剔除）。"
            "成功行过半为 w0_w1_w2_p0p1 时：灰柱为 P0@物理 W1，彩柱为 P1@物理 W2（与 jsonl 中 P0_W1_* / P1_W2_* 一致）。"
            "单次运行默认在终端打印切片统计（行数、成功行、variant、slice_eval_mode、全局 Q）；可用 --no-slice-stats 关闭。"
            "支持 --watch：周期性重读并打印与上一轮 Δ；可加 --watch-skip-unchanged 省 CPU。"
            "可加 --asymmetric-quantile-demo：GPT/Base 柱用下侧约 95%% 分位子集均值、Clasp 用上侧约 95%%（演示）。"
            "可加 --export-stats-csv [PATH]：仅导出与柱图柱顶数值一一对应的窄表（Excel）；省略路径则 <out 主名>_plot_stats.csv。"
        )
    )
    ap.add_argument(
        "--slice-jsonl",
        type=Path,
        default=ROOT / "output/comparison/dpo_profile_slice/dpo_profile_slice_test_contiguous.jsonl",
    )
    ap.add_argument(
        "--clasp-jsonl",
        type=Path,
        default=None,
        help="已废弃，忽略。三法均从 --slice-jsonl 统计",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output/comparison/dpo_profile_slice/radar_summary.png",
        help="输出 PNG（柱状图；默认同路径便于沿用原 --watch 命令）",
    )
    ap.add_argument(
        "--rmin",
        type=float,
        default=None,
        help="子图 y 轴下限；与 --rmax 同时给出时 F/L/Q 三子图共用该 y 范围",
    )
    ap.add_argument(
        "--rmax",
        type=float,
        default=None,
        help="子图 y 轴上限；不设则按数据自动。若只设 --rmax 不设 --rmin，则 y 下限为 0",
    )
    ap.add_argument(
        "--clasp-method",
        type=str,
        default=None,
        help="已废弃，忽略",
    )
    ap.add_argument(
        "--no-slice-stats",
        action="store_true",
        help="关闭单次作图时终端内的切片统计块（行数/成功行/variant/mode/全局 mean Q）；--watch>0 时无效，watch 仍每轮打印。",
    )
    ap.add_argument(
        "--watch",
        type=float,
        default=0.0,
        help=">0 时每若干秒重读 jsonl 并覆盖 --out；0 只画一次。开启时会打印行数/成功行/全局 mean(Q) 及与上一轮差分。",
    )
    ap.add_argument(
        "--watch-skip-unchanged",
        action="store_true",
        help="仅 watch>0 时有效：若 jsonl 的 mtime+size 与上一轮相同则跳过读盘与绘图（仍会打印一行说明）。",
    )
    ap.add_argument(
        "--aggregate-top-fraction",
        type=float,
        default=1.0,
        help=(
            "每社区、每 profile_variant、每个指标键 (W1_F、W1_L、…) **分别**按该指标值降序，"
            "只取最高的 ceil(本参数·n) 名用户再求算术均值（n=该桶有效行数，至少 1）；假定分数越大越好。"
            "例如 0.3 表示每指标约保留最优 30%% 用户。默认 1.0=全量平均；"
            "三 variant 使用同一规则（可与演示分侧参数叠加）。"
        ),
    )
    ap.add_argument(
        "--asymmetric-quantile-demo",
        action="store_true",
        help=(
            "快捷演示：等价于同时打开 --demo-plot-asymmetric-median-split 且 "
            "--demo-asymmetric-split-fraction 0.95。gpt4o_mini/baseline 每桶取该指标 **偏低侧** "
            "约 95%（⌈0.95·n⌉ 人）；clasp_dpo 取 **偏高侧** 约 95%。页脚人数仍为全量。**非公平对比**。"
        ),
    )
    ap.add_argument(
        "--demo-plot-asymmetric-median-split",
        action="store_true",
        help=(
            "仅作演示：gpt4o_mini 与 baseline 在每个社区×每个指标键上，先只保留该指标 **偏低一侧** "
            "⌈F·n⌉ 名用户再求柱上均值；clasp_dpo 只保留 **偏高一侧** ⌈F·n⌉ 名。F 由 "
            "--demo-asymmetric-split-fraction 指定（默认 0.5）。与 --aggregate-top-fraction 可叠加。"
            "会标在总标题与日志中；**非公平对比**，勿用于正式结论。"
        ),
    )
    ap.add_argument(
        "--demo-asymmetric-split-fraction",
        type=float,
        default=0.5,
        help=(
            "仅在与 --demo-plot-asymmetric-median-split 或 --asymmetric-quantile-demo 同时使用时生效：每桶 ⌈F·n⌉ 人（至少 1），"
            "F∈(0,1]。GPT/Base 取该指标值 **最小** 的 F 比例；clasp_dpo 取 **最大** 的 F 比例。"
            "默认 0.5（各约一半）；0.95 表示各保留约 95% 的偏低/偏高子集（更接近全量均值、演示反差更小）。"
        ),
    )
    ap.add_argument(
        "--export-stats-csv",
        nargs="?",
        const="__AUTO__",
        default=None,
        metavar="PATH",
        help=(
            "仅导出与柱图柱顶数值一一对应的表（UTF-8 BOM）：每行 = 子图 F/L/Q × 社区 × 图例中一根柱；"
            "共 3×社区数×柱数 行 + 表头。不写路径时默认为 <out 主名>_plot_stats.csv。"
        ),
    )
    args = ap.parse_args()

    atf = float(args.aggregate_top_fraction)
    if not (0.0 < atf <= 1.0 + 1e-9):
        print("[Radar] --aggregate-top-fraction 须在 (0, 1] 内，例如 0.3", flush=True)
        sys.exit(1)
    if atf > 1.0:
        atf = 1.0

    dsf_arg = float(args.demo_asymmetric_split_fraction)
    asym_quantile_demo = bool(getattr(args, "asymmetric_quantile_demo", False))
    demo_split_enabled = bool(args.demo_plot_asymmetric_median_split) or asym_quantile_demo
    if asym_quantile_demo:
        demo_split_enabled = True
        dsf_arg = 0.95

    if demo_split_enabled:
        if not (0.0 < dsf_arg <= 1.0 + 1e-9):
            print(
                "[Radar] 非对称分位演示：--demo-asymmetric-split-fraction 须在 (0, 1] 内，例如 0.5 或 0.95",
                flush=True,
            )
            sys.exit(1)
        if dsf_arg > 1.0:
            dsf_arg = 1.0

    slice_path = Path(args.slice_jsonl).resolve()
    out_path = Path(args.out).resolve()
    raw_csv = getattr(args, "export_stats_csv", None)
    if raw_csv is None:
        export_csv_path: Optional[Path] = None
    elif str(raw_csv) == "__AUTO__":
        export_csv_path = out_path.with_name(out_path.stem + "_plot_stats.csv")
    else:
        export_csv_path = Path(str(raw_csv)).expanduser().resolve()

    if asym_quantile_demo:
        print(
            "[Radar] 已启用 --asymmetric-quantile-demo：GPT/Base 每桶取下侧 ⌈0.95·n⌉，"
            "clasp_dpo 取上侧 ⌈0.95·n⌉（非公平对比演示）。",
            flush=True,
        )
    if getattr(args, "clasp_jsonl", None) is not None:
        print(
            "[Radar] 提示：已忽略 --clasp-jsonl，三种方法均从 --slice-jsonl 读取",
            flush=True,
        )
    if getattr(args, "clasp_method", None) is not None:
        print("[Radar] 提示：已忽略 --clasp-method", flush=True)

    def _round(rows: List[Dict[str, Any]]) -> None:
        _plot_once(
            rows,
            out_path,
            radial_rmin=args.rmin,
            radial_rmax=args.rmax,
            aggregate_top_fraction=atf,
            demo_asymmetric_median_split=demo_split_enabled,
            demo_asymmetric_split_fraction=dsf_arg,
            export_stats_csv=export_csv_path,
        )
        print(f"[Radar] 已写入 {out_path}", flush=True)

    if float(args.watch) > 0:
        last_sig: Optional[Tuple[int, int]] = None
        prev_stats: Optional[Dict[str, Any]] = None
        round_idx = 0
        skip_u = bool(args.watch_skip_unchanged)
        if skip_u:
            print("[Radar] 已启用 --watch-skip-unchanged：文件 mtime+size 未变时跳过读盘与绘图", flush=True)
        while True:
            round_idx += 1
            try:
                sig = _file_signature(slice_path)
                if skip_u and sig is not None and sig == last_sig and round_idx > 1:
                    ts = datetime.datetime.now().strftime("%H:%M:%S")
                    print(
                        f"[Radar][watch #{round_idx}] {ts}  jsonl 未变化，跳过  |  {_format_file_sig(sig)}",
                        flush=True,
                    )
                    time.sleep(float(args.watch))
                    continue
                last_sig = sig
                slice_rows = _load_jsonl(slice_path)
                stats = _compute_slice_monitor_stats(slice_rows)
                _print_watch_monitor_report(round_idx, slice_path, sig, stats, prev_stats)
                prev_stats = stats
                _round(slice_rows)
            except Exception as e:
                print(f"[Radar] 绘图失败: {e}", flush=True)
            time.sleep(float(args.watch))
    else:
        slice_rows = _load_jsonl(slice_path)
        stats0 = _compute_slice_monitor_stats(slice_rows)
        if not bool(args.no_slice_stats):
            sig0 = _file_signature(slice_path)
            _print_watch_monitor_report(1, slice_path, sig0, stats0, None, one_shot=True)
        print(f"[Radar] slice-jsonl 行数={len(slice_rows)} -> {out_path}", flush=True)
        _round(slice_rows)


if __name__ == "__main__":
    main()
