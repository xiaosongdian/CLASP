#!/usr/bin/env python3
"""
Window splitter: split user action sequence into num_windows windows.

Mode contiguous (default): split sequentially by time, window_size actions per window,
requires actions >= window_size * num_windows.

Mode monthly_chain: consecutive MONTHLY_CHAIN_NUM_MONTHS (default 6) natural months, MONTHLY_CHAIN_WINDOWS_PER_MONTH (default 1) windows per month,
evenly sample actions_per_window actions within full month time span (total windows = NUM_WINDOWS_EVAL_CHAIN).
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.config import (
    MONTHLY_CHAIN_NUM_MONTHS,
    MONTHLY_CHAIN_WINDOWS_PER_MONTH,
    NUM_WINDOWS,
    NUM_WINDOWS_EVAL_CHAIN,
    WINDOW_SIZE,
)

MonthKey = Tuple[int, int]


def _action_timestamp(action: Dict) -> Optional[float]:
    """Extract timestamp from action. date=YYYYMMDDHHMM format, fallback to timestamp field."""
    d = action.get("date")
    if isinstance(d, str) and len(d) >= 8 and d[:8].isdigit():
        y, mo, day = int(d[0:4]), int(d[4:6]), int(d[6:8])
        h, mi = 0, 0
        if len(d) >= 12:
            h, mi = int(d[8:10]), int(d[10:12])
        try:
            return datetime(y, mo, day, h, mi).timestamp()
        except ValueError:
            pass
    ts = action.get("timestamp") or ""
    if isinstance(ts, str) and ts.strip():
        s = ts.strip().replace("/", "-")
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(s[:19], fmt).timestamp()
            except ValueError:
                continue
        try:
            return datetime.strptime(s[:10], "%Y-%m-%d").timestamp()
        except ValueError:
            pass
    return None


def _sort_actions_by_time(actions: List[Dict]) -> List[Dict]:
    """Sort actions by time; stable sort for same timestamp."""

    def key(a: Dict) -> Tuple[float, str, str]:
        t = _action_timestamp(a)
        return (
            t if t is not None else float("inf"),
            str(a.get("date", "")),
            str(a.get("timestamp", "")),
        )

    return sorted(actions, key=key)


def _evenly_spaced_sample_by_time(sorted_actions: List[Dict], n: int) -> Optional[List[Dict]]:
    """
    On time interval [t_first, t_last] of this action segment, evenly space n target times,
    for each time select the temporally nearest action (no duplicate selection).
    If time collapses to a point, fall back to index-based even sampling.
    """
    m = len(sorted_actions)
    if m < n or n <= 0:
        return None
    ts = [_action_timestamp(a) for a in sorted_actions]
    if any(x is None for x in ts):
        return None
    t_lo, t_hi = ts[0], ts[-1]
    if t_hi <= t_lo:
        return _evenly_spaced_sample(sorted_actions, n)

    used: Set[int] = set()
    out: List[Dict] = []
    for i in range(n):
        tgt = t_lo + i * (t_hi - t_lo) / max(n - 1, 1)
        best_j: Optional[int] = None
        best_d = float("inf")
        for j in range(m):
            if j in used:
                continue
            dd = abs(ts[j] - tgt)
            if dd < best_d:
                best_d = dd
                best_j = j
        if best_j is None:
            return None
        used.add(best_j)
        out.append(sorted_actions[best_j])
    out.sort(key=lambda a: (_action_timestamp(a) or 0.0, str(a.get("date", ""))))
    return out


def _month_key(action: Dict) -> Optional[MonthKey]:
    """Extract (year, month) from date / timestamp field; return None if unable."""
    d = action.get("date")
    if isinstance(d, str) and len(d) >= 6 and d[:6].isdigit():
        return int(d[:4]), int(d[4:6])
    ts = action.get("timestamp") or ""
    if isinstance(ts, str) and ts.strip():
        part = ts.strip().replace("/", "-").split()[0]
        bits = part.split("-")
        if len(bits) >= 2 and bits[0].isdigit() and bits[1].isdigit():
            try:
                return int(bits[0]), int(bits[1])
            except ValueError:
                pass
    return None


def _next_month(y: int, m: int) -> MonthKey:
    if m >= 12:
        return y + 1, 1
    return y, m + 1


def _evenly_spaced_sample(sorted_actions: List[Dict], n: int) -> Optional[List[Dict]]:
    """Evenly sample n actions from sorted list by index."""
    m = len(sorted_actions)
    if m < n or n <= 0:
        return None
    if n == 1:
        return [sorted_actions[m // 2]]
    out: List[Dict] = []
    for i in range(n):
        idx = int(round(i * (m - 1) / (n - 1)))
        out.append(sorted_actions[idx])
    return out


def split_user_into_windows(
    actions: List[Dict],
    window_size: int = WINDOW_SIZE,
    num_windows: int = NUM_WINDOWS,
) -> Optional[List[List[Dict]]]:
    """
    Split one user's action sequence into num_windows equal-length windows.
    Return [W0, W1, ..., W(num_windows-1)], each window is a list of window_size actions.
    Return None if insufficient actions.
    """
    required = window_size * num_windows
    if len(actions) < required:
        return None
    windows = []
    for i in range(num_windows):
        start = i * window_size
        end = start + window_size
        windows.append(actions[start:end])
    return windows


def split_user_into_windows_monthly_chain(
    actions: List[Dict],
    actions_per_window: int,
    *,
    num_months: int = MONTHLY_CHAIN_NUM_MONTHS,
    windows_per_month: int = MONTHLY_CHAIN_WINDOWS_PER_MONTH,
) -> Optional[Tuple[List[List[Dict]], Dict]]:
    """
    Consecutive natural month window chain: find earliest consecutive num_months calendar months.

    - windows_per_month==1 (default): 1 window per month, evenly sample actions_per_window actions within that month's full time span.
    - windows_per_month==2: split each month by time midpoint into two halves (equal duration), evenly sample actions_per_window from each half.

    Total windows must be num_months * windows_per_month (default 6×1=6).
    """
    if actions_per_window <= 0 or num_months <= 0:
        return None
    if windows_per_month not in (1, 2):
        return None

    total_windows = num_months * windows_per_month
    min_per_month = actions_per_window * windows_per_month

    by_month: Dict[MonthKey, List[Dict]] = defaultdict(list)
    for a in actions:
        mk = _month_key(a)
        if mk is None:
            continue
        by_month[mk].append(a)

    for lst in by_month.values():
        sorted_lst = _sort_actions_by_time(lst)
        lst.clear()
        lst.extend(sorted_lst)

    if not by_month:
        return None

    months_sorted = sorted(by_month.keys())
    for start_m in months_sorted:
        cur_y, cur_m = start_m
        seq_keys: List[MonthKey] = []
        for _ in range(num_months):
            mk = (cur_y, cur_m)
            acts = by_month.get(mk)
            if acts is None or len(acts) < min_per_month:
                break
            seq_keys.append(mk)
            cur_y, cur_m = _next_month(cur_y, cur_m)
        else:
            windows: List[List[Dict]] = []
            ok_build = True
            for mk in seq_keys:
                month_acts = by_month[mk]
                ts_m = [_action_timestamp(a) for a in month_acts]
                if any(t is None for t in ts_m):
                    ok_build = False
                    break

                if windows_per_month == 1:
                    w = _evenly_spaced_sample_by_time(month_acts, actions_per_window)
                    if w is None:
                        return None
                    windows.append(w)
                    continue

                t_min, t_max = ts_m[0], ts_m[-1]
                t_mid = (t_min + t_max) / 2
                idx = bisect.bisect_right(ts_m, t_mid)
                first_half = month_acts[:idx]
                second_half = month_acts[idx:]
                if (
                    len(first_half) < actions_per_window
                    or len(second_half) < actions_per_window
                ):
                    ok_build = False
                    break
                w1 = _evenly_spaced_sample_by_time(first_half, actions_per_window)
                w2 = _evenly_spaced_sample_by_time(second_half, actions_per_window)
                if w1 is None or w2 is None:
                    return None
                windows.extend([w1, w2])

            if ok_build and len(windows) == total_windows:
                y0, m0 = seq_keys[0]
                split_desc = (
                    "equal_time_span_halves"
                    if windows_per_month == 2
                    else "single_window_even_time_per_month"
                )
                meta = {
                    "month_block_start": f"{y0:04d}-{m0:02d}",
                    "month_keys_used": [f"{y:04d}-{m:02d}" for y, m in seq_keys],
                    "num_months": num_months,
                    "windows_per_month": windows_per_month,
                    "actions_per_window": actions_per_window,
                    "month_sampling": split_desc,
                }
                return windows, meta

    return None


def prepare_windowed_data(
    input_file: str,
    output_file: str,
    window_size: int = WINDOW_SIZE,
    num_windows: int = NUM_WINDOWS,
    *,
    split_mode: str = "contiguous",
    actions_per_month: Optional[int] = None,
) -> Dict:
    """
    Read a community's jsonl file, perform window splitting for each user,
    output jsonl file with windows field.
    Return statistics summary.

    split_mode:
      - contiguous: window_size consecutive actions per window;
      - monthly_chain: consecutive MONTHLY_CHAIN_NUM_MONTHS months, MONTHLY_CHAIN_WINDOWS_PER_MONTH windows per month,
        evenly sample on full month (or half-month) time axis; total windows = NUM_WINDOWS_EVAL_CHAIN.
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_users = 0
    kept_users = 0
    skipped_users = 0

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                user = json.loads(line)
            except json.JSONDecodeError:
                continue

            total_users += 1
            actions = user.get("actions", [])
            monthly_meta: Optional[Dict] = None

            if split_mode == "monthly_chain":
                apw = (
                    int(actions_per_month) if actions_per_month is not None else window_size
                )
                nw_chain = MONTHLY_CHAIN_NUM_MONTHS * MONTHLY_CHAIN_WINDOWS_PER_MONTH
                got = split_user_into_windows_monthly_chain(actions, apw)
                if got is None:
                    skipped_users += 1
                    continue
                windows, monthly_meta = got
                used_n = apw * nw_chain
            else:
                windows = split_user_into_windows(actions, window_size, num_windows)
                if windows is None:
                    skipped_users += 1
                    continue
                used_n = window_size * num_windows

            kept_users += 1
            record = {
                "community_id": user.get("community_id"),
                "user_id": user.get("user_id"),
                "windows": {
                    f"W{i}": w for i, w in enumerate(windows)
                },
                "total_actions": len(actions),
                "used_actions": used_n,
                "split_mode": split_mode,
            }
            if split_mode == "monthly_chain" and monthly_meta:
                record["split_meta"] = monthly_meta
                record["window_size_effective"] = (
                    int(actions_per_month) if actions_per_month is not None else window_size
                )
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "total_users": total_users,
        "kept_users": kept_users,
        "skipped_users": skipped_users,
        "window_size": window_size,
        "num_windows": (
            MONTHLY_CHAIN_NUM_MONTHS * MONTHLY_CHAIN_WINDOWS_PER_MONTH
            if split_mode == "monthly_chain"
            else num_windows
        ),
        "split_mode": split_mode,
        "actions_per_month": actions_per_month,
    }
    print(
        f"[WindowSplitter] {input_path.name}: "
        f"total={total_users}, kept={kept_users}, skipped={skipped_users}"
    )
    return summary


def batch_prepare(
    input_dir: str,
    output_dir: str,
    split: str = "test",
    *,
    window_size: int = WINDOW_SIZE,
    num_windows: Optional[int] = None,
    split_mode: str = "contiguous",
    actions_per_month: Optional[int] = None,
) -> None:
    """ split 。"""
    nw = int(num_windows) if num_windows is not None else int(NUM_WINDOWS)
    in_dir = Path(input_dir) / split
    out_dir = Path(output_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp in sorted(in_dir.glob("community_*.jsonl")):
        out_file = out_dir / fp.name
        prepare_windowed_data(
            str(fp),
            str(out_file),
            window_size,
            nw,
            split_mode=split_mode,
            actions_per_month=actions_per_month,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input", required=True, help=" jsonl ")
    parser.add_argument("--output", required=True, help=" jsonl ")
    parser.add_argument("--split", default=None, help="split （）")
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help=f"（ {WINDOW_SIZE}）",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=None,
        help=(
            "； config.NUM_WINDOWS。"
            f" S0→W1… W0..W5  {NUM_WINDOWS_EVAL_CHAIN}（）。"
        ),
    )
    parser.add_argument(
        "--split-mode",
        choices=("contiguous", "monthly_chain"),
        default="contiguous",
        help="contiguous=；monthly_chain=（ config MONTHLY_CHAIN_*）",
    )
    parser.add_argument(
        "--actions-per-month",
        type=int,
        default=None,
        metavar="N",
        help=(
            " monthly_chain：； --window-size；"
            " 6 =config.MONTHLY_CHAIN_NUM_MONTHS×MONTHLY_CHAIN_WINDOWS_PER_MONTH"
        ),
    )
    args = parser.parse_args()

    nw = args.num_windows
    if nw is None:
        nw = NUM_WINDOWS

    if args.split_mode == "monthly_chain":
        apw = (
            args.actions_per_month
            if args.actions_per_month is not None
            else args.window_size
        )
        if apw <= 0:
            print("monthly_chain：（--actions-per-month  --window-size）")
            raise SystemExit(2)
    else:
        if args.actions_per_month is not None:
            print(": --actions-per-month  --split-mode monthly_chain ")

    inp = Path(args.input)
    if inp.is_file():
        prepare_windowed_data(
            args.input,
            args.output,
            args.window_size,
            int(nw),
            split_mode=args.split_mode,
            actions_per_month=args.actions_per_month,
        )
    elif inp.is_dir() and args.split:
        batch_prepare(
            args.input,
            args.output,
            args.split,
            window_size=args.window_size,
            num_windows=int(nw),
            split_mode=args.split_mode,
            actions_per_month=args.actions_per_month,
        )
    else:
        print(" jsonl ， + --split ")
