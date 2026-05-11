#!/usr/bin/env python3
"""
窗口切分器：将用户动作序列切分为 num_windows 个窗口。

模式 contiguous（默认）：按时间顺序连续切块，每窗 window_size 条，
需 actions 不少于 window_size * num_windows。

模式 monthly_chain：连续 MONTHLY_CHAIN_NUM_MONTHS（默认 6）个自然月，每月 MONTHLY_CHAIN_WINDOWS_PER_MONTH（默认 1）窗，
在全月时间跨度内均匀抽取 actions_per_window 条（总窗数 = NUM_WINDOWS_EVAL_CHAIN）。
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
    """解析动作为可用于排序的时间戳（秒）。优先 date=YYYYMMDDHHMM，其次 timestamp 字符串。"""
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
    """按解析时间排序；无法解析的排在后面（一般不会进入月度分组）。"""

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
    在该段动作的时间区间 [t_first, t_last] 上等距取 n 个目标时刻，
    每个时刻选时间上最近的一条动作（不重复选同一条）。
    若时间坍缩为一点则退回按索引均匀抽样。
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
    """从动作的 date / timestamp 解析到 (year, month)；失败返回 None。"""
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
    """在时间有序的列表上均匀取 n 条（含首尾倾向），避免扎堆在同一天。"""
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
    将一个用户的动作序列切分为 num_windows 个等长窗口。
    返回 [W0, W1, ..., W(num_windows-1)]，每个窗口是 window_size 条动作的列表。
    动作数不足时返回 None。
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
    连续自然月窗口链：找到最早的连续 num_months 个日历月。

    - windows_per_month==1（默认）：每月 1 窗，在该月整段动作的时间跨度内均匀抽 actions_per_window 条。
    - windows_per_month==2：每月按时间中点分为两半（两半时长相等），两半各均匀抽 actions_per_window 条。

    总窗数须为 num_months * windows_per_month（默认 6×1=6）。
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
    读取一个社区的 jsonl 文件，对每个用户做窗口切分，
    输出包含 windows 字段的 jsonl 文件。
    返回统计摘要。

    split_mode:
      - contiguous: 每窗 window_size 条连续动作；
      - monthly_chain: 连续 MONTHLY_CHAIN_NUM_MONTHS 个月、每月 MONTHLY_CHAIN_WINDOWS_PER_MONTH 窗，
        在全月（或半月）时间轴上均匀抽样；总窗数 = NUM_WINDOWS_EVAL_CHAIN。
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
    """对指定 split 目录下的所有社区文件做窗口切分。"""
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
    parser = argparse.ArgumentParser(description="窗口切分用户动作序列")
    parser.add_argument("--input", required=True, help="输入 jsonl 文件或目录")
    parser.add_argument("--output", required=True, help="输出 jsonl 文件或目录")
    parser.add_argument("--split", default=None, help="split 名称（目录模式时使用）")
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help=f"每窗动作数（默认 {WINDOW_SIZE}）",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=None,
        help=(
            "窗口个数；默认与训练一致 config.NUM_WINDOWS。"
            f"评估链 S0→W1…需 W0..W5 时请设 {NUM_WINDOWS_EVAL_CHAIN}（或显式传参）。"
        ),
    )
    parser.add_argument(
        "--split-mode",
        choices=("contiguous", "monthly_chain"),
        default="contiguous",
        help="contiguous=顺序切块；monthly_chain=连续自然月链（见 config MONTHLY_CHAIN_*）",
    )
    parser.add_argument(
        "--actions-per-month",
        type=int,
        default=None,
        metavar="N",
        help=(
            "仅 monthly_chain：每个时间窗均匀抽取条数；默认用 --window-size；"
            "总 6 窗=config.MONTHLY_CHAIN_NUM_MONTHS×MONTHLY_CHAIN_WINDOWS_PER_MONTH"
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
            print("monthly_chain：每窗条数须为正（--actions-per-month 或 --window-size）")
            raise SystemExit(2)
    else:
        if args.actions_per_month is not None:
            print("提示: --actions-per-month 仅在 --split-mode monthly_chain 下使用")

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
        print("请指定单个 jsonl 文件，或指定目录 + --split 参数")
