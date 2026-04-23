#!/usr/bin/env python3
"""
窗口切分器：将用户动作序列按 WINDOW_SIZE 切分为 NUM_WINDOWS 个窗口
不足 MIN_ACTIONS 的用户自动跳过
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from src.config import WINDOW_SIZE, NUM_WINDOWS, MIN_ACTIONS


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


def prepare_windowed_data(
    input_file: str,
    output_file: str,
    window_size: int = WINDOW_SIZE,
    num_windows: int = NUM_WINDOWS,
) -> Dict:
    """
    读取一个社区的 jsonl 文件，对每个用户做窗口切分，
    输出包含 windows 字段的 jsonl 文件。
    返回统计摘要。
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
            windows = split_user_into_windows(actions, window_size, num_windows)

            if windows is None:
                skipped_users += 1
                continue

            kept_users += 1
            record = {
                "community_id": user.get("community_id"),
                "user_id": user.get("user_id"),
                "windows": {
                    f"W{i}": w for i, w in enumerate(windows)
                },
                "total_actions": len(actions),
                "used_actions": window_size * num_windows,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "input_file": str(input_path),
        "output_file": str(output_path),
        "total_users": total_users,
        "kept_users": kept_users,
        "skipped_users": skipped_users,
        "window_size": window_size,
        "num_windows": num_windows,
    }
    print(
        f"[WindowSplitter] {input_path.name}: "
        f"total={total_users}, kept={kept_users}, skipped={skipped_users}"
    )
    return summary


def batch_prepare(input_dir: str, output_dir: str, split: str = "test") -> None:
    """对指定 split 目录下的所有社区文件做窗口切分。"""
    in_dir = Path(input_dir) / split
    out_dir = Path(output_dir) / split
    out_dir.mkdir(parents=True, exist_ok=True)

    for fp in sorted(in_dir.glob("community_*.jsonl")):
        out_file = out_dir / fp.name
        prepare_windowed_data(str(fp), str(out_file))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="窗口切分用户动作序列")
    parser.add_argument("--input", required=True, help="输入 jsonl 文件或目录")
    parser.add_argument("--output", required=True, help="输出 jsonl 文件或目录")
    parser.add_argument("--split", default=None, help="split 名称（目录模式时使用）")
    args = parser.parse_args()

    inp = Path(args.input)
    if inp.is_file():
        prepare_windowed_data(args.input, args.output)
    elif inp.is_dir() and args.split:
        batch_prepare(args.input, args.output, args.split)
    else:
        print("请指定单个 jsonl 文件，或指定目录 + --split 参数")
