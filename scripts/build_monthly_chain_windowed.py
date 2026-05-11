#!/usr/bin/env python3
"""
合并 data/test 与 data/eval_unseen 下各 community_*.jsonl（按社区 id），
用 monthly_chain 切分后写入 output/windowed/test/monthly_chain_community_{id}.jsonl。

命名仅含方法前缀 monthly_chain，不区分来源 split。
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import NUM_WINDOWS_EVAL_CHAIN, WINDOW_SIZE
from src.window_splitter import prepare_windowed_data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="数据根目录（其下含 test/、eval_unseen/）",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "windowed" / "test",
        help="输出目录",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=WINDOW_SIZE,
        help="monthly_chain 每窗条数（默认 config.WINDOW_SIZE）",
    )
    parser.add_argument(
        "--clean-old-monthly",
        action="store_true",
        help="删除输出目录内旧的 monthly_chain_*.jsonl 再写入",
    )
    args = parser.parse_args()

    data_root = args.data_root.resolve()
    out_dir = args.output_dir.resolve()
    src_dirs = [data_root / "test", data_root / "eval_unseen"]

    by_cid: dict[int, list[Path]] = defaultdict(list)
    for d in src_dirs:
        if not d.is_dir():
            print(f"[skip] 不存在: {d}", flush=True)
            continue
        for fp in sorted(d.glob("community_*.jsonl")):
            try:
                cid = int(fp.stem.replace("community_", ""))
            except ValueError:
                continue
            by_cid[cid].append(fp)

    if not by_cid:
        print("未找到 test/eval_unseen 下的 community_*.jsonl", flush=True)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)
    if args.clean_old_monthly:
        for fp in out_dir.glob("monthly_chain_*.jsonl"):
            fp.unlink()
            print(f"[删除] {fp.name}", flush=True)

    for cid in sorted(by_cid.keys()):
        tmp = out_dir / f".merge_community_{cid}.jsonl"
        try:
            with tmp.open("w", encoding="utf-8") as fout:
                for src_fp in sorted(by_cid[cid]):
                    with src_fp.open(encoding="utf-8") as fin:
                        for line in fin:
                            if line.strip():
                                fout.write(line)
            out_fp = out_dir / f"monthly_chain_community_{cid}.jsonl"
            prepare_windowed_data(
                str(tmp),
                str(out_fp),
                window_size=int(args.window_size),
                num_windows=NUM_WINDOWS_EVAL_CHAIN,
                split_mode="monthly_chain",
                actions_per_month=None,
            )
            print(
                f"[ok] community {cid}: 合并 {len(by_cid[cid])} 个源文件 -> {out_fp.name}",
                flush=True,
            )
        finally:
            if tmp.exists():
                tmp.unlink()


if __name__ == "__main__":
    main()
