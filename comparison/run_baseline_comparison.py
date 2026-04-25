#!/usr/bin/env python3
"""
统一测试集上的多基线窗口链评估（不生成 DPO 对）。

默认基线：
  - static_s0：W0 初始画像固定不变；
  - prefix_refresh：每步用已观测前缀 W0..W_{k-1} 重算「初始画像」；
  - clasp_online：每步用下一窗口预测误差精炼画像，与主 pipeline 纠偏逻辑一致（无 DPO 构造）。

示例（仓库根目录）：
  python -m comparison.run_baseline_comparison \\
    --split test \\
    --methods static_s0,prefix_refresh,clasp_online \\
    --output output/comparison/baseline_chain_test.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dpo_pipeline import preflight_check
from src.scorer import SemanticScorer
from src.window_splitter import batch_prepare

from comparison.window_chain_eval import VALID_METHODS, evaluate_user_window_chain


def _parse_methods(s: str) -> List[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    bad = [p for p in parts if p not in VALID_METHODS]
    if bad:
        print(f"[BaselineChain] 未知 method: {bad}，可选: {sorted(VALID_METHODS)}", flush=True)
        sys.exit(1)
    return parts


def _print_aggregate(rows: List[Dict[str, Any]]) -> None:
    """按 method 汇总 mean_Q / 各 step 平均 Q。"""
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        m = r.get("method")
        if m:
            by_method[str(m)].append(r)

    print("[BaselineChain] ========== 汇总（跳过含 error 的记录）==========", flush=True)
    for method in sorted(by_method.keys()):
        items = by_method[method]
        n = len(items)
        overall = [float(x["mean_Q"]) for x in items if x.get("mean_Q") is not None]
        mean_overall = sum(overall) / len(overall) if overall else 0.0

        step_sums: Dict[int, List[float]] = defaultdict(list)
        for x in items:
            for st in x.get("steps") or []:
                si = int(st.get("step_index", -1))
                if si >= 0:
                    step_sums[si].append(float(st["Q"]))

        step_means = {
            si: sum(vs) / len(vs) for si, vs in sorted(step_sums.items())
        }
        print(
            f"  [{method}] 用户数={n} | 平均 mean_Q={mean_overall:.4f} | "
            f"各步平均Q={ {k: round(v, 4) for k, v in step_means.items()} }",
            flush=True,
        )


def run(
    *,
    split: str,
    data_dir: Path,
    windowed_root: Path,
    output_jsonl: Path,
    methods: List[str],
    max_users: Optional[int],
    skip_preflight: bool,
    skip_window_split: bool,
    num_candidates: int,
    workers: int,
    scorer_device: Optional[str],
) -> None:
    data_dir = data_dir.resolve()
    raw_split_dir = data_dir / split
    if not raw_split_dir.is_dir():
        print(f"[BaselineChain] 不存在目录: {raw_split_dir}", flush=True)
        sys.exit(1)

    out_split = windowed_root / split
    if not skip_window_split:
        out_split.mkdir(parents=True, exist_ok=True)
        print(f"[BaselineChain] 窗口切分: {raw_split_dir} -> {out_split}", flush=True)
        batch_prepare(str(data_dir), str(windowed_root), split)
    else:
        if not out_split.is_dir():
            print(
                f"[BaselineChain] 已指定 --skip-window-split 但缺少: {out_split}",
                flush=True,
            )
            sys.exit(1)

    if not skip_preflight and not preflight_check():
        print("[BaselineChain] 预检失败", flush=True)
        sys.exit(1)

    sem_dev = scorer_device
    if sem_dev is not None and str(sem_dev).strip() == "":
        sem_dev = None
    semantic_scorer = SemanticScorer(device=sem_dev)
    profile_model, profile_tokenizer = None, None
    action_model, action_tokenizer = None, None

    files = sorted(out_split.glob("community_*.jsonl"))
    if not files:
        print(f"[BaselineChain] 无窗口化文件: {out_split}/community_*.jsonl", flush=True)
        sys.exit(1)

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    all_rows: List[Dict[str, Any]] = []
    total_lines = 0
    t0 = time.time()

    with output_jsonl.open("w", encoding="utf-8") as fp:
        for fp_in in files:
            with fp_in.open("r", encoding="utf-8") as fin:
                for line in fin:
                    if max_users is not None and total_lines >= max_users:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        user = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    for method in methods:
                        r = evaluate_user_window_chain(
                            user,
                            method,
                            semantic_scorer,
                            profile_model=profile_model,
                            profile_tokenizer=profile_tokenizer,
                            action_model=action_model,
                            action_tokenizer=action_tokenizer,
                            num_candidates=num_candidates,
                            workers=workers,
                        )
                        r["source_file"] = fp_in.name
                        r["split"] = split
                        fp.write(json.dumps(r, ensure_ascii=False) + "\n")
                        fp.flush()
                        all_rows.append(r)

                    total_lines += 1
                    if total_lines % 10 == 0:
                        print(
                            f"[BaselineChain] 已处理 {total_lines} 个用户"
                            f"（每用户 {len(methods)} 条方法结果）…",
                            flush=True,
                        )

            if max_users is not None and total_lines >= max_users:
                break

    dt = time.time() - t0
    print(
        f"[BaselineChain] 完成: {total_lines} 用户, {len(all_rows)} 行输出, "
        f"耗时 {dt:.1f}s -> {output_jsonl}",
        flush=True,
    )
    _print_aggregate(all_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="测试集窗口链：多基线 F/L/Q（不构造 DPO）"
    )
    parser.add_argument(
        "--split",
        default="test",
        help="data 下子目录名，如 test / eval_unseen",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="含原始 jsonl 的数据根目录",
    )
    parser.add_argument(
        "--windowed-root",
        type=Path,
        default=ROOT / "output" / "windowed_eval_chain",
        help="窗口化输出根目录：<root>/<split>/community_*.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="JSONL 输出路径；默认 output/comparison/baseline_chain_<split>.jsonl",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="static_s0,prefix_refresh,clasp_online",
        help=f"逗号分隔，可选: {','.join(sorted(VALID_METHODS))}",
    )
    parser.add_argument("--max-users", type=int, default=None, help="最多评估用户数（原始用户数）")
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=None,
        help="clasp_online 每步候选数；默认用 config.NUM_CANDIDATE_PROFILES",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="候选画像线程数；默认 config.DPO_WORKERS",
    )
    parser.add_argument(
        "--scorer-device",
        default=None,
        help="SemanticScorer 设备，如 cpu / cuda；默认自动",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="跳过 vLLM / ST 预检",
    )
    parser.add_argument(
        "--skip-window-split",
        action="store_true",
        help="跳过窗口切分，直接使用 --windowed-root/<split>",
    )
    args = parser.parse_args()

    from src.config import DPO_WORKERS as _DW
    from src.config import NUM_CANDIDATE_PROFILES as _NC

    out_path = args.output
    if out_path is None:
        out_path = ROOT / "output" / "comparison" / f"baseline_chain_{args.split}.jsonl"

    run(
        split=args.split,
        data_dir=args.data_dir,
        windowed_root=args.windowed_root,
        output_jsonl=out_path,
        methods=_parse_methods(args.methods),
        max_users=args.max_users,
        skip_preflight=args.skip_preflight,
        skip_window_split=args.skip_window_split,
        num_candidates=int(args.num_candidates if args.num_candidates is not None else _NC),
        workers=int(args.workers if args.workers is not None else _DW),
        scorer_device=args.scorer_device,
    )


if __name__ == "__main__":
    main()
