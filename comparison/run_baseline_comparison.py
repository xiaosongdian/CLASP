#!/usr/bin/env python3
"""
统一测试集上的多基线窗口链评估（不生成 DPO 对）。

默认基线（`--methods` 可选）：
  - static_s0：W0 初始画像固定不变；
  - prefix_refresh：每步用已观测前缀 W0..W_{t+1} 重算「初始画像」；
  - clasp_online：每步按预测误差精炼画像；
  - incremental_persona：S_{t-1} + 当前窗行为（无误差信号）精炼。

注意：所有方法已统一历史输入机制（profile_suffix），确保公平对比画像更新策略。

默认每种 method 单独目录（避免混在同一 jsonl）：
  output/comparison/<method>/baseline_chain_<split>.jsonl
  作图：output/comparison/<method>/<plot 文件名>_F|L|Q.png

示例（仓库根目录）：
  # 多方法 → clasp_online、prefix_refresh、static_s0 各一份 jsonl
  python -m comparison.run_baseline_comparison \\
    --split test --skip-window-split \\
    --windowed-root output/windowed \\
    --methods static_s0,prefix_refresh,clasp_online

  # 单社区 + 作图（图在同 method 目录下）
  python -m comparison.run_baseline_comparison \\
    --input-jsonl output/windowed/test/community_3.jsonl \\
    --methods clasp_online \\
    --plot clasp_c3.png

  # 旧版：所有 method 合并为一个 jsonl
  python -m comparison.run_baseline_comparison \\
    --combined-jsonl \\
    --output output/comparison/baseline_chain_test.jsonl \\
    --methods static_s0,prefix_refresh,clasp_online
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.dpo_pipeline import preflight_check
from src.scorer import SemanticScorer
from src.window_splitter import batch_prepare

from comparison.window_chain_eval import VALID_METHODS, evaluate_user_window_chain
from comparison.window_chain_plot import (
    aggregate_flq_by_step,
    filter_rows_for_plot_tails,
    plot_flq_separate_figures,
    print_step_table,
)


def _parse_methods(s: str) -> List[str]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    bad = [p for p in parts if p not in VALID_METHODS]
    if bad:
        print(f"[BaselineChain] 未知 method: {bad}，可选: {sorted(VALID_METHODS)}", flush=True)
        sys.exit(1)
    return parts


def _print_aggregate(rows: List[Dict[str, Any]]) -> None:
    """按 method 汇总 mean_Q，以及各 step 的 F/L/Q。"""
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
        chain_only = [
            float(x["mean_Q_chain"])
            for x in items
            if x.get("mean_Q_chain") is not None
        ]
        mean_chain = sum(chain_only) / len(chain_only) if chain_only else 0.0
        means, _ = aggregate_flq_by_step(items, method=None)
        step_q_only = {k: round(v["Q"], 4) for k, v in sorted(means.items())}
        print(
            f"  [{method}] 用户数={n} | 平均 mean_Q(全前向步)={mean_overall:.4f} | "
            f"mean_Q_chain(同左，兼容字段)={mean_chain:.4f} | 各步平均Q={step_q_only}",
            flush=True,
        )
        print_step_table(means, label=method)


def run(
    *,
    split: str,
    data_dir: Path,
    windowed_root: Path,
    methods: List[str],
    max_users: Optional[int],
    skip_preflight: bool,
    skip_window_split: bool,
    refinement_variants: int,
    workers: int,
    user_processes: int,
    user_process_stagger: float,
    use_parallel: bool,
    scorer_device: Optional[str],
    input_jsonl: Optional[Path] = None,
    file_glob: str = "community_*.jsonl",
    plot_path: Optional[Path] = None,
    always_accept_refinement: bool = False,
    plot_trim_each_tail: Optional[float] = None,
    num_windows: Optional[int] = None,
    window_size: Optional[int] = None,
    comparison_root: Path,
    separate_by_method: bool,
    output_jsonl: Optional[Path],
    output_stem: str,
) -> None:
    data_dir = data_dir.resolve()
    windowed_root = windowed_root.resolve()
    comparison_root = Path(comparison_root).resolve()

    if always_accept_refinement and "clasp_online" in methods:
        print(
            "[BaselineChain] clasp_online 使用 --always-accept-refinement："
            "每步始终采用新精炼画像（空则保留旧画像），不比 Q。",
            flush=True,
        )

    if input_jsonl is not None:
        input_jsonl = Path(input_jsonl).resolve()
        if not input_jsonl.is_file():
            print(f"[BaselineChain] 不是文件: {input_jsonl}", flush=True)
            sys.exit(1)
        files = [input_jsonl]
    else:
        raw_split_dir = data_dir / split
        if not raw_split_dir.is_dir():
            print(f"[BaselineChain] 不存在目录: {raw_split_dir}", flush=True)
            sys.exit(1)

        out_split = windowed_root / split
        if not skip_window_split:
            out_split.mkdir(parents=True, exist_ok=True)
            print(f"[BaselineChain] 窗口切分: {raw_split_dir} -> {out_split}", flush=True)
            from src.config import NUM_WINDOWS_EVAL_CHAIN, WINDOW_SIZE as _WS

            nw = int(num_windows) if num_windows is not None else int(NUM_WINDOWS_EVAL_CHAIN)
            ws = int(window_size) if window_size is not None else int(_WS)
            print(
                f"[BaselineChain] 每窗 {ws} 动作, num_windows={nw}（W0..W{nw - 1}）",
                flush=True,
            )
            batch_prepare(
                str(data_dir),
                str(windowed_root),
                split,
                window_size=ws,
                num_windows=nw,
            )
        else:
            if not out_split.is_dir():
                print(
                    f"[BaselineChain] 已指定 --skip-window-split 但缺少: {out_split}",
                    flush=True,
                )
                sys.exit(1)

        files = sorted(out_split.glob(file_glob))
        if not files:
            print(
                f"[BaselineChain] 无匹配文件: {out_split}/{file_glob}",
                flush=True,
            )
            sys.exit(1)

    if not skip_preflight and not preflight_check():
        print("[BaselineChain] 预检失败", flush=True)
        sys.exit(1)

    # 决定是否使用并行化
    if use_parallel and user_processes > 1:
        print(f"[BaselineChain] 使用并行模式: {user_processes} 个进程", flush=True)
        from comparison.run_baseline_parallel import run_baseline_comparison_parallel

        run_baseline_comparison_parallel(
            input_files=files,
            methods=methods,
            output_stem=output_stem,
            comparison_root=comparison_root,
            max_users=max_users,
            workers=workers,
            user_processes=user_processes,
            user_process_stagger_sec=user_process_stagger,
            scorer_device=scorer_device or "cpu",
            split=split,
        )

        # 并行模式不支持绘图，如果需要绘图，提示用户
        if plot_path is not None:
            print("[BaselineChain] 注意: 并行模式暂不支持绘图，请使用串行模式 (--no-parallel)", flush=True)

        return

    # 串行模式
    print(f"[BaselineChain] 使用串行模式", flush=True)

    sem_dev = scorer_device
    if sem_dev is None or str(sem_dev).strip() == "":
        sem_dev = "cpu"
    print(f"[BaselineChain] SemanticScorer device={sem_dev}", flush=True)
    semantic_scorer = SemanticScorer(device=sem_dev)
    profile_model, profile_tokenizer = None, None
    action_model, action_tokenizer = None, None

    all_rows: List[Dict[str, Any]] = []
    total_lines = 0
    t0 = time.time()

    if separate_by_method:
        comparison_root.mkdir(parents=True, exist_ok=True)
        out_suffix = ".jsonl"
        method_paths = {
            m: comparison_root / m / f"{output_stem}{out_suffix}" for m in methods
        }
        for p in method_paths.values():
            p.parent.mkdir(parents=True, exist_ok=True)
        print(
            "[BaselineChain] 按方法分目录输出: "
            + ", ".join(f"{m} -> {method_paths[m]}" for m in methods),
            flush=True,
        )
    else:
        assert output_jsonl is not None
        output_jsonl.parent.mkdir(parents=True, exist_ok=True)
        method_paths = {}

    with ExitStack() as stack:
        writers: Dict[str, Any] = {}
        if separate_by_method:
            for m in methods:
                writers[m] = stack.enter_context(
                    method_paths[m].open("w", encoding="utf-8")
                )
        else:
            writers["__all__"] = stack.enter_context(
                output_jsonl.open("w", encoding="utf-8")
            )

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
                            refinement_variants=refinement_variants,
                            workers=workers,
                            always_accept_refinement=always_accept_refinement,
                        )
                        r["source_file"] = fp_in.name
                        r["split"] = split
                        w = writers[method] if separate_by_method else writers["__all__"]
                        w.write(json.dumps(r, ensure_ascii=False) + "\n")
                        w.flush()
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
    if separate_by_method:
        print(
            f"[BaselineChain] 完成: {total_lines} 用户, {len(all_rows)} 行输出, "
            f"耗时 {dt:.1f}s；见各 method 目录下 {output_stem}.jsonl",
            flush=True,
        )
    else:
        print(
            f"[BaselineChain] 完成: {total_lines} 用户, {len(all_rows)} 行输出, "
            f"耗时 {dt:.1f}s -> {output_jsonl}",
            flush=True,
        )
    _print_aggregate(all_rows)

    if plot_path is not None and all_rows:
        trim_t = plot_trim_each_tail
        for m in methods:
            sub = [r for r in all_rows if r.get("method") == m and not r.get("error")]
            if not sub:
                continue
            sub_plot = sub
            if trim_t is not None and trim_t > 0:
                sub_plot, tmeta = filter_rows_for_plot_tails(
                    sub, tail_fraction=trim_t, key="mean_Q"
                )
                print(
                    f"[BaselineChain] 作图去极值 mean_Q 各侧 {trim_t*100:.1f}%: "
                    f"dropped={tmeta.get('dropped', 0)} plot_users={len(sub_plot)}/{len(sub)}",
                    flush=True,
                )
                if not sub_plot:
                    sub_plot = sub
            means, _ = aggregate_flq_by_step(sub_plot, method=None)
            if not means:
                continue
            steps_sorted = sorted(means.keys())
            labels: List[str] = []
            for si in steps_sorted:
                tw = None
                for st in sub_plot[0].get("steps") or []:
                    if int(st.get("step_index", -1)) == si:
                        tw = st.get("target_window")
                        break
                labels.append(str(tw) if tw else f"step{si}")
            if separate_by_method:
                fig_base = comparison_root / m / plot_path.name
                fig_base.parent.mkdir(parents=True, exist_ok=True)
            elif len(methods) == 1:
                fig_base = Path(plot_path)
            else:
                fig_base = plot_path.parent / f"{plot_path.stem}_{m}{plot_path.suffix}"
            paths = plot_flq_separate_figures(
                means,
                fig_base,
                title_prefix=m,
                window_labels=labels,
                n_users=len(sub_plot),
            )
            for p in paths:
                print(f"[BaselineChain] 已保存: {p}", flush=True)


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
        "--comparison-root",
        type=Path,
        default=ROOT / "output" / "comparison",
        help="评估结果根目录；默认按 method 写入 <root>/<method>/…",
    )
    parser.add_argument(
        "--combined-jsonl",
        action="store_true",
        help="所有 method 合并写入同一个 jsonl（旧版行为）；默认每种方法单独目录+单独文件",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "合并模式(--combined-jsonl)：完整输出文件路径。"
            "分目录模式：仅用作输出文件名主干（默认 baseline_chain_<split|输入stem>），"
            "实际路径为 <comparison-root>/<method>/<stem>.jsonl"
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="static_s0,prefix_refresh,clasp_online",
        help=f"逗号分隔，可选: {','.join(sorted(VALID_METHODS))}",
    )
    parser.add_argument("--max-users", type=int, default=None, help="最多评估用户数（原始用户数）")
    parser.add_argument(
        "--refinement-variants",
        "--num-candidates",
        type=int,
        default=None,
        dest="refinement_variants",
        help="clasp_online 每步精炼次数（默认 1=单次纠偏；>1 为消融，训练期 DPO 才需多份）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="候选画像线程数；默认 config.DPO_WORKERS",
    )
    parser.add_argument(
        "--user-processes",
        type=int,
        default=None,
        help="并行处理的用户进程数；默认 config.DPO_USER_PROCESSES（多进程加速）",
    )
    parser.add_argument(
        "--user-process-stagger",
        type=float,
        default=0.5,
        help="多进程启动错开时间（秒），减轻 API 洪峰；默认 0.5s",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="禁用多进程并行，使用串行模式（调试用）",
    )
    parser.add_argument(
        "--scorer-device",
        default="cpu",
        help="SentenceTransformer 语义分设备（默认 cpu，避免与 GPU 上 vLLM 等争显存）；可设 cuda、cuda:0",
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
    parser.add_argument(
        "--num-windows",
        type=int,
        default=None,
        help="窗口切分时窗口个数；默认 config.NUM_WINDOWS_EVAL_CHAIN（6=W0..W5）",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="窗口切分每窗动作数；默认 config.WINDOW_SIZE",
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="直接指定已窗口化的单个 jsonl（不要求 data/<split> 存在）",
    )
    parser.add_argument(
        "--file-glob",
        type=str,
        default="community_*.jsonl",
        help="目录模式下在 --windowed-root/<split> 下的 glob",
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="保存 F、L、Q 三张折线图（需 matplotlib）；如 output/c.png -> c_F.png, c_L.png, c_Q.png",
    )
    parser.add_argument(
        "--plot-trim-each-tail",
        type=float,
        default=None,
        metavar="P",
        help=(
            "仅作图：按 mean_Q 去掉最低/最高各该比例用户；0=不去极值；"
            "省略则用 config.PLOT_TRIM_EACH_TAIL；不写回 jsonl"
        ),
    )
    parser.add_argument(
        "--always-accept-refinement",
        action="store_true",
        help="仅 clasp_online：每步精炼后始终采用新画像，不与旧画像比 Q；精炼为空则保留旧画像",
    )
    args = parser.parse_args()

    from src.config import DPO_WORKERS as _DW
    from src.config import DPO_USER_PROCESSES as _DUP
    from src.config import PLOT_TRIM_EACH_TAIL as _PLOT_TRIM

    methods = _parse_methods(args.methods)
    comparison_root = Path(args.comparison_root).resolve()
    separate_by_method = not bool(args.combined_jsonl)

    if args.output is not None:
        output_stem = Path(args.output).stem
        if not output_stem:
            output_stem = "baseline_chain"
    else:
        tag = Path(args.input_jsonl).stem if args.input_jsonl is not None else args.split
        output_stem = f"baseline_chain_{tag}"

    out_single: Optional[Path] = None
    if not separate_by_method:
        if args.output is not None:
            out_single = Path(args.output).resolve()
        else:
            out_single = comparison_root / f"{output_stem}.jsonl"

    run(
        split=args.split,
        data_dir=args.data_dir,
        windowed_root=args.windowed_root,
        methods=methods,
        max_users=args.max_users,
        skip_preflight=args.skip_preflight,
        skip_window_split=args.skip_window_split,
        refinement_variants=int(args.refinement_variants if args.refinement_variants is not None else 1),
        workers=int(args.workers if args.workers is not None else _DW),
        user_processes=int(args.user_processes if args.user_processes is not None else _DUP),
        user_process_stagger=float(args.user_process_stagger),
        use_parallel=(not args.no_parallel),
        scorer_device=args.scorer_device,
        input_jsonl=args.input_jsonl,
        file_glob=args.file_glob,
        plot_path=args.plot,
        always_accept_refinement=bool(args.always_accept_refinement),
        plot_trim_each_tail=(
            float(_PLOT_TRIM if args.plot_trim_each_tail is None else args.plot_trim_each_tail)
            if args.plot is not None
            else None
        ),
        num_windows=args.num_windows,
        window_size=args.window_size,
        comparison_root=comparison_root,
        separate_by_method=separate_by_method,
        output_jsonl=out_single,
        output_stem=output_stem,
    )


if __name__ == "__main__":
    main()
