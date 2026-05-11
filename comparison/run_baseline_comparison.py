#!/usr/bin/env python3
"""
统一测试集上的多基线窗口链评估（不生成 DPO 对）。

默认基线（`--methods` 可选）：
  - static_s0：W0 初始画像固定不变；
  - prefix_refresh：每步用已观测前缀 W0..W_{t+1} 重算「初始画像」；
  - clasp_online：每步按预测误差精炼画像；
  - clasp_online_no_hist：与 clasp_online 相同，但动作 prompt 不含观测历史（与 ``--no-action-prompt-observed-history`` 效果一致，且仅作用于该方法）；
  - history_only：不生成画像；每步将 **W0..W_t** 全部已观测动作写入动作侧「画像」槽位预测 W_{t+1}，Recent user actions 置空（避免重复），总长受 config 约束；
  - incremental_persona：S_{t-1} + 当前窗行为（无误差信号）精炼。

注意：所有方法已统一历史输入机制（profile_suffix），确保公平对比画像更新策略。

默认每种 method 单独目录（避免混在同一 jsonl）：
  output/comparison/<method>/baseline_chain_<split或标签>_<数据集类型>.jsonl
  数据集类型：contiguous（顺序切块窗口化）或 monthly_chain（自然月链窗口化）
  作图：output/comparison/<method>/<plot 文件名>_F|L|Q.png

示例（仓库根目录）：
  # 多方法 → clasp_online、prefix_refresh、static_s0 各一份 jsonl
  python -m comparison.run_baseline_comparison \\
    --split test --skip-window-split \\
    --windowed-root output/windowed \\
    --methods static_s0,prefix_refresh,clasp_online

  # 使用 monthly_chain 窗口化数据（默认 glob：monthly_chain_community_*.jsonl）
  python -m comparison.run_baseline_comparison \\
    --split test --skip-window-split \\
    --windowed-root output/windowed \\
    --windowed-dataset monthly_chain \\
    --methods static_s0

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

  # 断点续跑：跳过输出 jsonl 中已成功完成的用户（无 error），追加新结果
  python -m comparison.run_baseline_comparison --split test --resume \\
    --skip-window-split --comparison-root output/comparison

  # 动作 prompt 不载入观测历史（消融 profile 后历史块 / Recent user actions）
  python -m comparison.run_baseline_comparison --methods clasp_online \\
    --no-action-prompt-observed-history ...

  # 关闭链末三窗口评估（减少额外动作 API；输出无 three_window_evaluation）
  python -m comparison.run_baseline_comparison --methods static_s0 \\
    --no-three-window-evaluation ...

  # 每社区只评测前 100 个用户（默认即 100；全量可设 --max-users-per-community 0）
  python -m comparison.run_baseline_comparison --max-users-per-community 100 ...
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 目录模式下默认匹配的窗口化 jsonl（可被 --file-glob 覆盖）
WINDOWED_DATASET_FILE_GLOBS = {
    "contiguous": "community_*.jsonl",
    "monthly_chain": "monthly_chain_community_*.jsonl",
}


def _output_stem_with_dataset(base_stem: str, windowed_dataset: str) -> str:
    """为输出 jsonl 主干追加 _<数据集类型>，避免 contiguous / monthly_chain 结果互相覆盖。"""
    suf = f"_{windowed_dataset}"
    return base_stem if base_stem.endswith(suf) else base_stem + suf


from src.dpo_pipeline import preflight_check
from src.scorer import SemanticScorer
from src.window_splitter import batch_prepare

from comparison.window_chain_eval import (
    CLASP_ONLINE_VARIANTS,
    CLASP_PROFILE_SNAPSHOT_FILENAME,
    VALID_METHODS,
    evaluate_user_window_chain,
)
from comparison.baseline_resume import (
    filter_users_per_community,
    load_all_prior_rows,
    load_completed_keys_per_method,
    serialize_user_key,
)
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
    plot_trim_sides: str = "both",
    plot_trim_scope: str = "user",
    plot_step_trim_basis: str = "deviation",
    num_windows: Optional[int] = None,
    window_size: Optional[int] = None,
    window_split_mode: str = "contiguous",
    actions_per_month: Optional[int] = None,
    comparison_root: Path,
    separate_by_method: bool,
    output_jsonl: Optional[Path],
    output_stem: str,
    resume: bool = False,
    record_profile_snapshots: bool = True,
    action_prompt_include_observed_history: bool = True,
    enable_three_window_evaluation: bool = True,
    max_users_per_community: int = 100,
) -> None:
    data_dir = data_dir.resolve()
    windowed_root = windowed_root.resolve()
    comparison_root = Path(comparison_root).resolve()

    if always_accept_refinement and any(m in CLASP_ONLINE_VARIANTS for m in methods):
        print(
            "[BaselineChain] clasp_online / clasp_online_no_hist 使用 --always-accept-refinement："
            "每步始终采用新精炼画像（空则保留旧画像），不比 Q。",
            flush=True,
        )

    if not action_prompt_include_observed_history:
        print(
            "[BaselineChain] --no-action-prompt-observed-history："
            "动作预测 prompt 不含观测历史（仅画像 + Current scenario）。",
            flush=True,
        )

    if not enable_three_window_evaluation:
        print(
            "[BaselineChain] --no-three-window-evaluation："
            "跳过链末三窗口对比（jsonl 中无 three_window_evaluation）。",
            flush=True,
        )

    if max_users_per_community > 0:
        print(
            f"[BaselineChain] 每社区最多评测用户数: {max_users_per_community} "
            f"（超出输入顺序跳过；0=不限制）",
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
            from src.config import (
                MONTHLY_CHAIN_NUM_MONTHS,
                MONTHLY_CHAIN_WINDOWS_PER_MONTH,
                NUM_WINDOWS_EVAL_CHAIN,
                WINDOW_SIZE as _WS,
            )

            nw = int(num_windows) if num_windows is not None else int(NUM_WINDOWS_EVAL_CHAIN)
            ws = int(window_size) if window_size is not None else int(_WS)
            apm: Optional[int] = (
                int(actions_per_month)
                if actions_per_month is not None
                else (ws if window_split_mode == "monthly_chain" else None)
            )
            if window_split_mode == "monthly_chain":
                if apm is None or apm <= 0:
                    print(
                        "[BaselineChain] monthly_chain 需要有效的每窗条数：请设 "
                        "--actions-per-month，或使用默认 --window-size（等于 config.WINDOW_SIZE）。",
                        flush=True,
                    )
                    sys.exit(1)
                print(
                    f"[BaselineChain] 切分模式=monthly_chain：连续 {MONTHLY_CHAIN_NUM_MONTHS} 个自然月，"
                    f"每月 {MONTHLY_CHAIN_WINDOWS_PER_MONTH} 个时间窗，每窗 {apm} 条 "
                    f"（共 {apm * NUM_WINDOWS_EVAL_CHAIN} 条，{NUM_WINDOWS_EVAL_CHAIN} 窗 W0..W5）",
                    flush=True,
                )
            else:
                print(
                    f"[BaselineChain] 切分模式=contiguous：每窗 {ws} 动作, "
                    f"num_windows={nw}（W0..W{nw - 1}）",
                    flush=True,
                )
            batch_prepare(
                str(data_dir),
                str(windowed_root),
                split,
                window_size=ws,
                num_windows=nw,
                split_mode=window_split_mode,
                actions_per_month=apm,
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

    if not skip_preflight and not preflight_check(comparison_methods=methods):
        print("[BaselineChain] 预检失败", flush=True)
        sys.exit(1)

    do_parallel = bool(use_parallel and user_processes > 1)
    if resume and do_parallel:
        print("[BaselineChain] --resume：并行模式将追加写入并跳过已完成用户。", flush=True)

    # 决定是否使用并行化
    if do_parallel:
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
            resume=resume,
            refinement_variants=refinement_variants,
            always_accept_refinement=always_accept_refinement,
            record_profile_snapshots=record_profile_snapshots,
            action_prompt_include_observed_history=action_prompt_include_observed_history,
            enable_three_window_evaluation=enable_three_window_evaluation,
            max_users_per_community=max_users_per_community,
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

    new_rows: List[Dict[str, Any]] = []
    prior_rows: List[Dict[str, Any]] = []
    completed_by_m: Dict[str, Set[str]] = {m: set() for m in methods}
    total_lines = 0
    per_comm_done: Dict[Any, int] = defaultdict(int)
    t0 = time.time()

    clasp_snap_by_method: Dict[str, Path] = {}
    if record_profile_snapshots:
        for sm in methods:
            if sm not in CLASP_ONLINE_VARIANTS:
                continue
            snap_dir = comparison_root / sm / "profile_snapshots" / output_stem
            snap_dir.mkdir(parents=True, exist_ok=True)
            _snap_fp = snap_dir / CLASP_PROFILE_SNAPSHOT_FILENAME
            if not resume:
                _snap_fp.unlink(missing_ok=True)
                print(
                    f"[BaselineChain] {sm} 画像快照（单文件）: {_snap_fp}",
                    flush=True,
                )
            else:
                print(
                    f"[BaselineChain] {sm} 画像快照追加: {_snap_fp}",
                    flush=True,
                )
            clasp_snap_by_method[sm] = snap_dir

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

    if resume:
        completed_by_m = load_completed_keys_per_method(
            separate_by_method=separate_by_method,
            methods=methods,
            method_paths=method_paths,
            combined_jsonl=output_jsonl if not separate_by_method else None,
        )
        prior_rows = load_all_prior_rows(
            separate_by_method=separate_by_method,
            methods=methods,
            method_paths=method_paths,
            combined_jsonl=output_jsonl if not separate_by_method else None,
        )
        stats = {m: len(completed_by_m.get(m) or set()) for m in methods}
        print(f"[BaselineChain] --resume: 各 method 已成功用户数（将跳过）: {stats}", flush=True)

    write_mode = "a" if resume else "w"

    with ExitStack() as stack:
        writers: Dict[str, Any] = {}
        if separate_by_method:
            for m in methods:
                writers[m] = stack.enter_context(
                    method_paths[m].open(write_mode, encoding="utf-8")
                )
        else:
            writers["__all__"] = stack.enter_context(
                output_jsonl.open(write_mode, encoding="utf-8")
            )

        for fp_in in files:
            with fp_in.open("r", encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        user = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    cid = user.get("community_id")
                    if (
                        max_users_per_community > 0
                        and per_comm_done[cid] >= max_users_per_community
                    ):
                        continue

                    if max_users is not None and total_lines >= max_users:
                        break

                    ukey = serialize_user_key(user)
                    for method in methods:
                        if resume and ukey in completed_by_m.get(method, set()):
                            continue
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
                            profile_snapshot_dir=clasp_snap_by_method.get(method),
                            action_prompt_include_observed_history=action_prompt_include_observed_history,
                            enable_three_window_evaluation=enable_three_window_evaluation,
                        )
                        r["source_file"] = fp_in.name
                        r["split"] = split
                        w = writers[method] if separate_by_method else writers["__all__"]
                        w.write(json.dumps(r, ensure_ascii=False) + "\n")
                        w.flush()
                        new_rows.append(r)
                        if not r.get("error"):
                            completed_by_m.setdefault(method, set()).add(ukey)

                    per_comm_done[cid] += 1
                    total_lines += 1
                    if total_lines % 10 == 0:
                        print(
                            f"[BaselineChain] 已扫描 {total_lines} 个输入用户"
                            f"（本 run 新写入 {len(new_rows)} 行）…",
                            flush=True,
                        )

            if max_users is not None and total_lines >= max_users:
                break

    all_rows = prior_rows + new_rows

    dt = time.time() - t0
    if separate_by_method:
        print(
            f"[BaselineChain] 完成: 扫描输入用户 {total_lines} 个, 本 run 新写入 {len(new_rows)} 行, "
            f"汇总共 {len(all_rows)} 行, 耗时 {dt:.1f}s；见各 method 目录下 {output_stem}.jsonl",
            flush=True,
        )
    else:
        print(
            f"[BaselineChain] 完成: 扫描输入用户 {total_lines} 个, 本 run 新写入 {len(new_rows)} 行, "
            f"汇总共 {len(all_rows)} 行, 耗时 {dt:.1f}s -> {output_jsonl}",
            flush=True,
        )
    _print_aggregate(all_rows)

    if plot_path is not None and all_rows:
        trim_t = plot_trim_each_tail
        scope = str(plot_trim_scope).lower().strip()
        if scope not in ("user", "step"):
            scope = "user"
        for m in methods:
            sub = [r for r in all_rows if r.get("method") == m and not r.get("error")]
            if not sub:
                continue
            sub_plot = sub
            if trim_t is not None and trim_t > 0 and scope == "user":
                sub_plot, tmeta = filter_rows_for_plot_tails(
                    sub,
                    tail_fraction=trim_t,
                    key="mean_Q",
                    trim_sides=str(plot_trim_sides),
                )
                print(
                    f"[BaselineChain] 作图去极值 mean_Q "
                    f"(sides={plot_trim_sides}) {trim_t*100:.1f}%: "
                    f"dropped={tmeta.get('dropped', 0)} plot_users={len(sub_plot)}/{len(sub)}",
                    flush=True,
                )
                if not sub_plot:
                    sub_plot = sub
            elif trim_t is not None and trim_t > 0 and scope == "step":
                print(
                    f"[BaselineChain] 作图 per-step trim (sides={plot_trim_sides}) "
                    f"{trim_t*100:.1f}% basis={plot_step_trim_basis} | users={len(sub)}",
                    flush=True,
                )
            st_tail = float(trim_t) if (trim_t is not None and trim_t > 0 and scope == "step") else 0.0
            means, _ = aggregate_flq_by_step(
                sub_plot,
                method=None,
                step_trim_each_tail=st_tail,
                step_trim_sides=str(plot_trim_sides),
                step_trim_basis=str(plot_step_trim_basis),
            )
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
        help=(
            "窗口化 jsonl 根目录：读取 <root>/<split>/；文件名由 "
            "--windowed-dataset 或 --file-glob 决定（如 output/windowed/test）"
        ),
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
            "合并模式(--combined-jsonl)：完整输出文件路径（文件名自定，建议含数据集类型以免混淆）。"
            "分目录模式：仅用作输出文件名主干（默认 baseline_chain_<split|输入stem>_<windowed-dataset>），"
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
        "--max-users-per-community",
        type=int,
        default=100,
        help=(
            "每个 community_id 仅评测前 K 个用户（按各输入文件中的出现顺序）；"
            "节省 API 时间。默认 100；设为 0 表示不限制"
        ),
    )
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
        help="窗口切分每窗动作数；默认 config.WINDOW_SIZE（contiguous）；monthly_chain 下可作每月抽取条数备用默认值",
    )
    parser.add_argument(
        "--window-split-mode",
        choices=("contiguous", "monthly_chain"),
        default="contiguous",
        help=(
            "窗口切分策略：contiguous=顺序切块；"
            "monthly_chain=连续自然月、每月均匀抽取（见 --actions-per-month）"
        ),
    )
    parser.add_argument(
        "--actions-per-month",
        type=int,
        default=None,
        metavar="N",
        help=(
            "仅 monthly_chain：每个时间窗的动作条数（默认与 --window-size 或 config.WINDOW_SIZE 一致）；"
            "总窗数固定为 config.NUM_WINDOWS_EVAL_CHAIN（默认 6=连续 6 个月×每月 1 窗）。"
        ),
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="直接指定已窗口化的单个 jsonl（不要求 data/<split> 存在）",
    )
    parser.add_argument(
        "--windowed-dataset",
        choices=tuple(WINDOWED_DATASET_FILE_GLOBS.keys()),
        default="contiguous",
        help=(
            "已窗口化测试集类型；在未指定 --file-glob 时决定匹配模式："
            "contiguous=顺序切块 community_*.jsonl；"
            "monthly_chain=自然月链 monthly_chain_community_*.jsonl（见 scripts/build_monthly_chain_windowed.py）"
        ),
    )
    parser.add_argument(
        "--file-glob",
        type=str,
        default=None,
        metavar="PATTERN",
        help=(
            "目录模式下匹配 --windowed-root/<split>/ 下文件；默认由 --windowed-dataset 决定"
            "（contiguous→community_*.jsonl，monthly_chain→monthly_chain_community_*.jsonl）"
        ),
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
            "仅作图：按 mean_Q 裁剪尾部比例用户；0=不去极值；"
            "双侧默认最低/最高各 P；单侧见 --plot-trim-sides；"
            "省略则用 config.PLOT_TRIM_EACH_TAIL；不写回 jsonl"
        ),
    )
    parser.add_argument(
        "--plot-trim-sides",
        choices=("both", "lower", "upper"),
        default="both",
        help=(
            "与 --plot-trim-each-tail 配合：both=最低与最高各去掉该比例；"
            "lower=只去掉最低比例（保留高分侧）；upper=只去掉最高比例"
        ),
    )
    parser.add_argument(
        "--plot-trim-scope",
        choices=("user", "step"),
        default="user",
        help=(
            "仅作图：user=按 mean_Q 整行删用户后聚合；"
            "step=每链上窗口内去尾后再聚合（见 --plot-step-trim-basis）"
        ),
    )
    parser.add_argument(
        "--plot-step-trim-basis",
        choices=("deviation", "value"),
        default="deviation",
        help="plot-trim-scope=step 时：deviation=Q−当步均值；value=当步 Q 分位",
    )
    parser.add_argument(
        "--always-accept-refinement",
        action="store_true",
        help="仅 clasp_online：每步精炼后始终采用新画像，不与旧画像比 Q；精炼为空则保留旧画像",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "断点续跑：读取当前输出 jsonl（无 error 的行视为已完成），跳过已有用户，仅追加未完成的方法；"
            "须与本次 --comparison-root / --output / output_stem 一致。"
        ),
    )
    parser.add_argument(
        "--no-profile-snapshots",
        action="store_true",
        help=(
            "关闭 Clasp 系画像快照（默认在运行列表含 clasp_online / clasp_online_no_hist 时，"
            "分别写入 <comparison-root>/<method>/profile_snapshots/<output_stem>/profiles.jsonl）"
        ),
    )
    parser.add_argument(
        "--no-action-prompt-observed-history",
        action="store_true",
        help=(
            "动作预测 prompt 不载入观测历史：不拼画像后的本窗行为块，"
            "也不在 Recent user actions 中使用历史滑窗（仍保留 Current scenario 中的待预测动作上下文）；"
            "对本次 --methods 列表中所有方法生效。"
            "若要在同一评测中并列「有/无观测历史」的 Clasp，请使用 --methods 含 "
            "clasp_online 与 clasp_online_no_hist（后者始终无观测历史，不受本开关反向打开历史）。"
        ),
    )
    parser.add_argument(
        "--no-three-window-evaluation",
        action="store_true",
        help=(
            "关闭链末三窗口评估（past/current/future 上旧画像 vs 新画像）；"
            "减少额外动作 API 调用，输出 jsonl 不含 three_window_evaluation"
        ),
    )
    args = parser.parse_args()

    from src.config import DPO_WORKERS as _DW
    from src.config import DPO_USER_PROCESSES as _DUP
    from src.config import PLOT_TRIM_EACH_TAIL as _PLOT_TRIM

    methods = _parse_methods(args.methods)
    comparison_root = Path(args.comparison_root).resolve()
    separate_by_method = not bool(args.combined_jsonl)

    file_glob_resolved = (
        args.file_glob
        if args.file_glob is not None
        else WINDOWED_DATASET_FILE_GLOBS[str(args.windowed_dataset)]
    )
    if args.input_jsonl is None:
        print(
            f"[BaselineChain] 窗口化数据集类型={args.windowed_dataset} → glob={file_glob_resolved}",
            flush=True,
        )

    if args.output is not None:
        raw_path = Path(args.output)
        base_stem = raw_path.stem if raw_path.stem else "baseline_chain"
        output_stem = _output_stem_with_dataset(base_stem, str(args.windowed_dataset))
        if raw_path.stem and output_stem != raw_path.stem:
            print(
                f"[BaselineChain] 输出文件名含数据集类型区分: "
                f"{raw_path.stem}.jsonl → {output_stem}.jsonl",
                flush=True,
            )
    else:
        tag = Path(args.input_jsonl).stem if args.input_jsonl is not None else args.split
        output_stem = f"baseline_chain_{tag}_{args.windowed_dataset}"

    out_single: Optional[Path] = None
    if not separate_by_method:
        if args.output is not None:
            raw_path = Path(args.output).resolve()
            out_single = raw_path.parent / f"{output_stem}{raw_path.suffix}"
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
        file_glob=file_glob_resolved,
        plot_path=args.plot,
        always_accept_refinement=bool(args.always_accept_refinement),
        plot_trim_each_tail=(
            float(_PLOT_TRIM if args.plot_trim_each_tail is None else args.plot_trim_each_tail)
            if args.plot is not None
            else None
        ),
        plot_trim_sides=str(args.plot_trim_sides),
        plot_trim_scope=str(args.plot_trim_scope),
        plot_step_trim_basis=str(args.plot_step_trim_basis),
        num_windows=args.num_windows,
        window_size=args.window_size,
        window_split_mode=str(args.window_split_mode),
        actions_per_month=args.actions_per_month,
        comparison_root=comparison_root,
        separate_by_method=separate_by_method,
        output_jsonl=out_single,
        output_stem=output_stem,
        resume=bool(args.resume),
        record_profile_snapshots=not bool(args.no_profile_snapshots),
        action_prompt_include_observed_history=not bool(args.no_action_prompt_observed_history),
        enable_three_window_evaluation=not bool(args.no_three_window_evaluation),
        max_users_per_community=int(args.max_users_per_community),
    )


if __name__ == "__main__":
    main()
