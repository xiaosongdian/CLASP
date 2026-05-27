#!/usr/bin/env python3

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

WINDOWED_DATASET_FILE_GLOBS = {
    "contiguous": "community_*.jsonl",
    "monthly_chain": "monthly_chain_community_*.jsonl",
}


def _output_stem_with_dataset(base_stem: str, windowed_dataset: str) -> str:
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
        print(f"[BaselineChain] Unknown method: {bad}, available: {sorted(VALID_METHODS)}", flush=True)
        sys.exit(1)
    return parts


def _print_aggregate(rows: List[Dict[str, Any]]) -> None:
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("error"):
            continue
        m = r.get("method")
        if m:
            by_method[str(m)].append(r)

    print("[BaselineChain] ========== Summary (skipping records with errors) ==========", flush=True)
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
            f"  [{method}] num_users={n} | avg mean_Q(all forward steps)={mean_overall:.4f} | "
            f"mean_Q_chain(same as above, compat field)={mean_chain:.4f} | per-step avg Q={step_q_only}",
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
            "[BaselineChain] Clasp family (clasp_online / *_ablate_*) with --always-accept-refinement: "
            "Always adopt new refined persona at each step (keep old if empty), no Q comparison.",
            flush=True,
        )

    if not action_prompt_include_observed_history:
        print(
            "[BaselineChain] --no-action-prompt-observed-history: "
            "Action prediction prompt excludes observed history (only persona + Current scenario).",
            flush=True,
        )

    if not enable_three_window_evaluation:
        print(
            "[BaselineChain] --no-three-window-evaluation: "
            "Skip three-window comparison at chain end (no three_window_evaluation in jsonl).",
            flush=True,
        )

    if max_users_per_community > 0:
        print(
            f"[BaselineChain] Max users per community: {max_users_per_community} "
            f"(skip excess in input order; 0=no limit)",
            flush=True,
        )

    if input_jsonl is not None:
        input_jsonl = Path(input_jsonl).resolve()
        if not input_jsonl.is_file():
            print(f"[BaselineChain] Not a file: {input_jsonl}", flush=True)
            sys.exit(1)
        files = [input_jsonl]
    else:
        raw_split_dir = data_dir / split
        if not raw_split_dir.is_dir():
            print(f"[BaselineChain] Directory does not exist: {raw_split_dir}", flush=True)
            sys.exit(1)

        out_split = windowed_root / split
        if not skip_window_split:
            out_split.mkdir(parents=True, exist_ok=True)
            print(f"[BaselineChain] Window splitting: {raw_split_dir} -> {out_split}", flush=True)
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
                        "[BaselineChain] monthly_chain requires valid actions per window: please set "
                        "--actions-per-month, or use default --window-size (equals config.WINDOW_SIZE).",
                        flush=True,
                    )
                    sys.exit(1)
                print(
                    f"[BaselineChain] Split mode=monthly_chain: {MONTHLY_CHAIN_NUM_MONTHS} consecutive natural months, "
                    f"{MONTHLY_CHAIN_WINDOWS_PER_MONTH} windows per month, {apm} actions per window "
                    f"(total {apm * NUM_WINDOWS_EVAL_CHAIN} actions, {NUM_WINDOWS_EVAL_CHAIN} windows W0..W5)",
                    flush=True,
                )
            else:
                print(
                    f"[BaselineChain] Split mode=contiguous: {ws} actions per window, "
                    f"num_windows={nw} (W0..W{nw - 1})",
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
                    f"[BaselineChain] --skip-window-split specified but missing: {out_split}",
                    flush=True,
                )
                sys.exit(1)

        files = sorted(out_split.glob(file_glob))
        if not files:
            print(
                f"[BaselineChain] No matching files: {out_split}/{file_glob}",
                flush=True,
            )
            sys.exit(1)

    if not skip_preflight and not preflight_check(comparison_methods=methods):
        print("[BaselineChain] Preflight check failed", flush=True)
        sys.exit(1)

    do_parallel = bool(use_parallel and user_processes > 1)
    if resume and do_parallel:
        print("[BaselineChain] --resume: Parallel mode will append and skip completed users.", flush=True)

    # Decide whether to use parallelization
    if do_parallel:
        print(f"[BaselineChain] Using parallel mode: {user_processes} processes", flush=True)
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

        # Parallel mode does not support plotting; notify user if plotting is requested
        if plot_path is not None:
            print("[BaselineChain] Note: Parallel mode does not support plotting, please use serial mode (--no-parallel)", flush=True)

        return

    # Serial mode
    print(f"[BaselineChain] Using serial mode", flush=True)

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
                    f"[BaselineChain] {sm} persona snapshots (single file): {_snap_fp}",
                    flush=True,
                )
            else:
                print(
                    f"[BaselineChain] {sm} persona snapshots append: {_snap_fp}",
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
            "[BaselineChain] Output by method directory: "
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
        print(f"[BaselineChain] --resume: Completed users per method (will skip): {stats}", flush=True)

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
                            f"[BaselineChain] Scanned {total_lines} input users "
                            f"(this run wrote {len(new_rows)} new lines)...",
                            flush=True,
                        )

            if max_users is not None and total_lines >= max_users:
                break

    all_rows = prior_rows + new_rows

    dt = time.time() - t0
    if separate_by_method:
        print(
            f"[BaselineChain] Done: Scanned {total_lines} input users, this run wrote {len(new_rows)} new lines, "
            f"total {len(all_rows)} lines, elapsed {dt:.1f}s; see {output_stem}.jsonl under each method directory",
            flush=True,
        )
    else:
        print(
            f"[BaselineChain] Done: Scanned {total_lines} input users, this run wrote {len(new_rows)} new lines, "
            f"total {len(all_rows)} lines, elapsed {dt:.1f}s -> {output_jsonl}",
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
                    f"[BaselineChain] Plot outlier removal mean_Q "
                    f"(sides={plot_trim_sides}) {trim_t*100:.1f}%: "
                    f"dropped={tmeta.get('dropped', 0)} plot_users={len(sub_plot)}/{len(sub)}",
                    flush=True,
                )
                if not sub_plot:
                    sub_plot = sub
            elif trim_t is not None and trim_t > 0 and scope == "step":
                print(
                    f"[BaselineChain] Plot per-step trim (sides={plot_trim_sides}) "
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
                print(f"[BaselineChain] Saved: {p}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test set window chain: multi-baseline F/L/Q (no DPO construction)"
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Subdirectory name under data, e.g. test / eval_unseen",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Data root directory containing raw jsonl files",
    )
    parser.add_argument(
        "--windowed-root",
        type=Path,
        default=ROOT / "output" / "windowed_eval_chain",
        help=(
            "Windowed jsonl root directory: reads <root>/<split>/; filename determined by "
            "--windowed-dataset or --file-glob (e.g. output/windowed/test)"
        ),
    )
    parser.add_argument(
        "--comparison-root",
        type=Path,
        default=ROOT / "output" / "comparison",
        help="Evaluation results root directory; by default writes to <root>/<method>/...",
    )
    parser.add_argument(
        "--combined-jsonl",
        action="store_true",
        help="Write all methods to same jsonl (legacy behavior); default is separate directory+file per method",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Combined mode (--combined-jsonl): full output file path (custom filename, suggest including dataset type to avoid confusion). "
            "Separate directory mode: used as output filename stem only (default baseline_chain_<split|input_stem>_<windowed-dataset>), "
            "actual path is <comparison-root>/<method>/<stem>.jsonl"
        ),
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="static_s0,prefix_refresh,clasp_online",
        help=f"Comma-separated, available: {','.join(sorted(VALID_METHODS))}",
    )
    parser.add_argument("--max-users", type=int, default=None, help="Max number of users to evaluate (raw user count)")
    parser.add_argument(
        "--max-users-per-community",
        type=int,
        default=100,
        help=(
            "Evaluate only first K users per community_id (by appearance order in each input file); "
            "saves API time. Default 100; set to 0 for no limit"
        ),
    )
    parser.add_argument(
        "--refinement-variants",
        "--num-candidates",
        type=int,
        default=None,
        dest="refinement_variants",
        help="clasp_online refinement iterations per step (default 1=single correction; >1 for ablation, DPO training needs multiple)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Candidate persona thread count; default config.DPO_WORKERS",
    )
    parser.add_argument(
        "--user-processes",
        type=int,
        default=None,
        help="Number of user processes for parallel processing; default config.DPO_USER_PROCESSES (multi-process acceleration)",
    )
    parser.add_argument(
        "--user-process-stagger",
        type=float,
        default=0.5,
        help="Multi-process startup stagger time (seconds) to reduce API burst; default 0.5s",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable multi-process parallelism, use serial mode (for debugging)",
    )
    parser.add_argument(
        "--scorer-device",
        default="cpu",
        help="SentenceTransformer semantic scorer device (default cpu to avoid GPU memory contention with vLLM); can set cuda, cuda:0",
    )
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip vLLM / ST preflight check",
    )
    parser.add_argument(
        "--skip-window-split",
        action="store_true",
        help="Skip window splitting, directly use --windowed-root/<split>",
    )
    parser.add_argument(
        "--num-windows",
        type=int,
        default=None,
        help="Number of windows for window splitting; default config.NUM_WINDOWS_EVAL_CHAIN (6=W0..W5)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Actions per window for window splitting; default config.WINDOW_SIZE (contiguous); for monthly_chain can serve as fallback default for actions per month",
    )
    parser.add_argument(
        "--window-split-mode",
        choices=("contiguous", "monthly_chain"),
        default="contiguous",
        help=(
            "Window splitting strategy: contiguous=sequential chunks; "
            "monthly_chain=consecutive natural months, evenly sampled per month (see --actions-per-month)"
        ),
    )
    parser.add_argument(
        "--actions-per-month",
        type=int,
        default=None,
        metavar="N",
        help=(
            "monthly_chain only: actions per time window (default matches --window-size or config.WINDOW_SIZE); "
            "total window count fixed at config.NUM_WINDOWS_EVAL_CHAIN (default 6=6 consecutive months×1 window per month)."
        ),
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=None,
        help="Directly specify a single windowed jsonl (data/<split> not required to exist)",
    )
    parser.add_argument(
        "--windowed-dataset",
        choices=tuple(WINDOWED_DATASET_FILE_GLOBS.keys()),
        default="contiguous",
        help=(
            "Windowed test set type; determines matching pattern when --file-glob not specified: "
            "contiguous=sequential chunks community_*.jsonl; "
            "monthly_chain=natural month chain monthly_chain_community_*.jsonl (see scripts/build_monthly_chain_windowed.py)"
        ),
    )
    parser.add_argument(
        "--file-glob",
        type=str,
        default=None,
        metavar="PATTERN",
        help=(
            "Directory mode file matching pattern for --windowed-root/<split>/; default determined by --windowed-dataset "
            "(contiguous→community_*.jsonl, monthly_chain→monthly_chain_community_*.jsonl)"
        ),
    )
    parser.add_argument(
        "--plot",
        type=Path,
        default=None,
        help="Save F, L, Q three line plots (requires matplotlib); e.g. output/c.png -> c_F.png, c_L.png, c_Q.png",
    )
    parser.add_argument(
        "--plot-trim-each-tail",
        type=float,
        default=None,
        metavar="P",
        help=(
            "Plot only: trim tail fraction of users by mean_Q; 0=no outlier removal; "
            "default both sides remove P% from low/high; single side see --plot-trim-sides; "
            "omit to use config.PLOT_TRIM_EACH_TAIL; not written back to jsonl"
        ),
    )
    parser.add_argument(
        "--plot-trim-sides",
        choices=("both", "lower", "upper"),
        default="both",
        help=(
            "With --plot-trim-each-tail: both=remove that fraction from low and high; "
            "lower=remove only low fraction (keep high scores); upper=remove only high fraction"
        ),
    )
    parser.add_argument(
        "--plot-trim-scope",
        choices=("user", "step"),
        default="user",
        help=(
            "Plot only: user=delete entire rows by mean_Q then aggregate; "
            "step=trim tails within each window on chain then aggregate (see --plot-step-trim-basis)"
        ),
    )
    parser.add_argument(
        "--plot-step-trim-basis",
        choices=("deviation", "value"),
        default="deviation",
        help="plot-trim-scope=step: deviation=Q−step mean; value=step Q quantile",
    )
    parser.add_argument(
        "--always-accept-refinement",
        action="store_true",
        help="clasp_online only: always adopt new persona after refinement, no Q comparison with old; keep old if refinement is empty",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from checkpoint: read current output jsonl (rows without error treated as completed), skip existing users, only append incomplete methods; "
            "must match this run's --comparison-root / --output / output_stem."
        ),
    )
    parser.add_argument(
        "--no-profile-snapshots",
        action="store_true",
        help=(
            "Disable Clasp family persona snapshots (default enabled when run list contains clasp_online / clasp_online_no_hist / "
            "clasp_online_ablate_*, writes to <comparison-root>/<method>/profile_snapshots/<output_stem>/profiles.jsonl respectively)"
        ),
    )
    parser.add_argument(
        "--no-action-prompt-observed-history",
        action="store_true",
        help=(
            "Action prediction prompt excludes observed history: skip behavior block after persona in this window, "
            "also skip historical sliding window in Recent user actions (still keep action context in Current scenario); "
            "applies to all methods in this --methods list. "
            "To compare 'with/without observed history' Clasp in same evaluation, use --methods containing both "
            "clasp_online and clasp_online_no_hist (latter always excludes history, unaffected by this switch)."
        ),
    )
    parser.add_argument(
        "--no-three-window-evaluation",
        action="store_true",
        help=(
            "Disable three-window evaluation at chain end (past/current/future old persona vs new persona); "
            "reduces extra action API calls, output jsonl omits three_window_evaluation"
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
            f"[BaselineChain] {args.windowed_dataset} → glob={file_glob_resolved}",
            flush=True,
        )

    if args.output is not None:
        raw_path = Path(args.output)
        base_stem = raw_path.stem if raw_path.stem else "baseline_chain"
        output_stem = _output_stem_with_dataset(base_stem, str(args.windowed_dataset))
        if raw_path.stem and output_stem != raw_path.stem:
            print(
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
