#!/usr/bin/env python3
"""
Parallelized baseline comparison evaluation

Parallelization strategy (reference dpo_pipeline.py):
1. Multi-process parallel processing of different users (ProcessPoolExecutor)
2. Within each user, multiple methods execute serially
3. Within each method, candidate persona evaluation uses thread pool (ThreadPoolExecutor)
"""

import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from comparison.baseline_resume import (
    filter_users_per_community,
    load_completed_keys_per_method,
    serialize_user_key,
)
from comparison.window_chain_eval import (
    CLASP_ONLINE_VARIANTS,
    CLASP_PROFILE_SNAPSHOT_FILENAME,
    evaluate_user_window_chain,
)
from src.config import DPO_USER_PROCESSES, DPO_WORKERS
from src.scorer import SemanticScorer


def _baseline_user_worker(job: tuple) -> tuple:
    """
    Subprocess worker function: evaluate all methods for a single user

    Args:
        job: (
            user_index, user_data, methods, workers, scorer_device, stagger_sec,
            completed_by_method, refinement_variants, always_accept_refinement,
            comparison_root_str, output_stem, record_profile_snapshots,
            action_prompt_include_observed_history,
            enable_three_window_evaluation,
        )

    Returns:
        (user_index, results_dict, elapsed_time)
    """
    (
        idx,
        user_data,
        methods,
        workers,
        scorer_device,
        stagger_sec,
        completed_by_method,
        refinement_variants,
        always_accept_refinement,
        comparison_root_str,
        output_stem,
        record_profile_snapshots,
        action_prompt_include_observed_history,
        enable_three_window_evaluation,
    ) = job

    # Stagger startup to reduce API burst
    if stagger_sec > 0:
        time.sleep(idx * stagger_sec)

    t0 = time.time()

    # Each subprocess loads its own SemanticScorer
    semantic_scorer = SemanticScorer(device=scorer_device)

    def _uk(u: Dict[str, Any]) -> str:
        return f"{u.get('user_id')}\t{u.get('community_id')}"

    ukey = _uk(user_data)

    comparison_root = Path(comparison_root_str)

    # Evaluate methods not yet completed for this user
    results = {}
    for method in methods:
        done = completed_by_method.get(method) or set()
        if ukey in done:
            continue
        try:
            snap_dir: Optional[Path] = None
            if record_profile_snapshots and method in CLASP_ONLINE_VARIANTS:
                snap_dir = comparison_root / method / "profile_snapshots" / output_stem
            r = evaluate_user_window_chain(
                user_data,
                method,
                semantic_scorer,
                profile_model=None,  # Use vLLM API
                profile_tokenizer=None,
                action_model=None,
                action_tokenizer=None,
                refinement_variants=int(refinement_variants),
                workers=workers,
                always_accept_refinement=bool(always_accept_refinement),
                profile_snapshot_dir=snap_dir,
                action_prompt_include_observed_history=bool(
                    action_prompt_include_observed_history
                ),
                enable_three_window_evaluation=bool(
                    enable_three_window_evaluation
                ),
            )
            results[method] = r
        except Exception as e:
            results[method] = {
                "user_id": user_data.get("user_id"),
                "community_id": user_data.get("community_id"),
                "method": method,
                "error": f"{type(e).__name__}: {str(e)}",
                "steps": [],
            }

    elapsed = time.time() - t0
    return idx, results, elapsed


def run_baseline_comparison_parallel(
    input_files: List[Path],
    methods: List[str],
    output_stem: str,
    comparison_root: Path,
    max_users: Optional[int] = None,
    workers: int = DPO_WORKERS,
    user_processes: int = DPO_USER_PROCESSES,
    user_process_stagger_sec: float = 0.5,
    scorer_device: str = "cpu",
    split: str = "test",
    *,
    resume: bool = False,
    refinement_variants: int = 1,
    always_accept_refinement: bool = False,
    record_profile_snapshots: bool = True,
    action_prompt_include_observed_history: bool = True,
    enable_three_window_evaluation: bool = True,
    max_users_per_community: int = 100,
) -> None:
    """
    Parallelized baseline comparison evaluation

    Args:
        input_files: List of input files
        methods: List of evaluation methods
        output_stem: Output filename prefix
        comparison_root: Output root directory
        max_users: Maximum number of users
        workers: Number of candidate evaluation threads per user
        user_processes: Number of parallel user processes
        user_process_stagger_sec: Process startup stagger time (seconds)
        scorer_device: Semantic scorer device (cpu/cuda)
        split: Dataset split
        resume: When True, don't overwrite existing jsonl, only append incomplete users; skip completed rows per method file
        refinement_variants / always_accept_refinement: Consistent with serial CLI
        record_profile_snapshots: When True and methods contain any ``CLASP_ONLINE_VARIANTS`` (clasp_online / no_hist / *_ablate_*), write persona snapshot directories under each method
        action_prompt_include_observed_history: When False, action prompt excludes observed history (consistent with serial CLI)
        enable_three_window_evaluation: When False, skip three-window evaluation at chain end (consistent with serial CLI)
        max_users_per_community: When >0, keep only first K users per community (input order); 0 means no limit (consistent with serial CLI)
    """
    print(f"\n[BaselineChain] Parallel evaluation started", flush=True)
    print(f"  Methods: {', '.join(methods)}", flush=True)
    print(f"  User processes: {user_processes}", flush=True)
    print(f"  Candidate evaluation threads: {workers}", flush=True)
    print(f"  Semantic scorer: {scorer_device}", flush=True)
    if max_users_per_community > 0:
        print(
            f"  Max users per community: {max_users_per_community} (0=no limit)",
            flush=True,
        )

    # Load all users
    users = []
    for fp_in in input_files:
        with fp_in.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    user = json.loads(line)
                    user["source_file"] = fp_in.name
                    users.append(user)
                except json.JSONDecodeError:
                    continue

    n_loaded = len(users)
    users = filter_users_per_community(users, max_users_per_community)
    if max_users_per_community > 0 and len(users) < n_loaded:
        print(
            f"[BaselineChain] Trim by community: {n_loaded} -> {len(users)} users "
            f"(≤{max_users_per_community} per community)",
            flush=True,
        )

    if max_users:
        users = users[:max_users]

    if record_profile_snapshots:
        for sm in methods:
            if sm not in CLASP_ONLINE_VARIANTS:
                continue
            d = comparison_root / sm / "profile_snapshots" / output_stem
            d.mkdir(parents=True, exist_ok=True)
            _p = d / CLASP_PROFILE_SNAPSHOT_FILENAME
            if not resume:
                _p.unlink(missing_ok=True)
            print(
                f"[BaselineChain] {sm} persona snapshots (single file, resume={'append' if resume else 'new file'}): {_p}",
                flush=True,
            )

    method_paths: Dict[str, Path] = {}
    for m in methods:
        method_dir = comparison_root / m
        method_dir.mkdir(parents=True, exist_ok=True)
        method_paths[m] = method_dir / f"{output_stem}.jsonl"

    completed_by_m: Dict[str, Set[str]] = {m: set() for m in methods}
    if resume:
        completed_by_m = load_completed_keys_per_method(
            separate_by_method=True,
            methods=methods,
            method_paths=method_paths,
            combined_jsonl=None,
        )
        stats = {m: len(completed_by_m.get(m) or set()) for m in methods}
        print(f"[BaselineChain] --resume: Completed users per method {stats}", flush=True)

    def _needs_work(user: Dict[str, Any]) -> bool:
        uk = serialize_user_key(user)
        return any(uk not in (completed_by_m.get(m) or set()) for m in methods)

    users_work = [u for u in users if _needs_work(u)]
    n_users = len(users_work)
    print(f"[BaselineChain] Users pending evaluation: {n_users} (total loaded {len(users)})", flush=True)

    if n_users == 0:
        print("[BaselineChain] No users pending evaluation, exiting", flush=True)
        return

    # Create output directories and file handles (don't truncate on resume)
    method_files = {}
    for m in methods:
        mp = method_paths[m]
        if not resume:
            mp.open("w", encoding="utf-8").close()
        method_files[m] = mp.open("a", encoding="utf-8")

    print(f"[BaselineChain] Output directory: {comparison_root}", flush=True)

    # Prepare parallel tasks
    effective_procs = min(user_processes, n_users)
    jobs = [
        (
            i,
            user,
            methods,
            workers,
            scorer_device,
            user_process_stagger_sec,
            completed_by_m,
            refinement_variants,
            always_accept_refinement,
            str(comparison_root.resolve()),
            output_stem,
            record_profile_snapshots,
            action_prompt_include_observed_history,
            enable_three_window_evaluation,
        )
        for i, user in enumerate(users_work)
    ]

    print(f"[BaselineChain] Starting {effective_procs} parallel processes...", flush=True)

    # Execute parallel evaluation
    run_start = time.time()
    done_count = 0

    try:
        with ProcessPoolExecutor(max_workers=effective_procs) as pool:
            futs = {pool.submit(_baseline_user_worker, job): job[0] for job in jobs}

            for fut in as_completed(futs):
                idx, results, user_elapsed = fut.result()
                done_count += 1

                # Write results immediately
                for method, result in results.items():
                    result["split"] = split
                    method_files[method].write(json.dumps(result, ensure_ascii=False) + "\n")
                    method_files[method].flush()
                    os.fsync(method_files[method].fileno())

                # Print progress
                batch_elapsed = time.time() - run_start
                pct = 100.0 * done_count / n_users
                # Wall clock total elapsed / completed users (consistent with ETA's "average interval per completion")
                avg_per_user = batch_elapsed / done_count
                if done_count < n_users:
                    eta_sec = avg_per_user * (n_users - done_count)
                    eta_str = f"{int(eta_sec // 60)}m{int(eta_sec % 60)}s"
                else:
                    eta_str = "—"

                if results:
                    user_id = next(iter(results.values())).get("user_id", "?")
                else:
                    user_id = users_work[idx].get("user_id", "?") if 0 <= idx < len(users_work) else "?"

                print(
                    f"[{done_count}/{n_users}] ({pct:.1f}%) "
                    f"user={user_id} | "
                    f"this_user={user_elapsed:.1f}s | "
                    f"avg_user={avg_per_user:.1f}s | "
                    f"total_elapsed={int(batch_elapsed)}s | "
                    f"ETA={eta_str}",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n[BaselineChain] Received interrupt (Ctrl+C), shutting down...", flush=True)
        raise

    finally:
        # Close all files
        for f in method_files.values():
            f.close()

    total_time = time.time() - run_start
    print(f"\n[BaselineChain] ✅ Done!", flush=True)
    print(f"  Processed users: {n_users}", flush=True)
    print(f"  Total elapsed: {int(total_time // 60)}m{int(total_time % 60)}s", flush=True)
    print(f"  Average per user: {total_time / n_users:.1f}s", flush=True)

    # Print output files for each method
    print(f"\n[BaselineChain] Output files:", flush=True)
    for m in methods:
        print(f"  {m}: {method_paths[m]}", flush=True)
