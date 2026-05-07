#!/usr/bin/env python3
"""
并行化优化的基线对比评估

参考 dpo_pipeline.py 的并行化策略：
1. 多进程并行处理不同用户（ProcessPoolExecutor）
2. 每个用户内部，多个方法串行执行
3. 每个方法内部，候选画像评估使用线程池（ThreadPoolExecutor）
"""

import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from comparison.baseline_resume import load_completed_keys_per_method, serialize_user_key
from comparison.window_chain_eval import evaluate_user_window_chain
from src.config import DPO_USER_PROCESSES, DPO_WORKERS
from src.scorer import SemanticScorer


def _baseline_user_worker(job: tuple) -> tuple:
    """
    子进程工作函数：处理单个用户的所有方法评估

    Args:
        job: (
            user_index, user_data, methods, workers, scorer_device, stagger_sec,
            completed_by_method, refinement_variants, always_accept_refinement,
            clasp_profile_snapshot_dir, action_prompt_include_observed_history,
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
        clasp_profile_snapshot_dir,
        action_prompt_include_observed_history,
    ) = job

    # 错开启动，减轻 API 洪峰
    if stagger_sec > 0:
        time.sleep(idx * stagger_sec)

    t0 = time.time()

    # 每个子进程加载自己的 SemanticScorer
    semantic_scorer = SemanticScorer(device=scorer_device)

    def _uk(u: Dict[str, Any]) -> str:
        return f"{u.get('user_id')}\t{u.get('community_id')}"

    ukey = _uk(user_data)

    snap_dir: Optional[Path] = (
        Path(clasp_profile_snapshot_dir)
        if clasp_profile_snapshot_dir
        else None
    )

    # 对该用户评估尚未完成的方法
    results = {}
    for method in methods:
        done = completed_by_method.get(method) or set()
        if ukey in done:
            continue
        try:
            r = evaluate_user_window_chain(
                user_data,
                method,
                semantic_scorer,
                profile_model=None,  # 使用 vLLM API
                profile_tokenizer=None,
                action_model=None,
                action_tokenizer=None,
                refinement_variants=int(refinement_variants),
                workers=workers,
                always_accept_refinement=bool(always_accept_refinement),
                profile_snapshot_dir=(
                    snap_dir if method == "clasp_online" else None
                ),
                action_prompt_include_observed_history=bool(
                    action_prompt_include_observed_history
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
) -> None:
    """
    并行化的基线对比评估

    Args:
        input_files: 输入文件列表
        methods: 评估方法列表
        output_stem: 输出文件名前缀
        comparison_root: 输出根目录
        max_users: 最大用户数
        workers: 每个用户内部的候选评估线程数
        user_processes: 并行处理的用户进程数
        user_process_stagger_sec: 进程启动错开时间（秒）
        scorer_device: 语义评分器设备（cpu/cuda）
        split: 数据集划分
        resume: True 时不覆盖已有 jsonl，仅追加未完成用户；按各 method 文件跳过已成功行
        refinement_variants / always_accept_refinement: 与串行 CLI 一致
        record_profile_snapshots: True 且 methods 含 clasp_online 时写入画像快照目录
        action_prompt_include_observed_history: False 时动作 prompt 不含观测历史（与串行 CLI 一致）
    """
    print(f"\n[BaselineChain] 并行化评估启动", flush=True)
    print(f"  方法: {', '.join(methods)}", flush=True)
    print(f"  用户进程数: {user_processes}", flush=True)
    print(f"  候选评估线程数: {workers}", flush=True)
    print(f"  语义评分器: {scorer_device}", flush=True)

    # 加载所有用户
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

    if max_users:
        users = users[:max_users]

    clasp_snap_dir: Optional[Path] = None
    clasp_snap_str: Optional[str] = None
    if record_profile_snapshots and "clasp_online" in methods:
        clasp_snap_dir = (
            comparison_root / "clasp_online" / "profile_snapshots" / output_stem
        )
        clasp_snap_dir.mkdir(parents=True, exist_ok=True)
        clasp_snap_str = str(clasp_snap_dir.resolve())
        print(
            f"[BaselineChain] clasp_online 画像快照目录: {clasp_snap_dir}",
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
        print(f"[BaselineChain] --resume: 各 method 已成功用户数 {stats}", flush=True)

    def _needs_work(user: Dict[str, Any]) -> bool:
        uk = serialize_user_key(user)
        return any(uk not in (completed_by_m.get(m) or set()) for m in methods)

    users_work = [u for u in users if _needs_work(u)]
    n_users = len(users_work)
    print(f"[BaselineChain] 待评估用户: {n_users}（总加载 {len(users)}）", flush=True)

    if n_users == 0:
        print("[BaselineChain] 无待评估用户，退出", flush=True)
        return

    # 创建输出目录与文件句柄（resume 时不截断）
    method_files = {}
    for m in methods:
        mp = method_paths[m]
        if not resume:
            mp.open("w", encoding="utf-8").close()
        method_files[m] = mp.open("a", encoding="utf-8")

    print(f"[BaselineChain] 输出目录: {comparison_root}", flush=True)

    # 准备并行任务
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
            clasp_snap_str,
            action_prompt_include_observed_history,
        )
        for i, user in enumerate(users_work)
    ]

    print(f"[BaselineChain] 启动 {effective_procs} 个并行进程...", flush=True)

    # 执行并行评估
    run_start = time.time()
    done_count = 0

    try:
        with ProcessPoolExecutor(max_workers=effective_procs) as pool:
            futs = {pool.submit(_baseline_user_worker, job): job[0] for job in jobs}

            for fut in as_completed(futs):
                idx, results, user_elapsed = fut.result()
                done_count += 1

                # 立即写入结果
                for method, result in results.items():
                    result["split"] = split
                    method_files[method].write(json.dumps(result, ensure_ascii=False) + "\n")
                    method_files[method].flush()
                    os.fsync(method_files[method].fileno())

                # 打印进度
                batch_elapsed = time.time() - run_start
                pct = 100.0 * done_count / n_users
                if done_count < n_users:
                    avg_time = batch_elapsed / done_count
                    eta_sec = avg_time * (n_users - done_count)
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
                    f"用时={user_elapsed:.1f}s | "
                    f"总耗时={int(batch_elapsed)}s | "
                    f"ETA={eta_str}",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\n[BaselineChain] 收到中断 (Ctrl+C)，正在结束...", flush=True)
        raise

    finally:
        # 关闭所有文件
        for f in method_files.values():
            f.close()

    total_time = time.time() - run_start
    print(f"\n[BaselineChain] ✅ 完成！", flush=True)
    print(f"  处理用户: {n_users}", flush=True)
    print(f"  总耗时: {int(total_time // 60)}m{int(total_time % 60)}s", flush=True)
    print(f"  平均每用户: {total_time / n_users:.1f}s", flush=True)

    # 打印各方法的输出文件
    print(f"\n[BaselineChain] 输出文件:", flush=True)
    for m in methods:
        print(f"  {m}: {method_paths[m]}", flush=True)
