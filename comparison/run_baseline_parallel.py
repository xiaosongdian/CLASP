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
from typing import Any, Dict, List, Optional

from comparison.window_chain_eval import evaluate_user_window_chain
from src.config import DPO_USER_PROCESSES, DPO_WORKERS
from src.scorer import SemanticScorer


def _baseline_user_worker(job: tuple) -> tuple:
    """
    子进程工作函数：处理单个用户的所有方法评估

    Args:
        job: (user_index, user_data, methods, workers, scorer_device, stagger_sec)

    Returns:
        (user_index, results_dict, elapsed_time)
    """
    idx, user_data, methods, workers, scorer_device, stagger_sec = job

    # 错开启动，减轻 API 洪峰
    if stagger_sec > 0:
        time.sleep(idx * stagger_sec)

    t0 = time.time()

    # 每个子进程加载自己的 SemanticScorer
    semantic_scorer = SemanticScorer(device=scorer_device)

    # 对该用户评估所有方法
    results = {}
    for method in methods:
        try:
            r = evaluate_user_window_chain(
                user_data,
                method,
                semantic_scorer,
                profile_model=None,  # 使用 vLLM API
                profile_tokenizer=None,
                action_model=None,
                action_tokenizer=None,
                refinement_variants=1,
                workers=workers,
                always_accept_refinement=False,
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

    n_users = len(users)
    print(f"[BaselineChain] 加载 {n_users} 个用户", flush=True)

    if n_users == 0:
        print("[BaselineChain] 无用户数据，退出", flush=True)
        return

    # 创建输出目录
    comparison_root.mkdir(parents=True, exist_ok=True)
    method_paths = {}
    method_files = {}

    for m in methods:
        method_dir = comparison_root / m
        method_dir.mkdir(parents=True, exist_ok=True)
        method_path = method_dir / f"{output_stem}.jsonl"
        method_paths[m] = method_path
        # 清空输出文件
        method_path.open("w", encoding="utf-8").close()
        method_files[m] = method_path.open("a", encoding="utf-8")

    print(f"[BaselineChain] 输出目录: {comparison_root}", flush=True)

    # 准备并行任务
    effective_procs = min(user_processes, n_users)
    jobs = [
        (i, user, methods, workers, scorer_device, user_process_stagger_sec)
        for i, user in enumerate(users)
    ]

    print(f"[BaselineChain] 启动 {effective_procs} 个并行进程...", flush=True)

    # 执行并行评估
    run_start = time.time()
    done_count = 0
    results_by_idx = {}

    try:
        with ProcessPoolExecutor(max_workers=effective_procs) as pool:
            futs = {pool.submit(_baseline_user_worker, job): job[0] for job in jobs}

            for fut in as_completed(futs):
                idx, results, user_elapsed = fut.result()
                done_count += 1
                results_by_idx[idx] = results

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

                user_id = results[methods[0]].get("user_id", "?")
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
