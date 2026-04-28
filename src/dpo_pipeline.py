#!/usr/bin/env python3
"""
DPO 全流程 Pipeline
步骤：
  1. 加载窗口化数据
  2. 用 W0 生成初始画像 S0
  3. 用 S0 在 W1 上预测，计算 base Q(S0)
  4. 构建偏差信号，生成 N=15 候选画像
  5. 对每个候选画像在 W0/W1/W2 上评分
  6. 构造 DPO 正负对并保存
"""

import argparse
import json
import multiprocessing
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import src.config as cfg
from src.config import (
    ACTION_API_BASE,
    ACTION_API_MODEL,
    ALPHA,
    ABS_DELTA,
    DELTA,
    DPO_ROUNDS,
    DPO_USER_PROCESSES,
    DPO_USER_PROCESS_STAGGER_SEC,
    DPO_WORKERS,
    NUM_CANDIDATE_PROFILES,
    PROFILE_API_BASE,
    PROFILE_API_MODEL,
    SENTENCE_TRANSFORMER_MODEL,
    TAU_MINUS,
    TAU_PLUS,
    TEMPERATURE_ACTION,
)
from src.action_predictor import (
    build_behavior_discrepancies,
    predict_actions_for_window,
)
from src.profile_generator import (
    generate_candidate_profiles,
    generate_initial_profile,
    profile_candidate_source,
)
from src.scorer import SemanticScorer, evaluate_predictions


def _format_duration_s(sec: float) -> str:
    """将秒数格式化为易读短字符串；无效输入返回 "—"."""
    if sec is None or sec != sec or sec < 0:  # nan or neg
        return "—"
    if sec < 60:
        return f"{sec:.1f}s"
    if sec < 3600:
        m, s = divmod(sec, 60.0)
        return f"{int(m)}m{s:.0f}s"
    h, r = divmod(sec, 3600.0)
    m, s = divmod(r, 60.0)
    return f"{int(h)}h{int(m)}m{s:.0f}s"


def _dbg_print(*args, **kwargs) -> None:
    if cfg.DEBUG_LLM:
        kwargs.setdefault("flush", True)
        print(*args, **kwargs)


def _print_user_finish_report(
    result: Dict[str, Any],
    done_in_batch: int,
    total_in_batch: int,
    user_elapsed_sec: float,
    batch_elapsed_sec: float,
) -> None:
    """
    非 debug 模式下每完成一个用户打印一行：进度、DPO 对数/候选/各轮、本用户与本批耗时、ETA。
    """
    n_pairs = int(result.get("num_dpo_pairs", 0) or 0)
    n_cand = int(result.get("num_candidates", 0) or 0)
    rdist = result.get("round_pair_distribution")
    if rdist is None and result.get("rounds"):
        rsum = result["rounds"]
        rdist = {f"round_{x['round_idx']}": x.get("num_dpo_pairs", 0) for x in rsum if isinstance(x, dict)}
    rdist = rdist or {}
    if total_in_batch <= 0:
        pct, eta_s = 0.0, 0.0
    else:
        pct = 100.0 * done_in_batch / total_in_batch
        rem = max(0, total_in_batch - done_in_batch)
        if rem > 0 and done_in_batch > 0 and batch_elapsed_sec > 0:
            avg = batch_elapsed_sec / done_in_batch
            eta_s = avg * rem
        else:
            eta_s = 0.0
    eta_str = (
        _format_duration_s(eta_s)
        if (total_in_batch and done_in_batch < total_in_batch)
        else "0s"
    )
    print(
        f"[Pipeline] 完成 {done_in_batch}/{total_in_batch} ({pct:.1f}%) | "
        f"DPO对={n_pairs} 末轮候选={n_cand} 各轮{rdist} | "
        f"本用户={_format_duration_s(user_elapsed_sec)} 本批={_format_duration_s(batch_elapsed_sec)} "
        f"预计剩余={eta_str}",
        flush=True,
    )


def _dpo_terminate_orphan_processes() -> None:
    """终止当前 Python 仍知的 multiprocessing 子进程，避免 Ctrl+C 后主进程退出后子进程仍存活。"""
    for p in list(multiprocessing.active_children()):
        try:
            p.terminate()
        except Exception:
            pass
    for p in list(multiprocessing.active_children()):
        try:
            p.join(timeout=2.0)
        except Exception:
            pass
    for p in list(multiprocessing.active_children()):
        if not p.is_alive():
            continue
        try:
            if hasattr(p, "kill"):
                p.kill()  # Py3.7+
        except Exception:
            pass
        try:
            p.join(timeout=1.0)
        except Exception:
            pass


def _dpo_shutdown_process_pool(
    pool: Optional[ProcessPoolExecutor],
    pending_futs: Any,
) -> None:
    """取消未起任务、关闭进程池；随后清理残留子进程（尽量温和→强制）。"""
    if pool is None:
        return
    if isinstance(pending_futs, dict):
        for fut in list(pending_futs.keys()):
            try:
                fut.cancel()
            except Exception:
                pass
    try:
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except TypeError:
            pool.shutdown(wait=False)
    except Exception:
        pass
    time.sleep(0.15)
    _dpo_terminate_orphan_processes()


# ============================================================================
# 启动前模型连通性检测
# ============================================================================

def _check_api(api_base: str, model_name: str, api_key: str = "not-needed", label: str = "") -> bool:
    """向 OpenAI 兼容 API 发送一个最小请求，检测连通性。"""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key=api_key, timeout=15)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
            temperature=0.01,
        )
        text = resp.choices[0].message.content.strip()
        print(f"  [✓] {label} 连通成功（响应: {text[:50]}）", flush=True)
        return True
    except Exception as e:
        print(f"  [✗] {label} 连通失败: {e}", flush=True)
        return False


def _check_local_path(path: str, label: str = "") -> bool:
    """检查本地模型路径是否存在。"""
    p = Path(path)
    if p.exists():
        print(f"  [✓] {label} 路径存在: {path}", flush=True)
        return True
    print(f"  [✗] {label} 路径不存在: {path}", flush=True)
    return False


def preflight_check() -> bool:
    """
    启动前检测：
    1. 两个 vLLM 服务（画像生成 + 动作预测）是否可访问
    2. Sentence Transformer 本地路径是否存在
    """
    print("\n[Preflight] 模型连通性检测 ...", flush=True)

    ok1 = _check_api(PROFILE_API_BASE, PROFILE_API_MODEL, label="画像生成模型")
    ok2 = _check_api(ACTION_API_BASE, ACTION_API_MODEL, label="动作预测模型")
    ok3 = _check_local_path(SENTENCE_TRANSFORMER_MODEL, "Sentence Transformer")

    all_ok = ok1 and ok2 and ok3

    if all_ok:
        print("[Preflight] 全部检测通过 ✓\n", flush=True)
    else:
        print("[Preflight] 存在不可用的服务，请检查后重试 ✗\n", flush=True)

    return all_ok


# ============================================================================
# 单个用户的窗口评估
# ============================================================================

def evaluate_profile_on_window(
    profile: str,
    history: List[Dict],
    targets: List[Dict],
    action_model,
    action_tokenizer,
    semantic_scorer: SemanticScorer,
    profile_suffix: Optional[str] = None,
) -> Tuple[float, float, float]:
    """
    在单个窗口上评估画像，返回 (F, L, Q)。
    history: 前序窗口动作（作为上下文）
    targets: 目标窗口动作（待预测）
    profile_suffix: 见 action_predictor.predict_actions_for_window
    """
    predictions = predict_actions_for_window(
        action_model, action_tokenizer,
        profile, history, targets,
        temperature=TEMPERATURE_ACTION,
        profile_suffix=profile_suffix,
    )
    return evaluate_predictions(predictions, targets, semantic_scorer, ALPHA)


# ============================================================================
# DPO 对构造
# ============================================================================

def construct_dpo_pairs(
    s0_profile: str,
    candidates: List[str],
    s0_scores: Dict[str, Tuple[float, float, float]],
    candidate_scores: List[Dict[str, Tuple[float, float, float]]],
    *,
    baseline_profile_source: str = "base",
    behavior_discrepancies: str = "",
) -> List[Dict]:
    """
    构造 DPO 对。

    每条样本含 chosen/rejected 的 profile_source（base | commercial），
    以及 baseline_profile_source（首轮 S0 为 base；后续轮为上一轮最佳候选的来源）。
    behavior_discrepancies：本轮用于精炼的预测-真实行为偏差全文，写入每条样本的 discrepancies，
    便于后续 SFT / 带条件的 DPO 拼 prompt（与 generate_candidate_profiles 入参一致）。

    s0_scores: {"W0": (F,L,Q), "W1": (F,L,Q), "W2": (F,L,Q)}
    candidate_scores: [{"W0": (F,L,Q), ...}, ...]  length = N

    对每个候选 Si:
      r_pre = Q(Si,W0) - Q(S0,W0)
      r_cur = Q(Si,W1) - Q(S0,W1)
      r_fut = Q(Si,W2) - Q(S0,W2)
      r_all = r_pre + r_cur + r_fut
    """
    q_s0 = {w: scores[2] for w, scores in s0_scores.items()}

    reward_list = []
    for i, cand_scores in enumerate(candidate_scores):
        q_si = {w: scores[2] for w, scores in cand_scores.items()}
        r_pre = q_si.get("W0", 0) - q_s0.get("W0", 0)
        r_cur = q_si.get("W1", 0) - q_s0.get("W1", 0)
        r_fut = q_si.get("W2", 0) - q_s0.get("W2", 0)
        r_all = r_pre + r_cur + r_fut
        reward_list.append({
            "index": i,
            "profile": candidates[i],
            "r_pre": r_pre,
            "r_cur": r_cur,
            "r_fut": r_fut,
            "r_all": r_all,
            "scores": {w: {"F": s[0], "L": s[1], "Q": s[2]} for w, s in cand_scores.items()},
        })

    positive = [r for r in reward_list if r["r_all"] > TAU_PLUS]
    negative = [r for r in reward_list if r["r_all"] < TAU_MINUS]

    n_cand = len(candidates)

    def _build_row(pos: Dict, neg: Dict, rule: str) -> Dict:
        return {
            "chosen": {
                "profile": pos["profile"],
                "profile_source": profile_candidate_source(pos["index"], n_cand),
                "r_all": pos["r_all"],
                "r_pre": pos["r_pre"],
                "r_cur": pos["r_cur"],
                "r_fut": pos["r_fut"],
                "scores": pos["scores"],
            },
            "rejected": {
                "profile": neg["profile"],
                "profile_source": profile_candidate_source(neg["index"], n_cand),
                "r_all": neg["r_all"],
                "r_pre": neg["r_pre"],
                "r_cur": neg["r_cur"],
                "r_fut": neg["r_fut"],
                "scores": neg["scores"],
            },
            "baseline_profile": s0_profile,
            "baseline_profile_source": baseline_profile_source,
            "baseline_scores": {w: {"F": s[0], "L": s[1], "Q": s[2]} for w, s in s0_scores.items()},
            "discrepancies": behavior_discrepancies,
            "margin": pos["r_all"] - neg["r_all"],
            "pair_rule": rule,
        }

    dpo_pairs = []
    seen_pair_indices = set()

    # 规则 A：原始阈值法（TAU+/TAU- + DELTA）
    for pos in positive:
        for neg in negative:
            if pos["r_all"] - neg["r_all"] > DELTA:
                key = (pos["index"], neg["index"])
                if key in seen_pair_indices:
                    continue
                seen_pair_indices.add(key)
                dpo_pairs.append(_build_row(pos, neg, "tau_delta"))

    # 规则 B：补充绝对值差法
    # 任意两候选中，若一正一负，且 |r_big - r_small| > ABS_DELTA，则也纳入
    n = len(reward_list)
    for i in range(n):
        for j in range(i + 1, n):
            a = reward_list[i]
            b = reward_list[j]
            hi, lo = (a, b) if a["r_all"] >= b["r_all"] else (b, a)
            if not (hi["r_all"] > 0 and lo["r_all"] < 0):
                continue
            abs_gap = abs(hi["r_all"] - lo["r_all"])
            if abs_gap <= ABS_DELTA:
                continue
            key = (hi["index"], lo["index"])
            if key in seen_pair_indices:
                continue
            seen_pair_indices.add(key)
            dpo_pairs.append(_build_row(hi, lo, "abs_delta"))

    return dpo_pairs


def _compute_candidate_rewards(
    s0_scores: Dict[str, Tuple[float, float, float]],
    candidate_scores: List[Dict[str, Tuple[float, float, float]]],
) -> List[float]:
    """计算每个候选相对 baseline 的 r_all。"""
    q_s0 = {w: scores[2] for w, scores in s0_scores.items()}
    vals = []
    for cand_scores in candidate_scores:
        q_si = {w: scores[2] for w, scores in cand_scores.items()}
        r_pre = q_si.get("W0", 0) - q_s0.get("W0", 0)
        r_cur = q_si.get("W1", 0) - q_s0.get("W1", 0)
        r_fut = q_si.get("W2", 0) - q_s0.get("W2", 0)
        vals.append(r_pre + r_cur + r_fut)
    return vals


# ============================================================================
# 单用户 DPO 流程
# ============================================================================

def process_single_user(
    user_data: Dict,
    profile_model,
    profile_tokenizer,
    action_model,
    action_tokenizer,
    semantic_scorer: SemanticScorer,
    workers: int = DPO_WORKERS,
    rounds: int = 2,
) -> Dict:
    """
    对单个用户执行完整 DPO 流程。
    user_data: {"user_id", "community_id", "windows": {"W0":[...], ...}}
    """
    uid = user_data["user_id"]
    cid = user_data["community_id"]
    windows = user_data["windows"]
    window_keys = sorted(
        [k for k in windows.keys() if k.startswith("W")],
        key=lambda x: int(x[1:]),
    )
    max_rounds_by_data = max(0, len(window_keys) - 2)
    effective_rounds = max(1, min(rounds, max_rounds_by_data)) if max_rounds_by_data > 0 else 0

    _dbg_print(f"\n{'='*60}")
    _dbg_print(f"[User {uid}] 社区={cid}")
    _dbg_print(
        f"[User {uid}] 计划轮次={rounds}，可用轮次={effective_rounds} "
        f"(windows={window_keys})",
    )
    if effective_rounds == 0:
        _dbg_print(f"[User {uid}] 可用窗口不足，跳过")
        return {
            "user_id": uid,
            "community_id": cid,
            "s0_profile": "",
            "s0_scores": {},
            "num_candidates": 0,
            "num_dpo_pairs": 0,
            "dpo_pairs": [],
            "rounds": [],
        }

    # === Step 1: 生成初始画像 S0 ===
    _dbg_print(f"[User {uid}] Step 1: 生成初始画像 S0 ...")
    w0 = windows[window_keys[0]]
    s0 = generate_initial_profile(profile_model, profile_tokenizer, w0)
    _dbg_print(f"[User {uid}] S0 长度: {len(s0)} chars")
    current_profile = s0
    all_round_dpo_pairs: List[Dict[str, Any]] = []
    round_summaries: List[Dict[str, Any]] = []
    s0_scores_for_return: Dict[str, Tuple[float, float, float]] = {}
    num_candidates_last_round = 0
    prev_best_idx: Optional[int] = None
    prev_n_candidates: Optional[int] = None

    for ridx in range(effective_rounds):
        w_pre = windows[window_keys[ridx]]
        w_cur = windows[window_keys[ridx + 1]]
        w_fut = windows[window_keys[ridx + 2]]

        _dbg_print(
            f"[User {uid}] Round {ridx + 1}/{effective_rounds} "
            f"(windows={window_keys[ridx]},{window_keys[ridx+1]},{window_keys[ridx+2]})",
        )

        # baseline scores for current round profile
        cf0, cl0, cq0 = evaluate_profile_on_window(
            current_profile, [], w_pre, action_model, action_tokenizer, semantic_scorer
        )
        cf1, cl1, cq1 = evaluate_profile_on_window(
            current_profile, w_pre, w_cur, action_model, action_tokenizer, semantic_scorer
        )
        cf2, cl2, cq2 = evaluate_profile_on_window(
            current_profile, w_cur, w_fut, action_model, action_tokenizer, semantic_scorer
        )
        base_scores = {"W0": (cf0, cl0, cq0), "W1": (cf1, cl1, cq1), "W2": (cf2, cl2, cq2)}
        if ridx == 0:
            s0_scores_for_return = base_scores
        _dbg_print(
            f"[User {uid}] Round {ridx + 1} baseline scores: "
            f"{window_keys[ridx]}(rel_W0)=Q{cq0:.4f}, "
            f"{window_keys[ridx+1]}(rel_W1)=Q{cq1:.4f}, "
            f"{window_keys[ridx+2]}(rel_W2)=Q{cq2:.4f}",
        )

        # discrepancies on current window
        preds_cur = predict_actions_for_window(
            action_model, action_tokenizer, current_profile, w_pre, w_cur,
            temperature=TEMPERATURE_ACTION,
        )
        discrepancies = build_behavior_discrepancies(preds_cur, w_cur, w_pre)

        if cfg.DEBUG_LLM:
            _dbg_print(
                "\n"
                + "=" * 72
                + f"\n[DEBUG][User {uid}] Round {ridx + 1} 行为偏差全文\n"
                + "-" * 72
                + "\n"
                + discrepancies
                + "\n"
                + "=" * 72
                + "\n",
            )

        # candidate generation
        candidates = generate_candidate_profiles(
            profile_model, profile_tokenizer,
            current_profile, discrepancies,
            n=NUM_CANDIDATE_PROFILES,
            workers=workers,
        )
        num_candidates_last_round = len(candidates)

        # candidate quick screening: 先只算 W1（当前窗口）
        quick_w1_scores: List[Optional[Tuple[float, float, float]]] = [None] * len(candidates)

        def _score_w1_one(i: int, cand: str) -> tuple[int, Tuple[float, float, float]]:
            f1_, l1_, q1_ = evaluate_profile_on_window(
                cand, w_pre, w_cur, action_model, action_tokenizer, semantic_scorer
            )
            return i, (f1_, l1_, q1_)

        eff_workers = max(1, min(int(workers), len(candidates)))
        if eff_workers == 1:
            for i, cand in enumerate(candidates):
                idx, w1_score = _score_w1_one(i, cand)
                quick_w1_scores[idx] = w1_score
        else:
            with ThreadPoolExecutor(max_workers=eff_workers, thread_name_prefix="candquick") as pool:
                futs = {pool.submit(_score_w1_one, i, cand): i for i, cand in enumerate(candidates)}
                for fut in as_completed(futs):
                    idx, w1_score = fut.result()
                    quick_w1_scores[idx] = w1_score

        improved_indices: List[int] = []
        for i, w1_score in enumerate(quick_w1_scores):
            if w1_score is None:
                continue
            q1 = w1_score[2]
            gain = q1 - cq1
            if gain > 0:
                improved_indices.append(i)
            _dbg_print(
                f"  Round{ridx+1} 候选 {i+1}: quick_Q_W1={q1:.4f} "
                f"(vs baseline {cq1:.4f}, gain={gain:+.4f})",
            )

        if not improved_indices:
            _dbg_print(
                f"[User {uid}] Round {ridx + 1} 所有候选在 {window_keys[ridx+1]}(rel_W1) 均未超过 baseline，"
                f"跳过本轮剩余窗口计算与DPO构造",
            )
            round_summaries.append(
                {
                    "round_idx": ridx + 1,
                    "window_triplet": [window_keys[ridx], window_keys[ridx + 1], window_keys[ridx + 2]],
                    "num_candidates": len(candidates),
                    "num_dpo_pairs": 0,
                    "best_candidate_idx": None,
                    "best_r_all": None,
                    "skip_reason": "no_candidate_improves_w1_over_baseline",
                }
            )
            if ridx == 0 and effective_rounds > 1:
                _dbg_print(
                    f"[User {uid}] 首轮 DPO=0（无 W1 改进），跳过后续 {effective_rounds - 1} 轮",
                )
            if ridx == 0:
                break
            continue

        # candidate full scoring (parallel)
        candidate_scores_list: List[Optional[Dict[str, Tuple[float, float, float]]]] = [None] * len(candidates)

        def _score_one(i: int, cand: str) -> tuple[int, Dict[str, Tuple[float, float, float]], float]:
            # 复用 quick 阶段已计算的 W1，避免重复调用
            w1_cached = quick_w1_scores[i]
            if w1_cached is None:
                raise RuntimeError(f"Round{ridx+1} 候选 {i+1} 缺少 quick_W1 缓存结果")
            s0_, s1_, s2_ = evaluate_profile_on_window(cand, [], w_pre, action_model, action_tokenizer, semantic_scorer)
            s3_, s4_, s5_ = w1_cached
            s6_, s7_, s8_ = evaluate_profile_on_window(cand, w_cur, w_fut, action_model, action_tokenizer, semantic_scorer)
            cand_scores = {
                "W0": (s0_, s1_, s2_),
                "W1": (s3_, s4_, s5_),
                "W2": (s6_, s7_, s8_),
            }
            r_all = (s2_ - cq0) + (s5_ - cq1) + (s8_ - cq2)
            return i, cand_scores, r_all

        eff_workers = max(1, min(int(workers), len(candidates)))
        if eff_workers == 1:
            for i, cand in enumerate(candidates):
                idx, cand_scores, r_all = _score_one(i, cand)
                candidate_scores_list[idx] = cand_scores
                c0, c1, c2 = cand_scores["W0"][2], cand_scores["W1"][2], cand_scores["W2"][2]
                _dbg_print(
                    f"  Round{ridx+1} 候选 {idx+1}: Q_W0={c0:.4f} Q_W1={c1:.4f} Q_W2={c2:.4f} r_all={r_all:+.4f}"
                )
        else:
            with ThreadPoolExecutor(max_workers=eff_workers, thread_name_prefix="candscore") as pool:
                futs = {pool.submit(_score_one, i, cand): i for i, cand in enumerate(candidates)}
                for fut in as_completed(futs):
                    idx, cand_scores, r_all = fut.result()
                    candidate_scores_list[idx] = cand_scores
                    c0, c1, c2 = cand_scores["W0"][2], cand_scores["W1"][2], cand_scores["W2"][2]
                    _dbg_print(
                        f"  Round{ridx+1} 候选 {idx+1}: Q_W0={c0:.4f} Q_W1={c1:.4f} Q_W2={c2:.4f} r_all={r_all:+.4f}"
                    )
        candidate_scores_list = [s for s in candidate_scores_list if s is not None]

        # DPO pairs for this round
        if ridx == 0:
            baseline_src = "base"
        else:
            if prev_best_idx is None or prev_n_candidates is None:
                baseline_src = "base"
            else:
                baseline_src = profile_candidate_source(prev_best_idx, prev_n_candidates)
        round_pairs = construct_dpo_pairs(
            current_profile,
            candidates,
            base_scores,
            candidate_scores_list,
            baseline_profile_source=baseline_src,
            behavior_discrepancies=discrepancies,
        )
        for p in round_pairs:
            p["round_idx"] = ridx + 1
            p["window_triplet"] = [window_keys[ridx], window_keys[ridx + 1], window_keys[ridx + 2]]
        all_round_dpo_pairs.extend(round_pairs)
        _dbg_print(
            f"[User {uid}] Round {ridx + 1} 生成 DPO 对数量: {len(round_pairs)}",
        )

        # choose best candidate as next-round profile
        rewards = _compute_candidate_rewards(base_scores, candidate_scores_list)
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i]) if rewards else 0
        best_r_all = rewards[best_idx] if rewards else 0.0

        round_summaries.append(
            {
                "round_idx": ridx + 1,
                "window_triplet": [window_keys[ridx], window_keys[ridx + 1], window_keys[ridx + 2]],
                "num_candidates": len(candidates),
                "num_dpo_pairs": len(round_pairs),
                "best_candidate_idx": best_idx + 1 if rewards else None,
                "best_r_all": best_r_all,
            }
        )

        if ridx == 0 and len(round_pairs) == 0 and effective_rounds > 1:
            _dbg_print(
                f"[User {uid}] 首轮 DPO=0（阈值下未构出对），跳过后续 {effective_rounds - 1} 轮",
            )
            break

        if ridx < effective_rounds - 1:
            current_profile = candidates[best_idx]
            _dbg_print(
                f"[User {uid}] Round {ridx + 1} 选择最佳候选 idx={best_idx + 1} "
                f"作为 S{ridx + 1} (r_all={best_r_all:+.4f})",
            )

        prev_best_idx = best_idx
        prev_n_candidates = len(candidates)

    round_pair_dist = {f"round_{r['round_idx']}": r["num_dpo_pairs"] for r in round_summaries}
    _dbg_print(f"[User {uid}] 轮次DPO分布: {round_pair_dist}")
    _dbg_print(
        f"[User {uid}] 总计生成 {len(all_round_dpo_pairs)} 个 DPO 对 "
        f"（执行 {len(round_summaries)}/{effective_rounds} 轮）",
    )
    return {
        "user_id": uid,
        "community_id": cid,
        "s0_profile": s0,
        "s0_scores": {w: {"F": s[0], "L": s[1], "Q": s[2]} for w, s in s0_scores_for_return.items()},
        "num_candidates": num_candidates_last_round,
        "num_dpo_pairs": len(all_round_dpo_pairs),
        "dpo_pairs": all_round_dpo_pairs,
        "rounds": round_summaries,
        "round_pair_distribution": round_pair_dist,
    }


# ============================================================================
# 多进程子进程入口（需模块顶层，便于 ProcessPool 可 pickle；子进程内仍用 DPO_WORKERS 线程评候选）
# ============================================================================


def _dpo_user_worker(
    job: Tuple[int, Dict[str, Any], int, int, Optional[str], float],
) -> Tuple[int, Dict[str, Any], float]:
    """
    在独立进程中为单个用户跑完整 DPO 流程；进程内再按候选数用 ThreadPool（见 process_single_user）。

    job: (下标, user_data, workers, rounds, semantic_scorer_device, stagger_sec)
    stagger_sec: 第 i 个用户先 sleep i*stagger_sec 再加载模型/调 API，减轻多进程同时洪峰；0 表示不等待。
    返回: (下标, result_dict, 处理耗时秒，不含错开等待)
    """
    idx, user_data, workers, rounds, sem_device, stagger_sec = job
    if stagger_sec > 0.0 and idx > 0:
        time.sleep(float(idx) * float(stagger_sec))
    t0 = time.time()
    semantic_scorer = SemanticScorer(device=sem_device)
    profile_model, profile_tokenizer = None, None
    action_model, action_tokenizer = None, None
    result = process_single_user(
        user_data,
        profile_model,
        profile_tokenizer,
        action_model,
        action_tokenizer,
        semantic_scorer,
        workers=workers,
        rounds=rounds,
    )
    return idx, result, time.time() - t0


def _scorer_device_for_parallel_runs(
    override: Optional[str] = None,
) -> Optional[str]:
    """
    多用户多进程时每个子进程会各自加载一份 Sentence-Transformer，默认用 CPU 避免多份同驻 GPU 导致 OOM。
    可传 override 或 config.DPO_SCORER_DEVICE 覆盖。
    """
    if override is not None and str(override).strip() != "":
        return override
    cfg_dev = getattr(cfg, "DPO_SCORER_DEVICE", None)
    if cfg_dev:
        return cfg_dev
    return "cpu"


# ============================================================================
# 主入口
# ============================================================================

def run_dpo_pipeline(
    input_file: str,
    output_dir: str,
    max_users: int = None,
    random_seed: Optional[int] = None,
    workers: int = DPO_WORKERS,
    user_processes: int = DPO_USER_PROCESSES,
    user_process_stagger_sec: Optional[float] = None,
    rounds: int = DPO_ROUNDS,
    do_preflight: bool = True,
    resume: bool = False,
    scorer_device: Optional[str] = None,
) -> None:
    """
    DPO Pipeline 主入口。
    input_file: 窗口化后的 jsonl 文件（来自 window_splitter）
    output_dir: DPO 对输出目录
    """
    if do_preflight and not preflight_check():
        print("[Pipeline] 预检失败，退出", flush=True)
        sys.exit(1)

    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载数据
    users = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                users.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    n_loaded = len(users)
    if random_seed is not None:
        rng = random.Random(random_seed)
        rng.shuffle(users)
        print(
            f"[Pipeline] 已用 --seed {random_seed} 将 {n_loaded} 个用户打乱顺序",
            flush=True,
        )
    if max_users:
        users = users[:max_users]
    n_up = int(max(1, user_processes or 1))
    print(
        f"[Pipeline] 处理 {len(users)} 个用户（本 run），候选线程 workers={workers}，"
        f"用户进程数 user_processes={n_up}，rounds={rounds}",
        flush=True,
    )

    # LLM 已通过 vLLM 服务事先启动，无需本地加载
    profile_model, profile_tokenizer = None, None
    action_model, action_tokenizer = None, None

    # 结果文件路径
    stem = input_path.stem
    pairs_file = output_path / f"dpo_pairs_{stem}.jsonl"
    detail_progress_file = output_path / f"dpo_detail_{stem}.jsonl"

    # 处理 resume：从 detail 进度文件恢复已完成用户，再重建 pairs 文件，避免重复与脏数据
    processed_user_ids = set()
    all_results = []
    total_pairs_written = 0
    if resume and detail_progress_file.exists():
        with detail_progress_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                all_results.append(r)
                processed_user_ids.add(str(r.get("user_id")))
                for p in r.get("dpo_pairs", []):
                    p.setdefault("round_idx", 1)
                    p.setdefault("round_tag", f"round_{p['round_idx']}")
                    p.setdefault("user_id", r.get("user_id"))
                    p.setdefault("community_id", r.get("community_id"))
                    total_pairs_written += 1

        # 按恢复结果重建 pairs 文件
        with pairs_file.open("w", encoding="utf-8") as f:
            for r in all_results:
                for p in r.get("dpo_pairs", []):
                    p.setdefault("round_idx", 1)
                    p.setdefault("round_tag", f"round_{p['round_idx']}")
                    p.setdefault("user_id", r.get("user_id"))
                    p.setdefault("community_id", r.get("community_id"))
                    f.write(json.dumps(p, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

        print(
            f"[Pipeline] resume 模式：已恢复 {len(all_results)} 个用户，"
            f"重建 DPO 对 {total_pairs_written} 条",
            flush=True,
        )
    else:
        # 非 resume，或无可恢复进度：清空输出文件，开始新任务
        pairs_file.open("w", encoding="utf-8").close()
        detail_progress_file.open("w", encoding="utf-8").close()
        print(f"[Pipeline] 增量写入 DPO 对文件: {pairs_file}", flush=True)
        print(f"[Pipeline] 增量写入进度文件: {detail_progress_file}", flush=True)

    # 处理每个用户（resume 时跳过已完成用户）
    pending_users = [
        u for u in users if str(u.get("user_id")) not in processed_user_ids
    ]
    if processed_user_ids:
        print(
            f"[Pipeline] 已跳过 {len(users) - len(pending_users)} 个已完成用户，"
            f"本次待处理 {len(pending_users)} 个",
            flush=True,
        )

    effective_procs = min(n_up, len(pending_users)) if len(pending_users) else 0
    use_parallel_users = effective_procs > 1
    semantic_scorer: Optional[SemanticScorer] = None
    sem_parallel: Optional[str] = None
    if len(pending_users) > 0:
        if use_parallel_users:
            sem_parallel = _scorer_device_for_parallel_runs(override=scorer_device)
            print(
                f"[Pipeline] 多进程: {effective_procs} 个进程并行不同用户；"
                f"每进程内候选评估仍用 ThreadPool（最多 {workers} 线程）。"
                f"Sentence-Transformer: {sem_parallel}",
                flush=True,
            )
        else:
            print("[Pipeline] 加载 Semantic Scorer（主进程、单路用户）...", flush=True)
            d_single = scorer_device
            if d_single is None and getattr(cfg, "DPO_SCORER_DEVICE", None):
                d_single = str(cfg.DPO_SCORER_DEVICE)
            if d_single is not None and str(d_single).strip() == "":
                d_single = None
            semantic_scorer = SemanticScorer(device=d_single)

    stagger_resolved = float(
        DPO_USER_PROCESS_STAGGER_SEC
        if user_process_stagger_sec is None
        else (user_process_stagger_sec or 0.0)
    )
    if use_parallel_users and len(pending_users) > 0 and stagger_resolved > 0:
        print(
            f"[Pipeline] 多用户错开启动: 第 i 个用户任务在子进程内先等待 i×{stagger_resolved:.2f}s，"
            f"再加载模型/请求（减轻 API 同时洪峰）",
            flush=True,
        )

    def _flush_one_user(
        result: Dict[str, Any],
        fp_pairs,
        fp_detail,
    ) -> int:
        """
        将单用户结果落盘（detail 整包一行 + 每条 DPO 对），并 flush+fsync。
        串行/多进程路径均在「每个用户处理完毕」时调用，故 --resume 可按用户粒度恢复未完成的 user。
        """
        nonlocal total_pairs_written
        fp_detail.write(json.dumps(result, ensure_ascii=False) + "\n")
        fp_detail.flush()
        os.fsync(fp_detail.fileno())
        user_pairs_written = 0
        for p in result["dpo_pairs"]:
            p.setdefault("round_idx", 1)
            p.setdefault("round_tag", f"round_{p['round_idx']}")
            p.setdefault("user_id", result.get("user_id"))
            p.setdefault("community_id", result.get("community_id"))
            fp_pairs.write(json.dumps(p, ensure_ascii=False) + "\n")
            user_pairs_written += 1
        fp_pairs.flush()
        os.fsync(fp_pairs.fileno())
        total_pairs_written += user_pairs_written
        return user_pairs_written

    run_start = time.time()
    with pairs_file.open("a", encoding="utf-8") as fp_pairs, detail_progress_file.open("a", encoding="utf-8") as fp_detail:
        if not pending_users:
            pass
        elif use_parallel_users:
            jobs = [
                (i, u, workers, rounds, sem_parallel, stagger_resolved)
                for i, u in enumerate(pending_users)
            ]
            done_ct = 0
            n_pending = len(pending_users)
            results_by_idx: Dict[int, Dict[str, Any]] = {}
            pool: Optional[ProcessPoolExecutor] = None
            futs_dict: Dict[Any, int] = {}
            try:
                pool = ProcessPoolExecutor(max_workers=effective_procs)
                futs_dict = {pool.submit(_dpo_user_worker, job): job[0] for job in jobs}
                for fut in as_completed(futs_dict):
                    idx, result, user_wall_sec = fut.result()
                    done_ct += 1
                    results_by_idx[idx] = result
                    pws = _flush_one_user(result, fp_pairs, fp_detail)
                    batch_elapsed = time.time() - run_start
                    if cfg.DEBUG_LLM:
                        print(
                            f"\n[Pipeline] 多进程完成 {done_ct}/{n_pending} "
                            f"(user_id={result.get('user_id')})",
                            flush=True,
                        )
                        print(
                            f"[Pipeline] 用户 {result['user_id']} DPO轮次分布: "
                            f"{result.get('round_pair_distribution', {})}",
                            flush=True,
                        )
                        print(
                            f"[Pipeline] 用户 {result['user_id']} DPO 对已增量写入 {pws} 条 "
                            f"(累计 {total_pairs_written} 条)",
                            flush=True,
                        )
                    else:
                        _print_user_finish_report(
                            result,
                            done_ct,
                            n_pending,
                            user_wall_sec,
                            batch_elapsed,
                        )
            except KeyboardInterrupt:
                print(
                    "\n[Pipeline] 收到中断 (Ctrl+C)，正在结束子进程并取消未完成任务…",
                    flush=True,
                )
                _dpo_shutdown_process_pool(pool, futs_dict)
                raise
            except Exception:
                _dpo_shutdown_process_pool(pool, futs_dict)
                raise
            else:
                if pool is not None:
                    try:
                        try:
                            pool.shutdown(wait=True, cancel_futures=False)
                        except TypeError:
                            pool.shutdown(wait=True)
                    except Exception:
                        pass
            finally:
                _dpo_terminate_orphan_processes()
            for k in range(n_pending):
                if k not in results_by_idx:
                    raise RuntimeError(f"缺失待处理下标 {k} 的结果")
            for k in range(n_pending):
                all_results.append(results_by_idx[k])
                processed_user_ids.add(str(results_by_idx[k].get("user_id")))
        else:
            for i, user_data in enumerate(pending_users):
                if cfg.DEBUG_LLM:
                    print(f"\n[Pipeline] 处理用户 {i+1}/{len(pending_users)}", flush=True)
                t0 = time.time()
                if semantic_scorer is None:
                    raise RuntimeError("Internal: semantic_scorer 未初始化")
                result = process_single_user(
                    user_data,
                    profile_model, profile_tokenizer,
                    action_model, action_tokenizer,
                    semantic_scorer,
                    workers=workers,
                    rounds=rounds,
                )
                elapsed = time.time() - t0
                all_results.append(result)
                processed_user_ids.add(str(result.get("user_id")))
                pws = _flush_one_user(result, fp_pairs, fp_detail)
                batch_elapsed = time.time() - run_start
                if cfg.DEBUG_LLM:
                    print(f"[Pipeline] 用户 {result['user_id']} 完成，耗时 {elapsed:.1f}s", flush=True)
                    print(
                        f"[Pipeline] 用户 {result['user_id']} DPO轮次分布: "
                        f"{result.get('round_pair_distribution', {})}",
                        flush=True,
                    )
                    print(
                        f"[Pipeline] 用户 {result['user_id']} DPO 对已增量写入 {pws} 条 "
                        f"(累计 {total_pairs_written} 条)",
                        flush=True,
                    )
                else:
                    _print_user_finish_report(
                        result,
                        i + 1,
                        len(pending_users),
                        elapsed,
                        batch_elapsed,
                    )

    total_elapsed = time.time() - run_start
    avg_user_elapsed = (total_elapsed / len(pending_users)) if pending_users else 0.0
    print(
        f"\n[Pipeline] 总耗时: {total_elapsed:.1f}s | 平均每用户耗时: {avg_user_elapsed:.1f}s",
        flush=True,
    )

    # 保存结果
    print(f"\n[Pipeline] DPO 对写入完成: {pairs_file} ({total_pairs_written} 对)", flush=True)

    detail_file = output_path / f"dpo_detail_{stem}.json"
    with detail_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[Pipeline] 详细结果已保存: {detail_file}", flush=True)

    summary = {
        "input_file": str(input_path),
        "total_users": len(all_results),
        "total_dpo_pairs": total_pairs_written,
        "users_with_pairs": sum(1 for r in all_results if r["num_dpo_pairs"] > 0),
        "total_elapsed_seconds": round(total_elapsed, 3),
        "avg_user_elapsed_seconds": round(avg_user_elapsed, 3),
        "config": {
            "tau_plus": TAU_PLUS,
            "tau_minus": TAU_MINUS,
            "delta": DELTA,
            "alpha": ALPHA,
            "num_candidates": NUM_CANDIDATE_PROFILES,
            "debug_llm": cfg.DEBUG_LLM,
            "debug_llm_include_actions": getattr(cfg, "DEBUG_LLM_INCLUDE_ACTIONS", False),
            "random_seed": random_seed,
            "workers": workers,
            "user_processes": n_up,
            "user_parallel": use_parallel_users,
            "effective_user_processes": effective_procs,
            "scorer_device": (
                sem_parallel
                if use_parallel_users
                else (
                    scorer_device
                    or getattr(cfg, "DPO_SCORER_DEVICE", None)
                    or "auto"
                )
            ),
            "rounds": rounds,
            "user_process_stagger_sec": stagger_resolved,
        },
    }
    summary_file = output_path / f"dpo_summary_{stem}.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Pipeline] 统计摘要已保存: {summary_file}", flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO 对构造 Pipeline")
    parser.add_argument("--input", default=None, help="窗口化后的 jsonl 文件")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="窗口化 jsonl 所在目录（与 --input 二选一或同时使用）",
    )
    parser.add_argument(
        "--input-glob",
        default="community_*.jsonl",
        help="批量模式下用于匹配文件的 glob（默认: community_*.jsonl）",
    )
    parser.add_argument("--output-dir", default="output/dpo", help="DPO 对输出目录")
    parser.add_argument(
        "--max-users",
        "--max_user",
        dest="max_users",
        type=int,
        default=None,
        help="最多处理用户数（调试用）",
    )
    parser.add_argument("--test", action="store_true", help="测试模式：用远程 API 单模型跑通 pipeline")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试：每次 LLM 调用在终端打印步骤、模型类型(model_role)、模型名、请求与完整输出",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子：加载全部用户后先按此种子打乱顺序，再取前 --max-users 条，便于换用户做重复实验",
    )
    parser.add_argument(
        "--debug-actions",
        action="store_true",
        help="与 --debug 联用：额外打印每次「动作预测」LLM 的完整调试块（含 scenario 里 reply 原文；输出量很大）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DPO_WORKERS,
        help="单用户内、候选级并行：线程数（画像与评分，默认 config.DPO_WORKERS）",
    )
    parser.add_argument(
        "--user-processes",
        type=int,
        default=DPO_USER_PROCESSES,
        dest="user_processes",
        help="多用户时并行进程数；每进程内仍用 --workers 做候选级线程。1=按用户串行。",
    )
    parser.add_argument(
        "--scorer-device",
        default=None,
        help="Sentence-Transformer 设备：cpu / cuda 等。多进程默认各子进程 cpu；单进程未指定则自动。",
    )
    parser.add_argument(
        "--user-process-stagger",
        type=float,
        default=None,
        dest="user_process_stagger_sec",
        help="多用户多进程时第 i 个任务先等待 i×秒 再开始（默认同 config；0=关闭错开）",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DPO_ROUNDS,
        help="DPO 滚动轮次（默认 2）：每轮选择 r_all 最大候选作为下一轮基线画像",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="断点续跑：从 dpo_detail_<stem>.jsonl 恢复已完成用户并继续处理未完成用户",
    )
    args = parser.parse_args()

    if args.test:
        cfg.TEST_MODE = True
    if args.debug:
        cfg.DEBUG_LLM = True
    if args.debug_actions:
        cfg.DEBUG_LLM_INCLUDE_ACTIONS = True

    input_files: List[str] = []
    if args.input:
        input_files.append(args.input)
    if args.input_dir:
        dir_path = Path(args.input_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            print(f"[Pipeline] 输入目录不存在或不是目录: {dir_path}", flush=True)
            sys.exit(1)
        matched = sorted(dir_path.glob(args.input_glob))
        input_files.extend([str(p) for p in matched if p.is_file()])

    # 去重并保持顺序
    input_files = list(dict.fromkeys(input_files))
    if not input_files:
        print("[Pipeline] 请至少提供 --input 或 --input-dir", flush=True)
        sys.exit(1)

    if len(input_files) == 1:
        run_dpo_pipeline(
            input_files[0],
            args.output_dir,
            max_users=args.max_users,
            random_seed=args.seed,
            workers=args.workers,
            user_processes=args.user_processes,
            rounds=args.rounds,
            do_preflight=True,
            resume=args.resume,
            scorer_device=args.scorer_device,
            user_process_stagger_sec=args.user_process_stagger_sec,
        )
    else:
        print(f"[Pipeline] 批量模式: 共 {len(input_files)} 个输入文件", flush=True)
        if not preflight_check():
            print("[Pipeline] 预检失败，退出", flush=True)
            sys.exit(1)
        for i, fp in enumerate(input_files, 1):
            print(f"\n[Batch] ({i}/{len(input_files)}) 开始处理: {fp}", flush=True)
            run_dpo_pipeline(
                fp,
                args.output_dir,
                max_users=args.max_users,
                random_seed=args.seed,
                workers=args.workers,
                user_processes=args.user_processes,
                rounds=args.rounds,
                do_preflight=False,
                resume=args.resume,
                scorer_device=args.scorer_device,
                user_process_stagger_sec=args.user_process_stagger_sec,
            )
