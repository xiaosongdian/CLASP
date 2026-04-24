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
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
)
from src.scorer import SemanticScorer, evaluate_predictions


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
) -> Tuple[float, float, float]:
    """
    在单个窗口上评估画像，返回 (F, L, Q)。
    history: 前序窗口动作（作为上下文）
    targets: 目标窗口动作（待预测）
    """
    predictions = predict_actions_for_window(
        action_model, action_tokenizer,
        profile, history, targets,
        temperature=TEMPERATURE_ACTION,
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
) -> List[Dict]:
    """
    构造 DPO 对。

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

    def _build_row(pos: Dict, neg: Dict, rule: str) -> Dict:
        return {
            "chosen": {
                "profile": pos["profile"],
                "r_all": pos["r_all"],
                "r_pre": pos["r_pre"],
                "r_cur": pos["r_cur"],
                "r_fut": pos["r_fut"],
                "scores": pos["scores"],
            },
            "rejected": {
                "profile": neg["profile"],
                "r_all": neg["r_all"],
                "r_pre": neg["r_pre"],
                "r_cur": neg["r_cur"],
                "r_fut": neg["r_fut"],
                "scores": neg["scores"],
            },
            "baseline_profile": s0_profile,
            "baseline_scores": {w: {"F": s[0], "L": s[1], "Q": s[2]} for w, s in s0_scores.items()},
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

    print(f"\n{'='*60}", flush=True)
    print(f"[User {uid}] 社区={cid}", flush=True)
    print(
        f"[User {uid}] 计划轮次={rounds}，可用轮次={effective_rounds} "
        f"(windows={window_keys})",
        flush=True,
    )
    if effective_rounds == 0:
        print(f"[User {uid}] 可用窗口不足，跳过", flush=True)
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
    print(f"[User {uid}] Step 1: 生成初始画像 S0 ...", flush=True)
    w0 = windows[window_keys[0]]
    s0 = generate_initial_profile(profile_model, profile_tokenizer, w0)
    print(f"[User {uid}] S0 长度: {len(s0)} chars", flush=True)
    current_profile = s0
    all_round_dpo_pairs: List[Dict[str, Any]] = []
    round_summaries: List[Dict[str, Any]] = []
    s0_scores_for_return: Dict[str, Tuple[float, float, float]] = {}
    num_candidates_last_round = 0

    for ridx in range(effective_rounds):
        w_pre = windows[window_keys[ridx]]
        w_cur = windows[window_keys[ridx + 1]]
        w_fut = windows[window_keys[ridx + 2]]

        print(
            f"[User {uid}] Round {ridx + 1}/{effective_rounds} "
            f"(windows={window_keys[ridx]},{window_keys[ridx+1]},{window_keys[ridx+2]})",
            flush=True,
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
        print(
            f"[User {uid}] Round {ridx + 1} baseline scores: "
            f"{window_keys[ridx]}(rel_W0)=Q{cq0:.4f}, "
            f"{window_keys[ridx+1]}(rel_W1)=Q{cq1:.4f}, "
            f"{window_keys[ridx+2]}(rel_W2)=Q{cq2:.4f}",
            flush=True,
        )

        # discrepancies on current window
        preds_cur = predict_actions_for_window(
            action_model, action_tokenizer, current_profile, w_pre, w_cur,
            temperature=TEMPERATURE_ACTION,
        )
        discrepancies = build_behavior_discrepancies(preds_cur, w_cur, w_pre)

        if cfg.DEBUG_LLM:
            print(
                "\n"
                + "=" * 72
                + f"\n[DEBUG][User {uid}] Round {ridx + 1} 行为偏差全文\n"
                + "-" * 72
                + "\n"
                + discrepancies
                + "\n"
                + "=" * 72
                + "\n",
                flush=True,
            )

        # candidate generation
        candidates = generate_candidate_profiles(
            profile_model, profile_tokenizer,
            current_profile, discrepancies,
            n=NUM_CANDIDATE_PROFILES,
            workers=workers,
        )
        num_candidates_last_round = len(candidates)

        # candidate scoring (parallel)
        candidate_scores_list: List[Optional[Dict[str, Tuple[float, float, float]]]] = [None] * len(candidates)

        def _score_one(i: int, cand: str) -> tuple[int, Dict[str, Tuple[float, float, float]], float]:
            s0_, s1_, s2_ = evaluate_profile_on_window(cand, [], w_pre, action_model, action_tokenizer, semantic_scorer)
            s3_, s4_, s5_ = evaluate_profile_on_window(cand, w_pre, w_cur, action_model, action_tokenizer, semantic_scorer)
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
                print(f"  Round{ridx+1} 候选 {idx+1}: Q_W0={c0:.4f} Q_W1={c1:.4f} Q_W2={c2:.4f} r_all={r_all:+.4f}", flush=True)
        else:
            with ThreadPoolExecutor(max_workers=eff_workers, thread_name_prefix="candscore") as pool:
                futs = {pool.submit(_score_one, i, cand): i for i, cand in enumerate(candidates)}
                for fut in as_completed(futs):
                    idx, cand_scores, r_all = fut.result()
                    candidate_scores_list[idx] = cand_scores
                    c0, c1, c2 = cand_scores["W0"][2], cand_scores["W1"][2], cand_scores["W2"][2]
                    print(f"  Round{ridx+1} 候选 {idx+1}: Q_W0={c0:.4f} Q_W1={c1:.4f} Q_W2={c2:.4f} r_all={r_all:+.4f}", flush=True)
        candidate_scores_list = [s for s in candidate_scores_list if s is not None]

        # DPO pairs for this round
        round_pairs = construct_dpo_pairs(current_profile, candidates, base_scores, candidate_scores_list)
        for p in round_pairs:
            p["round_idx"] = ridx + 1
            p["window_triplet"] = [window_keys[ridx], window_keys[ridx + 1], window_keys[ridx + 2]]
        all_round_dpo_pairs.extend(round_pairs)
        print(
            f"[User {uid}] Round {ridx + 1} 生成 DPO 对数量: {len(round_pairs)}",
            flush=True,
        )

        # choose best candidate as next-round profile
        rewards = _compute_candidate_rewards(base_scores, candidate_scores_list)
        best_idx = max(range(len(rewards)), key=lambda i: rewards[i]) if rewards else 0
        best_r_all = rewards[best_idx] if rewards else 0.0
        if ridx < effective_rounds - 1:
            current_profile = candidates[best_idx]
            print(
                f"[User {uid}] Round {ridx + 1} 选择最佳候选 idx={best_idx + 1} "
                f"作为 S{ridx + 1} (r_all={best_r_all:+.4f})",
                flush=True,
            )

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

    round_pair_dist = {f"round_{r['round_idx']}": r["num_dpo_pairs"] for r in round_summaries}
    print(
        f"[User {uid}] 轮次DPO分布: {round_pair_dist}",
        flush=True,
    )
    print(f"[User {uid}] 总计生成 {len(all_round_dpo_pairs)} 个 DPO 对（{effective_rounds} 轮）", flush=True)
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
# 主入口
# ============================================================================

def run_dpo_pipeline(
    input_file: str,
    output_dir: str,
    max_users: int = None,
    random_seed: Optional[int] = None,
    workers: int = DPO_WORKERS,
    rounds: int = DPO_ROUNDS,
    do_preflight: bool = True,
    resume: bool = False,
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
    print(
        f"[Pipeline] 处理 {len(users)} 个用户（本 run），候选并发 workers={workers}，rounds={rounds}",
        flush=True,
    )

    # 加载 semantic scorer（唯一需要本地加载的模型）
    print("[Pipeline] 加载 Semantic Scorer ...", flush=True)
    semantic_scorer = SemanticScorer()

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

    run_start = time.time()
    with pairs_file.open("a", encoding="utf-8") as fp_pairs, detail_progress_file.open("a", encoding="utf-8") as fp_detail:
        for i, user_data in enumerate(pending_users):
            print(f"\n[Pipeline] 处理用户 {i+1}/{len(pending_users)}", flush=True)
            t0 = time.time()
            result = process_single_user(
                user_data,
                profile_model, profile_tokenizer,
                action_model, action_tokenizer,
                semantic_scorer,
                workers=workers,
                rounds=rounds,
            )
            elapsed = time.time() - t0
            print(f"[Pipeline] 用户 {result['user_id']} 完成，耗时 {elapsed:.1f}s", flush=True)
            print(
                f"[Pipeline] 用户 {result['user_id']} DPO轮次分布: "
                f"{result.get('round_pair_distribution', {})}",
                flush=True,
            )

            all_results.append(result)
            processed_user_ids.add(str(result.get("user_id")))

            # 先写入 detail 进度，标记该用户已完成（用于 resume 跳过）
            fp_detail.write(json.dumps(result, ensure_ascii=False) + "\n")
            fp_detail.flush()
            os.fsync(fp_detail.fileno())

            user_pairs_written = 0
            for p in result["dpo_pairs"]:
                # 兜底：确保每条 DPO 对都带轮次信息
                p.setdefault("round_idx", 1)
                p.setdefault("round_tag", f"round_{p['round_idx']}")
                p.setdefault("user_id", result.get("user_id"))
                p.setdefault("community_id", result.get("community_id"))
                fp_pairs.write(json.dumps(p, ensure_ascii=False) + "\n")
                user_pairs_written += 1

            # 每完成一个用户就落盘，尽量避免中途中断导致已生成 DPO 对丢失
            fp_pairs.flush()
            os.fsync(fp_pairs.fileno())
            total_pairs_written += user_pairs_written
            print(
                f"[Pipeline] 用户 {result['user_id']} DPO 对已增量写入 {user_pairs_written} 条 "
                f"(累计 {total_pairs_written} 条)",
                flush=True,
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
            "rounds": rounds,
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
        help="候选画像生成与评分的并发线程数（默认来自 config.DPO_WORKERS）",
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
            args.max_users,
            args.seed,
            args.workers,
            args.rounds,
            do_preflight=True,
            resume=args.resume,
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
                args.max_users,
                args.seed,
                args.workers,
                args.rounds,
                do_preflight=False,
                resume=args.resume,
            )
