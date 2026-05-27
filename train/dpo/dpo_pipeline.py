#!/usr/bin/env python3


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
    """Format duration in seconds to human-readable string; return "—" for invalid values."""
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
    In non-debug mode, print one line per completed user: progress, DPO pair count/candidates/per-round,
    user and batch elapsed time, and ETA.
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
        f"[Pipeline]  {done_in_batch}/{total_in_batch} ({pct:.1f}%) | "
        f"DPO={n_pairs} ={n_cand} {rdist} | "
        f"={_format_duration_s(user_elapsed_sec)} ={_format_duration_s(batch_elapsed_sec)} "
        f"={eta_str}",
        flush=True,
    )


def _dpo_terminate_orphan_processes() -> None:
    """Terminate orphan child processes from multiprocessing; handles Ctrl+C cleanup."""
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
    """Shutdown process pool gracefully; cancel pending futures and terminate orphan processes."""
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
# API and Path Validation
# ============================================================================

def _check_api(api_base: str, model_name: str, api_key: str = "not-needed", label: str = "") -> bool:
    """Test OpenAI-compatible API endpoint connectivity."""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key=api_key, timeout=15)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "Hello~ who are you?"}],
            max_tokens=5,
            temperature=0.01,
        )
        text = resp.choices[0].message.content.strip()
        print(f"  [✓] {label} （: {text[:50]}）", flush=True)
        return True
    except Exception as e:
        print(f"  [✗] {label} : {e}", flush=True)
        return False


def _check_local_path(path: str, label: str = "") -> bool:
    """Check if local file path exists."""
    p = Path(path)
    if p.exists():
        print(f"  [✓] {label} : {path}", flush=True)
        return True
    print(f"  [✗] {label} : {path}", flush=True)
    return False


def preflight_check(comparison_methods: Optional[List[str]] = None) -> bool:
    """
    Pre-startup validation:
    1. Check if two vLLM services (profile + action) are accessible
    2. Check if Sentence Transformer local path exists

    comparison_methods:
        If provided (e.g., from run_baseline_comparison's --methods), probe based on actual models used by window_chain:
        baselines needing profiles (static/prefix/incremental) use COMPARISON_BASELINE_VLLM_MODEL for profile+action probes;
        history_only only probes action endpoint with same model; clasp_online and clasp_online_ablate_* probe with their respective PROFILE/ACTION combinations.
        If not provided, use config's PROFILE_API_MODEL / ACTION_API_MODEL (main DPO pipeline, etc.).
    """
    print("\n[Preflight] Checking prerequisites...", flush=True)

    api_ok: List[bool] = []

    if comparison_methods is not None and len(comparison_methods) > 0:
        ms = set(comparison_methods)
        need_profile_baseline = bool(
            ms & {"static_s0", "prefix_refresh", "incremental_persona"}
        )
        need_history_only = bool(ms & {"history_only"})
        need_clasp = bool(
            ms
            & {
                "clasp_online",
                "clasp_online_no_hist",
                "clasp_online_ablate_action_base",
                "clasp_online_ablate_profile_base",
                "clasp_online_ablate_profile_stage1",
            }
        )
        need_ablate_action_base = bool(ms & {"clasp_online_ablate_action_base"})
        need_ablate_profile_base = bool(ms & {"clasp_online_ablate_profile_base"})
        need_ablate_profile_stage1 = bool(ms & {"clasp_online_ablate_profile_stage1"})
        need_clasp_full = bool(ms & {"clasp_online", "clasp_online_no_hist"})

        if need_profile_baseline or need_history_only:
            bp = str(cfg.COMPARISON_BASELINE_VLLM_MODEL)
            if need_profile_baseline:
                api_ok.append(
                    _check_api(
                        PROFILE_API_BASE,
                        bp,
                        label=f"Profile vLLM (static/prefix/incremental, model={bp})",
                    )
                )
            for ab in cfg.effective_action_api_bases():
                api_ok.append(
                    _check_api(
                        ab,
                        bp,
                        label=(
                            f"Action vLLM ({'/history_only' if need_history_only else ''}, model={bp}) @ {ab}"
                        ),
                    )
                )
        if need_clasp:
            cp = str(cfg.COMPARISON_CLASP_PROFILE_VLLM_MODEL)
            ca = str(cfg.COMPARISON_CLASP_ACTION_VLLM_MODEL)
            bp = str(cfg.COMPARISON_BASELINE_VLLM_MODEL)
            c1 = str(getattr(cfg, "COMPARISON_CLASP_PROFILE_STAGE1_VLLM_MODEL", ""))

            if need_clasp_full or need_ablate_action_base:
                api_ok.append(
                    _check_api(
                        PROFILE_API_BASE,
                        cp,
                        label=f"Profile vLLM (clasp_online / ablate_action_base, model={cp})",
                    )
                )
            if need_ablate_profile_base:
                api_ok.append(
                    _check_api(
                        PROFILE_API_BASE,
                        bp,
                        label=f"Profile vLLM (ablate_profile_base, model={bp})",
                    )
                )
            if need_ablate_profile_stage1 and c1:
                api_ok.append(
                    _check_api(
                        PROFILE_API_BASE,
                        c1,
                        label=f"Profile vLLM (ablate_profile_stage1, model={c1})",
                    )
                )

            if need_clasp_full or need_ablate_profile_base or need_ablate_profile_stage1:
                for ab in cfg.effective_action_api_bases():
                    api_ok.append(
                        _check_api(
                            ab,
                            ca,
                            label=f"Action vLLM (clasp SFT / ablate_profile_*, model={ca}) @ {ab}",
                        )
                    )
            if need_ablate_action_base:
                for ab in cfg.effective_action_api_bases():
                    api_ok.append(
                        _check_api(
                            ab,
                            bp,
                            label=f"Action vLLM (ablate_action_base, model={bp}) @ {ab}",
                        )
                    )

        if not need_profile_baseline and not need_history_only and not need_clasp:
            api_ok.append(
                _check_api(PROFILE_API_BASE, PROFILE_API_MODEL, label="")
            )
            for ab in cfg.effective_action_api_bases():
                api_ok.append(
                    _check_api(
                        ab,
                        ACTION_API_MODEL,
                        label=f" @ {ab}",
                    )
                )
    else:
        ok1 = _check_api(
            PROFILE_API_BASE, PROFILE_API_MODEL, label="Profile vLLM"
        )
        api_ok.append(ok1)
        for ab in cfg.effective_action_api_bases():
            api_ok.append(
                _check_api(
                    ab,
                    ACTION_API_MODEL,
                    label=f"Action vLLM @ {ab}",
                )
            )

    ok3 = _check_local_path(SENTENCE_TRANSFORMER_MODEL, "Sentence Transformer")
    all_ok = all(api_ok) and ok3

    if all_ok:
        print("[Preflight] ✓ All checks passed\n", flush=True)
    else:
        print("[Preflight] ✗ Some checks failed\n", flush=True)

    return all_ok


# ============================================================================
# Profile Evaluation on Window
# ============================================================================

def evaluate_profile_on_window(
    profile: str,
    history: List[Dict],
    targets: List[Dict],
    action_model,
    action_tokenizer,
    semantic_scorer: SemanticScorer,
    profile_suffix: Optional[str] = None,
    include_observed_history: Optional[bool] = None,
) -> Tuple[float, float, float]:
    """
    Evaluate profile on a single window, return (F, L, Q).
    history: prior window actions (as context)
    targets: target window actions (to predict)
    profile_suffix: see action_predictor.predict_actions_for_window
    include_observed_history: see predict_actions_for_window; None reads config.
    """
    predictions = predict_actions_for_window(
        action_model, action_tokenizer,
        profile, history, targets,
        temperature=TEMPERATURE_ACTION,
        profile_suffix=profile_suffix,
        include_observed_history=include_observed_history,
    )
    return evaluate_predictions(predictions, targets, semantic_scorer, ALPHA)


# ============================================================================
# DPO Pair Construction
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
    Construct DPO pairs.

    Each sample contains chosen/rejected profile_source (base | commercial),
    and baseline_profile_source (first round S0 is base; subsequent rounds use best candidate source from prior round).
    behavior_discrepancies: full text of prediction-actual behavior discrepancies for this round, written to each sample's discrepancies field,
    for later SFT / conditional DPO prompt construction (consistent with generate_candidate_profiles parameter).

    s0_scores: {"W0": (F,L,Q), "W1": (F,L,Q), "W2": (F,L,Q)}
    candidate_scores: [{"W0": (F,L,Q), ...}, ...]  length = N

    For each candidate Si:
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

    # Rule A: (TAU+/TAU- + DELTA)
    for pos in positive:
        for neg in negative:
            if pos["r_all"] - neg["r_all"] > DELTA:
                key = (pos["index"], neg["index"])
                if key in seen_pair_indices:
                    continue
                seen_pair_indices.add(key)
                dpo_pairs.append(_build_row(pos, neg, "tau_delta"))

    # Rule B: Candidates with opposite signs (one positive, one negative)
    # If one is positive and one is negative, and their absolute difference > ABS_DELTA,
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
    """Compute reward (r_all) for each candidate relative to baseline."""
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
# Single User DPO Processing
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
    Execute complete DPO pipeline for a single user.
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
    _dbg_print(f"[User {uid}] community_id={cid}")
    _dbg_print(
        f"[User {uid}] requested_rounds={rounds}, effective_rounds={effective_rounds} "
        f"(windows={window_keys})",
    )
    if effective_rounds == 0:
        _dbg_print(f"[User {uid}] Insufficient windows for DPO processing")
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

    # === Step 1: Generate Initial Profile S0 ===
    _dbg_print(f"[User {uid}] Step 1: Generating initial profile S0 ...")
    w0 = windows[window_keys[0]]
    s0 = generate_initial_profile(profile_model, profile_tokenizer, w0)
    _dbg_print(f"[User {uid}] S0 generated: {len(s0)} characters")
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

        # Evaluate baseline profile on current round windows
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
            f"{window_keys[ridx]}(W0)=Q{cq0:.4f}, "
            f"{window_keys[ridx+1]}(W1)=Q{cq1:.4f}, "
            f"{window_keys[ridx+2]}(W2)=Q{cq2:.4f}",
        )

        # Compute behavior discrepancies on current window
        preds_cur = predict_actions_for_window(
            action_model, action_tokenizer, current_profile, w_pre, w_cur,
            temperature=TEMPERATURE_ACTION,
        )
        discrepancies = build_behavior_discrepancies(preds_cur, w_cur, w_pre)

        if cfg.DEBUG_LLM:
            _dbg_print(
                "\n"
                + "=" * 72
                + f"\n[DEBUG][User {uid}] Round {ridx + 1} \n"
                + "-" * 72
                + "\n"
                + discrepancies
                + "\n"
                + "=" * 72
                + "\n",
            )

        # Generate candidate profiles
        candidates = generate_candidate_profiles(
            profile_model, profile_tokenizer,
            current_profile, discrepancies,
            n=NUM_CANDIDATE_PROFILES,
            workers=workers,
        )
        num_candidates_last_round = len(candidates)

        # Quick screening of candidates on W1 (filter by current window performance)
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
                f"  Round{ridx+1} Candidate {i+1}: quick_Q_W1={q1:.4f} "
                f"(vs baseline {cq1:.4f}, gain={gain:+.4f})",
            )

        if not improved_indices:
            _dbg_print(
                f"[User {uid}] Round {ridx + 1} No candidates improved {window_keys[ridx+1]}(W1) over baseline, "
                f"skipping DPO pair generation",
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
                    f"[User {uid}] Round 1 generated 0 DPO pairs (no W1 improvement), "
                    f"skipping remaining {effective_rounds - 1} rounds",
                )
            if ridx == 0:
                break
            continue

        # Full scoring of candidates (parallel)
        candidate_scores_list: List[Optional[Dict[str, Tuple[float, float, float]]]] = [None] * len(candidates)

        def _score_one(i: int, cand: str) -> tuple[int, Dict[str, Tuple[float, float, float]], float]:
            # Reuse quick W1 score, compute W0 and W2
            w1_cached = quick_w1_scores[i]
            if w1_cached is None:
                raise RuntimeError(f"Round{ridx+1} Candidate {i+1} missing quick_W1 score")
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
                    f"  Round{ridx+1} Candidate {idx+1}: Q_W0={c0:.4f} Q_W1={c1:.4f} Q_W2={c2:.4f} r_all={r_all:+.4f}"
                )
        else:
            with ThreadPoolExecutor(max_workers=eff_workers, thread_name_prefix="candscore") as pool:
                futs = {pool.submit(_score_one, i, cand): i for i, cand in enumerate(candidates)}
                for fut in as_completed(futs):
                    idx, cand_scores, r_all = fut.result()
                    candidate_scores_list[idx] = cand_scores
                    c0, c1, c2 = cand_scores["W0"][2], cand_scores["W1"][2], cand_scores["W2"][2]
                    _dbg_print(
                        f"  Round{ridx+1} Candidate {idx+1}: Q_W0={c0:.4f} Q_W1={c1:.4f} Q_W2={c2:.4f} r_all={r_all:+.4f}"
                    )
        candidate_scores_list = [s for s in candidate_scores_list if s is not None]

        # Construct DPO pairs for this round
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
            f"[User {uid}] Round {ridx + 1} generated DPO pairs: {len(round_pairs)}",
        )

        # Select best candidate as next-round profile
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
                f"[User {uid}] Round 1 generated 0 DPO pairs (no qualifying pairs), "
                f"skipping remaining {effective_rounds - 1} rounds",
            )
            break

        if ridx < effective_rounds - 1:
            current_profile = candidates[best_idx]
            _dbg_print(
                f"[User {uid}] Round {ridx + 1} selected candidate idx={best_idx + 1} "
                f"as S{ridx + 1} (r_all={best_r_all:+.4f})",
            )

        prev_best_idx = best_idx
        prev_n_candidates = len(candidates)

    round_pair_dist = {f"round_{r['round_idx']}": r["num_dpo_pairs"] for r in round_summaries}
    _dbg_print(f"[User {uid}] DPO summary: {round_pair_dist}")
    _dbg_print(
        f"[User {uid}] Completed {len(all_round_dpo_pairs)} total DPO pairs "
        f"(from {len(round_summaries)}/{effective_rounds} rounds)",
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
# Multiprocessing Worker (ProcessPool + ThreadPool for candidates; DPO_WORKERS controls thread count)
# ============================================================================


def _dpo_user_worker(
    job: Tuple[int, Dict[str, Any], int, int, Optional[str], float],
) -> Tuple[int, Dict[str, Any], float]:
    """
    Run complete DPO pipeline for a single user in an independent process; within process use ThreadPool for candidates (see process_single_user).

    job: (index, user_data, workers, rounds, semantic_scorer_device, stagger_sec)
    stagger_sec: user i sleeps i*stagger_sec before loading models/calling APIs, to reduce multiprocess API flood; 0 means no wait.
    Returns: (index, result_dict, processing time in seconds, excluding stagger wait)

    Note: underlying LLM/SDK exceptions (e.g., openai.APIConnectionError) when pickled back to main process often raise TypeError
    due to exception __reduce__/__init__ mismatch with library versions, masking real failures.
    Here we convert to RuntimeError to ensure main process receives readable error and maintains stable shutdown.
    """
    idx, user_data, workers, rounds, sem_device, stagger_sec = job
    uid = user_data.get("user_id", "?")
    cid = user_data.get("community_id", "?")
    ctx = f"[user_id={uid}, community_id={cid}]"
    if stagger_sec > 0.0 and idx > 0:
        time.sleep(float(idx) * float(stagger_sec))
    t0 = time.time()
    try:
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
    except Exception as e:
        raise RuntimeError(
            f"DPO processing failed for {ctx} ({type(e).__name__}): {e}",
        ) from None


def _scorer_device_for_parallel_runs(
    override: Optional[str] = None,
) -> Optional[str]:
    """
    When running multiple users in parallel processes, each subprocess loads its own Sentence-Transformer.
    Default to CPU to avoid multiple copies on GPU causing OOM.
    Can override with override parameter or config.DPO_SCORER_DEVICE.
    """
    if override is not None and str(override).strip() != "":
        return override
    cfg_dev = getattr(cfg, "DPO_SCORER_DEVICE", None)
    if cfg_dev:
        return cfg_dev
    return "cpu"


# ============================================================================
# Main DPO Pipeline
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
    Main DPO Pipeline entry point.
    input_file: windowed jsonl file (from window_splitter)
    output_dir: output directory for DPO pairs
    """
    if do_preflight and not preflight_check():
        print("[Pipeline] Preflight check failed, exiting", flush=True)
        sys.exit(1)

    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load users from input file
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
            f"[Pipeline] Loaded {n_loaded} users, shuffled with seed {random_seed}",
            flush=True,
        )
    if max_users:
        users = users[:max_users]
    n_up = int(max(1, user_processes or 1))
    print(
        f"[Pipeline] Processing {len(users)} users (parallel run), workers={workers}, "
        f"user_processes={n_up}, rounds={rounds}",
        flush=True,
    )

    # Load LLM models (None when using vLLM API)
    profile_model, profile_tokenizer = None, None
    action_model, action_tokenizer = None, None

    # Output file paths
    stem = input_path.stem
    pairs_file = output_path / f"dpo_pairs_{stem}.jsonl"
    detail_progress_file = output_path / f"dpo_detail_{stem}.jsonl"

    # Resume logic: read detail file, extract processed user IDs, rewrite pairs file
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

        # Rewrite pairs file from loaded results
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
            f"[Pipeline] Resume mode: loaded {len(all_results)} completed users, "
            f"DPO pairs written: {total_pairs_written}",
            flush=True,
        )
    else:
        # Fresh start: clear output files
        pairs_file.open("w", encoding="utf-8").close()
        detail_progress_file.open("w", encoding="utf-8").close()
        print(f"[Pipeline] Output DPO pairs: {pairs_file}", flush=True)
        print(f"[Pipeline] Output detail: {detail_progress_file}", flush=True)

    # Filter to pending users (not yet processed)
    pending_users = [
        u for u in users if str(u.get("user_id")) not in processed_user_ids
    ]
    if processed_user_ids:
        print(
            f"[Pipeline] Already processed {len(users) - len(pending_users)} users, "
            f"pending: {len(pending_users)}",
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
                f"[Pipeline] Parallel mode: {effective_procs} processes; "
                f"ThreadPool within each (workers={workers}). "
                f"Sentence-Transformer device: {sem_parallel}",
                flush=True,
            )
        else:
            print("[Pipeline] Loading Sentence Scorer (single process mode)...", flush=True)
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
            f"[Pipeline] Stagger enabled: user i sleeps i×{stagger_resolved:.2f}s "
            f"before loading models/calling APIs",
            flush=True,
        )

    def _flush_one_user(
        result: Dict[str, Any],
        fp_pairs,
        fp_detail,
    ) -> int:
        """
        Write single user result to disk (detail as one line + each DPO pair), then flush+fsync.
        Called when each user completes in both serial/parallel paths, enabling --resume to recover at user granularity.
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
                            f"\n[Pipeline] Completed {done_ct}/{n_pending} "
                            f"(user_id={result.get('user_id')})",
                            flush=True,
                        )
                        print(
                            f"[Pipeline] User {result['user_id']} DPO: "
                            f"{result.get('round_pair_distribution', {})}",
                            flush=True,
                        )
                        print(
                            f"[Pipeline] User {result['user_id']} wrote {pws} DPO pairs "
                            f"(total {total_pairs_written})",
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
                    "\n[Pipeline] Interrupted (Ctrl+C), shutting down...",
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
                    raise RuntimeError(f"Missing result for user index {k}")
            for k in range(n_pending):
                all_results.append(results_by_idx[k])
                processed_user_ids.add(str(results_by_idx[k].get("user_id")))
        else:
            for i, user_data in enumerate(pending_users):
                if cfg.DEBUG_LLM:
                    print(f"\n[Pipeline] Processing {i+1}/{len(pending_users)}", flush=True)
                t0 = time.time()
                if semantic_scorer is None:
                    raise RuntimeError("Internal error: semantic_scorer not initialized")
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
                    print(f"[Pipeline] User {result['user_id']} completed in {elapsed:.1f}s", flush=True)
                    print(
                        f"[Pipeline] User {result['user_id']} DPO: "
                        f"{result.get('round_pair_distribution', {})}",
                        flush=True,
                    )
                    print(
                        f"[Pipeline] User {result['user_id']} wrote {pws} DPO pairs "
                        f"(total {total_pairs_written})",
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
        f"\n[Pipeline] Total time: {total_elapsed:.1f}s | Average per user: {avg_user_elapsed:.1f}s",
        flush=True,
    )

    # Output summary
    print(f"\n[Pipeline] DPO pairs output: {pairs_file} ({total_pairs_written} pairs)", flush=True)

    detail_file = output_path / f"dpo_detail_{stem}.json"
    with detail_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[Pipeline] Detail output: {detail_file}", flush=True)

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
    print(f"[Pipeline] Summary output: {summary_file}", flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO Pipeline")
    parser.add_argument("--input", default=None, help="Input windowed jsonl file")
    parser.add_argument(
        "--input-dir",
        default=None,
        help="Input directory with jsonl files (alternative to --input)",
    )
    parser.add_argument(
        "--input-glob",
        default="community_*.jsonl",
        help="Glob pattern for input files (default: community_*.jsonl)",
    )
    parser.add_argument("--output-dir", default="output/dpo", help="Output directory for DPO pairs")
    parser.add_argument(
        "--max-users",
        "--max_user",
        dest="max_users",
        type=int,
        default=None,
        help="Maximum number of users to process (optional)",
    )
    parser.add_argument("--test", action="store_true", help="Test mode: use test API endpoint")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: print LLM calls, model roles, prompts, outputs",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for shuffling users; requires --max-users for reproducibility",
    )
    parser.add_argument(
        "--debug-actions",
        action="store_true",
        help="With --debug: also print full action prediction LLM calls (scenario + reply text)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DPO_WORKERS,
        help="Thread workers for candidate scoring (default from config.DPO_WORKERS)",
    )
    parser.add_argument(
        "--user-processes",
        type=int,
        default=DPO_USER_PROCESSES,
        dest="user_processes",
        help="Number of parallel user processes; --workers controls threads within each. 1=serial.",
    )
    parser.add_argument(
        "--scorer-device",
        default=None,
        help="Sentence-Transformer device: cpu / cuda. Default cpu; auto-detect if empty.",
    )
    parser.add_argument(
        "--user-process-stagger",
        type=float,
        default=None,
        dest="user_process_stagger_sec",
        help="Stagger delay: user i sleeps i× seconds (from config if not set; 0=no stagger)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=DPO_ROUNDS,
        help="DPO rounds per user (default 2): refine profile based on r_all reward",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume mode: skip already-processed users from dpo_detail_<stem>.jsonl",
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
            print(f"[Pipeline] Input directory not found: {dir_path}", flush=True)
            sys.exit(1)
        matched = sorted(dir_path.glob(args.input_glob))
        input_files.extend([str(p) for p in matched if p.is_file()])

    # Deduplicate input files
    input_files = list(dict.fromkeys(input_files))
    if not input_files:
        print("[Pipeline] No input files specified; use --input or --input-dir", flush=True)
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
        print(f"[Pipeline] Batch mode: processing {len(input_files)} files", flush=True)
        if not preflight_check():
            print("[Pipeline] Preflight check failed, exiting", flush=True)
            sys.exit(1)
        for i, fp in enumerate(input_files, 1):
            print(f"\n[Batch] ({i}/{len(input_files)}) Processing: {fp}", flush=True)
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
