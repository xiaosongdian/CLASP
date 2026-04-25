#!/usr/bin/env python3
"""
按时间窗口链评估画像策略（不构造 DPO 对）。

协议（与 src/dpo_pipeline 中单窗历史一致）：
- 用 W0 得到初始画像（或策略规定的画像）。
- 对 k=1..N-1：在「历史 = W_{k-1}」「目标 = W_k」上算 F/L/Q；
  若还有后续窗口且策略需要更新画像：
  - clasp_online：根据本窗口预测误差生成候选画像，在当前 (W_{k-1}, W_k) 上选 Q 最高者作为下一轮的画像；
  - prefix_refresh：用已观测动作 W0..W_k 重新做一次「初始画像」生成；
  - static_s0：画像始终不变。
"""
from __future__ import annotations

from typing import Any, Dict, List

from src.action_predictor import (
    build_behavior_discrepancies,
    predict_actions_for_window,
)
from src.config import DPO_WORKERS, NUM_CANDIDATE_PROFILES, TEMPERATURE_ACTION
from src.dpo_pipeline import evaluate_profile_on_window
from src.profile_generator import generate_candidate_profiles, generate_initial_profile
from src.scorer import SemanticScorer

VALID_METHODS = frozenset({"static_s0", "prefix_refresh", "clasp_online"})


def _sorted_window_keys(windows: Dict[str, Any]) -> List[str]:
    return sorted(
        [k for k in windows.keys() if k.startswith("W")],
        key=lambda x: int(x[1:]),
    )


def _concat_windows(
    windows: Dict[str, Any],
    keys: List[str],
    start_i: int,
    end_i: int,
) -> List[Dict]:
    """闭区间 [start_i, end_i] 在 keys 上的动作拼接。"""
    out: List[Dict] = []
    for i in range(start_i, end_i + 1):
        out.extend(windows[keys[i]])
    return out


def evaluate_user_window_chain(
    user_record: Dict[str, Any],
    method: str,
    semantic_scorer: SemanticScorer,
    *,
    profile_model=None,
    profile_tokenizer=None,
    action_model=None,
    action_tokenizer=None,
    num_candidates: int = NUM_CANDIDATE_PROFILES,
    workers: int = DPO_WORKERS,
) -> Dict[str, Any]:
    """
    对单条窗口化用户记录执行窗口链评估。

    method:
      - static_s0: 仅用 W0 生成的初始画像预测 W1..W_{T-1}，不更新。
      - prefix_refresh: 预测 W_k 前，用 W0..W_{k-1} 上观测到的动作整段重算初始画像。
      - clasp_online: 每步预测后按行为偏差精炼候选，在当前窗口任务上取 Q 最高的画像进入下一步。
    """
    uid = user_record.get("user_id")
    cid = user_record.get("community_id")
    windows = user_record.get("windows") or {}

    if method not in VALID_METHODS:
        return {
            "user_id": uid,
            "community_id": cid,
            "method": method,
            "error": f"unknown_method (valid: {sorted(VALID_METHODS)})",
            "steps": [],
        }

    keys = _sorted_window_keys(windows)
    if len(keys) < 2:
        return {
            "user_id": uid,
            "community_id": cid,
            "method": method,
            "error": "need_at_least_2_windows",
            "steps": [],
        }

    w0_actions = windows[keys[0]]
    profile = generate_initial_profile(profile_model, profile_tokenizer, w0_actions)

    steps_out: List[Dict[str, Any]] = []

    for step_idx in range(1, len(keys)):
        hist = windows[keys[step_idx - 1]]
        targets = windows[keys[step_idx]]

        f_s, l_s, q_s = evaluate_profile_on_window(
            profile,
            hist,
            targets,
            action_model,
            action_tokenizer,
            semantic_scorer,
        )
        steps_out.append(
            {
                "step_index": step_idx,
                "history_window": keys[step_idx - 1],
                "target_window": keys[step_idx],
                "F": f_s,
                "L": l_s,
                "Q": q_s,
            }
        )

        is_last = step_idx >= len(keys) - 1
        if is_last:
            break

        if method == "static_s0":
            continue

        if method == "prefix_refresh":
            prefix_actions = _concat_windows(windows, keys, 0, step_idx)
            profile = generate_initial_profile(
                profile_model, profile_tokenizer, prefix_actions
            )
            continue

        if method == "clasp_online":
            preds = predict_actions_for_window(
                action_model,
                action_tokenizer,
                profile,
                hist,
                targets,
                temperature=TEMPERATURE_ACTION,
            )
            discrepancies = build_behavior_discrepancies(preds, targets, hist)
            candidates = generate_candidate_profiles(
                profile_model,
                profile_tokenizer,
                profile,
                discrepancies,
                n=num_candidates,
                workers=workers,
            )
            best_profile = profile
            best_q = q_s
            for cand in candidates:
                if not (cand or "").strip():
                    continue
                fc, lc, qc = evaluate_profile_on_window(
                    cand,
                    hist,
                    targets,
                    action_model,
                    action_tokenizer,
                    semantic_scorer,
                )
                if qc > best_q:
                    best_q = qc
                    best_profile = cand
            profile = best_profile
            continue

    qs = [float(s["Q"]) for s in steps_out]
    fs = [float(s["F"]) for s in steps_out]
    mean_q = sum(qs) / len(qs) if qs else None
    mean_f = sum(fs) / len(fs) if fs else None

    return {
        "user_id": uid,
        "community_id": cid,
        "method": method,
        "window_keys": keys,
        "num_candidates_config": num_candidates,
        "steps": steps_out,
        "mean_Q": mean_q,
        "mean_F": mean_f,
    }
