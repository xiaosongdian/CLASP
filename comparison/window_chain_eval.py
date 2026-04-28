#!/usr/bin/env python3
"""
按时间窗口链评估画像策略（不构造 DPO 对）。

协议（W0 建画像 + 前向跨窗预测）：
- 用 **W0** 得到初始画像 **S0**。
- 共 N 窗 W0..W_{N-1} 时评估 N-1 步：`step_index=0..N-2`，第 t 步为
  「历史 W_t → 预测 W_{t+1}」（S0→W1，S1→W2，…）。
  N=6（W0..W5，共 6*T 条动作）时共 5 个采样点，至 S4 预测 W5。
- 每步结束后（非最后一步）若策略需要更新画像：
  - clasp_online：按预测误差精炼；每步动作模型只调用一轮 predict。
  - prefix_refresh：用已观测 W0..W_{t+1} 重算初始画像；
  - static_s0：画像保持 S0。
  - incremental_persona：用 S_{t-1} + **当前窗**短期行为（无预测误差）精炼得到 S_t。
  - s0_sliding_history：画像固定 S0，预测时在 prompt 中附加 **当前窗**行为全文。
  - user_full_history：无单独长期画像，仅用 **W0..W_t 拼接**的全量行为块驱动预测。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.action_predictor import (
    build_behavior_discrepancies,
    predict_actions_for_window,
)
from src.config import ALPHA, DPO_WORKERS, TEMPERATURE_ACTION
from src.dpo_pipeline import evaluate_profile_on_window
from src.config import ACTION_PROMPT_HISTORY_MAX_CHARS
from src.profile_generator import (
    format_behavior_data,
    generate_candidate_profiles,
    generate_initial_profile,
    truncate_behavior_plaintext,
)
from src.scorer import SemanticScorer, evaluate_predictions

VALID_METHODS = frozenset(
    {
        "static_s0",
        "prefix_refresh",
        "clasp_online",
        "incremental_persona",
        "s0_sliding_history",
        "user_full_history",
    }
)


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


def _incremental_refine_block(hist_actions: List[Dict], window_key: str) -> str:
    """增量画像：无预测误差，仅用当前窗真实行为驱动精炼。"""
    return (
        "(Incremental persona update: no predicted-vs-actual errors.)\n"
        f"Align the persona with these **actual behaviors in {window_key}**:\n\n"
        + format_behavior_data(hist_actions)
    )


def evaluate_user_window_chain(
    user_record: Dict[str, Any],
    method: str,
    semantic_scorer: SemanticScorer,
    *,
    profile_model=None,
    profile_tokenizer=None,
    action_model=None,
    action_tokenizer=None,
    refinement_variants: int = 1,
    workers: int = DPO_WORKERS,
    always_accept_refinement: bool = False,
) -> Dict[str, Any]:
    """
    对单条窗口化用户记录执行窗口链评估。

    method:
      - static_s0: 仅用 W0 生成的 S0，逐步以 W_t 为历史预测 W_{t+1}，不更新画像。
      - prefix_refresh: 每步预测后，用已观测的 W0..W_{t+1} 重算初始画像再进入下一步。
      - clasp_online: 每步按误差精炼画像；默认 refinement_variants=1。
        默认与旧画像比 Q 取优；always_accept_refinement=True 时总是采用新精炼（多份时取首个非空）。
      - incremental_persona: 每步后用 S_{t-1} 与当前窗行为（无误差信号）单次精炼更新画像。
      - s0_sliding_history: 画像固定 S0；每步在动作 prompt 中附加**已观测历史窗 W_t** 的行为全文（不含待预测的 W_{t+1}）。
      - user_full_history: 不显式维护画像；每步在 prompt 中附加 **W0..W_t** 拼接行为（不含目标窗）。
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
    s0_fixed = generate_initial_profile(profile_model, profile_tokenizer, w0_actions)
    profile = s0_fixed

    steps_out: List[Dict[str, Any]] = []

    n_keys = len(keys)
    for step_idx in range(n_keys - 1):
        # 协议：hist = 已观测窗 W_t，targets = 待预测窗 W_{t+1}；不得把 targets 写入 profile_suffix。
        hist = windows[keys[step_idx]]
        targets = windows[keys[step_idx + 1]]
        wkey = keys[step_idx]

        profile_suffix: Optional[str] = None
        eval_profile = profile
        if method == "s0_sliding_history":
            eval_profile = s0_fixed
            # 仅附加「当前历史窗」W_t 的行为全文，不含目标窗 W_{t+1}（避免标签泄漏）。
            profile_suffix = (
                f"### Explicit recent behaviors (observed window {wkey}, not the prediction target)\n"
                + format_behavior_data(hist)
            )
        elif method == "user_full_history":
            eval_profile = (
                "No separate long-term persona. Infer user preferences and likely actions "
                "only from the cumulative behavior log below."
            )
            # W0..W_t 闭区间，即截至已观测最后一窗；不含 W_{t+1}。
            cum_actions = _concat_windows(windows, keys, 0, step_idx)
            profile_suffix = (
                "### Cumulative user behavior (W0 through observed window only)\n"
                + format_behavior_data(cum_actions)
            )

        if profile_suffix and int(ACTION_PROMPT_HISTORY_MAX_CHARS) > 0:
            profile_suffix = truncate_behavior_plaintext(
                profile_suffix, int(ACTION_PROMPT_HISTORY_MAX_CHARS)
            )

        # clasp_online：本步只预测一次，评分与 discrepancy 共用同一组 preds（避免重复打动作 API）
        preds_for_refine: Optional[List[Dict]] = None
        if method == "clasp_online":
            preds_for_refine = predict_actions_for_window(
                action_model,
                action_tokenizer,
                profile,
                hist,
                targets,
                temperature=TEMPERATURE_ACTION,
                profile_suffix=None,
            )
            f_s, l_s, q_s = evaluate_predictions(
                preds_for_refine, targets, semantic_scorer, ALPHA
            )
        else:
            f_s, l_s, q_s = evaluate_profile_on_window(
                eval_profile,
                hist,
                targets,
                action_model,
                action_tokenizer,
                semantic_scorer,
                profile_suffix=profile_suffix,
            )
        steps_out.append(
            {
                "step_index": step_idx,
                "history_window": keys[step_idx],
                "target_window": keys[step_idx + 1],
                "F": f_s,
                "L": l_s,
                "Q": q_s,
            }
        )

        is_last = step_idx >= n_keys - 2
        if is_last:
            break

        if method in ("static_s0", "s0_sliding_history", "user_full_history"):
            continue

        if method == "prefix_refresh":
            prefix_actions = _concat_windows(windows, keys, 0, step_idx + 1)
            profile = generate_initial_profile(
                profile_model, profile_tokenizer, prefix_actions
            )
            continue

        if method == "incremental_persona":
            block = _incremental_refine_block(hist, wkey)
            candidates = generate_candidate_profiles(
                profile_model,
                profile_tokenizer,
                profile,
                block,
                n=1,
                workers=1,
            )
            if candidates and (candidates[0] or "").strip():
                profile = candidates[0]
            continue

        if method == "clasp_online":
            discrepancies = build_behavior_discrepancies(
                preds_for_refine, targets, hist
            )
            n_var = max(1, int(refinement_variants))
            candidates = generate_candidate_profiles(
                profile_model,
                profile_tokenizer,
                profile,
                discrepancies,
                n=n_var,
                workers=min(workers, n_var),
            )
            if always_accept_refinement:
                new_p = profile
                for cand in candidates:
                    if (cand or "").strip():
                        new_p = cand
                        break
                profile = new_p
            else:
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
    # 全部为前向跨窗步，与 mean_Q / mean_F 相同；保留字段供下游脚本兼容
    chain_steps = list(steps_out)
    qc = [float(s["Q"]) for s in chain_steps]
    fc = [float(s["F"]) for s in chain_steps]
    mean_q_chain = sum(qc) / len(qc) if qc else None
    mean_f_chain = sum(fc) / len(fc) if fc else None

    return {
        "user_id": uid,
        "community_id": cid,
        "method": method,
        "window_keys": keys,
        "refinement_variants": refinement_variants,
        "always_accept_refinement": always_accept_refinement,
        "steps": steps_out,
        "mean_Q": mean_q,
        "mean_F": mean_f,
        "mean_Q_chain": mean_q_chain,
        "mean_F_chain": mean_f_chain,
    }
