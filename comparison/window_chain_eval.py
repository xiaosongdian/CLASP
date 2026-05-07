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

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import src.config as cfg

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


def evaluate_three_windows(
    old_profile: str,
    new_profile: str,
    windows: Dict[str, Any],
    keys: List[str],
    step_idx: int,
    action_model,
    action_tokenizer,
    semantic_scorer: SemanticScorer,
    current_step_scores: Optional[Dict[str, float]] = None,
    *,
    action_prompt_include_observed_history: bool = True,
) -> Dict[str, Any]:
    """
    在三个窗口上评估旧画像和新画像的性能。

    Args:
        old_profile: 更新前的画像
        new_profile: 更新后的画像
        windows: 所有窗口数据
        keys: 窗口键列表（如 ['W0', 'W1', 'W2', 'W3', 'W4', 'W5']）
        step_idx: 当前步骤索引
        action_model: 动作预测模型
        action_tokenizer: 动作预测分词器
        semantic_scorer: 语义评分器
        current_step_scores: 当前步骤的评估结果 {"F": f_s, "L": l_s, "Q": q_s}，
                            可以复用作为未来窗口的 new_profile 评估结果

    Returns:
        包含三个窗口评估结果的字典
    """
    result = {}

    # 过去窗口：step_idx - 1（如果存在）
    if step_idx > 0:
        # 过去窗口：用 W_{i-2} 预测 W_{i-1}
        # history = W_{i-2}, target = W_{i-1}
        if step_idx > 1:
            past_history = windows[keys[step_idx - 2]]
        else:
            past_history = []
        past_target = windows[keys[step_idx - 1]]

        # 用旧画像评估
        f_old, l_old, q_old = evaluate_profile_on_window(
            old_profile,
            past_history,
            past_target,
            action_model,
            action_tokenizer,
            semantic_scorer,
            profile_suffix=None,
            include_observed_history=action_prompt_include_observed_history,
        )

        # 用新画像评估
        f_new, l_new, q_new = evaluate_profile_on_window(
            new_profile,
            past_history,
            past_target,
            action_model,
            action_tokenizer,
            semantic_scorer,
            profile_suffix=None,
            include_observed_history=action_prompt_include_observed_history,
        )

        result["past_window"] = {
            "history": keys[step_idx - 2] if step_idx > 1 else "empty",
            "target": keys[step_idx - 1],
            "with_old_profile": {"F": f_old, "L": l_old, "Q": q_old},
            "with_new_profile": {"F": f_new, "L": l_new, "Q": q_new},
            "gain": {"ΔF": f_new - f_old, "ΔL": l_new - l_old, "ΔQ": q_new - q_old},
        }

    # 当前窗口：step_idx
    # 用 W_{i-1} 预测 W_i
    # history = W_{i-1}, target = W_i
    if step_idx > 0:
        current_history = windows[keys[step_idx - 1]]
    else:
        current_history = []
    current_target = windows[keys[step_idx]]

    # 用旧画像评估
    f_old, l_old, q_old = evaluate_profile_on_window(
        old_profile,
        current_history,
        current_target,
        action_model,
        action_tokenizer,
        semantic_scorer,
        profile_suffix=None,
        include_observed_history=action_prompt_include_observed_history,
    )

    # 用新画像评估
    f_new, l_new, q_new = evaluate_profile_on_window(
        new_profile,
        current_history,
        current_target,
        action_model,
        action_tokenizer,
        semantic_scorer,
        profile_suffix=None,
        include_observed_history=action_prompt_include_observed_history,
    )

    result["current_window"] = {
        "history": keys[step_idx - 1] if step_idx > 0 else "empty",
        "target": keys[step_idx],
        "with_old_profile": {"F": f_old, "L": l_old, "Q": q_old},
        "with_new_profile": {"F": f_new, "L": l_new, "Q": q_new},
        "gain": {"ΔF": f_new - f_old, "ΔL": l_new - l_old, "ΔQ": q_new - q_old},
    }

    # 未来窗口：step_idx + 1（如果存在）
    if step_idx + 1 < len(keys):
        # 用 W_i 预测 W_{i+1}
        # history = W_i, target = W_{i+1}
        # 注意：这正是当前步骤正在评估的，可以复用结果
        future_history = windows[keys[step_idx]]
        future_target = windows[keys[step_idx + 1]]

        # 用旧画像评估
        f_old, l_old, q_old = evaluate_profile_on_window(
            old_profile,
            future_history,
            future_target,
            action_model,
            action_tokenizer,
            semantic_scorer,
            profile_suffix=None,
            include_observed_history=action_prompt_include_observed_history,
        )

        # 用新画像评估
        # 如果提供了 current_step_scores，可以直接复用（避免重复计算）
        if current_step_scores is not None:
            f_new = current_step_scores["F"]
            l_new = current_step_scores["L"]
            q_new = current_step_scores["Q"]
        else:
            f_new, l_new, q_new = evaluate_profile_on_window(
                new_profile,
                future_history,
                future_target,
                action_model,
                action_tokenizer,
                semantic_scorer,
                profile_suffix=None,
                include_observed_history=action_prompt_include_observed_history,
            )

        result["future_window"] = {
            "history": keys[step_idx],
            "target": keys[step_idx + 1],
            "with_old_profile": {"F": f_old, "L": l_old, "Q": q_old},
            "with_new_profile": {"F": f_new, "L": l_new, "Q": q_new},
            "gain": {"ΔF": f_new - f_old, "ΔL": l_new - l_old, "ΔQ": q_new - q_old},
        }

    return result

VALID_METHODS = frozenset(
    {
        "static_s0",
        "prefix_refresh",
        "clasp_online",
        "incremental_persona",
    }
)


def _sorted_window_keys(windows: Dict[str, Any]) -> List[str]:
    return sorted(
        [k for k in windows.keys() if k.startswith("W")],
        key=lambda x: int(x[1:]),
    )


def _append_clasp_profile_snapshot(
    snap_dir: Path,
    uid: Any,
    cid: Any,
    record: Dict[str, Any],
) -> None:
    """将一行 JSON 追加到该用户的 clasp_online 画像快照文件。"""
    snap_dir.mkdir(parents=True, exist_ok=True)
    fn = snap_dir / f"user_{uid}_c_{cid}.jsonl"
    with fn.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


@contextmanager
def _comparison_vllm_model_scope(method: str):
    """
    baseline 对比：clasp_online 与其它方法使用不同 vLLM model id（读 cfg 运行时字段）。
    须与 PROFILE_API_BASE / ACTION_API_BASE 上各服务端注册的模型名一致。
    """
    old_p, old_a = cfg.PROFILE_API_MODEL, cfg.ACTION_API_MODEL
    try:
        if method == "clasp_online":
            cfg.PROFILE_API_MODEL = cfg.COMPARISON_CLASP_PROFILE_VLLM_MODEL
            cfg.ACTION_API_MODEL = cfg.COMPARISON_CLASP_ACTION_VLLM_MODEL
        else:
            cfg.PROFILE_API_MODEL = cfg.COMPARISON_BASELINE_VLLM_MODEL
            cfg.ACTION_API_MODEL = cfg.COMPARISON_BASELINE_VLLM_MODEL
        yield
    finally:
        cfg.PROFILE_API_MODEL, cfg.ACTION_API_MODEL = old_p, old_a


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
    profile_snapshot_dir: Optional[Path] = None,
    action_prompt_include_observed_history: bool = True,
) -> Dict[str, Any]:
    """
    对单条窗口化用户记录执行窗口链评估。

    method:
      - static_s0: 仅用 W0 生成的 S0，逐步以 W_t 为历史预测 W_{t+1}，不更新画像。
      - prefix_refresh: 每步预测后，用已观测的 W0..W_{t+1} 重算初始画像再进入下一步。
      - clasp_online: 每步按误差精炼画像；默认 refinement_variants=1。
        默认与旧画像比 Q 取优；always_accept_refinement=True 时总是采用新精炼（多份时取首个非空）。
      - incremental_persona: 每步后用 S_{t-1} 与当前窗行为（无误差信号）单次精炼更新画像。

    注意：所有方法都使用统一的历史输入机制（profile_suffix），确保公平对比。
    若 action_prompt_include_observed_history=False，则动作 API prompt 中不再附带观测到的历史动作
    （既不拼 profile_suffix，也不在 Recent user actions 中使用滑窗），用于消融实验。

    profile_snapshot_dir:
      仅当 method 为 clasp_online 且目录非空时，写入每用户一个 ``user_<id>_c_<cid>.jsonl``
      （含 W0 初始一行 + 每链一步一行）；其它 method 忽略此参数。
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

    snap_dir: Optional[Path] = None
    if method == "clasp_online" and profile_snapshot_dir is not None:
        snap_dir = Path(profile_snapshot_dir).resolve()

    with _comparison_vllm_model_scope(method):
        if snap_dir is not None:
            snap_dir.mkdir(parents=True, exist_ok=True)
            (snap_dir / f"user_{uid}_c_{cid}.jsonl").unlink(missing_ok=True)

        w0_actions = windows[keys[0]]
        s0_fixed = generate_initial_profile(profile_model, profile_tokenizer, w0_actions)
        profile = s0_fixed

        if snap_dir is not None:
            _append_clasp_profile_snapshot(
                snap_dir,
                uid,
                cid,
                {
                    "user_id": uid,
                    "community_id": cid,
                    "method": method,
                    "phase": "after_W0_initial",
                    "step_index": None,
                    "history_window": None,
                    "target_window": None,
                    "profile": s0_fixed,
                    "profile_length": len(s0_fixed or ""),
                },
            )
    
        steps_out: List[Dict[str, Any]] = []
    
        n_keys = len(keys)
        for step_idx in range(n_keys - 1):
            profile_at_step_start = profile
            # 协议：hist = 已观测窗 W_t，targets = 待预测窗 W_{t+1}；不得把 targets 写入 profile_suffix。
            hist = windows[keys[step_idx]]
            targets = windows[keys[step_idx + 1]]
            wkey = keys[step_idx]
    
            # === 统一历史机制：所有方法都使用相同的 profile_suffix（可关闭以仅保留画像+scenario）===
            if action_prompt_include_observed_history:
                profile_suffix = (
                    f"### Recent behaviors (observed window {wkey})\n"
                    + format_behavior_data(hist)
                )

                if profile_suffix and int(ACTION_PROMPT_HISTORY_MAX_CHARS) > 0:
                    profile_suffix = truncate_behavior_plaintext(
                        profile_suffix, int(ACTION_PROMPT_HISTORY_MAX_CHARS)
                    )
            else:
                profile_suffix = None
    
            # 根据方法设置画像
            if method == "static_s0":
                eval_profile = s0_fixed
            else:
                eval_profile = profile
    
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
                    profile_suffix=profile_suffix,
                    include_observed_history=action_prompt_include_observed_history,
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
                    include_observed_history=action_prompt_include_observed_history,
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
    
            # 判断是否是最后一步
            is_last = step_idx >= n_keys - 2
    
            # 保存旧画像用于三窗口对比
            old_profile = profile
    
            # 画像更新（所有步骤都执行，包括最后一步）
            if method == "static_s0":
                # static_s0: 画像不变
                pass
    
            elif method == "prefix_refresh":
                prefix_actions = _concat_windows(windows, keys, 0, step_idx + 1)
                profile = generate_initial_profile(
                    profile_model, profile_tokenizer, prefix_actions
                )
    
            elif method == "incremental_persona":
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
    
            elif method == "clasp_online":
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
    
                    # 记录候选画像信息
                    candidate_scores = []
    
                    if always_accept_refinement:
                        new_p = profile
                        for idx, cand in enumerate(candidates):
                            if (cand or "").strip():
                                new_p = cand
                                steps_out[-1]["best_candidate_index"] = idx
                                break
                        profile = new_p
                        steps_out[-1]["profile_updated"] = (new_p != old_profile)
                    else:
                        best_profile = profile
                        best_q = q_s
                        best_idx = -1
    
                        # 使用下一步窗口评估候选画像，避免在训练集（当前窗口）上过拟合
                        # 下一步：hist = W_{t+1}（当前 targets），target = W_{t+2}
                        # 注意：需要检查 step_idx + 2 是否在范围内
                        if step_idx + 2 < len(keys):
                            next_hist = targets
                            next_targets = windows[keys[step_idx + 2]]
                            if action_prompt_include_observed_history:
                                next_suffix = (
                                    f"### Recent behaviors (observed window {keys[step_idx + 1]})\n"
                                    + format_behavior_data(next_hist)
                                )
                                if next_suffix and int(ACTION_PROMPT_HISTORY_MAX_CHARS) > 0:
                                    next_suffix = truncate_behavior_plaintext(
                                        next_suffix, int(ACTION_PROMPT_HISTORY_MAX_CHARS)
                                    )
                            else:
                                next_suffix = None
    
                            for idx, cand in enumerate(candidates):
                                if not (cand or "").strip():
                                    continue
                                fc, lc, qc = evaluate_profile_on_window(
                                    cand,
                                    next_hist,
                                    next_targets,
                                    action_model,
                                    action_tokenizer,
                                    semantic_scorer,
                                    profile_suffix=next_suffix,
                                    include_observed_history=action_prompt_include_observed_history,
                                )
                                candidate_scores.append({
                                    "index": idx,
                                    "F": fc,
                                    "L": lc,
                                    "Q": qc,
                                })
                                if qc > best_q:
                                    best_q = qc
                                    best_profile = cand
                                    best_idx = idx
                        else:
                            # 最后一步：没有下一个窗口，直接使用当前窗口的 Q 值选择最佳候选
                            # 或者直接接受第一个候选
                            if candidates and (candidates[0] or "").strip():
                                best_profile = candidates[0]
                                best_idx = 0
    
                        profile = best_profile
                        steps_out[-1]["profile_updated"] = (best_profile != old_profile)
                        steps_out[-1]["best_candidate_index"] = best_idx
                        steps_out[-1]["candidate_scores"] = candidate_scores
    
                    # 记录画像长度
                    steps_out[-1]["profile_length"] = len(profile)
                    steps_out[-1]["num_candidates"] = len(candidates)
    
            if snap_dir is not None:
                _append_clasp_profile_snapshot(
                    snap_dir,
                    uid,
                    cid,
                    {
                        "user_id": uid,
                        "community_id": cid,
                        "method": method,
                        "phase": "after_chain_step",
                        "step_index": step_idx,
                        "history_window": keys[step_idx],
                        "target_window": keys[step_idx + 1],
                        "profile_before_prediction": profile_at_step_start,
                        "profile_after_step": profile,
                        "profile_length_before": len(profile_at_step_start or ""),
                        "profile_length_after": len(profile or ""),
                    },
                )

            # === 三窗口评估（所有方法）===
            # 在最后一步进行三窗口评估
            # 最后一步：当前评估 W_{n-1} → W_n，可以复用这个结果作为未来窗口
            is_last_step = (step_idx >= n_keys - 2)
    
            # 记录画像变化前后在三个窗口上的表现
            should_evaluate_three_windows = False
    
            if method == "static_s0":
                # static_s0: 只在最后一步记录一次（作为基线）
                should_evaluate_three_windows = is_last_step
            else:
                # 其他方法: 只在最后一步且画像有更新时记录
                if is_last_step and old_profile != profile:
                    should_evaluate_three_windows = True
    
            if should_evaluate_three_windows:
                # 传入当前步骤的评估结果，可以复用作为未来窗口的 new_profile 评估
                three_window_eval = evaluate_three_windows(
                    old_profile=old_profile,
                    new_profile=profile,
                    windows=windows,
                    keys=keys,
                    step_idx=step_idx,
                    action_model=action_model,
                    action_tokenizer=action_tokenizer,
                    semantic_scorer=semantic_scorer,
                    current_step_scores={"F": f_s, "L": l_s, "Q": q_s},
                    action_prompt_include_observed_history=action_prompt_include_observed_history,
                )
                steps_out[-1]["three_window_evaluation"] = three_window_eval
                steps_out[-1]["profile_changed"] = (old_profile != profile)
    
        qs = [float(s["Q"]) for s in steps_out]
        fs = [float(s["F"]) for s in steps_out]
        mean_q = sum(qs) / len(qs) if qs else None
        mean_f = sum(fs) / len(fs) if fs else None
    
        return {
            "user_id": uid,
            "community_id": cid,
            "method": method,
            "window_keys": keys,
            "refinement_variants": refinement_variants,
            "always_accept_refinement": always_accept_refinement,
            "action_prompt_include_observed_history": action_prompt_include_observed_history,
            "steps": steps_out,
            "mean_Q": mean_q,
            "mean_F": mean_f,
            # 保留 mean_Q_chain / mean_F_chain 字段供下游脚本兼容（与 mean_Q / mean_F 相同）
            "mean_Q_chain": mean_q,
            "mean_F_chain": mean_f,
        }
