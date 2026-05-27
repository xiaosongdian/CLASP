#!/usr/bin/env python3
"""
Parallelized action predictor

Key improvements:
1. When predicting window W_{t+1}, all actions use same history window W_t
2. Don't use sliding history within window
3. Can predict all actions in window in parallel
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Tuple
import src.config as cfg
from src.action_predictor import (
    build_decision_prompt,
    build_content_prompt,
    parse_action_type,
    invoke_action_llm,
)


def predict_single_action(
    model,
    tokenizer,
    user_profile: str,
    history_actions: List[Dict],
    target_action: Dict,
    max_new_tokens_decision: int,
    max_new_tokens_content: int,
    temperature: float,
    action_idx: int,
    total_actions: int,
) -> Tuple[int, Dict]:
    """
    Predict single action (for parallel invocation)

    Args:
        action_idx: Action index (for sorting)
        total_actions: Total action count (for debug)

    Returns:
        (action_idx, prediction)
    """
    actual_type = target_action.get("action_type", "")  # Get actual type

    # Decision prediction
    inst, inp = build_decision_prompt(user_profile, history_actions, target_action)
    raw_decision = invoke_action_llm(
        model,
        tokenizer,
        inst,
        inp,
        max_new_tokens_decision,
        temperature,
        debug_step=f"action_prediction:decision#{action_idx + 1}/{total_actions}",
    )
    pred_type = parse_action_type(raw_decision)

    # Content prediction (judge by actual type, not predicted type)
    pred_content = None
    if actual_type in ("post", "reply"):  # Key change: use actual_type
        inst_c, inp_c = build_content_prompt(user_profile, history_actions, target_action)
        pred_content = invoke_action_llm(
            model,
            tokenizer,
            inst_c,
            inp_c,
            max_new_tokens_content,
            temperature,
            debug_step=f"action_prediction:content#{action_idx + 1}/{total_actions}",
        )

    return action_idx, {
        "action_type": pred_type,
        "content": pred_content,
    }


def predict_actions_for_window_parallel(
    model,
    tokenizer,
    profile: str,
    history_actions: List[Dict],
    target_actions: List[Dict],
    max_new_tokens_decision: int = 128,
    max_new_tokens_content: int = 512,
    temperature: float = 0.3,
    profile_suffix: Optional[str] = None,
    workers: int = 10,
) -> List[Dict]:
    """
    Predict all actions in window in parallel

    Key improvements:
    - All actions use same history_actions (history window)
    - Don't use sliding history within window
    - Can predict all actions in parallel

    Args:
        history_actions: History window (W_t); if already set to empty by upper layer, no Recent user actions in prompt
        target_actions: Target window (W_{t+1})
        workers: Number of parallel threads

    Returns:
        Prediction list: [{"action_type": str, "content": str|None}, ...]
    """
    user_profile = (profile + (f"\n\n{profile_suffix}" if (profile_suffix or "").strip() else "")).strip()
    n = len(target_actions)

    if n == 0:
        return []

    # Truncate history to window size
    hw = max(1, int(getattr(cfg, "ACTION_PREDICTION_HISTORY_WINDOW", 5)))
    history_context = history_actions[-hw:] if history_actions else []

    # Prepare predictions dict
    predictions_dict = {}
    eff_workers = max(1, min(workers, n))

    if eff_workers == 1:
        # Serial mode (single worker)
        for i, target in enumerate(target_actions):
            idx, pred = predict_single_action(
                model,
                tokenizer,
                user_profile,
                history_context,
                target,
                max_new_tokens_decision,
                max_new_tokens_content,
                temperature,
                i,
                n,
            )
            predictions_dict[idx] = pred
    else:
        # Parallel mode (multiple workers)
        with ThreadPoolExecutor(max_workers=eff_workers, thread_name_prefix="action_pred") as pool:
            futs = {
                pool.submit(
                    predict_single_action,
                    model,
                    tokenizer,
                    user_profile,
                    history_context,
                    target,
                    max_new_tokens_decision,
                    max_new_tokens_content,
                    temperature,
                    i,
                    n,
                ): i
                for i, target in enumerate(target_actions)
            }

            for fut in as_completed(futs):
                idx, pred = fut.result()
                predictions_dict[idx] = pred

    # Restore order
    predictions = [predictions_dict[i] for i in range(n)]
    return predictions
