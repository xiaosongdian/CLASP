#!/usr/bin/env python3
"""
并行化的动作预测器

关键改进：
1. 预测窗口 W_{t+1} 时，所有动作使用相同的历史窗口 W_t
2. 不使用窗口内部的滑动历史
3. 可以并行预测窗口内的所有动作
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
    预测单个动作（用于并行调用）

    Args:
        action_idx: 动作索引（用于排序）
        total_actions: 总动作数（用于 debug）

    Returns:
        (action_idx, prediction)
    """
    # 决策预测
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

    # 内容预测（仅 post/reply）
    pred_content = None
    if pred_type in ("post", "reply"):
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
    并行预测窗口内的所有动作

    关键改进：
    - 所有动作使用相同的 history_actions（历史窗口）
    - 不使用窗口内部的滑动历史
    - 可以并行预测所有动作

    Args:
        history_actions: 历史窗口（W_t）；若由上层已置空则 prompt 中无 Recent user actions
        target_actions: 目标窗口（W_{t+1}）
        workers: 并行线程数

    Returns:
        预测列表：[{"action_type": str, "content": str|None}, ...]
    """
    user_profile = (profile + (f"\n\n{profile_suffix}" if (profile_suffix or "").strip() else "")).strip()
    n = len(target_actions)

    if n == 0:
        return []

    # 使用历史窗口的最后几条作为上下文（固定）
    hw = max(1, int(getattr(cfg, "ACTION_PREDICTION_HISTORY_WINDOW", 5)))
    history_context = history_actions[-hw:] if history_actions else []

    # 并行预测所有动作
    predictions_dict = {}
    eff_workers = max(1, min(workers, n))

    if eff_workers == 1:
        # 串行模式（调试用）
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
        # 并行模式
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

    # 按索引排序返回
    predictions = [predictions_dict[i] for i in range(n)]
    return predictions
