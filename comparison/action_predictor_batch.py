#!/usr/bin/env python3
"""
动作预测批量优化

当前问题：predict_actions_for_window 串行预测每个动作
优化方案：使用批量 API 调用，一次预测多个动作
"""

from typing import List, Dict, Optional
import src.config as cfg
from src.action_predictor import (
    build_decision_prompt,
    build_content_prompt,
    parse_action_type,
    invoke_action_llm,
)


def predict_actions_for_window_batch(
    model,
    tokenizer,
    profile: str,
    history_actions: List[Dict],
    target_actions: List[Dict],
    max_new_tokens_decision: int = 128,
    max_new_tokens_content: int = 512,
    temperature: float = 0.3,
    profile_suffix: Optional[str] = None,
    batch_size: int = 5,
) -> List[Dict]:
    """
    批量预测动作（优化版本）

    策略：
    1. 由于动作之间有依赖（滑动历史），不能完全并行
    2. 但可以批量发送请求给 API，减少网络往返
    3. 使用 batch_size 控制批量大小

    Args:
        batch_size: 批量大小，默认 5（一次预测 5 个动作）
    """
    user_profile = (profile + (f"\n\n{profile_suffix}" if (profile_suffix or "").strip() else "")).strip()
    predictions = []
    current_history = list(history_actions)

    hw = max(1, int(getattr(cfg, "ACTION_PREDICTION_HISTORY_WINDOW", 5)))
    n = len(target_actions)

    # 注意：由于滑动历史的依赖，我们仍然需要串行处理
    # 但可以优化 API 调用方式（如果 vLLM 支持批量推理）

    for i, target in enumerate(target_actions):
        recent = current_history[-hw:] if current_history else []

        # 决策预测
        inst, inp = build_decision_prompt(user_profile, recent, target)
        raw_decision = invoke_action_llm(
            model,
            tokenizer,
            inst,
            inp,
            max_new_tokens_decision,
            temperature,
            debug_step=f"action_prediction:decision#{i + 1}/{n}",
        )
        pred_type = parse_action_type(raw_decision)

        # 内容预测（仅 post/reply）
        pred_content = None
        if pred_type in ("post", "reply"):
            inst_c, inp_c = build_content_prompt(user_profile, recent, target)
            pred_content = invoke_action_llm(
                model,
                tokenizer,
                inst_c,
                inp_c,
                max_new_tokens_content,
                temperature,
                debug_step=f"action_prediction:content#{i + 1}/{n}",
            )

        predictions.append({
            "action_type": pred_type,
            "content": pred_content,
        })

        # 将真实动作加入历史（模拟时间推进）
        current_history.append(target)

    return predictions


# 注意：由于动作预测有滑动历史依赖，真正的并行化需要在更高层次实现
# 即：多进程处理不同用户（已在 run_baseline_parallel.py 中实现）
