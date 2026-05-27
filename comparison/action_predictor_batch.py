#!/usr/bin/env python3
"""
: predict_actions_for_window 
: uses API, 
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
    user_profile = (profile + (f"\n\n{profile_suffix}" if (profile_suffix or "").strip() else "")).strip()
    predictions = []
    current_history = list(history_actions)

    hw = max(1, int(getattr(cfg, "ACTION_PREDICTION_HISTORY_WINDOW", 5)))
    n = len(target_actions)
    for i, target in enumerate(target_actions):
        recent = current_history[-hw:] if current_history else []
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

        current_history.append(target)

    return predictions


