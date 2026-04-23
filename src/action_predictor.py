#!/usr/bin/env python3
"""
动作预测器：基于 action_generation_model，
给定画像 + 历史动作，预测目标窗口内每条动作的类型和内容。
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import src.config as cfg
from src.config import (
    TEXT_LONG,
    USE_VLLM_API,
    TEST_MODE,
    TEST_API_BASE,
    TEST_API_KEY,
    TEST_API_MODEL,
    ACTION_API_BASE,
    ACTION_API_MODEL,
)
from src.prompts import (
    AVAILABLE_ACTIONS,
    CONTENT_INPUT_TEMPLATE,
    CONTENT_INSTRUCTION,
    DECISION_INPUT_TEMPLATE,
    DECISION_INSTRUCTION,
    DISCREPANCY_TEMPLATE,
)


# ============================================================================
# 动作格式化（与 sft_data_generator 保持一致）
# ============================================================================

def format_action(a: Dict) -> str:
    """将单条动作格式化为可读字符串，用于构造 history。"""
    ts = a.get("timestamp", "")
    action_type = a.get("action_type", "")
    target = a.get("target") or ""
    action_text = a.get("action_text") or ""

    if action_type == "reply":
        orig = target[:TEXT_LONG] if target else ""
        content = action_text[:TEXT_LONG]
        if orig:
            return f'[{ts}] Reply to original: "{orig}" | reply text: "{content}"'
        return f'[{ts}] User commented (context unknown): "{content}"'
    elif action_type == "post":
        content = action_text[:TEXT_LONG]
        return f'[{ts}] User posted: "{content}"'
    elif action_type == "like":
        return f'[{ts}] User liked: "{target[:TEXT_LONG]}..."'
    elif action_type == "repost":
        return f'[{ts}] User reposted: "{target[:TEXT_LONG]}..."'
    else:
        return f'[{ts}] User performed {action_type} on: "{target[:TEXT_LONG]}..."'


def format_history(actions: List[Dict]) -> str:
    return "\n".join(format_action(a) for a in actions)


# ============================================================================
# 构造 prompt（决策 / 内容）
# ============================================================================

def build_decision_scenario(target_action: Dict) -> str:
    """根据目标动作的上下文构造决策场景描述。reply 须包含被回复对象原文（target）。"""
    action_type = target_action.get("action_type", "")
    target = (target_action.get("target") or "").strip()
    action_text = (target_action.get("action_text") or "").strip()

    if action_type == "post":
        context = (action_text or target)[:TEXT_LONG]
        return (
            f"Draft/surface text associated with the next action (e.g. post body):\n\"{context}\"\n"
            f"Which type of user behavior is this most likely generated from?"
        )
    if action_type == "reply":
        orig = target[:TEXT_LONG] if target else (action_text[:TEXT_LONG] if action_text else "(unknown)")
        return (
            "The user is about to send a **reply**. The **original post or comment they are replying to** is:\n"
            f"\"{orig}\"\n"
            "(The reply text the user will publish is predicted separately; use the quoted text as the object of the reply.)\n"
            "Which type of user behavior is this most likely: reply, or a different action?"
        )

    # like / repost / others
    context = (target or action_text)[:TEXT_LONG]
    return (
        f"Context of the next action (e.g. target post text):\n\"{context}\"\n"
        f"Which type of user behavior is this most likely generated from?"
    )


def build_content_scenario(target_action: Dict) -> str:
    """根据目标动作构造内容生成场景描述。reply 必须带被回复对象全文（截断在 TEXT_LONG）。"""
    action_type = target_action.get("action_type", "")
    target = (target_action.get("target") or "").strip()

    if action_type == "post":
        return "Write a post, what content?"
    if action_type == "reply":
        orig = target[:TEXT_LONG] if target else "(original post to reply to is missing in data — infer briefly)"
        return (
            "Output **only** the text of the user's reply (no meta-commentary).\n"
            "Original post/comment the user is replying to (must respond to this):\n"
            f"\"{orig}\"\n"
            "What is the reply text?"
        )
    return "What content to generate?"


def build_decision_prompt(
    user_profile: str, history: List[Dict], target_action: Dict
) -> Tuple[str, str]:
    """返回 (instruction, input_text) 用于决策预测。"""
    scenario = build_decision_scenario(target_action)
    input_text = DECISION_INPUT_TEMPLATE.format(
        user_profile=user_profile,
        scenario=scenario,
        available_actions=AVAILABLE_ACTIONS,
    )
    return DECISION_INSTRUCTION, input_text


def build_content_prompt(
    user_profile: str, history: List[Dict], target_action: Dict
) -> Tuple[str, str]:
    """返回 (instruction, input_text) 用于内容预测。"""
    scenario = build_content_scenario(target_action)
    input_text = CONTENT_INPUT_TEMPLATE.format(
        user_profile=user_profile,
        scenario=scenario,
    )
    return CONTENT_INSTRUCTION, input_text


# ============================================================================
# 模型推理封装
# ============================================================================

def _trunc_for_debug(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [截断，共 {len(text)} 字符]"


def _head_tail_for_debug(text: str, head: int, tail: int) -> str:
    """长文本只打印头+尾，中间省略，便于看结构又不刷屏。"""
    if not text:
        return ""
    n = len(text)
    if n <= head + tail + 80:
        return text
    mid = n - head - tail
    return (
        text[:head]
        + f"\n\n... [已省略中间 {mid} 字符] ...\n\n"
        + text[-tail:]
    )


def _shorten_action_debug_user_text(text: str) -> str:
    """
    动作预测里 user 往往带超长画像：调试时只保留画像头尾+完整 scenario 段。
    """
    if len(text) <= cfg.DEBUG_LLM_ACTION_USER_MAX:
        return text
    marker = "Target user profile:"
    sc = "Current scenario:"
    if marker in text and sc in text:
        try:
            a = text.index(marker) + len(marker)
            b = text.index(sc, a)
            profile = text[a:b].strip()
            rest = text[b:]
            ph = _head_tail_for_debug(profile, 1000, 600)
            return (
                f"{marker}\n{ph}\n\n{rest}\n"
                f"(调试摘要: 原画像 {len(profile)} 字符，已做头尾节选；scenario 及以下完整保留)"
            )
        except ValueError:
            pass
    return _trunc_for_debug(text, cfg.DEBUG_LLM_MAX_USER_CHARS)


def _format_debug_user_profile_mode(focus: Dict[str, Any]) -> str:
    """画像类请求：重点展示行为历史或（精炼时）误差块。"""
    ftype = focus.get("type")
    if ftype == "profile_initial":
        bd = focus.get("behavior_data") or ""
        rc = focus.get("record_count", "?")
        snippet = _head_tail_for_debug(
            bd, cfg.DEBUG_LLM_BEHAVIOR_HEAD, cfg.DEBUG_LLM_BEHAVIOR_TAIL
        )
        return (
            f"[画像 · 初始 S0] 行为记录数: {rc}\n"
            f"[重点: 行为历史节选 — 头/尾，总长 {len(bd)} 字符]\n{snippet}"
        )
    if ftype == "profile_refine":
        disc = (focus.get("discrepancies") or "").strip()
        if len(disc) > int(cfg.DEBUG_LLM_DISCREPANCY_MAX):
            disc_show = _trunc_for_debug(disc, int(cfg.DEBUG_LLM_DISCREPANCY_MAX))
        else:
            disc_show = disc
        op = focus.get("old_profile") or ""
        op_snip = _head_tail_for_debug(
            op, cfg.DEBUG_LLM_OLD_PERSONA_HEAD, cfg.DEBUG_LLM_OLD_PERSONA_TAIL
        )
        return (
            "[画像 · 精炼] \n"
            f"[重点: 行为预测误差 (predicted vs actual)，总长 {len(disc)} 字符 — 尽量完整展示]\n"
            f"{disc_show}\n\n"
            f"[原画像（节选: 头+尾）总长 {len(op)} 字符]\n{op_snip}"
        )
    return ""


def _format_debug_model_output(
    step: str,
    model_output: str,
    debug_focus: Optional[Dict[str, Any]],
) -> str:
    """画像长输出用头尾；动作用配置决定是否全长。"""
    if debug_focus and debug_focus.get("type") in ("profile_initial", "profile_refine"):
        return _head_tail_for_debug(
            model_output,
            cfg.DEBUG_LLM_PROFILE_OUTPUT_HEAD,
            cfg.DEBUG_LLM_PROFILE_OUTPUT_TAIL,
        )
    if not getattr(cfg, "DEBUG_LLM_PRINT_FULL_OUTPUT", True):
        return _trunc_for_debug(model_output, 4000)
    if step.startswith("action_prediction") and len(model_output) > 5000:
        return _head_tail_for_debug(model_output, 2000, 1500)
    return model_output


def emit_llm_debug(
    step: str,
    model_role: str,
    model_id: str,
    backend: str,
    instruction: str,
    user_content: str,
    model_output: str,
    *,
    debug_focus: Optional[Dict[str, Any]] = None,
) -> None:
    """
    DEBUG 模式下打印 LLM 调用。画像类通过 debug_focus 重点打印误差/行为节选，长画像输出用头+尾。
    """
    if not cfg.DEBUG_LLM:
        return
    mi = _trunc_for_debug(instruction, cfg.DEBUG_LLM_MAX_INSTRUCTION_CHARS)

    if debug_focus and debug_focus.get("type") in ("profile_initial", "profile_refine"):
        mu = _format_debug_user_profile_mode(debug_focus)
    elif step.startswith("action_prediction") and len(user_content) > cfg.DEBUG_LLM_ACTION_USER_MAX:
        mu = _shorten_action_debug_user_text(user_content)
    else:
        mu = _trunc_for_debug(user_content, cfg.DEBUG_LLM_MAX_USER_CHARS)

    out = _format_debug_model_output(step, model_output, debug_focus)

    print(
        "\n" + "=" * 72
        + f"\n[LLM-DEBUG] 步骤: {step}"
        + f"\n[LLM-DEBUG] 模型类型(model_role): {model_role}"
        + f"\n[LLM-DEBUG] 模型标识(model_id): {model_id}"
        + f"\n[LLM-DEBUG] 后端(backend): {backend}"
        + "\n" + "-" * 72
        + f"\n[system / instruction]\n{mi}"
        + "\n" + "-" * 72
        + f"\n[user / input — 调试摘要]\n{mu}"
        + "\n" + "-" * 72
        + f"\n[assistant / 模型输出 — 长文本已按规则节选]\n{out}"
        + "\n" + "=" * 72 + "\n",
        flush=True,
    )


def call_llm(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    *,
    debug_step: str = "local_llm",
    debug_focus: Optional[Dict[str, Any]] = None,
    debug_emit: bool = True,
) -> str:
    """
    调用本地 LLM（transformers 格式），返回生成文本。
    使用 Llama-3 chat template: <|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|eot_id|>
    """
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": input_text},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    import torch
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if temperature > 0 else 1.0,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    out = tokenizer.decode(generated, skip_special_tokens=True).strip()
    mid = getattr(tokenizer, "name_or_path", None) or str(type(model).__name__)
    if debug_emit:
        emit_llm_debug(
            debug_step,
            "local_transformers",
            str(mid),
            "HuggingFace transformers + generate()",
            instruction,
            input_text,
            out,
            debug_focus=debug_focus,
        )
    return out


def call_llm_api(
    api_base: str,
    model_name: str,
    instruction: str,
    input_text: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    api_key: str = "not-needed",
    *,
    debug_step: str = "openai_api",
    model_role: str = "openai_compatible",
    debug_focus: Optional[Dict[str, Any]] = None,
    debug_emit: bool = True,
) -> str:
    """
    通过 OpenAI 兼容 API 调用模型，返回生成文本。
    适用于 vLLM / 任意 OpenAI 兼容服务。
    """
    from openai import OpenAI

    client = OpenAI(base_url=api_base, api_key=api_key)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": instruction},
            {"role": "user", "content": input_text},
        ],
        max_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else 0.01,
    )
    out = response.choices[0].message.content.strip()
    if debug_emit:
        emit_llm_debug(
            debug_step,
            model_role,
            model_name,
            f"OpenAI-compatible API | base={api_base}",
            instruction,
            input_text,
            out,
            debug_focus=debug_focus,
        )
    return out


def invoke_action_llm(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int = 512,
    temperature: float = 0.3,
    *,
    debug_step: str = "action",
) -> str:
    """统一调度：根据 TEST_MODE / vLLM / 本地 自动选择调用方式。"""
    action_debug = bool(
        cfg.DEBUG_LLM and getattr(cfg, "DEBUG_LLM_INCLUDE_ACTIONS", False)
    )
    if TEST_MODE:
        return call_llm_api(
            TEST_API_BASE,
            TEST_API_MODEL,
            instruction,
            input_text,
            max_new_tokens,
            temperature,
            api_key=TEST_API_KEY,
            debug_step=debug_step,
            model_role="test_remote_api(action+profile共用)",
            debug_emit=action_debug,
        )
    if USE_VLLM_API or model is None or tokenizer is None:
        return call_llm_api(
            ACTION_API_BASE,
            ACTION_API_MODEL,
            instruction,
            input_text,
            max_new_tokens,
            temperature,
            debug_step=debug_step,
            model_role="vllm_action_prediction",
            debug_emit=action_debug,
        )
    return call_llm(
        model,
        tokenizer,
        instruction,
        input_text,
        max_new_tokens,
        temperature,
        debug_step=debug_step,
        debug_emit=action_debug,
    )


def parse_action_type(raw_output: str) -> str:
    """从模型输出中提取动作类型。"""
    raw = raw_output.strip().lower()
    for action in ["post", "reply", "repost", "like", "not interested"]:
        if action in raw:
            return action
    return "not interested"


# ============================================================================
# 窗口级动作预测
# ============================================================================

def predict_actions_for_window(
    model,
    tokenizer,
    profile: str,
    history_actions: List[Dict],
    target_actions: List[Dict],
    max_new_tokens_decision: int = 128,
    max_new_tokens_content: int = 512,
    temperature: float = 0.3,
) -> List[Dict]:
    """
    对目标窗口中每条动作进行预测。
    使用滑动历史：随着预测推进，前面的真实动作加入历史。

    返回预测列表：[{"action_type": str, "content": str|None}, ...]
    """
    predictions = []
    current_history = list(history_actions)

    n = len(target_actions)
    for i, target in enumerate(target_actions):
        # 决策预测
        inst, inp = build_decision_prompt(profile, current_history[-10:], target)
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
            inst_c, inp_c = build_content_prompt(profile, current_history[-10:], target)
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


# ============================================================================
# 偏差信号生成（用于画像精炼）
# ============================================================================

def build_behavior_discrepancies(
    predictions: List[Dict],
    actuals: List[Dict],
    history_actions: List[Dict],
) -> str:
    """
    对比预测与真实，构造偏差信号文本，供画像精炼 prompt 使用。
    """
    parts = []
    for i, (pred, actual) in enumerate(zip(predictions, actuals)):
        actual_type = actual.get("action_type", "unknown")
        actual_text = actual.get("action_text") or actual.get("target") or ""
        pred_type = pred.get("action_type", "unknown")
        pred_text = pred.get("content") or ""

        # 只记录存在偏差的场景
        type_diff = pred_type != actual_type
        text_diff = actual_type in ("post", "reply")

        if type_diff or text_diff:
            scenario_ctx = ""
            predicted = f"{pred_type}" + (f': "{pred_text[:200]}"' if pred_text else "")
            # reply：action_text 为用户回复；target 为被回复对象，须单独展示
            if actual_type == "reply":
                u_reply = (actual.get("action_text") or "")[:200]
                actual_str = f"{actual_type} (user reply: \"{u_reply}\")" if u_reply else f"{actual_type}"
            else:
                actual_str = f"{actual_type}" + (f': "{actual_text[:200]}"' if actual_text else "")

            object_block = ""
            if actual_type == "reply":
                replied_to = (actual.get("target") or "")[:500]
                if replied_to:
                    object_block = f'Replied-to original post/comment: "{replied_to}"\n'

            parts.append(DISCREPANCY_TEMPLATE.format(
                idx=len(parts) + 1,
                scenario_context=scenario_ctx,
                object_block=object_block,
                predicted_action=predicted,
                actual_action=actual_str,
            ))

    if not parts:
        return "No significant discrepancies detected."
    return "\n".join(parts)
