#!/usr/bin/env python3
"""
动作预测器：基于 action_generation_model，
给定画像 + 历史动作，预测目标窗口内每条动作的类型和内容。
"""

import re
import threading
from typing import Any, Dict, List, Optional, Tuple

import src.config as cfg
from src.config import (
    TEXT_LONG,
    USE_VLLM_API,
    TEST_MODE,
    TEST_API_BASE,
    TEST_API_KEY,
    TEST_API_MODEL,
)

_ACTION_API_RR_LOCK = threading.Lock()
_ACTION_API_RR_IDX = 0


def _pick_action_api_base() -> str:
    """从 effective_action_api_bases() 中轮询选一个端点（线程安全）。"""
    global _ACTION_API_RR_IDX
    bases = cfg.effective_action_api_bases()
    if not bases:
        return "http://127.0.0.1:8002/v1"
    if len(bases) == 1:
        return bases[0]
    with _ACTION_API_RR_LOCK:
        b = bases[_ACTION_API_RR_IDX % len(bases)]
        _ACTION_API_RR_IDX += 1
        return b
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

def format_action(a: Dict, *, include_timestamp: bool = True) -> str:
    """
    将单条动作格式化为可读字符串，用于构造 history。

    include_timestamp=False 时去掉行首 `[时间]`，用于动作预测 prompt 内的「近期行为」块以节省 token；
    画像生成等仍可默认 True。
    """
    ts = a.get("timestamp", "")
    action_type = a.get("action_type", "")
    target = a.get("target") or ""
    action_text = a.get("action_text") or ""
    lead = f"[{ts}] " if include_timestamp else ""

    if action_type == "reply":
        orig = target[:TEXT_LONG] if target else ""
        content = action_text[:TEXT_LONG]
        if orig:
            return f'{lead}Reply to original: "{orig}" | reply text: "{content}"'
        return f'{lead}User commented (context unknown): "{content}"'
    elif action_type == "post":
        content = action_text[:TEXT_LONG]
        return f'{lead}User posted: "{content}"'
    elif action_type == "like":
        return f'{lead}User liked: "{target[:TEXT_LONG]}..."'
    elif action_type == "repost":
        return f'{lead}User reposted: "{target[:TEXT_LONG]}..."'
    else:
        return f'{lead}User performed {action_type} on: "{target[:TEXT_LONG]}..."'


def format_history(actions: List[Dict], *, include_timestamp: bool = True) -> str:
    return "\n".join(format_action(a, include_timestamp=include_timestamp) for a in actions)


def _truncate_action_history_for_prompt(text: str) -> str:
    """超长历史头尾截断（与画像共用逻辑；运行时懒导入防与 profile_generator 循环引用）。"""
    mc = int(getattr(cfg, "ACTION_PROMPT_HISTORY_MAX_CHARS", 6000) or 0)
    from src.profile_generator import truncate_behavior_plaintext

    return truncate_behavior_plaintext(text, mc)


def _format_action_history_block(history: List[Dict]) -> str:
    if not history:
        return "(No recent actions in this window.)"
    return _truncate_action_history_for_prompt(
        format_history(history, include_timestamp=False)
    )


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
    """返回 (instruction, input_text) 用于决策预测。history 为当前步之前的近期真实动作序列（已格式化进 prompt）。"""
    scenario = build_decision_scenario(target_action)
    action_history = _format_action_history_block(history)
    input_text = DECISION_INPUT_TEMPLATE.format(
        user_profile=user_profile,
        action_history=action_history,
        scenario=scenario,
        available_actions=AVAILABLE_ACTIONS,
    )
    return DECISION_INSTRUCTION, input_text


def build_content_prompt(
    user_profile: str, history: List[Dict], target_action: Dict
) -> Tuple[str, str]:
    """返回 (instruction, input_text) 用于内容预测。history 为当前步之前的近期真实动作序列（已格式化进 prompt）。"""
    scenario = build_content_scenario(target_action)
    action_history = _format_action_history_block(history)
    input_text = CONTENT_INPUT_TEMPLATE.format(
        user_profile=user_profile,
        action_history=action_history,
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
    max_context_tokens: Optional[int] = None,
) -> str:
    """
    通过 OpenAI 兼容 API 调用模型，返回生成文本。
    适用于 vLLM / 任意 OpenAI 兼容服务。

    max_context_tokens: 远端窗口上限；None 时使用 config.ACTION_API_MAX_CONTEXT_TOKENS。
    会在请求前用字符粗估 prompt 长度并收缩 max_tokens，避免 「max_tokens 大于 context−input」 的 400。
    """

    def _estimate_prompt_tokens(instr: str, user: str) -> int:
        total = len(instr or "") + len(user or "")
        cpte = float(getattr(cfg, "ACTION_API_CHARS_PER_TOKEN_ESTIMATE", 3.0) or 3.0)
        cpte = max(1.5, min(cpte, 8.0))
        return max(1, int(total / cpte))

    from openai import APIError, BadRequestError, OpenAI

    def _parse_allowed_max_tokens_from_error(msg: str) -> Optional[int]:
        """
        从常见上下文超限报错中解析“当前请求最多还能给多少 completion tokens”。
        典型格式：
        "... maximum context length is 4096 tokens and your request has 2148 input tokens ..."
        """
        if not msg:
            return None
        m = re.search(
            r"maximum context length is\s+(\d+)\s+tokens.*?your request has\s+(\d+)\s+input tokens",
            msg,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not m:
            return None
        try:
            max_ctx = int(m.group(1))
            input_tokens = int(m.group(2))
        except (TypeError, ValueError):
            return None
        # 预留少量安全缓冲，避免边界抖动再次触发 400
        return max(1, max_ctx - input_tokens - 16)

    mc = (
        max_context_tokens
        if max_context_tokens is not None
        else int(getattr(cfg, "ACTION_API_MAX_CONTEXT_TOKENS", 8192))
    )
    margin = int(getattr(cfg, "ACTION_API_COMPLETION_SAFETY_MARGIN", 64))

    client = OpenAI(base_url=api_base, api_key=api_key)
    est_in = _estimate_prompt_tokens(instruction, input_text)
    allowed_completion = max(1, mc - est_in - margin)
    wanted = max(1, int(max_new_tokens))
    max_tokens_req = max(1, min(wanted, allowed_completion))
    if max_tokens_req < wanted and getattr(cfg, "DEBUG_LLM", False):
        print(
            f"[LLM-API] max_tokens 预判收缩: {wanted} -> {max_tokens_req} "
            f"(est_input≈{est_in}, context={mc}, model={model_name})",
            flush=True,
        )
    last_err: Optional[Exception] = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": input_text},
                ],
                max_tokens=max_tokens_req,
                temperature=temperature if temperature > 0 else 0.01,
            )
            break
        except (BadRequestError, APIError) as e:
            last_err = e
            err_msg = str(e)
            if getattr(e, "body", None) and isinstance(e.body, dict):
                err_msg = err_msg + " " + str(e.body)
            allowed = _parse_allowed_max_tokens_from_error(err_msg)
            # 仅对“max_tokens 过大”类问题做一次动态缩小重试
            if attempt == 0 and allowed is not None and allowed < max_tokens_req:
                print(
                    f"[LLM-API] max_tokens 动态收缩: {max_tokens_req} -> {allowed} "
                    f"(model={model_name})",
                    flush=True,
                )
                max_tokens_req = max(1, allowed)
                continue
            raise
    else:
        # 理论上不会走到这里；保险兜底
        if last_err is not None:
            raise last_err
        raise RuntimeError("LLM API 调用失败：未知错误")

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
            _pick_action_api_base(),
            cfg.ACTION_API_MODEL,
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
    profile_suffix: Optional[str] = None,
    use_parallel: Optional[bool] = None,
    workers: Optional[int] = None,
    include_observed_history: Optional[bool] = None,
) -> List[Dict]:
    """
    对目标窗口中每条动作进行预测。

    Args:
        use_parallel: 是否使用并行预测（默认从 config.ACTION_PREDICTION_PARALLEL 读取）
        workers: 并行线程数（默认从 config.ACTION_PREDICTION_WORKERS 读取）

    注意：
    - 并行模式：所有动作使用相同的 history_actions，可以并行预测
    - 串行模式（旧版）：使用滑动历史，动作之间有依赖

    profile_suffix: 可选，拼在画像文本后（如显式近期/全量行为块），供对比实验 S0+历史、全量历史等。
    include_observed_history: False 时不拼 profile_suffix，且 Recent user actions 块为空占位；
        None 时读 config.ACTION_PROMPT_INCLUDE_OBSERVED_HISTORY。

    返回预测列表：[{"action_type": str, "content": str|None}, ...]
    """
    # 从配置读取默认值
    if use_parallel is None:
        use_parallel = getattr(cfg, "ACTION_PREDICTION_PARALLEL", True)
    if workers is None:
        workers = getattr(cfg, "ACTION_PREDICTION_WORKERS", 10)

    _ih = (
        include_observed_history
        if include_observed_history is not None
        else bool(getattr(cfg, "ACTION_PROMPT_INCLUDE_OBSERVED_HISTORY", True))
    )
    eff_suffix = profile_suffix if _ih else None
    eff_history_src = history_actions if _ih else []

    # 并行模式：所有动作使用相同的历史窗口
    if use_parallel:
        from src.action_predictor_parallel import predict_actions_for_window_parallel
        return predict_actions_for_window_parallel(
            model,
            tokenizer,
            profile,
            eff_history_src,
            target_actions,
            max_new_tokens_decision,
            max_new_tokens_content,
            temperature,
            eff_suffix,
            workers,
        )

    # 串行模式（旧版）：使用滑动历史
    user_profile = (profile + (f"\n\n{eff_suffix}" if (eff_suffix or "").strip() else "")).strip()
    predictions = []
    current_history = list(eff_history_src)

    hw = max(1, int(getattr(cfg, "ACTION_PREDICTION_HISTORY_WINDOW", 5)))
    n = len(target_actions)
    for i, target in enumerate(target_actions):
        recent = current_history[-hw:] if current_history and _ih else []
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
    seen_signatures = set()

    def _normalized_scenario_for_dedup(s: str) -> str:
        """
        去重时忽略首行的误差标签（如 [TEXT_GENERATION]/[DECISION_ONLY]），
        只比较后续核心内容，避免“内容完全相同但标签不同”重复输出。
        """
        lines = [ln.strip() for ln in (s or "").splitlines() if ln.strip()]
        if lines and lines[0].startswith("[") and lines[0].endswith("]"):
            lines = lines[1:]
        return "\n".join(lines)

    def _append_unique(
        scenario_context: str,
        object_block: str,
        predicted_action: str,
        actual_action: str,
    ) -> None:
        sig = (
            _normalized_scenario_for_dedup(scenario_context),
            (object_block or "").strip(),
            (predicted_action or "").strip(),
            (actual_action or "").strip(),
        )
        if sig in seen_signatures:
            return
        seen_signatures.add(sig)
        parts.append(DISCREPANCY_TEMPLATE.format(
            idx=len(parts) + 1,
            scenario_context=scenario_context,
            object_block=object_block,
            predicted_action=predicted_action,
            actual_action=actual_action,
        ))
    for i, (pred, actual) in enumerate(zip(predictions, actuals)):
        actual_type = actual.get("action_type", "unknown")
        actual_text = actual.get("action_text") or actual.get("target") or ""
        pred_type = pred.get("action_type", "unknown")
        pred_text = pred.get("content") or ""

        # 一条动作可拆分为两类误差：
        # 1) TEXT_GENERATION：post/reply 的文本生成误差
        # 2) DECISION_ONLY：动作类型判定误差
        # 这样总条数不再固定受窗口长度限制（可 > len(window)）
        type_diff = pred_type != actual_type

        predicted = f"{pred_type}" + (f': "{pred_text[:200]}"' if pred_text else "")
        # reply：action_text 为用户回复；target 为被回复对象，须单独展示
        if actual_type == "reply":
            u_reply = (actual.get("action_text") or "")[:200]
            actual_str = f'{actual_type}: "{u_reply}"' if u_reply else f"{actual_type}"
        else:
            actual_str = f"{actual_type}" + (f': "{actual_text[:200]}"' if actual_text else "")

        object_block = ""
        if actual_type == "reply":
            replied_to = (actual.get("target") or "")[:500]
            if replied_to:
                object_block = f'Replied-to original post/comment: "{replied_to}"\n'

        # 文本生成类：post/reply 都记录（反映内容层信号）
        if actual_type in ("post", "reply"):
            scenario_ctx = (
                "[TEXT_GENERATION] \n"
                f"Decision type predicted: {pred_type}; actual type: {actual_type}"
            )
            _append_unique(
                scenario_context=scenario_ctx,
                object_block=object_block,
                predicted_action=predicted,
                actual_action=actual_str,
            )

        # 交互决策类：只在类型误判时记录（包括 post/reply）
        if type_diff:
            scenario_ctx = (
                "[DECISION_ONLY] \n"
                f"Decision type predicted: {pred_type}; actual type: {actual_type}"
            )
            _append_unique(
                scenario_context=scenario_ctx,
                object_block=object_block,
                predicted_action=predicted,
                actual_action=actual_str,
            )

    if not parts:
        return "No significant discrepancies detected."
    return "\n".join(parts)
