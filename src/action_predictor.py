#!/usr/bin/env python3
"""
Action predictor: based on action_generation_model,
given persona + historical actions, predict action type and content for each action in target window.
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

# For computing semantic similarity
_SEMANTIC_SCORER = None
_SEMANTIC_SCORER_LOCK = threading.Lock()


def _get_semantic_scorer():
    """Lazy load semantic similarity calculator (thread-safe)"""
    global _SEMANTIC_SCORER
    if _SEMANTIC_SCORER is None:
        with _SEMANTIC_SCORER_LOCK:
            if _SEMANTIC_SCORER is None:  # Double-check
                try:
                    from src.scorer import SemanticScorer
                    _SEMANTIC_SCORER = SemanticScorer(device="cpu")
                except Exception as e:
                    print(f"[Warning] Failed to load SemanticScorer: {e}")
                    _SEMANTIC_SCORER = False  # Mark as failed to avoid retry
    return _SEMANTIC_SCORER if _SEMANTIC_SCORER is not False else None


def compute_semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two text segments (cosine similarity, range [-1, 1])

    Returns:
        Similarity score, returns float('nan') if computation fails
    """
    if not text1 or not text2:
        return float('nan')

    scorer = _get_semantic_scorer()
    if scorer is None:
        return float('nan')

    try:
        similarity = scorer.cosine_similarity(text1, text2)
        return float(similarity)
    except Exception as e:
        print(f"[Warning] Semantic similarity computation failed: {e}")
        return float('nan')
        return -1.0

_ACTION_API_RR_LOCK = threading.Lock()
_ACTION_API_RR_IDX = 0


def _pick_action_api_base() -> str:
    """Round-robin select one endpoint from effective_action_api_bases() (thread-safe)."""
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
# Action formatting (consistent with sft_data_generator)
# ============================================================================

def format_action(a: Dict, *, include_timestamp: bool = True) -> str:
    """
    Format single action into readable string for constructing history.

    When include_timestamp=False, remove `[timestamp]` at line start, used in action prediction prompt's "recent actions" block to save tokens;
    persona generation etc. can use default True.
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
    """Truncate long history head and tail (shared logic with persona; lazy import at runtime to prevent circular import with profile_generator)."""
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
# Construct prompt (decision / content)
# ============================================================================

def build_decision_scenario(target_action: Dict) -> str:
    """Construct decision scenario description based on target action context. reply must include original text of replied object (target)."""
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
    """Construct content generation scenario description based on target action. reply must include full text of replied object (truncated at TEXT_LONG)."""
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


def build_content_scenario_for_discrepancy(target_action: Dict) -> str:
    """Construct simplified content scenario for discrepancy record (remove redundant information)."""
    action_type = target_action.get("action_type", "")
    target = (target_action.get("target") or "").strip()

    if action_type == "post":
        return "Generate post content"
    if action_type == "reply":
        orig = target[:TEXT_LONG] if target else "(original post missing)"
        return f"Generate reply to: \"{orig}\""
    return "Generate content"


def build_decision_scenario_for_discrepancy(target_action: Dict) -> str:
    """Construct simplified decision scenario for discrepancy record (remove redundant context repetition)."""
    action_type = target_action.get("action_type", "")

    if action_type == "post":
        return "Predict action type for post scenario"
    if action_type == "reply":
        return "Predict action type for reply scenario"
    if action_type == "repost":
        return "Predict action type for repost scenario"
    if action_type == "like":
        return "Predict action type for like scenario"
    return "Predict action type"


def build_decision_prompt(
    user_profile: str, history: List[Dict], target_action: Dict
) -> Tuple[str, str]:
    """Return (instruction, input_text) for decision prediction. history is recent actual action sequence before current step (already formatted into prompt)."""
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
    """Return (instruction, input_text) for content prediction. history is recent actual action sequence before current step (already formatted into prompt)."""
    scenario = build_content_scenario(target_action)
    action_history = _format_action_history_block(history)
    input_text = CONTENT_INPUT_TEMPLATE.format(
        user_profile=user_profile,
        action_history=action_history,
        scenario=scenario,
    )
    return CONTENT_INSTRUCTION, input_text


# ============================================================================
# Model inference wrapper
# ============================================================================

def _trunc_for_debug(text: str, max_chars: int) -> str:
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [truncated, total {len(text)} characters]"


def _head_tail_for_debug(text: str, head: int, tail: int) -> str:
    """For long text, only print head + tail with middle omitted, convenient for viewing structure without flooding output."""
    if not text:
        return ""
    n = len(text)
    if n <= head + tail + 80:
        return text
    mid = n - head - tail
    return (
        text[:head]
        + f"\n\n... [omitted middle {mid} characters] ...\n\n"
        + text[-tail:]
    )


def _shorten_action_debug_user_text(text: str) -> str:
    """
    In action prediction, user text often contains very long persona: in debug mode, only keep persona head/tail + complete scenario section.
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
                f"(debug summary: original persona {len(profile)} characters, head/tail selected; scenario and below kept complete)"
            )
        except ValueError:
            pass
    return _trunc_for_debug(text, cfg.DEBUG_LLM_MAX_USER_CHARS)


def _format_debug_user_profile_mode(focus: Dict[str, Any]) -> str:
    """Persona request: highlight behavior history or (during refinement) discrepancy block."""
    ftype = focus.get("type")
    if ftype == "profile_initial":
        bd = focus.get("behavior_data") or ""
        rc = focus.get("record_count", "?")
        snippet = _head_tail_for_debug(
            bd, cfg.DEBUG_LLM_BEHAVIOR_HEAD, cfg.DEBUG_LLM_BEHAVIOR_TAIL
        )
        return (
            f"[Persona · Initial S0] Behavior record count: {rc}\n"
            f"[Focus: behavior history excerpt — head/tail, total {len(bd)} characters]\n{snippet}"
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
            "[Persona · Refinement] \n"
            f"[Focus: behavior prediction discrepancy (predicted vs actual), total {len(disc)} characters — show as complete as possible]\n"
            f"{disc_show}\n\n"
            f"[Original persona (excerpt: head+tail) total {len(op)} characters]\n{op_snip}"
        )
    return ""


def _format_debug_model_output(
    step: str,
    model_output: str,
    debug_focus: Optional[Dict[str, Any]],
) -> str:
    """Persona long output uses head/tail; action uses config to decide whether to show full length."""
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
    In DEBUG mode, print LLM calls. Persona type highlights discrepancy/behavior excerpt via debug_focus, long persona output uses head+tail.
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
        + f"\n[LLM-DEBUG] Step: {step}"
        + f"\n[LLM-DEBUG] Model type (model_role): {model_role}"
        + f"\n[LLM-DEBUG] Model ID (model_id): {model_id}"
        + f"\n[LLM-DEBUG] Backend: {backend}"
        + "\n" + "-" * 72
        + f"\n[system / instruction]\n{mi}"
        + "\n" + "-" * 72
        + f"\n[user / input — debug summary]\n{mu}"
        + "\n" + "-" * 72
        + f"\n[assistant / model output — long text excerpted per rules]\n{out}"
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
    Call local LLM (transformers format), return generated text.
    Use Llama-3 chat template: <|begin_of_text|><|start_header_id|>system<|end_header_id|>...<|eot_id|>
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
    Call model via OpenAI compatible API, return generated text.
    Applicable to vLLM / any OpenAI compatible service.

    max_context_tokens: Remote window limit; when None use config.ACTION_API_MAX_CONTEXT_TOKENS.
    Estimates prompt length by character count before request and shrinks max_tokens to avoid "max_tokens > context−input" 400 error.
    """

    def _estimate_prompt_tokens(instr: str, user: str) -> int:
        total = len(instr or "") + len(user or "")
        cpte = float(getattr(cfg, "ACTION_API_CHARS_PER_TOKEN_ESTIMATE", 3.0) or 3.0)
        cpte = max(1.5, min(cpte, 8.0))
        return max(1, int(total / cpte))

    from openai import APIError, BadRequestError, OpenAI

    def _parse_allowed_max_tokens_from_error(msg: str) -> Optional[int]:
        """
        Parse "how many completion tokens can current request still have" from common context limit errors.
        Typical format:
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
        # Reserve small safety buffer to avoid boundary jitter triggering 400 again
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
            f"[LLM-API] max_tokens pre-judged shrink: {wanted} -> {max_tokens_req} "
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
            # Only retry once for "max_tokens too large" type issues with dynamic shrinking
            if attempt == 0 and allowed is not None and allowed < max_tokens_req:
                print(
                    f"[LLM-API] max_tokens dynamic shrink: {max_tokens_req} -> {allowed} "
                    f"(model={model_name})",
                    flush=True,
                )
                max_tokens_req = max(1, allowed)
                continue
            raise
    else:
        # Theoretically should not reach here; safety fallback
        if last_err is not None:
            raise last_err
        raise RuntimeError("LLM API call failed: unknown error")

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
    """Unified dispatch: automatically select invocation method based on TEST_MODE / vLLM / local."""
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
            model_role="test_remote_api(action+profile shared)",
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
    """Extract action type from model output."""
    raw = raw_output.strip().lower()
    for action in ["post", "reply", "repost", "like", "not interested"]:
        if action in raw:
            return action
    return "not interested"


# ============================================================================
# Window-level action prediction
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
    Predict each action in target window.

    Args:
        use_parallel: Whether to use parallel prediction (default read from config.ACTION_PREDICTION_PARALLEL)
        workers: Number of parallel threads (default read from config.ACTION_PREDICTION_WORKERS)

    Note:
    - Parallel mode: all actions use same history_actions, can predict in parallel
    - Serial mode (legacy): use sliding history, actions have dependencies

    profile_suffix: Optional, append after persona text (e.g. explicit recent/full behavior block) for comparison experiments S0+history, full history, etc.
    include_observed_history: When False, don't append profile_suffix, and Recent user actions block is empty placeholder;
        When None, read config.ACTION_PROMPT_INCLUDE_OBSERVED_HISTORY.

    Return prediction list: [{"action_type": str, "content": str|None}, ...]
    """
    # Read default values from config
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

    # Parallel mode: all actions use same history window
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

    # Serial mode (legacy): use sliding history
    user_profile = (profile + (f"\n\n{eff_suffix}" if (eff_suffix or "").strip() else "")).strip()
    predictions = []
    current_history = list(eff_history_src)

    hw = max(1, int(getattr(cfg, "ACTION_PREDICTION_HISTORY_WINDOW", 5)))
    n = len(target_actions)
    for i, target in enumerate(target_actions):
        actual_type = target.get("action_type", "")  # Get actual type
        recent = current_history[-hw:] if current_history and _ih else []

        # Decision prediction
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

        # Content prediction (judge by actual type, not predicted type)
        pred_content = None
        if actual_type in ("post", "reply"):  # Key change: use actual_type
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

        # Add actual action to history (simulate time progression)
        current_history.append(target)

    return predictions


# ============================================================================
# Discrepancy signal generation (for persona refinement)
# ============================================================================

def build_behavior_discrepancies(
    predictions: List[Dict],
    actuals: List[Dict],
    history_actions: List[Dict],
) -> str:
    """
    Compare predictions vs actual, construct discrepancy signal text for persona refinement prompt use.

    Also compute and save semantic similarity to predictions (avoid redundant computation later).
    """
    parts = []
    seen_signatures = set()

    def _normalized_scenario_for_dedup(s: str) -> str:
        """
        When deduplicating, ignore error label in first line (e.g. [TEXT_GENERATION]/[DECISION_ONLY]),
        only compare subsequent core content to avoid "identical content but different label" duplicate output.
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
        pred_content = pred.get("content") or ""

        type_diff = pred_type != actual_type

        # ============================================
        # Discrepancy 1: TEXT_GENERATION (content generation discrepancy)
        # Only focus on content quality, not include type information
        # ============================================
        if actual_type in ("post", "reply"):
            # Use simplified scenario construction (remove redundant information)
            content_scenario = build_content_scenario_for_discrepancy(actual)

            # Only show content, no type prefix
            predicted_content = pred_content[:200] if pred_content else "(empty)"

            # Construct actual content
            if actual_type == "reply":
                actual_content = (actual.get("action_text") or "")[:200]
                # reply scenario description already includes replied object, no need for object_block
                object_block = ""
            else:  # post
                actual_content = actual_text[:200]
                object_block = ""

            actual_content = actual_content if actual_content else "(empty)"

            # Compute semantic similarity (use full text, no truncation)
            # If already computed, use directly
            if "semantic_similarity" not in pred:
                full_pred_content = pred_content if pred_content else ""
                # For post type, prefer action_text, then target
                # For reply type, action_text is reply content, target is replied object
                if actual_type == "reply":
                    full_actual_content = actual.get("action_text") or ""
                else:  # post
                    full_actual_content = actual.get("action_text") or actual.get("target") or ""

                # Only compute similarity when both texts are non-empty
                if full_pred_content and full_actual_content:
                    similarity = compute_semantic_similarity(full_pred_content, full_actual_content)
                else:
                    similarity = float('nan')  # Mark as uncomputable

                # Save to predictions to avoid redundant computation later
                pred["semantic_similarity"] = similarity
            else:
                similarity = pred["semantic_similarity"]

            # Add similarity info to scenario description
            # Use math.isnan() to check if valid value
            import math
            if not math.isnan(similarity):
                content_scenario_with_sim = f"{content_scenario} [Similarity: {similarity:.3f}]"
            else:
                content_scenario_with_sim = content_scenario

            # Record discrepancy
            _append_unique(
                scenario_context=f"[TEXT_GENERATION] {content_scenario_with_sim}",
                object_block=object_block,
                predicted_action=f'"{predicted_content}"',  # Content only
                actual_action=f'"{actual_content}"',        # Content only
            )

        # ============================================
        # Discrepancy 2: DECISION_ONLY (decision discrepancy)
        # Only focus on type judgment, not include content information
        # ============================================
        if type_diff:
            # Use simplified scenario construction (remove redundant context repetition)
            decision_scenario = build_decision_scenario_for_discrepancy(actual)

            # Only show type, no content
            predicted_type = pred_type
            actual_type_str = actual_type

            # Simplified context information (only provide when necessary)
            object_block = ""
            if actual_type == "reply":
                replied_to = (actual.get("target") or "")[:200]
                if replied_to:
                    object_block = f'Reply context: "{replied_to}"\n'
            elif actual_type in ("post", "like", "repost"):
                # Other types provide brief context
                context = (actual.get("target") or actual.get("action_text") or "")[:150]
                if context:
                    object_block = f'Content: "{context}"\n'

            # Record discrepancy
            _append_unique(
                scenario_context=f"[DECISION_ONLY] {decision_scenario}",
                object_block=object_block,
                predicted_action=predicted_type,  # Type only
                actual_action=actual_type_str,    # Type only
            )

    if not parts:
        return "No significant discrepancies detected."
    return "\n".join(parts)
