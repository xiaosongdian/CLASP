#!/usr/bin/env python3
"""
画像生成器
- 初始画像生成：基于 W0 动作序列 + profile_generation_model_raw
- 画像精炼：基于预测偏差生成 N 个候选画像
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import src.config as cfg
from src.config import (
    COMMERCIAL_PROFILE_RATIO,
    DPO_WORKERS,
    ENABLE_COMMERCIAL_PROFILE,
    MAX_NEW_TOKENS_PROFILE,
    NUM_CANDIDATE_PROFILES,
    OPENAI_API_KEY,
    OPENAI_BASE_URL,
    PROFILE_API_BASE,
    PROFILE_API_MODEL,
    PROFILE_MODEL,
    TEMPERATURE_PROFILE,
    TEST_API_BASE,
    TEST_API_KEY,
    TEST_API_MODEL,
    TEST_MODE,
    TEST_NUM_CANDIDATES,
    USE_VLLM_API,
)
from src.prompts import (
    FREE_FORM_PROMPT,
    PROFILE_REFINEMENT_PROMPT,
    SYSTEM_INSTRUCTION_PROFILE,
    SYSTEM_INSTRUCTION_REFINEMENT,
)
from src.action_predictor import call_llm, call_llm_api, format_action


def _invoke_profile_llm(
    model,
    tokenizer,
    instruction: str,
    input_text: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    *,
    debug_step: str = "profile",
    debug_focus: Optional[Dict[str, Any]] = None,
    debug_emit: bool = False,
) -> str:
    """统一调度：根据 TEST_MODE / vLLM / 本地 自动选择调用方式。debug_emit 为 True 时打印 LLM-DEBUG（仅画像精炼等需观察时打开）。"""
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
            model_role="test_remote_api_profile",
            debug_focus=debug_focus,
            debug_emit=debug_emit,
        )
    if USE_VLLM_API or model is None or tokenizer is None:
        return call_llm_api(
            PROFILE_API_BASE,
            PROFILE_API_MODEL,
            instruction,
            input_text,
            max_new_tokens,
            temperature,
            debug_step=debug_step,
            model_role="vllm_user_profile",
            debug_focus=debug_focus,
            debug_emit=debug_emit,
        )
    return call_llm(
        model,
        tokenizer,
        instruction,
        input_text,
        max_new_tokens,
        temperature,
        debug_step=debug_step,
        debug_focus=debug_focus,
        debug_emit=debug_emit,
    )


def format_behavior_data(actions: List[Dict]) -> str:
    """将动作列表格式化为画像生成所需的行为数据字符串。"""
    return "\n".join(format_action(a) for a in actions)


def generate_initial_profile(
    model,
    tokenizer,
    actions: List[Dict],
    max_new_tokens: int = MAX_NEW_TOKENS_PROFILE,
) -> str:
    """
    使用 W0 窗口动作生成初始用户画像 S0。
    """
    behavior_data = format_behavior_data(actions)
    prompt = FREE_FORM_PROMPT.format(
        action_count=len(actions),
        behavior_data=behavior_data,
    )
    return _invoke_profile_llm(
        model,
        tokenizer,
        SYSTEM_INSTRUCTION_PROFILE,
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        debug_step="user_profile:initial_s0",
        debug_emit=False,
        debug_focus={
            "type": "profile_initial",
            "behavior_data": behavior_data,
            "record_count": len(actions),
        },
    )


def _invoke_commercial_profile_llm(
    instruction: str,
    input_text: str,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    *,
    debug_step: str = "profile_commercial",
    debug_focus: Optional[Dict[str, Any]] = None,
    debug_emit: bool = False,
) -> str:
    """调用商用画像模型（OpenAI 兼容 API）。"""
    return call_llm_api(
        OPENAI_BASE_URL,
        PROFILE_MODEL,
        instruction,
        input_text,
        max_new_tokens,
        temperature,
        api_key=OPENAI_API_KEY,
        debug_step=debug_step,
        model_role="commercial_profile_api",
        debug_focus=debug_focus,
        debug_emit=debug_emit,
    )


def generate_candidate_profiles(
    model,
    tokenizer,
    old_profile: str,
    behavior_discrepancies: str,
    n: int = NUM_CANDIDATE_PROFILES,
    max_new_tokens: int = MAX_NEW_TOKENS_PROFILE,
    temperature: float = TEMPERATURE_PROFILE,
    workers: int = DPO_WORKERS,
) -> List[str]:
    """
    基于预测偏差对原始画像进行精炼，生成 N 个候选画像。
    通过调高 temperature 获得多样性。
    """
    if TEST_MODE:
        n = TEST_NUM_CANDIDATES

    refinement_prompt = PROFILE_REFINEMENT_PROMPT.format(
        old_persona=old_profile,
        behavior_discrepancies=behavior_discrepancies,
    )

    effective_ratio = float(COMMERCIAL_PROFILE_RATIO) if ENABLE_COMMERCIAL_PROFILE else 0.0
    n_commercial = int(round(n * effective_ratio))
    n_commercial = max(0, min(n, n_commercial))
    n_base = n - n_commercial
    if cfg.DEBUG_LLM:
        print(
            f"  [ProfileGen] 候选画像来源分配：base={n_base}, commercial={n_commercial} "
            f"(ratio={effective_ratio}, enabled={ENABLE_COMMERCIAL_PROFILE})",
            flush=True,
        )

    debug_focus = {
        "type": "profile_refine",
        "discrepancies": behavior_discrepancies,
        "old_profile": old_profile,
    }

    def _gen_one(i: int) -> tuple[int, str]:
        use_commercial = i >= n_base
        provider = "commercial" if use_commercial else "base"
        if cfg.DEBUG_LLM:
            print(f"  [ProfileGen] 生成候选画像 {i+1}/{n} (source={provider}) ...", flush=True)
        if use_commercial:
            profile = _invoke_commercial_profile_llm(
                SYSTEM_INSTRUCTION_REFINEMENT,
                refinement_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                debug_step=f"user_profile:refine_candidate_{i + 1}_of_{n}:commercial",
                debug_emit=True,
                debug_focus=debug_focus,
            )
        else:
            profile = _invoke_profile_llm(
                model,
                tokenizer,
                SYSTEM_INSTRUCTION_REFINEMENT,
                refinement_prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                debug_step=f"user_profile:refine_candidate_{i + 1}_of_{n}:base",
                debug_emit=True,
                debug_focus=debug_focus,
            )
        return i, profile

    workers = max(1, min(int(workers), n))
    results: List[Optional[str]] = [None] * n
    if workers == 1:
        for i in range(n):
            idx, prof = _gen_one(i)
            results[idx] = prof
    else:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="candgen") as pool:
            futures = {pool.submit(_gen_one, i): i for i in range(n)}
            for fut in as_completed(futures):
                idx, prof = fut.result()
                results[idx] = prof
    return [r or "" for r in results]
