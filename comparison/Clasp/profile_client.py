#!/usr/bin/env python3
"""
DPO 微调后画像：OpenAI 兼容 API 调用，与主项目 `src.config` 中
PROFILE_API_BASE / PROFILE_API_MODEL 对齐；可在构造时显式覆盖，避免改全局 config。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.action_predictor import call_llm_api
from src.config import (
    MAX_NEW_TOKENS_PROFILE,
    TEMPERATURE_PROFILE,
    PROFILE_API_BASE,
    PROFILE_API_MODEL,
)
from src.profile_generator import format_behavior_data
from src.prompts import FREE_FORM_PROMPT, SYSTEM_INSTRUCTION_PROFILE


@dataclass
class ProfileServiceConfig:
    """画像生成服务端点；默认从 src.config 读取。"""

    api_base: str
    model_name: str
    max_new_tokens: int = MAX_NEW_TOKENS_PROFILE
    temperature: float = 0.7


def load_default_profile_config() -> ProfileServiceConfig:
    return ProfileServiceConfig(
        api_base=PROFILE_API_BASE,
        model_name=PROFILE_API_MODEL,
        max_new_tokens=MAX_NEW_TOKENS_PROFILE,
        temperature=0.7,
    )


class FinetunedProfileClient:
    """
    用微调后的画像服务，从**单窗口动作序列**生成用户画像 S。
    与 `generate_initial_profile` 的 prompt/格式一致，仅 api_base/model 可单独指定。
    """

    def __init__(self, cfg: Optional[ProfileServiceConfig] = None):
        self.cfg = cfg or load_default_profile_config()

    def generate_from_window_actions(
        self,
        actions: List[Dict[str, Any]],
        *,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        if not actions:
            return ""
        behavior_data = format_behavior_data(actions)
        prompt = FREE_FORM_PROMPT.format(
            action_count=len(actions),
            behavior_data=behavior_data,
        )
        m = max_new_tokens if max_new_tokens is not None else self.cfg.max_new_tokens
        t = temperature if temperature is not None else self.cfg.temperature
        return call_llm_api(
            self.cfg.api_base,
            self.cfg.model_name,
            SYSTEM_INSTRUCTION_PROFILE,
            prompt,
            max_new_tokens=int(m),
            temperature=float(t),
            debug_step="comparison:finetuned_profile_s",
            model_role="comparison_finetuned_profile",
            debug_emit=False,
        )
