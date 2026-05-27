#!/usr/bin/env python3
"""
DPO ：OpenAI  API ， `src.config` 
PROFILE_API_BASE / PROFILE_API_MODEL ；， config。
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
    """； src.config 。"""

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
    ，**** S。
     `generate_initial_profile`  prompt/， api_base/model 。
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
