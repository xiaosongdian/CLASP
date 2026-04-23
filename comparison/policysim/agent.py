from dataclasses import dataclass, field
import json
import re
from typing import Any, Optional

from policysim.memory import AgentMemory
from policysim.prompt.agent_prompt import build_action_prompt
from policysim.types import ActionRecord
from policysim.llm.base import BaseChatModel, ChatMessage


@dataclass
class AgentProfile:
    user_id: str
    attributes: dict[str, Any] = field(default_factory=dict)
    tweets: list[str] = field(default_factory=list)
    following: list[str] = field(default_factory=list)
    followers: list[str] = field(default_factory=list)


class PolicyAgent:
    def __init__(self, profile: AgentProfile, model: BaseChatModel):
        self.profile = profile
        self.model = model
        self.memory = AgentMemory()

    def _safe_parse_actions(self, text: str) -> list[ActionRecord]:
        if not text:
            return [ActionRecord(action="do_nothing", content="")]
        block = re.search(r"```json\s*([\s\S]*?)\s*```", text)
        json_text = block.group(1) if block else text
        container = re.search(r"(\[[\s\S]*\])", json_text)
        json_text = container.group(1) if container else json_text
        try:
            raw = json.loads(json_text)
            if not isinstance(raw, list):
                return [ActionRecord(action="do_nothing", content="")]
            result: list[ActionRecord] = []
            for item in raw:
                action = item.get("action", "do_nothing")
                content = item.get("content", "")
                if action not in {
                    "post",
                    "retweet",
                    "reply",
                    "like",
                    "dislike",
                    "follow",
                    "unfollow",
                    "do_nothing",
                }:
                    action = "do_nothing"
                result.append(ActionRecord(action=action, content=content))
            return result or [ActionRecord(action="do_nothing", content="")]
        except Exception:
            return [ActionRecord(action="do_nothing", content="")]

    def act(
        self,
        topic: str,
        trigger_news: str = "",
        incoming_message: Optional[str] = None,
    ) -> list[ActionRecord]:
        prompt = build_action_prompt(
            profile=self.profile.attributes,
            memory_short=self.memory.short_term,
            memory_actions=self.memory.actions,
            topic=topic,
            trigger_news=trigger_news,
            incoming_message=incoming_message,
        )
        response = self.model.generate(
            [
                ChatMessage(role="system", content="你是一个真实社交平台用户。"),
                ChatMessage(role="user", content=prompt),
            ]
        )
        actions = self._safe_parse_actions(response)
        self.memory.remember_short(incoming_message or trigger_news)
        self.memory.remember_action(json.dumps([a.__dict__ for a in actions], ensure_ascii=False))
        return actions

