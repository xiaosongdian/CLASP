from typing import List
from openai import OpenAI

from policysim.config import ModelConfig
from policysim.llm.base import BaseChatModel, ChatMessage


class OpenAIChatModel(BaseChatModel):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def generate(self, messages: List[ChatMessage]) -> str:
        payload = [{"role": m.role, "content": m.content} for m in messages]
        resp = self.client.chat.completions.create(
            model=self.config.model_name,
            messages=payload,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return resp.choices[0].message.content or ""

