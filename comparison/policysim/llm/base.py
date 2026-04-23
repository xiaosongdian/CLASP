from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List


@dataclass
class ChatMessage:
    role: str
    content: str


class BaseChatModel(ABC):
    @abstractmethod
    def generate(self, messages: List[ChatMessage]) -> str:
        raise NotImplementedError

