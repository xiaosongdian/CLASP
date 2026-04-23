from dataclasses import dataclass, field
from typing import List


@dataclass
class AgentMemory:
    short_term: List[str] = field(default_factory=list)
    long_term: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)

    def remember_short(self, text: str, max_items: int = 30) -> None:
        if not text:
            return
        self.short_term.append(text)
        if len(self.short_term) > max_items:
            self.short_term = self.short_term[-max_items:]

    def remember_long(self, text: str, max_items: int = 200) -> None:
        if not text:
            return
        self.long_term.append(text)
        if len(self.long_term) > max_items:
            self.long_term = self.long_term[-max_items:]

    def remember_action(self, text: str, max_items: int = 200) -> None:
        if not text:
            return
        self.actions.append(text)
        if len(self.actions) > max_items:
            self.actions = self.actions[-max_items:]

