from dataclasses import dataclass, field
from typing import List, Literal, Optional


AgentAction = Literal[
    "post",
    "retweet",
    "reply",
    "like",
    "dislike",
    "follow",
    "unfollow",
    "do_nothing",
]


@dataclass
class ActionRecord:
    action: AgentAction
    content: str = ""
    target_user_id: str = ""


@dataclass
class InteractionMessage:
    sender_id: str
    content: str
    action: AgentAction = "post"
    parent_message_id: Optional[str] = None
    receiver_ids: List[str] = field(default_factory=list)

