import json
from typing import Optional


def build_profile_text(profile: dict) -> str:
    chunks = []
    for k, v in profile.items():
        chunks.append(f"- {k}: {v}")
    return "\n".join(chunks)


def build_action_prompt(
    profile: dict,
    memory_short: list[str],
    memory_actions: list[str],
    topic: str,
    trigger_news: str = "",
    incoming_message: Optional[str] = None,
) -> str:
    output_example = json.dumps(
        [{"action": "reply", "content": "I see your point, but I disagree."}],
        ensure_ascii=False,
    )
    return (
        "你是一个社交平台用户智能体。\n"
        f"【主题】{topic}\n"
        f"【用户画像】\n{build_profile_text(profile)}\n"
        f"【短期记忆】{memory_short[-8:]}\n"
        f"【历史动作】{memory_actions[-10:]}\n"
        f"【触发新闻】{trigger_news}\n"
        f"【收到消息】{incoming_message or '无'}\n\n"
        "请输出 JSON 数组，每个元素包含 action 和 content（无内容可为空字符串）。\n"
        "可选 action: post, retweet, reply, like, dislike, follow, unfollow, do_nothing。\n"
        "只输出 JSON，不要输出任何解释。\n"
        f"示例: {output_example}"
    )

