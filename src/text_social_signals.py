#!/usr/bin/env python3
"""
社交文本中的 emoji / #hashtag / @mention 统计（用于评测汇总）。

- emoji：用 `regex` 的 `\\p{Extended_Pictographic}` 匹配（避免 `\\p{Emoji}` 把 `#` 误计为 emoji）。
- hashtag：`#` 开头到空白为止（支持 Unicode 标签）。
- mention：`@` + 常见 handle 字符（Bluesky / Twitter 风格）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import regex

# Unicode Extended_Pictographic：比 \\p{Emoji} 更不易把 #、数字等算成 emoji
_RE_EMOJI = regex.compile(r"\p{Extended_Pictographic}", regex.V1)
_RE_HASHTAG = regex.compile(r"#[^\s#]+", regex.UNICODE)
_RE_MENTION = regex.compile(r"@[\w.\-]+", regex.UNICODE)

_SIGNAL_META = (
    "仅统计 post/reply 生成步中、该侧非空正文「条」数；各信号为「至少出现一次即计 1 条」；"
    "presence_ratio = rows_with_signal / total_rows（如 10 条里 5 条有 emoji 即为 0.5）。"
    "不做字符占比。emoji 用 Unicode Extended_Pictographic（与 #、@ 区分开）。"
    "text_length：对非空条求字符数之和，mean_chars_per_row = total_chars / total_rows（Python len 字符数）。"
)


@dataclass
class SocialSignalBucket:
    generation_steps: int = 0
    total_chars: int = 0
    steps_with_emoji: int = 0
    steps_with_hashtag: int = 0
    steps_with_mention: int = 0


def empty_signal_bucket() -> SocialSignalBucket:
    return SocialSignalBucket()


def _feed_text(b: SocialSignalBucket, text: str) -> None:
    """将单条正文累加：每条最多各计一次「含 emoji/#/@」。"""
    s = text or ""
    n = len(s)
    if n <= 0:
        return
    b.generation_steps += 1
    b.total_chars += n

    em = len(_RE_EMOJI.findall(s))
    ht = len(_RE_HASHTAG.findall(s))
    mn = len(_RE_MENTION.findall(s))

    if em > 0:
        b.steps_with_emoji += 1
    if ht > 0:
        b.steps_with_hashtag += 1
    if mn > 0:
        b.steps_with_mention += 1


def accumulate_from_generation_rows(
    rows: List[Dict[str, Any]],
    *,
    human_key: str = "user_content",
    model_key: str = "model_content",
) -> tuple[SocialSignalBucket, SocialSignalBucket]:
    """
    rows 来自 eval_detail 的 generation 数组项。
    分别统计人类正文与模型正文的聚合指标（仅非空正文参与步数统计；模型 null 当作空跳过该侧步数）。
    """
    human_b = SocialSignalBucket()
    model_b = SocialSignalBucket()
    for row in rows:
        u = row.get(human_key)
        if isinstance(u, str) and u.strip():
            _feed_text(human_b, u)
        m = row.get(model_key)
        if isinstance(m, str) and m.strip():
            _feed_text(model_b, m)
    return human_b, model_b


def _bucket_to_report(b: SocialSignalBucket) -> Dict[str, Any]:
    st = b.generation_steps

    def presence(used_steps: int) -> float:
        return (used_steps / st) if st > 0 else 0.0

    mean_chars = (b.total_chars / st) if st > 0 else 0.0
    return {
        "total_rows": st,
        "text_length": {
            "total_chars": b.total_chars,
            "mean_chars_per_row": round(mean_chars, 4),
        },
        "emoji": {
            "rows_with_signal": b.steps_with_emoji,
            "presence_ratio": presence(b.steps_with_emoji),
        },
        "hashtag": {
            "rows_with_signal": b.steps_with_hashtag,
            "presence_ratio": presence(b.steps_with_hashtag),
        },
        "mention": {
            "rows_with_signal": b.steps_with_mention,
            "presence_ratio": presence(b.steps_with_mention),
        },
    }


def bucket_add_into(dst: SocialSignalBucket, src: SocialSignalBucket) -> None:
    """将 src 的可加总字段并入 dst。"""
    dst.generation_steps += src.generation_steps
    dst.total_chars += src.total_chars
    dst.steps_with_emoji += src.steps_with_emoji
    dst.steps_with_hashtag += src.steps_with_hashtag
    dst.steps_with_mention += src.steps_with_mention


def finalize_pair(human_b: SocialSignalBucket, model_b: SocialSignalBucket) -> Dict[str, Any]:
    """输出 human / model 两套报告。"""
    return {
        "_meta": _SIGNAL_META,
        "human": _bucket_to_report(human_b),
        "model": _bucket_to_report(model_b),
    }


def finalize_compare(
    human_b: SocialSignalBucket,
    model_baseline_b: SocialSignalBucket,
    model_finetuned_b: SocialSignalBucket,
) -> Dict[str, Any]:
    """双模型对比：人类一套、基座模型、微调模型各一套。"""
    return {
        "_meta": _SIGNAL_META,
        "human": _bucket_to_report(human_b),
        "model_baseline": _bucket_to_report(model_baseline_b),
        "model_finetuned": _bucket_to_report(model_finetuned_b),
    }
