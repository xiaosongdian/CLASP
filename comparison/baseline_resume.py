"""Baseline jsonl 断点续跑：从已有输出统计各 method 已成功完成的用户。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


def serialize_user_key(rec: Dict[str, Any]) -> str:
    return f"{rec.get('user_id')}\t{rec.get('community_id')}"


def load_completed_keys_per_method(
    *,
    separate_by_method: bool,
    methods: List[str],
    method_paths: Dict[str, Path],
    combined_jsonl: Optional[Path],
    skip_error_rows: bool = True,
) -> Dict[str, Set[str]]:
    """
    各 method 下已成功写入的用户键（无 error 视为成功）。
    skip_error_rows=True：含 error 的行下次会重跑。
    """
    out: Dict[str, Set[str]] = {m: set() for m in methods}
    if separate_by_method:
        for m in methods:
            p = method_paths.get(m)
            if p is None or not p.is_file():
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if skip_error_rows and obj.get("error"):
                        continue
                    out[m].add(serialize_user_key(obj))
    else:
        if combined_jsonl is None or not combined_jsonl.is_file():
            return out
        with combined_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                meth = obj.get("method")
                if meth not in out:
                    continue
                if skip_error_rows and obj.get("error"):
                    continue
                out[str(meth)].add(serialize_user_key(obj))
    return out


def load_all_prior_rows(
    *,
    separate_by_method: bool,
    methods: List[str],
    method_paths: Dict[str, Path],
    combined_jsonl: Optional[Path],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if separate_by_method:
        for m in methods:
            p = method_paths.get(m)
            if p is None or not p.is_file():
                continue
            with p.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    else:
        if combined_jsonl is None or not combined_jsonl.is_file():
            return rows
        with combined_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows
