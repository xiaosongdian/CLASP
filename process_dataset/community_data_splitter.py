#!/usr/bin/env python3
"""
按社区切分用户并导出序列动作数据。

目标：
1) 训练/测试：社区 0,1,3,4,5，按每个社区 70%/30% 切分用户。
2) 未见评估：社区 6,7 随机抽取 500 个用户。
3) 导出到 data 目录，按 split + 社区文件组织（仅保留用户动作数据）。
"""

import argparse
import json
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import psycopg2


DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "11860")),
    "dbname": os.getenv("DB_NAME", "user_actions_db"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

DB_FETCH_BATCH = 100000
TEXT_LONG = 500


def get_db_connection():
    return psycopg2.connect(
        **DB_CONFIG,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=3,
        connect_timeout=10,
        options="-c statement_timeout=60000",
    )


def parse_cids(cid_str: str) -> List[int]:
    return [int(x.strip()) for x in cid_str.split(",") if x.strip()]


def row_to_action(row: Tuple) -> Tuple[int, Dict]:
    user_id, action_type, post_id, _target_post_id, action_text, original_text, date = row
    date = str(date)
    target = None
    if action_type != "post":
        if action_type == "reply":
            target = str(original_text)[:TEXT_LONG] if original_text else (str(action_text)[:TEXT_LONG] if action_text else f"post_{post_id}")
        else:
            target = str(action_text)[:TEXT_LONG] if action_text else (str(original_text)[:TEXT_LONG] if original_text else f"post_{post_id}")
    year, month, day = date[0:4], date[4:6], date[6:8]
    hour, minute = date[8:10], date[10:12]
    action = {
        "timestamp": f"{year}-{month}-{day} {hour}:{minute}",
        "action_type": action_type,
        "target": target,
        "action_text": action_text,
        "date": date,
    }
    return user_id, action


def fetch_users_by_communities(community_ids: List[int]) -> Dict[int, List[int]]:
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT s.community_id, s.user_id
        FROM user_stats s
        JOIN user_actions_sampled a ON a.user_id = s.user_id
        WHERE s.community_id = ANY(%s)
        ORDER BY s.community_id, s.user_id
        """,
        (community_ids,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    out: Dict[int, List[int]] = defaultdict(list)
    for cid, uid in rows:
        out[cid].append(uid)
    return out


def split_train_test_per_community(
    users_by_community: Dict[int, List[int]],
    train_ratio: float,
    rng: random.Random,
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    train_users: Dict[int, List[int]] = {}
    test_users: Dict[int, List[int]] = {}
    for cid, users in users_by_community.items():
        copied = users[:]
        rng.shuffle(copied)
        train_n = int(len(copied) * train_ratio)
        train_users[cid] = copied[:train_n]
        test_users[cid] = copied[train_n:]
    return train_users, test_users


def sample_eval_users(
    users_by_community: Dict[int, List[int]],
    total_eval_users: int,
    rng: random.Random,
) -> Dict[int, List[int]]:
    pooled = []
    for cid, users in users_by_community.items():
        pooled.extend((cid, uid) for uid in users)
    if not pooled:
        return {}

    sample_size = min(total_eval_users, len(pooled))
    sampled = rng.sample(pooled, sample_size)
    out: Dict[int, List[int]] = defaultdict(list)
    for cid, uid in sampled:
        out[cid].append(uid)
    return out


def stream_actions_for_users(community_id: int, user_ids: List[int]) -> Dict[int, List[Dict]]:
    if not user_ids:
        return {}

    conn = get_db_connection()
    cur = conn.cursor(name=f"split_stream_{community_id}_{os.getpid()}")
    cur.itersize = DB_FETCH_BATCH
    cur.execute(
        """
        SELECT s.user_id, a.action_type, a.post_id, a.target_post_id,
               a.action_text, a.original_text, a.date
        FROM user_actions_sampled a
        JOIN user_stats s ON a.user_id = s.user_id
        WHERE s.community_id = %s AND s.user_id = ANY(%s)
        ORDER BY a.user_id, a.date
        """,
        (community_id, user_ids),
    )

    user_actions: Dict[int, List[Dict]] = defaultdict(list)
    while True:
        batch = cur.fetchmany(DB_FETCH_BATCH)
        if not batch:
            break
        for row in batch:
            uid, action = row_to_action(row)
            user_actions[uid].append(action)

    cur.close()
    conn.close()
    return user_actions


def ensure_dirs(base: Path, split_name: str) -> None:
    (base / split_name).mkdir(parents=True, exist_ok=True)


def write_jsonl(path: Path, records: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def export_split(base: Path, split_name: str, users_by_community: Dict[int, List[int]]) -> Dict:
    ensure_dirs(base, split_name)

    split_summary = {
        "split": split_name,
        "communities": {},
        "total_users": 0,
        "total_actions": 0,
    }

    for cid, user_ids in sorted(users_by_community.items()):
        user_actions = stream_actions_for_users(cid, user_ids)
        user_records = []

        for uid in user_ids:
            actions = sorted(
                user_actions.get(uid, []),
                key=lambda a: (a.get("date", ""), a.get("timestamp", "")),
            )
            user_records.append(
                {
                    "community_id": cid,
                    "user_id": uid,
                    "actions": actions,
                    "action_count": len(actions),
                }
            )

        users_file = base / split_name / f"community_{cid}.jsonl"
        write_jsonl(users_file, user_records)

        action_total = sum(r["action_count"] for r in user_records)
        split_summary["communities"][str(cid)] = {
            "user_count": len(user_ids),
            "action_count": action_total,
            "users_file": str(users_file),
        }
        split_summary["total_users"] += len(user_ids)
        split_summary["total_actions"] += action_total

        print(
            f"[{split_name}] 社区 {cid}: users={len(user_ids)}, actions={action_total}, "
            f"users_file={users_file}"
        )

    return split_summary


def main():
    parser = argparse.ArgumentParser(description="按社区拆分用户序列并导出")
    parser.add_argument("--train-communities", default="0,1,3,4,5", help="训练/测试社区ID，逗号分隔")
    parser.add_argument("--eval-communities", default="6,7", help="未见评估社区ID，逗号分隔")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="训练比例")
    parser.add_argument("--eval-users", type=int, default=500, help="未见评估用户总数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output-dir", default="data", help="输出目录")
    parser.add_argument("--expected-train-users", type=int, default=3248, help="期望训练用户数")
    parser.add_argument("--expected-test-users", type=int, default=1392, help="期望测试用户数")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cids = parse_cids(args.train_communities)
    eval_cids = parse_cids(args.eval_communities)

    print(f"训练/测试社区: {train_cids}")
    print(f"评估社区(未见): {eval_cids}")

    train_pool = fetch_users_by_communities(train_cids)
    train_users, test_users = split_train_test_per_community(train_pool, args.train_ratio, rng)

    train_summary = export_split(output_dir, "train", train_users)
    test_summary = export_split(output_dir, "test", test_users)

    eval_pool = fetch_users_by_communities(eval_cids)
    eval_users = sample_eval_users(eval_pool, args.eval_users, rng)
    eval_summary = export_split(output_dir, "eval_unseen", eval_users)

    expected_notes = {
        "expected_train_users": args.expected_train_users,
        "expected_test_users": args.expected_test_users,
        "actual_train_users": train_summary["total_users"],
        "actual_test_users": test_summary["total_users"],
        "match_expected_train": train_summary["total_users"] == args.expected_train_users,
        "match_expected_test": test_summary["total_users"] == args.expected_test_users,
    }

    summary = {
        "config": {
            "train_communities": train_cids,
            "eval_communities": eval_cids,
            "train_ratio": args.train_ratio,
            "eval_users": args.eval_users,
            "seed": args.seed,
            "output_dir": str(output_dir),
        },
        "train": train_summary,
        "test": test_summary,
        "eval_unseen": eval_summary,
        "expected_notes": expected_notes,
    }

    summary_file = output_dir / "split_summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== 完成 ===")
    print(f"Train users: {train_summary['total_users']}")
    print(f"Test users: {test_summary['total_users']}")
    print(f"Eval users: {eval_summary['total_users']}")
    print(f"Summary: {summary_file}")

    if not expected_notes["match_expected_train"] or not expected_notes["match_expected_test"]:
        print(
            "[提示] 实际用户数与期望(3248/1392)不一致。通常是因为库中可用用户总量变化，"
            "可检查 split_summary.json。"
        )


if __name__ == "__main__":
    main()
