#!/usr/bin/env python3
"""
按社区汇总「类 Bluetrack」数据集概览（本库实际表结构）：

- 用户数：`user_stats`（按 `community_id`）
- 关注边：`followers_list` — 语义为 `follower_id` 关注 `user_id`（被关注者）
- 动作：`user_actions` 与 `user_stats` 按 `user_id` 关联后按社区聚合；并给出各 `action_type` 计数

边统计（有向边，端点映射到 `user_stats.community_id`）：
- `edges_within`：两端同社区且均在 `user_stats` 中有社区
- `edges_out`：关注者在社区 C，被关注者不在 C 或未映射
- `edges_in`：被关注者在 C，关注者不在 C 或未映射
- `edges_touching`：至少一端映射到 C 的边在 C 上计一次（社区内边只计一次）

用法（与 `community_data_splitter` 一致的环境变量）：

  export DB_HOST=127.0.0.1 DB_PORT=11860 DB_NAME=user_actions_db DB_USER=postgres DB_PASSWORD=...
  python scripts/print_bluetrack_community_overview.py
  python scripts/print_bluetrack_community_overview.py --out-md output/bluetrack_community_overview.md --out-csv output/bluetrack_community_overview.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from process_dataset.community_data_splitter import get_db_connection


def fetch_action_types(cur) -> List[str]:
    cur.execute(
        """
        SELECT action_type
        FROM user_actions
        GROUP BY action_type
        ORDER BY action_type
        """
    )
    return [str(r[0]) for r in cur.fetchall()]


def fetch_users_per_community(cur) -> Dict[int, int]:
    cur.execute(
        """
        SELECT community_id::int, COUNT(*)::bigint
        FROM user_stats
        WHERE community_id IS NOT NULL
        GROUP BY community_id
        ORDER BY community_id
        """
    )
    return {int(r[0]): int(r[1]) for r in cur.fetchall()}


def fetch_actions_pivot(cur, action_types: Sequence[str]) -> Dict[int, Dict[str, Any]]:
    if not action_types:
        cur.execute(
            """
            SELECT s.community_id::int, COUNT(*)::bigint AS n_actions_total
            FROM user_actions a
            INNER JOIN user_stats s ON s.user_id = a.user_id
            WHERE s.community_id IS NOT NULL
            GROUP BY s.community_id
            ORDER BY s.community_id
            """
        )
        return {int(r[0]): {"n_actions_total": int(r[1])} for r in cur.fetchall()}

    safe_aliases = []
    parts = []
    params: List[str] = []
    for t in action_types:
        al = "t_" + "".join(c if c.isalnum() else "_" for c in t)[:60]
        safe_aliases.append(al)
        parts.append(f"COUNT(*) FILTER (WHERE a.action_type = %s) AS {al}")
        params.append(t)
    cols = ", ".join(parts)
    cur.execute(
        f"""
        SELECT s.community_id::int, COUNT(*)::bigint AS n_actions_total, {cols}
        FROM user_actions a
        INNER JOIN user_stats s ON s.user_id = a.user_id
        WHERE s.community_id IS NOT NULL
        GROUP BY s.community_id
        ORDER BY s.community_id
        """,
        params,
    )
    desc = [d[0] for d in cur.description]
    out: Dict[int, Dict[str, Any]] = {}
    for row in cur.fetchall():
        d = dict(zip(desc, row))
        cid = int(d.pop("community_id"))
        for k, v in d.items():
            d[k] = int(v) if v is not None else 0
        out[cid] = d
    return out


def fetch_edge_stats(cur) -> Dict[int, Dict[str, int]]:
    """
    返回 community_id -> {edges_within, edges_out, edges_in, edges_touching}
    """
    cur.execute(
        """
        WITH m AS (
          SELECT
            sf.community_id AS src_c,
            st.community_id AS dst_c
          FROM followers_list f
          LEFT JOIN user_stats sf ON sf.user_id = f.follower_id
          LEFT JOIN user_stats st ON st.user_id = f.user_id
        ),
        within_c AS (
          SELECT src_c::int AS cid, COUNT(*)::bigint AS n
          FROM m
          WHERE src_c IS NOT NULL AND src_c = dst_c
          GROUP BY src_c
        ),
        out_c AS (
          SELECT src_c::int AS cid, COUNT(*)::bigint AS n
          FROM m
          WHERE src_c IS NOT NULL
            AND (dst_c IS DISTINCT FROM src_c OR dst_c IS NULL)
          GROUP BY src_c
        ),
        in_c AS (
          SELECT dst_c::int AS cid, COUNT(*)::bigint AS n
          FROM m
          WHERE dst_c IS NOT NULL
            AND (src_c IS DISTINCT FROM dst_c OR src_c IS NULL)
          GROUP BY dst_c
        ),
        touch AS (
          SELECT cid::int, SUM(n)::bigint AS n
          FROM (
            SELECT src_c AS cid, COUNT(*)::bigint AS n
            FROM m
            WHERE src_c IS NOT NULL
            GROUP BY src_c
            UNION ALL
            SELECT dst_c AS cid, COUNT(*)::bigint AS n
            FROM m
            WHERE dst_c IS NOT NULL AND dst_c IS DISTINCT FROM src_c
            GROUP BY dst_c
          ) u
          GROUP BY cid
        )
        SELECT c.community_id::int,
               COALESCE(w.n, 0)::bigint,
               COALESCE(o.n, 0)::bigint,
               COALESCE(i.n, 0)::bigint,
               COALESCE(t.n, 0)::bigint
        FROM (SELECT DISTINCT community_id FROM user_stats WHERE community_id IS NOT NULL) c(community_id)
        LEFT JOIN within_c w ON w.cid = c.community_id
        LEFT JOIN out_c o ON o.cid = c.community_id
        LEFT JOIN in_c i ON i.cid = c.community_id
        LEFT JOIN touch t ON t.cid = c.community_id
        ORDER BY c.community_id
        """
    )
    out: Dict[int, Dict[str, int]] = {}
    for cid, ew, eo, ei, et in cur.fetchall():
        out[int(cid)] = {
            "edges_within": int(ew),
            "edges_out": int(eo),
            "edges_in": int(ei),
            "edges_touching": int(et),
        }
    return out


def fetch_global_action_summary(cur) -> Tuple[int, List[Tuple[str, int]]]:
    cur.execute("SELECT COUNT(*)::bigint FROM user_actions")
    total = int(cur.fetchone()[0])
    cur.execute(
        """
        SELECT action_type, COUNT(*)::bigint AS n
        FROM user_actions
        GROUP BY action_type
        ORDER BY n DESC
        """
    )
    dist = [(str(r[0]), int(r[1])) for r in cur.fetchall()]
    return total, dist


def main() -> None:
    p = argparse.ArgumentParser(description="按社区打印 user_stats / followers_list / user_actions 汇总表")
    p.add_argument("--out-md", type=Path, default=None, help="写入 Markdown 表路径")
    p.add_argument("--out-csv", type=Path, default=None, help="写入 CSV 路径")
    args = p.parse_args()

    conn = get_db_connection()
    cur = conn.cursor()
    # 全表聚合可能超过 community_data_splitter 默认 60s statement_timeout
    cur.execute("SET statement_timeout = 0")

    try:
        action_types = fetch_action_types(cur)
        users_c = fetch_users_per_community(cur)
        actions_c = fetch_actions_pivot(cur, action_types)
        edges_c = fetch_edge_stats(cur)
        global_total, global_dist = fetch_global_action_summary(cur)
    finally:
        cur.close()
        conn.close()

    all_cids = sorted(set(users_c) | set(actions_c) | set(edges_c))

    # CSV 列
    base_cols = [
        "community_id",
        "n_users",
        "edges_within",
        "edges_out",
        "edges_in",
        "edges_touching",
        "n_actions_total",
    ]
    type_cols = [f"n_act_{t}" for t in action_types]
    header = base_cols + type_cols

    rows: List[Dict[str, Any]] = []
    for cid in all_cids:
        ad = actions_c.get(cid, {})
        ed = edges_c.get(
            cid,
            {"edges_within": 0, "edges_out": 0, "edges_in": 0, "edges_touching": 0},
        )
        row = {
            "community_id": cid,
            "n_users": users_c.get(cid, 0),
            "edges_within": ed["edges_within"],
            "edges_out": ed["edges_out"],
            "edges_in": ed["edges_in"],
            "edges_touching": ed["edges_touching"],
            "n_actions_total": ad.get("n_actions_total", 0),
        }
        for t in action_types:
            al = "t_" + "".join(c if c.isalnum() else "_" for c in t)[:60]
            row[f"n_act_{t}"] = ad.get(al, 0)
        rows.append(row)

    # 终端摘要
    print(f"user_actions 全局总行数: {global_total}")
    print("全局 action_type 分布（降序）:")
    for t, n in global_dist:
        print(f"  {t!r}: {n}")
    print()
    print("| " + " | ".join(header) + " |")
    print("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        print("| " + " | ".join(str(row[c]) for c in header) + " |")

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rows)
        print(f"\n[done] CSV: {args.out_csv.resolve()}", file=sys.stderr)

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Bluetrack-style 社区数据概览",
            "",
            f"- 数据源：`user_stats`、`followers_list`（`follower_id` → `user_id`）、`user_actions`",
            f"- `user_actions` 全局记录数: **{global_total}**",
            "",
            "## 全局动作类型分布",
            "",
            "| action_type | count |",
            "| --- | ---: |",
        ]
        for t, n in global_dist:
            lines.append(f"| `{t}` | {n} |")
        lines.extend(["", "## 按社区汇总", "", "| " + " | ".join(header) + " |", "| " + " | ".join("---:" for _ in header) + " |"])
        for row in rows:
            lines.append("| " + " | ".join(str(row[c]) for c in header) + " |")
        lines.append("")
        args.out_md.write_text("\n".join(lines), encoding="utf-8")
        print(f"[done] Markdown: {args.out_md.resolve()}", file=sys.stderr)


if __name__ == "__main__":
    main()
