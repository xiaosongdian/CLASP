#!/usr/bin/env python3
"""
从 PostgreSQL 读取 final_users（用户→社区）与 followers_list（关注边），
聚合为「社区—社区」关注网络并绘图。

依赖：psycopg2、networkx、matplotlib（与项目内 community_data_splitter 相同的数据库环境变量）

示例：
  export DB_HOST=localhost DB_PORT=11860 DB_NAME=user_actions_db DB_USER=postgres DB_PASSWORD=postgres
  python scripts/plot_community_network_from_db.py --output output/community_network.png

若表字段名不同，可用参数覆盖，例如：
  --user-col uid --community-col cluster_id \\
  --follower-col src --followee-col dst
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

# 保证从仓库根目录直接运行本脚本时可 import process_dataset
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from process_dataset.community_data_splitter import get_db_connection

_IDENT = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def _sql_ident(name: str, what: str) -> str:
    if not name or not all(c in _IDENT for c in name):
        raise ValueError(f"非法 SQL 标识符 {what}: {name!r}（仅允许字母数字下划线）")
    return name


def fetch_user_community(
    cur,
    table: str,
    user_col: str,
    community_col: str,
) -> Dict[int, int]:
    t, u, c = _sql_ident(table, "table"), _sql_ident(user_col, "user_col"), _sql_ident(
        community_col, "community_col"
    )
    cur.execute(
        f"""
        SELECT {u}::bigint, {c}::int
        FROM {t}
        WHERE {c} IS NOT NULL
        """
    )
    out: Dict[int, int] = {}
    for uid, cid in cur.fetchall():
        try:
            out[int(uid)] = int(cid)
        except (TypeError, ValueError):
            continue
    return out


def fetch_follow_edges(
    cur,
    table: str,
    follower_col: str,
    followee_col: str,
) -> List[Tuple[int, int]]:
    cur.execute(
        f"""
        SELECT {follower_col}::bigint, {followee_col}::bigint
        FROM {table}
        WHERE {follower_col} IS NOT NULL AND {followee_col} IS NOT NULL
        """
    )
    edges: List[Tuple[int, int]] = []
    for a, b in cur.fetchall():
        try:
            edges.append((int(a), int(b)))
        except (TypeError, ValueError):
            continue
    return edges


def aggregate_community_edges(
    user_community: Dict[int, int],
    edges: Iterable[Tuple[int, int]],
    *,
    self_loops: bool,
) -> Tuple[Counter, int, int]:
    """
    返回 ( (src_c, dst_c) -> 次数, 跳过边数(缺映射), 总边数 )
    """
    skipped = 0
    total = 0
    weight: Counter = Counter()
    uc = user_community
    for a, b in edges:
        total += 1
        ca = uc.get(a)
        cb = uc.get(b)
        if ca is None or cb is None:
            skipped += 1
            continue
        if not self_loops and ca == cb:
            continue
        weight[(ca, cb)] += 1
    return weight, skipped, total


def plot_network(
    weight: Counter,
    output_path: Path,
    *,
    figsize: Tuple[float, float],
    min_edge: int,
    layout_seed: int,
    show_labels: bool,
    title: str,
) -> None:
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    for (u, v), w in weight.items():
        if w < min_edge:
            continue
        G.add_edge(u, v, weight=w)

    if G.number_of_nodes() == 0:
        raise SystemExit("图中无节点：请检查 min-edge 是否过大，或库中是否无跨社区边。")

    pos = nx.spring_layout(G, seed=layout_seed, k=2.0 / max(1, G.number_of_nodes()) ** 0.5)

    plt.figure(figsize=figsize)
    ax = plt.gca()

    node_sizes = []
    for n in G.nodes():
        ins = sum(G[u][n]["weight"] for u in G.predecessors(n))
        outs = sum(G[n][v]["weight"] for v in G.successors(n))
        node_sizes.append(300 + 40 * (ins + outs) ** 0.5)
    nx.draw_networkx_nodes(G, pos, node_color="#4ecdc4", node_size=node_sizes, alpha=0.9, ax=ax)
    edges_draw = list(G.edges(data=True))
    widths = [max(0.5, min(8.0, d["weight"] ** 0.5 * 0.8)) for _, _, d in edges_draw]
    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v) for u, v, _ in edges_draw],
        width=widths,
        alpha=0.55,
        arrows=True,
        arrowsize=14,
        connectionstyle="arc3,rad=0.08",
        ax=ax,
    )
    if show_labels:
        nx.draw_networkx_labels(G, pos, font_size=10, ax=ax)
    plt.title(title)
    plt.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    p = argparse.ArgumentParser(description="从 DB 绘制社区关注关系网（社区聚合）")
    p.add_argument("--output", "-o", type=Path, default=Path("output/community_network.png"))
    p.add_argument("--users-table", default="final_users")
    p.add_argument("--followers-table", default="followers_list")
    p.add_argument("--user-col", default="user_id", help="final_users 中用户 ID 列名")
    p.add_argument("--community-col", default="community_id", help="final_users 中社区 ID 列名")
    p.add_argument("--follower-col", default="follower_id", help="followers：发起关注的一方")
    p.add_argument("--followee-col", default="followee_id", help="followers：被关注的一方")
    p.add_argument(
        "--include-within-community",
        action="store_true",
        help="保留同社区内部的边（默认只画跨社区）",
    )
    p.add_argument("--min-edge", type=int, default=1, help="聚合后边权低于此值的边不画")
    p.add_argument("--seed", type=int, default=42, help="spring_layout 随机种子")
    p.add_argument("--max-follow-rows", type=int, default=0, help="仅调试用：最多读取关注行数，0 表示不限")
    p.add_argument("--figsize", nargs=2, type=float, default=[14.0, 10.0])
    p.add_argument("--no-label", action="store_true", help="不画社区编号标签")

    args = p.parse_args()
    figsize = (float(args.figsize[0]), float(args.figsize[1]))

    try:
        import networkx as _nx  # noqa: F401
        import matplotlib.pyplot as _plt  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "请先安装绘图依赖：pip install networkx matplotlib\n",
        )
        raise SystemExit(1)

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        print("[info] 读取 final_users ...", flush=True)
        user_community = fetch_user_community(
            cur, args.users_table, args.user_col, args.community_col
        )
        print(f"       用户映射行数（去重 user）: {len(user_community)}", flush=True)

        print("[info] 读取 followers_list ...", flush=True)
        if args.max_follow_rows and args.max_follow_rows > 0:
            cur.execute(
                f"""
                SELECT {args.follower_col}::bigint, {args.followee_col}::bigint
                FROM {args.followers_table}
                WHERE {args.follower_col} IS NOT NULL AND {args.followee_col} IS NOT NULL
                LIMIT %s
                """,
                (args.max_follow_rows,),
            )
            edges: List[Tuple[int, int]] = []
            for a, b in cur.fetchall():
                try:
                    edges.append((int(a), int(b)))
                except (TypeError, ValueError):
                    continue
        else:
            edges = fetch_follow_edges(
                cur, args.followers_table, args.follower_col, args.followee_col
            )
        print(f"       关注边数: {len(edges)}", flush=True)
    finally:
        cur.close()
        conn.close()

    weight, skipped, total = aggregate_community_edges(
        user_community,
        edges,
        self_loops=args.include_within_community,
    )
    print(
        f"[info] 聚合后边数: {sum(weight.values())} 条有向（社区对）, "
        f"原始边 {total} 条, 因缺用户/社区映射跳过 {skipped} 条",
        flush=True,
    )

    n_communities = len({c for p in weight for c in p})
    title = (
        f"社区关注网络（有向，边宽≈权重） | 社区数≈{n_communities} | "
        f"min_edge={args.min_edge}"
    )
    plot_network(
        weight,
        args.output,
        figsize=figsize,
        min_edge=args.min_edge,
        layout_seed=args.seed,
        show_labels=not args.no_label,
        title=title,
    )
    print(f"[done] 已保存: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
