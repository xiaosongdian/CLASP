#!/usr/bin/env python3
"""离线重算 dpo_pairs jsonl 的 Q、r_*、margin（仅用已存 F、L）。旧数据常为 α=0.3 且 L'=(L+1)/2。"""
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import ABS_DELTA, ALPHA, DELTA, NORMALIZE_L_TO_UNIT, TAU_MINUS, TAU_PLUS


def q_fl(f: float, l: float, *, alpha: float, norm_l: bool) -> float:
    le = (float(l) + 1.0) / 2.0 if norm_l else float(l)
    return float(alpha) * float(f) + (1.0 - float(alpha)) * le


def rw(
    s: Dict[str, Dict[str, Any]],
    b: Dict[str, Dict[str, Any]],
    w: str,
    *,
    alpha: float,
    norm_l: bool,
) -> float:
    return q_fl(s[w]["F"], s[w]["L"], alpha=alpha, norm_l=norm_l) - q_fl(
        b[w]["F"], b[w]["L"], alpha=alpha, norm_l=norm_l
    )


def side_apply(side: Dict[str, Any], bs: Dict[str, Any], *, alpha: float, norm_l: bool) -> None:
    sc = side["scores"]
    for w in ("W0", "W1", "W2"):
        sc[w]["Q"] = q_fl(sc[w]["F"], sc[w]["L"], alpha=alpha, norm_l=norm_l)
    side["r_pre"] = rw(sc, bs, "W0", alpha=alpha, norm_l=norm_l)
    side["r_cur"] = rw(sc, bs, "W1", alpha=alpha, norm_l=norm_l)
    side["r_fut"] = rw(sc, bs, "W2", alpha=alpha, norm_l=norm_l)
    side["r_all"] = side["r_pre"] + side["r_cur"] + side["r_fut"]


def valid(p: Dict[str, Any]) -> bool:
    cr, rr = float(p["chosen"]["r_all"]), float(p["rejected"]["r_all"])
    m = cr - rr
    r = p.get("pair_rule")
    if r == "tau_delta":
        return cr > TAU_PLUS and rr < TAU_MINUS and m > DELTA
    if r == "abs_delta":
        return cr > rr and cr > 0 and rr < 0 and m > ABS_DELTA
    return True


def one(obj: Dict[str, Any], *, alpha: float, norm_l: bool) -> Dict[str, Any]:
    o = deepcopy(obj)
    bs = o["baseline_scores"]
    for w in ("W0", "W1", "W2"):
        bs[w]["Q"] = q_fl(bs[w]["F"], bs[w]["L"], alpha=alpha, norm_l=norm_l)
    side_apply(o["chosen"], bs, alpha=alpha, norm_l=norm_l)
    side_apply(o["rejected"], bs, alpha=alpha, norm_l=norm_l)
    o["margin"] = float(o["chosen"]["r_all"]) - float(o["rejected"]["r_all"])
    return o


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--alpha", type=float, default=None)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--normalize-l", action="store_true")
    g.add_argument("--no-normalize-l", action="store_true")
    ap.add_argument("--filter-valid", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    alpha = float(ALPHA if a.alpha is None else a.alpha)
    if a.normalize_l:
        nl = True
    elif a.no_normalize_l:
        nl = False
    else:
        nl = bool(NORMALIZE_L_TO_UNIT)

    inp = a.input.resolve()
    if not inp.is_file():
        print(f"不存在: {inp}", file=sys.stderr)
        sys.exit(1)
    if a.output is None and not a.report:
        print("需 --output 和/或 --report", file=sys.stderr)
        sys.exit(1)

    tot = k = flip = drop = 0
    out = a.output.resolve().open("w", encoding="utf-8") if a.output else None
    try:
        with inp.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                tot += 1
                n = one(json.loads(line), alpha=alpha, norm_l=nl)
                if float(n["chosen"]["r_all"]) <= float(n["rejected"]["r_all"]):
                    flip += 1
                if a.filter_valid and not valid(n):
                    drop += 1
                    continue
                k += 1
                if out:
                    out.write(json.dumps(n, ensure_ascii=False) + "\n")
    finally:
        if out:
            out.close()

    if a.report or a.output:
        print(
            json.dumps(
                {
                    "input": str(inp),
                    "output": str(a.output) if a.output else None,
                    "alpha": alpha,
                    "normalize_l": nl,
                    "filter_valid": a.filter_valid,
                    "total": tot,
                    "written": k,
                    "flipped_preference": flip,
                    "dropped_by_filter": drop,
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
