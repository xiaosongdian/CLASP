#!/usr/bin/env python3
"""
单用户 clasp_online 个案：重跑窗口链（调用画像 + 动作 API），导出每轮
  - F / L / Q（与 baseline_chain 一致）
  - 用于预测的画像全文、精炼后画像全文
  - 预测 vs 真实的偏差文本（build_behavior_discrepancies）

默认在 --windowed-dir 下扫描 community_*.jsonl，若主输入文件不含该 user_id 则自动换文件
（例如用户实际在 community_6.jsonl 而非 community_1.jsonl）。

用法（仓库根目录）：

1）**离线合并**（不调用 LLM，从已有 ``baseline_chain_*.jsonl`` + ``profiles.jsonl`` 拼 TXT；
   含每轮 F/L/Q 与画像；**不含** ``build_behavior_discrepancies`` 原文，见文件内说明）：
  python3 -m comparison.export_case_study_clasp_txt \\
    --user-id 274402 --offline \\
    --out output/comparison/case_study_user_274402_clasp_online.txt

2）**在线重算**（需 vLLM；会写出真实 ``behavior_discrepancies``）：
  python3 -m comparison.export_case_study_clasp_txt \\
    --user-id 274402 \\
    --input-jsonl output/windowed/test/community_1.jsonl

窗口化数据：若主文件无该用户，会在 ``--windowed-dir`` 下自动扫描 ``community_*.jsonl``（274402 在 **community_6**）。

依赖：在线模式与 ``run_baseline_comparison`` 相同；离线模式仅需本仓库与已有输出文件。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from comparison.window_chain_eval import evaluate_user_window_chain
from src.dpo_pipeline import preflight_check
from src.scorer import SemanticScorer


def _find_user_record(
    user_id: int,
    primary: Path,
    scan_dir: Path,
) -> Tuple[Dict[str, Any], Path]:
    """先在 primary 中找 user_id；找不到则在 scan_dir 下所有 community_*.jsonl 中找。"""
    candidates = [primary]
    if primary.is_file():
        pass
    else:
        candidates = []
    for p in sorted(scan_dir.glob("community_*.jsonl")):
        if p not in candidates:
            candidates.append(p)

    for fp in candidates:
        if not fp.is_file():
            continue
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(o.get("user_id", -1)) == user_id:
                    return o, fp.resolve()
    raise FileNotFoundError(
        f"在 {primary} 及目录 {scan_dir} 的 community_*.jsonl 中均未找到 user_id={user_id}"
    )


def _fmt_step_txt(step: Dict[str, Any]) -> str:
    lines = []
    hw, tw = step.get("history_window"), step.get("target_window")
    lines.append(f"  step_index={step.get('step_index')}  {hw} -> {tw}")
    lines.append(
        f"  F={step.get('F')}  L={step.get('L')}  Q={step.get('Q')}"
    )
    if "profile_updated" in step:
        lines.append(f"  profile_updated={step.get('profile_updated')}")
    if "best_candidate_index" in step:
        lines.append(f"  best_candidate_index={step.get('best_candidate_index')}")
    if "profile_length" in step:
        lines.append(f"  profile_length={step.get('profile_length')}")
    lines.append("")
    pu = step.get("profile_used_for_prediction")
    if pu is not None:
        lines.append("  --- profile_used_for_prediction（本步预测前画像）---")
        lines.append(str(pu))
        lines.append("")
    disc = step.get("behavior_discrepancies")
    if disc is not None:
        lines.append("  --- behavior_discrepancies（预测 vs 真实，精炼信号）---")
        lines.append(str(disc))
        lines.append("")
    pa = step.get("profile_after_refinement")
    if pa is not None:
        lines.append("  --- profile_after_refinement（本步精炼后画像）---")
        lines.append(str(pa))
        lines.append("")
    return "\n".join(lines)


def _pick_baseline_row(baseline_path: Path, user_id: int, method: str) -> Dict[str, Any]:
    rows: list[Dict[str, Any]] = []
    with baseline_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(o.get("user_id", -1)) != user_id:
                continue
            if str(o.get("method")) != method:
                continue
            rows.append(o)
    if not rows:
        raise FileNotFoundError(
            f"{baseline_path} 中无 user_id={user_id} 且 method={method} 的记录"
        )
    true_rows = [r for r in rows if r.get("always_accept_refinement") is True]
    return true_rows[-1] if true_rows else rows[-1]


def _load_profile_snapshots(
    profiles_path: Path, user_id: int, method: str
) -> Tuple[str, Dict[int, str]]:
    """返回 (W0 后初始画像, step_index -> 该步精炼后画像)。"""
    initial = ""
    after_step: Dict[int, str] = {}
    with profiles_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if int(o.get("user_id", -1)) != user_id:
                continue
            if str(o.get("method")) != method:
                continue
            phase = o.get("phase")
            prof = str(o.get("profile") or "")
            if phase == "after_W0_initial":
                initial = prof
            elif phase == "after_chain_step":
                si = o.get("step_index")
                if isinstance(si, int):
                    after_step[si] = prof
    return initial, after_step


def _merge_offline_baseline_profiles(
    baseline_row: Dict[str, Any],
    initial_profile: str,
    after_step: Dict[int, str],
    *,
    user_id: int,
) -> Dict[str, Any]:
    """构造与在线 case_study_capture 接近的 result 字典。"""
    steps_in = baseline_row.get("steps") or []
    merged: list[Dict[str, Any]] = []
    disc_note = (
        "（离线合并：预测–真实偏差全文未写入 baseline/profiles 产物。\n"
        "若需要 ``build_behavior_discrepancies`` 原文，请在本机 vLLM 可用时运行：\n"
        f"  python3 -m comparison.export_case_study_clasp_txt --user-id {user_id} "
        "--input-jsonl output/windowed/test/community_6.jsonl\n"
        "  不要加 --offline。）"
    )
    for si, step in enumerate(steps_in):
        d = dict(step)
        d["profile_used_for_prediction"] = (
            initial_profile if si == 0 else after_step.get(si - 1, f"[缺失：无 step_index={si-1} 快照]")
        )
        d["profile_after_refinement"] = after_step.get(
            si, f"[缺失：无 step_index={si} 快照]"
        )
        d["behavior_discrepancies"] = disc_note
        merged.append(d)
    out = {k: v for k, v in baseline_row.items() if k != "steps"}
    out["steps"] = merged
    out["case_study_initial_profile"] = initial_profile
    return out


def _build_report(
    result: Dict[str, Any],
    source_jsonl: Path,
    user_id: int,
) -> str:
    blocks = []
    blocks.append("=" * 80)
    blocks.append("Case study: clasp_online 窗口链（单用户）")
    blocks.append("=" * 80)
    blocks.append(f"user_id: {result.get('user_id')}")
    blocks.append(f"community_id (来自窗口化数据): {result.get('community_id')}")
    blocks.append(f"数据源: {source_jsonl}")
    note = result.get("_provenance_note")
    if note:
        blocks.append(f"说明: {note}")
    blocks.append(f"method: {result.get('method')}")
    blocks.append(f"window_keys: {result.get('window_keys')}")
    blocks.append(f"mean_F: {result.get('mean_F')}  mean_Q: {result.get('mean_Q')}")
    blocks.append(f"always_accept_refinement: {result.get('always_accept_refinement')}")
    blocks.append(f"action_prompt_include_observed_history: {result.get('action_prompt_include_observed_history')}")
    blocks.append("")

    if result.get("error"):
        blocks.append(f"[错误] {result.get('error')}")
        return "\n".join(blocks)

    s0 = result.get("case_study_initial_profile")
    if s0 is not None:
        blocks.append("-" * 80)
        blocks.append("【W0 后初始画像 S0】case_study_initial_profile")
        blocks.append("-" * 80)
        blocks.append(str(s0))
        blocks.append("")

    blocks.append("-" * 80)
    blocks.append("【链上各步】F/L/Q + 画像 + 预测偏差")
    blocks.append("-" * 80)
    for step in result.get("steps") or []:
        blocks.append(_fmt_step_txt(step))
        blocks.append("-" * 40)

    return "\n".join(blocks)


def main() -> None:
    p = argparse.ArgumentParser(description="导出 clasp_online 单用户个案 txt（画像 + 预测偏差 + F/L/Q）")
    p.add_argument("--user-id", type=int, required=True)
    p.add_argument(
        "--offline",
        action="store_true",
        help="从已有 baseline_chain jsonl + profiles.jsonl 合并，不调用 API",
    )
    p.add_argument(
        "--baseline-jsonl",
        type=Path,
        default=Path("output/comparison/clasp_online/baseline_chain_test_contiguous.jsonl"),
        help="--offline 时读取链上 F/L/Q",
    )
    p.add_argument(
        "--profiles-jsonl",
        type=Path,
        default=Path(
            "output/comparison/clasp_online/profile_snapshots/"
            "baseline_chain_test_contiguous/profiles.jsonl"
        ),
        help="--offline 时读取画像快照",
    )
    p.add_argument(
        "--input-jsonl",
        type=Path,
        default=Path("output/windowed/test/community_1.jsonl"),
        help="首选窗口化 jsonl；若不含该用户则在 --windowed-dir 下自动扫描",
    )
    p.add_argument(
        "--windowed-dir",
        type=Path,
        default=Path("output/windowed/test"),
        help="自动查找用户时的目录（默认与 test split 一致）",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="输出 txt；默认 output/comparison/case_study_user_<id>_clasp_online.txt",
    )
    p.add_argument("--scorer-device", type=str, default="cpu")
    p.add_argument("--skip-preflight", action="store_true")
    p.add_argument(
        "--method",
        choices=("clasp_online", "clasp_online_no_hist"),
        default="clasp_online",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    if not str(args.input_jsonl).startswith("/"):
        primary = (root / args.input_jsonl).resolve()
    else:
        primary = args.input_jsonl.resolve()
    windowed_dir = (
        (root / args.windowed_dir).resolve()
        if not str(args.windowed_dir).startswith("/")
        else args.windowed_dir.resolve()
    )

    uid = int(args.user_id)
    out = args.out
    if out is None:
        suf = "_clasp_online_offline.txt" if args.offline else "_clasp_online_rerun.txt"
        out = root / "output" / "comparison" / f"case_study_user_{uid}{suf}"
    else:
        out = out.resolve() if str(out).startswith("/") else (root / out).resolve()

    if args.offline:
        baseline_p = (
            (root / args.baseline_jsonl).resolve()
            if not str(args.baseline_jsonl).startswith("/")
            else args.baseline_jsonl.resolve()
        )
        prof_p = (
            (root / args.profiles_jsonl).resolve()
            if not str(args.profiles_jsonl).startswith("/")
            else args.profiles_jsonl.resolve()
        )
        if not baseline_p.is_file():
            print(f"[错误] 不存在: {baseline_p}", file=sys.stderr)
            sys.exit(1)
        if not prof_p.is_file():
            print(f"[错误] 不存在: {prof_p}", file=sys.stderr)
            sys.exit(1)
        try:
            row = _pick_baseline_row(baseline_p, uid, args.method)
        except FileNotFoundError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)
        initial, after_map = _load_profile_snapshots(prof_p, uid, args.method)
        result = _merge_offline_baseline_profiles(
            row, initial, after_map, user_id=uid
        )
        # 窗口化文件路径（仅作文档；离线不依赖其内容）
        try:
            _, found_path = _find_user_record(uid, primary, windowed_dir)
        except FileNotFoundError:
            found_path = Path("(未在 windowed/test 中找到该 user_id)")
        result["_provenance_note"] = (
            f"离线合并 | F/L/Q: {baseline_p} | 画像: {prof_p} | 窗口化参考: {found_path}"
        )
        text = _build_report(result, found_path, uid)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"[CaseStudy][offline] 已写入: {out}", flush=True)
        return

    try:
        user, found_path = _find_user_record(uid, primary, windowed_dir)
    except FileNotFoundError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)

    if found_path != primary:
        print(
            f"[提示] user_id={uid} 不在 {primary} 中，已从实际文件载入: {found_path}",
            flush=True,
        )

    if not args.skip_preflight and not preflight_check(comparison_methods=[args.method]):
        print("[错误] 预检失败（API/模型配置）。可加 --skip-preflight 跳过（仍可能在调用时报错）。", file=sys.stderr)
        sys.exit(1)

    print(f"[CaseStudy] SemanticScorer device={args.scorer_device}", flush=True)
    scorer = SemanticScorer(device=args.scorer_device)

    result = evaluate_user_window_chain(
        user,
        args.method,
        scorer,
        profile_model=None,
        profile_tokenizer=None,
        action_model=None,
        action_tokenizer=None,
        refinement_variants=1,
        workers=1,
        always_accept_refinement=True,
        profile_snapshot_dir=None,
        action_prompt_include_observed_history=True,
        enable_three_window_evaluation=False,
        case_study_capture=True,
    )

    text = _build_report(result, found_path, uid)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"[CaseStudy] 已写入: {out}", flush=True)


if __name__ == "__main__":
    main()
