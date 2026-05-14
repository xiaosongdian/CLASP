#!/usr/bin/env python3
"""
Clasp DPO 画像有效性 · 切片评估（非全量窗口链）

默认评估「P0 用 W0 建画像 → 预测 W1 → 按误差精炼得 P1 → 预测 W2」两跳。加 ``--slice-eval-mode w0_w1_w2_p0p1`` 时：P0 分别评 **W0/W1/W2**，再用 **W1** 上预测误差精炼得 P1，P1 再评三窗；**W0 从不**向动作模型注入近期行为（history 与观测块均为空）；**W1/W2**：未加 ``--no-action-prompt-observed-history`` 时与旧 ``w1_w2`` 切片一致（上一窗 history + 观测块），加上该参数则 W1/W2 也不注入。行内另写 ``P0_W0_F``…``P1_W2_Q``，并填 ``W1_*``/``W2_*`` 为 **P0 / P1 各自三窗**的算术均值（便于全链路粗看）；**柱状图** ``plot_dpo_profile_slice_radar`` 对 p0p1 成功行改为聚合 **``P0_W1_*``（P0@W1）** 与 **``P1_W2_*``（P1@W2）** 单窗得分。加 ``--slice-eval-mode w2_w3`` 时需四个连续窗口：
仍用 W1 误差精炼得 P1，再评 **物理 W2** 并写入 **W1_***（与柱状图「基线柱」语义对齐），再用 W2 误差精炼得 P2 后评 **物理 W3** 写入 **W2_***；输出文件名为 ``…_w2w3.jsonl``，勿与 ``w1_w2`` 混在同一 jsonl 里 ``--resume``。

动作侧统一使用 **Clasp 动作 checkpoint**（与 window_chain 中 clasp_online 一致），
只切换 **画像精炼（P0→P1）** 后端，便于对比 DPO 画像 vs 基座 vLLM vs gpt-4o-mini。

**默认（``--initial-p0 shared_gpt``）**：三种 variant **共用同一份**由 **GPT-4o-mini**（``cfg.PROFILE_MODEL`` + 商用 API）对 W0 **只调用一次**生成的初始画像 P0；
随后各 variant 仍用各自的精炼得到 P1；在 ``w1_w2`` 下 **W2_F/L/Q** 反映「同一起点下的不同精炼 + 对 W2 的预测」。若需旧行为（各 variant 自己从 W0 生成 P0），加 ``--initial-p0 per_variant``。

共享 P0 默认写入 ``<output-dir>/shared_gpt_p0/shared_gpt_p0_store.json``：按 ``user_id`` + ``community_id`` 存多条记录（含 ``p0_text``、``window_keys_used`` 等），避免每用户一对文件导致数量爆炸。仍兼容读取旧版 ``p0__uid_*__cid_*.txt``。可用 ``--p0-cache-dir`` 指定目录；``--p0-cache-read`` 优先读缓存跳过 GPT；``--p0-no-disk-cache`` 不写盘（仍可配合显式 ``--p0-cache-dir`` 只读）。

每种画像变体输出一行 JSON（同一 user 三行，便于按 community 聚合）：
  - profile_variant: clasp_dpo | baseline | gpt4o_mini
  - W1_* / W2_*：``w1_w2`` 时为物理 W1、W2 窗得分；``w2_w3`` 时为物理 **W2**、**W3**（字段名不变，兼容现有聚合/绘图）；``w0_w1_w2_p0p1`` 时为 **P0 三窗**、**P1 三窗**的 F/L/Q 算术均值（细粒度见 ``P0_W*_*`` / ``P1_W*_*``）；柱状图脚本对 p0p1 另行读 **P0@W1 / P1@W2** 单窗。
  - slice_eval_mode: w1_w2 | w2_w3 | w0_w1_w2_p0p1
  - slice_initial_p0 / slice_p0_backend：初始画像来源说明
  - action_prompt_include_observed_history：是否向动作模型附带观测历史（见 ``--no-action-prompt-observed-history``）

输入：已窗口化 jsonl（``w1_w2`` / ``w0_w1_w2_p0p1`` 至少 W0..W2；``w2_w3`` 至少 W0..W3），默认扫齐 6 个社区 `community_*.jsonl`。

示例（两画像变体 + 多进程，与 baseline 对比实验类似）：
  python3 -m comparison.run_dpo_profile_slice_eval \\
    --split test --windowed-root output/windowed \\
    --output-dir output/comparison/dpo_profile_slice \\
    --variants baseline,gpt4o_mini \\
    --max-users-per-community 100 --user-processes 5 --user-process-stagger 0.5

默认 ``--initial-p0 shared_gpt``：换用 ``--initial-p0 per_variant`` 可恢复「各 variant 自建 P0」。
若曾用旧逻辑跑满 ``--resume``，更换 ``--initial-p0`` 后建议换 ``--output-dir`` 或删旧 jsonl，以免同文件内混用两种协议。

加 ``--no-action-prompt-observed-history`` 时动作 prompt **不**附带观测到的历史行为块与滑窗历史（与 ``run_baseline_comparison`` 同名开关一致），便于在控制变量下更纯粹对比画像文本对预测的影响。在 ``w0_w1_w2_p0p1`` 下该开关仅作用于 **W1/W2** 预测；**W0 恒不注入**近期动作。

模型约定（与 comparison 窗口链一致）：
  - **动作**：全程 ``cfg.COMPARISON_CLASP_ACTION_VLLM_MODEL``（如 Meta-Llama-3-8B-Instruct-bluesky-sft）。
  - **baseline 变体画像**：``cfg.COMPARISON_BASELINE_VLLM_MODEL``（原始 Instruct 基座，走 PROFILE_API）。
  - **clasp_dpo 变体画像**：``cfg.COMPARISON_CLASP_PROFILE_VLLM_MODEL``。
  - **gpt4o_mini 变体画像**：``cfg.OPENAI_BASE_URL`` + ``cfg.PROFILE_MODEL``（与主流程商用画像一致）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.config as cfg
from src.action_predictor import (
    build_behavior_discrepancies,
    predict_actions_for_window,
)
from src.config import (
    ACTION_PROMPT_HISTORY_MAX_CHARS,
    ALPHA,
    PROFILE_BEHAVIOR_TEXT_MAX_CHARS,
    TEMPERATURE_ACTION,
)
from src.dpo_pipeline import evaluate_predictions
from src.profile_generator import (
    format_behavior_data,
    generate_candidate_profiles,
    generate_initial_profile,
    truncate_behavior_plaintext,
)
from src.profile_generator import _invoke_commercial_profile_llm
from src.prompts import (
    FREE_FORM_PROMPT,
    PROFILE_REFINEMENT_PROMPT,
    SYSTEM_INSTRUCTION_PROFILE,
    SYSTEM_INSTRUCTION_REFINEMENT,
)
from src.scorer import SemanticScorer

from comparison.baseline_resume import filter_users_per_community


PROFILE_VARIANTS = ("clasp_dpo", "baseline", "gpt4o_mini")

# 切片协议：jsonl 中始终写 W1_* / W2_*，语义随 slice_eval_mode 变化（见 --slice-eval-mode）
SLICE_EVAL_MODE_W1_W2 = "w1_w2"
SLICE_EVAL_MODE_W2_W3 = "w2_w3"
SLICE_EVAL_MODE_W0_W1_W2_P0P1 = "w0_w1_w2_p0p1"


@contextmanager
def _action_clasp_scope() -> Any:
    old = cfg.ACTION_API_MODEL
    cfg.ACTION_API_MODEL = cfg.COMPARISON_CLASP_ACTION_VLLM_MODEL
    try:
        yield
    finally:
        cfg.ACTION_API_MODEL = old


@contextmanager
def _profile_vllm_scope(kind: str) -> Any:
    """kind: clasp_dpo | baseline — 切换 vLLM 画像 model id。"""
    old = cfg.PROFILE_API_MODEL
    try:
        if kind == "clasp_dpo":
            cfg.PROFILE_API_MODEL = cfg.COMPARISON_CLASP_PROFILE_VLLM_MODEL
        elif kind == "baseline":
            cfg.PROFILE_API_MODEL = cfg.COMPARISON_BASELINE_VLLM_MODEL
        else:
            raise ValueError(f"_profile_vllm_scope: {kind}")
        yield
    finally:
        cfg.PROFILE_API_MODEL = old


def _observed_suffix(window_key: str, actions: List[Dict]) -> Optional[str]:
    block = f"### Recent behaviors (observed window {window_key})\n" + format_behavior_data(actions)
    if int(ACTION_PROMPT_HISTORY_MAX_CHARS) > 0:
        block = truncate_behavior_plaintext(block, int(ACTION_PROMPT_HISTORY_MAX_CHARS))
    return block or None


def _p0p1_triple_window_predict_args(
    window_index: int,
    keys: List[str],
    ws: List[List[Dict]],
    *,
    inject_recent_w1w2: bool,
) -> Tuple[List[Dict], List[Dict], bool, Optional[str]]:
    """
    w0_w1_w2_p0p1：对单窗构造 predict_actions_for_window 参数。

    - **W0（index=0）**：恒不注入近期动作（history 空、无 profile_suffix、include_observed_history=False）。
    - **W1/W2**：仅当 inject_recent_w1w2 为 True 时与旧 w1_w2 切片一致（上一窗作 history + 观测块）；
      为 False 时与 W0 相同（纯画像 + 目标窗）。
    """
    if window_index == 0:
        return [], ws[0], False, None
    if not inject_recent_w1w2:
        return [], ws[window_index], False, None
    if window_index == 1:
        return ws[0], ws[1], True, _observed_suffix(keys[0], ws[0])
    return ws[1], ws[2], True, _observed_suffix(keys[1], ws[1])


def _mean_three(a: Any, b: Any, c: Any) -> Optional[float]:
    xs: List[float] = []
    for v in (a, b, c):
        if isinstance(v, (int, float)):
            xs.append(float(v))
    return sum(xs) / 3.0 if len(xs) == 3 else None


def _commercial_initial_profile(actions: List[Dict]) -> str:
    behavior_data = format_behavior_data(actions)
    behavior_data = truncate_behavior_plaintext(
        behavior_data, int(PROFILE_BEHAVIOR_TEXT_MAX_CHARS)
    )
    prompt = FREE_FORM_PROMPT.format(
        action_count=len(actions),
        behavior_data=behavior_data,
    )
    return _invoke_commercial_profile_llm(
        SYSTEM_INSTRUCTION_PROFILE,
        prompt,
        max_new_tokens=2048,
        temperature=0.7,
        debug_step="dpo_slice:initial_s0_commercial",
        debug_emit=False,
    )


def _commercial_refine_profile(old_profile: str, discrepancies: str) -> str:
    old_t = truncate_behavior_plaintext(
        old_profile, int(getattr(cfg, "PROFILE_REFINEMENT_OLD_PERSONA_MAX_CHARS", 3500))
    )
    disc_t = truncate_behavior_plaintext(
        discrepancies, int(getattr(cfg, "PROFILE_REFINEMENT_DISCREPANCY_MAX_CHARS", 3500))
    )
    refinement_prompt = PROFILE_REFINEMENT_PROMPT.format(
        old_persona=old_t,
        behavior_discrepancies=disc_t,
    )
    return _invoke_commercial_profile_llm(
        SYSTEM_INSTRUCTION_REFINEMENT,
        refinement_prompt,
        max_new_tokens=2048,
        temperature=float(cfg.TEMPERATURE_PROFILE),
        debug_step="dpo_slice:refine_commercial",
        debug_emit=False,
    )


def _slice_windowed_chain(
    user_record: Dict[str, Any],
    *,
    n: int,
) -> Tuple[Optional[str], Optional[List[str]], Optional[List[List[Dict]]]]:
    """
    取前 n 个按编号排序的窗口 W*。成功返回 (None, keys, [w0, w1, ...])；
    失败返回 (error_code, None, None)。
    """
    windows = user_record.get("windows") or {}
    keys = sorted(
        [k for k in windows if k.startswith("W")],
        key=lambda x: int(x[1:]),
    )
    if len(keys) < n:
        msg = "need_W0_W1_W2" if n == 3 else "need_W0_W1_W2_W3"
        return msg, None, None
    ws = [windows[keys[i]] for i in range(n)]
    return None, keys, ws


def _slice_w0_w1_w2(
    user_record: Dict[str, Any],
) -> Tuple[Optional[str], Optional[List[Dict]], Optional[List[Dict]], Optional[List[Dict]], Optional[List[str]]]:
    """
    从窗口化用户记录取出 W0/W1/W2 与排序后的键。
    失败时返回 (error_code, None, None, None, None)；成功时首项为 None。
    """
    err, keys, ws = _slice_windowed_chain(user_record, n=3)
    if err or ws is None:
        return err or "need_W0_W1_W2", None, None, None, None
    return None, ws[0], ws[1], ws[2], keys


def _safe_p0_cache_segment(value: Any, *, max_len: int = 220) -> str:
    s = str(value).strip().replace("/", "_").replace("\\", "_").replace(":", "_")
    return (s[:max_len] if s else "unknown").strip() or "unknown"


def _p0_cache_paths(cache_dir: Path, uid: Any, cid: Any) -> Tuple[Path, Path]:
    """旧版每用户一对文件路径（仅用于兼容读取）。"""
    base = cache_dir / f"p0__uid_{_safe_p0_cache_segment(uid)}__cid_{_safe_p0_cache_segment(cid, max_len=80)}"
    return base.with_suffix(".txt"), base.with_suffix(".meta.json")


def _p0_store_json_path(cache_dir: Path) -> Path:
    return cache_dir / "shared_gpt_p0_store.json"


def _p0_store_lock_path(cache_dir: Path) -> Path:
    return cache_dir / ".shared_gpt_p0_store.lock"


def _p0_store_entry_key(uid: Any, cid: Any) -> str:
    return f"uid_{_safe_p0_cache_segment(uid)}__cid_{_safe_p0_cache_segment(cid, max_len=80)}"


def _flock_lock(f: Any, *, exclusive: bool) -> None:
    try:
        import fcntl
    except ImportError:
        return
    op = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(f.fileno(), op)


def _flock_unlock(f: Any) -> None:
    try:
        import fcntl
    except ImportError:
        return
    fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def _read_p0_store_body(store_path: Path) -> Dict[str, Any]:
    if not store_path.is_file():
        return {"version": 1, "entries": {}}
    try:
        with store_path.open("r", encoding="utf-8") as sf:
            data = json.load(sf)
    except (json.JSONDecodeError, OSError):
        return {"version": 1, "entries": {}}
    if not isinstance(data, dict):
        return {"version": 1, "entries": {}}
    data.setdefault("version", 1)
    ent = data.get("entries")
    if not isinstance(ent, dict):
        data["entries"] = {}
    return data


def _write_p0_store_atomic(store_path: Path, data: Dict[str, Any]) -> None:
    store_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(
        dir=str(store_path.parent),
        prefix=".shared_gpt_p0_store_",
        suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as tf:
            tf.write(payload)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(tmp, store_path)
    except Exception:
        try:
            if os.path.isfile(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        raise


def _try_load_p0_cache(txt_path: Path) -> Optional[str]:
    """旧版单用户 .txt 缓存。"""
    if not txt_path.is_file():
        return None
    return txt_path.read_text(encoding="utf-8")


def _try_load_p0_from_json_store(cache_dir: Path, uid: Any, cid: Any) -> Optional[str]:
    """优先读聚合 json；若无则读旧版 per-user .txt。"""
    store_p = _p0_store_json_path(cache_dir)
    lock_p = _p0_store_lock_path(cache_dir)
    key = _p0_store_entry_key(uid, cid)
    lock_p.parent.mkdir(parents=True, exist_ok=True)
    hit: Optional[str] = None
    with lock_p.open("a+", encoding="utf-8") as lf:
        _flock_lock(lf, exclusive=False)
        try:
            data = _read_p0_store_body(store_p)
            ent = data.get("entries", {}).get(key)
            if isinstance(ent, dict) and "p0_text" in ent:
                hit = ent["p0_text"] if isinstance(ent["p0_text"], str) else None
        finally:
            _flock_unlock(lf)
    if hit is not None:
        return hit
    txt_p, _ = _p0_cache_paths(cache_dir, uid, cid)
    return _try_load_p0_cache(txt_p)


def _save_p0_to_json_store(
    cache_dir: Path,
    *,
    uid: Any,
    cid: Any,
    p0_text: str,
    user_record: Dict[str, Any],
    window_keys_used: List[str],
) -> None:
    store_p = _p0_store_json_path(cache_dir)
    lock_p = _p0_store_lock_path(cache_dir)
    key = _p0_store_entry_key(uid, cid)
    new_entry = {
        "user_id": user_record.get("user_id"),
        "community_id": user_record.get("community_id"),
        "window_keys_used": window_keys_used[:3],
        "p0_text": p0_text or "",
        "encoding": "utf-8",
        "note": "GPT-4o-mini (cfg.PROFILE_MODEL) 初始画像；聚合于 shared_gpt_p0_store.json",
    }
    lock_p.parent.mkdir(parents=True, exist_ok=True)
    with lock_p.open("a+", encoding="utf-8") as lf:
        _flock_lock(lf, exclusive=True)
        try:
            data = _read_p0_store_body(store_p)
            data["entries"][key] = new_entry
            _write_p0_store_atomic(store_p, data)
        finally:
            _flock_unlock(lf)


def _obtain_shared_gpt_p0(
    user: Dict[str, Any],
    w0: List[Dict],
    keys: List[str],
    *,
    p0_cache_dir: Optional[Path],
    p0_cache_read: bool,
    p0_cache_write: bool,
) -> Tuple[Optional[str], Optional[str]]:
    """
    获取共享 GPT 初始画像：可选先读盘，否则调 API 并写入缓存。
    返回 (p0 正文, error 字符串)；成功时 error 为 None。
    """
    uid, cid = user.get("user_id"), user.get("community_id")
    if p0_cache_read and p0_cache_dir is not None:
        hit = _try_load_p0_from_json_store(p0_cache_dir, uid, cid)
        if hit is not None:
            return hit, None
    try:
        p0 = _commercial_initial_profile(w0)
    except Exception as e:
        return None, f"p0_shared_gpt_failed:{type(e).__name__}: {e}"
    if p0_cache_write and p0_cache_dir is not None:
        _save_p0_to_json_store(
            p0_cache_dir,
            uid=uid,
            cid=cid,
            p0_text=p0,
            user_record=user,
            window_keys_used=keys,
        )
    return p0, None


def _refine_one_base_candidate(
    p0: str,
    discrepancies: str,
    *,
    profile_kind: str,
) -> str:
    """单次精炼、仅 vLLM base 槽位（不调商用 API）。"""
    old_ratio = float(cfg.COMMERCIAL_PROFILE_RATIO)
    old_enable = bool(cfg.ENABLE_COMMERCIAL_PROFILE)
    try:
        cfg.COMMERCIAL_PROFILE_RATIO = 0.0
        cfg.ENABLE_COMMERCIAL_PROFILE = False
        with _profile_vllm_scope(profile_kind):
            cands = generate_candidate_profiles(
                None,
                None,
                p0,
                discrepancies,
                n=1,
                workers=1,
            )
    finally:
        cfg.COMMERCIAL_PROFILE_RATIO = old_ratio
        cfg.ENABLE_COMMERCIAL_PROFILE = old_enable
    if cands and (cands[0] or "").strip():
        return cands[0]
    return p0


def evaluate_user_slice_one_variant(
    user_record: Dict[str, Any],
    profile_variant: str,
    semantic_scorer: SemanticScorer,
    *,
    p0_shared: Optional[str] = None,
    action_prompt_include_observed_history: bool = True,
    slice_eval_mode: str = SLICE_EVAL_MODE_W1_W2,
) -> Dict[str, Any]:
    uid = user_record.get("user_id")
    cid = user_record.get("community_id")
    need_n = 4 if slice_eval_mode == SLICE_EVAL_MODE_W2_W3 else 3
    err, keys, ws = _slice_windowed_chain(user_record, n=need_n)
    if err or ws is None or keys is None:
        return {
            "user_id": uid,
            "community_id": cid,
            "profile_variant": profile_variant,
            "error": err or "need_windows",
            "slice_action_model": str(cfg.COMPARISON_CLASP_ACTION_VLLM_MODEL),
            "slice_initial_p0_arg": "shared_gpt" if p0_shared is not None else "per_variant",
            "action_prompt_include_observed_history": bool(action_prompt_include_observed_history),
            "slice_eval_mode": slice_eval_mode,
        }

    w0, w1, w2 = ws[0], ws[1], ws[2]
    w3 = ws[3] if need_n == 4 else None

    out: Dict[str, Any] = {
        "user_id": uid,
        "community_id": cid,
        "profile_variant": profile_variant,
        "window_keys_used": keys[:need_n],
        "slice_action_model": str(cfg.COMPARISON_CLASP_ACTION_VLLM_MODEL),
        "slice_initial_p0_arg": "shared_gpt" if p0_shared is not None else "per_variant",
        "action_prompt_include_observed_history": bool(action_prompt_include_observed_history),
        "slice_eval_mode": slice_eval_mode,
    }

    try:
        with _action_clasp_scope():
            ih = bool(action_prompt_include_observed_history)
            if p0_shared is not None:
                p0 = p0_shared
                out["slice_initial_p0"] = "shared_gpt4o_mini_once"
                out["slice_p0_backend"] = f"openai:{cfg.PROFILE_MODEL}"
            elif profile_variant == "gpt4o_mini":
                p0 = _commercial_initial_profile(w0)
                out["slice_initial_p0"] = "per_variant"
                out["slice_p0_backend"] = f"openai:{cfg.PROFILE_MODEL}"
            elif profile_variant in ("clasp_dpo", "baseline"):
                with _profile_vllm_scope(profile_variant):
                    p0 = generate_initial_profile(None, None, w0)
                out["slice_initial_p0"] = "per_variant"
                if profile_variant == "baseline":
                    out["slice_p0_backend"] = f"vllm:{cfg.COMPARISON_BASELINE_VLLM_MODEL}"
                else:
                    out["slice_p0_backend"] = f"vllm:{cfg.COMPARISON_CLASP_PROFILE_VLLM_MODEL}"
            else:
                return {**out, "error": f"unknown_profile_variant:{profile_variant}"}

            if slice_eval_mode == SLICE_EVAL_MODE_W0_W1_W2_P0P1:
                inject_w1w2 = ih
                out["slice_w0_never_action_history"] = True
                out["action_history_injection_w1w2"] = bool(inject_w1w2)
                preds_p0_by_w: List[List[Dict]] = []
                for wi in range(3):
                    hist, tgt, use_ih, suff = _p0p1_triple_window_predict_args(
                        wi, keys, ws, inject_recent_w1w2=inject_w1w2
                    )
                    pr = predict_actions_for_window(
                        None,
                        None,
                        p0,
                        hist,
                        tgt,
                        temperature=TEMPERATURE_ACTION,
                        profile_suffix=suff,
                        include_observed_history=use_ih,
                    )
                    preds_p0_by_w.append(pr)
                    f_, l_, q_ = evaluate_predictions(pr, tgt, semantic_scorer, ALPHA)
                    out[f"P0_W{wi}_F"], out[f"P0_W{wi}_L"], out[f"P0_W{wi}_Q"] = f_, l_, q_

                disc1 = build_behavior_discrepancies(preds_p0_by_w[1], w1, w0)
                if profile_variant == "gpt4o_mini":
                    p1 = _commercial_refine_profile(p0, disc1)
                else:
                    p1 = _refine_one_base_candidate(p0, disc1, profile_kind=profile_variant)

                for wi in range(3):
                    hist, tgt, use_ih, suff = _p0p1_triple_window_predict_args(
                        wi, keys, ws, inject_recent_w1w2=inject_w1w2
                    )
                    pr = predict_actions_for_window(
                        None,
                        None,
                        p1,
                        hist,
                        tgt,
                        temperature=TEMPERATURE_ACTION,
                        profile_suffix=suff,
                        include_observed_history=use_ih,
                    )
                    f_, l_, q_ = evaluate_predictions(pr, tgt, semantic_scorer, ALPHA)
                    out[f"P1_W{wi}_F"], out[f"P1_W{wi}_L"], out[f"P1_W{wi}_Q"] = f_, l_, q_

                out["W1_F"] = _mean_three(out["P0_W0_F"], out["P0_W1_F"], out["P0_W2_F"])
                out["W1_L"] = _mean_three(out["P0_W0_L"], out["P0_W1_L"], out["P0_W2_L"])
                out["W1_Q"] = _mean_three(out["P0_W0_Q"], out["P0_W1_Q"], out["P0_W2_Q"])
                out["W2_F"] = _mean_three(out["P1_W0_F"], out["P1_W1_F"], out["P1_W2_F"])
                out["W2_L"] = _mean_three(out["P1_W0_L"], out["P1_W1_L"], out["P1_W2_L"])
                out["W2_Q"] = _mean_three(out["P1_W0_Q"], out["P1_W1_Q"], out["P1_W2_Q"])
                out["profile_p0_chars"] = len(p0 or "")
                out["profile_p1_chars"] = len(p1 or "")
            else:
                preds_w1 = predict_actions_for_window(
                    None,
                    None,
                    p0,
                    w0,
                    w1,
                    temperature=TEMPERATURE_ACTION,
                    profile_suffix=_observed_suffix(keys[0], w0) if ih else None,
                    include_observed_history=ih,
                )
                disc1 = build_behavior_discrepancies(preds_w1, w1, w0)
                if profile_variant == "gpt4o_mini":
                    p1 = _commercial_refine_profile(p0, disc1)
                else:
                    p1 = _refine_one_base_candidate(p0, disc1, profile_kind=profile_variant)

                if slice_eval_mode == SLICE_EVAL_MODE_W1_W2:
                    f1, l1, q1 = evaluate_predictions(preds_w1, w1, semantic_scorer, ALPHA)
                    out["W1_F"], out["W1_L"], out["W1_Q"] = f1, l1, q1

                    preds_w2 = predict_actions_for_window(
                        None,
                        None,
                        p1,
                        w1,
                        w2,
                        temperature=TEMPERATURE_ACTION,
                        profile_suffix=_observed_suffix(keys[1], w1) if ih else None,
                        include_observed_history=ih,
                    )
                    f2, l2, q2 = evaluate_predictions(preds_w2, w2, semantic_scorer, ALPHA)
                    out["W2_F"], out["W2_L"], out["W2_Q"] = f2, l2, q2
                    out["profile_p0_chars"] = len(p0 or "")
                    out["profile_p1_chars"] = len(p1 or "")
                else:
                    preds_on_w2 = predict_actions_for_window(
                        None,
                        None,
                        p1,
                        w1,
                        w2,
                        temperature=TEMPERATURE_ACTION,
                        profile_suffix=_observed_suffix(keys[1], w1) if ih else None,
                        include_observed_history=ih,
                    )
                    f_w2, l_w2, q_w2 = evaluate_predictions(
                        preds_on_w2, w2, semantic_scorer, ALPHA
                    )
                    out["W1_F"], out["W1_L"], out["W1_Q"] = f_w2, l_w2, q_w2

                    disc2 = build_behavior_discrepancies(preds_on_w2, w2, w1)
                    if profile_variant == "gpt4o_mini":
                        p2 = _commercial_refine_profile(p1, disc2)
                    else:
                        p2 = _refine_one_base_candidate(p1, disc2, profile_kind=profile_variant)

                    if w3 is None:
                        return {**out, "error": "need_W0_W1_W2_W3"}
                    preds_on_w3 = predict_actions_for_window(
                        None,
                        None,
                        p2,
                        w2,
                        w3,
                        temperature=TEMPERATURE_ACTION,
                        profile_suffix=_observed_suffix(keys[2], w2) if ih else None,
                        include_observed_history=ih,
                    )
                    f3, l3, q3 = evaluate_predictions(
                        preds_on_w3, w3, semantic_scorer, ALPHA
                    )
                    out["W2_F"], out["W2_L"], out["W2_Q"] = f3, l3, q3
                    out["profile_p0_chars"] = len(p0 or "")
                    out["profile_p1_chars"] = len(p1 or "")
                    out["profile_p2_chars"] = len(p2 or "")

            if profile_variant == "baseline":
                out["slice_profile_backend"] = f"vllm:{cfg.COMPARISON_BASELINE_VLLM_MODEL}"
            elif profile_variant == "clasp_dpo":
                out["slice_profile_backend"] = f"vllm:{cfg.COMPARISON_CLASP_PROFILE_VLLM_MODEL}"
            else:
                out["slice_profile_backend"] = f"openai:{cfg.PROFILE_MODEL}"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _load_completed_variant_keys(path: Path) -> Set[Tuple[str, str, str]]:
    """(user_id, community_id str, profile_variant) 已成功写入。"""
    done: Set[Tuple[str, str, str]] = set()
    if not path.is_file():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("error"):
                continue
            pv = o.get("profile_variant")
            if not pv:
                continue
            done.add((str(o.get("user_id")), str(o.get("community_id")), str(pv)))
    return done


def _print_community_summary(rows: List[Dict[str, Any]], variants: Tuple[str, ...]) -> None:
    """按 community_id × profile_variant 打印 W1_Q / W2_Q 均值。"""
    by_c: Dict[Any, Dict[Tuple[str, str], List[float]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("error"):
            continue
        pv = str(r.get("profile_variant"))
        if pv not in variants:
            continue
        cid = r.get("community_id")
        for step in ("W1", "W2"):
            kq = f"{step}_Q"
            v = r.get(kq)
            if v is not None:
                by_c[cid][(pv, step)].append(float(v))

    print("\n[DpoSlice] ========== 按 community_id 汇总 mean(Q) ==========", flush=True)
    for cid in sorted(by_c.keys(), key=lambda x: str(x)):
        parts = [f"community={cid}"]
        for v in variants:
            for step in ("W1", "W2"):
                xs = by_c[cid].get((v, step), [])
                mean = sum(xs) / len(xs) if xs else None
                tag = f"{v}/{step}_Q"
                parts.append(f"{tag}={mean:.4f}" if mean is not None else f"{tag}=n/a")
        print("  " + " | ".join(parts), flush=True)


def _user_needs_any_variant(
    user: Dict[str, Any],
    variants: Tuple[str, ...],
    completed: Set[Tuple[str, str, str]],
    resume: bool,
) -> bool:
    if not resume:
        return True
    uid = str(user.get("user_id"))
    cid = str(user.get("community_id"))
    return any((uid, cid, pv) not in completed for pv in variants)


def _dpo_slice_user_worker(job: tuple) -> tuple:
    """
    子进程：单用户、多 variant 串行（每进程独立 SemanticScorer，避免 embedder 多线程争用）。

    job: (idx, user, variants, scorer_device, stagger_sec, completed_frozen, resume,
          initial_p0, p0_cache_dir, p0_cache_read, p0_cache_write, action_prompt_include_observed_history,
          slice_eval_mode)
    initial_p0: "shared_gpt" | "per_variant"
    返回 (_, rows, elapsed_sec)（idx 供调试，父进程可忽略）
    """
    (
        idx,
        user,
        variants,
        scorer_device,
        stagger_sec,
        completed_frozen,
        resume,
        initial_p0,
        p0_cache_dir,
        p0_cache_read,
        p0_cache_write,
        action_prompt_include_observed_history,
        slice_eval_mode,
    ) = job
    if stagger_sec > 0:
        time.sleep(float(idx) * float(stagger_sec))
    scorer = SemanticScorer(device=scorer_device)
    uid = user.get("user_id")
    cid = user.get("community_id")
    t0 = time.time()
    rows: List[Dict[str, Any]] = []

    need_n = 4 if slice_eval_mode == SLICE_EVAL_MODE_W2_W3 else 3
    err, keys_list, ws = _slice_windowed_chain(user, n=need_n)
    if err or ws is None:
        for pv in variants:
            key = (str(uid), str(cid), pv)
            if resume and key in completed_frozen:
                continue
            rows.append(
                {
                    "user_id": uid,
                    "community_id": cid,
                    "profile_variant": pv,
                    "error": err or "need_windows",
                    "slice_action_model": str(cfg.COMPARISON_CLASP_ACTION_VLLM_MODEL),
                    "slice_initial_p0_arg": initial_p0,
                    "slice_eval_mode": slice_eval_mode,
                }
            )
        return idx, rows, time.time() - t0
    w0 = ws[0]

    p0_once: Optional[str] = None
    if initial_p0 == "shared_gpt":
        p0_once, err_p0 = _obtain_shared_gpt_p0(
            user,
            w0,
            keys_list or [],
            p0_cache_dir=p0_cache_dir,
            p0_cache_read=bool(p0_cache_read),
            p0_cache_write=bool(p0_cache_write),
        )
        if err_p0 is not None:
            for pv in variants:
                key = (str(uid), str(cid), pv)
                if resume and key in completed_frozen:
                    continue
                rows.append(
                    {
                        "user_id": uid,
                        "community_id": cid,
                        "profile_variant": pv,
                        "error": err_p0,
                        "slice_action_model": str(cfg.COMPARISON_CLASP_ACTION_VLLM_MODEL),
                        "slice_initial_p0_arg": initial_p0,
                        "slice_eval_mode": slice_eval_mode,
                    }
                )
            return idx, rows, time.time() - t0

    for pv in variants:
        key = (str(uid), str(cid), pv)
        if resume and key in completed_frozen:
            continue
        rows.append(
            evaluate_user_slice_one_variant(
                user,
                pv,
                scorer,
                p0_shared=p0_once,
                action_prompt_include_observed_history=action_prompt_include_observed_history,
                slice_eval_mode=slice_eval_mode,
            )
        )
    return idx, rows, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO 画像切片：P0→W1→精炼→P1→W2，三画像源对比")
    ap.add_argument("--split", default="test", help="windowed 子目录名，如 test")
    ap.add_argument(
        "--windowed-root",
        type=Path,
        default=ROOT / "output" / "windowed",
        help="窗口化数据根目录",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "comparison" / "dpo_profile_slice",
        help="输出目录（写入单个 jsonl）",
    )
    ap.add_argument(
        "--file-glob",
        default="community_*.jsonl",
        help="在 <windowed-root>/<split>/ 下匹配的 glob",
    )
    ap.add_argument(
        "--variants",
        default=",".join(PROFILE_VARIANTS),
        help="逗号分隔：clasp_dpo,baseline,gpt4o_mini 的子集",
    )
    ap.add_argument("--max-users", type=int, default=None, help="全局最多用户数（跨文件合计）")
    ap.add_argument(
        "--max-users-per-community",
        type=int,
        default=0,
        help="每社区最多用户数；0=不限制（默认扫满各文件）",
    )
    ap.add_argument("--scorer-device", default="cpu")
    ap.add_argument("--resume", action="store_true", help="跳过已成功写入的 (user, community, variant)")
    ap.add_argument(
        "--user-processes",
        type=int,
        default=1,
        help=(
            "并行处理用户数；>1 时启多进程（每进程独立 SemanticScorer，"
            "与 run_baseline_comparison 的 --user-processes 类似）。默认 1=串行。"
        ),
    )
    ap.add_argument(
        "--user-process-stagger",
        type=float,
        default=0.5,
        help="多进程时按子进程序号错开启动的秒数，减轻画像/动作 API 洪峰",
    )
    ap.add_argument(
        "--initial-p0",
        choices=("shared_gpt", "per_variant"),
        default="shared_gpt",
        help=(
            "shared_gpt（默认）：每用户只调一次商用 GPT 初始画像，全 variant 共用后再各自精炼并评 W2；"
            "per_variant：各 variant 自己从 W0 生成 P0（旧行为）"
        ),
    )
    ap.add_argument(
        "--p0-cache-dir",
        type=Path,
        default=None,
        help="共享 GPT P0 落盘目录；默认 <output-dir>/shared_gpt_p0，写入 shared_gpt_p0_store.json（仅 initial-p0=shared_gpt 且未 --p0-no-disk-cache）",
    )
    ap.add_argument(
        "--p0-cache-read",
        action="store_true",
        help="若 shared_gpt_p0_store.json（或旧版 per-user .txt）中已有该用户 P0 则跳过 GPT、直接读取（须 shared_gpt 且配置了可解析的缓存路径）",
    )
    ap.add_argument(
        "--p0-no-disk-cache",
        action="store_true",
        help="不把新生成的 P0 写入磁盘；仍可用 --p0-cache-dir + --p0-cache-read 读取已有缓存",
    )
    ap.add_argument(
        "--no-action-prompt-observed-history",
        action="store_true",
        help=(
            "动作 prompt 不附带观测历史：不拼画像后的本窗行为块，且 Recent user actions 为空占位 "
            "（与 run_baseline_comparison 的同名开关一致）；便于控制变量、更纯粹对比画像对预测的影响。"
            "在 w0_w1_w2_p0p1 下仅关闭 **W1/W2** 上的注入；**W0 在该模式下本就不注入**。"
        ),
    )
    ap.add_argument(
        "--slice-eval-mode",
        choices=(
            SLICE_EVAL_MODE_W1_W2,
            SLICE_EVAL_MODE_W2_W3,
            SLICE_EVAL_MODE_W0_W1_W2_P0P1,
        ),
        default=SLICE_EVAL_MODE_W1_W2,
        help=(
            "w1_w2（默认）：P0→评 W1→精炼 P1→评 W2，写入 W1_* / W2_*。"
            "w2_w3：需 W0..W3 四窗；仍用 W1 误差精炼 P1，物理 W2 得分写入 W1_*（柱状图基线柱语义），"
            "再用 W2 误差精炼 P2→物理 W3 写入 W2_*；输出 …_w2w3.jsonl，勿与 w1_w2 混用 --resume。"
            "w0_w1_w2_p0p1：P0 评 W0/W1/W2，按 W1 误差精炼 P1 后再评三窗；另写 P0_W*_*/P1_W*_；"
            "W1_* / W2_* 写入 P0/P1 各自三窗均值（粗看全链路）；柱状图 ``plot_dpo_profile_slice_radar`` 对 p0p1 聚合 **P0_W1_*、P1_W2_***（单物理窗）。W0 从不注入动作历史；"
            "W1/W2 是否注入由 --no-action-prompt-observed-history 控制（默认注入）。"
            "输出 …_w0w1w2_p0p1.jsonl。"
        ),
    )
    args = ap.parse_args()

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    bad = [v for v in variants if v not in PROFILE_VARIANTS]
    if bad:
        print(f"[DpoSlice] 未知 variant: {bad}，可选: {PROFILE_VARIANTS}", flush=True)
        sys.exit(1)

    split_dir = (Path(args.windowed_root) / args.split).resolve()
    if not split_dir.is_dir():
        print(f"[DpoSlice] 无目录: {split_dir}", flush=True)
        sys.exit(1)
    files = sorted(split_dir.glob(args.file_glob))
    if not files:
        print(f"[DpoSlice] 无匹配文件: {split_dir}/{args.file_glob}", flush=True)
        sys.exit(1)

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.slice_eval_mode == SLICE_EVAL_MODE_W2_W3:
        out_name = f"dpo_profile_slice_{args.split}_w2w3.jsonl"
    elif args.slice_eval_mode == SLICE_EVAL_MODE_W0_W1_W2_P0P1:
        out_name = f"dpo_profile_slice_{args.split}_w0w1w2_p0p1.jsonl"
    else:
        out_name = f"dpo_profile_slice_{args.split}_contiguous.jsonl"
    out_path = out_dir / out_name

    p0_cache_dir_resolved: Optional[Path] = None
    p0_cache_write = not bool(args.p0_no_disk_cache)
    if args.initial_p0 == "shared_gpt":
        if args.p0_cache_dir is not None:
            p0_cache_dir_resolved = Path(args.p0_cache_dir).resolve()
            p0_cache_dir_resolved.mkdir(parents=True, exist_ok=True)
        elif p0_cache_write:
            p0_cache_dir_resolved = (out_dir / "shared_gpt_p0").resolve()
            p0_cache_dir_resolved.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    users_total = 0
    for fp in files:
        with fp.open("r", encoding="utf-8") as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    u = json.loads(line)
                except json.JSONDecodeError:
                    continue
                all_rows.append(u)
                users_total += 1
                if args.max_users is not None and users_total >= int(args.max_users):
                    break
        if args.max_users is not None and users_total >= int(args.max_users):
            break

    if int(args.max_users_per_community) > 0:
        users_in = filter_users_per_community(all_rows, int(args.max_users_per_community))
    else:
        users_in = all_rows

    print(
        f"[DpoSlice] 输入文件数={len(files)} 读取行={len(all_rows)} "
        f"评测用户={len(users_in)} variants={variants} slice-eval-mode={args.slice_eval_mode} -> {out_path}",
        flush=True,
    )
    print(
        f"[DpoSlice] 动作 API 固定为 COMPARISON_CLASP_ACTION_VLLM_MODEL="
        f"{cfg.COMPARISON_CLASP_ACTION_VLLM_MODEL}",
        flush=True,
    )
    if "baseline" in variants or "clasp_dpo" in variants:
        print(
            f"[DpoSlice] vLLM 画像：baseline 用 COMPARISON_BASELINE_VLLM_MODEL="
            f"{cfg.COMPARISON_BASELINE_VLLM_MODEL}；clasp_dpo 用 "
            f"{cfg.COMPARISON_CLASP_PROFILE_VLLM_MODEL}",
            flush=True,
        )
    if "gpt4o_mini" in variants:
        print(
            f"[DpoSlice] gpt4o_mini 画像：OpenAI 兼容 API model={cfg.PROFILE_MODEL} base={cfg.OPENAI_BASE_URL}",
            flush=True,
        )
    print(
        f"[DpoSlice] 初始画像 P0 模式: {args.initial_p0} "
        f"（shared_gpt=全 variant 共用一次 GPT 初始画像；per_variant=各 variant 自建 P0）",
        flush=True,
    )
    if args.initial_p0 == "shared_gpt":
        if p0_cache_dir_resolved is not None:
            print(
                f"[DpoSlice] 共享 P0 缓存目录: {p0_cache_dir_resolved}  "
                f"store={_p0_store_json_path(p0_cache_dir_resolved).name}  "
                f"write_disk={p0_cache_write}  --p0-cache-read={bool(args.p0_cache_read)}",
                flush=True,
            )
        else:
            print(
                "[DpoSlice] 共享 P0 不写盘（已加 --p0-no-disk-cache 且未指定 --p0-cache-dir）；"
                "仅 API 生成、无法从默认路径读缓存",
                flush=True,
            )

    if args.no_action_prompt_observed_history:
        print(
            "[DpoSlice] --no-action-prompt-observed-history：动作 prompt 不附带观测历史（与 run_baseline_comparison 一致）",
            flush=True,
        )
    if args.slice_eval_mode == SLICE_EVAL_MODE_W2_W3:
        print(
            "[DpoSlice] slice-eval-mode=w2_w3：每行 W1_*=物理 W2、W2_*=物理 W3；输入需至少四窗；"
            f"输出 {out_name}",
            flush=True,
        )
    elif args.slice_eval_mode == SLICE_EVAL_MODE_W0_W1_W2_P0P1:
        print(
            "[DpoSlice] slice-eval-mode=w0_w1_w2_p0p1：P0/P1 各评 W0–W2；"
            "W0 恒不注入动作历史；W1/W2 注入="
            f"{not bool(args.no_action_prompt_observed_history)}；"
            f"输出 {out_name}",
            flush=True,
        )

    if not args.resume and out_path.is_file():
        # 覆盖式重来：清空内容、保留路径（不 unlink）；写入阶段仍用 open("a")。
        out_path.open("w", encoding="utf-8").close()
        print(f"[DpoSlice] 非 --resume：已覆盖清空主输出 {out_path}", flush=True)

    completed = _load_completed_variant_keys(out_path) if args.resume else set()
    if args.resume and completed:
        print(f"[DpoSlice] --resume：已从输出加载成功键 {len(completed)} 条", flush=True)

    users_work = [
        u
        for u in users_in
        if _user_needs_any_variant(u, variants, completed, args.resume)
    ]
    n_work = len(users_work)
    print(f"[DpoSlice] 待评估用户: {n_work} / {len(users_in)}", flush=True)

    written = 0
    procs = max(1, int(args.user_processes))

    if n_work == 0:
        print("[DpoSlice] 无待办用户。", flush=True)
    elif procs <= 1:
        scorer = SemanticScorer(device=args.scorer_device)
        with out_path.open("a", encoding="utf-8") as fout:
            for u in users_work:
                uid = u.get("user_id")
                cid = u.get("community_id")
                need_n = (
                    4 if args.slice_eval_mode == SLICE_EVAL_MODE_W2_W3 else 3
                )
                err_u, keys_u, ws_u = _slice_windowed_chain(u, n=need_n)
                w0_u = ws_u[0] if (not err_u and ws_u is not None) else None
                p0_once: Optional[str] = None
                if args.initial_p0 == "shared_gpt" and w0_u is not None:
                    p0_once, err_gpt = _obtain_shared_gpt_p0(
                        u,
                        w0_u,
                        keys_u or [],
                        p0_cache_dir=p0_cache_dir_resolved,
                        p0_cache_read=bool(args.p0_cache_read),
                        p0_cache_write=p0_cache_write,
                    )
                    if err_gpt is not None:
                        for pv in variants:
                            key = (str(uid), str(cid), pv)
                            if args.resume and key in completed:
                                continue
                            rec = {
                                "user_id": uid,
                                "community_id": cid,
                                "profile_variant": pv,
                                "error": err_gpt,
                                "slice_action_model": str(cfg.COMPARISON_CLASP_ACTION_VLLM_MODEL),
                                "slice_initial_p0_arg": args.initial_p0,
                                "slice_eval_mode": args.slice_eval_mode,
                            }
                            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                            fout.flush()
                            written += 1
                        continue
                elif err_u or ws_u is None:
                    for pv in variants:
                        key = (str(uid), str(cid), pv)
                        if args.resume and key in completed:
                            continue
                        rec = {
                            "user_id": uid,
                            "community_id": cid,
                            "profile_variant": pv,
                            "error": err_u or "need_windows",
                            "slice_action_model": str(cfg.COMPARISON_CLASP_ACTION_VLLM_MODEL),
                            "slice_initial_p0_arg": args.initial_p0,
                            "slice_eval_mode": args.slice_eval_mode,
                        }
                        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        fout.flush()
                        written += 1
                    continue

                for pv in variants:
                    key = (str(uid), str(cid), pv)
                    if args.resume and key in completed:
                        continue
                    rec = evaluate_user_slice_one_variant(
                        u,
                        pv,
                        scorer,
                        p0_shared=p0_once,
                        action_prompt_include_observed_history=not bool(
                            args.no_action_prompt_observed_history
                        ),
                        slice_eval_mode=args.slice_eval_mode,
                    )
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    written += 1
                    if not rec.get("error"):
                        completed.add(key)
    else:
        eff = min(procs, n_work)
        completed_frozen = frozenset(completed)
        action_ih = not bool(args.no_action_prompt_observed_history)
        jobs = [
            (
                i,
                u,
                variants,
                args.scorer_device,
                float(args.user_process_stagger),
                completed_frozen,
                args.resume,
                args.initial_p0,
                p0_cache_dir_resolved,
                bool(args.p0_cache_read),
                p0_cache_write,
                action_ih,
                args.slice_eval_mode,
            )
            for i, u in enumerate(users_work)
        ]
        print(
            f"[DpoSlice] 多进程：{eff} workers，错开 {args.user_process_stagger}s/进程",
            flush=True,
        )
        run_start = time.time()
        done_n = 0
        with out_path.open("a", encoding="utf-8") as fout, ProcessPoolExecutor(
            max_workers=eff
        ) as pool:
            futs = {pool.submit(_dpo_slice_user_worker, j): j[0] for j in jobs}
            for fut in as_completed(futs):
                _idx, rows, user_elapsed = fut.result()
                done_n += 1
                for rec in rows:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    if not rec.get("error"):
                        completed.add(
                            (
                                str(rec.get("user_id")),
                                str(rec.get("community_id")),
                                str(rec.get("profile_variant")),
                            )
                        )
                fout.flush()
                batch_elapsed = time.time() - run_start
                avg = batch_elapsed / done_n if done_n else 0.0
                eta = avg * (n_work - done_n) if done_n < n_work else 0.0
                uid_show = rows[0].get("user_id", "?") if rows else "?"
                print(
                    f"[DpoSlice] [{done_n}/{n_work}] user={uid_show} "
                    f"本用户={user_elapsed:.1f}s 平均={avg:.1f}s/人 "
                    f"ETA≈{int(eta // 60)}m{int(eta % 60)}s",
                    flush=True,
                )

    print(f"[DpoSlice] 本 run 新写入行数: {written}", flush=True)

    # 重读输出文件做汇总（含历史行）
    final_rows: List[Dict[str, Any]] = []
    if out_path.is_file():
        with out_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    final_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    _print_community_summary(final_rows, variants)


if __name__ == "__main__":
    main()
