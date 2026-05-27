#!/usr/bin/env python3
"""
Clasp DPO Persona Validity · Slice Evaluation (non-full window chain)

Default evaluation: "P0 builds persona using W0 → predict W1 → refine by error to get P1 → predict W2" two hops.
With ``--slice-eval-mode w0_w1_w2_p0p1``: P0 evaluates **W0/W1/W2** separately, then refines using **W1** prediction error to get P1, P1 evaluates three windows;
**W0 never** injects recent behaviors to action model (history and observed block both empty); **W1/W2**: without ``--no-action-prompt-observed-history`` consistent with old ``w1_w2`` slice (previous window history + observed block),
with that parameter W1/W2 also don't inject. Write ``P0_W0_F``…``P1_W2_Q`` inline, fill ``W1_*``/``W2_*`` as arithmetic mean of **P0 / P1 three-window** scores (for rough full-path view);
**bar chart** ``plot_dpo_profile_slice_radar`` for p0p1 success rows aggregates **``P0_W1_*`` (P0@W1)** and **``P1_W2_*`` (P1@W2)** single-window scores.
With ``--slice-eval-mode w2_w3`` need four consecutive windows: still refine P1 using W1 error, evaluate **physical W2** write to **W1_*** (semantically aligned with bar chart "baseline bar"),
then refine P2 using W2 error, evaluate **physical W3** write to **W2_***; output filename ``…_w2w3.jsonl``, don't mix with ``w1_w2`` in same jsonl with ``--resume``.

Action side uniformly uses **Clasp action checkpoint** (consistent with clasp_online in window_chain),
only switches **persona refinement (P0→P1)** backend, facilitating comparison of DPO persona vs base vLLM vs gpt-4o-mini.

**Default (``--initial-p0 shared_gpt``)**: three variants **share same** initial persona generated **once** by **GPT-4o-mini** (``cfg.PROFILE_MODEL`` + commercial API) on W0;
subsequently each variant still refines to get P1; under ``w1_w2`` **W2_F/L/Q** reflects "different refinement from same starting point + prediction on W2".
For old behavior (each variant generates P0 from W0 itself), add ``--initial-p0 per_variant``.

Shared P0 default writes to ``<output-dir>/shared_gpt_p0/shared_gpt_p0_store.json``: multiple records per ``user_id`` + ``community_id`` (containing ``p0_text``, ``window_keys_used`` etc),
avoiding explosion of per-user file pairs. Still compatible reading old ``p0__uid_*__cid_*.txt``. Can specify directory with ``--p0-cache-dir``;
``--p0-cache-read`` prioritizes reading cache skipping GPT; ``--p0-no-disk-cache`` doesn't write disk (still compatible with explicit ``--p0-cache-dir`` read-only).

Each persona variant outputs one JSON line (three lines per user, convenient for per-community aggregation):
  - profile_variant: clasp_dpo | baseline | gpt4o_mini
  - W1_* / W2_*: under ``w1_w2`` physical W1, W2 window scores; under ``w2_w3`` physical **W2**, **W3** (field names unchanged, compatible with existing aggregation/plotting);
    under ``w0_w1_w2_p0p1`` arithmetic mean of **P0 three-window** and **P1 three-window** F/L/Q (fine-grained see ``P0_W*_*`` / ``P1_W*_*``);
    bar chart script for p0p1 separately reads **P0@W1 / P1@W2** single-window.
  - slice_eval_mode: w1_w2 | w2_w3 | w0_w1_w2_p0p1
  - slice_initial_p0 / slice_p0_backend: initial persona source description
  - action_prompt_include_observed_history: whether action model includes observed history (see ``--no-action-prompt-observed-history``)

Input: windowed jsonl (``w1_w2`` / ``w0_w1_w2_p0p1`` at least W0..W2; ``w2_w3`` at least W0..W3), default scans all 6 communities `community_*.jsonl`.

Example (two persona variants + multi-process, similar to baseline comparison experiment):
  python3 -m comparison.run_dpo_profile_slice_eval \\
    --split test --windowed-root output/windowed \\
    --output-dir output/comparison/dpo_profile_slice \\
    --variants baseline,gpt4o_mini \\
    --max-users-per-community 100 --user-processes 5 --user-process-stagger 0.5

Default ``--initial-p0 shared_gpt``: use ``--initial-p0 per_variant`` to restore "each variant builds P0".
If previously ran full ``--resume`` with old logic, after switching ``--initial-p0`` recommend changing ``--output-dir`` or deleting old jsonl to avoid mixing two protocols in same file.

With ``--no-action-prompt-observed-history`` action prompt **doesn't** include observed behavior block and sliding-window history (consistent with same-name switch in ``run_baseline_comparison``),
facilitating purer comparison of persona text impact on prediction under controlled variables. Under ``w0_w1_w2_p0p1`` this switch only affects **W1/W2** prediction; **W0 never injects** recent actions.

Model conventions (consistent with comparison window chain):
  - **Action**: throughout ``cfg.COMPARISON_CLASP_ACTION_VLLM_MODEL`` (e.g. Meta-Llama-3.1-8B-Instruct-bluesky-sft).
  - **baseline variant persona**: ``cfg.COMPARISON_BASELINE_VLLM_MODEL`` (original Instruct base, goes through PROFILE_API).
  - **clasp_dpo variant persona**: ``cfg.COMPARISON_CLASP_PROFILE_VLLM_MODEL``.
  - **gpt4o_mini variant persona**: ``cfg.OPENAI_BASE_URL`` + ``cfg.PROFILE_MODEL`` (consistent with main flow commercial persona).
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


PROFILE_VARIANTS = ("clasp_dpo", "baseline", "gpt4o_mini", "incremental_persona", "regeneration_persona")

# Slice protocol: always write W1_* / W2_* in jsonl, semantics vary by slice_eval_mode (see --slice-eval-mode)
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
def _action_base_scope() -> Any:
    """Switch to base action model (for incremental_persona / regeneration_persona)."""
    old = cfg.ACTION_API_MODEL
    cfg.ACTION_API_MODEL = cfg.COMPARISON_BASELINE_VLLM_MODEL
    try:
        yield
    finally:
        cfg.ACTION_API_MODEL = old


@contextmanager
def _profile_vllm_scope(kind: str) -> Any:
    """kind: clasp_dpo | baseline — switch vLLM persona model id."""
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
    w0_w1_w2_p0p1: construct predict_actions_for_window parameters for single window.

    - **W0 (index=0)**: never inject recent actions (history empty, no profile_suffix, include_observed_history=False).
    - **W1/W2**: only when inject_recent_w1w2 is True consistent with old w1_w2 slice (previous window as history + observed block);
      when False same as W0 (pure persona + target window).
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
    Get first n windows sorted by number W*. Success returns (None, keys, [w0, w1, ...]);
    failure returns (error_code, None, None).
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
    Extract W0/W1/W2 and sorted keys from windowed user record.
    Failure returns (error_code, None, None, None, None); success first item is None.
    """
    err, keys, ws = _slice_windowed_chain(user_record, n=3)
    if err or ws is None:
        return err or "need_W0_W1_W2", None, None, None, None
    return None, ws[0], ws[1], ws[2], keys


def _safe_p0_cache_segment(value: Any, *, max_len: int = 220) -> str:
    s = str(value).strip().replace("/", "_").replace("\\", "_").replace(":", "_")
    return (s[:max_len] if s else "unknown").strip() or "unknown"


def _p0_cache_paths(cache_dir: Path, uid: Any, cid: Any) -> Tuple[Path, Path]:
    """Old version per-user file pair paths (only for backward compatibility reading)."""
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
    """Old version single-user .txt cache."""
    if not txt_path.is_file():
        return None
    return txt_path.read_text(encoding="utf-8")


def _try_load_p0_from_json_store(cache_dir: Path, uid: Any, cid: Any) -> Optional[str]:
    """Prioritize reading aggregated json; if not, read old per-user .txt."""
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
        "note": "GPT-4o-mini (cfg.PROFILE_MODEL) initial persona; aggregated in shared_gpt_p0_store.json",
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
    Obtain shared GPT initial persona: optionally read disk first, otherwise call API and write to cache.
    Returns (p0 text, error string); success when error is None.
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
    """Single refinement, vLLM base slot only (don't call commercial API)."""
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


def _incremental_refine_block(hist_actions: List[Dict], window_key: str) -> str:
    """Incremental persona: no prediction error, only driven by current window actual behaviors."""
    return (
        "(Incremental persona update: no predicted-vs-actual errors.)\n"
        f"Align the persona with these **actual behaviors in {window_key}**:\n\n"
        + format_behavior_data(hist_actions)
    )


def evaluate_user_slice_one_variant(
    user_record: Dict[str, Any],
    profile_variant: str,
    semantic_scorer: SemanticScorer,
    *,
    p0_shared: Optional[str] = None,
    action_prompt_include_observed_history: bool = True,
    slice_eval_mode: str = SLICE_EVAL_MODE_W1_W2,
    save_profile_text: bool = False,
    force_clasp_action: bool = False,
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
        "slice_initial_p0_arg": "shared_gpt" if p0_shared is not None else "per_variant",
        "action_prompt_include_observed_history": bool(action_prompt_include_observed_history),
        "slice_eval_mode": slice_eval_mode,
    }

    try:
        # Unified use of Clasp action model (all variants share)
        action_scope = _action_clasp_scope()
        out["slice_action_model"] = str(cfg.COMPARISON_CLASP_ACTION_VLLM_MODEL)

        with action_scope:
            ih = bool(action_prompt_include_observed_history)
            if p0_shared is not None:
                p0 = p0_shared
                out["slice_initial_p0"] = "shared_gpt4o_mini_once"
                out["slice_p0_backend"] = f"openai:{cfg.PROFILE_MODEL}"
            elif profile_variant == "gpt4o_mini":
                p0 = _commercial_initial_profile(w0)
                out["slice_initial_p0"] = "per_variant"
                out["slice_p0_backend"] = f"openai:{cfg.PROFILE_MODEL}"
            elif profile_variant in ("clasp_dpo", "baseline", "incremental_persona", "regeneration_persona"):
                # incremental_persona and regeneration_persona use baseline model
                kind = "baseline" if profile_variant in ("incremental_persona", "regeneration_persona") else profile_variant
                with _profile_vllm_scope(kind):
                    p0 = generate_initial_profile(None, None, w0)
                out["slice_initial_p0"] = "per_variant"
                if kind == "baseline":
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
                    out["slice_profile_backend"] = f"openai:{cfg.PROFILE_MODEL}"
                elif profile_variant == "incremental_persona":
                    # Incremental persona: refine using W1 actual behaviors (no error signal)
                    incr_block = _incremental_refine_block(w1, keys[1])
                    p1 = _refine_one_base_candidate(p0, incr_block, profile_kind="baseline")
                    out["slice_profile_backend"] = f"vllm:{cfg.COMPARISON_BASELINE_VLLM_MODEL}"
                elif profile_variant == "regeneration_persona":
                    # Regeneration persona: regenerate using all observed behaviors from W0+W1
                    all_actions = w0 + w1
                    with _profile_vllm_scope("baseline"):
                        p1 = generate_initial_profile(None, None, all_actions)
                    out["slice_profile_backend"] = f"vllm:{cfg.COMPARISON_BASELINE_VLLM_MODEL}"
                else:
                    p1 = _refine_one_base_candidate(p0, disc1, profile_kind=profile_variant)
                    if profile_variant == "baseline":
                        out["slice_profile_backend"] = f"vllm:{cfg.COMPARISON_BASELINE_VLLM_MODEL}"
                    else:
                        out["slice_profile_backend"] = f"vllm:{cfg.COMPARISON_CLASP_PROFILE_VLLM_MODEL}"

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
                elif profile_variant == "incremental_persona":
                    # Incremental persona: refine using W1 actual behaviors (no error signal)
                    incr_block = _incremental_refine_block(w1, keys[1])
                    p1 = _refine_one_base_candidate(p0, incr_block, profile_kind="baseline")
                elif profile_variant == "regeneration_persona":
                    # Regeneration persona: regenerate using all observed behaviors from W0+W1
                    all_actions = w0 + w1
                    with _profile_vllm_scope("baseline"):
                        p1 = generate_initial_profile(None, None, all_actions)
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
                    elif profile_variant == "incremental_persona":
                        # Incremental persona: refine using W2 actual behaviors (no error signal)
                        incr_block = _incremental_refine_block(w2, keys[2])
                        p2 = _refine_one_base_candidate(p1, incr_block, profile_kind="baseline")
                    elif profile_variant == "regeneration_persona":
                        # Regeneration persona: regenerate using all observed behaviors from W0+W1+W2
                        all_actions = w0 + w1 + w2
                        with _profile_vllm_scope("baseline"):
                            p2 = generate_initial_profile(None, None, all_actions)
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

            # Save persona text (optional, for analysis)
            if save_profile_text:
                out["profile_p0_text"] = p0 if 'p0' in locals() else None
                out["profile_p1_text"] = p1 if 'p1' in locals() else None
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def _load_completed_variant_keys(path: Path) -> Set[Tuple[str, str, str]]:
    """(user_id, community_id str, profile_variant) successfully written."""
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
    """Print mean W1_Q / W2_Q by community_id × profile_variant."""
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

    print("\n[DpoSlice] ========== Summary mean(Q) by community_id ==========", flush=True)
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
    Subprocess: single user, multiple variants serial (each process independent SemanticScorer to avoid embedder multi-thread contention).

    job: (idx, user, variants, scorer_device, stagger_sec, completed_frozen, resume,
          initial_p0, p0_cache_dir, p0_cache_read, p0_cache_write, action_prompt_include_observed_history,
          slice_eval_mode, save_profile_text, force_clasp_action)
    initial_p0: "shared_gpt" | "per_variant"
    Returns (_, rows, elapsed_sec) (idx for debugging, parent can ignore)
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
        save_profile_text,
        force_clasp_action,
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
                save_profile_text=save_profile_text,
                force_clasp_action=force_clasp_action,
            )
        )
    return idx, rows, time.time() - t0


def main() -> None:
    ap = argparse.ArgumentParser(description="DPO Persona Slice: P0→W1→refine→P1→W2, three persona source comparison")
    ap.add_argument("--split", default="test", help="windowed subdirectory name, e.g. test")
    ap.add_argument(
        "--windowed-root",
        type=Path,
        default=ROOT / "output" / "windowed",
        help="Windowed data root directory",
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "output" / "comparison" / "dpo_profile_slice",
        help="Output directory (write single jsonl)",
    )
    ap.add_argument(
        "--file-glob",
        default="community_*.jsonl",
        help="Glob pattern to match under <windowed-root>/<split>/",
    )
    ap.add_argument(
        "--variants",
        default=",".join(PROFILE_VARIANTS),
        help="Comma-separated: subset of clasp_dpo,baseline,gpt4o_mini",
    )
    ap.add_argument("--max-users", type=int, default=None, help="Global max users (cross-file total)")
    ap.add_argument(
        "--max-users-per-community",
        type=int,
        default=0,
        help="Max users per community; 0=no limit (default scan full each file)",
    )
    ap.add_argument("--scorer-device", default="cpu")
    ap.add_argument("--resume", action="store_true", help="Skip already successfully written (user, community, variant)")
    ap.add_argument(
        "--append",
        action="store_true",
        help="Append mode: don't clear existing data, suitable for multi-batch runs with different variants then merge",
    )
    ap.add_argument(
        "--user-processes",
        type=int,
        default=1,
        help=(
            "Parallel user processing; >1 enables multi-process (each process independent SemanticScorer, "
            "similar to run_baseline_comparison --user-processes). Default 1=serial."
        ),
    )
    ap.add_argument(
        "--user-process-stagger",
        type=float,
        default=0.5,
        help="Multi-process startup stagger seconds by subprocess index, reduce persona/action API burst",
    )
    ap.add_argument(
        "--initial-p0",
        choices=("shared_gpt", "per_variant"),
        default="shared_gpt",
        help=(
            "shared_gpt (default): call commercial GPT initial persona once per user, all variants share then each refines and evaluates W2; "
            "per_variant: each variant generates P0 from W0 (old behavior)"
        ),
    )
    ap.add_argument(
        "--p0-cache-dir",
        type=Path,
        default=None,
        help="Shared GPT P0 disk directory; default <output-dir>/shared_gpt_p0, write shared_gpt_p0_store.json (only initial-p0=shared_gpt and not --p0-no-disk-cache)",
    )
    ap.add_argument(
        "--p0-cache-read",
        action="store_true",
        help="If shared_gpt_p0_store.json (or old per-user .txt) already has user P0 skip GPT, read directly (requires shared_gpt and configured parseable cache path)",
    )
    ap.add_argument(
        "--p0-no-disk-cache",
        action="store_true",
        help="Don't write newly generated P0 to disk; still can use --p0-cache-dir + --p0-cache-read to read existing cache",
    )
    ap.add_argument(
        "--no-action-prompt-observed-history",
        action="store_true",
        help=(
            "Action prompt doesn't include observed history: don't concatenate current window behavior block after persona, "
            "and Recent user actions is empty placeholder "
            "(consistent with same-name switch in run_baseline_comparison); facilitates controlled variable comparison, purer comparison of persona impact on prediction. "
            "Under w0_w1_w2_p0p1 only disables injection on **W1/W2**; **W0 in this mode never injects anyway**."
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
            "w1_w2 (default): P0→evaluate W1→refine P1→evaluate W2, write W1_* / W2_*. "
            "w2_w3: need W0..W3 four windows; still refine P1 using W1 error, physical W2 score write to W1_* (bar chart baseline bar semantic), "
            "then refine P2 using W2 error→physical W3 write to W2_*; output …_w2w3.jsonl, don't mix with w1_w2 in --resume. "
            "w0_w1_w2_p0p1: P0 evaluate W0/W1/W2, refine P1 by W1 error then evaluate three windows; also write P0_W*_*/P1_W*_*; "
            "W1_* / W2_* write P0/P1 three-window means (rough full-path view); bar chart ``plot_dpo_profile_slice_radar`` for p0p1 aggregates **P0_W1_*, P1_W2_*** (single physical window). W0 never injects action history; "
            "W1/W2 injection controlled by --no-action-prompt-observed-history (default inject). "
            "Output …_w0w1w2_p0p1.jsonl."
        ),
    )
    ap.add_argument(
        "--save-profile-text",
        action="store_true",
        help="Save persona text in output JSONL (p0_text, p1_text) for subsequent analysis",
    )
    ap.add_argument(
        "--force-clasp-action",
        action="store_true",
        help="Force all variants (including incremental_persona and regeneration_persona) use Clasp action model",
    )
    args = ap.parse_args()

    variants = tuple(v.strip() for v in args.variants.split(",") if v.strip())
    bad = [v for v in variants if v not in PROFILE_VARIANTS]
    if bad:
        print(f"[DpoSlice] Unknown variants: {bad}, available: {PROFILE_VARIANTS}", flush=True)
        sys.exit(1)

    split_dir = (Path(args.windowed_root) / args.split).resolve()
    if not split_dir.is_dir():
        print(f"[DpoSlice] No directory: {split_dir}", flush=True)
        sys.exit(1)
    files = sorted(split_dir.glob(args.file_glob))
    if not files:
        print(f"[DpoSlice] No matching files: {split_dir}/{args.file_glob}", flush=True)
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
        f"[DpoSlice] Input files={len(files)} read_lines={len(all_rows)} "
        f"eval_users={len(users_in)} variants={variants} slice-eval-mode={args.slice_eval_mode} -> {out_path}",
        flush=True,
    )
    # Distinguish action models used by different variants
    base_action_variants = [v for v in variants if v in ("incremental_persona", "regeneration_persona")]
    clasp_action_variants = [v for v in variants if v not in ("incremental_persona", "regeneration_persona")]
    if clasp_action_variants:
        print(
            f"[DpoSlice] Action API ({', '.join(clasp_action_variants)}): "
            f"COMPARISON_CLASP_ACTION_VLLM_MODEL={cfg.COMPARISON_CLASP_ACTION_VLLM_MODEL}",
            flush=True,
        )
    if base_action_variants:
        print(
            f"[DpoSlice] Action API ({', '.join(base_action_variants)}): "
            f"COMPARISON_BASELINE_VLLM_MODEL={cfg.COMPARISON_BASELINE_VLLM_MODEL}",
            flush=True,
        )
    if "baseline" in variants or "clasp_dpo" in variants:
        print(
            f"[DpoSlice] vLLM persona: baseline uses COMPARISON_BASELINE_VLLM_MODEL="
            f"{cfg.COMPARISON_BASELINE_VLLM_MODEL}; clasp_dpo uses "
            f"{cfg.COMPARISON_CLASP_PROFILE_VLLM_MODEL}",
            flush=True,
        )
    if "gpt4o_mini" in variants:
        print(
            f"[DpoSlice] gpt4o_mini persona: OpenAI-compatible API model={cfg.PROFILE_MODEL} base={cfg.OPENAI_BASE_URL}",
            flush=True,
        )
    print(
        f"[DpoSlice] Initial persona P0 mode: {args.initial_p0} "
        f"(shared_gpt=all variants share one GPT initial persona; per_variant=each variant builds P0)",
        flush=True,
    )
    if args.initial_p0 == "shared_gpt":
        if p0_cache_dir_resolved is not None:
            print(
                f"[DpoSlice] Shared P0 cache directory: {p0_cache_dir_resolved}  "
                f"store={_p0_store_json_path(p0_cache_dir_resolved).name}  "
                f"write_disk={p0_cache_write}  --p0-cache-read={bool(args.p0_cache_read)}",
                flush=True,
            )
        else:
            print(
                "[DpoSlice] Shared P0 not written to disk (added --p0-no-disk-cache and no --p0-cache-dir); "
                "only API generated, can't read cache from default path",
                flush=True,
            )

    if args.no_action_prompt_observed_history:
        print(
            "[DpoSlice] --no-action-prompt-observed-history: action prompt doesn't include observed history (consistent with run_baseline_comparison)",
            flush=True,
        )
    if args.slice_eval_mode == SLICE_EVAL_MODE_W2_W3:
        print(
            "[DpoSlice] slice-eval-mode=w2_w3: each row W1_*=physical W2, W2_*=physical W3; input needs at least four windows; "
            f"output {out_name}",
            flush=True,
        )
    elif args.slice_eval_mode == SLICE_EVAL_MODE_W0_W1_W2_P0P1:
        print(
            "[DpoSlice] slice-eval-mode=w0_w1_w2_p0p1: P0/P1 each evaluate W0–W2; "
            "W0 never injects action history; W1/W2 inject="
            f"{not bool(args.no_action_prompt_observed_history)}; "
            f"output {out_name}",
            flush=True,
        )

    if not args.resume and not args.append and out_path.is_file():
        # Overwrite mode: clear content, keep path (don't unlink); write phase still uses open("a").
        out_path.open("w", encoding="utf-8").close()
        print(f"[DpoSlice] Not --resume/--append: cleared main output {out_path}", flush=True)
    elif args.resume and out_path.is_file():
        print(f"[DpoSlice] --resume: append mode, keep existing data", flush=True)

    completed = _load_completed_variant_keys(out_path) if args.resume else set()
    if args.resume and completed:
        print(f"[DpoSlice] --resume: loaded {len(completed)} success keys from output", flush=True)

    users_work = [
        u
        for u in users_in
        if _user_needs_any_variant(u, variants, completed, args.resume)
    ]
    n_work = len(users_work)
    print(f"[DpoSlice] Users pending evaluation: {n_work} / {len(users_in)}", flush=True)

    written = 0
    procs = max(1, int(args.user_processes))

    if n_work == 0:
        print("[DpoSlice] No pending users.", flush=True)
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
                        save_profile_text=args.save_profile_text,
                        force_clasp_action=args.force_clasp_action,
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
                args.save_profile_text,
                args.force_clasp_action,
            )
            for i, u in enumerate(users_work)
        ]
        print(
            f"[DpoSlice] Multi-process: {eff} workers, stagger {args.user_process_stagger}s/process",
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
                    f"this_user={user_elapsed:.1f}s avg={avg:.1f}s/user "
                    f"ETA≈{int(eta // 60)}m{int(eta % 60)}s",
                    flush=True,
                )

    print(f"[DpoSlice] New rows written this run: {written}", flush=True)

    # Re-read output file for summary (includes history rows)
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
