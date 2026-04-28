#!/usr/bin/env python3
"""
动作模型评测：用前 N 条动作经「画像 API」生成用户画像，再在动作后端上预测后续 M 条，
计算 F(S)、L(S)、Q(S)。

默认只评测一个动作模型；明细 `eval_detail_<模型名>.jsonl` 每用户含：
`scores`（F/L/Q）、画像字符长度、送入动作 prompt 的截断后长度、生成类 post/reply 对照。

画像全文会按 `ACTION_PROMPT_PROFILE_MAX_CHARS` 截断后再写入动作 API，避免 4k 上下文 400；
`call_llm_api` 亦会按估算输入长度收缩 max_tokens。
预测时对每条后续动作采用**滑动历史**：使用该步之前的最近 `ACTION_PREDICTION_HISTORY_WINDOW`（默认 5）条真实动作，随步推进将已发生的真实标签追加进历史。
`social_signals` 统计条级「是否含 emoji/#/@」：`presence_ratio` = 含该信号的条数 / 该侧有效条（非字符占比）。

可选 `--compare-baseline-finetuned`：基座+微调各写独立 jsonl，`eval_summary.json` 汇总 Δ 与画像长度均值。

默认 N=30，M=20；数据默认 data/test/community_5.jsonl。微调侧可为 API 或 `--finetuned-action-lora`。

调试可加 `--print-llm-io`（可选 `--print-llm-io-max-chars`，0 为不截断），在终端打印动作模型每次决策/内容生成的 SYSTEM 指令、user 侧拼好的正文与模型原始输出。
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import src.config as cfg
from src.action_predictor import (
    build_content_prompt,
    build_decision_prompt,
    call_llm,
    call_llm_api,
    parse_action_type,
)
from src.config import (
    ACTION_API_BASE,
    ACTION_API_MODEL,
    ACTION_GENERATION_MODEL,
    ACTION_PREDICTION_HISTORY_WINDOW,
    ACTION_PROMPT_PROFILE_MAX_CHARS,
    ALPHA,
    MAX_NEW_TOKENS_ACTION,
    TEMPERATURE_ACTION,
)
from src.profile_generator import generate_initial_profile, truncate_behavior_plaintext
from src.scorer import SemanticScorer, evaluate_predictions
from src.text_social_signals import (
    accumulate_from_generation_rows,
    bucket_add_into,
    empty_signal_bucket,
    finalize_compare,
    finalize_pair,
)

# completion: (instruction, input_text, max_new_tokens, debug_step) -> str
CompleteFn = Callable[..., str]


def _trunc_for_print(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n... [已截断，原长 {len(text)} 字符]\n"


def _print_llm_io(
    *,
    step: str,
    instruction: str,
    input_text: str,
    output: str,
    max_chars: int,
) -> None:
    """终端打印单次动作模型调用的 system / user / assistant 侧内容，便于核对提示词与输出。"""
    print(
        f"\n{'=' * 72}\n[动作模型 LLM-I/O] {step}\n"
        f"{'-' * 72}\n[SYSTEM / instruction]\n{_trunc_for_print(instruction, max_chars)}\n"
        f"{'-' * 72}\n[user 侧 input_text]\n{_trunc_for_print(input_text, max_chars)}\n"
        f"{'-' * 72}\n[模型输出]\n{_trunc_for_print(output or '', max_chars)}\n"
        f"{'=' * 72}\n",
        flush=True,
    )


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _make_complete_api(
    api_base: str,
    model_name: str,
    *,
    print_llm_io: bool = False,
    print_llm_io_max_chars: int = 12_000,
) -> CompleteFn:
    def _complete(
        instruction: str,
        input_text: str,
        max_new_tokens: int,
        *,
        debug_step: str,
    ) -> str:
        out = call_llm_api(
            api_base,
            model_name,
            instruction,
            input_text,
            max_new_tokens,
            TEMPERATURE_ACTION,
            debug_step=debug_step,
            debug_emit=False,
        )
        if print_llm_io:
            _print_llm_io(
                step=debug_step,
                instruction=instruction,
                input_text=input_text,
                output=out,
                max_chars=print_llm_io_max_chars,
            )
        return out

    return _complete


def _make_complete_local(
    model: Any,
    tokenizer: Any,
    *,
    print_llm_io: bool = False,
    print_llm_io_max_chars: int = 12_000,
) -> CompleteFn:
    def _complete(
        instruction: str,
        input_text: str,
        max_new_tokens: int,
        *,
        debug_step: str,
    ) -> str:
        out = call_llm(
            model,
            tokenizer,
            instruction,
            input_text,
            max_new_tokens,
            TEMPERATURE_ACTION,
            debug_step=debug_step,
            debug_emit=False,
        )
        if print_llm_io:
            _print_llm_io(
                step=debug_step,
                instruction=instruction,
                input_text=input_text,
                output=out,
                max_chars=print_llm_io_max_chars,
            )
        return out

    return _complete


def _predict_window_with_trace(
    complete: CompleteFn,
    profile: str,
    history_actions: List[Dict],
    target_actions: List[Dict],
    *,
    trace_label: str,
    history_window: int = ACTION_PREDICTION_HISTORY_WINDOW,
) -> Tuple[List[Dict], List[Dict[str, Any]]]:
    user_profile = profile.strip()
    predictions: List[Dict] = []
    trace: List[Dict[str, Any]] = []
    current_history = list(history_actions)
    hw = max(1, int(history_window))
    n = len(target_actions)

    for i, target in enumerate(target_actions):
        recent = current_history[-hw:] if current_history else []
        inst, inp = build_decision_prompt(user_profile, recent, target)
        raw_decision = complete(
            inst,
            inp,
            128,
            debug_step=f"{trace_label}:decision#{i + 1}/{n}",
        )
        pred_type = parse_action_type(raw_decision)
        raw_content_full: Optional[str] = None
        pred_content: Optional[str] = None

        if pred_type in ("post", "reply"):
            inst_c, inp_c = build_content_prompt(user_profile, recent, target)
            raw_content_full = complete(
                inst_c,
                inp_c,
                MAX_NEW_TOKENS_ACTION,
                debug_step=f"{trace_label}:content#{i + 1}/{n}",
            )
            pred_content = raw_content_full

        predictions.append({"action_type": pred_type, "content": pred_content})
        trace.append({
            "step_index": i,
            "trace_label": trace_label,
            "ground_truth": {
                "action_type": target.get("action_type"),
                "action_text": target.get("action_text"),
                "target": target.get("target"),
                "timestamp": target.get("timestamp"),
            },
            "raw_decision_llm_output": raw_decision,
            "predicted_action_type": pred_type,
            "raw_content_llm_output": raw_content_full,
            "predicted_content": pred_content,
        })
        current_history.append(target)

    return predictions, trace


def _sanitize_model_slug(name: str) -> str:
    """用于输出文件名：避免路径字符与过长。"""
    s = (name or "model").strip().replace("\\", "_").replace("/", "_")
    s = "_".join(s.split()) or "model"
    return s[:120]


def _generation_only_for_detail(
    trace: List[Dict[str, Any]],
    test_actions: List[Dict],
) -> List[Dict[str, Any]]:
    """
    仅「真实标签为 post/reply」的步骤写入明细；每条只含三个字段：
    step_index、user_content（真实正文）、model_content（模型生成；若模型未走内容分支则为 null）。
    like/repost 等决策类步骤不落库。
    """
    out: List[Dict[str, Any]] = []
    for i, target in enumerate(test_actions):
        at = target.get("action_type") or ""
        if at not in ("post", "reply"):
            continue
        if i >= len(trace):
            break
        t = trace[i]
        out.append({
            "step_index": i,
            "user_content": (target.get("action_text") or "").strip(),
            "model_content": t.get("predicted_content"),
        })
    return out


def _default_summary_path_from_detail(detail_path: Path) -> Path:
    """eval_detail_xxx.jsonl → eval_summary_xxx.json（同目录）。"""
    name = detail_path.name
    if name.startswith("eval_detail_") and name.endswith(".jsonl"):
        return detail_path.with_name(
            name.replace("eval_detail_", "eval_summary_", 1).replace(".jsonl", ".json")
        )
    return detail_path.with_suffix(".eval_summary.json")


def _load_action_lora(lora_path: Path) -> Tuple[Any, Any]:
    """动作 LoRA：基座见 adapter_config.json 的 base_model_name_or_path（一般为 Bluesky 指令模型）。"""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError("需要 torch、peft、transformers 才能加载动作 LoRA") from e

    adapter_cfg = lora_path / "adapter_config.json"
    if not adapter_cfg.is_file():
        raise FileNotFoundError(f"未找到 {adapter_cfg}")

    with adapter_cfg.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    base_name = str(meta.get("base_model_name_or_path") or ACTION_GENERATION_MODEL)

    tokenizer = AutoTokenizer.from_pretrained(str(lora_path), trust_remote_code=True)
    dt = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=dt,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, str(lora_path))
    model.eval()
    return model, tokenizer


def run() -> None:
    p = argparse.ArgumentParser(
        description="动作模型评测：默认单模型，明细 eval_detail_<模型>.jsonl 仅含生成类对比",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=_REPO_ROOT / "data/test/community_5.jsonl",
        help="用户 jsonl",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=_REPO_ROOT / "Action_model_experiment/output",
        help="明细与汇总 JSON/JSONL 输出目录",
    )
    p.add_argument(
        "--compare-baseline-finetuned",
        action="store_true",
        help="同时评测「基座 + 微调」两套后端并各写独立 jsonl；默认仅评测单个动作模型",
    )
    p.add_argument(
        "--eval-detail-suffix",
        type=str,
        default=None,
        help="覆盖输出文件名中的模型标识（默认同 --action-model 净化后的片段）",
    )
    p.add_argument(
        "--eval-detail-jsonl",
        type=Path,
        default=None,
        help="明细 jsonl 的完整路径；指定则不再使用 eval_detail_<标识>.jsonl 默认命名",
    )
    p.add_argument(
        "--eval-summary-json",
        type=Path,
        default=None,
        help="汇总 JSON 路径；默认与明细配对（单模型：eval_summary_<标识>.json 或由明细文件名推导）",
    )
    p.add_argument("--history-len", type=int, default=30, help="建画像的前 N 条动作")
    p.add_argument("--predict-len", type=int, default=20, help="预测的后续 M 条")
    p.add_argument(
        "--action-history-window",
        type=int,
        default=ACTION_PREDICTION_HISTORY_WINDOW,
        help="预测时滑动历史：当前步之前最近几条「真实」动作（随步推进追加；默认见 config）",
    )
    p.add_argument(
        "--max-users",
        type=int,
        default=0,
        help="最多评测多少个用户（在数据文件中从上到下、且满足动作条数≥history+predict 的用户里取前 K 个）；0 表示全部",
    )
    p.add_argument(
        "--action-api-base",
        "--baseline-action-api-base",
        type=str,
        default="http://127.0.0.1:8002/v1",
        dest="action_api_base",
        metavar="URL",
        help="单模型模式（或双模型中的基座）动作 API 根 URL。兼容别名 --baseline-action-api-base",
    )
    p.add_argument(
        "--action-model",
        "--baseline-action-model",
        type=str,
        default="Meta-Llama-3-8B-Instruct",
        dest="action_model",
        metavar="NAME",
        help="单模型或基座在服务端注册的 --served-model-name。兼容别名 --baseline-action-model",
    )
    p.add_argument(
        "--finetuned-action-api-base",
        type=str,
        default=ACTION_API_BASE,
        help="微调动作模型 API 根 URL；与 --finetuned-action-lora 互斥",
    )
    p.add_argument(
        "--finetuned-action-model",
        type=str,
        default=ACTION_API_MODEL,
        help="微调动作模型名（API 模式）",
    )
    p.add_argument(
        "--finetuned-action-lora",
        type=Path,
        default=None,
        help="若指定，则微调侧走本地 LoRA（覆盖 finetuned API）",
    )
    p.add_argument(
        "--save-profile-text",
        action="store_true",
        help="在明细 jsonl 每行追加 profile_text（体积大）",
    )
    p.add_argument(
        "--scorer-device",
        type=str,
        default=None,
        help="SentenceTransformer 设备，默认自动",
    )
    p.add_argument(
        "--print-llm-io",
        action="store_true",
        help="打印动作模型每一次 decision/content 调用的 SYSTEM 指令、user 正文与模型输出（便于核对提示词问题）",
    )
    p.add_argument(
        "--print-llm-io-max-chars",
        type=int,
        default=12_000,
        help="与 --print-llm-io 联用：每一段（instruction/input/output）单独截断上限，0 表示不截断",
    )
    args = p.parse_args()

    if not args.data.is_file():
        print(f"[错误] 数据不存在: {args.data}", flush=True)
        sys.exit(1)

    rows = _read_jsonl(args.data)
    need = args.history_len + args.predict_len
    eligible = [r for r in rows if len(r.get("actions") or []) >= need]
    if not eligible:
        print(f"[错误] 没有用户动作数 >= {need}", flush=True)
        sys.exit(1)
    if args.max_users > 0:
        eligible = eligible[: args.max_users]

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.print_llm_io:
        print(
            "[提示] --print-llm-io：将打印动作模型每次 decision/content 的提示词与输出；"
            "画像 API 单次调用仍不经过此开关（需在 src/config 开 DEBUG_LLM 等另行观察）。",
            flush=True,
        )

    io_kw = dict(
        print_llm_io=args.print_llm_io,
        print_llm_io_max_chars=args.print_llm_io_max_chars,
    )
    complete_baseline = _make_complete_api(
        args.action_api_base,
        args.action_model,
        **io_kw,
    )

    ft_lora_path: Optional[Path] = args.finetuned_action_lora
    complete_finetuned: Optional[CompleteFn] = None
    ft_mode = "api"

    if args.compare_baseline_finetuned:
        if ft_lora_path is not None:
            if not ft_lora_path.is_dir():
                print(f"[错误] --finetuned-action-lora 无效: {ft_lora_path}", flush=True)
                sys.exit(1)
            print(f"[加载] 微调动作 LoRA: {ft_lora_path}", flush=True)
            m_ft, tok_ft = _load_action_lora(ft_lora_path)
            complete_finetuned = _make_complete_local(m_ft, tok_ft, **io_kw)
            ft_mode = "local_lora"
        else:
            complete_finetuned = _make_complete_api(
                args.finetuned_action_api_base,
                args.finetuned_action_model,
                **io_kw,
            )
            ft_mode = "api"
    else:
        ft_lora_path = None  # 单模型模式不加载微调

    semantic = SemanticScorer(device=args.scorer_device)
    summary_users: List[Dict[str, Any]] = []

    def _detail_line(
        uid: str,
        profile_full: str,
        trace: List[Dict[str, Any]],
        test: List[Dict],
        f: float,
        l: float,
        q: float,
        *,
        profile_length_chars: int,
        profile_for_action_chars: int,
    ) -> str:
        """明细：scores + 画像长度（全长 / 送入动作 prompt 的长度）+ 生成类步骤。"""
        rec: Dict[str, Any] = {
            "user_id": uid,
            "scores": {"F": f, "L": l, "Q": q},
            "profile_length_chars": profile_length_chars,
            "profile_for_action_prompt_chars": profile_for_action_chars,
            "generation": _generation_only_for_detail(trace, test),
        }
        if args.save_profile_text:
            rec["profile_text"] = profile_full
        return json.dumps(rec, ensure_ascii=False) + "\n"

    if not args.compare_baseline_finetuned:
        slug_single = (
            _sanitize_model_slug(args.eval_detail_suffix)
            if args.eval_detail_suffix
            else _sanitize_model_slug(args.action_model)
        )
        detail_path = args.output_dir / f"eval_detail_{slug_single}.jsonl"
        sig_human = empty_signal_bucket()
        sig_model = empty_signal_bucket()
        with detail_path.open("w", encoding="utf-8") as detail_f:
            for row in eligible:
                uid = row.get("user_id", "")
                actions = list(row["actions"])
                train = actions[: args.history_len]
                test = actions[args.history_len : args.history_len + args.predict_len]

                print(
                    f"[用户] {uid} | 画像 {len(train)} 条 | 预测 {len(test)} 条",
                    flush=True,
                )

                profile = generate_initial_profile(None, None, train)
                plen = len(profile or "")
                profile_for_action = truncate_behavior_plaintext(
                    (profile or "").strip(),
                    int(ACTION_PROMPT_PROFILE_MAX_CHARS),
                )
                pac = len(profile_for_action)

                preds_b, trace_b = _predict_window_with_trace(
                    complete_baseline,
                    profile_for_action,
                    train,
                    test,
                    trace_label="baseline_action_model",
                    history_window=args.action_history_window,
                )
                f_b, l_b, q_b = evaluate_predictions(preds_b, test, semantic, ALPHA)

                gen_rows = _generation_only_for_detail(trace_b, test)
                h_b, m_b = accumulate_from_generation_rows(gen_rows)
                bucket_add_into(sig_human, h_b)
                bucket_add_into(sig_model, m_b)

                detail_f.write(
                    _detail_line(
                        uid,
                        profile,
                        trace_b,
                        test,
                        f_b,
                        l_b,
                        q_b,
                        profile_length_chars=plen,
                        profile_for_action_chars=pac,
                    )
                )
                detail_f.flush()

                summary_users.append(
                    {
                        "user_id": uid,
                        "F": f_b,
                        "L": l_b,
                        "Q": q_b,
                        "profile_length_chars": plen,
                        "profile_for_action_prompt_chars": pac,
                    }
                )

        summary_path = args.output_dir / f"eval_summary_{slug_single}.json"
        fs = [u["F"] for u in summary_users]
        ls = [u["L"] for u in summary_users]
        qs = [u["Q"] for u in summary_users]
        summary = {
            "config": {
                "compare_mode": False,
                "data": str(args.data),
                "history_len": args.history_len,
                "predict_len": args.predict_len,
                "alpha": ALPHA,
                "action_api_base": args.action_api_base,
                "action_model": args.action_model,
                "action_history_window": args.action_history_window,
                "action_prompt_profile_max_chars": int(ACTION_PROMPT_PROFILE_MAX_CHARS),
                "print_llm_io": args.print_llm_io,
                "print_llm_io_max_chars": args.print_llm_io_max_chars,
            },
            "user_count": len(eligible),
            "users": summary_users,
            "aggregate": {
                "mean_F": statistics.mean(fs) if fs else 0.0,
                "mean_L": statistics.mean(ls) if ls else 0.0,
                "mean_Q": statistics.mean(qs) if qs else 0.0,
                "median_F": statistics.median(fs) if fs else 0.0,
                "median_L": statistics.median(ls) if ls else 0.0,
                "median_Q": statistics.median(qs) if qs else 0.0,
                "mean_profile_length_chars": statistics.mean(
                    [u["profile_length_chars"] for u in summary_users]
                )
                if summary_users
                else 0.0,
                "mean_profile_for_action_prompt_chars": statistics.mean(
                    [u["profile_for_action_prompt_chars"] for u in summary_users]
                )
                if summary_users
                else 0.0,
            },
            "social_signals": finalize_pair(sig_human, sig_model),
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        agg = summary["aggregate"]
        print(f"\n[完成] detail: {detail_path}", flush=True)
        print(f"[完成] summary: {summary_path}", flush=True)
        print(
            f"[汇总] F/L/Q：mean F={agg['mean_F']:.4f}  L={agg['mean_L']:.4f}  Q={agg['mean_Q']:.4f}",
            flush=True,
        )
        ss = summary.get("social_signals") or {}
        if ss and "human" in ss and "model" in ss:
            h, m = ss["human"], ss["model"]
            print(
                "[信号] 人类 vs 模型 — 条级占比 presence_ratio：emoji "
                f"{h['emoji']['presence_ratio']:.3f} / {m['emoji']['presence_ratio']:.3f}，hashtag "
                f"{h['hashtag']['presence_ratio']:.3f} / {m['hashtag']['presence_ratio']:.3f}，mention "
                f"{h['mention']['presence_ratio']:.3f} / {m['mention']['presence_ratio']:.3f}",
                flush=True,
            )
            print(
                "[长度] 非空生成条上平均字符数（human / model）："
                f"{h['text_length']['mean_chars_per_row']:.2f} / {m['text_length']['mean_chars_per_row']:.2f}",
                flush=True,
            )
        return

    # ---- 基座 + 微调对比：两套独立明细，不合并到同一行 ----
    if args.eval_detail_suffix:
        slug_b = _sanitize_model_slug(f"{args.eval_detail_suffix}_baseline")
        slug_f = _sanitize_model_slug(f"{args.eval_detail_suffix}_finetuned")
    else:
        slug_b = _sanitize_model_slug(args.action_model)
        if ft_mode == "local_lora" and ft_lora_path is not None:
            slug_f = _sanitize_model_slug(f"lora_{ft_lora_path.name}")
        else:
            slug_f = _sanitize_model_slug(args.finetuned_action_model)

    assert complete_finetuned is not None

    detail_b = args.output_dir / f"eval_detail_{slug_b}.jsonl"
    detail_f_path = args.output_dir / f"eval_detail_{slug_f}.jsonl"
    sig_human = empty_signal_bucket()
    sig_model_b = empty_signal_bucket()
    sig_model_f = empty_signal_bucket()
    with detail_b.open("w", encoding="utf-8") as fb, detail_f_path.open(
        "w", encoding="utf-8",
    ) as ff:
        for row in eligible:
            uid = row.get("user_id", "")
            actions = list(row["actions"])
            train = actions[: args.history_len]
            test = actions[args.history_len : args.history_len + args.predict_len]

            print(
                f"[用户] {uid} | 画像 {len(train)} 条 | 预测 {len(test)} 条",
                flush=True,
            )

            profile = generate_initial_profile(None, None, train)
            plen = len(profile or "")
            profile_for_action = truncate_behavior_plaintext(
                (profile or "").strip(),
                int(ACTION_PROMPT_PROFILE_MAX_CHARS),
            )
            pac = len(profile_for_action)

            preds_b, trace_b = _predict_window_with_trace(
                complete_baseline,
                profile_for_action,
                train,
                test,
                trace_label="baseline_action_model",
                history_window=args.action_history_window,
            )
            f_b, l_b, q_b = evaluate_predictions(preds_b, test, semantic, ALPHA)

            preds_ft, trace_ft = _predict_window_with_trace(
                complete_finetuned,
                profile_for_action,
                train,
                test,
                trace_label="finetuned_action_model",
                history_window=args.action_history_window,
            )
            f_f, l_f, q_f = evaluate_predictions(preds_ft, test, semantic, ALPHA)

            rows_b = _generation_only_for_detail(trace_b, test)
            rows_f = _generation_only_for_detail(trace_ft, test)
            h_a, mb_a = accumulate_from_generation_rows(rows_b)
            _, mf_a = accumulate_from_generation_rows(rows_f)
            bucket_add_into(sig_human, h_a)
            bucket_add_into(sig_model_b, mb_a)
            bucket_add_into(sig_model_f, mf_a)

            fb.write(
                _detail_line(
                    uid,
                    profile,
                    trace_b,
                    test,
                    f_b,
                    l_b,
                    q_b,
                    profile_length_chars=plen,
                    profile_for_action_chars=pac,
                )
            )
            ff.write(
                _detail_line(
                    uid,
                    profile,
                    trace_ft,
                    test,
                    f_f,
                    l_f,
                    q_f,
                    profile_length_chars=plen,
                    profile_for_action_chars=pac,
                )
            )
            fb.flush()
            ff.flush()

            summary_users.append(
                {
                    "user_id": uid,
                    "F_baseline": f_b,
                    "L_baseline": l_b,
                    "Q_baseline": q_b,
                    "F_finetuned": f_f,
                    "L_finetuned": l_f,
                    "Q_finetuned": q_f,
                    "dF": f_f - f_b,
                    "dL": l_f - l_b,
                    "dQ": q_f - q_b,
                    "profile_length_chars": plen,
                    "profile_for_action_prompt_chars": pac,
                }
            )

    dfs = [u["dF"] for u in summary_users]
    dls = [u["dL"] for u in summary_users]
    dqs = [u["dQ"] for u in summary_users]
    summary = {
        "config": {
            "compare_mode": True,
            "data": str(args.data),
            "history_len": args.history_len,
            "predict_len": args.predict_len,
            "alpha": ALPHA,
            "baseline_action_api_base": args.action_api_base,
            "baseline_action_model": args.action_model,
            "detail_baseline_jsonl": str(detail_b.name),
            "detail_finetuned_jsonl": str(detail_f_path.name),
            "finetuned_mode": ft_mode,
            "finetuned_action_api_base": args.finetuned_action_api_base
            if ft_mode == "api"
            else None,
            "finetuned_action_model": args.finetuned_action_model
            if ft_mode == "api"
            else None,
            "finetuned_action_lora": str(ft_lora_path) if ft_lora_path else None,
            "action_prompt_profile_max_chars": int(ACTION_PROMPT_PROFILE_MAX_CHARS),
            "action_history_window": args.action_history_window,
            "print_llm_io": args.print_llm_io,
            "print_llm_io_max_chars": args.print_llm_io_max_chars,
        },
        "user_count": len(eligible),
        "users": summary_users,
        "aggregate": {
            "mean_dF": statistics.mean(dfs) if dfs else 0.0,
            "mean_dL": statistics.mean(dls) if dls else 0.0,
            "mean_dQ": statistics.mean(dqs) if dqs else 0.0,
            "median_dF": statistics.median(dfs) if dfs else 0.0,
            "median_dL": statistics.median(dls) if dls else 0.0,
            "median_dQ": statistics.median(dqs) if dqs else 0.0,
            "mean_profile_length_chars": statistics.mean(
                [u["profile_length_chars"] for u in summary_users]
            )
            if summary_users
            else 0.0,
            "mean_profile_for_action_prompt_chars": statistics.mean(
                [u["profile_for_action_prompt_chars"] for u in summary_users]
            )
            if summary_users
            else 0.0,
        },
        "social_signals": finalize_compare(sig_human, sig_model_b, sig_model_f),
    }

    summary_path = args.output_dir / "eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    agg = summary["aggregate"]
    print(f"\n[完成] baseline detail: {detail_b}", flush=True)
    print(f"[完成] finetuned detail: {detail_f_path}", flush=True)
    print(f"[完成] summary: {summary_path}", flush=True)
    print(
        f"[汇总] 微调 − 基座：mean ΔF={agg['mean_dF']:.4f}  ΔL={agg['mean_dL']:.4f}  ΔQ={agg['mean_dQ']:.4f}",
        flush=True,
    )
    ss = summary.get("social_signals") or {}
    if ss and "human" in ss:
        hum = ss["human"]
        b = ss.get("model_baseline") or {}
        f = ss.get("model_finetuned") or {}
        if b and f:
            print(
                "[信号] 人类 / 基座 / 微调 — emoji 条级占比 "
                f"{hum['emoji']['presence_ratio']:.3f} / {b['emoji']['presence_ratio']:.3f} / {f['emoji']['presence_ratio']:.3f}",
                flush=True,
            )
            print(
                "[长度] 平均字符数 human / baseline / finetuned："
                f"{hum['text_length']['mean_chars_per_row']:.2f} / "
                f"{b['text_length']['mean_chars_per_row']:.2f} / {f['text_length']['mean_chars_per_row']:.2f}",
                flush=True,
            )


if __name__ == "__main__":
    run()
