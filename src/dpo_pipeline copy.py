#!/usr/bin/env python3
"""
DPO 全流程 Pipeline
步骤：
  1. 加载窗口化数据
  2. 用 W0 生成初始画像 S0
  3. 用 S0 在 W1 上预测，计算 base Q(S0)
  4. 构建偏差信号，生成 N=15 候选画像
  5. 对每个候选画像在 W0/W1/W2 上评分
  6. 构造 DPO 正负对并保存
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import src.config as cfg
from src.config import (
    ACTION_API_BASE,
    ACTION_API_MODEL,
    ALPHA,
    DELTA,
    DPO_BETA,
    DPO_SFT_LOSS_WEIGHT,
    NUM_CANDIDATE_PROFILES,
    PROFILE_API_BASE,
    PROFILE_API_MODEL,
    SENTENCE_TRANSFORMER_MODEL,
    TAU_MINUS,
    TAU_PLUS,
    TEMPERATURE_ACTION,
)
from src.action_predictor import (
    build_behavior_discrepancies,
    predict_actions_for_window,
)
from src.profile_generator import (
    generate_candidate_profiles,
    generate_initial_profile,
    profile_candidate_source,
)
from src.prompts import build_profile_refinement_prompt_messages
from src.scorer import SemanticScorer, evaluate_predictions


# Debug：与画像基座一致的 tokenizer，用于统计 persona 文本 token 数（仅懒加载一次）
_profile_tokenizer_for_debug = None


def _get_profile_tokenizer_for_debug():
    """返回 tokenizer；失败时返回 False 表示不可用。"""
    global _profile_tokenizer_for_debug
    if _profile_tokenizer_for_debug is not None:
        return _profile_tokenizer_for_debug
    try:
        from transformers import AutoTokenizer

        _profile_tokenizer_for_debug = AutoTokenizer.from_pretrained(
            cfg.PROFILE_GENERATION_MODEL_RAW,
        )
    except Exception as e:
        print(
            f"[DEBUG][Tokenizer] 无法从本地加载计数用 tokenizer "
            f"({cfg.PROFILE_GENERATION_MODEL_RAW}): {e}",
            flush=True,
        )
        _profile_tokenizer_for_debug = False
    return _profile_tokenizer_for_debug


def _print_profile_token_lens_debug(uid, s0: str, candidates: List[str]) -> None:
    """Debug：打印 S0 与各候选画像的 token/字符长度（便于检查过短、崩坏或异常冗长）。"""
    print(f"\n[DEBUG][User {uid}] Step5 结束 — 画像长度（token 以基座分词器计，add_special_tokens=False）", flush=True)
    tok = _get_profile_tokenizer_for_debug()
    if tok is False:
        print(f"  S0 chars={len(s0)}", flush=True)
        for i, c in enumerate(candidates):
            print(f"  候选 {i + 1} chars={len(c)} (Δ chars vs S0={len(c) - len(s0):+d})", flush=True)
        print("", flush=True)
        return
    n0 = len(tok.encode(s0 or "", add_special_tokens=False))
    print(f"  S0 tokens={n0}, chars={len(s0 or '')}", flush=True)
    for i, c in enumerate(candidates):
        nc = len(tok.encode(c or "", add_special_tokens=False))
        print(
            f"  候选 {i + 1} tokens={nc} (Δ vs S0={nc - n0:+d}), chars={len(c or '')}",
            flush=True,
        )
    print("", flush=True)


# ============================================================================
# 启动前模型连通性检测
# ============================================================================

def _check_api(api_base: str, model_name: str, api_key: str = "not-needed", label: str = "") -> bool:
    """向 OpenAI 兼容 API 发送一个最小请求，检测连通性。"""
    try:
        from openai import OpenAI
        client = OpenAI(base_url=api_base, api_key=api_key, timeout=15)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=5,
            temperature=0.01,
        )
        text = resp.choices[0].message.content.strip()
        print(f"  [✓] {label} 连通成功（响应: {text[:50]}）", flush=True)
        return True
    except Exception as e:
        print(f"  [✗] {label} 连通失败: {e}", flush=True)
        return False


def _check_local_path(path: str, label: str = "") -> bool:
    """检查本地模型路径是否存在。"""
    p = Path(path)
    if p.exists():
        print(f"  [✓] {label} 路径存在: {path}", flush=True)
        return True
    print(f"  [✗] {label} 路径不存在: {path}", flush=True)
    return False


def preflight_check() -> bool:
    """
    启动前检测：
    1. 两个 vLLM 服务（画像生成 + 动作预测）是否可访问
    2. Sentence Transformer 本地路径是否存在
    """
    print("\n[Preflight] 模型连通性检测 ...", flush=True)

    ok1 = _check_api(PROFILE_API_BASE, PROFILE_API_MODEL, label="画像生成模型")
    ok2 = _check_api(ACTION_API_BASE, ACTION_API_MODEL, label="动作预测模型")
    ok3 = _check_local_path(SENTENCE_TRANSFORMER_MODEL, "Sentence Transformer")

    all_ok = ok1 and ok2 and ok3

    if all_ok:
        print("[Preflight] 全部检测通过 ✓\n", flush=True)
    else:
        print("[Preflight] 存在不可用的服务，请检查后重试 ✗\n", flush=True)

    return all_ok


# ============================================================================
# 单个用户的窗口评估
# ============================================================================

def evaluate_profile_on_window(
    profile: str,
    history: List[Dict],
    targets: List[Dict],
    action_model,
    action_tokenizer,
    semantic_scorer: SemanticScorer,
) -> Tuple[float, float, float]:
    """
    在单个窗口上评估画像，返回 (F, L, Q)。
    history: 前序窗口动作（作为上下文）
    targets: 目标窗口动作（待预测）
    """
    predictions = predict_actions_for_window(
        action_model, action_tokenizer,
        profile, history, targets,
        temperature=TEMPERATURE_ACTION,
    )
    return evaluate_predictions(predictions, targets, semantic_scorer, ALPHA)


# ============================================================================
# DPO 对构造
# ============================================================================

def construct_dpo_pairs(
    s0_profile: str,
    candidates: List[str],
    s0_scores: Dict[str, Tuple[float, float, float]],
    candidate_scores: List[Dict[str, Tuple[float, float, float]]],
    prompt_messages: Optional[List[Dict[str, str]]] = None,
    *,
    baseline_profile_source: str = "base",
) -> List[Dict]:
    """
    构造 DPO 对。

    s0_scores: {"W0": (F,L,Q), "W1": (F,L,Q), "W2": (F,L,Q)}
    candidate_scores: [{"W0": (F,L,Q), ...}, ...]  length = N

    对每个候选 Si:
      r_pre = Q(Si,W0) - Q(S0,W0)
      r_cur = Q(Si,W1) - Q(S0,W1)
      r_fut = Q(Si,W2) - Q(S0,W2)
      r_all = r_pre + r_cur + r_fut

    prompt_messages: 与画像精炼一致的多轮消息列表（system + user），
    user 中含旧人格与行为偏差，写入每条 DPO 样本的 `prompt` 供微调使用。
    """
    q_s0 = {w: scores[2] for w, scores in s0_scores.items()}

    reward_list = []
    for i, cand_scores in enumerate(candidate_scores):
        q_si = {w: scores[2] for w, scores in cand_scores.items()}
        r_pre = q_si.get("W0", 0) - q_s0.get("W0", 0)
        r_cur = q_si.get("W1", 0) - q_s0.get("W1", 0)
        r_fut = q_si.get("W2", 0) - q_s0.get("W2", 0)
        r_all = r_pre + r_cur + r_fut
        reward_list.append({
            "index": i,
            "profile": candidates[i],
            "r_pre": r_pre,
            "r_cur": r_cur,
            "r_fut": r_fut,
            "r_all": r_all,
            "scores": {w: {"F": s[0], "L": s[1], "Q": s[2]} for w, s in cand_scores.items()},
        })

    positive = [r for r in reward_list if r["r_all"] > TAU_PLUS]
    negative = [r for r in reward_list if r["r_all"] < TAU_MINUS]

    n_cand = len(candidates)
    dpo_pairs = []
    for pos in positive:
        for neg in negative:
            if pos["r_all"] - neg["r_all"] > DELTA:
                row = {
                    "chosen": {
                        "profile": pos["profile"],
                        "profile_source": profile_candidate_source(pos["index"], n_cand),
                        "r_all": pos["r_all"],
                        "r_pre": pos["r_pre"],
                        "r_cur": pos["r_cur"],
                        "r_fut": pos["r_fut"],
                        "scores": pos["scores"],
                    },
                    "rejected": {
                        "profile": neg["profile"],
                        "profile_source": profile_candidate_source(neg["index"], n_cand),
                        "r_all": neg["r_all"],
                        "r_pre": neg["r_pre"],
                        "r_cur": neg["r_cur"],
                        "r_fut": neg["r_fut"],
                        "scores": neg["scores"],
                    },
                    "baseline_profile": s0_profile,
                    "baseline_profile_source": baseline_profile_source,
                    "baseline_scores": {w: {"F": s[0], "L": s[1], "Q": s[2]} for w, s in s0_scores.items()},
                    "margin": pos["r_all"] - neg["r_all"],
                }
                # 输入上下文 x：与画像精炼时一致（旧人格 + 预测/真实偏差），供 TRL DPO 对话式 prompt
                if prompt_messages is not None:
                    row["prompt"] = prompt_messages
                dpo_pairs.append(row)

    return dpo_pairs


# ============================================================================
# 单用户 DPO 流程
# ============================================================================

def process_single_user(
    user_data: Dict,
    profile_model,
    profile_tokenizer,
    action_model,
    action_tokenizer,
    semantic_scorer: SemanticScorer,
) -> Dict:
    """
    对单个用户执行完整 DPO 流程。
    user_data: {"user_id", "community_id", "windows": {"W0":[...], ...}}
    """
    uid = user_data["user_id"]
    cid = user_data["community_id"]
    windows = user_data["windows"]
    w0, w1, w2 = windows["W0"], windows["W1"], windows["W2"]

    print(f"\n{'='*60}", flush=True)
    print(f"[User {uid}] 社区={cid}", flush=True)

    # === Step 1: 生成初始画像 S0 ===
    print(f"[User {uid}] Step 1: 生成初始画像 S0 ...", flush=True)
    s0 = generate_initial_profile(profile_model, profile_tokenizer, w0)
    print(f"[User {uid}] S0 长度: {len(s0)} chars", flush=True)

    # === Step 2: 用 S0 在 W0/W1/W2 上评估 baseline ===
    print(f"[User {uid}] Step 2: 计算 S0 baseline 得分 ...", flush=True)
    # W0 评估：空历史起，对 W0 内全部动作逐步预测（每步后将该步真实动作并入历史）
    f0, l0, q0 = evaluate_profile_on_window(
        s0, [], w0, action_model, action_tokenizer, semantic_scorer
    )
    # W1 评估：用 W0 作为历史
    f1, l1, q1 = evaluate_profile_on_window(
        s0, w0, w1, action_model, action_tokenizer, semantic_scorer
    )
    # W2 评估：用 W1 作为历史
    f2, l2, q2 = evaluate_profile_on_window(
        s0, w1, w2, action_model, action_tokenizer, semantic_scorer
    )

    s0_scores = {
        "W0": (f0, l0, q0),
        "W1": (f1, l1, q1),
        "W2": (f2, l2, q2),
    }
    print(
        f"[User {uid}] S0 scores: "
        f"W0=Q{q0:.4f}, W1=Q{q1:.4f}, W2=Q{q2:.4f}",
        flush=True,
    )

    # === Step 3: 构建偏差信号 ===
    print(f"[User {uid}] Step 3: 构建 W1 偏差信号 ...", flush=True)
    preds_w1 = predict_actions_for_window(
        action_model, action_tokenizer, s0, w0, w1,
        temperature=TEMPERATURE_ACTION,
    )
    discrepancies = build_behavior_discrepancies(preds_w1, w1, w0)

    if cfg.DEBUG_LLM:
        print(
            "\n"
            + "=" * 72
            + f"\n[DEBUG][User {uid}] Step3 行为偏差全文（reply 含 Replied-to original；将送入画像精炼）\n"
            + "-" * 72
            + "\n"
            + discrepancies
            + "\n"
            + "=" * 72
            + "\n",
            flush=True,
        )

    # === Step 4: 生成 N 个候选画像 ===
    print(f"[User {uid}] Step 4: 生成 {NUM_CANDIDATE_PROFILES} 个候选画像 ...", flush=True)
    candidates = generate_candidate_profiles(
        profile_model, profile_tokenizer,
        s0, discrepancies,
        n=NUM_CANDIDATE_PROFILES,
    )

    # === Step 5: 对每个候选画像在 W0/W1/W2 上评分 ===
    print(f"[User {uid}] Step 5: 评分候选画像 ...", flush=True)
    candidate_scores_list = []
    for i, cand in enumerate(candidates):
        cf0, cl0, cq0 = evaluate_profile_on_window(
            cand, [], w0, action_model, action_tokenizer, semantic_scorer
        )
        cf1, cl1, cq1 = evaluate_profile_on_window(
            cand, w0, w1, action_model, action_tokenizer, semantic_scorer
        )
        cf2, cl2, cq2 = evaluate_profile_on_window(
            cand, w1, w2, action_model, action_tokenizer, semantic_scorer
        )
        cand_scores = {
            "W0": (cf0, cl0, cq0),
            "W1": (cf1, cl1, cq1),
            "W2": (cf2, cl2, cq2),
        }
        candidate_scores_list.append(cand_scores)
        r_all = (cq0 - q0) + (cq1 - q1) + (cq2 - q2)
        print(
            f"  候选 {i+1}: Q_W0={cq0:.4f} Q_W1={cq1:.4f} Q_W2={cq2:.4f} r_all={r_all:+.4f}",
            flush=True,
        )

    if cfg.DEBUG_LLM:
        _print_profile_token_lens_debug(uid, s0, candidates)

    # === Step 6: 构造 DPO 对（附带精炼任务上下文 x）===
    print(f"[User {uid}] Step 6: 构造 DPO 对 ...", flush=True)
    prompt_messages = build_profile_refinement_prompt_messages(s0, discrepancies)
    dpo_pairs = construct_dpo_pairs(
        s0, candidates, s0_scores, candidate_scores_list,
        prompt_messages=prompt_messages,
    )
    for p in dpo_pairs:
        p["user_id"] = uid
        p["community_id"] = cid
    print(f"[User {uid}] 生成 {len(dpo_pairs)} 个 DPO 对", flush=True)

    return {
        "user_id": uid,
        "community_id": cid,
        "s0_profile": s0,
        "s0_scores": {w: {"F": s[0], "L": s[1], "Q": s[2]} for w, s in s0_scores.items()},
        "num_candidates": len(candidates),
        "num_dpo_pairs": len(dpo_pairs),
        "dpo_pairs": dpo_pairs,
    }


# ============================================================================
# 主入口
# ============================================================================

def run_dpo_pipeline(
    input_file: str,
    output_dir: str,
    max_users: int = None,
    random_seed: Optional[int] = None,
) -> None:
    """
    DPO Pipeline 主入口。
    input_file: 窗口化后的 jsonl 文件（来自 window_splitter）
    output_dir: DPO 对输出目录
    """
    if not preflight_check():
        print("[Pipeline] 预检失败，退出", flush=True)
        sys.exit(1)

    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 加载数据
    users = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                users.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    n_loaded = len(users)
    if random_seed is not None:
        rng = random.Random(random_seed)
        rng.shuffle(users)
        print(
            f"[Pipeline] 已用 --seed {random_seed} 将 {n_loaded} 个用户打乱顺序",
            flush=True,
        )
    if max_users:
        users = users[:max_users]
    print(f"[Pipeline] 处理 {len(users)} 个用户（本 run）", flush=True)

    # 加载 semantic scorer（唯一需要本地加载的模型）
    print("[Pipeline] 加载 Semantic Scorer ...", flush=True)
    semantic_scorer = SemanticScorer()

    # LLM 已通过 vLLM 服务事先启动，无需本地加载
    profile_model, profile_tokenizer = None, None
    action_model, action_tokenizer = None, None

    # 处理每个用户
    all_results = []
    all_dpo_pairs = []
    for i, user_data in enumerate(users):
        print(f"\n[Pipeline] 处理用户 {i+1}/{len(users)}", flush=True)
        t0 = time.time()
        result = process_single_user(
            user_data,
            profile_model, profile_tokenizer,
            action_model, action_tokenizer,
            semantic_scorer,
        )
        elapsed = time.time() - t0
        print(f"[Pipeline] 用户 {result['user_id']} 完成，耗时 {elapsed:.1f}s", flush=True)

        all_results.append(result)
        all_dpo_pairs.extend(result["dpo_pairs"])

    # 保存结果
    stem = input_path.stem
    pairs_file = output_path / f"dpo_pairs_{stem}.jsonl"
    with pairs_file.open("w", encoding="utf-8") as f:
        for pair in all_dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"\n[Pipeline] DPO 对已保存: {pairs_file} ({len(all_dpo_pairs)} 对)", flush=True)

    detail_file = output_path / f"dpo_detail_{stem}.json"
    with detail_file.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"[Pipeline] 详细结果已保存: {detail_file}", flush=True)

    summary = {
        "input_file": str(input_path),
        "total_users": len(users),
        "total_dpo_pairs": len(all_dpo_pairs),
        "users_with_pairs": sum(1 for r in all_results if r["num_dpo_pairs"] > 0),
        "config": {
            "tau_plus": TAU_PLUS,
            "tau_minus": TAU_MINUS,
            "delta": DELTA,
            "alpha": ALPHA,
            "num_candidates": NUM_CANDIDATE_PROFILES,
            "dpo_beta": DPO_BETA,
            "dpo_sft_loss_weight": DPO_SFT_LOSS_WEIGHT,
            "debug_llm": cfg.DEBUG_LLM,
            "debug_llm_include_actions": getattr(cfg, "DEBUG_LLM_INCLUDE_ACTIONS", False),
            "random_seed": random_seed,
        },
    }
    summary_file = output_path / f"dpo_summary_{stem}.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[Pipeline] 统计摘要已保存: {summary_file}", flush=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPO 对构造 Pipeline")
    parser.add_argument("--input", required=True, help="窗口化后的 jsonl 文件")
    parser.add_argument("--output-dir", default="output/dpo", help="DPO 对输出目录")
    parser.add_argument("--max-users", type=int, default=None, help="最多处理用户数（调试用）")
    parser.add_argument("--test", action="store_true", help="测试模式：用远程 API 单模型跑通 pipeline")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="调试：每次 LLM 调用在终端打印步骤、模型类型(model_role)、模型名、请求与完整输出",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子：加载全部用户后先按此种子打乱顺序，再取前 --max-users 条，便于换用户做重复实验",
    )
    parser.add_argument(
        "--debug-actions",
        action="store_true",
        help="与 --debug 联用：额外打印每次「动作预测」LLM 的完整调试块（含 scenario 里 reply 原文；输出量很大）",
    )
    args = parser.parse_args()

    if args.test:
        cfg.TEST_MODE = True
    if args.debug:
        cfg.DEBUG_LLM = True
    if args.debug_actions:
        cfg.DEBUG_LLM_INCLUDE_ACTIONS = True

    run_dpo_pipeline(args.input, args.output_dir, args.max_users, args.seed)
