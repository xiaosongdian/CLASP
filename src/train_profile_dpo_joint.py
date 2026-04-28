#!/usr/bin/env python3
"""
画像生成模型 DPO + SFT 联合微调入口（TRL）。

损失：
  L = L_DPO(sigmoid, beta) + a * L_SFT(chosen NLL)
对应 TRL 配置：
  loss_type=["sigmoid", "sft"]
  loss_weights=[1.0, sft_weight]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from transformers import TrainerCallback

from src.config import (
    DPO_BETA,
    DPO_SFT_LOSS_WEIGHT,
    PROFILE_GENERATION_MODEL_RAW,
    PROFILE_REFINEMENT_DISCREPANCY_MAX_CHARS,
    PROFILE_REFINEMENT_OLD_PERSONA_MAX_CHARS,
)
from src.profile_generator import truncate_behavior_plaintext
from src.prompts import build_profile_refinement_prompt_messages


class _GradNormLogCallback(TrainerCallback):
    """
    在 optimizer.step() 之前打印梯度 L2 范数（只读 .grad，不改训练数学）。
    需 bind_trainer：HF 回调拿不到 model，训练开始后再绑定即可。
    """

    def __init__(self, every_n: int, lora_only: bool) -> None:
        super().__init__()
        self.every_n = max(0, int(every_n))
        self.lora_only = bool(lora_only)
        self._trainer = None  # type: ignore

    def bind_trainer(self, trainer: Any) -> None:
        self._trainer = trainer

    def on_pre_optimizer_step(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
        if self.every_n <= 0 or self._trainer is None:
            return control
        # 本次即将完成的 global_step（step 在 step 之后才会 state.global_step += 1）
        upcoming = int(state.global_step) + 1
        if upcoming % self.every_n != 0:
            return control

        import torch

        model = self._trainer.model
        if hasattr(self._trainer, "accelerator"):
            model = self._trainer.accelerator.unwrap_model(model)

        sq = 0.0
        n_params = 0
        max_layer = 0.0
        max_name = ""
        for name, p in model.named_parameters():
            if not p.requires_grad or p.grad is None:
                continue
            if self.lora_only and "lora" not in name.lower():
                continue
            g = p.grad.detach()
            # 统一在 float32 上算范数，避免半精度溢出/误差
            gn = float(g.float().norm(2).item())
            sq += gn * gn
            n_params += 1
            if gn > max_layer:
                max_layer = gn
                max_name = name

        total = sq ** 0.5
        scope = "lora" if self.lora_only else "all_trainable"
        print(
            f"[Grad] step={upcoming} scope={scope} total_l2={total:.6f} "
            f"params_with_grad={n_params} max_layer_l2={max_layer:.6f} ({max_name})",
            flush=True,
        )
        return control


def _prompt_to_text(prompt_obj: Any) -> str:
    """
    兼容 dpo_pipeline 导出的 prompt 格式：
    - list[{"role","content"}, ...]
    - 纯字符串
    统一转换为纯文本，避免某些 TRL 版本对结构化 prompt 处理异常。
    """
    if isinstance(prompt_obj, list):
        chunks: List[str] = []
        for msg in prompt_obj:
            if isinstance(msg, dict):
                role = str(msg.get("role", "")).strip()
                content = str(msg.get("content", "")).strip()
                if role:
                    chunks.append(f"[{role}] {content}")
                else:
                    chunks.append(content)
            else:
                chunks.append(str(msg))
        return "\n".join(chunks).strip()
    return str(prompt_obj or "")


def _prompt_from_baseline_and_discrepancies(obj: Dict[str, Any]) -> str:
    """与 profile 精炼线上逻辑一致：system + user（旧画像 + 行为误差）。"""
    baseline = obj.get("baseline_profile")
    if baseline is None or not str(baseline).strip():
        return ""
    disc_raw = obj.get("discrepancies")
    disc = "" if disc_raw is None else str(disc_raw)
    old_t = truncate_behavior_plaintext(
        str(baseline), int(PROFILE_REFINEMENT_OLD_PERSONA_MAX_CHARS)
    )
    disc_t = truncate_behavior_plaintext(
        disc, int(PROFILE_REFINEMENT_DISCREPANCY_MAX_CHARS)
    )
    messages = build_profile_refinement_prompt_messages(old_t, disc_t)
    return _prompt_to_text(messages)


def _r_all_preference_ok(chosen: Dict[str, Any], rejected: Dict[str, Any]) -> bool:
    """数据里若带 r_all，则 chosen 应严格优于 rejected（与 DPO 构造一致）。"""
    if "r_all" not in chosen or "r_all" not in rejected:
        return True
    try:
        c = float(chosen["r_all"])
        r = float(rejected["r_all"])
    except (TypeError, ValueError):
        return True
    return c > r


def _load_preference_rows(path: Path, *, allow_inverted_pairs: bool) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    skipped_no_prompt = 0
    skipped_inverted = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            chosen = obj.get("chosen")
            rejected = obj.get("rejected")
            if not isinstance(chosen, dict) or not isinstance(rejected, dict):
                print("[Warn] 跳过 chosen/rejected 非对象结构样本", flush=True)
                continue
            if "profile" not in chosen or "profile" not in rejected:
                print("[Warn] 跳过缺少 chosen/rejected.profile 的样本", flush=True)
                continue
            if not allow_inverted_pairs and not _r_all_preference_ok(chosen, rejected):
                skipped_inverted += 1
                continue
            if "prompt" in obj and obj["prompt"] is not None:
                prompt_text = _prompt_to_text(obj["prompt"])
            else:
                prompt_text = _prompt_from_baseline_and_discrepancies(obj)
                if not prompt_text:
                    skipped_no_prompt += 1
                    print(
                        "[Warn] 跳过无 prompt 且无 baseline_profile 的样本",
                        flush=True,
                    )
                    continue
            rows.append(
                {
                    "prompt": prompt_text,
                    "chosen": str(chosen["profile"]),
                    "rejected": str(rejected["profile"]),
                }
            )
    if skipped_inverted:
        print(
            f"[Load] 已跳过 r_all 顺序异常（chosen<=rejected）样本: {skipped_inverted} 条"
            f"（可用 --allow-inverted-pairs 保留）",
            flush=True,
        )
    if skipped_no_prompt:
        print(
            f"[Load] 因缺少可构造上下文跳过的样本: {skipped_no_prompt} 条",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="画像模型 DPO + SFT 联合微调（TRL）")
    parser.add_argument("--data", required=True, help="dpo_pairs_*.jsonl")
    parser.add_argument(
        "--model",
        default=PROFILE_GENERATION_MODEL_RAW,
        help="基座模型路径或 HuggingFace 模型名（默认 config.PROFILE_GENERATION_MODEL_RAW）",
    )
    parser.add_argument("--output", required=True, help="输出目录")

    # 训练超参（参考你给的参数风格）
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--beta", type=float, default=DPO_BETA, help="DPO beta")
    parser.add_argument("--sft-weight", type=float, default=DPO_SFT_LOSS_WEIGHT, help="联合损失中的 SFT 权重")
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=4096,
        help="精炼任务含长旧画像+误差，默认放宽；显存不足可改小",
    )
    parser.add_argument(
        "--max-completion-length",
        type=int,
        default=2048,
        help="修订后 persona 可能较长；0 表示交给 TRL 不截断 completion",
    )
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--eval-strategy", default="steps")
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--optimizer", default="adamw_torch")
    parser.add_argument(
        "--allow-inverted-pairs",
        action="store_true",
        help="保留 chosen.r_all<=rejected.r_all 的样本（默认跳过，避免 DPO 标签反了）",
    )

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.2)
    parser.add_argument("--lora-target", default="all", help="LoRA target modules，默认 all")

    # 精度
    parser.add_argument("--bf16", action="store_true", help="强制 bf16（需设备支持）")
    parser.add_argument("--fp16", action="store_true", help="强制 fp16（需 CUDA）")
    parser.add_argument(
        "--log-grad-every",
        type=int,
        default=0,
        help="每 N 次优化步（已含 grad accum）在 optimizer.step 前打印梯度 L2；0=关闭，不影响更新",
    )
    parser.add_argument(
        "--log-grad-lora-only",
        action="store_true",
        help="与 --log-grad-every 联用：只统计名字含 lora 的参数梯度",
    )
    args = parser.parse_args()

    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, TaskType
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import DPOConfig, DPOTrainer
    except ImportError as exc:
        print("缺少依赖，请安装: pip install trl transformers datasets accelerate peft", file=sys.stderr)
        raise SystemExit(1) from exc

    rows = _load_preference_rows(
        Path(args.data), allow_inverted_pairs=args.allow_inverted_pairs
    )
    if not rows:
        print("无可用训练样本。", file=sys.stderr)
        raise SystemExit(2)

    # 划分训练/验证
    n_total = len(rows)
    n_eval = int(round(n_total * max(0.0, min(0.9, args.eval_ratio))))
    if n_eval >= n_total:
        n_eval = max(0, n_total - 1)
    train_rows = rows[n_eval:]
    eval_rows = rows[:n_eval] if n_eval > 0 else []
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows) if eval_rows else None

    # GPU/精度策略
    has_cuda = torch.cuda.is_available()
    bf16_supported = bool(has_cuda and torch.cuda.is_bf16_supported())
    if args.bf16 and args.fp16:
        print("请勿同时指定 --bf16 和 --fp16", file=sys.stderr)
        raise SystemExit(2)
    if args.bf16:
        if not has_cuda or not bf16_supported:
            print("当前环境不支持 bf16", file=sys.stderr)
            raise SystemExit(2)
        use_bf16, use_fp16 = True, False
    elif args.fp16:
        if not has_cuda:
            print("当前环境不支持 fp16（未检测到 CUDA）", file=sys.stderr)
            raise SystemExit(2)
        use_bf16, use_fp16 = False, True
    else:
        if has_cuda:
            use_bf16, use_fp16 = (True, False) if bf16_supported else (False, True)
        else:
            use_bf16, use_fp16 = False, False

    print(
        f"[Train] CUDA={has_cuda}, bf16_supported={bf16_supported}, "
        f"bf16={use_bf16}, fp16={use_fp16}",
        flush=True,
    )
    print(
        f"[Train] 混合损失已启用: loss_type={['sigmoid', 'sft']} "
        f"loss_weights={[1.0, args.sft_weight]}",
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, Any] = {}
    if has_cuda:
        model_kwargs["device_map"] = "auto"
        if use_bf16:
            model_kwargs["torch_dtype"] = torch.bfloat16
        elif use_fp16:
            model_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

    target_modules = "all-linear" if args.lora_target == "all" else args.lora_target
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )

    dpo_args = DPOConfig(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        optim=args.optimizer,
        beta=args.beta,
        loss_type=["sigmoid", "sft"],
        loss_weights=[1.0, args.sft_weight],
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length if args.max_completion_length > 0 else None,
        bf16=use_bf16,
        fp16=use_fp16,
        gradient_checkpointing=not args.no_gradient_checkpointing,
    )

    grad_cb: _GradNormLogCallback | None = None
    callbacks = None
    if args.log_grad_every > 0:
        grad_cb = _GradNormLogCallback(args.log_grad_every, args.log_grad_lora_only)
        callbacks = [grad_cb]

    # TRL：PEFT + DPO 时不要传独立 ref_model（否则会 ValueError）；参考策略由 trainer 内部处理
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=callbacks,
    )
    if grad_cb is not None:
        grad_cb.bind_trainer(trainer)

    # PEFT 官方建议：get_peft_model 之后再 enable_input_require_grads，配合 gradient checkpointing
    if hasattr(trainer.model, "enable_input_require_grads"):
        trainer.model.enable_input_require_grads()

    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(
        f"[Done] 训练完成，模型已保存: {args.output} "
        f"(train={len(train_rows)}, eval={len(eval_rows)})",
        flush=True,
    )


if __name__ == "__main__":
    main()

