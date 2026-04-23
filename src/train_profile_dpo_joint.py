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

from src.config import DPO_BETA, DPO_SFT_LOSS_WEIGHT, PROFILE_GENERATION_MODEL_RAW


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


def _load_preference_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "prompt" not in obj:
                print("[Warn] 跳过无 prompt 字段样本", flush=True)
                continue
            chosen = obj.get("chosen")
            rejected = obj.get("rejected")
            if not isinstance(chosen, dict) or not isinstance(rejected, dict):
                print("[Warn] 跳过 chosen/rejected 非对象结构样本", flush=True)
                continue
            if "profile" not in chosen or "profile" not in rejected:
                print("[Warn] 跳过缺少 chosen/rejected.profile 的样本", flush=True)
                continue
            rows.append(
                {
                    "prompt": _prompt_to_text(obj["prompt"]),
                    "chosen": str(chosen["profile"]),
                    "rejected": str(rejected["profile"]),
                }
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
    parser.add_argument("--max-prompt-length", type=int, default=1408)
    parser.add_argument("--max-completion-length", type=int, default=640)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--warmup-steps", type=int, default=250)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--eval-strategy", default="steps")
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--optimizer", default="adamw_torch")

    # LoRA
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.2)
    parser.add_argument("--lora-target", default="all", help="LoRA target modules，默认 all")

    # 精度
    parser.add_argument("--bf16", action="store_true", help="强制 bf16（需设备支持）")
    parser.add_argument("--fp16", action="store_true", help="强制 fp16（需 CUDA）")
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

    rows = _load_preference_rows(Path(args.data))
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
    ref_model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)

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
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
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

