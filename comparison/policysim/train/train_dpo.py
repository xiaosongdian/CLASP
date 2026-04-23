import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments


def main() -> None:
    parser = argparse.ArgumentParser(description="DPO training for PolicySim agent model.")
    parser.add_argument("--model", required=True, help="SFT model path or model name")
    parser.add_argument("--data", required=True, help="DPO JSONL file path")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.1)
    args = parser.parse_args()

    try:
        from trl import DPOTrainer
    except ImportError as exc:
        raise ImportError(
            "缺少 trl 依赖，请先执行: pip install trl"
        ) from exc

    dataset = load_dataset("json", data_files=args.data)["train"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        beta=args.beta,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"DPO training done. Model saved to {args.output}")


if __name__ == "__main__":
    main()

