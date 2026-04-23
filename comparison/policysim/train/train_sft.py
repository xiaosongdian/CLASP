import argparse
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments


def format_row(example: dict) -> str:
    return (
        f"Instruction: {example['instruction']}\n"
        f"Input: {example['input']}\n"
        f"Output: {example['output']}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="SFT training for PolicySim agent model.")
    parser.add_argument("--model", required=True, help="Base model name or local path")
    parser.add_argument("--data", required=True, help="SFT JSONL file path")
    parser.add_argument("--output", required=True, help="Output model directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    dataset = load_dataset("json", data_files=args.data)["train"]
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(example: dict) -> dict:
        text = format_row(example)
        tokenized = tokenizer(text, truncation=True, max_length=1024)
        tokenized["labels"] = tokenized["input_ids"][:]
        return tokenized

    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_strategy="epoch",
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
    )
    trainer.train()
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"SFT training done. Model saved to {args.output}")


if __name__ == "__main__":
    main()

