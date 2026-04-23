import argparse
import json
from pathlib import Path


def build_sft_sample(event: str, profile: dict, action: str, content: str) -> dict:
    instruction = "根据事件和用户画像，生成用户在社交平台上的动作与文本。"
    model_input = {
        "event": event,
        "profile": profile,
    }
    model_output = {
        "action": action,
        "content": content,
    }
    return {
        "instruction": instruction,
        "input": json.dumps(model_input, ensure_ascii=False),
        "output": json.dumps(model_output, ensure_ascii=False),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build SFT dataset for PolicySim agent.")
    parser.add_argument("--source", required=True, help="Raw event-user-action JSON file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    src_path = Path(args.source)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = json.loads(src_path.read_text(encoding="utf-8"))
    with out_path.open("w", encoding="utf-8") as f:
        for row in records:
            sample = build_sft_sample(
                event=row["event"],
                profile=row["profile"],
                action=row["action"],
                content=row.get("content", ""),
            )
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"SFT dataset generated: {out_path}")


if __name__ == "__main__":
    main()

