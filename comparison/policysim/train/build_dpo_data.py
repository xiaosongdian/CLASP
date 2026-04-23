import argparse
import json
from pathlib import Path


def build_dpo_sample(event: str, profile: dict, chosen: dict, rejected: dict) -> dict:
    prompt = json.dumps({"event": event, "profile": profile}, ensure_ascii=False)
    chosen_text = json.dumps(chosen, ensure_ascii=False)
    rejected_text = json.dumps(rejected, ensure_ascii=False)
    return {"prompt": prompt, "chosen": chosen_text, "rejected": rejected_text}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build DPO preference dataset for PolicySim agent.")
    parser.add_argument("--source", required=True, help="Raw preference JSON file path")
    parser.add_argument("--output", required=True, help="Output JSONL file path")
    args = parser.parse_args()

    src_path = Path(args.source)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records = json.loads(src_path.read_text(encoding="utf-8"))
    with out_path.open("w", encoding="utf-8") as f:
        for row in records:
            sample = build_dpo_sample(
                event=row["event"],
                profile=row["profile"],
                chosen=row["chosen"],
                rejected=row["rejected"],
            )
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"DPO dataset generated: {out_path}")


if __name__ == "__main__":
    main()

