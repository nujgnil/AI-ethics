from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def build_replay(input_csv: Path, output_jsonl: Path) -> dict[str, int]:
    total = 0
    filled = 0
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with input_csv.open("r", encoding="utf-8", newline="") as in_fh, output_jsonl.open(
        "w", encoding="utf-8", newline=""
    ) as out_fh:
        reader = csv.DictReader(in_fh)
        for row in reader:
            item_id = (row.get("item_id") or "").strip()
            response_text = (row.get("manual_response") or "").strip()
            if not item_id:
                continue
            total += 1
            if response_text:
                filled += 1
            out_fh.write(json.dumps({"item_id": item_id, "response_text": response_text}, ensure_ascii=False) + "\n")
    return {"total": total, "filled": filled}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a replay JSONL file from a prompts CSV with manual_response filled in.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    stats = build_replay(Path(args.input_csv), Path(args.output_jsonl))
    print(json.dumps({"input_csv": args.input_csv, "output_jsonl": args.output_jsonl, **stats}, ensure_ascii=False))


if __name__ == "__main__":
    main()
