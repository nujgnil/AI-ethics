from pathlib import Path

import _common


def main() -> None:
    path = _common.RAW_ROOT / "real-toxicity-prompts-master" / "data" / "list_of_naughty_and_bad_words.txt"
    if not path.exists():
        print(f"Missing file: {path}")
        return

    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            rows.append(
                {
                    "text": text,
                    "source": "real-toxicity-prompts",
                    "source_file": str(path.relative_to(_common.RAW_ROOT)),
                }
            )

    out_dir = _common.PROCESSED_ROOT / "real_toxicity_prompts"
    _common.write_jsonl(out_dir / "wordlist.jsonl", rows)
    _common.write_csv(out_dir / "wordlist.csv", rows)
    print(f"Wrote {len(rows)} records to {out_dir}")


if __name__ == "__main__":
    main()
