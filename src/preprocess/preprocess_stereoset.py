from pathlib import Path
import json

import _common


def _load_dev(path: Path, section: str):
    rows = []
    obj = json.loads(path.read_text(encoding="utf-8"))
    for item in obj.get("data", {}).get(section, []):
        context_id = item.get("id")
        for sent in item.get("sentences", []):
            rows.append(
                {
                    "id": sent.get("id"),
                    "context_id": context_id,
                    "context": item.get("context"),
                    "sentence": sent.get("sentence"),
                    "gold_label": sent.get("gold_label"),
                    "bias_type": item.get("bias_type"),
                    "target": item.get("target"),
                    "section": section,
                    "split": "dev",
                    "labels": sent.get("labels"),
                }
            )
    return rows


def main() -> None:
    data_dir = _common.RAW_ROOT / "StereoSet-master" / "data"
    dev_path = data_dir / "dev.json"
    terms_path = data_dir / "test_terms.txt"

    if not dev_path.exists():
        print(f"Missing file: {dev_path}")
        return

    rows = []
    rows.extend(_load_dev(dev_path, "intersentence"))
    rows.extend(_load_dev(dev_path, "intrasentence"))

    out_dir = _common.PROCESSED_ROOT / "stereoset"
    _common.write_jsonl(out_dir / "dev.jsonl", rows)
    _common.write_csv(out_dir / "dev.csv", rows)
    print(f"Wrote {len(rows)} dev records to {out_dir}")

    if terms_path.exists():
        term_rows = []
        with terms_path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                term = line.strip()
                if not term:
                    continue
                term_rows.append(
                    {
                        "term": term,
                        "source": "stereoset",
                        "source_file": str(terms_path.relative_to(_common.RAW_ROOT)),
                    }
                )
        _common.write_jsonl(out_dir / "test_terms.jsonl", term_rows)
        _common.write_csv(out_dir / "test_terms.csv", term_rows)
        print(f"Wrote {len(term_rows)} test term records to {out_dir}")


if __name__ == "__main__":
    main()
