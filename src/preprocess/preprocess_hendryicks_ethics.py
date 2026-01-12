from pathlib import Path
from collections import Counter, defaultdict
import hashlib
import re

import _common


def _split_from_name(name: str):
    for suffix in ["train", "test_hard", "test", "ambig"]:
        needle = f"_{suffix}"
        if name.endswith(needle):
            return name[: -len(needle)], suffix
    return name, ""


def _pick_first_field(row, candidates):
    for key in candidates:
        if key in row and row.get(key) not in (None, ""):
            return key, row.get(key)
    return "", ""


def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def main() -> None:
    base = _common.RAW_ROOT / "hendryicks-ethics"
    if not base.exists():
        print(f"Missing dataset root: {base}")
        return

    rows = []
    summary_rows = []
    dup_index = defaultdict(list)

    text_fields = [
        "input",
        "scenario",
        "prompt",
        "question",
        "text",
        "sentence",
        "story",
        "statement",
    ]
    label_fields = ["label", "answer", "gold", "gold_label", "target", "output"]

    for path in sorted(base.rglob("*.csv")):
        rel = path.relative_to(base)
        category = rel.parts[0]
        name = path.stem
        task, split = _split_from_name(name)
        file_rows = _common.read_csv_rows(path, errors="replace")
        label_counts = Counter()
        text_lengths = []
        missing_text = 0
        missing_label = 0
        total = 0

        for row in file_rows:
            row = dict(row)
            total += 1
            text_key, text_value = _pick_first_field(row, text_fields)
            label_key, label_value = _pick_first_field(row, label_fields)

            if text_value in (None, ""):
                missing_text += 1
            else:
                norm_text = _normalize_text(str(text_value))
                text_lengths.append(len(norm_text))
                text_hash = _hash_text(norm_text)
                dup_index[text_hash].append(
                    {
                        "category": category,
                        "task": task,
                        "split": split,
                        "source_file": str(rel),
                        "text_field": text_key,
                    }
                )
                row["text_normalized"] = norm_text
                row["text_hash"] = text_hash

            if label_value in (None, ""):
                missing_label += 1
            else:
                label_counts[str(label_value).strip()] += 1
                row["label_normalized"] = str(label_value).strip()

            row.update(
                {
                    "task": task,
                    "split": split,
                    "category": category,
                    "source": "hendryicks-ethics",
                    "source_file": str(rel),
                }
            )
            rows.append(row)

        avg_len = round(sum(text_lengths) / len(text_lengths), 2) if text_lengths else 0
        summary_rows.append(
            {
                "category": category,
                "task": task,
                "split": split,
                "source_file": str(rel),
                "rows": total,
                "missing_text": missing_text,
                "missing_label": missing_label,
                "avg_text_len": avg_len,
                "label_distribution": dict(label_counts),
            }
        )

    dup_rows = []
    for text_hash, entries in dup_index.items():
        if len(entries) < 2:
            continue
        dup_rows.append(
            {
                "text_hash": text_hash,
                "occurrences": len(entries),
                "locations": entries,
            }
        )

    out_dir = _common.PROCESSED_ROOT / "hendryicks_ethics"
    _common.write_jsonl(out_dir / "hendryicks_ethics.jsonl", rows)
    _common.write_csv(out_dir / "hendryicks_ethics.csv", rows)
    _common.write_jsonl(out_dir / "summary.jsonl", summary_rows)
    _common.write_csv(out_dir / "summary.csv", summary_rows)
    _common.write_jsonl(out_dir / "dup_report.jsonl", dup_rows)
    _common.write_csv(out_dir / "dup_report.csv", dup_rows)
    print(f"Wrote {len(rows)} records to {out_dir}")


if __name__ == "__main__":
    main()
