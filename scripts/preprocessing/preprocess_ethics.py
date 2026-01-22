from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import csv
import hashlib
import json
import re

from preprocess_utils import write_csv, write_jsonl, summarize_labels


ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = ROOT / "data" / "raw" / "hendryicks-ethics"
OUT_DIR = ROOT / "data" / "processed" / "ethics"


TEXT_FIELDS = [
    "input",
    "scenario",
    "prompt",
    "question",
    "text",
    "sentence",
    "story",
    "statement",
]
LABEL_FIELDS = ["label", "answer", "gold", "gold_label", "target", "output"]


def _split_from_name(name: str) -> tuple[str, str]:
    for suffix in ["train", "test_hard", "test", "ambig"]:
        needle = f"_{suffix}"
        if name.endswith(needle):
            return name[: -len(needle)], suffix
    return name, ""


def _pick_first_field(row: Dict[str, Any], candidates: List[str]) -> tuple[str, Any]:
    for key in candidates:
        if key in row and row.get(key) not in (None, ""):
            return key, row.get(key)
    return "", ""


def _normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def _read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    if not RAW_ROOT.exists():
        print(f"Missing dataset root: {RAW_ROOT}")
        return

    rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    dup_index: Dict[str, List[Dict[str, Any]]] = {}

    for path in sorted(RAW_ROOT.rglob("*.csv")):
        rel = path.relative_to(RAW_ROOT)
        category = rel.parts[0]
        name = path.stem
        task, split = _split_from_name(name)
        file_rows = _read_csv(path)
        label_counts: Dict[str, int] = {}
        text_lengths: List[int] = []
        missing_text = 0
        missing_label = 0

        for row in file_rows:
            row = dict(row)
            text_key, text_value = _pick_first_field(row, TEXT_FIELDS)
            label_key, label_value = _pick_first_field(row, LABEL_FIELDS)

            text_hash = ""
            norm_text = ""
            if text_value in (None, ""):
                missing_text += 1
            else:
                norm_text = _normalize_text(str(text_value))
                text_lengths.append(len(norm_text))
                text_hash = _hash_text(norm_text)
                dup_index.setdefault(text_hash, []).append(
                    {
                        "category": category,
                        "task": task,
                        "split": split,
                        "source_file": str(rel),
                        "text_field": text_key,
                    }
                )

            if label_value in (None, ""):
                missing_label += 1
                label_norm = ""
            else:
                label_norm = str(label_value).strip()
                label_counts[label_norm] = label_counts.get(label_norm, 0) + 1

            out_row = {
                "text": norm_text,
                "label": label_norm,
                "dataset": "ethics",
                "task": task,
                "split": split,
                "source_file": str(rel),
                "metadata": {
                    "category": category,
                    "text_field": text_key,
                    "label_field": label_key,
                    "text_hash": text_hash,
                },
            }
            rows.append(out_row)

        avg_len = round(sum(text_lengths) / len(text_lengths), 2) if text_lengths else 0
        summary_rows.append(
            {
                "category": category,
                "task": task,
                "split": split,
                "source_file": str(rel),
                "rows": len(file_rows),
                "missing_text": missing_text,
                "missing_label": missing_label,
                "avg_text_len": avg_len,
                "label_distribution": label_counts,
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

    write_jsonl(OUT_DIR / "ethics.jsonl", rows)
    write_csv(OUT_DIR / "ethics.csv", rows)
    write_jsonl(OUT_DIR / "summary.jsonl", summary_rows)
    write_csv(OUT_DIR / "summary.csv", summary_rows)
    write_jsonl(OUT_DIR / "dup_report.jsonl", dup_rows)
    write_csv(OUT_DIR / "dup_report.csv", dup_rows)

    label_summary = summarize_labels(rows)
    write_jsonl(OUT_DIR / "label_summary.jsonl", [{"labels": label_summary}])
    write_csv(OUT_DIR / "label_summary.csv", [{"labels": label_summary}])

    print(f"Wrote {len(rows)} records to {OUT_DIR}")


if __name__ == "__main__":
    main()
