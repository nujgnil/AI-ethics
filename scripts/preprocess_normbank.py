from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pyarrow.parquet as pq

from preprocess_utils import write_csv, write_jsonl, summarize_labels


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "normbank"
OUT_DIR = ROOT / "data" / "processed" / "normbank"


def _read_parquet(path: Path) -> List[Dict[str, Any]]:
    table = pq.read_table(path)
    return table.to_pylist()


def main() -> None:
    if not RAW_DIR.exists():
        print(f"Missing dataset root: {RAW_DIR}")
        return

    rows: List[Dict[str, Any]] = []
    for path in sorted(RAW_DIR.glob("*.parquet")):
        split = path.stem
        file_rows = _read_parquet(path)
        for row in file_rows:
            text = str(row.get("norm", "")).strip()
            label = row.get("label")
            out_row = {
                "text": text,
                "label": "" if label is None else str(label),
                "dataset": "normbank",
                "task": "norm_classification",
                "split": split,
                "source_file": str(path.relative_to(RAW_DIR)),
                "metadata": {k: v for k, v in row.items() if k not in {"norm", "label"}},
            }
            rows.append(out_row)

    write_jsonl(OUT_DIR / "normbank.jsonl", rows)
    write_csv(OUT_DIR / "normbank.csv", rows)
    label_summary = summarize_labels(rows)
    write_jsonl(OUT_DIR / "label_summary.jsonl", [{"labels": label_summary}])
    write_csv(OUT_DIR / "label_summary.csv", [{"labels": label_summary}])

    print(f"Wrote {len(rows)} records to {OUT_DIR}")


if __name__ == "__main__":
    main()
