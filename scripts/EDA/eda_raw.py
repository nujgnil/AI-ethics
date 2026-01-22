from __future__ import annotations

from pathlib import Path
import csv
import json
import statistics


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "eda" / "raw"

TEXT_HINTS = {
    "text",
    "prompt",
    "question",
    "sentence",
    "story",
    "statement",
    "input",
    "scenario",
    "norm",
    "rule",
    "vignette",
    "post",
    "comment",
    "body",
}
LABEL_HINTS = {
    "label",
    "answer",
    "gold",
    "target",
    "class",
    "category",
    "judgment",
    "rating",
    "score",
    "acceptability",
}


def _pick_column(columns: list[str], hints: set[str]) -> str:
    lower = {c.lower(): c for c in columns}
    for name in columns:
        key = name.lower()
        if key in hints:
            return name
    for key, name in lower.items():
        for hint in hints:
            if hint in key:
                return name
    return ""


def _text_len_stats(lengths: list[int]) -> dict:
    if not lengths:
        return {"min": 0, "median": 0, "p95": 0}
    lengths_sorted = sorted(lengths)
    p95_index = int(len(lengths_sorted) * 0.95) - 1
    p95_index = max(0, min(p95_index, len(lengths_sorted) - 1))
    return {
        "min": lengths_sorted[0],
        "median": int(statistics.median(lengths_sorted)),
        "p95": lengths_sorted[p95_index],
    }


def _count_csv(path: Path, delimiter: str = ",") -> tuple[int, list[str], dict]:
    rows = 0
    columns: list[str] = []
    lengths: list[int] = []
    empty_text = 0
    empty_label = 0
    label_counts: dict[str, int] = {}
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        columns = reader.fieldnames or []
        text_col = _pick_column(columns, TEXT_HINTS)
        label_col = _pick_column(columns, LABEL_HINTS)
        for row in reader:
            rows += 1
            if text_col:
                value = (row.get(text_col) or "").strip()
                if not value:
                    empty_text += 1
                else:
                    lengths.append(len(value))
            if label_col:
                label = (row.get(label_col) or "").strip()
                if not label:
                    empty_label += 1
                else:
                    label_counts[label] = label_counts.get(label, 0) + 1
    stats = {
        "text_col": text_col,
        "label_col": label_col,
        "empty_text": empty_text,
        "empty_label": empty_label,
        "text_len": _text_len_stats(lengths),
        "label_counts": label_counts,
    }
    return rows, columns, stats


def _count_jsonl(path: Path) -> tuple[int, list[str], dict]:
    rows = 0
    keys: list[str] = []
    lengths: list[int] = []
    empty_text = 0
    empty_label = 0
    label_counts: dict[str, int] = {}
    text_col = ""
    label_col = ""
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rows == 1 and isinstance(obj, dict):
                keys = list(obj.keys())
                text_col = _pick_column(keys, TEXT_HINTS)
                label_col = _pick_column(keys, LABEL_HINTS)
            if isinstance(obj, dict):
                if text_col:
                    value = str(obj.get(text_col, "")).strip()
                    if not value:
                        empty_text += 1
                    else:
                        lengths.append(len(value))
                if label_col:
                    label = str(obj.get(label_col, "")).strip()
                    if not label:
                        empty_label += 1
                    else:
                        label_counts[label] = label_counts.get(label, 0) + 1
    stats = {
        "text_col": text_col,
        "label_col": label_col,
        "empty_text": empty_text,
        "empty_label": empty_label,
        "text_len": _text_len_stats(lengths),
        "label_counts": label_counts,
    }
    return rows, keys, stats


def _count_txt(path: Path) -> int:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def _count_parquet(path: Path) -> tuple[int, list[str], dict]:
    try:
        import pyarrow.parquet as pq
    except Exception:
        return 0, [], {}
    pf = pq.ParquetFile(path)
    columns = list(pf.schema.names)
    text_col = _pick_column(columns, TEXT_HINTS)
    label_col = _pick_column(columns, LABEL_HINTS)
    lengths: list[int] = []
    empty_text = 0
    empty_label = 0
    label_counts: dict[str, int] = {}
    cols_to_read = [c for c in [text_col, label_col] if c]
    if cols_to_read:
        table = pq.read_table(path, columns=cols_to_read)
        data = table.to_pydict()
        text_values = data.get(text_col, []) if text_col else []
        label_values = data.get(label_col, []) if label_col else []
        if text_col:
            for value in text_values:
                value = "" if value is None else str(value).strip()
                if not value:
                    empty_text += 1
                else:
                    lengths.append(len(value))
        if label_col:
            for value in label_values:
                value = "" if value is None else str(value).strip()
                if not value:
                    empty_label += 1
                else:
                    label_counts[value] = label_counts.get(value, 0) + 1
    stats = {
        "text_col": text_col,
        "label_col": label_col,
        "empty_text": empty_text,
        "empty_label": empty_label,
        "text_len": _text_len_stats(lengths),
        "label_counts": label_counts,
    }
    return pf.metadata.num_rows, columns, stats


def _file_summary(dataset: str, path: Path) -> dict:
    info: dict = {
        "dataset": dataset,
        "file": str(path.relative_to(RAW_DIR)),
        "bytes": path.stat().st_size,
        "rows": 0,
        "columns": [],
        "type": path.suffix.lower().lstrip("."),
        "text_col": "",
        "label_col": "",
        "text_len_min": 0,
        "text_len_median": 0,
        "text_len_p95": 0,
        "empty_text": 0,
        "empty_label": 0,
        "label_counts": {},
    }
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows, cols, stats = _count_csv(path, delimiter=",")
        info["rows"] = rows
        info["columns"] = cols
        info.update(
            {
                "text_col": stats["text_col"],
                "label_col": stats["label_col"],
                "text_len_min": stats["text_len"]["min"],
                "text_len_median": stats["text_len"]["median"],
                "text_len_p95": stats["text_len"]["p95"],
                "empty_text": stats["empty_text"],
                "empty_label": stats["empty_label"],
                "label_counts": stats["label_counts"],
            }
        )
    elif suffix == ".tsv":
        rows, cols, stats = _count_csv(path, delimiter="\t")
        info["rows"] = rows
        info["columns"] = cols
        info.update(
            {
                "text_col": stats["text_col"],
                "label_col": stats["label_col"],
                "text_len_min": stats["text_len"]["min"],
                "text_len_median": stats["text_len"]["median"],
                "text_len_p95": stats["text_len"]["p95"],
                "empty_text": stats["empty_text"],
                "empty_label": stats["empty_label"],
                "label_counts": stats["label_counts"],
            }
        )
    elif suffix == ".jsonl":
        rows, cols, stats = _count_jsonl(path)
        info["rows"] = rows
        info["columns"] = cols
        info.update(
            {
                "text_col": stats["text_col"],
                "label_col": stats["label_col"],
                "text_len_min": stats["text_len"]["min"],
                "text_len_median": stats["text_len"]["median"],
                "text_len_p95": stats["text_len"]["p95"],
                "empty_text": stats["empty_text"],
                "empty_label": stats["empty_label"],
                "label_counts": stats["label_counts"],
            }
        )
    elif suffix == ".parquet":
        rows, cols, stats = _count_parquet(path)
        info["rows"] = rows
        info["columns"] = cols
        info.update(
            {
                "text_col": stats["text_col"],
                "label_col": stats["label_col"],
                "text_len_min": stats["text_len"]["min"],
                "text_len_median": stats["text_len"]["median"],
                "text_len_p95": stats["text_len"]["p95"],
                "empty_text": stats["empty_text"],
                "empty_label": stats["empty_label"],
                "label_counts": stats["label_counts"],
            }
        )
    elif suffix in {".txt", ".dic"}:
        info["rows"] = _count_txt(path)
    return info


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {
        "hendrycks_ethics": RAW_DIR / "hendryicks-ethics",
        "normbank": RAW_DIR / "normbank",
        "mfd2": RAW_DIR / "mfd2",
        "mfrc": RAW_DIR / "mfrc",
        "moralbench": RAW_DIR / "moralbench",
    }

    file_rows = []
    dataset_rows = []

    for name, path in datasets.items():
        if not path.exists():
            dataset_rows.append(
                {
                    "dataset": name,
                    "status": "missing",
                    "files": 0,
                    "rows": 0,
                    "empty_text": 0,
                    "empty_label": 0,
                    "notes": "path not found",
                }
            )
            continue

        files = [p for p in path.rglob("*") if p.is_file()]
        total_rows = 0
        total_empty_text = 0
        total_empty_label = 0
        for p in files:
            info = _file_summary(name, p)
            total_rows += int(info.get("rows") or 0)
            total_empty_text += int(info.get("empty_text") or 0)
            total_empty_label += int(info.get("empty_label") or 0)
            file_rows.append(info)

        dataset_rows.append(
            {
                "dataset": name,
                "status": "ok",
                "files": len(files),
                "rows": total_rows,
                "empty_text": total_empty_text,
                "empty_label": total_empty_label,
                "notes": "",
            }
        )

    with (OUT_DIR / "raw_file_summary.json").open("w", encoding="utf-8") as f:
        json.dump(file_rows, f, indent=2)

    with (OUT_DIR / "raw_dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_rows, f, indent=2)

    with (OUT_DIR / "raw_file_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "file",
                "bytes",
                "rows",
                "columns",
                "type",
                "text_col",
                "label_col",
                "text_len_min",
                "text_len_median",
                "text_len_p95",
                "empty_text",
                "empty_label",
                "label_counts",
            ],
        )
        writer.writeheader()
        for row in file_rows:
            out = dict(row)
            if isinstance(out.get("columns"), list):
                out["columns"] = ",".join(out["columns"])
            if isinstance(out.get("label_counts"), dict):
                out["label_counts"] = json.dumps(out["label_counts"], ensure_ascii=True)
            writer.writerow(out)

    with (OUT_DIR / "raw_dataset_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "status",
                "files",
                "rows",
                "empty_text",
                "empty_label",
                "notes",
            ],
        )
        writer.writeheader()
        for row in dataset_rows:
            writer.writerow(row)

    print(f"Wrote raw EDA summaries to {OUT_DIR}")


if __name__ == "__main__":
    main()
