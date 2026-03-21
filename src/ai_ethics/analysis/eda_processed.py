from __future__ import annotations

from pathlib import Path
import csv
import json


ROOT = Path(__file__).resolve().parents[3]


def _data_root() -> Path:
    for candidate in (ROOT / "Data", ROOT / "data"):
        if candidate.exists():
            return candidate
    return ROOT / "Data"


DATA_ROOT = _data_root()
PROCESSED_DIR = DATA_ROOT / "processed"
OUT_DIR = DATA_ROOT / "eda" / "processed"


def _count_csv(path: Path) -> tuple[int, list[str]]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
        rows = sum(1 for _ in reader)
    return rows, header


def _count_jsonl(path: Path) -> tuple[int, list[str]]:
    rows = 0
    keys: list[str] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows += 1
            if rows == 1:
                try:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        keys = list(obj.keys())
                except json.JSONDecodeError:
                    pass
    return rows, keys


def _count_parquet(path: Path) -> tuple[int, list[str]]:
    try:
        import pyarrow.parquet as pq
    except Exception:
        return 0, []
    pf = pq.ParquetFile(path)
    return pf.metadata.num_rows, pf.schema.names


def _file_summary(path: Path) -> dict:
    info: dict = {
        "file": str(path.relative_to(PROCESSED_DIR)),
        "bytes": path.stat().st_size,
        "rows": 0,
        "columns": [],
        "type": path.suffix.lower().lstrip("."),
    }
    suffix = path.suffix.lower()
    if suffix == ".csv":
        rows, cols = _count_csv(path)
        info["rows"] = rows
        info["columns"] = cols
    elif suffix == ".jsonl":
        rows, cols = _count_jsonl(path)
        info["rows"] = rows
        info["columns"] = cols
    elif suffix == ".parquet":
        rows, cols = _count_parquet(path)
        info["rows"] = rows
        info["columns"] = cols
    return info


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not PROCESSED_DIR.exists():
        print(f"Missing processed dir: {PROCESSED_DIR}")
        return

    files = [p for p in PROCESSED_DIR.rglob("*") if p.is_file()]
    file_rows = []
    for p in files:
        info = _file_summary(p)
        file_rows.append(info)

    with (OUT_DIR / "processed_file_summary.json").open("w", encoding="utf-8") as f:
        json.dump(file_rows, f, indent=2)

    with (OUT_DIR / "processed_file_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["file", "bytes", "rows", "columns", "type"],
        )
        writer.writeheader()
        for row in file_rows:
            out = dict(row)
            if isinstance(out.get("columns"), list):
                out["columns"] = ",".join(out["columns"])
            writer.writerow(out)

    print(f"Wrote processed EDA summaries to {OUT_DIR}")


if __name__ == "__main__":
    main()
