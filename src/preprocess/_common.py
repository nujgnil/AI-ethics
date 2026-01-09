from __future__ import annotations

from pathlib import Path
import csv
import json
from typing import Any, Dict, Iterable, List


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


RAW_ROOT = repo_root() / "data" / "raw"
PROCESSED_ROOT = repo_root() / "data" / "processed"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_for_csv(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str] | None = None) -> None:
    ensure_dir(path.parent)
    if not rows:
        if fieldnames:
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
        return
    if fieldnames is None:
        keys: List[str] = []
        seen = set()
        for row in rows:
            for key in row.keys():
                if key not in seen:
                    seen.add(key)
                    keys.append(key)
        fieldnames = keys
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {k: normalize_for_csv(row.get(k)) for k in fieldnames}
            writer.writerow(out)


def read_csv_rows(path: Path, encoding: str = "utf-8", errors: str = "strict") -> List[Dict[str, Any]]:
    with path.open("r", encoding=encoding, errors=errors, newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def read_tsv_rows(path: Path, encoding: str = "utf-8", errors: str = "strict") -> List[Dict[str, Any]]:
    with path.open("r", encoding=encoding, errors=errors, newline="") as f:
        reader = csv.DictReader(f, delimiter="	")
        return list(reader)


def read_json_rows(path: Path, encoding: str = "utf-8", errors: str = "strict") -> List[Dict[str, Any]]:
    with path.open("r", encoding=encoding, errors=errors) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return [item if isinstance(item, dict) else {"value": item} for item in obj]
    if isinstance(obj, dict):
        return [obj]
    return [{"value": obj}]


def read_jsonl_rows(path: Path, encoding: str = "utf-8", errors: str = "strict") -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding=encoding, errors=errors) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
            else:
                rows.append({"value": obj})
    return rows
