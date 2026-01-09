from pathlib import Path

import _common


def _read_rows(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _common.read_csv_rows(path, errors="replace")
    if suffix == ".jsonl":
        return _common.read_jsonl_rows(path, errors="replace")
    if suffix == ".json":
        return _common.read_json_rows(path, errors="replace")
    return []


def main() -> None:
    dataset_root = _common.RAW_ROOT / "do-not-answer-main"
    if not dataset_root.exists():
        print(f"Missing dataset root: {dataset_root}")
        return

    data_dirs = [dataset_root / "datasets", dataset_root / "cdna"]
    files = []
    for data_dir in data_dirs:
        if not data_dir.exists():
            continue
        for path in sorted(data_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in {".csv", ".json", ".jsonl"}:
                files.append(path)

    if not files:
        print("No data files found under do-not-answer-main/datasets or do-not-answer-main/cdna")
        return

    for path in files:
        rows = _read_rows(path)
        if not rows:
            print(f"No rows parsed from {path}")
            continue
        rel = path.relative_to(dataset_root)
        for row in rows:
            row["source_file"] = str(rel)
        out_base = (_common.PROCESSED_ROOT / "do_not_answer" / rel).with_suffix("")
        _common.write_jsonl(out_base.with_suffix(".jsonl"), rows)
        _common.write_csv(out_base.with_suffix(".csv"), rows)
        print(f"Wrote {len(rows)} records from {rel}")


if __name__ == "__main__":
    main()
