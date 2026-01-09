from pathlib import Path

import _common


def _collect_files(data_dir: Path):
    exts = {".csv", ".tsv", ".json", ".jsonl"}
    files = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            files.append(path)
    return files


def _read_rows(path: Path):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _common.read_csv_rows(path, errors="replace")
    if suffix == ".tsv":
        return _common.read_tsv_rows(path, errors="replace")
    if suffix == ".jsonl":
        return _common.read_jsonl_rows(path, errors="replace")
    if suffix == ".json":
        return _common.read_json_rows(path, errors="replace")
    return []


def main() -> None:
    base = _common.RAW_ROOT / "social-chemistry-101-main"
    if not base.exists():
        print(f"Missing dataset root: {base}")
        return

    data_dir = base / "data"
    if not data_dir.exists():
        print("No data directory found under social-chemistry-101-main. Dataset files appear missing.")
        return

    files = _collect_files(data_dir)
    if not files:
        print("No dataset files found under social-chemistry-101-main/data")
        return

    for path in files:
        rows = _read_rows(path)
        rel = path.relative_to(base)
        for row in rows:
            row["source_file"] = str(rel)
        out_base = (_common.PROCESSED_ROOT / "social_chemistry_101" / rel).with_suffix("")
        _common.write_jsonl(out_base.with_suffix(".jsonl"), rows)
        _common.write_csv(out_base.with_suffix(".csv"), rows)
        print(f"Wrote {len(rows)} records from {rel}")


if __name__ == "__main__":
    main()
