from pathlib import Path

from . import common as _common


def _input_path() -> Path | None:
    candidates = [
        _common.RAW_ROOT / "morebench" / "morebench_theory.csv",
        _common.RAW_ROOT / "morebench_theory.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    print("Missing MoreBench theory file. Checked:")
    for candidate in candidates:
        print(f"- {candidate}")
    return None


def main() -> None:
    path = _input_path()
    if path is None:
        return

    rows = _common.read_csv_rows(path, errors="replace")
    for row in rows:
        row["source"] = "morebench_theory"
        row["source_file"] = str(path.relative_to(_common.RAW_ROOT))

    out_dir = _common.PROCESSED_ROOT / "morebench_theory"
    _common.write_jsonl(out_dir / "morebench_theory.jsonl", rows)
    _common.write_csv(out_dir / "morebench_theory.csv", rows)
    print(f"Wrote {len(rows)} records to {out_dir}")


if __name__ == "__main__":
    main()
