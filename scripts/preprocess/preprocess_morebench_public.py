from pathlib import Path

import _common


def main() -> None:
    path = _common.RAW_ROOT / "morebench_public.csv"
    if not path.exists():
        print(f"Missing file: {path}")
        return

    rows = _common.read_csv_rows(path, errors="replace")
    for row in rows:
        row["source"] = "morebench_public"
        row["source_file"] = str(path.relative_to(_common.RAW_ROOT))

    out_dir = _common.PROCESSED_ROOT / "morebench_public"
    _common.write_jsonl(out_dir / "morebench_public.jsonl", rows)
    _common.write_csv(out_dir / "morebench_public.csv", rows)
    print(f"Wrote {len(rows)} records to {out_dir}")


if __name__ == "__main__":
    main()
