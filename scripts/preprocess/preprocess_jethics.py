from pathlib import Path

import _common


def _parse_name(stem: str):
    if "_" not in stem:
        return stem, "", ""
    task, split_info = stem.split("_", 1)
    split = "train" if split_info.startswith("train") else "test" if split_info.startswith("test") else ""
    size = "".join(ch for ch in split_info if ch.isdigit())
    return task, split, size


def main() -> None:
    base = _common.RAW_ROOT / "jethics-main" / "samples"
    if not base.exists():
        print(f"Missing dataset root: {base}")
        return

    rows = []
    for path in sorted(base.rglob("*.csv")):
        rel = path.relative_to(_common.RAW_ROOT)
        task, split, sample_size = _parse_name(path.stem)
        for row in _common.read_csv_rows(path, errors="replace"):
            row = dict(row)
            row.update(
                {
                    "task": task,
                    "split": split,
                    "sample_size": sample_size,
                    "source": "jethics",
                    "source_file": str(rel),
                }
            )
            rows.append(row)

    out_dir = _common.PROCESSED_ROOT / "jethics"
    _common.write_jsonl(out_dir / "jethics.jsonl", rows)
    _common.write_csv(out_dir / "jethics.csv", rows)
    print(f"Wrote {len(rows)} records to {out_dir}")


if __name__ == "__main__":
    main()
