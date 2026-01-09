from pathlib import Path

import _common


def _split_from_name(name: str):
    for suffix in ["train", "test_hard", "test", "ambig"]:
        needle = f"_{suffix}"
        if name.endswith(needle):
            return name[: -len(needle)], suffix
    return name, ""


def main() -> None:
    base = _common.RAW_ROOT / "hendryicks-ethics"
    if not base.exists():
        print(f"Missing dataset root: {base}")
        return

    rows = []
    for path in sorted(base.rglob("*.csv")):
        rel = path.relative_to(base)
        category = rel.parts[0]
        name = path.stem
        task, split = _split_from_name(name)
        for row in _common.read_csv_rows(path, errors="replace"):
            row = dict(row)
            row.update(
                {
                    "task": task,
                    "split": split,
                    "category": category,
                    "source": "hendryicks-ethics",
                    "source_file": str(rel),
                }
            )
            rows.append(row)

    out_dir = _common.PROCESSED_ROOT / "hendryicks_ethics"
    _common.write_jsonl(out_dir / "hendryicks_ethics.jsonl", rows)
    _common.write_csv(out_dir / "hendryicks_ethics.csv", rows)
    print(f"Wrote {len(rows)} records to {out_dir}")


if __name__ == "__main__":
    main()
