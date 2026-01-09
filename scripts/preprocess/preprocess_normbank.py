from pathlib import Path

import _common


def main() -> None:
    base = _common.RAW_ROOT / "normbank-main" / "data" / "raw"
    if not base.exists():
        print(f"Missing dataset root: {base}")
        return

    rows = []
    for path in sorted(base.rglob("*.txt")):
        rel = path.relative_to(base)
        category = rel.parts[0]
        subcategory = "/".join(rel.parts[1:-1]) if len(rel.parts) > 2 else ""
        file_name = rel.stem
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                rows.append(
                    {
                        "text": text,
                        "category": category,
                        "subcategory": subcategory,
                        "file": file_name,
                        "source": "normbank",
                        "source_file": str(rel),
                    }
                )

    out_dir = _common.PROCESSED_ROOT / "normbank"
    _common.write_jsonl(out_dir / "normbank.jsonl", rows)
    _common.write_csv(out_dir / "normbank.csv", rows)
    print(f"Wrote {len(rows)} records to {out_dir}")


if __name__ == "__main__":
    main()
