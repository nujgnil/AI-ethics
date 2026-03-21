from pathlib import Path

from . import common as _common


def _from_normbank_csv(path: Path):
    rows = []
    for row in _common.read_csv_rows(path, errors="replace"):
        metadata = {
            key: row.get(key, "")
            for key in [
                "setting",
                "behavior",
                "constraints",
                "constraints_given",
                "constraint_predict",
            ]
            if row.get(key, "") not in (None, "")
        }
        rows.append(
            {
                "text": str(row.get("norm", "")).strip(),
                "label": str(row.get("label", "")).strip(),
                "dataset": "normbank",
                "task": "norm_classification",
                "split": str(row.get("split", "")).strip(),
                "source_file": path.name,
                "metadata": metadata,
            }
        )
    return rows


def _from_legacy_text_tree(base: Path):
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
    return rows


def main() -> None:
    csv_path = _common.RAW_ROOT / "normbank" / "NormBank.csv"
    legacy_base = _common.RAW_ROOT / "normbank-main" / "data" / "raw"

    if csv_path.exists():
        rows = _from_normbank_csv(csv_path)
    elif legacy_base.exists():
        rows = _from_legacy_text_tree(legacy_base)
    else:
        print(f"Missing NormBank raw data. Checked: {csv_path} and {legacy_base}")
        return

    out_dir = _common.PROCESSED_ROOT / "normbank"
    _common.write_jsonl(out_dir / "normbank.jsonl", rows)
    _common.write_csv(out_dir / "normbank.csv", rows)
    print(f"Wrote {len(rows)} records to {out_dir}")


if __name__ == "__main__":
    main()
