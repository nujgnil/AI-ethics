from pathlib import Path

import _common


def _normalize_id(rows):
    out = []
    for idx, row in enumerate(rows):
        row = dict(row)
        raw_id = row.get("") or row.get("Unnamed: 0")
        if "" in row:
            row.pop("")
        if "Unnamed: 0" in row:
            row.pop("Unnamed: 0")
        row["id"] = raw_id if raw_id not in (None, "") else str(idx)
        out.append(row)
    return out


def main() -> None:
    data_dir = _common.RAW_ROOT / "crows-pairs-master" / "data"
    pairs_path = data_dir / "crows_pairs_anonymized.csv"
    prompts_path = data_dir / "prompts.csv"

    if not pairs_path.exists():
        print(f"Missing file: {pairs_path}")
        return

    pairs_rows = _normalize_id(_common.read_csv_rows(pairs_path))
    for row in pairs_rows:
        row["source"] = "crows-pairs"
        row["source_file"] = str(pairs_path.relative_to(_common.RAW_ROOT))

    out_dir = _common.PROCESSED_ROOT / "crows_pairs"
    _common.write_jsonl(out_dir / "pairs.jsonl", pairs_rows)
    _common.write_csv(out_dir / "pairs.csv", pairs_rows)
    print(f"Wrote {len(pairs_rows)} pair records to {out_dir}")

    if prompts_path.exists():
        prompts_rows = _normalize_id(_common.read_csv_rows(prompts_path))
        for row in prompts_rows:
            row["source"] = "crows-pairs"
            row["source_file"] = str(prompts_path.relative_to(_common.RAW_ROOT))
        _common.write_jsonl(out_dir / "prompts.jsonl", prompts_rows)
        _common.write_csv(out_dir / "prompts.csv", prompts_rows)
        print(f"Wrote {len(prompts_rows)} prompt records to {out_dir}")


if __name__ == "__main__":
    main()
