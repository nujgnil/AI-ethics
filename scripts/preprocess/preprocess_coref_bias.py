from pathlib import Path
import re

import _common


def main() -> None:
    data_dir = _common.RAW_ROOT / "corefBias-master" / "WinoBias" / "wino" / "data"
    if not data_dir.exists():
        print(f"Missing data dir: {data_dir}")
        return

    rows = []
    pattern = re.compile(r"(pro|anti)_stereotyped_type(\d+)")

    for path in sorted(data_dir.glob("*stereotyped_type*.txt.*")):
        name = path.name
        split = name.split(".")[-1]
        base = name.split(".")[0]
        match = pattern.match(base)
        if not match:
            continue
        stereotype, type_id = match.group(1), match.group(2)

        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                line_match = re.match(r"^(\d+)\s+(.*)$", line)
                if line_match:
                    item_id, text = line_match.group(1), line_match.group(2)
                else:
                    item_id, text = None, line

                rows.append(
                    {
                        "id": item_id,
                        "text": text,
                        "stereotype": stereotype,
                        "type": type_id,
                        "split": split,
                        "source": "corefBias",
                        "source_file": str(path.relative_to(_common.RAW_ROOT)),
                    }
                )

    out_dir = _common.PROCESSED_ROOT / "corefBias"
    _common.write_jsonl(out_dir / "corefBias.jsonl", rows)
    _common.write_csv(out_dir / "corefBias.csv", rows)
    print(f"Wrote {len(rows)} records to {out_dir}")


if __name__ == "__main__":
    main()
