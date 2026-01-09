from pathlib import Path
import re

import _common


def _parse_prompt_file(path: Path):
    records = []
    key_pattern = re.compile(r'"([^"]+)"\s*:\s*"""')
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        match = key_pattern.search(line)
        if not match:
            idx += 1
            continue
        key = match.group(1)
        parts = []
        after = line.split('"""', 1)[1]
        if after.strip():
            parts.append(after)
        idx += 1
        while idx < len(lines):
            if '"""' in lines[idx]:
                before = lines[idx].split('"""', 1)[0]
                parts.append(before)
                break
            parts.append(lines[idx])
            idx += 1
        prompt_text = "
".join(parts).strip("
")
        records.append(
            {
                "prompt_key": key,
                "prompt_text": prompt_text,
                "source": "unimoral",
                "source_file": str(path.relative_to(_common.RAW_ROOT)),
            }
        )
        idx += 1
    return records


def main() -> None:
    base = _common.RAW_ROOT / "UniMoral-main"
    if not base.exists():
        print(f"Missing dataset root: {base}")
        return

    prompt_files = sorted(base.glob("PROMPTS*.txt"))
    if not prompt_files:
        print("No PROMPTS*.txt files found under UniMoral-main")
        return

    rows = []
    for path in prompt_files:
        rows.extend(_parse_prompt_file(path))

    for idx, row in enumerate(rows):
        row["id"] = str(idx)

    out_dir = _common.PROCESSED_ROOT / "unimoral"
    _common.write_jsonl(out_dir / "prompts.jsonl", rows)
    _common.write_csv(out_dir / "prompts.csv", rows)
    print(f"Wrote {len(rows)} prompt records to {out_dir}")


if __name__ == "__main__":
    main()
