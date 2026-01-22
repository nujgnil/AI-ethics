from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from preprocess_utils import write_csv, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "moralbench"
OUT_DIR = ROOT / "data" / "processed" / "moralbench"


def _is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _parse_metadata(path: Path) -> Dict[str, str]:
    parts = path.parts
    metadata: Dict[str, str] = {}
    if "questions" in parts:
        idx = parts.index("questions")
        if idx + 1 < len(parts):
            metadata["collection"] = parts[idx + 1]
        if idx + 2 < len(parts):
            foundation = parts[idx + 2].split("_")[0]
            metadata["foundation"] = foundation
    return metadata


def main() -> None:
    if not RAW_DIR.exists():
        print(f"Missing dataset root: {RAW_DIR}")
        return

    rows: List[Dict[str, str]] = []
    for path in sorted(RAW_DIR.rglob("*.txt")):
        if _is_hidden(path):
            continue
        if "ipynb_checkpoints" in path.as_posix():
            continue
        rel = path.relative_to(RAW_DIR)
        metadata = _parse_metadata(rel)
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                text = line.strip()
                if not text:
                    continue
                rows.append(
                    {
                        "text": text,
                        "label": "",
                        "dataset": "moralbench",
                        "task": metadata.get("collection", "prompt_set"),
                        "split": "",
                        "source_file": str(rel),
                        "metadata": metadata,
                    }
                )

    write_jsonl(OUT_DIR / "moralbench.jsonl", rows)
    write_csv(OUT_DIR / "moralbench.csv", rows)

    print(f"Wrote {len(rows)} records to {OUT_DIR}")


if __name__ == "__main__":
    main()
