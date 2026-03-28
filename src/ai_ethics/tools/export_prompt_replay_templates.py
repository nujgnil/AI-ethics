from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

from ..data.loader import get_processed_csv_path


DEFAULT_DATASETS = ["moralbench", "morebench_public", "interpretive"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _rows_for_dataset(dataset: str, limit: int) -> list[dict[str, str]]:
    path = get_processed_csv_path(dataset)
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "item_id": (row.get("item_id") or "").strip(),
                    "text": (row.get("text") or "").strip(),
                    "response_format": (row.get("response_format") or "").strip(),
                    "expected_behavior": (row.get("expected_behavior") or "").strip(),
                    "metric_id": (row.get("metric_id") or "").strip(),
                    "scenario_group": (row.get("scenario_group") or "").strip(),
                }
            )
            if len(rows) >= limit:
                break
    return rows


def _write_prompt_csv(path: Path, rows: Iterable[dict[str, str]]) -> None:
    fieldnames = [
        "item_id",
        "text",
        "response_format",
        "expected_behavior",
        "metric_id",
        "scenario_group",
        "manual_response",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({**row, "manual_response": ""})


def _write_replay_template(path: Path, rows: Iterable[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        for row in rows:
            record = {
                "item_id": row["item_id"],
                "response_text": "",
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def _write_readme(path: Path, datasets: list[str], limit: int) -> None:
    lines = [
        "Manual replay workflow",
        "",
        f"These files were generated for the first {limit} items of: {', '.join(datasets)}.",
        "",
        "Files per dataset:",
        "- <dataset>_prompts.csv: copy prompts out and paste manual responses into the manual_response column.",
        "- <dataset>_replay_template.jsonl: final replay file for provider=replay; fill response_text for each item_id.",
        "",
        "Run replay after filling the JSONL files:",
        "",
    ]
    for dataset in datasets:
        run_id = f"{dataset}_replay_manual_{limit}"
        jsonl_name = f"{dataset}_replay_template.jsonl"
        lines.extend(
            [
                f"python src/prompt_eval.py run --dataset {dataset} --provider replay --model replay_manual --replay-file results/prompt_eval_manual/{jsonl_name} --run-id {run_id} --limit {limit}",
                f"python src/prompt_eval.py score --run-id {run_id}",
                f"python src/prompt_eval.py aggregate --run-id {run_id}",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def export_templates(output_dir: Path, datasets: list[str], limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for dataset in datasets:
        rows = _rows_for_dataset(dataset, limit)
        if not rows:
            raise ValueError(f"No rows found for dataset={dataset}")
        _write_prompt_csv(output_dir / f"{dataset}_prompts.csv", rows)
        _write_replay_template(output_dir / f"{dataset}_replay_template.jsonl", rows)
    _write_readme(output_dir / "README_manual_replay.txt", datasets, limit)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export prompt subsets and replay templates for manual prompt-eval runs.")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS, choices=["moralbench", "morebench_public", "morebench_theory", "interpretive"])
    parser.add_argument("--output-dir", default=str(_project_root() / "results" / "prompt_eval_manual"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_templates(Path(args.output_dir), args.datasets, args.limit)
    print(json.dumps({"output_dir": str(Path(args.output_dir)), "datasets": args.datasets, "limit": args.limit}, ensure_ascii=False))


if __name__ == "__main__":
    main()
