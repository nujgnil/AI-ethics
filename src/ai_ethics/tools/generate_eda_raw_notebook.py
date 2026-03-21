from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = ROOT / "notebooks" / "eda_raw.ipynb"


def markdown_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _lines(source),
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": _lines(source),
        "execution_count": None,
        "outputs": [],
    }


def _lines(source: str) -> list[str]:
    source = textwrap.dedent(source).strip("\n")
    if not source:
        return []
    lines = source.splitlines(keepends=True)
    if not source.endswith("\n"):
        lines[-1] += "\n"
    return lines


def notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # Raw Data EDA

            This notebook profiles the raw datasets before cleaning or modeling. It is set up to answer four useful questions:

            1. Which datasets are actually large enough to matter?
            2. What kinds of files make up each raw dataset?
            3. How long are the text fields the models will eventually see?
            4. How much label structure is already present in the raw files?

            Set `SELECTED_DATASETS` in the setup cell if you want to focus on a subset such as `['normbank', 'morebench']`.
            """
        ),
        code_cell(
            """
            from __future__ import annotations

            from collections import Counter
            from pathlib import Path
            import csv
            import json
            import statistics
            import sys

            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            from IPython.display import Markdown, display

            def set_csv_field_limit() -> None:
                limit = sys.maxsize
                while True:
                    try:
                        csv.field_size_limit(limit)
                        return
                    except OverflowError:
                        limit //= 10

            set_csv_field_limit()
            sns.set_theme(style="whitegrid", context="talk")
            pd.set_option("display.max_columns", 50)
            pd.set_option("display.max_colwidth", 120)

            def find_project_root() -> tuple[Path, Path]:
                for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
                    for data_name in ("Data", "data"):
                        data_root = candidate / data_name
                        if (data_root / "raw").exists():
                            return candidate, data_root
                raise FileNotFoundError("Could not find a Data/raw or data/raw directory from the current working directory.")

            ROOT, DATA_ROOT = find_project_root()
            RAW_DIR = DATA_ROOT / "raw"
            OUT_DIR = DATA_ROOT / "eda" / "raw"

            SELECTED_DATASETS = None

            DATASET_PATHS = {
                "hendrycks_ethics": RAW_DIR / "hendryicks-ethics",
                "normbank": RAW_DIR / "normbank",
                "mfd2": RAW_DIR / "mfd2",
                "mfrc": RAW_DIR / "mfrc",
                "moralbench": RAW_DIR / "moralbench",
                "morebench": RAW_DIR / "morebench",
            }

            DATASET_LABELS = {
                "hendrycks_ethics": "Hendrycks Ethics",
                "normbank": "NormBank",
                "mfd2": "MFD2",
                "mfrc": "MFRC",
                "moralbench": "MoralBench",
                "morebench": "MoReBench",
            }

            TEXT_HINTS = {
                "text",
                "prompt",
                "question",
                "sentence",
                "story",
                "statement",
                "input",
                "scenario",
                "norm",
                "rule",
                "vignette",
                "post",
                "comment",
                "body",
            }

            LABEL_HINTS = {
                "label",
                "answer",
                "gold",
                "target",
                "class",
                "category",
                "judgment",
                "rating",
                "score",
                "acceptability",
            }

            print("Project root:", ROOT)
            print("Data root:", DATA_ROOT)
            print("Raw data dir:", RAW_DIR)
            print("EDA output dir:", OUT_DIR)
            print("Selected datasets:", SELECTED_DATASETS or "all")
            """
        ),
        code_cell(
            """
            def _pick_column(columns: list[str], hints: set[str]) -> str:
                lower = {c.lower(): c for c in columns}
                for name in columns:
                    key = name.lower()
                    if key in hints:
                        return name
                for key, name in lower.items():
                    for hint in hints:
                        if hint in key:
                            return name
                return ""


            def _text_len_stats(lengths: list[int]) -> dict[str, int]:
                if not lengths:
                    return {"min": 0, "median": 0, "p95": 0}
                lengths_sorted = sorted(lengths)
                p95_index = int(len(lengths_sorted) * 0.95) - 1
                p95_index = max(0, min(p95_index, len(lengths_sorted) - 1))
                return {
                    "min": lengths_sorted[0],
                    "median": int(statistics.median(lengths_sorted)),
                    "p95": lengths_sorted[p95_index],
                }
            """
        ),
        code_cell(
            """
            def _count_csv(path: Path, delimiter: str = ",") -> tuple[int, list[str], dict]:
                rows = 0
                columns: list[str] = []
                lengths: list[int] = []
                empty_text = 0
                empty_label = 0
                label_counts: dict[str, int] = {}
                with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                    reader = csv.DictReader(f, delimiter=delimiter)
                    columns = reader.fieldnames or []
                    text_col = _pick_column(columns, TEXT_HINTS)
                    label_col = _pick_column(columns, LABEL_HINTS)
                    for row in reader:
                        rows += 1
                        if text_col:
                            value = (row.get(text_col) or "").strip()
                            if not value:
                                empty_text += 1
                            else:
                                lengths.append(len(value))
                        if label_col:
                            label = (row.get(label_col) or "").strip()
                            if not label:
                                empty_label += 1
                            else:
                                label_counts[label] = label_counts.get(label, 0) + 1
                stats = {
                    "text_col": text_col,
                    "label_col": label_col,
                    "empty_text": empty_text,
                    "empty_label": empty_label,
                    "text_len": _text_len_stats(lengths),
                    "label_counts": label_counts,
                }
                return rows, columns, stats


            def _count_jsonl(path: Path) -> tuple[int, list[str], dict]:
                rows = 0
                keys: list[str] = []
                lengths: list[int] = []
                empty_text = 0
                empty_label = 0
                label_counts: dict[str, int] = {}
                text_col = ""
                label_col = ""
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        rows += 1
                        if rows == 1 and isinstance(obj, dict):
                            keys = list(obj.keys())
                            text_col = _pick_column(keys, TEXT_HINTS)
                            label_col = _pick_column(keys, LABEL_HINTS)
                        if isinstance(obj, dict):
                            if text_col:
                                value = str(obj.get(text_col, "")).strip()
                                if not value:
                                    empty_text += 1
                                else:
                                    lengths.append(len(value))
                            if label_col:
                                label = str(obj.get(label_col, "")).strip()
                                if not label:
                                    empty_label += 1
                                else:
                                    label_counts[label] = label_counts.get(label, 0) + 1
                stats = {
                    "text_col": text_col,
                    "label_col": label_col,
                    "empty_text": empty_text,
                    "empty_label": empty_label,
                    "text_len": _text_len_stats(lengths),
                    "label_counts": label_counts,
                }
                return rows, keys, stats


            def _count_txt(path: Path) -> int:
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    return sum(1 for _ in f)
            """
        ),
        code_cell(
            """
            def _count_parquet(path: Path) -> tuple[int, list[str], dict]:
                try:
                    import pyarrow.parquet as pq
                except Exception:
                    pq = None

                if pq is None:
                    df = pd.read_parquet(path)
                    columns = list(df.columns)
                    text_col = _pick_column(columns, TEXT_HINTS)
                    label_col = _pick_column(columns, LABEL_HINTS)
                    lengths: list[int] = []
                    empty_text = 0
                    empty_label = 0
                    label_counts: dict[str, int] = {}

                    if text_col:
                        for value in df[text_col].tolist():
                            value = "" if value is None else str(value).strip()
                            if not value:
                                empty_text += 1
                            else:
                                lengths.append(len(value))
                    if label_col:
                        for value in df[label_col].tolist():
                            value = "" if value is None else str(value).strip()
                            if not value:
                                empty_label += 1
                            else:
                                label_counts[value] = label_counts.get(value, 0) + 1

                    stats = {
                        "text_col": text_col,
                        "label_col": label_col,
                        "empty_text": empty_text,
                        "empty_label": empty_label,
                        "text_len": _text_len_stats(lengths),
                        "label_counts": label_counts,
                    }
                    return len(df), columns, stats

                pf = pq.ParquetFile(path)
                columns = list(pf.schema.names)
                text_col = _pick_column(columns, TEXT_HINTS)
                label_col = _pick_column(columns, LABEL_HINTS)
                lengths: list[int] = []
                empty_text = 0
                empty_label = 0
                label_counts: dict[str, int] = {}
                cols_to_read = [c for c in [text_col, label_col] if c]

                if cols_to_read:
                    table = pq.read_table(path, columns=cols_to_read)
                    data = table.to_pydict()
                    text_values = data.get(text_col, []) if text_col else []
                    label_values = data.get(label_col, []) if label_col else []
                    if text_col:
                        for value in text_values:
                            value = "" if value is None else str(value).strip()
                            if not value:
                                empty_text += 1
                            else:
                                lengths.append(len(value))
                    if label_col:
                        for value in label_values:
                            value = "" if value is None else str(value).strip()
                            if not value:
                                empty_label += 1
                            else:
                                label_counts[value] = label_counts.get(value, 0) + 1

                stats = {
                    "text_col": text_col,
                    "label_col": label_col,
                    "empty_text": empty_text,
                    "empty_label": empty_label,
                    "text_len": _text_len_stats(lengths),
                    "label_counts": label_counts,
                }
                return pf.metadata.num_rows, columns, stats


            def _file_summary(dataset: str, path: Path) -> dict:
                info = {
                    "dataset": dataset,
                    "file": str(path.relative_to(RAW_DIR)),
                    "bytes": path.stat().st_size,
                    "rows": 0,
                    "columns": [],
                    "type": path.suffix.lower().lstrip("."),
                    "text_col": "",
                    "label_col": "",
                    "text_len_min": 0,
                    "text_len_median": 0,
                    "text_len_p95": 0,
                    "empty_text": 0,
                    "empty_label": 0,
                    "label_counts": {},
                }

                def _merge_stats(stats: dict) -> None:
                    info.update(
                        {
                            "text_col": stats.get("text_col", ""),
                            "label_col": stats.get("label_col", ""),
                            "text_len_min": stats.get("text_len", {}).get("min", 0),
                            "text_len_median": stats.get("text_len", {}).get("median", 0),
                            "text_len_p95": stats.get("text_len", {}).get("p95", 0),
                            "empty_text": stats.get("empty_text", 0),
                            "empty_label": stats.get("empty_label", 0),
                            "label_counts": stats.get("label_counts", {}),
                        }
                    )

                suffix = path.suffix.lower()
                if suffix == ".csv":
                    rows, cols, stats = _count_csv(path, delimiter=",")
                    info["rows"] = rows
                    info["columns"] = cols
                    _merge_stats(stats)
                elif suffix == ".tsv":
                    rows, cols, stats = _count_csv(path, delimiter="\\t")
                    info["rows"] = rows
                    info["columns"] = cols
                    _merge_stats(stats)
                elif suffix == ".jsonl":
                    rows, cols, stats = _count_jsonl(path)
                    info["rows"] = rows
                    info["columns"] = cols
                    _merge_stats(stats)
                elif suffix == ".parquet":
                    rows, cols, stats = _count_parquet(path)
                    info["rows"] = rows
                    info["columns"] = cols
                    _merge_stats(stats)
                elif suffix in {".txt", ".dic"}:
                    info["rows"] = _count_txt(path)
                return info
            """
        ),
        code_cell(
            """
            OUT_DIR.mkdir(parents=True, exist_ok=True)

            file_rows = []
            dataset_rows = []

            for name, path in DATASET_PATHS.items():
                if not path.exists():
                    dataset_rows.append(
                        {
                            "dataset": name,
                            "status": "missing",
                            "files": 0,
                            "data_files": 0,
                            "rows": 0,
                            "bytes": 0,
                            "empty_text": 0,
                            "empty_label": 0,
                            "notes": "path not found",
                        }
                    )
                    continue

                files = [p for p in path.rglob("*") if p.is_file() and "ipynb_checkpoints" not in p.as_posix()]
                total_rows = 0
                total_bytes = 0
                total_empty_text = 0
                total_empty_label = 0
                total_data_files = 0

                for p in files:
                    info = _file_summary(name, p)
                    total_rows += int(info.get("rows") or 0)
                    total_bytes += int(info.get("bytes") or 0)
                    total_empty_text += int(info.get("empty_text") or 0)
                    total_empty_label += int(info.get("empty_label") or 0)
                    total_data_files += int((info.get("rows") or 0) > 0)
                    file_rows.append(info)

                dataset_rows.append(
                    {
                        "dataset": name,
                        "status": "ok",
                        "files": len(files),
                        "data_files": total_data_files,
                        "rows": total_rows,
                        "bytes": total_bytes,
                        "empty_text": total_empty_text,
                        "empty_label": total_empty_label,
                        "notes": "",
                    }
                )

            with (OUT_DIR / "raw_file_summary.json").open("w", encoding="utf-8") as f:
                json.dump(file_rows, f, indent=2)

            with (OUT_DIR / "raw_dataset_summary.json").open("w", encoding="utf-8") as f:
                json.dump(dataset_rows, f, indent=2)

            with (OUT_DIR / "raw_file_summary.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "dataset",
                        "file",
                        "bytes",
                        "rows",
                        "columns",
                        "type",
                        "text_col",
                        "label_col",
                        "text_len_min",
                        "text_len_median",
                        "text_len_p95",
                        "empty_text",
                        "empty_label",
                        "label_counts",
                    ],
                )
                writer.writeheader()
                for row in file_rows:
                    out = dict(row)
                    if isinstance(out.get("columns"), list):
                        out["columns"] = ",".join(out["columns"])
                    if isinstance(out.get("label_counts"), dict):
                        out["label_counts"] = json.dumps(out["label_counts"], ensure_ascii=True)
                    writer.writerow(out)

            with (OUT_DIR / "raw_dataset_summary.csv").open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "dataset",
                        "status",
                        "files",
                        "data_files",
                        "rows",
                        "bytes",
                        "empty_text",
                        "empty_label",
                        "notes",
                    ],
                )
                writer.writeheader()
                for row in dataset_rows:
                    writer.writerow(row)

            summary_preview = pd.DataFrame(dataset_rows).sort_values(["status", "rows"], ascending=[True, False])
            display(summary_preview)
            print(f"Wrote raw EDA summaries to {OUT_DIR}")
            """
        ),
        markdown_cell(
            """
            ## Load And Shape The Summary Tables

            This section converts the raw summaries into analysis-ready tables for plotting. The derived metrics are intended to make comparisons meaningful, especially across datasets with very different file structures.
            """
        ),
        code_cell(
            """
            raw_file_csv = OUT_DIR / "raw_file_summary.csv"
            raw_dataset_csv = OUT_DIR / "raw_dataset_summary.csv"

            if not raw_file_csv.exists() or not raw_dataset_csv.exists():
                raise FileNotFoundError("Run the summary-building cell above first.")

            files_df = pd.read_csv(raw_file_csv)
            datasets_df = pd.read_csv(raw_dataset_csv)

            numeric_file_cols = [
                "bytes",
                "rows",
                "text_len_min",
                "text_len_median",
                "text_len_p95",
                "empty_text",
                "empty_label",
            ]
            numeric_dataset_cols = ["files", "data_files", "rows", "bytes", "empty_text", "empty_label"]

            for col in numeric_file_cols:
                files_df[col] = pd.to_numeric(files_df[col], errors="coerce").fillna(0)
            for col in numeric_dataset_cols:
                datasets_df[col] = pd.to_numeric(datasets_df[col], errors="coerce").fillna(0)

            def parse_counts(value: object) -> dict[str, int]:
                if pd.isna(value) or value in ("", "{}"):
                    return {}
                if isinstance(value, dict):
                    return value
                try:
                    parsed = json.loads(value)
                except Exception:
                    return {}
                return {str(k): int(v) for k, v in parsed.items()}

            files_df["columns_list"] = files_df["columns"].fillna("").apply(
                lambda value: [part.strip() for part in str(value).split(",") if part.strip()]
            )
            files_df["column_count"] = files_df["columns_list"].apply(len)
            files_df["label_counts_dict"] = files_df["label_counts"].apply(parse_counts)
            files_df["unique_labels"] = files_df["label_counts_dict"].apply(len)
            files_df["label_total"] = files_df["label_counts_dict"].apply(lambda counts: sum(counts.values()))
            files_df["dominant_label_share"] = files_df["label_counts_dict"].apply(
                lambda counts: max(counts.values()) / sum(counts.values()) if counts else 0.0
            )
            files_df["has_text"] = files_df["text_col"].fillna("").astype(str).str.strip().ne("")
            files_df["has_label"] = files_df["label_col"].fillna("").astype(str).str.strip().ne("")
            files_df["is_data_file"] = files_df["rows"] > 0
            files_df["dataset_label"] = files_df["dataset"].map(DATASET_LABELS).fillna(files_df["dataset"])
            datasets_df["dataset_label"] = datasets_df["dataset"].map(DATASET_LABELS).fillna(datasets_df["dataset"])
            datasets_df["avg_rows_per_data_file"] = datasets_df.apply(
                lambda row: row["rows"] / row["data_files"] if row["data_files"] else 0,
                axis=1,
            )

            selected = SELECTED_DATASETS or list(DATASET_PATHS)
            selected = [name for name in selected if name in DATASET_PATHS]
            if not selected:
                raise ValueError("SELECTED_DATASETS did not match any known dataset keys.")

            viz_files_df = files_df[files_df["dataset"].isin(selected)].copy()
            viz_datasets_df = datasets_df[datasets_df["dataset"].isin(selected)].copy()
            core_files_df = viz_files_df[viz_files_df["is_data_file"]].copy()
            text_files_df = core_files_df[core_files_df["has_text"]].copy()
            label_files_df = core_files_df[core_files_df["has_label"]].copy()

            dataset_order = viz_datasets_df.sort_values("rows", ascending=False)["dataset_label"].tolist()

            overview_df = (
                viz_datasets_df[
                    [
                        "dataset_label",
                        "files",
                        "data_files",
                        "rows",
                        "bytes",
                        "avg_rows_per_data_file",
                        "empty_text",
                        "empty_label",
                    ]
                ]
                .sort_values("rows", ascending=False)
                .reset_index(drop=True)
            )

            type_mix_df = (
                core_files_df.groupby(["dataset_label", "type"])
                .size()
                .reset_index(name="file_count")
                .sort_values(["dataset_label", "file_count"], ascending=[True, False])
            )

            display(Markdown("### Dataset Overview"))
            display(overview_df)

            display(Markdown("### Raw File Type Mix"))
            display(type_mix_df)
            """
        ),
        markdown_cell(
            """
            ## Visualizations

            The charts below are aimed at interpretation, not decoration. Each one compares a different aspect of raw dataset readiness.
            """
        ),
        code_cell(
            """
            plot_df = overview_df.sort_values("rows", ascending=True)

            plt.figure(figsize=(10, 6))
            sns.barplot(data=plot_df, x="rows", y="dataset_label", color="#4C78A8")
            plt.xscale("log")
            plt.title("Raw Row Count By Dataset")
            plt.xlabel("Rows (log scale)")
            plt.ylabel("")
            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            plt.figure(figsize=(10, 6))
            sns.barplot(data=plot_df, x="data_files", y="dataset_label", color="#F58518")
            plt.title("Data-Bearing Files By Dataset")
            plt.xlabel("Files with rows")
            plt.ylabel("")
            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            plt.figure(figsize=(10, 6))
            sns.barplot(data=plot_df, x="avg_rows_per_data_file", y="dataset_label", color="#54A24B")
            plt.xscale("log")
            plt.title("Average Rows Per Data File")
            plt.xlabel("Rows per file (log scale)")
            plt.ylabel("")
            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            dataset_type_counts = (
                core_files_df.groupby(["dataset_label", "type"])
                .size()
                .unstack(fill_value=0)
                .reindex(index=dataset_order)
            )

            plt.figure(figsize=(12, 7))
            dataset_type_counts.plot(kind="barh", stacked=True, colormap="tab20")
            plt.title("File Type Composition")
            plt.xlabel("Number of files")
            plt.ylabel("")
            plt.legend(title="Type", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            dataset_type_rows = (
                core_files_df.groupby(["dataset_label", "type"])["rows"]
                .sum()
                .unstack(fill_value=0)
                .reindex(index=dataset_order)
            )

            plt.figure(figsize=(12, 7))
            dataset_type_rows.plot(kind="barh", stacked=True, colormap="tab20c")
            plt.title("Rows Contributed By File Type")
            plt.xlabel("Rows")
            plt.xscale("log")
            plt.ylabel("")
            plt.legend(title="Type", bbox_to_anchor=(1.02, 1), loc="upper left")
            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            if text_files_df.empty:
                print("No text-bearing raw files were detected for the current dataset selection.")
            else:
                plt.figure(figsize=(11, 7))
                sns.boxplot(
                    data=text_files_df,
                    y="dataset_label",
                    x="text_len_median",
                    orient="h",
                    color="#72B7B2",
                )
                sns.stripplot(
                    data=text_files_df,
                    y="dataset_label",
                    x="text_len_p95",
                    orient="h",
                    size=7,
                    color="#E45756",
                    alpha=0.75,
                )
                plt.title("Typical And Long-Tail Text Length")
                plt.xlabel("Characters")
                plt.ylabel("")
                plt.tight_layout()
                plt.show()
            """
        ),
        code_cell(
            """
            if text_files_df.empty:
                print("No text-bearing raw files were detected for the current dataset selection.")
            else:
                plt.figure(figsize=(11, 7))
                sns.scatterplot(
                    data=text_files_df,
                    x="rows",
                    y="text_len_p95",
                    hue="dataset_label",
                    style="type",
                    size="bytes",
                    sizes=(80, 500),
                    alpha=0.8,
                )
                plt.xscale("log")
                plt.title("Text Length vs File Scale")
                plt.xlabel("Rows (log scale)")
                plt.ylabel("95th percentile text length")
                plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
                plt.tight_layout()
                plt.show()
            """
        ),
        code_cell(
            """
            coverage_rows = []
            for dataset, group in core_files_df.groupby("dataset", sort=False):
                total_rows = group["rows"].sum()
                coverage_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS.get(dataset, dataset),
                        "text_file_share_pct": group["has_text"].mean() * 100,
                        "label_file_share_pct": group["has_label"].mean() * 100,
                        "empty_text_rate_pct": (group["empty_text"].sum() / total_rows * 100) if total_rows else 0,
                        "empty_label_rate_pct": (group["empty_label"].sum() / total_rows * 100) if total_rows else 0,
                    }
                )

            coverage_df = pd.DataFrame(coverage_rows).set_index("dataset_label").reindex(dataset_order)

            plt.figure(figsize=(11, 7))
            sns.heatmap(
                coverage_df[
                    [
                        "text_file_share_pct",
                        "label_file_share_pct",
                        "empty_text_rate_pct",
                        "empty_label_rate_pct",
                    ]
                ],
                annot=True,
                fmt=".1f",
                cmap="YlGnBu",
                cbar_kws={"label": "Percent"},
            )
            plt.title("Metadata Coverage And Missingness")
            plt.xlabel("")
            plt.ylabel("")
            plt.tight_layout()
            plt.show()
            """
        ),
        code_cell(
            """
            label_profile_rows = []
            for dataset, group in label_files_df.groupby("dataset", sort=False):
                counter = Counter()
                for counts in group["label_counts_dict"]:
                    counter.update(counts)
                total = sum(counter.values())
                label_profile_rows.append(
                    {
                        "dataset": dataset,
                        "dataset_label": DATASET_LABELS.get(dataset, dataset),
                        "unique_labels": len(counter),
                        "dominant_label_share_pct": (max(counter.values()) / total * 100) if total else 0,
                    }
                )

            label_profile_plot = pd.DataFrame(label_profile_rows)
            if label_profile_plot.empty:
                print("No label-bearing files for the current selection.")
            else:
                label_profile_plot = label_profile_plot.set_index("dataset_label").reindex(
                    [label for label in dataset_order if label in label_profile_plot["dataset_label"].tolist()]
                ).reset_index()

                plt.figure(figsize=(10, 6))
                sns.barplot(
                    data=label_profile_plot,
                    y="dataset_label",
                    x="unique_labels",
                    color="#B279A2",
                )
                plt.title("Observed Label Diversity")
                plt.xlabel("Unique raw labels")
                plt.ylabel("")
                plt.tight_layout()
                plt.show()
            """
        ),
        code_cell(
            """
            if label_profile_plot.empty:
                print("No label-bearing files for the current selection.")
            else:
                plt.figure(figsize=(10, 6))
                sns.scatterplot(
                    data=label_profile_plot,
                    y="dataset_label",
                    x="dominant_label_share_pct",
                    color="#FF9DA6",
                    s=140,
                )
                plt.xlim(0, 100)
                plt.title("Dominant Label Share")
                plt.xlabel("Dominant label share (%)")
                plt.ylabel("")
                plt.tight_layout()
                plt.show()
            """
        ),
    ]

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "venv (3.10.11)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
