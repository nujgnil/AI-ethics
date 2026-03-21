from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = ROOT / "notebooks" / "cleaning"


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


def setup_cell(raw_subpath: str, out_subpath: str, needs_pyarrow: bool = False) -> str:
    pyarrow_import = """
try:
    import pyarrow.parquet as pq
except Exception:
    pq = None
""" if needs_pyarrow else ""

    return f"""
from __future__ import annotations

from pathlib import Path
from ast import literal_eval
import csv
import hashlib
import json
import re
import shutil

import matplotlib.pyplot as plt
import seaborn as sns

try:
    import pandas as pd
except Exception as exc:
    raise RuntimeError("pandas is required to use this cleaning notebook.") from exc

from IPython.display import display

{pyarrow_import}

sns.set_theme(style="whitegrid")
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_colwidth", 120)

def find_project_root() -> Path:
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        if (candidate / "Data").exists() or (candidate / "data").exists():
            return candidate
    return Path.cwd().resolve()


ROOT = find_project_root()
DATA_ROOT = ROOT / "Data" if (ROOT / "Data").exists() else ROOT / "data"
RAW_DIR = DATA_ROOT / "{raw_subpath}"
OUT_DIR = DATA_ROOT / "processed" / "{out_subpath}"
SAVE_OUTPUTS = False

print("Project root:", ROOT)
print("Raw dir:", RAW_DIR)
print("Output dir:", OUT_DIR)
print("SAVE_OUTPUTS:", SAVE_OUTPUTS)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_for_csv(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_jsonl(path: Path, rows) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")


def write_csv(path: Path, rows, fieldnames=None) -> None:
    ensure_dir(path.parent)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({{k: normalize_for_csv(row.get(k)) for k in fieldnames}})


def plot_count(series, title: str, top_n: int = 15):
    counts = series.fillna("<missing>").astype(str).value_counts().head(top_n)
    if counts.empty:
        print(f"No values available for {{title}}")
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=counts.index, y=counts.values, ax=ax, color="#4C72B0")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.show()


def plot_text_length(series, title: str):
    lengths = series.fillna("").astype(str).str.len()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(lengths, bins=30, ax=ax, color="#55A868")
    ax.set_title(title)
    ax.set_xlabel("characters")
    plt.tight_layout()
    plt.show()
"""


def ethics_notebook() -> dict:
    text_fields = '["input", "scenario", "prompt", "question", "text", "sentence", "story", "statement"]'
    label_fields = '["label", "answer", "gold", "gold_label", "target", "output"]'
    cells = [
        markdown_cell(
            """
            # ETHICS Cleaning

            This notebook inspects the raw Hendrycks ETHICS files, shows cleaning diagnostics,
            visualizes class/split coverage, and builds the standardized `ethics.csv` / `ethics.jsonl` outputs.
            """
        ),
        code_cell(setup_cell("raw/hendryicks-ethics", "ethics")),
        code_cell(
            f"""
            TEXT_FIELDS = {text_fields}
            LABEL_FIELDS = {label_fields}


            def split_from_name(name: str) -> tuple[str, str]:
                for suffix in ["train", "test_hard", "test", "ambig"]:
                    needle = f"_{{suffix}}"
                    if name.endswith(needle):
                        return name[: -len(needle)], suffix
                return name, ""


            raw_frames = []
            for path in sorted(RAW_DIR.rglob("*.csv")):
                rel = path.relative_to(RAW_DIR)
                category = rel.parts[0]
                task, split = split_from_name(path.stem)
                df = pd.read_csv(path)
                df["source_file"] = str(rel)
                df["category"] = category
                df["task"] = task
                df["split"] = split
                raw_frames.append(df)

            raw_df = pd.concat(raw_frames, ignore_index=True, sort=False)
            print("Raw shape:", raw_df.shape)
            display(raw_df.head())
            display(pd.DataFrame({{"column": raw_df.columns, "missing": raw_df.isna().sum().values}}).sort_values("missing", ascending=False).head(15))
            """
        ),
        code_cell(
            f"""
            def pick_first_value(row, candidates):
                for key in candidates:
                    value = row.get(key)
                    if pd.notna(value) and str(value).strip():
                        return key, str(value).strip()
                return "", ""


            analysis_rows = []
            for row in raw_df.to_dict(orient="records"):
                text_key, text_value = pick_first_value(row, TEXT_FIELDS)
                label_key, label_value = pick_first_value(row, LABEL_FIELDS)
                analysis_rows.append({{
                    "category": row["category"],
                    "task": row["task"],
                    "split": row["split"],
                    "source_file": row["source_file"],
                    "text_field": text_key,
                    "label_field": label_key,
                    "text": text_value,
                    "label": label_value,
                }})

            analysis_df = pd.DataFrame(analysis_rows)
            analysis_df["text_len"] = analysis_df["text"].str.len()
            analysis_df["text_hash"] = analysis_df["text"].map(lambda x: hashlib.sha1(x.encode("utf-8", errors="ignore")).hexdigest() if x else "")

            display(analysis_df.head())
            display(analysis_df[["text", "label"]].isna().sum())
            display(analysis_df[["text", "label"]].eq("").sum().rename("blank_count").to_frame())

            plot_count(analysis_df["category"], "ETHICS rows by category")
            plot_count(analysis_df["split"], "ETHICS rows by split")
            plot_count(analysis_df.loc[analysis_df["label"] != "", "label"], "ETHICS label distribution")
            plot_text_length(analysis_df["text"], "ETHICS text length distribution")

            dup_counts = analysis_df.loc[analysis_df["text_hash"] != "", "text_hash"].value_counts()
            print("Duplicate text hashes with more than one occurrence:", int((dup_counts > 1).sum()))
            """
        ),
        code_cell(
            """
            cleaned_df = analysis_df.loc[:, ["text", "label", "task", "split", "source_file", "category", "text_field", "label_field", "text_hash"]].copy()
            cleaned_df.insert(2, "dataset", "ethics")
            cleaned_df["metadata"] = cleaned_df.apply(
                lambda row: {
                    "category": row["category"],
                    "text_field": row["text_field"],
                    "label_field": row["label_field"],
                    "text_hash": row["text_hash"],
                },
                axis=1,
            )
            cleaned_df = cleaned_df[["text", "label", "dataset", "task", "split", "source_file", "metadata"]]

            summary_df = (
                analysis_df.assign(has_text=analysis_df["text"] != "", has_label=analysis_df["label"] != "")
                .groupby(["category", "task", "split", "source_file"], dropna=False)
                .agg(
                    rows=("text", "size"),
                    missing_text=("has_text", lambda s: int((~s).sum())),
                    missing_label=("has_label", lambda s: int((~s).sum())),
                    avg_text_len=("text_len", "mean"),
                )
                .reset_index()
            )

            dup_report_df = (
                analysis_df.loc[analysis_df["text_hash"] != "", ["text_hash", "category", "task", "split", "source_file", "text_field"]]
                .groupby("text_hash")
                .agg(
                    occurrences=("text_hash", "size"),
                    locations=("source_file", lambda s: list(s)),
                )
                .reset_index()
            )
            dup_report_df = dup_report_df.loc[dup_report_df["occurrences"] > 1]

            print("Cleaned shape:", cleaned_df.shape)
            display(cleaned_df.head())
            display(summary_df.head())
            display(dup_report_df.head())
            """
        ),
        code_cell(
            """
            if SAVE_OUTPUTS:
                write_jsonl(OUT_DIR / "ethics.jsonl", cleaned_df.to_dict(orient="records"))
                write_csv(OUT_DIR / "ethics.csv", cleaned_df.to_dict(orient="records"))
                write_jsonl(OUT_DIR / "summary.jsonl", summary_df.to_dict(orient="records"))
                write_csv(OUT_DIR / "summary.csv", summary_df.to_dict(orient="records"))
                write_jsonl(OUT_DIR / "dup_report.jsonl", dup_report_df.to_dict(orient="records"))
                write_csv(OUT_DIR / "dup_report.csv", dup_report_df.to_dict(orient="records"))
                label_summary = [{"labels": cleaned_df.loc[cleaned_df["label"] != "", "label"].value_counts().to_dict()}]
                write_jsonl(OUT_DIR / "label_summary.jsonl", label_summary)
                write_csv(OUT_DIR / "label_summary.csv", label_summary)
                print("Wrote cleaned ETHICS outputs to", OUT_DIR)
            else:
                print("Preview only. Set SAVE_OUTPUTS = True and rerun this cell to write cleaned files.")
            """
        ),
    ]
    return notebook("ETHICS Cleaning", cells)


def normbank_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # NormBank Cleaning

            This notebook inspects the NormBank raw file, visualizes label/split balance,
            standardizes the schema, and prepares the processed outputs.
            """
        ),
        code_cell(setup_cell("raw/normbank", "normbank", needs_pyarrow=True)),
        code_cell(
            """
            parquet_files = sorted(RAW_DIR.glob("*.parquet"))
            csv_files = sorted(RAW_DIR.glob("*.csv"))

            if parquet_files:
                if pq is None:
                    raise RuntimeError("pyarrow is required to read parquet files.")
                raw_df = pd.concat([pq.read_table(path).to_pandas() for path in parquet_files], ignore_index=True)
                raw_df["source_file"] = raw_df.get("split", "parquet")
                raw_df["split"] = raw_df.get("split", "")
            elif csv_files:
                raw_df = pd.concat([pd.read_csv(path).assign(source_file=path.name) for path in csv_files], ignore_index=True)
            else:
                raise FileNotFoundError(f"No CSV or parquet files found under {RAW_DIR}")

            print("Raw shape:", raw_df.shape)
            display(raw_df.head())
            display(pd.DataFrame({"column": raw_df.columns, "missing": raw_df.isna().sum().values}).sort_values("missing", ascending=False))
            """
        ),
        code_cell(
            """
            text_series = raw_df.get("norm", raw_df.get("setting-behavior", pd.Series(dtype=str))).fillna("").astype(str).str.strip()
            label_series = raw_df.get("label", pd.Series(dtype=str)).fillna("").astype(str).str.strip()
            split_series = raw_df.get("split", pd.Series(dtype=str)).fillna("").astype(str).str.strip()

            plot_count(label_series[label_series != ""], "NormBank label distribution")
            plot_count(split_series.replace("", "<blank>"), "NormBank split distribution")
            plot_text_length(text_series, "NormBank text length distribution")

            display(pd.DataFrame({
                "blank_text": [int((text_series == "").sum())],
                "blank_label": [int((label_series == "").sum())],
                "blank_split": [int((split_series == "").sum())],
            }))
            """
        ),
        code_cell(
            """
            cleaned_df = pd.DataFrame({
                "text": raw_df.get("norm", raw_df.get("setting-behavior", "")).fillna("").astype(str).str.strip(),
                "label": raw_df.get("label", "").fillna("").astype(str).str.strip(),
                "dataset": "normbank",
                "task": "norm_classification",
                "split": raw_df.get("split", "").fillna("").astype(str).str.strip(),
                "source_file": raw_df["source_file"].astype(str),
            })
            metadata_cols = [c for c in raw_df.columns if c not in {"norm", "setting-behavior", "label", "split", "source_file"}]
            cleaned_df["metadata"] = raw_df[metadata_cols].to_dict(orient="records")

            print("Cleaned shape:", cleaned_df.shape)
            display(cleaned_df.head())
            plot_count(cleaned_df["label"], "Cleaned NormBank label distribution")
            """
        ),
        code_cell(
            """
            if SAVE_OUTPUTS:
                write_jsonl(OUT_DIR / "normbank.jsonl", cleaned_df.to_dict(orient="records"))
                write_csv(OUT_DIR / "normbank.csv", cleaned_df.to_dict(orient="records"))
                label_summary = [{"labels": cleaned_df["label"].value_counts().to_dict()}]
                write_jsonl(OUT_DIR / "label_summary.jsonl", label_summary)
                write_csv(OUT_DIR / "label_summary.csv", label_summary)
                print("Wrote cleaned NormBank outputs to", OUT_DIR)
            else:
                print("Preview only. Set SAVE_OUTPUTS = True and rerun this cell to write cleaned files.")
            """
        ),
    ]
    return notebook("NormBank Cleaning", cells)


def mfrc_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # MFRC Cleaning

            This notebook inspects the raw parquet splits, visualizes text coverage and split counts,
            and prepares the current standardized MFRC format used in this repo.
            """
        ),
        code_cell(setup_cell("raw/mfrc", "mfrc", needs_pyarrow=True)),
        code_cell(
            """
            if pq is None:
                raise RuntimeError("pyarrow is required to read MFRC parquet files.")

            frames = []
            for path in sorted(RAW_DIR.glob("*.parquet")):
                df = pq.read_table(path).to_pandas()
                df["split"] = path.stem
                df["source_file"] = path.name
                frames.append(df)

            raw_df = pd.concat(frames, ignore_index=True)
            print("Raw shape:", raw_df.shape)
            display(raw_df.head())
            display(pd.DataFrame({"column": raw_df.columns, "missing": raw_df.isna().sum().values}).sort_values("missing", ascending=False))
            """
        ),
        code_cell(
            """
            text_series = raw_df.get("text", pd.Series(dtype=str)).fillna("").astype(str).str.strip()

            plot_count(raw_df["split"], "MFRC split distribution")
            plot_text_length(text_series, "MFRC text length distribution")

            metadata_columns = [c for c in raw_df.columns if c not in {"text", "split", "source_file"}]
            unique_counts = pd.Series({col: raw_df[col].nunique(dropna=True) for col in metadata_columns}).sort_values(ascending=False)
            display(unique_counts.to_frame("nunique").head(15))

            for column in metadata_columns[:4]:
                if raw_df[column].dtype == "object":
                    plot_count(raw_df[column].astype(str), f"MFRC top values for {column}")
            """
        ),
        code_cell(
            """
            metadata_columns = [c for c in raw_df.columns if c not in {"text", "split", "source_file"}]
            cleaned_df = pd.DataFrame({
                "text": raw_df["text"].fillna("").astype(str).str.strip(),
                "label": "",
                "dataset": "mfrc",
                "task": "moral_sentiment_multilabel",
                "split": raw_df["split"].astype(str),
                "source_file": raw_df["source_file"].astype(str),
                "metadata": raw_df[metadata_columns].to_dict(orient="records"),
            })

            print("Cleaned shape:", cleaned_df.shape)
            display(cleaned_df.head())
            print("MFRC note: this format is prompt-usable, but not directly supervised in the current pipeline because labels remain nested in metadata.")
            """
        ),
        code_cell(
            """
            if SAVE_OUTPUTS:
                write_jsonl(OUT_DIR / "mfrc.jsonl", cleaned_df.to_dict(orient="records"))
                write_csv(OUT_DIR / "mfrc.csv", cleaned_df.to_dict(orient="records"))
                print("Wrote cleaned MFRC outputs to", OUT_DIR)
            else:
                print("Preview only. Set SAVE_OUTPUTS = True and rerun this cell to write cleaned files.")
            """
        ),
    ]
    return notebook("MFRC Cleaning", cells)


def moralbench_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # MoralBench Cleaning

            This notebook inventories the prompt files under `Data/raw/moralbench`,
            shows collection/foundation coverage, and standardizes them into the prompt-only processed format.
            """
        ),
        code_cell(setup_cell("raw/moralbench", "moralbench")),
        code_cell(
            """
            def is_hidden(path: Path) -> bool:
                return any(part.startswith(".") for part in path.parts)


            records = []
            for path in sorted(RAW_DIR.rglob("*.txt")):
                if is_hidden(path) or "ipynb_checkpoints" in path.as_posix():
                    continue
                rel = path.relative_to(RAW_DIR)
                parts = rel.parts
                metadata = {}
                if "questions" in parts:
                    idx = parts.index("questions")
                    if idx + 1 < len(parts):
                        metadata["collection"] = parts[idx + 1]
                    if idx + 2 < len(parts):
                        metadata["foundation"] = parts[idx + 2].split("_")[0]
                with path.open("r", encoding="utf-8", errors="replace") as f:
                    for line_number, line in enumerate(f, start=1):
                        text = line.strip()
                        if not text:
                            continue
                        records.append({
                            "text": text,
                            "source_file": str(rel),
                            "line_number": line_number,
                            "collection": metadata.get("collection", "prompt_set"),
                            "foundation": metadata.get("foundation", "unknown"),
                        })

            raw_df = pd.DataFrame(records)
            print("Raw prompt rows:", len(raw_df))
            display(raw_df.head())
            """
        ),
        code_cell(
            """
            plot_count(raw_df["collection"], "MoralBench rows by collection")
            plot_count(raw_df["foundation"], "MoralBench rows by foundation")
            plot_text_length(raw_df["text"], "MoralBench text length distribution")

            display(raw_df.groupby(["collection", "foundation"]).size().reset_index(name="rows").sort_values("rows", ascending=False))
            """
        ),
        code_cell(
            """
            cleaned_df = pd.DataFrame({
                "text": raw_df["text"],
                "label": "",
                "dataset": "moralbench",
                "task": raw_df["collection"],
                "split": "",
                "source_file": raw_df["source_file"],
                "metadata": raw_df[["collection", "foundation", "line_number"]].to_dict(orient="records"),
            })

            print("Cleaned shape:", cleaned_df.shape)
            display(cleaned_df.head())
            """
        ),
        code_cell(
            """
            if SAVE_OUTPUTS:
                write_jsonl(OUT_DIR / "moralbench.jsonl", cleaned_df.to_dict(orient="records"))
                write_csv(OUT_DIR / "moralbench.csv", cleaned_df.to_dict(orient="records"))
                print("Wrote cleaned MoralBench outputs to", OUT_DIR)
            else:
                print("Preview only. Set SAVE_OUTPUTS = True and rerun this cell to write cleaned files.")
            """
        ),
    ]
    return notebook("MoralBench Cleaning", cells)


def morebench_notebook(dataset_name: str) -> dict:
    raw_file = f"raw/morebench/{dataset_name}.csv"
    out_subpath = dataset_name
    pretty = dataset_name.replace("_", " ").title()
    cells = [
        markdown_cell(
            f"""
            # {pretty} Cleaning

            This notebook inspects the raw `{dataset_name}.csv` file, visualizes the main categorical fields,
            and standardizes it into the repo's processed prompt/evaluation format.
            """
        ),
        code_cell(setup_cell(raw_file, out_subpath)),
        code_cell(
            """
            raw_df = pd.read_csv(RAW_DIR)
            print("Raw shape:", raw_df.shape)
            display(raw_df.head())
            display(pd.DataFrame({"column": raw_df.columns, "missing": raw_df.isna().sum().values}).sort_values("missing", ascending=False))
            """
        ),
        code_cell(
            """
            for column in ["DILEMMA_SOURCE", "DILEMMA_TYPE", "THEORY", "ROLE_DOMAIN", "CONTEXT"]:
                if column in raw_df.columns:
                    plot_count(raw_df[column].astype(str), f"{column} distribution")

            plot_text_length(raw_df["DILEMMA"].fillna("").astype(str), "DILEMMA length distribution")
            """
        ),
        code_cell(
            f"""
            cleaned_df = pd.DataFrame({{
                "text": raw_df["DILEMMA"].fillna("").astype(str).str.strip(),
                "label": "",
                "dataset": "{dataset_name}",
                "task": raw_df.get("DILEMMA_TYPE", pd.Series(["prompt_eval"] * len(raw_df))).fillna("prompt_eval").astype(str),
                "split": "",
                "source_file": "{Path(raw_file).name}",
            }})
            metadata_cols = [c for c in raw_df.columns if c != "DILEMMA"]
            cleaned_df["metadata"] = raw_df[metadata_cols].to_dict(orient="records")

            print("Cleaned shape:", cleaned_df.shape)
            display(cleaned_df.head())

            # Optional rubric parsing preview for spot checks.
            if "RUBRIC" in raw_df.columns:
                rubric_lengths = raw_df["RUBRIC"].fillna("").astype(str).str.len()
                plot_text_length(rubric_lengths.astype(str), "Rubric string length distribution")
            """
        ),
        code_cell(
            f"""
            if SAVE_OUTPUTS:
                write_jsonl(OUT_DIR / "{dataset_name}.jsonl", cleaned_df.to_dict(orient="records"))
                write_csv(OUT_DIR / "{dataset_name}.csv", cleaned_df.to_dict(orient="records"))
                print("Wrote cleaned {pretty} outputs to", OUT_DIR)
            else:
                print("Preview only. Set SAVE_OUTPUTS = True and rerun this cell to write cleaned files.")
            """
        ),
    ]
    return notebook(f"{pretty} Cleaning", cells)


def mfd2_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # MFD2 Cleaning

            This notebook inspects the Moral Foundations Dictionary 2.0 lexicon, parses the category map,
            visualizes term coverage by category, and copies the standardized files into `Data/processed/mfd2`.
            """
        ),
        code_cell(setup_cell("raw/mfd2", "mfd2")),
        code_cell(
            """
            dic_files = sorted(RAW_DIR.glob("*.dic"))
            docx_files = sorted(RAW_DIR.glob("*.docx"))
            if not dic_files:
                raise FileNotFoundError(f"No .dic file found under {RAW_DIR}")

            dic_path = dic_files[0]
            print("Dictionary file:", dic_path.name)
            print("Summary docx:", docx_files[0].name if docx_files else "<none>")

            preview_lines = dic_path.read_text(encoding="utf-8", errors="replace").splitlines()[:20]
            print("\\n".join(preview_lines))
            """
        ),
        code_cell(
            """
            lines = dic_path.read_text(encoding="utf-8", errors="replace").splitlines()
            categories = {}
            entries = []
            mode = "categories"
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line == "%":
                    mode = "entries" if mode == "categories_done" else "categories_done"
                    continue
                if mode in {"categories", "categories_done"} and "\\t" in line and line.split("\\t", 1)[0].isdigit():
                    idx, label = line.split("\\t", 1)
                    categories[idx] = label
                    continue
                if mode == "entries" and "\\t" in line:
                    term, category_ids = line.split("\\t", 1)
                    for category_id in category_ids.split():
                        entries.append({
                            "term": term,
                            "category_id": category_id,
                            "category": categories.get(category_id, category_id),
                        })

            parsed_df = pd.DataFrame(entries)
            parsed_df["term_length"] = parsed_df["term"].str.len()
            print("Parsed entries:", len(parsed_df))
            display(parsed_df.head())
            """
        ),
        code_cell(
            """
            plot_count(parsed_df["category"], "MFD2 terms by category")
            plot_text_length(parsed_df["term"], "MFD2 term length distribution")
            display(parsed_df.groupby("category").size().reset_index(name="terms").sort_values("terms", ascending=False))
            """
        ),
        code_cell(
            """
            if SAVE_OUTPUTS:
                ensure_dir(OUT_DIR)
                shutil.copyfile(dic_path, OUT_DIR / "mfd2.dic")
                if docx_files:
                    shutil.copyfile(docx_files[0], OUT_DIR / "mfd2_summary.docx")
                print("Copied standardized MFD2 files to", OUT_DIR)
            else:
                print("Preview only. Set SAVE_OUTPUTS = True and rerun this cell to copy standardized files.")
            """
        ),
    ]
    return notebook("MFD2 Cleaning", cells)


def notebook(title: str, cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
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
                "version": "3.10",
            },
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_notebook(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    notebooks = {
        "clean_ethics.ipynb": ethics_notebook(),
        "clean_normbank.ipynb": normbank_notebook(),
        "clean_mfrc.ipynb": mfrc_notebook(),
        "clean_moralbench.ipynb": moralbench_notebook(),
        "clean_morebench_public.ipynb": morebench_notebook("morebench_public"),
        "clean_morebench_theory.ipynb": morebench_notebook("morebench_theory"),
        "clean_mfd2.ipynb": mfd2_notebook(),
    }

    for name, payload in notebooks.items():
        write_notebook(NOTEBOOK_DIR / name, payload)
        print((NOTEBOOK_DIR / name).relative_to(ROOT).as_posix())


if __name__ == "__main__":
    main()
