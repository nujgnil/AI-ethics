from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_DIR = ROOT / "notebooks" / "src" / "eda"


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


def notebook(title: str, cells: list[dict]) -> dict:
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


def base_setup_cell(raw_path: str, processed_path: str, title: str) -> str:
    return f"""
from __future__ import annotations

from ast import literal_eval
from collections import Counter
from pathlib import Path
import csv
import json
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
pd.set_option("display.max_colwidth", 160)


def find_project_root() -> tuple[Path, Path]:
    for candidate in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        for data_name in ("Data", "data"):
            data_root = candidate / data_name
            if data_root.exists():
                return candidate, data_root
    raise FileNotFoundError("Could not find project data directory.")


ROOT, DATA_ROOT = find_project_root()
RAW_PATH = DATA_ROOT / "{raw_path}"
PROCESSED_PATH = DATA_ROOT / "{processed_path}"

print("Notebook:", "{title}")
print("Project root:", ROOT)
print("Raw path:", RAW_PATH)
print("Processed path:", PROCESSED_PATH)


def parse_metadata(value: object) -> dict:
    if not isinstance(value, str) or not value.strip():
        return {{}}
    try:
        return json.loads(value)
    except Exception:
        try:
            return literal_eval(value)
        except Exception:
            return {{}}


def preview_frame(df: pd.DataFrame, n: int = 5) -> None:
    display(df.head(n))


def count_chars(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.len()


def markdown(text: str) -> None:
    display(Markdown(text))
"""


def ethics_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # ETHICS EDA

            The main story of ETHICS is not just that it is large. It is that the benchmark is assembled from five different moral tasks with different raw file formats, different sentence styles, and uneven label coverage. This notebook therefore starts by making the raw heterogeneity visible, then shows how preprocessing turns those files into one normalized project dataset.
            """
        ),
        code_cell(base_setup_cell("raw/hendryicks-ethics", "processed/ethics/ethics.csv", "ETHICS EDA")),
        code_cell(
            """
TEXT_FIELDS = ["input", "scenario", "sentence", "text", "question", "prompt", "story", "statement"]
LABEL_FIELDS = ["label", "answer", "gold", "gold_label", "target", "output"]

TASK_DESCRIPTIONS = {
    "commonsense": "everyday morality stories and social judgments",
    "deontology": "rule-based scenarios with explicit excuses",
    "justice": "fairness and interpersonal treatment scenarios",
    "utilitarianism": "paired outcomes with no explicit raw labels",
    "virtue": "scenario + trait judgments using [SEP]",
}

FORMAT_NOTES = {
    ("commonsense", "ambig"): "freeform ambiguous stories, no explicit labels",
    ("commonsense", "default"): "csv with label + input + boolean flags",
    ("deontology", "default"): "csv with label + scenario + excuse",
    ("justice", "default"): "csv with label + scenario",
    ("utilitarianism", "default"): "headerless paired outcomes, no explicit labels",
    ("virtue", "default"): "csv with label + scenario [SEP] trait",
}

def split_from_name(task_name: str, stem: str) -> str:
    for suffix in ["train", "test_hard", "test", "ambig"]:
        needle = f"_{suffix}"
        if stem.endswith(needle):
            return suffix
    return ""

def first_present(columns: list[str], candidates: list[str]) -> str:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name in lower:
            return lower[name]
    return ""

raw_frames = []
file_rows = []

for task_dir in [p for p in sorted(RAW_PATH.iterdir()) if p.is_dir()]:
    task_name = task_dir.name
    for path in sorted(task_dir.glob("*.csv")):
        split = split_from_name(task_name, path.stem)
        rel = str(path.relative_to(RAW_PATH))

        if task_name == "commonsense" and split == "ambig":
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                reader = csv.reader(f)
                rows = [row[0].strip() for row in reader if row and row[0].strip()]
            if rows:
                rows = rows[1:]
            df = pd.DataFrame({"text": rows})
            df["label"] = ""
            text_field = "freeform_line"
            label_field = ""
            raw_format = FORMAT_NOTES[(task_name, "ambig")]
        elif task_name == "utilitarianism":
            df = pd.read_csv(path, header=None, names=["option_a", "option_b"])
            if not df.empty:
                df = df.iloc[1:].reset_index(drop=True)
            df["text"] = (
                df["option_a"].fillna("").astype(str).str.strip()
                + " [ALT] "
                + df["option_b"].fillna("").astype(str).str.strip()
            ).str.strip()
            df["label"] = ""
            text_field = "option_a + option_b"
            label_field = ""
            raw_format = FORMAT_NOTES[(task_name, "default")]
        else:
            df = pd.read_csv(path)
            text_field = first_present(list(df.columns), TEXT_FIELDS)
            label_field = first_present(list(df.columns), LABEL_FIELDS)
            df["text"] = df[text_field].fillna("").astype(str).str.strip() if text_field else ""
            df["label"] = df[label_field].fillna("").astype(str).str.strip() if label_field else ""
            raw_format = FORMAT_NOTES.get((task_name, "default"), "csv with task-specific fields")

        df["task"] = task_name
        df["split"] = split
        df["source_file"] = rel
        df["raw_format"] = raw_format
        df["text_field"] = text_field
        df["label_field"] = label_field or "none"
        df["has_label"] = df["label"].astype(str).str.strip().ne("")
        df["char_len"] = count_chars(df["text"])
        raw_frames.append(df)

        file_rows.append({
            "task": task_name,
            "split": split,
            "source_file": rel,
            "rows": len(df),
            "raw_format": raw_format,
            "text_field": text_field,
            "label_field": label_field or "none",
        })

raw_df = pd.concat(raw_frames, ignore_index=True, sort=False)
file_inventory = pd.DataFrame(file_rows)

task_story_df = (
    raw_df.groupby("task", as_index=False)
    .agg(
        rows=("text", "size"),
        files=("source_file", "nunique"),
        labeled_rows=("has_label", "sum"),
        unlabeled_rows=("has_label", lambda s: int((~s).sum())),
        median_chars=("char_len", "median"),
    )
)
task_story_df["task_description"] = task_story_df["task"].map(TASK_DESCRIPTIONS)
task_story_df["label_readiness"] = task_story_df.apply(
    lambda row: "fully labeled raw task" if row["unlabeled_rows"] == 0 else "contains unlabeled raw rows",
    axis=1,
)

markdown("### The Raw Benchmark At A Glance")
display(task_story_df.sort_values("rows", ascending=False))
"""
        ),
        code_cell(
            """
markdown("### Why ETHICS Needs Preprocessing")
display(
    pd.DataFrame(
        [
            {
                "problem": "raw task formats differ",
                "evidence": "the five tasks store text differently, so one global dataframe view is misleading",
            },
            {
                "problem": "some raw rows have no explicit labels",
                "evidence": "the commonsense ambiguous file and all utilitarianism files are unlabeled in raw form",
            },
            {
                "problem": "training code expects one schema",
                "evidence": "the project normalizes everything to text / label / task / split / metadata",
            },
        ]
    )
)
"""
        ),
        code_cell(
            """
example_rows = []
for task_name, group in raw_df.groupby("task", sort=False):
    row = group.loc[group["text"].fillna("").astype(str).str.strip().ne("")].head(1)
    if row.empty:
        continue
    row = row.iloc[0]
    example_rows.append({
        "task": task_name,
        "what_it_looks_like": TASK_DESCRIPTIONS.get(task_name, ""),
        "source_file": row["source_file"],
        "raw_format": row["raw_format"],
        "label": row.get("label", "") or "unlabeled in raw file",
        "text": str(row.get("text", ""))[:320],
    })

markdown("### Representative Raw Examples By Task")
display(pd.DataFrame(example_rows))
"""
        ),
        code_cell(
            """
plt.figure(figsize=(10, 6))
sns.barplot(data=task_story_df.sort_values("rows", ascending=False), x="rows", y="task", color="#4C78A8")
plt.title("Raw Rows By ETHICS Task")
plt.xlabel("Rows")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
label_readiness_df = task_story_df.melt(
    id_vars=["task"],
    value_vars=["labeled_rows", "unlabeled_rows"],
    var_name="row_type",
    value_name="count",
)
plt.figure(figsize=(10, 6))
sns.barplot(data=label_readiness_df, x="count", y="task", hue="row_type")
plt.title("Raw Label Coverage By Task")
plt.xlabel("Rows")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
plt.figure(figsize=(11, 6))
sns.boxplot(data=raw_df, x="char_len", y="task", color="#72B7B2")
plt.title("Raw Text Length By Task")
plt.xlabel("Characters")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
labeled_only = raw_df.loc[raw_df["has_label"]].copy()
if labeled_only.empty:
    print("No explicitly labeled raw rows were detected.")
else:
    label_counts = labeled_only.groupby(["task", "label"]).size().reset_index(name="rows")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=label_counts, x="rows", y="task", hue="label")
    plt.title("Label Distribution For Raw Tasks With Explicit Labels")
    plt.xlabel("Rows")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
"""
        ),
        code_cell(
            """
processed_df = pd.read_csv(PROCESSED_PATH)
processed_df["text"] = processed_df["text"].fillna("").astype(str)
processed_df["label"] = processed_df["label"].fillna("").astype(str)

markdown("### Processed Snapshot")
display(pd.DataFrame({
    "processed_rows": [len(processed_df)],
    "usable_labeled_rows": [int(((processed_df["text"].str.strip() != "") & (processed_df["label"].str.strip() != "")).sum())],
    "task_types": [processed_df["task"].nunique()],
    "splits": [processed_df["split"].nunique()],
    "standard_schema_columns": [", ".join(processed_df.columns.tolist())],
}))
processed_examples = (
    processed_df.loc[(processed_df["text"].str.strip() != "") & (processed_df["label"].str.strip() != "")]
    .groupby("task", sort=False)
    .head(2)
    .loc[:, ["text", "label", "task", "split", "source_file"]]
    .reset_index(drop=True)
)
display(processed_examples)
"""
        ),
        code_cell(
            """
comparison_df = pd.DataFrame([
    {
        "stage": "raw",
        "rows": len(raw_df),
        "nonempty_text": int(raw_df["text"].fillna("").astype(str).str.strip().ne("").sum()),
        "nonempty_label": int(raw_df["label"].fillna("").astype(str).str.strip().ne("").sum()),
        "what_this_stage_means": "heterogeneous task files with uneven label coverage",
    },
    {
        "stage": "processed",
        "rows": len(processed_df),
        "nonempty_text": int(processed_df["text"].str.strip().ne("").sum()),
        "nonempty_label": int(processed_df["label"].str.strip().ne("").sum()),
        "what_this_stage_means": "single project schema ready for supervised filtering",
    },
])
markdown("### Raw vs Processed Comparison")
display(comparison_df)
"""
        ),
        markdown_cell(
            """
            The comparison above is the key pipeline move. The raw benchmark contains text for every task, but not every task provides explicit labels in raw form. Preprocessing does not invent new supervision; it standardizes the benchmark into one common schema and makes the already-labeled portions easy to filter, train on, and compare.
            """
        ),
        markdown_cell(
            """
            ## Key Takeaways

            - ETHICS should be read as five related benchmarks, not as one uniform CSV dataset.
            - The raw files mix several sentence styles: everyday stories, rule-based excuses, fairness scenarios, trait judgments, and paired utilitarian outcomes.
            - The raw benchmark is only partially label-complete, which is why the preprocessing step matters.
            - After normalization, ETHICS becomes one of the strongest trainable datasets in the project because the labeled portions can be filtered and compared under a common schema.
            """
        ),
    ]
    return notebook("ETHICS EDA", cells)


def normbank_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # NormBank EDA

            This notebook shows why NormBank needs context-aware EDA. The raw label is easy to count, but the actual meaning of each row comes from the setting, behavior, and constraints metadata.
            """
        ),
        code_cell(base_setup_cell("raw/normbank/NormBank.csv", "processed/normbank/normbank.csv", "NormBank EDA")),
        code_cell(
            """
raw_df = pd.read_csv(RAW_PATH)
markdown("### Raw Snapshot")
display(pd.DataFrame({
    "rows": [len(raw_df)],
    "columns": [len(raw_df.columns)],
    "unique_labels": [raw_df["label"].nunique(dropna=True)],
    "unique_settings": [raw_df["setting"].nunique(dropna=True)],
}))
preview_frame(raw_df)
"""
        ),
        code_cell(
            """
label_map = {"0": "taboo", "1": "normal", "2": "expected", 0: "taboo", 1: "normal", 2: "expected"}
raw_df["label_name"] = raw_df["label"].map(label_map).fillna(raw_df["label"].astype(str))
raw_df["case_text"] = (
    "Setting: " + raw_df["setting"].fillna("").astype(str)
    + " | Behavior: " + raw_df["behavior"].fillna("").astype(str)
    + " | Constraints: " + raw_df["constraints"].fillna("").astype(str)
)

markdown("### Representative Reconstructed Cases")
display(raw_df[["label_name", "norm", "setting", "behavior", "constraints", "case_text"]].head(8))
"""
        ),
        code_cell(
            """
plt.figure(figsize=(9, 6))
sns.countplot(data=raw_df, y="label_name", order=raw_df["label_name"].value_counts().index, color="#4C78A8")
plt.title("Raw NormBank Label Distribution")
plt.xlabel("Rows")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
top_settings = raw_df["setting"].fillna("unknown").value_counts().head(12).rename_axis("setting").reset_index(name="count")
plt.figure(figsize=(11, 7))
sns.barplot(data=top_settings, x="count", y="setting", color="#F58518")
plt.title("Most Common Settings In Raw NormBank")
plt.xlabel("Rows")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
top_behaviors = raw_df["behavior"].fillna("unknown").value_counts().head(12).rename_axis("behavior").reset_index(name="count")
plt.figure(figsize=(11, 7))
sns.barplot(data=top_behaviors, x="count", y="behavior", color="#54A24B")
plt.title("Most Common Behaviors In Raw NormBank")
plt.xlabel("Rows")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
heatmap_df = (
    raw_df.loc[raw_df["setting"].isin(top_settings["setting"])]
    .pivot_table(index="setting", columns="label_name", values="behavior", aggfunc="count", fill_value=0)
    .reindex(index=top_settings["setting"])
)
plt.figure(figsize=(11, 8))
sns.heatmap(heatmap_df, annot=False, cmap="YlGnBu")
plt.title("Norm Status By Setting")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
processed_df = pd.read_csv(PROCESSED_PATH)
processed_df["metadata_parsed"] = processed_df["metadata"].map(parse_metadata)
processed_df["context_preview"] = processed_df["metadata_parsed"].map(lambda meta: f"Setting: {meta.get('setting', '')} | Behavior: {meta.get('behavior', '')}"[:180])

markdown("### Processed Snapshot")
display(pd.DataFrame({
    "processed_rows": [len(processed_df)],
    "usable_labeled_rows": [int(((processed_df["text"].fillna('').str.strip() != '') & (processed_df["label"].fillna('').astype(str).str.strip() != '')).sum())],
    "label_values": [", ".join(sorted(processed_df["label"].astype(str).unique().tolist()))],
}))
display(processed_df[["text", "label", "split", "context_preview"]].head(8))
"""
        ),
        code_cell(
            """
comparison_df = pd.DataFrame([
    {"stage": "raw", "rows": len(raw_df), "text_like_field": "norm", "label_style": "0 / 1 / 2"},
    {"stage": "processed", "rows": len(processed_df), "text_like_field": "text", "label_style": "0 / 1 / 2"},
])
markdown("### Raw vs Processed Comparison")
display(comparison_df)
"""
        ),
        markdown_cell(
            """
            ## Key Takeaways

            - NormBank is not just short text classification; the real meaning of each row lives in the social context fields.
            - The label scheme is best read as `taboo`, `normal`, and `expected`, not simply `right` versus `wrong`.
            - The processed version keeps the pipeline simple by placing the norm status in `label` and pushing contextual detail into `metadata`.
            """
        ),
    ]
    return notebook("NormBank EDA", cells)


def mfrc_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # MFRC EDA

            MFRC differs from the benchmark-style datasets because it contains natural Reddit discourse rather than authored moral scenarios. This notebook shows the raw annotation structure, then checks why the processed version is still not yet ready for ordinary supervised training in the current pipeline.
            """
        ),
        code_cell(base_setup_cell("raw/mfrc/train.parquet", "processed/mfrc/mfrc.csv", "MFRC EDA")),
        code_cell(
            """
raw_df = pd.read_parquet(RAW_PATH)
markdown("### Raw Snapshot")
display(pd.DataFrame({
    "rows": [len(raw_df)],
    "columns": [len(raw_df.columns)],
    "subreddits": [raw_df["subreddit"].nunique() if "subreddit" in raw_df.columns else None],
    "annotations": [raw_df["annotation"].nunique() if "annotation" in raw_df.columns else None],
}))
preview_frame(raw_df)
"""
        ),
        code_cell(
            """
example_cols = [c for c in ["text", "subreddit", "bucket", "annotator", "annotation", "confidence"] if c in raw_df.columns]
markdown("### Representative Raw Reddit Examples")
display(raw_df[example_cols].head(8))
"""
        ),
        code_cell(
            """
raw_df["char_len"] = count_chars(raw_df["text"])
plt.figure(figsize=(10, 6))
sns.histplot(raw_df["char_len"], bins=40, kde=True, color="#4C78A8")
plt.title("Raw MFRC Text Length")
plt.xlabel("Characters")
plt.ylabel("Rows")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
top_subreddits = raw_df["subreddit"].fillna("unknown").value_counts().head(12).rename_axis("subreddit").reset_index(name="count")
plt.figure(figsize=(11, 7))
sns.barplot(data=top_subreddits, x="count", y="subreddit", color="#F58518")
plt.title("Most Common Subreddits In MFRC")
plt.xlabel("Rows")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
annotation_counts = raw_df["annotation"].fillna("unknown").value_counts().head(12).rename_axis("annotation").reset_index(name="count")
plt.figure(figsize=(11, 7))
sns.barplot(data=annotation_counts, x="count", y="annotation", color="#54A24B")
plt.title("Raw Moral Annotation Distribution")
plt.xlabel("Rows")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
if "confidence" in raw_df.columns:
    plt.figure(figsize=(9, 5))
    sns.countplot(data=raw_df, y="confidence", order=raw_df["confidence"].fillna("unknown").value_counts().index, color="#E45756")
    plt.title("Annotation Confidence Distribution")
    plt.xlabel("Rows")
    plt.ylabel("")
    plt.tight_layout()
    plt.show()
"""
        ),
        code_cell(
            """
repeat_examples = (
    raw_df.groupby("text")
    .size()
    .reset_index(name="annotation_count")
    .query("annotation_count > 1")
    .sort_values("annotation_count", ascending=False)
)
markdown("### Multi-Annotated Comments")
display(repeat_examples.head(5))
if not repeat_examples.empty:
    sample_text = repeat_examples.iloc[0]["text"]
    display(raw_df.loc[raw_df["text"] == sample_text, [c for c in ["text", "annotation", "annotator", "confidence"] if c in raw_df.columns]].head(10))
"""
        ),
        code_cell(
            """
processed_df = pd.read_csv(PROCESSED_PATH)
processed_df["metadata_parsed"] = processed_df["metadata"].map(parse_metadata)
processed_df["annotation"] = processed_df["metadata_parsed"].map(lambda meta: meta.get("labels", {}).get("annotation", ""))

markdown("### Processed Snapshot")
display(pd.DataFrame({
    "processed_rows": [len(processed_df)],
    "nonempty_text_rows": [int(processed_df["text"].fillna("").astype(str).str.strip().ne("").sum())],
    "nonempty_label_rows": [int(processed_df["label"].fillna("").astype(str).str.strip().ne("").sum())],
    "annotation_values_in_metadata": [int(processed_df["annotation"].astype(str).str.strip().ne("").sum())],
}))
display(processed_df[["text", "label", "task", "split", "annotation"]].head(8))
"""
        ),
        code_cell(
            """
comparison_df = pd.DataFrame([
    {"stage": "raw", "rows": len(raw_df), "ready_for_flat_supervised_training": "no", "reason": "multiple annotations / rich metadata"},
    {"stage": "processed", "rows": len(processed_df), "ready_for_flat_supervised_training": "no", "reason": "label column is blank; annotations remain nested"},
])
markdown("### Raw vs Processed Comparison")
display(comparison_df)
"""
        ),
        markdown_cell(
            """
            ## Key Takeaways

            - MFRC contains natural social media language, so the text is noisier and more realistic than benchmark-style ethics prompts.
            - The dataset is annotation-rich, but the annotations are not yet collapsed into one clean supervised label in the current pipeline.
            - This makes MFRC valuable for future multi-label or annotator-aware work, but not yet a direct plug-in replacement for ETHICS or NormBank.
            """
        ),
    ]
    return notebook("MFRC EDA", cells)


def moralbench_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # MoralBench EDA

            MoralBench is better understood as a prompt bank than as a standard sentence classification dataset. This notebook inventories the raw prompt files, shows what the prompt text looks like, and then compares that raw structure with the processed representation used in this project.
            """
        ),
        code_cell(base_setup_cell("raw/moralbench", "processed/moralbench/moralbench.csv", "MoralBench EDA")),
        code_cell(
            """
records = []
for path in sorted(RAW_PATH.rglob("*.txt")):
    rel = path.relative_to(RAW_PATH)
    lines = [line.strip() for line in path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()]
    for idx, line in enumerate(lines, 1):
        records.append({
            "source_file": str(rel),
            "collection": rel.parts[0] if len(rel.parts) > 0 else "",
            "foundation": rel.parts[2] if len(rel.parts) > 2 else "",
            "line_no": idx,
            "text": line,
            "char_len": len(line),
        })

raw_df = pd.DataFrame(records)
markdown("### Raw Snapshot")
display(pd.DataFrame({
    "prompt_lines": [len(raw_df)],
    "prompt_files": [raw_df["source_file"].nunique()],
    "collections": [raw_df["collection"].nunique()],
    "foundations": [raw_df["foundation"].nunique()],
}))
preview_frame(raw_df)
"""
        ),
        code_cell(
            """
markdown("### Representative Prompt Fragments")
display(raw_df[["collection", "foundation", "text"]].head(10))
"""
        ),
        code_cell(
            """
collection_counts = raw_df["collection"].value_counts().rename_axis("collection").reset_index(name="count")
plt.figure(figsize=(9, 5))
sns.barplot(data=collection_counts, x="count", y="collection", color="#4C78A8")
plt.title("Prompt Count By Collection")
plt.xlabel("Rows")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
foundation_counts = raw_df["foundation"].replace("", "unknown").value_counts().head(12).rename_axis("foundation").reset_index(name="count")
plt.figure(figsize=(10, 6))
sns.barplot(data=foundation_counts, x="count", y="foundation", color="#F58518")
plt.title("Prompt Count By Foundation / Source Folder")
plt.xlabel("Rows")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
plt.figure(figsize=(10, 6))
sns.histplot(raw_df["char_len"], bins=30, kde=True, color="#54A24B")
plt.title("Prompt Length Distribution")
plt.xlabel("Characters")
plt.ylabel("Rows")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
processed_df = pd.read_csv(PROCESSED_PATH)
processed_df["metadata_parsed"] = processed_df["metadata"].map(parse_metadata)
processed_df["foundation"] = processed_df["metadata_parsed"].map(lambda meta: meta.get("foundation", ""))

markdown("### Processed Snapshot")
display(pd.DataFrame({
    "processed_rows": [len(processed_df)],
    "nonempty_text_rows": [int(processed_df["text"].fillna("").astype(str).str.strip().ne("").sum())],
    "nonempty_label_rows": [int(processed_df["label"].fillna("").astype(str).str.strip().ne("").sum())],
    "task_values": [", ".join(sorted(processed_df["task"].fillna("").astype(str).unique().tolist()))],
}))
display(processed_df[["text", "label", "task", "foundation"]].head(10))
"""
        ),
        code_cell(
            """
comparison_df = pd.DataFrame([
    {"stage": "raw", "rows": len(raw_df), "structure": "prompt lines from text files", "training_ready": "no"},
    {"stage": "processed", "rows": len(processed_df), "structure": "standardized prompt table", "training_ready": "no"},
])
markdown("### Raw vs Processed Comparison")
display(comparison_df)
"""
        ),
        markdown_cell(
            """
            ## Key Takeaways

            - MoralBench is a prompt-oriented evaluation resource, not a flat-label training dataset.
            - The raw files are short prompt fragments and question items, so example displays matter more than generic label plots.
            - The processed file standardizes the prompt bank for downstream evaluation, but it remains label-free in the current pipeline.
            """
        ),
    ]
    return notebook("MoralBench EDA", cells)


def morebench_notebook(dataset_name: str, pretty_name: str) -> dict:
    cells = [
        markdown_cell(
            f"""
            # {pretty_name} EDA

            This notebook profiles `{pretty_name}` as an evaluation-style dilemma benchmark. It first shows the raw dilemma records and rubric structure, then compares them with the normalized version stored in this repository.
            """
        ),
        code_cell(base_setup_cell(f"raw/morebench/{dataset_name}.csv", f"processed/{dataset_name}/{dataset_name}.csv", f"{pretty_name} EDA")),
        code_cell(
            """
raw_df = pd.read_csv(RAW_PATH)

def parse_rubric_count(value: object) -> int:
    if not isinstance(value, str) or not value.strip():
        return 0
    try:
        parsed = literal_eval(value)
        if isinstance(parsed, list):
            return len(parsed)
    except Exception:
        pass
    return 0

raw_df["rubric_items"] = raw_df["RUBRIC"].map(parse_rubric_count)
raw_df["char_len"] = count_chars(raw_df["DILEMMA"])

markdown("### Raw Snapshot")
display(pd.DataFrame({
    "rows": [len(raw_df)],
    "columns": [len(raw_df.columns)],
    "dilemma_types": [raw_df["DILEMMA_TYPE"].nunique()],
    "role_domains": [raw_df["ROLE_DOMAIN"].nunique()],
    "contexts": [raw_df["CONTEXT"].nunique()],
}))
preview_frame(raw_df)
"""
        ),
        code_cell(
            """
sample_df = raw_df[["DILEMMA_TYPE", "THEORY", "ROLE_DOMAIN", "CONTEXT", "rubric_items", "DILEMMA"]].copy()
sample_df["DILEMMA"] = sample_df["DILEMMA"].astype(str).str.slice(0, 320)
markdown("### Representative Dilemmas")
display(sample_df.head(8))
"""
        ),
        code_cell(
            """
plt.figure(figsize=(9, 5))
sns.countplot(data=raw_df, y="DILEMMA_TYPE", order=raw_df["DILEMMA_TYPE"].value_counts().index, color="#4C78A8")
plt.title("Dilemma Type Distribution")
plt.xlabel("Rows")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
plt.figure(figsize=(9, 5))
sns.countplot(data=raw_df, y="ROLE_DOMAIN", order=raw_df["ROLE_DOMAIN"].value_counts().index, color="#F58518")
plt.title("Role Domain Distribution")
plt.xlabel("Rows")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
top_contexts = raw_df["CONTEXT"].fillna("unknown").value_counts().head(12).rename_axis("CONTEXT").reset_index(name="count")
plt.figure(figsize=(11, 7))
sns.barplot(data=top_contexts, x="count", y="CONTEXT", color="#54A24B")
plt.title("Most Common Contexts")
plt.xlabel("Rows")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
plt.figure(figsize=(10, 6))
sns.boxplot(data=raw_df, x="char_len", y="DILEMMA_TYPE", color="#72B7B2")
plt.title("Dilemma Length By Type")
plt.xlabel("Characters")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
plt.figure(figsize=(10, 6))
sns.histplot(raw_df["rubric_items"], bins=20, kde=False, color="#E45756")
plt.title("Rubric Item Count")
plt.xlabel("Rubric items per dilemma")
plt.ylabel("Rows")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
processed_df = pd.read_csv(PROCESSED_PATH)
processed_df["rubric_items"] = processed_df["RUBRIC"].map(parse_rubric_count)
processed_df["char_len"] = count_chars(processed_df["DILEMMA"])

markdown("### Processed Snapshot")
display(pd.DataFrame({
    "processed_rows": [len(processed_df)],
    "flat_label_column_present": ["label" in processed_df.columns],
    "dilemma_types": [processed_df["DILEMMA_TYPE"].nunique()],
    "theory_values": [processed_df["THEORY"].nunique()],
}))
display(processed_df[["DILEMMA_TYPE", "THEORY", "ROLE_DOMAIN", "CONTEXT", "rubric_items", "DILEMMA"]].head(8))
"""
        ),
        code_cell(
            """
comparison_df = pd.DataFrame([
    {"stage": "raw", "rows": len(raw_df), "evaluation_style": "rubric-based", "flat_training_label": "no"},
    {"stage": "processed", "rows": len(processed_df), "evaluation_style": "rubric-based", "flat_training_label": "no"},
])
markdown("### Raw vs Processed Comparison")
display(comparison_df)
"""
        ),
        markdown_cell(
            """
            ## Key Takeaways

            - These rows are open-ended dilemmas, so example displays and rubric summaries are more informative than ordinary class-balance plots.
            - The benchmark is evaluation-oriented rather than simple right/wrong classification.
            - The processed version mostly standardizes the table structure for downstream use; it does not convert the benchmark into a flat supervised label dataset.
            """
        ),
    ]
    return notebook(f"{pretty_name} EDA", cells)


def morebench_theory_notebook() -> dict:
    cells = morebench_notebook("morebench_theory", "MoReBench Theory")["cells"]
    insert_at = len(cells) - 2
    cells = cells[:insert_at] + [
        code_cell(
            """
theory_counts = raw_df["THEORY"].fillna("unknown").value_counts().rename_axis("THEORY").reset_index(name="count")
plt.figure(figsize=(11, 7))
sns.barplot(data=theory_counts, x="count", y="THEORY", color="#B279A2")
plt.title("Cases By Moral Theory")
plt.xlabel("Rows")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
theory_type = pd.crosstab(raw_df["THEORY"], raw_df["DILEMMA_TYPE"])
plt.figure(figsize=(11, 7))
sns.heatmap(theory_type, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Theory By Dilemma Type")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
    ] + cells[insert_at:]
    return notebook("MoReBench Theory EDA", cells)


def mfd2_notebook() -> dict:
    cells = [
        markdown_cell(
            """
            # MFD2 EDA

            MFD2 is not a sentence-level benchmark. It is a lexical resource, so the EDA should focus on category coverage and term organization rather than ordinary row labels or train/test splits.
            """
        ),
        code_cell(base_setup_cell("raw/mfd2", "processed/mfd2", "MFD2 EDA")),
        code_cell(
            """
dic_path = next(RAW_PATH.glob("*.dic"))
lines = dic_path.read_text(encoding="utf-8", errors="replace").splitlines()

categories = {}
terms = []
section = 0
for line in lines:
    line = line.strip()
    if not line:
        continue
    if line == "%":
        section += 1
        continue
    if section == 1:
        idx, label = line.split("\\t", 1)
        categories[idx] = label
    elif section >= 2:
        term, cat_id = line.split("\\t", 1)
        terms.append({"term": term, "category_id": cat_id, "category": categories.get(cat_id, cat_id)})

terms_df = pd.DataFrame(terms)
terms_df[["foundation", "polarity"]] = terms_df["category"].str.split(".", n=1, expand=True)

markdown("### Raw Snapshot")
display(pd.DataFrame({
    "dictionary_files": [len(list(RAW_PATH.glob('*')))],
    "categories": [len(categories)],
    "terms": [len(terms_df)],
    "foundations": [terms_df["foundation"].nunique()],
}))
preview_frame(terms_df)
"""
        ),
        code_cell(
            """
markdown("### Sample Lexicon Entries")
display(terms_df[["term", "category", "foundation", "polarity"]].head(12))
"""
        ),
        code_cell(
            """
category_counts = terms_df["category"].value_counts().rename_axis("category").reset_index(name="count")
plt.figure(figsize=(10, 6))
sns.barplot(data=category_counts, x="count", y="category", color="#4C78A8")
plt.title("Term Count By Category")
plt.xlabel("Terms")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
foundation_polarity = pd.crosstab(terms_df["foundation"], terms_df["polarity"])
plt.figure(figsize=(9, 6))
sns.heatmap(foundation_polarity, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Foundation By Polarity")
plt.xlabel("")
plt.ylabel("")
plt.tight_layout()
plt.show()
"""
        ),
        code_cell(
            """
processed_files = [{"file": path.name, "suffix": path.suffix, "bytes": path.stat().st_size} for path in sorted(PROCESSED_PATH.glob("*"))]

markdown("### Processed Snapshot")
display(pd.DataFrame(processed_files))
"""
        ),
        code_cell(
            """
comparison_df = pd.DataFrame([
    {"stage": "raw", "resource_type": "dictionary", "sentence_rows": "not applicable"},
    {"stage": "processed", "resource_type": "copied lexical resource", "sentence_rows": "not applicable"},
])
markdown("### Raw vs Processed Comparison")
display(comparison_df)
"""
        ),
        markdown_cell(
            """
            ## Key Takeaways

            - MFD2 should be presented as a lexical support resource rather than a normal benchmark dataset.
            - The useful EDA questions are about category coverage and term distribution, not label balance or split quality.
            - Its role in the project is interpretive and analytical rather than direct model training.
            """
        ),
    ]
    return notebook("MFD2 EDA", cells)


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    notebooks = {
        "eda_ethics.ipynb": ethics_notebook(),
        "eda_normbank.ipynb": normbank_notebook(),
        "eda_mfrc.ipynb": mfrc_notebook(),
        "eda_moralbench.ipynb": moralbench_notebook(),
        "eda_morebench_public.ipynb": morebench_notebook("morebench_public", "MoReBench Public"),
        "eda_morebench_theory.ipynb": morebench_theory_notebook(),
        "eda_mfd2.ipynb": mfd2_notebook(),
    }
    for name, payload in notebooks.items():
        path = NOTEBOOK_DIR / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
