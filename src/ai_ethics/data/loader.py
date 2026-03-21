from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from sklearn.model_selection import train_test_split


DATASET_TASK_MAP: Dict[str, str] = {
    "ethics": "single_label",
    "normbank": "single_label",
    "mfrc": "single_label",
    "moralbench": "prompt_only",
    "morebench_public": "prompt_only",
    "morebench_theory": "prompt_only",
    "interpretive": "prompt_only",
    "delphi": "single_label",
}

DATASET_PATH_MAP: Dict[str, tuple[str, ...]] = {
    "ethics": ("benchmark_supervised", "ethics", "ethics_labeled.csv"),
    "normbank": ("benchmark_supervised", "normbank", "normbank_readable.csv"),
    "mfrc": ("benchmark_supervised", "mfrc", "mfrc_aggregated.csv"),
    "moralbench": ("benchmark_reasoning", "moralbench", "moralbench_items.csv"),
    "morebench_public": ("benchmark_reasoning", "morebench_public", "morebench_public_structured.csv"),
    "morebench_theory": ("benchmark_reasoning", "morebench_theory", "morebench_theory_structured.csv"),
    "interpretive": ("benchmark_interpretive", "interpretive", "interpretive_benchmark.csv"),
}


@dataclass
class DatasetSplit:
    train: pd.DataFrame
    test: pd.DataFrame
    task_type: str
    dataset_name: str


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def get_data_root() -> Path:
    root = _project_root()
    for candidate in (root / "Data", root / "data"):
        if candidate.exists():
            return candidate
    return root / "Data"


def get_processed_csv_path(dataset: str) -> Path:
    data_root = get_data_root()
    mapped = DATASET_PATH_MAP.get(dataset)
    candidates = []
    if mapped is not None:
        candidates.append(data_root / "processed" / mapped[0] / mapped[1] / mapped[2])
    candidates.append(data_root / "processed" / dataset / f"{dataset}.csv")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Processed file not found for dataset={dataset}. Checked: {[str(path) for path in candidates]}"
    )


def _safe_json(value: str) -> dict:
    if not isinstance(value, str) or not value:
        return {}
    try:
        return json.loads(value)
    except Exception:
        return {}


def load_processed_dataset(dataset: str) -> pd.DataFrame:
    path = get_processed_csv_path(dataset)
    df = pd.read_csv(path)
    df["dataset"] = df.get("dataset", dataset).fillna(dataset).astype(str)
    df["text"] = df.get("text", "").fillna("").astype(str).str.strip()
    df["label"] = df.get("label", "").fillna("").astype(str).str.strip()
    df["split"] = df.get("split", "").fillna("").astype(str).str.strip()
    if "metadata" in df.columns:
        df["metadata_parsed"] = df["metadata"].map(_safe_json)
    else:
        df["metadata_parsed"] = [{} for _ in range(len(df))]
    return df


def _filter_labeled_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["text"] != "") & (df["label"] != "")].copy()


def _split_from_column(df: pd.DataFrame) -> Optional[DatasetSplit]:
    has_train = (df["split"] == "train").any()
    has_test = (df["split"] == "test").any() or (df["split"] == "test_hard").any()
    if not (has_train and has_test):
        return None

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"].isin(["test", "test_hard"])].copy()
    if train_df.empty or test_df.empty:
        return None

    return DatasetSplit(
        train=train_df,
        test=test_df,
        task_type="single_label",
        dataset_name=str(df["dataset"].iloc[0]) if not df.empty else "unknown",
    )


def get_single_label_split(
    dataset: str,
    test_size: float = 0.2,
    random_state: int = 42,
) -> DatasetSplit:
    df = load_processed_dataset(dataset)
    df = _filter_labeled_rows(df)
    if df.empty:
        raise ValueError(f"No labeled rows available for dataset={dataset}")

    split_from_col = _split_from_column(df)
    if split_from_col is not None:
        return split_from_col

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    return DatasetSplit(
        train=train_df.reset_index(drop=True),
        test=test_df.reset_index(drop=True),
        task_type="single_label",
        dataset_name=dataset,
    )


def available_datasets() -> List[str]:
    available: List[str] = []
    for name in DATASET_PATH_MAP:
        try:
            if get_processed_csv_path(name).exists():
                available.append(name)
        except FileNotFoundError:
            continue
    return sorted(available)
