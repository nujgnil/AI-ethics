from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

try:
    from .data_loader import DATASET_TASK_MAP, get_single_label_split
    from .evaluate_model import compute_classification_metrics
except ImportError:
    from data_loader import DATASET_TASK_MAP, get_single_label_split
    from evaluate_model import compute_classification_metrics


try:
    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )
except Exception:
    torch = None
    Dataset = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    Trainer = None
    TrainingArguments = None


MODEL_REGISTRY: Dict[str, Dict[str, str]] = {
    "tfidf_logreg": {"type": "sklearn"},
    "tfidf_linearsvc": {"type": "sklearn"},
    "bow_mnb": {"type": "sklearn"},
    "distilbert-base-uncased": {"type": "transformer"},
    "bert-base-uncased": {"type": "transformer"},
    "roberta-base": {"type": "transformer"},
    "microsoft/deberta-v3-base": {"type": "transformer"},
    "t5-small": {"type": "generative"},
    "facebook/bart-base": {"type": "generative"},
}


@dataclass
class TrainOutput:
    metrics: Dict[str, float]
    sample_errors: pd.DataFrame


def _results_dir() -> Path:
    root = Path(__file__).resolve().parents[1]
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _save_metrics_row(row: Dict[str, object]) -> None:
    out_file = _results_dir() / "metrics.csv"
    write_header = not out_file.exists() or out_file.stat().st_size == 0
    with out_file.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _append_qualitative_examples(dataset: str, model_name: str, df: pd.DataFrame) -> None:
    out_file = _results_dir() / "ualitative_examples.txt"
    with out_file.open("a", encoding="utf-8") as f:
        f.write(f"\n[{datetime.now(timezone.utc).isoformat()}] dataset={dataset} model={model_name}\n")
        if df.empty:
            f.write("No errors captured.\n")
            return
        for _, row in df.head(10).iterrows():
            f.write(
                f"- true={row['true_label']} pred={row['pred_label']} "
                f"confidence={row['confidence']:.4f} text={str(row['text'])[:280]}\n"
            )


def _build_sklearn_pipeline(model_name: str) -> Pipeline:
    if model_name == "tfidf_logreg":
        return Pipeline(
            [
                ("vec", TfidfVectorizer(ngram_range=(1, 2), max_features=100_000)),
                ("clf", LogisticRegression(max_iter=200, n_jobs=None)),
            ]
        )
    if model_name == "tfidf_linearsvc":
        base = Pipeline(
            [
                ("vec", TfidfVectorizer(ngram_range=(1, 2), max_features=100_000)),
                ("clf", LinearSVC()),
            ]
        )
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    if model_name == "bow_mnb":
        return Pipeline(
            [
                ("vec", CountVectorizer(ngram_range=(1, 2), max_features=100_000)),
                ("clf", MultinomialNB()),
            ]
        )
    raise ValueError(f"Unsupported sklearn model: {model_name}")


def _encode_labels(train_labels: pd.Series, test_labels: pd.Series) -> Tuple[np.ndarray, np.ndarray, LabelEncoder]:
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels.astype(str))
    y_test = le.transform(test_labels.astype(str))
    return y_train, y_test, le


def train_sklearn_model(
    dataset: str,
    model_name: str,
    max_train_samples: int | None = None,
) -> TrainOutput:
    split = get_single_label_split(dataset)
    train_df = split.train.copy()
    test_df = split.test.copy()

    if max_train_samples and max_train_samples < len(train_df):
        train_df = train_df.sample(n=max_train_samples, random_state=42).reset_index(drop=True)

    y_train, y_test, le = _encode_labels(train_df["label"], test_df["label"])
    model = _build_sklearn_pipeline(model_name)
    model.fit(train_df["text"], y_train)

    y_pred = model.predict(test_df["text"])
    probas = model.predict_proba(test_df["text"]) if hasattr(model, "predict_proba") else None
    metrics = compute_classification_metrics(y_true=y_test, y_pred=y_pred, probas=probas)

    errors = test_df.copy()
    errors["true_label"] = le.inverse_transform(y_test)
    errors["pred_label"] = le.inverse_transform(y_pred)
    if probas is not None:
        errors["confidence"] = probas.max(axis=1)
    else:
        errors["confidence"] = np.nan
    errors = errors[errors["true_label"] != errors["pred_label"]]
    return TrainOutput(metrics=metrics, sample_errors=errors)


def _tokenize_batch(batch, tokenizer, text_col: str = "text"):
    return tokenizer(batch[text_col], truncation=True, padding="max_length", max_length=256)


def train_transformer_model(
    dataset: str,
    model_name: str,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    epochs: int = 1,
) -> TrainOutput:
    if Trainer is None or torch is None:
        raise RuntimeError("transformers/torch not available. Install requirements first.")

    split = get_single_label_split(dataset)
    train_df = split.train.copy()
    test_df = split.test.copy()

    if max_train_samples and max_train_samples < len(train_df):
        train_df = train_df.sample(n=max_train_samples, random_state=42).reset_index(drop=True)
    if max_test_samples and max_test_samples < len(test_df):
        test_df = test_df.sample(n=max_test_samples, random_state=42).reset_index(drop=True)

    y_train, y_test, le = _encode_labels(train_df["label"], test_df["label"])
    train_df = train_df.assign(labels=y_train)
    test_df = test_df.assign(labels=y_test)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(le.classes_))

    train_ds = Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False)
    test_ds = Dataset.from_pandas(test_df[["text", "labels"]], preserve_index=False)
    train_ds = train_ds.map(lambda x: _tokenize_batch(x, tokenizer), batched=True)
    test_ds = test_ds.map(lambda x: _tokenize_batch(x, tokenizer), batched=True)
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    test_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    run_dir = _results_dir() / "checkpoints" / f"{dataset}_{model_name.replace('/', '_')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=20,
        save_strategy="no",
        evaluation_strategy="no",
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, tokenizer=tokenizer)
    trainer.train()

    pred_output = trainer.predict(test_ds)
    logits = pred_output.predictions
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probas = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    y_pred = probas.argmax(axis=1)
    metrics = compute_classification_metrics(y_true=y_test, y_pred=y_pred, probas=probas)

    errors = test_df.copy()
    errors["true_label"] = le.inverse_transform(y_test)
    errors["pred_label"] = le.inverse_transform(y_pred)
    errors["confidence"] = probas.max(axis=1)
    errors = errors[errors["true_label"] != errors["pred_label"]]
    return TrainOutput(metrics=metrics, sample_errors=errors)


def _task_check(dataset: str) -> None:
    task_type = DATASET_TASK_MAP.get(dataset, "single_label")
    if task_type == "multi_label":
        raise NotImplementedError(
            "MFRC is currently preprocessed without a unified supervised label column. "
            "Add label mapping before training."
        )
    if task_type == "prompt_only":
        raise NotImplementedError(
            "MoralBench is prompt-only in current processed form (no labels). "
            "Use it for qualitative/generative evaluation."
        )


def run_single_experiment(args: argparse.Namespace) -> Dict[str, object]:
    _task_check(args.dataset)
    model_type = MODEL_REGISTRY[args.model]["type"]
    if model_type == "generative":
        raise NotImplementedError("Generative model scaffolding is planned but not implemented in this baseline.")

    if model_type == "sklearn":
        output = train_sklearn_model(
            dataset=args.dataset,
            model_name=args.model,
            max_train_samples=args.max_train_samples,
        )
    else:
        output = train_transformer_model(
            dataset=args.dataset,
            model_name=args.model,
            max_train_samples=args.max_train_samples,
            max_test_samples=args.max_test_samples,
            epochs=args.epochs,
        )

    row: Dict[str, object] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "model": args.model,
        "train_limit": args.max_train_samples if args.max_train_samples else "",
        "test_limit": args.max_test_samples if args.max_test_samples else "",
    }
    row.update(output.metrics)

    _save_metrics_row(row)
    _append_qualitative_examples(args.dataset, args.model, output.sample_errors)
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline and transformer models for ethics datasets.")
    parser.add_argument("--dataset", required=True, help="Dataset key under Data/processed, e.g. ethics, normbank")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_REGISTRY.keys()))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    row = run_single_experiment(args)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
