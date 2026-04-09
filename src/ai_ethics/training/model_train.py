from __future__ import annotations

import argparse
import csv
import json
import inspect
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC

from ..data.loader import DATASET_TASK_MAP, get_single_label_split
from ..evaluation.metrics import compute_classification_metrics


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
    training_log: List[Dict[str, Any]] | None = None
    training_summary: Dict[str, Any] | None = None
    artifact_payload: Dict[str, Any] | None = None


def _results_dir() -> Path:
    root = Path(__file__).resolve().parents[3]
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


def _stringify_scalar(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return json.dumps(value, ensure_ascii=False)


def _write_jsonl(path: Path, records: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _save_training_artifacts(
    experiment_key: str,
    dataset: str,
    model_name: str,
    training_log: List[Dict[str, Any]] | None,
    training_summary: Dict[str, Any] | None,
) -> None:
    if not training_log and not training_summary:
        return

    base_dir = _results_dir() / "training_logs" / experiment_key
    base_dir.mkdir(parents=True, exist_ok=True)

    if training_log:
        # Persist per-step training history so runs can be inspected after training.
        normalized_log = [{k: _stringify_scalar(v) for k, v in record.items()} for record in training_log]
        _write_jsonl(base_dir / "history.jsonl", normalized_log)
        pd.DataFrame(normalized_log).to_csv(base_dir / "history.csv", index=False)

    if training_summary:
        # Persist a compact run summary for later comparison across experiments.
        normalized_summary = {k: _stringify_scalar(v) for k, v in training_summary.items()}
        (base_dir / "summary.json").write_text(
            json.dumps(normalized_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        out_file = _results_dir() / "training_run_summaries.csv"
        write_header = not out_file.exists() or out_file.stat().st_size == 0
        fieldnames = [
            "experiment_key",
            "dataset",
            "model",
            "model_type",
            "train_examples",
            "test_examples",
            "epochs",
            "global_step",
            "train_runtime",
            "train_samples_per_second",
            "train_steps_per_second",
            "total_flos",
            "train_loss",
            "summary_json",
        ]
        row = {
            "experiment_key": experiment_key,
            "dataset": dataset,
            "model": model_name,
            "model_type": normalized_summary.get("model_type", ""),
            "train_examples": normalized_summary.get("train_examples", ""),
            "test_examples": normalized_summary.get("test_examples", ""),
            "epochs": normalized_summary.get("epochs", ""),
            "global_step": normalized_summary.get("global_step", ""),
            "train_runtime": normalized_summary.get("train_runtime", ""),
            "train_samples_per_second": normalized_summary.get("train_samples_per_second", ""),
            "train_steps_per_second": normalized_summary.get("train_steps_per_second", ""),
            "total_flos": normalized_summary.get("total_flos", ""),
            "train_loss": normalized_summary.get("train_loss", ""),
            "summary_json": json.dumps(normalized_summary, ensure_ascii=False),
        }
        with out_file.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def _save_model_artifacts(
    experiment_key: str,
    dataset: str,
    model_name: str,
    artifact_payload: Dict[str, Any] | None,
    run_config: Dict[str, Any],
    metrics_row: Dict[str, object],
) -> None:
    if not artifact_payload:
        return

    artifact_type = artifact_payload.get("artifact_type", "")
    base_dir = _results_dir() / "models" / experiment_key
    base_dir.mkdir(parents=True, exist_ok=True)

    model_dir = base_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    label_classes = artifact_payload.get("label_classes", [])
    (base_dir / "label_mapping.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "model": model_name,
                "label_classes": label_classes,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (base_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (base_dir / "metrics_snapshot.json").write_text(
        json.dumps(metrics_row, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if artifact_type == "transformer":
        # Save the fine-tuned Hugging Face model and tokenizer together.
        model = artifact_payload.get("model")
        tokenizer = artifact_payload.get("tokenizer")
        if model is not None:
            model.save_pretrained(model_dir)
        if tokenizer is not None:
            tokenizer.save_pretrained(model_dir)
    elif artifact_type == "sklearn":
        # Save the fitted sklearn pipeline as a single serialized artifact.
        pipeline = artifact_payload.get("pipeline")
        if pipeline is not None:
            joblib.dump(pipeline, model_dir / "pipeline.joblib")


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
        # Sparse n-gram features with a linear probabilistic classifier.
        return Pipeline(
            [
                ("vec", TfidfVectorizer(ngram_range=(1, 2), max_features=100_000)),
                ("clf", LogisticRegression(max_iter=200, n_jobs=None)),
            ]
        )
    if model_name == "tfidf_linearsvc":
        # Calibrate LinearSVC so downstream metrics can use class probabilities.
        base = Pipeline(
            [
                ("vec", TfidfVectorizer(ngram_range=(1, 2), max_features=100_000)),
                ("clf", LinearSVC()),
            ]
        )
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    if model_name == "bow_mnb":
        # Count-based bag-of-words baseline with multinomial Naive Bayes.
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
    # Load the benchmark-ready split rather than raw data files.
    split = get_single_label_split(dataset)
    train_df = split.train.copy()
    test_df = split.test.copy()

    if max_train_samples and max_train_samples < len(train_df):
        train_df = train_df.sample(n=max_train_samples, random_state=42).reset_index(drop=True)

    # Encode string labels once so all models train on the same target mapping.
    y_train, y_test, le = _encode_labels(train_df["label"], test_df["label"])
    model = _build_sklearn_pipeline(model_name)
    model.fit(train_df["text"], y_train)

    # Run held-out inference and compute probability-aware metrics where possible.
    y_pred = model.predict(test_df["text"])
    probas = model.predict_proba(test_df["text"]) if hasattr(model, "predict_proba") else None
    metrics = compute_classification_metrics(y_true=y_test, y_pred=y_pred, probas=probas)

    # Keep misclassified examples for qualitative inspection alongside aggregate metrics.
    errors = test_df.copy()
    errors["true_label"] = le.inverse_transform(y_test)
    errors["pred_label"] = le.inverse_transform(y_pred)
    if probas is not None:
        errors["confidence"] = probas.max(axis=1)
    else:
        errors["confidence"] = np.nan
    errors = errors[errors["true_label"] != errors["pred_label"]]
    training_summary = {
        "train_examples": int(len(train_df)),
        "test_examples": int(len(test_df)),
        "epochs": None,
        "train_runtime": float("nan"),
        "train_loss": float("nan"),
        "global_step": None,
        "model_type": "sklearn",
    }
    return TrainOutput(
        metrics=metrics,
        sample_errors=errors,
        training_log=[],
        training_summary=training_summary,
        artifact_payload={
            "artifact_type": "sklearn",
            "pipeline": model,
            "label_classes": list(le.classes_),
        },
    )


def _tokenize_batch(batch, tokenizer, text_col: str = "text"):
    # Apply one shared tokenization policy across transformer experiments.
    return tokenizer(batch[text_col], truncation=True, padding="max_length", max_length=256)


def _build_training_arguments(run_dir: Path, epochs: int):
    kwargs = {
        "output_dir": str(run_dir),
        "num_train_epochs": epochs,
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "logging_strategy": "steps",
        "logging_steps": 20,
        "logging_first_step": True,
        "save_strategy": "no",
        "report_to": [],
    }

    # Support both older/newer transformers releases, which differ on this keyword.
    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        kwargs["evaluation_strategy"] = "no"
    elif "eval_strategy" in signature.parameters:
        kwargs["eval_strategy"] = "no"

    return TrainingArguments(**kwargs)


def train_transformer_model(
    dataset: str,
    model_name: str,
    max_train_samples: int | None = None,
    max_test_samples: int | None = None,
    epochs: int = 1,
) -> TrainOutput:
    if Trainer is None or torch is None:
        raise RuntimeError("transformers/torch not available. Install requirements first.")

    # Use the same benchmark-loading path as the classical baselines.
    split = get_single_label_split(dataset)
    train_df = split.train.copy()
    test_df = split.test.copy()

    if max_train_samples and max_train_samples < len(train_df):
        train_df = train_df.sample(n=max_train_samples, random_state=42).reset_index(drop=True)
    if max_test_samples and max_test_samples < len(test_df):
        test_df = test_df.sample(n=max_test_samples, random_state=42).reset_index(drop=True)

    # Encode labels first, then attach them as the supervised target column.
    y_train, y_test, le = _encode_labels(train_df["label"], test_df["label"])
    train_df = train_df.assign(labels=y_train)
    test_df = test_df.assign(labels=y_test)

    # Load the pretrained backbone and convert text rows into tokenized datasets.
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
    args = _build_training_arguments(run_dir, epochs)
    trainer = Trainer(model=model, args=args, train_dataset=train_ds, tokenizer=tokenizer)
    train_result = trainer.train()

    # Convert logits to probabilities so transformer runs use the same metric suite.
    pred_output = trainer.predict(test_ds)
    logits = pred_output.predictions
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probas = exp_logits / exp_logits.sum(axis=1, keepdims=True)
    y_pred = probas.argmax(axis=1)
    metrics = compute_classification_metrics(y_true=y_test, y_pred=y_pred, probas=probas)

    # Capture both hard errors and per-step training history for later analysis.
    errors = test_df.copy()
    errors["true_label"] = le.inverse_transform(y_test)
    errors["pred_label"] = le.inverse_transform(y_pred)
    errors["confidence"] = probas.max(axis=1)
    errors = errors[errors["true_label"] != errors["pred_label"]]
    training_log = [{k: _stringify_scalar(v) for k, v in record.items()} for record in trainer.state.log_history]
    training_summary: Dict[str, Any] = {
        "train_examples": int(len(train_df)),
        "test_examples": int(len(test_df)),
        "epochs": epochs,
        "global_step": int(getattr(trainer.state, "global_step", 0)),
        "model_type": "transformer",
    }
    training_summary.update({k: _stringify_scalar(v) for k, v in train_result.metrics.items()})
    return TrainOutput(
        metrics=metrics,
        sample_errors=errors,
        training_log=training_log,
        training_summary=training_summary,
        artifact_payload={
            "artifact_type": "transformer",
            "model": trainer.model,
            "tokenizer": tokenizer,
            "label_classes": list(le.classes_),
        },
    )


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
    started_at = datetime.now(timezone.utc)
    experiment_key = (
        f"{started_at.strftime('%Y%m%dT%H%M%S%fZ')}_"
        f"{args.dataset}_{args.model.replace('/', '_')}"
    )
    # Guardrails keep this script aligned to single-label supervised benchmarks only.
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
        "timestamp_utc": started_at.isoformat(),
        "dataset": args.dataset,
        "model": args.model,
        "train_limit": args.max_train_samples if args.max_train_samples else "",
        "test_limit": args.max_test_samples if args.max_test_samples else "",
    }
    row.update(output.metrics)

    # Store aggregate metrics, qualitative errors, training logs, and model artifacts together.
    _save_metrics_row(row)
    _append_qualitative_examples(args.dataset, args.model, output.sample_errors)
    _save_training_artifacts(
        experiment_key=experiment_key,
        dataset=args.dataset,
        model_name=args.model,
        training_log=output.training_log,
        training_summary=output.training_summary,
    )
    _save_model_artifacts(
        experiment_key=experiment_key,
        dataset=args.dataset,
        model_name=args.model,
        artifact_payload=output.artifact_payload,
        run_config={
            "timestamp_utc": started_at.isoformat(),
            "dataset": args.dataset,
            "model": args.model,
            "max_train_samples": args.max_train_samples,
            "max_test_samples": args.max_test_samples,
            "epochs": args.epochs,
        },
        metrics_row=row,
    )
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
