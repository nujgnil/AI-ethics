from __future__ import annotations

import argparse
import json
from typing import Dict, List

try:
    from .model_train import MODEL_REGISTRY, run_single_experiment
except ImportError:
    from src.model_train import MODEL_REGISTRY, run_single_experiment


MODEL_ALLOCATIONS: Dict[str, List[str]] = {
    "ethics": [
        "tfidf_logreg",
        "tfidf_linearsvc",
        "distilbert-base-uncased",
        "bert-base-uncased",
        "roberta-base",
        "microsoft/deberta-v3-base",
    ],
    "normbank": [
        "tfidf_logreg",
        "bow_mnb",
        "distilbert-base-uncased",
        "bert-base-uncased",
        "roberta-base",
    ],
    "mfrc": [
        "tfidf_logreg",
        "distilbert-base-uncased",
        "bert-base-uncased",
    ],
    "moralbench": [
        "t5-small",
        "facebook/bart-base",
    ],
    "delphi": [
        "tfidf_logreg",
        "distilbert-base-uncased",
        "bert-base-uncased",
        "t5-small",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the model-to-dataset allocations from the logbook.")
    parser.add_argument("--datasets", nargs="*", default=[], help="Subset of datasets to run.")
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    targets = args.datasets if args.datasets else list(MODEL_ALLOCATIONS.keys())
    outputs = []

    for dataset in targets:
        for model in MODEL_ALLOCATIONS.get(dataset, []):
            if model not in MODEL_REGISTRY:
                continue
            exp_args = argparse.Namespace(
                dataset=dataset,
                model=model,
                max_train_samples=args.max_train_samples,
                max_test_samples=args.max_test_samples,
                epochs=args.epochs,
            )
            try:
                row = run_single_experiment(exp_args)
                row["status"] = "ok"
            except Exception as exc:
                row = {
                    "dataset": dataset,
                    "model": model,
                    "status": "skipped",
                    "reason": str(exc),
                }
            outputs.append(row)
            print(json.dumps(row, ensure_ascii=False))

    print(json.dumps({"total": len(outputs)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
