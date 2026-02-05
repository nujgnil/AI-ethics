# Script Rundown

Date: 2026-02-05

This is a practical rundown of what each runnable/support file currently does.

## Root PowerShell workflow scripts

| File                        | What it does                                                                                                                           |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `setup.ps1`               | Creates a virtual env, activates it, upgrades pip, installs CUDA 11.8 PyTorch wheels, then installs packages from `requirement.txt`. |
| `start.ps1`               | Moves to project root and activates the virtual environment (`.\venv\Scripts\Activate.ps1`).                                         |
| `wheelhouse-download.ps1` | Pre-downloads all wheels (PyTorch +`requirement.txt`) into a local `wheelhouse/` for offline or repeat installs.                   |
| `wheelhouse-install.ps1`  | Installs PyTorch and all requirements from local `wheelhouse/` only (`--no-index`).                                                |

## Training and evaluation scripts (`src/`)

| File                        | What it does                                                                                                                                                                                                                                       |
| --------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `src/data_loader.py`      | Loads processed dataset CSVs, normalizes key columns, filters labeled rows, and returns train/test splits (from `split` column if available, otherwise stratified split).                                                                        |
| `src/evaluate_model.py`   | Computes evaluation metrics for classification runs: Accuracy, Macro/Micro-F1, Balanced Accuracy, MCC, AUROC, PR-AUC, Brier score, and ECE.                                                                                                        |
| `src/model_train.py`      | Main experiment runner for one dataset + one model. Trains sklearn baselines and transformer classifiers, evaluates results, appends metrics to `results/metrics.csv`, and writes misclassified examples to `results/ualitative_examples.txt`. |
| `src/run_allocations.py`  | Batch runner for the model-to-dataset allocation plan (from your model-selection logbook). Iterates allocations and runs `src/model_train.py` logic with skip reasons when not supported.                                                        |
| `src/preprocess_utils.py` | Small shared helpers for seed setting and metadata JSON parsing in the training pipeline.                                                                                                                                                          |

## Legacy/alternate preprocessing scripts in `src/preprocess/`

These appear to be an older preprocessing path and are not the main path used by `scripts/preprocessing/run_preprocess_all.py`.

| File                                               | What it does                                                                                                                   |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `src/preprocess/_common.py`                      | Shared utilities for the legacy preprocess flow (repo paths, CSV/JSONL writers, simple normalization helpers).                 |
| `src/preprocess/preprocess_hendryicks_ethics.py` | Legacy ETHICS preprocessing: reads raw task CSVs, extracts text/label fields, tracks duplicates, and writes processed outputs. |
| `src/preprocess/preprocess_morebench_public.py`  | Legacy conversion of `morebench_public.csv` to processed CSV/JSONL with source metadata.                                     |
| `src/preprocess/preprocess_morebench_theory.py`  | Legacy conversion of `morebench_theory.csv` to processed CSV/JSONL with source metadata.                                     |
| `src/preprocess/preprocess_normbank.py`          | Legacy NormBank text-file parser that extracts lines and setting/category metadata into processed outputs.                     |

## Active preprocessing scripts (`scripts/preprocessing/`)

| File                                               | What it does                                                                                                                                                         |
| -------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `scripts/preprocessing/preprocess_utils.py`      | Utility functions used by preprocessing scripts: directory creation, CSV/JSONL writing, field normalization, and label summary counting.                             |
| `scripts/preprocessing/preprocess_ethics.py`     | Active ETHICS preprocessing to normalized schema:`text,label,dataset,task,split,source_file,metadata`; also creates summary, label summary, and duplicate reports. |
| `scripts/preprocessing/preprocess_normbank.py`   | Active NormBank preprocessing (parquet or CSV) into normalized schema with label summaries.                                                                          |
| `scripts/preprocessing/preprocess_mfrc.py`       | Active MFRC preprocessing from parquet into normalized schema; stores original annotation payload under `metadata`, `label` currently left blank.                |
| `scripts/preprocessing/preprocess_mfd2.py`       | Copies MFD2 lexicon (`.dic`) and optional summary doc to processed folder.                                                                                         |
| `scripts/preprocessing/preprocess_moralbench.py` | Converts MoralBench question files into normalized prompt rows with extracted metadata (collection/foundation), no labels by default.                                |
| `scripts/preprocessing/run_preprocess_all.py`    | Orchestrates all active preprocessing scripts, then runs processed-data EDA generation.                                                                              |

## EDA scripts (`scripts/EDA/`)

| File                             | What it does                                                                                                  |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `scripts/EDA/eda_raw.py`       | Scans raw datasets and produces raw-level EDA summaries (file and dataset summaries) under `data/eda/raw/`. |
| `scripts/EDA/eda_processed.py` | Scans `data/processed/` outputs and writes processed-level EDA summaries under `data/eda/processed/`.     |

## Notes

- The active production preprocessing path is under `scripts/preprocessing/`.
- The model training/evaluation path is under `src/`.
- `mfrc` and `moralbench` currently need additional label/split handling before full supervised allocation runs.
