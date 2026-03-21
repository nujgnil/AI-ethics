# AI Ethics

This repository contains the code, notebooks, and data notes for comparing AI models on:

- supervised moral classification datasets
- open-ended moral reasoning benchmarks
- interpretive proxy metrics related to consciousness-like behaviour

## Repository Structure

```text
AI-ethics/
|-- Data/
|   |-- raw/
|   |-- processed/
|   |-- eda/
|-- notebooks/
|   |-- cleaning/
|   |-- src/
|-- src/
|   |-- ai_ethics/
|   |   |-- analysis/
|   |   |-- data/
|   |   |-- evaluation/
|   |   |-- preprocess/
|   |   |-- tools/
|   |   `-- training/
|   |-- build_research_benchmarks.py
|   |-- download_morebench.py
|   |-- eda_processed.py
|   |-- eda_raw.py
|   `-- run_preprocess_all.py
|-- admin/
|-- literature/
`-- results/
```

## Code Layout

There is now a single code root: `src/`.

- `src/ai_ethics/` contains the actual implementation.
- `src/*.py` contains thin entrypoints for running common tasks directly.
- the old `scripts/` tree has been removed.

## Main Commands

Run from the repository root:

```powershell
python src\download_morebench.py
python src\eda_raw.py
python src\eda_processed.py
python src\run_preprocess_all.py
python src\build_research_benchmarks.py
python src\model_train.py --dataset ethics --model tfidf_logreg
python src\run_allocations.py
```

## Data Layout

The current processed outputs are organised by benchmark role:

- `Data/processed/benchmark_supervised/`
  - `ethics`
  - `normbank`
  - `mfrc`
- `Data/processed/benchmark_reasoning/`
  - `moralbench`
  - `morebench_public`
  - `morebench_theory`
- `Data/processed/benchmark_interpretive/`
  - `interpretive`
- `Data/processed/resources/`
  - `mfd2`

## Notes

- `Data/` is ignored in git except for markdown notes.
- some archival notes under `admin/Logbook/` still mention the removed `scripts/` layout because they are historical records.
