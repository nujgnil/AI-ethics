# AI-Ethics Project Completion Checklist

Snapshot date: 2026-03-09

## 1) Repository and Environment

- [X] Git repository initialized and clean working tree (`git status` shows no pending changes before this checklist).
- [X] Virtual environment workflow scripts exist: `setup.ps1`, `start.ps1`.
- [X] Offline wheel workflow scripts exist: `wheelhouse-download.ps1`, `wheelhouse-install.ps1`.
- [X] Dependency file exists as `requirement.txt`.
- [ ] Align dependency naming between docs and files (`README.md` mentions `requirements.txt`, repo uses `requirement.txt`).

## 2) Raw Data Collection

- [X] Core raw datasets present under `Data/raw/`:
  - [X] `hendryicks-ethics`
  - [X] `normbank`
  - [X] `mfd2`
  - [X] `mfrc`
  - [X] `moralbench`
- [X] Additional archives collected under `Data/raw-zipped/`.
- [ ] Delphi dataset integrated into active pipeline (currently referenced in planning but not present in active processed outputs).
- [ ] Moral Foundations additional dataset integrated beyond MFD2 lexicon.

## 3) Preprocessing Pipeline

- [X] Active preprocessing code consolidated under `src/ai_ethics/preprocess/`.
- [X] End-to-end preprocessing runner implemented: `src/run_preprocess_all.py`.
- [X] ETHICS preprocessing implemented.
- [X] NormBank preprocessing implemented.
- [X] MFRC preprocessing implemented (raw-to-processed format).
- [X] MoralBench preprocessing implemented (prompt extraction).
- [X] MFD2 preprocessing implemented (resource standardization).
- [X] Shared preprocessing utilities implemented.
- [X] Consolidate/retire legacy preprocess path so one `src` tree is the main code location.

## 4) Processed Data Outputs

- [X] `Data/processed/ethics/` populated (csv/jsonl + summaries + duplicate reports).
- [X] `Data/processed/normbank/` populated (csv/jsonl + label summary).
- [X] `Data/processed/mfrc/` populated (csv/jsonl).
- [X] `Data/processed/moralbench/` populated (csv/jsonl).
- [X] `Data/processed/mfd2/` populated (`mfd2.dic`, summary doc).
- [ ] `Data/metadata/` populated (folder currently empty).

## 5) EDA Pipeline and Outputs

- [X] Raw EDA script implemented: `src/eda_raw.py`.
- [X] Processed EDA script implemented: `src/eda_processed.py`.
- [X] Raw EDA outputs generated under `Data/eda/raw/`.
- [X] Processed EDA outputs generated under `Data/eda/processed/`.
- [X] EDA documentation exists: `README_EDA.md`.
- [X] Visualization notebook exists: `notebooks/eda_visualization.ipynb`.
- [ ] Fully execute all notebook cells and freeze figure export workflow (`results/figures/` currently not standardized).

## 6) Modeling and Evaluation Code

- [X] Dataset loading/splitting implemented: `src/data_loader.py`.
- [X] Metrics implementation exists: `src/evaluate_model.py`.
- [X] Training runner exists: `src/model_train.py`.
- [X] Allocation runner exists: `src/run_allocations.py`.
- [X] Baseline sklearn models implemented (`tfidf_logreg`, `tfidf_linearsvc`, `bow_mnb`).
- [X] Transformer training path implemented for supported datasets.
- [ ] MFRC supervised training support completed (currently blocked by missing unified label mapping; raises `NotImplementedError`).
- [ ] MoralBench supervised training support completed (currently prompt-only; raises `NotImplementedError`).
- [ ] Generative model training/eval path completed (currently planned but not implemented; raises `NotImplementedError`).

## 7) Experiment Execution Outputs

- [ ] Populate `results/metrics.csv` with actual experiment runs (currently empty).
- [ ] Populate `results/ualitative_examples.txt` with misclassification/error samples (currently empty; filename also has typo).
- [ ] Produce non-empty checkpoint artifacts in `results/checkpoints/` (folder exists but contains no files).
- [ ] Add repeatable run logs for completed experiments (commands + config + seed + timestamp).

## 8) Data Readiness for Supervised Training

- [X] NormBank appears fully labeled and split-ready.
- [X] ETHICS labeled subset available for supervised use.
- [ ] Define and document policy for ETHICS unlabeled segments (`cm_ambig`, utilitarian splits with missing labels) in final methodology.
- [ ] Map MFRC multi-label source schema into trainable supervised targets.
- [ ] Decide MoralBench role formally: prompt-only evaluation vs labeled supervised task.

## 9) Documentation Quality

- [X] Main project README exists.
- [X] Preprocessing/EDA/admin logbook documentation is extensive under `admin/Logbook/`.
- [ ] Resolve README placeholders/TBD fields (`add source`, `TBD`, `<your-username>` links).
- [ ] Fix naming/typos and stale references (e.g., `qualitative` filename typo, report structure filename typo).
- [ ] Ensure README repo structure matches actual folders and script paths.

## 10) Testing and Quality Assurance

- [ ] Add automated tests for preprocessing and training utilities (no project test suite currently found).
- [ ] Add sanity-check script for processed schema/missingness/label distributions as CI-style gate.
- [ ] Add reproducibility checks (fixed seed validation run + expected metric range smoke test).

## 11) Suggested Immediate Next Actions

- [ ] Run baseline experiments and write first non-empty `results/metrics.csv`.
- [ ] Implement MFRC label mapping and rerun allocation for MFRC models.
- [ ] Clean documentation inconsistencies in `README.md`.
- [ ] Rename `ualitative_examples.txt` to `qualitative_examples.txt` (and update writer path in `src/model_train.py`).
- [ ] Add a minimal `tests/` folder with smoke tests for preprocessing and metric functions.
