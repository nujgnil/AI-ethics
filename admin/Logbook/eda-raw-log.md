# Raw EDA Log (Pre-Processing)

Date: YYYY-MM-DD

Purpose
Document a reproducible, dataset-level and file-level snapshot of raw inputs prior to any preprocessing. This log is intended to support methods reporting and traceability for later analysis and publication.

Data sources and paths
- Raw data root: `data/raw`
- Summary outputs used: `data/eda/raw/raw_dataset_summary.csv`, `data/eda/raw/raw_file_summary.csv`
- Script: `scripts/eda_raw.py`

Method summary
The raw EDA script performs a file-level inventory and extracts light-weight statistics without modifying data. For CSV/TSV/JSONL/Parquet it attempts to infer a primary text column and label column based on name hints, counts empty values, and computes text length statistics (min, median, p95). For TXT/DIC/Docx it records line counts only. Results are stored as dataset-level and file-level CSV and JSON summaries.

Dataset-level totals (from raw_dataset_summary.csv)
- hendrycks_ethics: 135,423 rows, 17 files
- normbank: 155,423 rows, 3 files
- mfd2: 2,116 rows, 2 files
- mfrc: 61,226 rows, 1 file
- moralbench: 792 rows, 177 files

Dataset storage footprint (approx)
- hendrycks_ethics: 35.6 MB
- normbank: 61.5 MB
- mfd2: 0.06 MB
- mfrc: 4.1 MB
- moralbench: 0.10 MB

File type inventory
- hendrycks_ethics: 16 CSV, 1 TXT
- normbank: 1 CSV, 1 JSON, 1 ZIP
- mfd2: 1 DIC, 1 DOCX
- mfrc: 1 Parquet
- moralbench: 176 TXT, 1 ZIP

Inferred schema signals (heuristic)
- hendrycks_ethics: text columns detected include `scenario` (majority) and `input` (minority); label column detected as `label` in some files.
- normbank: text column `norm`, label column `label`.
- mfrc: text column `text` detected; label columns not detected by the heuristic (likely multi-label columns or nonstandard naming).
- mfd2: lexicon format, no text/label columns expected.
- moralbench: question bank in TXT files, no label columns expected.

Text length statistics (approx, per dataset)
- hendrycks_ethics: min 10 chars, median ~183 chars, p95 ~813 chars.
- normbank: min 5 chars, median ~6 chars, p95 ~8 chars (very short norm strings).
- mfrc: min 33 chars, median ~157 chars, p95 ~472 chars.
- mfd2: not applicable.
- moralbench: not applicable (TXT prompts only).

Label distributions (where detected)
- hendrycks_ethics: labels detected only for files with a `label` column; counts aggregated show 0: 69,402 and 1: 42,200. This is incomplete because not all ETHICS tasks share the same schema or label naming. Final label accounting must be done in preprocessing per task.
- normbank: labels 0: 68,057; 1: 59,507; 2: 27,859 (three-class balance noted).
- mfrc: label columns not detected by heuristic; needs schema inspection in preprocessing.
- moralbench: no labels (prompt sets).

Data quality checks
- Empty text and empty label counts were zero across datasets based on heuristic detection.
- No missing dataset paths detected.
- Presence of ZIP files in `normbank` and `moralbench` suggests unused archives that should not be included in modeling pipelines.
- Presence of `.ipynb_checkpoints` files within `moralbench/questions/` (observed in file list) should be excluded during preprocessing.

Methodological implications
- ETHICS appears to contain multiple sub-tasks with different schemas; automated inference is partial. Preprocessing must map each task explicitly to canonical fields (text, label, split, metadata).
- NormBank consists of very short normative statements; models may require different tokenization settings and may be sensitive to punctuation and casing decisions.
- MFRC is a sentiment-style corpus; labels are likely multi-column and will require explicit mapping to a single target or a multi-label setup.
- MoralBench appears to be a prompt-only question bank; it is suitable for evaluation prompts but not supervised training without added labels.
- MFD2 is lexicon-based and should be treated as a resource, not a supervised dataset.

Recommendations for preprocessing
- ETHICS: define a task map that specifies which column is text and which column is label for each sub-task file; track original file name in metadata.
- NormBank: confirm label semantics (0/1/2) and document the meaning in the data card.
- MFRC: inspect parquet schema and decide whether to aggregate moral foundation dimensions into a single label or keep multi-label targets.
- MoralBench: exclude `.ipynb_checkpoints` and ZIP; collect TXT prompts into a unified prompt list with foundation metadata from directory names.
- MFD2: standardize the lexicon file into `data/processed/mfd2/mfd2.dic` and record a provenance note.

Reproducibility
Run raw EDA: `python scripts/eda_raw.py`
Outputs: `data/eda/raw/raw_dataset_summary.csv`, `data/eda/raw/raw_file_summary.csv`

Notes on limitations
The EDA uses heuristic column detection based on column names. Datasets with missing headers, unusual naming, or multi-label schemas may not be captured correctly. Detailed schema inspection is required during preprocessing for any dataset where inferred columns are empty or ambiguous.
