# Raw EDA Log (Pre-Processing)

Date: 2026-01-20

Purpose
Create a reproducible, dataset-level and file-level snapshot of raw inputs prior to any preprocessing. This log is meant to support methods reporting, traceability, and later discussion of data constraints in the dissertation.

Scope and inputs
- Raw data root: `Data/raw`
- EDA outputs referenced: `Data/eda/raw/raw_dataset_summary.csv`, `Data/eda/raw/raw_file_summary.csv`
- Script: `scripts/eda_raw.py`
- File types handled: CSV, TSV, JSONL, Parquet, TXT, DIC, DOCX, ZIP (inventory only)

Method summary
The raw EDA script inventories files and extracts light-weight statistics without modifying the underlying data. For structured files (CSV/TSV/JSONL/Parquet), it:
- Infers a candidate text column and label column using column-name heuristics.
- Counts empty values for inferred text/label.
- Computes text length statistics: min, median, p95.
For unstructured formats (TXT/DIC/DOCX), it records line counts only. ZIP files are listed but not parsed. Results are stored in dataset-level and file-level CSV/JSON summaries.

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
- hendrycks_ethics: labels detected only for files with a `label` column; aggregated counts show 0: 69,402 and 1: 42,200. Incomplete because ETHICS tasks use multiple schemas.
- normbank: labels 0: 68,057; 1: 59,507; 2: 27,859 (three-class balance noted).
- mfrc: label columns not detected by heuristic; requires schema inspection in preprocessing.
- moralbench: no labels (prompt sets).

Data quality checks
- Empty text and empty label counts were zero across datasets based on heuristic detection, except where labels were not detected.
- No missing dataset paths detected.
- Presence of ZIP files in `normbank` and `moralbench` indicates unused archives; these should be excluded from modeling pipelines.
- Presence of `.ipynb_checkpoints` files within `moralbench/questions/` (observed in file list) should be excluded during preprocessing.

Methodological implications
- ETHICS contains multiple sub-tasks with differing schemas. Automated inference is partial, so preprocessing must map each task explicitly to canonical fields (text, label, split, metadata).
- NormBank consists of very short normative statements; models may require different tokenization settings and will be sensitive to punctuation and casing decisions.
- MFRC is a sentiment-style corpus with likely multi-label schema; labels require explicit mapping to a single target or a multi-label setup.
- MoralBench appears to be a prompt-only question bank; it is suitable for evaluation prompts but not supervised training without labels.
- MFD2 is lexicon-based and should be treated as a resource rather than a supervised dataset.

Preprocessing requirements derived from this EDA
- ETHICS: build a task map specifying text and label columns per sub-task file; always record original `source_file` in metadata.
- NormBank: confirm label semantics (0/1/2) and document meanings in a data card.
- MFRC: inspect Parquet schema, select which labels to model, and document any aggregation rule.
- MoralBench: exclude `.ipynb_checkpoints` and ZIP; collect TXT prompts into a unified list with foundation metadata derived from folder names.
- MFD2: standardize the lexicon file into `Data/processed/mfd2/mfd2.dic` and record provenance.

Reproducibility
- Run raw EDA: `python scripts/eda_raw.py`
- Outputs: `Data/eda/raw/raw_dataset_summary.csv`, `Data/eda/raw/raw_file_summary.csv`
- Raw inputs are unchanged by this step.

Known limitations
- Column detection is heuristic and may miss nonstandard naming or multi-label schemas.
- Text-length stats are approximate and based on the inferred text column only.
- ZIP archives are not unpacked; they must be inspected separately if needed.

Decisions logged
- Raw EDA is treated as a non-destructive audit; it does not filter or clean any data.
- Any dataset without detected labels is deferred to preprocessing for manual schema mapping.
