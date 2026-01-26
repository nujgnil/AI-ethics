# Processed Data Report

Date: 2026-01-26

Purpose
Provide a comprehensive snapshot of processed datasets, including schema conformance, row counts, missing-field checks, label distributions, and data-quality observations. This report is intended to support later report writing and reproducibility claims.

Scope and inputs
- Processed data root: `Data/processed`
- Expected normalized schema: `text`, `label`, `dataset`, `task`, `split`, `source_file`, `metadata`
- Processed datasets reviewed: ethics, mfrc, moralbench, normbank, mfd2 (lexicon only)

Method summary (how checks were produced)
- Row counts and missing-field checks were computed directly from the processed CSVs.
- Header conformance checks confirm the expected 7-column schema.
- Dataset-specific summaries were read from:
  - `Data/processed/ethics/summary.csv`
  - `Data/processed/ethics/label_summary.csv`
  - `Data/processed/normbank/label_summary.csv`
  - `Data/processed/ethics/dup_report.csv`
- No edits to data were performed during this audit.

File inventory (processed)
- `Data/processed/ethics/`: `ethics.csv`, `ethics.jsonl`, `summary.csv`, `label_summary.csv`, `dup_report.csv` (and JSONL equivalents)
- `Data/processed/mfrc/`: `mfrc.csv`, `mfrc.jsonl`
- `Data/processed/moralbench/`: `moralbench.csv`, `moralbench.jsonl`
- `Data/processed/normbank/`: `normbank.csv`, `normbank.jsonl`, `label_summary.csv` (and JSONL equivalent)
- `Data/processed/mfd2/`: `mfd2.dic`, `mfd2_summary.docx`

Schema conformance (CSV header check)
All four normalized CSVs include the expected 7-column header:
`text,label,dataset,task,split,source_file,metadata`

Dataset-level row counts (CSV)
- ethics: 135,411 rows
- mfrc: 61,226 rows
- moralbench: 264 rows
- normbank: 155,423 rows

Missing-field checks (CSV)
- ethics: missing `text` and `label` in 23,809 rows; no missing `split`
- mfrc: missing `label` in 61,226 rows; no missing `text` or `split`
- moralbench: missing `label` in 264 rows; missing `split` in 264 rows; no missing `text`
- normbank: no missing `text`, `label`, or `split`

Dataset-specific notes and implications

ETHICS
- `summary.csv` indicates that missing text/label is fully localized to specific splits:
  - commonsense `cm_ambig` (994 rows) is fully unlabeled.
  - utilitarianism `util` splits (train/test/test_hard) are fully unlabeled: 13,737 + 4,807 + 4,271 rows.
- These unlabeled splits account for all 23,809 missing rows.
- `label_summary.csv` (labeled subset only): `{0: 69,402, 1: 42,200}`.
- `dup_report.csv` appears to contain only a header, suggesting no detected duplicates or none above the reporting threshold.
- Implication: these unlabeled splits should be excluded from supervised training or treated as unlabeled/prompt-only data with explicit mention in the report.

MFRC
- All rows have text; all labels are missing.
- This likely indicates that a multi-label schema exists in the source data but has not yet been mapped to `label`.
- Implication: MFRC is not yet usable for supervised modeling under the current schema and requires label mapping or multi-label encoding.

MoralBench
- All rows have text; all labels and splits are missing.
- This matches the dataset's nature as a question/prompt bank rather than a labeled dataset.
- Implication: suitable for evaluation prompts, not supervised training, unless labels and splits are added.

NormBank
- Fully populated `text`, `label`, and `split` fields.
- `label_summary.csv` reports label counts `{0: 68,057, 1: 59,507, 2: 27,859}`.
- Implication: ready for supervised modeling with three-class targets.

MFD2
- Processed output is a lexicon (`mfd2.dic`) plus a summary doc.
- Does not follow the normalized CSV schema and should be treated as a resource for lexicon-based features or analysis.

Data quality observations
- Missingness in ETHICS is fully explained by known unlabeled splits; no other missingness detected.
- MFRC and MoralBench are structurally consistent but missing target labels (by design or pending mapping).
- No duplicates detected in ETHICS based on `dup_report.csv`.
- All processed CSVs conform to the expected schema header.

Implications for modeling and analysis
- ETHICS: define a clear policy for `cm_ambig` and `util` splits (exclude from supervised training, use for unsupervised prompting, or ignore entirely).
- MFRC: decide on label construction (single-label vs multi-label). If multi-label, update schema or document multi-label handling.
- MoralBench: treat as evaluation prompts; optionally assign a split if used for reproducible evaluation.
- NormBank: proceed with supervised training and calibrated evaluation.
- MFD2: use as lexicon feature source rather than direct training data.

Recommended next steps
- MFRC: inspect original Parquet schema and implement label mapping into `label`.
- MoralBench: add `split` assignment (e.g., fixed 80/20 or custom) if needed for evaluation protocols.
- ETHICS: document in the methods section how unlabeled splits are handled.

Reproducibility notes
- Schema specification: `Data/processed/README.md`
- Preprocessing scripts: `scripts/preprocess_ethics.py`, `scripts/preprocess_normbank.py`, `scripts/preprocess_mfrc.py`, `scripts/preprocess_mfd2.py`, `scripts/preprocess_moralbench.py`
- This report reflects the state of processed data as of 2026-01-26.
