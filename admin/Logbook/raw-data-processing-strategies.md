# Raw Data Processing Strategies

Date: 2026-01-20

Purpose
Provide a comprehensive record of preprocessing decisions, pipeline stages, and quality checks applied to raw ethics datasets. This document serves as a methods reference for the dissertation and as a reproducibility guide.

Processing goals
- Normalize heterogeneous datasets into a shared schema.
- Preserve provenance and traceability for each record.
- Maintain a non-destructive pipeline with raw inputs unchanged.
- Provide quality checks and summary statistics for reporting.

Canonical processed schema
All supervised datasets are normalized to:
- `text`: input string used for modeling
- `label`: target label (string or numeric)
- `dataset`: dataset identifier (e.g., ethics, normbank)
- `task`: task subtype within dataset
- `split`: train/test/valid or dataset-specific split
- `source_file`: original file name or path
- `metadata`: JSON string with additional fields

Current strategies implemented in this repo
- Schema standardization: normalize into the shared schema above.
- Field selection heuristics: choose text/label columns based on known candidates.
- Text normalization: trim and collapse whitespace before hashing and length stats.
- Duplicate detection: hash normalized text to identify duplicates across files/splits.
- Missingness tracking: count missing text and label values per file/dataset.
- Label distribution summaries: compute counts for each label.
- Type-specific parsing: CSV/TSV/JSONL/Parquet/TXT strategies per dataset.
- Metadata enrichment: add category/task/split/source file and path-derived metadata.
- EDA summaries: per-file and per-dataset stats (rows, text length min/median/p95, empty counts, label counts).

Pipeline stages (conceptual)
1) Inventory and parse raw files
   - Identify file types and load using format-appropriate reader.
   - Record path, size, and inferred dataset.
2) Schema mapping
   - Map dataset-specific columns to `text` and `label`.
   - Attach `dataset`, `task`, `split`, `source_file`.
3) Cleaning and normalization
   - Normalize whitespace in text.
   - Optional: lowercasing (only if documented and aligned with baseline models).
4) Quality checks
   - Missing text / label counts.
   - Duplicate detection via text hash.
   - Split leakage check (if splits exist).
5) Outputs
   - Save to CSV and JSONL.
   - Write summary files (label counts, per-file stats).

Dataset-specific mappings (summary)
- ETHICS: multiple tasks (commonsense, deontology, justice, virtue, utilitarianism) with task-specific schema. Must map each task file explicitly.
- NormBank: `norm` as text, `label` as target; short strings only.
- MFRC: Parquet with likely multi-label fields; mapping required for chosen label strategy.
- MoralBench: TXT prompts, no labels; process into prompt list with metadata.
- MFD2: lexicon file only; store as resource rather than supervised dataset.

Decisions logged (current)
- Preserve raw inputs unmodified; all processed outputs live under `Data/processed`.
- Record `source_file` to preserve traceability and support error analysis.
- Keep `metadata` as JSON string to avoid schema fragmentation.
- Do not force label mapping across datasets unless explicitly justified.
- Treat prompt-only datasets as evaluation resources, not supervised training.

Known data-quality issues (from EDA)
- ETHICS has task splits with missing labels (e.g., ambig/util splits).
- MFRC labels are not yet mapped to the canonical `label` field.
- MoralBench contains no labels and no explicit train/test split.
- ZIP archives exist in raw data and should be excluded from modeling pipelines.

Recommended enhancements (future)
1) Label normalization and mapping
   - Map synonyms and numeric labels to a canonical set when merging datasets.
2) Split integrity checks
   - Ensure no text overlap across train/test/validation splits using hashes.
3) Text cleanup for artifacts
   - Remove HTML tags, boilerplate headers, and repeated punctuation if present.
4) Length-based filtering
   - Drop extreme outliers based on dataset-specific thresholds.
5) Class balancing (optional)
   - Downsample dominant labels or upsample minority labels for experiments.
6) Quality flags
   - Flag low-quality or ambiguous items rather than delete them.
7) Cross-dataset alignment
   - Align label spaces only when the research question requires it.

Audit and reproducibility checklist
- Script versions recorded in the logbook at time of run.
- Processed data includes `source_file` for every row.
- Raw EDA summaries saved prior to preprocessing.
- Dataset-specific mapping rules documented.
- Any exclusions or filters explicitly listed in logs.

Related scripts
- `scripts/preprocess_ethics.py`
- `scripts/preprocess_normbank.py`
- `scripts/preprocess_mfrc.py`
- `scripts/preprocess_moralbench.py`
- `scripts/preprocess_mfd2.py`
- `scripts/eda_raw.py`

Notes for reporting
When writing the dissertation methods section, explicitly state:
- Which datasets were used for supervised training vs evaluation-only prompts.
- How labels were mapped for each dataset.
- How missing labels and ambiguous entries were handled.
- What quality checks were performed and whether any filtering occurred.
