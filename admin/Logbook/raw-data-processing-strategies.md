# Raw Data Processing Strategies

This note summarizes the current processing strategies implemented in the repo and additional strategies you can add for raw ethics datasets.

## Current strategies in this repo
- Schema standardization: normalize records into a common structure (`text`, `label`, `dataset`, `task`, `split`, `source_file`, `metadata`).
- Field selection heuristics: pick the first matching text/label column based on known candidates.
- Text normalization: trim and collapse whitespace before hashing and length stats.
- Duplicate detection: hash normalized text to identify duplicates across files/splits.
- Missingness tracking: count missing text and label values per file/dataset.
- Label distribution summaries: compute counts for each label.
- Type-specific parsing: CSV/TSV/JSONL/Parquet/TXT strategies per dataset.
- Metadata enrichment: add category/task/split/source file and path-derived metadata.
- EDA summaries: per-file and per-dataset stats (rows, text length min/median/p95, empty counts, label counts).

## Additional strategies to consider
1) Label normalization/mapping
   - Map synonyms and numeric labels to a canonical set (e.g., "acceptable"/"yes"/"1" -> `acceptable`).

2) Split integrity checks
   - Ensure no text overlap across train/test/validation splits (extend hash checks across splits).

3) Text cleanup for artifacts
   - Remove HTML tags, repeated punctuation, and boilerplate text if present.

4) Length-based filtering
   - Drop extremely short or long items using thresholds derived from p95 stats.

5) Class balancing
   - Downsample dominant labels or upsample minority labels, depending on modeling goals.

6) Quality flags
   - Flag items with missing labels, low-quality text, or ambiguous labels for optional exclusion.

7) Cross-dataset alignment
   - Align label spaces and task definitions when merging datasets for training.

## Related scripts
- `scripts/preprocessing/preprocess_ethics.py`
- `scripts/preprocessing/preprocess_normbank.py`
- `scripts/preprocessing/preprocess_mfrc.py`
- `scripts/preprocessing/preprocess_moralbench.py`
- `scripts/preprocessing/preprocess_mfd2.py`
- `scripts/EDA/eda_raw.py`
