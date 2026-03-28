# AI Ethics Repository

This repository contains the code, datasets, notebooks, experiment artefacts, and thesis support material for an AI ethics project centered on three research questions:

1. How do different AI models perform on supervised moral classification datasets?
2. How do different AI systems behave on open-ended moral reasoning benchmarks?
3. Which observable proxy metrics might indicate consciousness-like behaviour in AI systems?

The active codebase is under `src/ai_ethics/`. The rest of the repository holds data, generated results, exploratory notebooks, report assets, and archival references.

## Repository At A Glance

```text
AI-ethics/
|-- src/
|   |-- ai_ethics/
|   |   |-- analysis/      # raw/processed EDA scanners
|   |   |-- data/          # dataset loading, benchmark building, downloads
|   |   |-- evaluation/    # metrics and prompt-eval pipeline
|   |   |-- preprocess/    # dataset-specific preprocessing
|   |   |-- tools/         # notebook/figure/prompt-export utilities
|   |   `-- training/      # sklearn + transformer training
|   `-- *.py               # thin entrypoint wrappers
|-- Data/
|   |-- raw/               # source datasets placed here
|   |-- processed/         # cleaned outputs + benchmark layers
|   |-- eda/               # generated EDA summaries
|   `-- metadata/          # reserved metadata area
|-- notebooks/             # exploratory, cleaning, and report notebooks
|-- results/               # metrics, models, logs, prompt-eval outputs, figures
|-- admin/                 # logbooks, report files, school/admin material
|-- literature/            # papers and reference material
|-- tmp_final_report_*     # temporary extracted report working folders
`-- venv/                  # local virtual environment in this workspace
```

## What Is Active Vs Historical

- Active implementation lives in `src/ai_ethics/` and the wrapper scripts in `src/`.
- `notebooks/` is active exploratory/report work derived from the same codebase.
- `admin/` is useful project context, but some older logbook files still refer to retired `scripts/` paths.
- `literature/` is reference material, not part of the runtime pipeline.
- `tmp_final_report_docx/`, `tmp_final_report_unzip/`, and `tmp_final_report_copy.zip` are temporary report-working artefacts, not core pipeline inputs.

## Environment Setup

This repo is currently organized around Windows PowerShell and `requirement.txt` (singular).

### Standard setup

If you want the helper scripts to line up with `start.ps1`, create the environment as `venv`:

```powershell
.\setup.ps1 -VenvPath venv
.\start.ps1
```

What `setup.ps1` does:

- creates a virtual environment
- upgrades `pip`
- installs CUDA 11.8 PyTorch wheels
- installs the packages from `requirement.txt`

### Manual setup

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirement.txt
```

### Offline wheelhouse workflow

```powershell
.\wheelhouse-download.ps1 -VenvPath venv
.\wheelhouse-install.ps1 -VenvPath venv
```

## Core Code Layout

### `src/ai_ethics/analysis`

- `eda_raw.py`: scans `Data/raw/` and writes dataset/file summaries under `Data/eda/raw/`
- `eda_processed.py`: scans `Data/processed/` and writes file summaries under `Data/eda/processed/`

### `src/ai_ethics/preprocess`

Dataset-specific preprocessing for:

- Hendrycks ETHICS
- NormBank
- MFRC
- MFD2
- MoralBench
- MoReBench public
- MoReBench theory

`run_preprocess_all.py` currently runs the main preprocessing path for ETHICS, NormBank, MFRC, MFD2, MoralBench, and then refreshes processed EDA.

### `src/ai_ethics/data`

- `loader.py`: resolves processed dataset paths and train/test splits
- `download_morebench.py`: downloads MoReBench CSVs into `Data/raw/morebench/`
- `build_research_benchmarks.py`: creates benchmark layers aligned to the research questions
- `interpretive_spec.py`: defines the synthetic interpretive benchmark and metric specs

### `src/ai_ethics/training`

- `model_train.py`: runs supervised baseline experiments
- `run_allocations.py`: batch runner over predefined dataset/model allocations

Supported supervised model families in the current code:

- `tfidf_logreg`
- `tfidf_linearsvc`
- `bow_mnb`
- `distilbert-base-uncased`
- `bert-base-uncased`
- `roberta-base`
- `microsoft/deberta-v3-base`

Generative entries such as `t5-small` and `facebook/bart-base` are scaffolded in the registry, but generative training is not implemented in the current baseline.

### `src/ai_ethics/evaluation`

- `metrics.py`: classification metrics including ECE and multiclass Brier score
- `prompt_eval.py`: prompt-benchmark execution, scoring, and aggregation for reasoning and interpretive layers

Prompt-eval providers currently supported:

- `echo`
- `replay`
- `openai`

### `src/ai_ethics/tools`

Utility scripts for:

- exporting prompt replay templates
- converting filled prompt CSVs into replay JSONL files
- converting source scripts into notebooks
- generating cleaning notebooks
- generating EDA notebooks
- generating thesis figures

## Data Layout

### Raw data

Expected raw inputs under `Data/raw/`:

- `hendryicks-ethics/`
- `normbank/`
- `mfd2/`
- `mfrc/`
- `moralbench/`
- `morebench/`

Only MoReBench has a downloader in this repo:

```powershell
python src\download_morebench.py
```

Other raw datasets are expected to be placed manually in the correct folders.

### Processed benchmark layers

The benchmark build step writes role-specific outputs under `Data/processed/`:

| Layer                      | Dataset / Artefact                |    Rows | Purpose                              |
| -------------------------- | --------------------------------- | ------: | ------------------------------------ |
| `benchmark_supervised`   | `ethics_labeled`                | 111,602 | Trainable ETHICS rows                |
| `benchmark_supervised`   | `ethics_eval_only`              |  23,813 | Unlabeled / ambiguous ETHICS rows    |
| `benchmark_supervised`   | `normbank_readable`             | 155,423 | Reconstructed NormBank text rows     |
| `benchmark_supervised`   | `mfrc_aggregated`               |  17,884 | Text-level dominant-label MFRC rows  |
| `benchmark_supervised`   | `mfrc_annotations`              |  61,226 | Annotator-level MFRC rows            |
| `benchmark_supervised`   | `mfrc_multilabel`               |  17,884 | Text-level MFRC multilabel rows      |
| `benchmark_reasoning`    | `moralbench_items`              |      88 | Cleaned MoralBench prompt items      |
| `benchmark_reasoning`    | `morebench_public_structured`   |     500 | Structured MoReBench public prompts  |
| `benchmark_reasoning`    | `morebench_public_rubric_items` |   8,125 | Rubric-expanded public rows          |
| `benchmark_reasoning`    | `morebench_theory_structured`   |     150 | Structured MoReBench theory prompts  |
| `benchmark_reasoning`    | `morebench_theory_rubric_items` |   2,299 | Rubric-expanded theory rows          |
| `benchmark_interpretive` | `interpretive_benchmark`        |      40 | Consciousness-proxy prompt benchmark |
| `benchmark_interpretive` | `metric_specs`                  |       5 | Proxy metric definitions             |
| `resources`              | `mfd2_terms`                    |   2,104 | Parsed moral lexicon terms           |

The manifest for all of these lives in:

- `Data/processed/benchmark_manifest.csv`
- `Data/processed/benchmark_manifest.jsonl`

### EDA outputs

Generated EDA summaries live under:

- `Data/eda/raw/`
- `Data/eda/processed/`

Important files:

- `raw_dataset_summary.csv`
- `raw_file_summary.csv`
- `processed_file_summary.csv`

## Main Workflows

### 1. Inspect raw data

```powershell
python src\eda_raw.py
```

### 2. Run preprocessing

```powershell
python src\run_preprocess_all.py
```

### 3. Build research benchmark layers

```powershell
python src\build_research_benchmarks.py
```

### 4. Run supervised experiments

Single run:

```powershell
python src\model_train.py --dataset ethics --model tfidf_logreg
```

Batch allocations:

```powershell
python src\run_allocations.py --datasets ethics normbank mfrc
```

### 5. Run prompt-based evaluation

Run:

```powershell
python src\prompt_eval.py run --dataset moralbench --provider openai --model <model_name>
```

Score:

```powershell
python src\prompt_eval.py score --run-id <run_id>
```

Aggregate:

```powershell
python src\prompt_eval.py aggregate --run-id <run_id>
```

### 6. Manual replay workflow for web-chat or copied responses

Export prompt templates:

```powershell
python src\export_prompt_replay_templates.py --limit 5
```

Convert a filled CSV into replay JSONL:

```powershell
python src\build_replay_from_prompt_csv.py --input-csv results\prompt_eval_manual\moralbench_prompts.csv --output-jsonl results\prompt_eval_manual\moralbench_replay_template.jsonl
```

Then run:

```powershell
python src\prompt_eval.py run --dataset moralbench --provider replay --model replay_manual --replay-file results\prompt_eval_manual\moralbench_replay_template.jsonl --run-id moralbench_replay_manual_5 --limit 5
python src\prompt_eval.py score --run-id moralbench_replay_manual_5
python src\prompt_eval.py aggregate --run-id moralbench_replay_manual_5
```

### 7. Generate notebooks and thesis figures

```powershell
python src\convert_src_to_notebooks.py
python -m ai_ethics.tools.generate_thesis_figures
```

## Notebooks

`notebooks/` contains:

- data cleaning notebooks under `notebooks/cleaning/`
- source-derived notebooks under `notebooks/src/`
- EDA notebooks such as `eda_raw.ipynb`, `eda_processed.ipynb`, `eda_visualization.ipynb`
- results/report notebooks such as `chapter8_rq1_results_visualization.ipynb`, `chapter9_results_visualization.ipynb`, `prompt_eval_analysis.ipynb`, and `thesis_results_visualization.ipynb`

These notebooks sit on top of saved data and results rather than replacing the package code.

## Results And Artefacts

### Supervised training outputs

The repository already contains saved experiment outputs under `results/`:

- `metrics.csv`: experiment-level metrics
- `training_run_summaries.csv`: compact run summaries
- `ualitative_examples.txt`: captured misclassification examples
- `models/`: saved sklearn pipelines and transformer checkpoints with config/metric snapshots
- `training_logs/`: per-run `history.csv`, `history.jsonl`, and `summary.json`
- `checkpoints/`: transformer trainer output directories

### Current saved supervised snapshot

Based on the current `results/metrics.csv`:

- ETHICS: the strongest saved run is `microsoft/deberta-v3-base` with `macro_f1` about `0.796`
- MFRC: the strongest saved run is `bert-base-uncased` with `macro_f1` about `0.322`
- NormBank: several saved runs reach `1.0` across major metrics, which is useful to document but should be treated carefully in analysis

### Prompt-eval outputs

`results/prompt_eval/` contains completed run directories such as:

- `moralbench_chatgpt_web_88`
- `moralbench_claude_web_88`
- `moralbench_deepseek_web_88`
- `morebench_public_chatgpt_web_50`
- `interpretive_chatgpt_web_40`

Each run directory typically contains:

- `config.json`
- `responses.jsonl`
- `item_scores.csv`
- `scenario_scores.csv`
- `model_summary.csv`
- `examples.jsonl`

### Figures

There are two figure areas in the repo:

- `results/figures/` for analysis/result figures already generated in this workspace
- `admin/Report/figures/` for thesis/report-ready figure exports and manifest files

## Admin And Supporting Material

### `admin/`

Contains project management and report material, including:

- `Logbook/`
- `Project management/`
- `Report/`
- `School/`

This is useful for project context and reporting, but it is not the runtime code path.

### `literature/`

Contains papers, benchmark references, and third-party comparison material. It supports the research process but is not part of the executable pipeline.

## Known Caveats

- `requirement.txt` is the real dependency file name.
- `start.ps1` activates `venv`, while `setup.ps1` defaults to `.venv`. Use `.\setup.ps1 -VenvPath venv` if you want the scripts to align without editing them.
- `results/ualitative_examples.txt` is spelled with the current on-disk typo and the training code writes to that exact filename.
- Supervised training currently targets single-label datasets. Reasoning and interpretive datasets are handled through `prompt_eval.py`, not `model_train.py`.
- `prompt_eval.py --provider openai` relies on the installed `openai` package and standard client authentication from the environment.
- Some historical documents under `admin/Logbook/` still mention older `scripts/` paths.

## Suggested Starting Point

If you are picking this project up fresh, the shortest reliable path is:

```powershell
.\setup.ps1 -VenvPath venv
.\start.ps1
python src\eda_raw.py
python src\run_preprocess_all.py
python src\build_research_benchmarks.py
python src\model_train.py --dataset ethics --model tfidf_logreg
```

Then inspect:

- `Data/processed/benchmark_manifest.csv`
- `results/metrics.csv`
- `results/training_run_summaries.csv`
- `results/prompt_eval/`
- `notebooks/thesis_results_visualization.ipynb`
