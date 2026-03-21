# Research Workflow and Rationale

This document explains the current project workflow, why the repository is structured this way, what each stage is trying to achieve, and what results should be expected from each stage. It is written around the three core research questions guiding this project.

## The Three Research Questions

### 1. How do different AI models perform across supervised moral datasets?

This question is about direct benchmark comparison. It focuses on datasets where rows can be treated as labeled examples and where model performance can be measured with standard classification metrics such as accuracy, macro F1, precision, and recall.

### 2. How do different AI models behave on open-ended moral reasoning benchmarks?

This question shifts from simple classification to reasoning quality. Some datasets in this repository are not best understood as right-or-wrong classification tasks. Instead, they contain dilemmas, prompts, or rubric-based evaluation items. These require a different style of evaluation.

### 3. What observable metrics might indicate consciousness-like behaviour in AI systems?

This question is interpretive rather than definitive. The goal is not to prove that an AI system is conscious. The goal is to build and test proxy metrics that capture properties often associated with conscious cognition, such as self-consistency, metacognitive awareness, integration of information, identity stability, and agency coherence.

## Why The Repository Is Structured This Way

The repository was reorganized so that there is a single code root under `src/`.

- `src/ai_ethics/` contains the real implementation
- `src/*.py` contains simple runnable entrypoints
- `notebooks/` contains exploratory and report-style notebook work
- `Data/` contains raw data, processed benchmark layers, and EDA outputs

This layout reduces duplication and makes the project easier to reason about. Previously, similar logic existed in both `scripts/` and `src/`. That split made the project harder to trust because it was not always clear which path was the active one. The current structure makes `src/` the only live code path.

## Dataset Strategy

The project does not treat all datasets as if they were the same. That would be methodologically weak because the datasets represent different task types.

### A. Supervised Comparison Layer

These are the datasets used to answer Research Question 1.

- `ETHICS`
- `NormBank`
- `MFRC`

These datasets are cleaned into trainable or near-trainable forms and stored under:

- `Data/processed/benchmark_supervised/`

### B. Reasoning Benchmark Layer

These are primarily used to answer Research Question 2.

- `MoralBench`
- `MoReBench Public`
- `MoReBench Theory`

These datasets are kept in prompt- or rubric-oriented form rather than being forced into artificial class labels. They are stored under:

- `Data/processed/benchmark_reasoning/`

### C. Interpretive Benchmark Layer

This layer supports Research Question 3.

- `interpretive_benchmark.csv`
- `metric_specs.csv`

These files define proxy-style evaluation prompts and dimensions related to consciousness-like behaviour. They are stored under:

- `Data/processed/benchmark_interpretive/interpretive/`

### D. Resource Layer

This contains supporting analytical resources rather than benchmark rows.

- `MFD2`

This is stored under:

- `Data/processed/resources/mfd2/`

## End-To-End Process

The workflow is best understood as a pipeline with six stages.

### Stage 1. Acquire And Organise Raw Data

The raw datasets are stored under `Data/raw/`. This stage ensures that the project has the original source material in place before any processing occurs.

Example command:

```powershell
python src\download_morebench.py
```

Reasoning:

- raw data should remain conceptually separate from processed data
- acquisition should be reproducible where possible
- source provenance matters for research transparency

Expected result:

- raw files exist in the expected dataset-specific folders
- the repo can proceed to preprocessing without path errors

### Stage 2. Exploratory Data Analysis

EDA is used to understand what each dataset actually is before forcing it into one shared modeling pipeline. This is one of the most important design choices in the project.

There are two kinds of EDA here:

- aggregate raw and processed summaries
- dataset-specific notebooks that explain the structure and meaning of each dataset

Commands:

```powershell
python src\eda_raw.py
python src\eda_processed.py
```

Notebook families:

- `notebooks/eda_raw.ipynb`
- `notebooks/eda_processed.ipynb`
- `notebooks/src/eda/*.ipynb`

Reasoning:

- EDA prevents false assumptions about row structure and label meaning
- some datasets are classification datasets, some are prompt datasets, some are lexicons
- this stage helps justify later cleaning and evaluation decisions

Expected result:

- the reader can see what each dataset looks like
- dataset-specific issues are identified early
- preprocessing choices become defensible rather than arbitrary

### Stage 3. Preprocess Each Dataset Into A Normalised Project Schema

The project uses preprocessing to convert heterogeneous raw sources into stable processed artifacts. The preprocessing logic lives under:

- `src/ai_ethics/preprocess/`

Main command:

```powershell
python src\run_preprocess_all.py
```

Reasoning:

- raw datasets vary widely in structure
- model code should not need to understand every original dataset format
- a normalised schema reduces downstream complexity

The standard processed fields are:

- `text`
- `label`
- `dataset`
- `task`
- `split`
- `source_file`
- `metadata`

Expected result:

- each dataset is converted into a stable processed representation
- dataset-level quirks are preserved in metadata rather than lost
- processed EDA can be rerun consistently

### Stage 4. Build Research Benchmark Layers

After preprocessing, the project creates benchmark layers aligned with the three research questions.

Command:

```powershell
python src\build_research_benchmarks.py
```

Reasoning:

- not every processed dataset should be used the same way
- the same file layout should not imply the same methodological use
- benchmark layers make the project analytically cleaner

The output structure is:

- `benchmark_supervised/`
- `benchmark_reasoning/`
- `benchmark_interpretive/`
- `resources/`

Expected result:

- trainable datasets are separated from reasoning benchmarks
- interpretive prompts are isolated as a distinct evaluation layer
- the next experiment stage can use data by research role rather than by file accident

### Stage 5. Run Supervised Model Experiments

This stage currently has the strongest implementation support and directly addresses Research Question 1.

Main commands:

```powershell
python src\model_train.py --dataset ethics --model tfidf_logreg
python src\model_train.py --dataset normbank --model tfidf_logreg
python src\model_train.py --dataset mfrc --model tfidf_logreg
```

Optional batch runner:

```powershell
python src\run_allocations.py --datasets ethics normbank mfrc
```

Reasoning:

- start with the trainable supervised layer first
- begin with transparent baselines such as TF-IDF + logistic regression
- use stronger baselines after the data path is confirmed working

Current supported supervised-style models include:

- `tfidf_logreg`
- `tfidf_linearsvc`
- `bow_mnb`
- transformer baselines such as DistilBERT, BERT, RoBERTa, and DeBERTa, when dependencies are available

Expected result:

- `results/metrics.csv` is populated with model comparison rows
- `results/ualitative_examples.txt` captures sample errors
- a first pass ranking of models across `ETHICS`, `NormBank`, and `MFRC` becomes available

### Stage 6. Evaluate Reasoning And Interpretive Benchmarks

This stage addresses Research Questions 2 and 3.

Important status note:

- the data layers for reasoning and interpretive evaluation are already built
- the baseline training script does not yet implement full prompt/rubric evaluation for these layers

This means the project is methodologically prepared for these questions, but the evaluator code for them should be treated as the next engineering step rather than something already complete.

Reasoning:

- prompt benchmarks should not be forced through a classifier-only pipeline
- rubric-based evaluation requires different scoring logic
- consciousness-like proxy metrics should be reported as interpretive measures, not proofs

Expected result once implemented:

- qualitative and rubric-based comparisons on `MoralBench` and `MoReBench`
- proxy-style scores for interpretive dimensions such as self-model consistency and metacognitive behaviour

## Why Each Dataset Is Used The Way It Is

### ETHICS

Used primarily for supervised comparison. It contains multiple moral subtasks and is one of the main trainable datasets after preprocessing.

Why:

- broad task coverage
- enough labeled data for meaningful comparison
- good for standard metrics

Expected result:

- useful benchmark of general moral classification behaviour
- model performance should differ by subtask complexity and text style

### NormBank

Used primarily for supervised comparison. It is a norm-status classification dataset rather than a simple right-versus-wrong benchmark.

Why:

- strong data volume
- clear labels
- rich contextual structure

Expected result:

- likely one of the most stable datasets for comparing classical baselines and transformers
- useful for understanding whether models capture social norm status

### MFRC

Used as an extended supervised dataset after aggregation. It is more naturalistic and less controlled than ETHICS or NormBank.

Why:

- real-world discourse matters
- it tests whether models can handle noisier moral language

Expected result:

- lower and less stable performance than cleaner benchmarks would be unsurprising
- disagreement and annotation complexity make this dataset analytically important

### MoralBench

Used as a reasoning benchmark rather than a classifier dataset.

Why:

- it is better suited to prompt-based or response-based evaluation
- classification metrics would flatten away too much of its meaning

Expected result:

- useful for comparing the style and consistency of model responses
- best interpreted qualitatively or through a dedicated evaluator

### MoReBench Public And Theory

Used as rubric-based reasoning benchmarks.

Why:

- the benchmark structure is built around dilemmas and rubric criteria
- the theory subset is especially useful for checking framework-sensitive reasoning

Expected result:

- stronger models may produce more coherent and complete responses
- theory-sensitive prompts may reveal whether models default to generic safety language rather than genuine ethical differentiation

### MFD2

Used as a support resource rather than a main benchmark.

Why:

- it helps with interpretation, lexical analysis, and potentially feature exploration

Expected result:

- useful for describing moral language patterns
- not suitable as a headline train/test benchmark

## Reasoning Behind The Three-Layer Evaluation Design

The project deliberately separates:

- predictive performance
- reasoning behaviour
- consciousness-like proxy behaviour

This matters because these are not interchangeable claims.

### For Research Question 1

The best evidence comes from:

- supervised benchmark metrics
- consistent train/test evaluation
- model comparison across datasets

Typical expected results:

- simpler baselines may remain competitive on some datasets
- transformer models may outperform classical baselines when enough data is available
- model rankings may vary by dataset rather than remaining globally stable

### For Research Question 2

The best evidence comes from:

- prompt quality
- rubric adherence
- theory sensitivity
- qualitative error analysis

Typical expected results:

- some models may appear strong in classification but weak in open-ended reasoning
- some models may produce safe-sounding but shallow responses
- theory-linked prompts may reveal lack of consistent moral framework use

### For Research Question 3

The best evidence comes from:

- proxy metrics, not metaphysical claims
- consistent self-description
- metacognitive calibration
- identity and agency coherence
- cross-context integration

Typical expected results:

- models may show partial consciousness-like traits without warranting claims of actual consciousness
- proxy scores may vary by prompt framing and evaluation design
- the project should present these findings as interpretive indicators, not proofs

## Recommended Current Run Order

If raw data or preprocessing logic changed:

```powershell
venv\Scripts\python.exe src\run_preprocess_all.py
venv\Scripts\python.exe src\build_research_benchmarks.py
```

Then run the current supervised experiments:

```powershell
venv\Scripts\python.exe src\model_train.py --dataset ethics --model tfidf_logreg
venv\Scripts\python.exe src\model_train.py --dataset normbank --model tfidf_logreg
venv\Scripts\python.exe src\model_train.py --dataset mfrc --model tfidf_logreg
venv\Scripts\python.exe src\model_train.py --dataset ethics --model tfidf_linearsvc
venv\Scripts\python.exe src\model_train.py --dataset normbank --model tfidf_linearsvc
```

Then inspect:

- `results/metrics.csv`
- `results/ualitative_examples.txt`

## Expected Near-Term Output Of The Project

If the current workflow is run successfully, the project should produce:

- cleaned dataset layers aligned with research purpose
- EDA notebooks that explain the data clearly
- first-pass supervised comparison metrics
- a reasoned justification for why prompt/rubric benchmarks need a different evaluator
- a prepared interpretive benchmark layer for consciousness-like proxy analysis

## Current Limitations

- reasoning-benchmark evaluation is prepared in data form but not yet fully implemented as a dedicated runner
- interpretive metrics are currently defined as project proxies, not universally accepted measures
- benchmark results should therefore be separated into:
  - completed supervised findings
  - planned reasoning findings
  - planned interpretive findings

## Summary

The project is designed to answer three related but distinct questions:

1. Which models perform best on supervised moral datasets?
2. Which models reason better on open-ended moral benchmarks?
3. Which observable traits suggest consciousness-like behaviour, if any?

The repository structure, preprocessing strategy, benchmark layering, and experiment flow are all built around keeping those questions separate enough to be methodologically defensible while still allowing them to inform one another.
