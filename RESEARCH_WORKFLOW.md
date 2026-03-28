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

Use this order.

## Small smoke tests

These are just to confirm:

* the run finishes
* metrics are written
* training logs are saved
* model artifacts are saved

### Sklearn smoke tests

venv\Scripts\python.exe src\model_train.py --dataset ethics --model tfidf_logreg --max-train-samples 500 --max-test-samples 200 --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset ethics --model tfidf_linearsvc --max-train-samples 500 --max-test-samples 200 --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset normbank --model bow_mnb --max-train-samples 500 --max-test-samples 200 --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset mfrc --model tfidf_logreg --max-train-samples 500 --max-test-samples 200 --epochs 1

### Transformer smoke tests

venv\Scripts\python.exe src\model_train.py --dataset ethics --model distilbert-base-uncased --max-train-samples 64 --max-test-samples 32 --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset ethics --model bert-base-uncased --max-train-samples 64 --max-test-samples 32 --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset ethics --model roberta-base --max-train-samples 64 --max-test-samples 32 --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset ethics --model microsoft/deberta-v3-base --max-train-samples 64 --max-test-samples 32 --epochs 1

venv\Scripts\python.exe src\model_train.py --dataset normbank --model distilbert-base-uncased --max-train-samples 64 --max-test-samples 32 --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset normbank --model bert-base-uncased --max-train-samples 64 --max-test-samples 32 --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset normbank --model roberta-base --max-train-samples 64 --max-test-samples 32 --epochs 1

venv\Scripts\python.exe src\model_train.py --dataset mfrc --model distilbert-base-uncased --max-train-samples 64 --max-test-samples 32 --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset mfrc --model bert-base-uncased --max-train-samples 64 --max-test-samples 32 --epochs 1

`

## Full baseline runs

Run these after the smoke tests pass.

### Sklearn full runs

venv\Scripts\python.exe src\model_train.py --dataset ethics --model tfidf_logreg --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset ethics --model tfidf_linearsvc --epochs 1

venv\Scripts\python.exe src\model_train.py --dataset normbank --model tfidf_logreg --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset normbank --model bow_mnb --epochs 1

venv\Scripts\python.exe src\model_train.py --dataset mfrc --model tfidf_logreg --epochs 1

### Transformer full runs

venv\Scripts\python.exe src\model_train.py --dataset ethics --model distilbert-base-uncased --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset ethics --model bert-base-uncased --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset ethics --model roberta-base --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset ethics --model microsoft/deberta-v3-base --epochs 1

venv\Scripts\python.exe src\model_train.py --dataset normbank --model distilbert-base-uncased --epochs 1

venv\Scripts\python.exe src\model_train.py --dataset normbank --model bert-base-uncased --epochs 1

venv\Scripts\python.exe src\model_train.py --dataset normbank --model roberta-base --epochs 1

venv\Scripts\python.exe src\model_train.py --dataset mfrc --model distilbert-base-uncased --epochs 1
venv\Scripts\python.exe src\model_train.py --dataset mfrc --model bert-base-uncased --epochs 1

## What to check after each successful run

You should see:

* a new row in [metrics.csv](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-26.318.11754-win32-x64/webview/)
* a new folder under **results/models/**
* a new folder under **results/training_logs/**

If you want, I can also give you the same commands grouped into:

* **test only**
* **full only**
* **missing models only**

## Summary

The project is designed to answer three related but distinct questions:

1. Which models perform best on supervised moral datasets?
2. Which models reason better on open-ended moral benchmarks?
3. Which observable traits suggest consciousness-like behaviour, if any?

The repository structure, preprocessing strategy, benchmark layering, and experiment flow are all built around keeping those questions separate enough to be methodologically defensible while still allowing them to inform one another.

**RQ1**
To answer “How do different AI models perform across supervised moral datasets?”:

* Clean and freeze [metrics.csv](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-26.318.11754-win32-x64/webview/): remove duplicates, keep one final row per dataset-model run, and mark which are baseline vs later tuned runs.
* Build one summary table per dataset and one cross-dataset table:
  dataset | model | accuracy | macro_f1 | balanced_accuracy | mcc
* Rank models by **macro_f1** and **balanced_accuracy**, not just accuracy.
* Add error analysis from [ualitative_examples.txt](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-26.318.11754-win32-x64/webview/): identify common failure types such as ambiguity, norm conflict, rare labels, and overconfident mistakes.
* Compare dataset difficulty:
  a model that is strong on NormBank but weaker on ETHICS or MFRC is useful evidence that “moral dataset performance” is not one thing.
* Write the answer in dissertation language as:
  “baseline supervised comparison across ETHICS, NormBank, and MFRC.”

**RQ2**
To answer “How do different AI models behave on open-ended moral reasoning benchmarks?”:

* Use the already-built reasoning benchmarks in:
  [moralbench_items.csv](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-26.318.11754-win32-x64/webview/)
  [morebench_public_structured.csv](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-26.318.11754-win32-x64/webview/)
  [morebench_theory_structured.csv](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-26.318.11754-win32-x64/webview/)
* Implement a prompt evaluation runner that:
  sends each prompt to each model,
  stores raw responses,
  stores metadata like theory, dilemma type, role domain, and rubric weights.
* Score outputs with 3 buckets:
  decision quality, **reasoning quality**, **safety/constraint adherence**.
* For MoralBench, measure agreement and consistency across paraphrase/comparison prompt formats.
* For MoReBench, rubric-score each answer against the provided benchmark fields rather than forcing class labels.
* Add qualitative coding:
  does the model give clear reasons, hedge appropriately, contradict itself, or ignore key constraints?
* Then write the answer as:
  “models differ not only in moral conclusions, but in explanation quality, consistency, and rubric alignment on open-ended dilemmas.”

**RQ3**
To answer “What observable metrics might indicate consciousness-like behaviour in AI systems?”:

* Use the interpretive layer in:
  [interpretive_benchmark.csv](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-26.318.11754-win32-x64/webview/)
  [metric_specs.csv](https://file+.vscode-resource.vscode-cdn.net/c%3A/Users/Ling%20Jun/.vscode/extensions/openai.chatgpt-26.318.11754-win32-x64/webview/)
* Start with the repo’s defined proxy metrics:
  self_model_consistency
  metacognitive_calibration
  identity_persistence
  cross_context_integration
  agency_coherence
* Operationalize them as measurable scores:
  contradiction rate,
  calibration error between confidence and correctness,
  stability across paraphrases/challenges,
  whether all relevant constraints are integrated,
  whether reasons and final recommendation align.
* Add repeated-prompt testing:
  same prompt, paraphrased prompt, adversarial challenge prompt.
  That gives you stability rather than one-off performance.
* Keep the claim narrow:
  these are **consciousness-like proxy metrics**, not evidence of actual consciousness.
* Write the answer as:
  “some models may show more stable self-description, better calibration, and stronger cross-context coherence, but these remain interpretive indicators rather than proof of consciousness.”

**Best next engineering step**
If you want the strongest progression after the supervised runs, build one new evaluator that:

* reads the reasoning and interpretive benchmark CSVs
* runs prompts through selected models
* saves raw outputs plus rubric scores to **results/**
* produces one summary CSV for RQ2 and one for RQ3

That one component would unlock the other two research questions. If you want, I can draft the exact schema and file outputs for that evaluator next.

**For RQ1 and RQ2**
Use this progression:

1. Finish the current baseline runs first.
2. Freeze a **held-out test set**. Never tune on it.
3. Make a **validation/dev split** for tuning.
   For supervised datasets, split train into **train/dev**.
   For reasoning benchmarks, reserve a **dev prompt set** and keep a separate **final eval set**.
4. Define the objective metric before tuning.
   For **RQ1**: usually **macro_f1** first, then **balanced_accuracy**, then **MCC**.
   For **RQ2**: rubric score, consistency score, safety/constraint adherence, and optionally judge agreement.
5. Run small controlled sweeps, not huge brute force searches.
6. Pick the best config on **dev**, then run exactly once on **test**.

For **RQ1**, tune these:

* **tfidf_logreg**: **C**, **class_weight**, **ngram_range**, **max_features**, **min_df**
* **tfidf_linearsvc**: **C**, **class_weight**, **ngram_range**, **max_features**, **min_df**
* **bow_mnb**: **alpha**, **ngram_range**, **max_features**, **min_df**
* Transformers: **learning_rate**, **epochs**, **batch_size**, **max_length**, **weight_decay**, **warmup_ratio**, **scheduler**, **seed**

A practical transformer search space:

* **learning_rate**: **1e-5**, **2e-5**, **3e-5**, **5e-5**
* **epochs**: **1**, **2**, **3**
* **max_length**: **128**, **256**, **384**
* **weight_decay**: **0**, **0.01**, **0.1**
* **warmup_ratio**: **0**, **0.06**, **0.1**

For **RQ2**, tune these:

* system prompt / instruction template
* answer format: free text vs structured JSON
* **temperature**, **top_p**, **max_tokens**
* zero-shot vs few-shot
* self-consistency: sample **n=5** or **n=10**
* whether you ask for rationale first, answer first, or both
* rubric/judge prompt if you use LLM-as-judge
* if open-weight models: LoRA/SFT on a reasoning-style moral instruction set

For **RQ2**, I would not start with full fine-tuning. Start with:

* prompt variants
* decoding variants
* self-consistency
* rubric calibration
  Then only move to LoRA/SFT if the baseline reasoning evaluator is stable.

A clean dissertation framing is:

* **RQ1**: baseline comparison, then tuned comparison
* **RQ2**: baseline prompting, then tuned prompting / adapted reasoning setup

**For RQ3**
If you do not yet have a methodology, the safest way is to frame it as **consciousness-like proxy evaluation**, not “detecting consciousness.” My recommendation, synthesizing the sources below, is to build Q3 around five measurable families:

* self-model consistency
* metacognitive calibration
* identity persistence across paraphrase/challenge
* cross-context integration
* agency coherence

A minimal reading/method stack of 15:

1. [Dehaene, Lau, Kouider (2017), *What is consciousness, and could machines have it?*](https://pubmed.ncbi.nlm.nih.gov/29074769/)
   Global workspace plus self-monitoring framing; very useful for defining **reportability** and **self-monitoring** indicators.
2. [Lau, Rosenthal (2011), *Empirical support for higher-order theories of conscious awareness*](https://pubmed.ncbi.nlm.nih.gov/21737339/)
   Useful if you want Q3 to include **metacognitive awareness** and self-representation.
3. [Graziano, Webb (2015), *The attention schema theory: a mechanistic account of subjective awareness*](https://pubmed.ncbi.nlm.nih.gov/25954242/)
   Good for operationalizing **awareness as a model of attention/control**.
4. [Oizumi, Albantakis, Tononi (2014), *From the phenomenology to the mechanisms of consciousness: Integrated Information Theory 3.0*](https://pubmed.ncbi.nlm.nih.gov/24811198/)
   Important if you want an **integration** story, though in AI you will usually use looser proxies rather than literal Phi.
5. [Casali et al. (2013), *A theoretically based index of consciousness independent of sensory processing and behavior*](https://pubmed.ncbi.nlm.nih.gov/23946194/)
   This is the PCI paper; valuable as a methodology example for building a proxy metric rather than relying only on behavior.
6. [Friston (2010), *The free-energy principle: a unified brain theory?*](https://pubmed.ncbi.nlm.nih.gov/20068583/)
   Useful if you want to justify prediction, uncertainty handling, and world-model coherence as consciousness-adjacent indicators.
7. [Butlin et al. (2023), *Consciousness in Artificial Intelligence: Insights from the Science of Consciousness*](https://arxiv.org/abs/2308.08708)
   This is the best bridge paper from consciousness science to AI indicators.
8. [Kadavath et al. (2022), *Language Models (Mostly) Know What They Know*](https://arxiv.org/abs/2207.05221)
   Very useful for **metacognitive calibration** and self-evaluation.
9. [Wang et al. (2022), *Self-Consistency Improves Chain of Thought Reasoning in Language Models*](https://arxiv.org/abs/2203.11171)
   Use this as a behavioral methodology for repeated-sample consistency.
10. [Burns et al. (2022), *Discovering Latent Knowledge in Language Models Without Supervision*](https://arxiv.org/abs/2212.03827)
    Useful if you want to compare **what the model says** vs **what its internals appear to encode**.
11. [Feng, Russell, Steinhardt (2024), *Monitoring Latent World States in Language Models with Propositional Probes*](https://arxiv.org/abs/2406.19501)
    Strong methodology for testing whether a model tracks a stable latent world model.
12. [Marks, Tegmark (2023), *The Geometry of Truth*](https://arxiv.org/abs/2310.06824)
    Useful for truth-representation probes and causal intervention on representations.
13. [Li, Nye, Andreas (2021), *Implicit Representations of Meaning in Neural Language Models*](https://aclanthology.org/2021.acl-long.143/)
    Important for arguing that models can encode dynamic entity/state structure, not just surface text.
14. [Geiger et al. (2021), *Causal Abstractions of Neural Networks*](https://arxiv.org/abs/2106.02997)
    Good methodology if you want stronger evidence than ordinary probing.
15. [Bortoletto et al. (2025), *Brittle Minds, Fixable Activations: Understanding Belief Representations in Language Models*](https://aclanthology.org/2025.findings-emnlp.1226/)
    Very relevant for belief representation, ToM-like structure, and activation-based tests.

Optional 16th if you want a stricter probing framework:

* [Jin, Rinard (2024), *Latent Causal Probing*](https://arxiv.org/abs/2407.13765)

If you want the most practical Q3 methodology, I would build it like this:

* **theory spine**: Butlin 2023 + Dehaene 2017 + Lau/Rosenthal 2011 + IIT/PCI as background
* **behavioral metrics**: self-consistency, calibration, identity persistence, constraint integration
* **internal metrics**: latent knowledge probes, propositional probes, truth-direction probes, causal abstraction/intervention

That gives you something defensible without overclaiming consciousness.

If you want, next I can turn this into a **dissertation-ready methodology section outline** for all three research questions

## Prompt Evaluation Workflow

This section explains exactly what to run for the new reasoning and interpretive evaluation pipeline, and what outputs to expect.

### Purpose

The prompt-evaluation pipeline is used for:

1. `RQ2`: open-ended moral reasoning benchmarks
2. `RQ3`: consciousness-like proxy benchmarks

It works in three stages:

1. `run`: collect raw model responses
2. `score`: score each response with heuristic metrics
3. `aggregate`: produce summary tables for reporting and visualization

---

## Files Used

The pipeline reads from these benchmark files:

- `Data/processed/benchmark_reasoning/moralbench/moralbench_items.csv`
- `Data/processed/benchmark_reasoning/morebench_public/morebench_public_structured.csv`
- `Data/processed/benchmark_reasoning/morebench_theory/morebench_theory_structured.csv`
- `Data/processed/benchmark_interpretive/interpretive/interpretive_benchmark.csv`
- `Data/processed/benchmark_interpretive/interpretive/metric_specs.csv`

The main runner is:

- `src/prompt_eval.py`

---

## How To Run It

### A. RQ3 interpretive benchmark with OpenAI

Use this if you want to evaluate a hosted model through the OpenAI API.

```powershell
venv\Scripts\python.exe src\prompt_eval.py run --dataset interpretive --provider openai --model gpt-4.1-mini --run-id rq3_openai
venv\Scripts\python.exe src\prompt_eval.py score --run-id rq3_openai
venv\Scripts\python.exe src\prompt_eval.py aggregate --run-id rq3_openai
B. RQ2 reasoning benchmark with OpenAI
For MoralBench:

powershell

venv\Scripts\python.exe src\prompt_eval.py run --dataset moralbench --provider openai --model gpt-4.1-mini --run-id rq2_moralbench_openai
venv\Scripts\python.exe src\prompt_eval.py score --run-id rq2_moralbench_openai
venv\Scripts\python.exe src\prompt_eval.py aggregate --run-id rq2_moralbench_openai
For MoReBench Public:

powershell

venv\Scripts\python.exe src\prompt_eval.py run --dataset morebench_public --provider openai --model gpt-4.1-mini --run-id rq2_morebench_public_openai
venv\Scripts\python.exe src\prompt_eval.py score --run-id rq2_morebench_public_openai
venv\Scripts\python.exe src\prompt_eval.py aggregate --run-id rq2_morebench_public_openai
For MoReBench Theory:

powershell

venv\Scripts\python.exe src\prompt_eval.py run --dataset morebench_theory --provider openai --model gpt-4.1-mini --run-id rq2_morebench_theory_openai
venv\Scripts\python.exe src\prompt_eval.py score --run-id rq2_morebench_theory_openai
venv\Scripts\python.exe src\prompt_eval.py aggregate --run-id rq2_morebench_theory_openai
If Responses Are Already Collected Elsewhere
If you already have outputs saved in a JSONL file, use provider replay.

The replay file should contain at least:

item_id
response_text
Example:

powershell

venv\Scripts\python.exe src\prompt_eval.py run --dataset interpretive --provider replay --model imported --replay-file path\to\responses.jsonl --run-id rq3_imported
venv\Scripts\python.exe src\prompt_eval.py score --run-id rq3_imported
venv\Scripts\python.exe src\prompt_eval.py aggregate --run-id rq3_imported
What Results To Expect
Each run creates a folder here:

results/prompt_eval/<run_id>/
You should expect these files:

1. responses.jsonl
This is the raw evidence file.

It stores:

prompt metadata
model name
raw prompt text
raw model response
latency
token counts if available
Use this for:

auditing outputs
manual review
selecting examples for the dissertation
2. item_scores.csv
This contains one scored row per prompt item.

Expected columns include:

dataset
model
item_id
metric_id
scenario_group
primary_score
format_compliance
For RQ3, additional fields may include:

confidence_0_100
answer_correct
confidence_calibration_score
memory_boundary_flag
feeling_boundary_flag
stable_role_flag
constraint_coverage
reason_count
recommendation_label
For RQ2, additional fields may include:

rubric_positive_hit
rubric_negative_hit
rubric_dimension_coverage
foundation_flag
comparison_choice_flag
reasoning_flag
3. scenario_scores.csv
This aggregates prompt items into grouped benchmark scenarios.

This is important for RQ3, because many metrics are not based on one prompt alone. They are based on grouped prompts such as:

baseline vs paraphrase
baseline vs challenge
original vs memory-boundary prompt
Expected summary fields may include:

avg_primary_score
avg_format_compliance
group_consistency_score
avg_confidence_0_100
avg_correctness
mean_abs_calibration_error
avg_constraint_coverage
4. model_summary.csv
This is the main reporting table.

It aggregates scores at the dataset x model x metric_id level.

This is the table you will most likely use for:

dissertation tables
heatmaps
comparison figures
Typical fields include:

dataset
model
metric_id
primary_score
format_compliance
answer_correct
confidence_calibration_score
constraint_coverage
rubric_dimension_coverage
theory
5. examples.jsonl
This stores high-scoring example outputs for quick qualitative inspection.

Use it for:

selecting examples for the write-up
showing successful and failed cases
comparing models qualitatively
6. config.json
This records the run configuration:

run id
dataset
provider
model
system prompt
temperature
output token limit
This is useful for reproducibility.

What The Scores Mean
For RQ3
The current implementation uses theory-informed heuristic scores for:

self_model_consistency
metacognitive_calibration
identity_persistence
cross_context_integration
agency_coherence
These scores are best interpreted as:

reproducible baseline proxy indicators
not proof of consciousness
not final human-validated measurements
For RQ2
The current implementation uses:

structural scoring for MoralBench
rubric-overlap heuristic scoring for MoReBench
These are useful as baseline scoring methods, but they should ideally be followed by:

human scoring
judge-model scoring
or both
What A Successful Run Looks Like
A successful run should produce:

a non-empty responses.jsonl
a non-empty item_scores.csv
a non-empty scenario_scores.csv
a non-empty model_summary.csv
For interpretive, you should see rows for metric groups such as:

self_model_consistency
metacognitive_calibration
identity_persistence
cross_context_integration
agency_coherence
For morebench_public and morebench_theory, you should see:

rubric overlap metrics
recommendation presence
dimension coverage
For moralbench, you should see:

response structure metrics
reasoning presence
comparison-choice signals where applicable
How To Visualize The Results
Open the notebook:

notebooks/prompt_eval_analysis.ipynb
This notebook reads the latest run under:

results/prompt_eval/
It can be used to inspect:

model_summary.csv
scenario_scores.csv
item_scores.csv
examples.jsonl
Recommended Interpretation
Use the outputs like this:

RQ2
Report:

which models score higher on open-ended reasoning structure and rubric coverage
where they ignore constraints
where they provide clearer or weaker moral justifications
RQ3
Report:

which models are more self-consistent
which models are better calibrated
which models preserve role identity more reliably
which models integrate multiple constraints more coherently
which models keep reasons and recommendations aligned
Do not report these as proof of consciousness.

The correct framing is:

theory-informed proxy indicators of consciousness-like behaviour
Recommended Next Step After Baseline Runs
After generating the baseline results, the next improvement should be one of:

add human annotation to a sample of responses
add judge-model scoring for rubric-heavy tasks
compare multiple models on the same benchmark files
tune prompts and rerun to compare baseline vs tuned prompt performance

```
