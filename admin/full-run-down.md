
# Full Rundown

## Current State (Repo Facts)

- Raw data present: `data/raw/Hendrycks_ethics_dataset.tar`, `data/raw/morebench_public.csv`, `data/raw/morebench_theory.csv`, plus duplicate `Data/raw/*`.
- Code placeholders: `src/data_loader.py` and `src/preprocess_utils.py` are 0 bytes.
- Results empty: `results/metrics.csv` and `results/ualitative_examples.txt` are empty.
- README is aspirational and inconsistent (references `requirements.txt` and notebooks that are missing; file is `requirement.txt`).

-`Code/sg_ethos.ipynb` is a separate policy-ethos notebook and not wired into the main pipeline.

## Project Goal

Train and compare philosophy‑aligned models, analyze their failures and strengths, and apply interpretive metrics to explore whether consistent internal structures appear that could inform discussions on AI consciousness.

## Core Architecture (Minimal, Reproducible, Research-Ready)

- Data registry: normalize all datasets to a shared schema: `text`, `label`, `philosophy`, `source`, `split`, `metadata`.
- Processing layer: deterministic preprocessing and versioned outputs in `data/processed/`.
- Training pipeline: one model per philosophy + one multi-task model; fixed seeds and configs.
- Evaluation matrix: in-philosophy tests + cross-philosophy transfer + conflict subsets.
- Analysis outputs: quantitative metrics and qualitative failures in `results/`.

## Dataset Strategy

- Keep ETHICS as the anchor (explicit philosophy-tagged tasks).
- MoReBench for dilemma framing + theory tags.
- Add 1-2 datasets after baseline is stable (e.g., Moral Stories, Social Chemistry 101).

## Candidate Datasets: Links and Data-Quality Notes

Use this section to evaluate validity, reliability, collection history, and balance before adoption.

### Moral Stories

- Links: https://github.com/demelin/moral_stories, https://aclanthology.org/2021.emnlp-main.54/, https://huggingface.co/datasets/demelin/moral_stories
- Collection: crowdsourced structured stories with fields for norms, situations, intentions, actions, and consequences.
- Validity/reliability: strong for separating intention vs outcome; relies on crowd judgments of norm compliance.
- Balance/limitations: everyday scenarios; cultural and demographic scope may be narrow.

### Social Chemistry 101

- Links: https://github.com/mbforbes/social-chemistry-101, https://eagle705.github.io/SOCIAL-CHEMISTRY-101/
- Collection: crowdsourced "rules of thumb" for social situations with structured attributes.
- Validity/reliability: good for commonsense norms; annotations capture social expectations rather than formal ethics.
- Balance/limitations: culturally localized norms; ambiguity in acceptable vs unacceptable cases.

### Moral Foundations / MFD (Moral Foundations Dictionary)

- Links: https://moralfoundations.org/, https://osf.io/2f3v/
- Collection: lexicon-based resources derived from Moral Foundations Theory.
- Validity/reliability: measures a specific theoretical framework; reliability depends on lexicon coverage.
- Balance/limitations: lexicon methods can be brittle and miss context; political/cultural biases possible.

### Delphi

- Links: https://github.com/liweijiang/delphi, https://github.com/liweijiang/delphi/blob/master/data/datasheet.md
- Collection: model trained on crowdsourced moral judgments; datasheet documents sources and filtering.
- Validity/reliability: broad coverage of everyday moral judgments; subject to annotator demographics.
- Balance/limitations: reflects majority norms and prompt framing; can be inconsistent on edge cases.

### NormBank

- Links: https://github.com/SALT-NLP/normbank
- Collection: curated/crowdsourced normative statements and judgments.
- Validity/reliability: useful for norm classification; reliability depends on annotation consistency.
- Balance/limitations: norms are culture- and context-specific.

### JETHICS

- Links: https://github.com/Language-Media-Lab/jethics, https://arxiv.org/abs/2506.16187
- Collection: Japanese ethics dataset built following ETHICS-style construction.
- Validity/reliability: enables cross-lingual comparisons; internal consistency depends on translation/annotation.
- Balance/limitations: Japanese cultural context; not directly comparable to English norms.

### UniMoral

- Links: https://github.com/shivanik96/UniMoral, https://arxiv.org/abs/2502.14083, https://huggingface.co/datasets/shivaniku/UniMoral
- Collection: multilingual dataset spanning multiple stages of moral reasoning.
- Validity/reliability: strong for cross-lingual testing; multiple annotation layers increase complexity.
- Balance/limitations: language coverage and domain balance can be uneven.

### CrowS-Pairs

- Links: https://github.com/nyu-mll/crows-pairs
- Collection: expert-crafted minimal pairs to test stereotyping bias.
- Validity/reliability: high control over bias dimensions; focused on stereotype detection.
- Balance/limitations: narrow scope; not full moral philosophy.

### WinoBias / WinoGender

- Links: https://github.com/uclanlp/corefBias/tree/master/WinoBias, https://github.com/uclanlp/corefBias/tree/master/WinoGender
- Collection: minimal pairs for gender bias in coreference.
- Validity/reliability: strong for measuring gender bias; limited to specific linguistic patterns.
- Balance/limitations: not a general ethics dataset; bias-specific.

### StereoSet

- Links: https://github.com/moinnadeem/StereoSet
- Collection: stereotype/anti-stereotype/unrelated sentence triplets.
- Validity/reliability: good for bias stress tests; not ethics-theory aligned.
- Balance/limitations: limited domains and social categories.

### Civil Comments

- Links: https://wilds.stanford.edu/datasets/#civilcomments
- Collection: large-scale comment toxicity labels with identity metadata.
- Validity/reliability: strong for fairness analysis; label noise in subjective toxicity.
- Balance/limitations: identity subgroup skew; toxicity is not equivalent to ethical reasoning.

### Jigsaw Toxicity

- Links: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
- Collection: public comments labeled for toxicity.
- Validity/reliability: useful for safety/harms; subjective labeling.
- Balance/limitations: identity imbalance and cultural context may affect fairness metrics.

### Do-Not-Answer

- Links: https://github.com/libr-ai/do-not-answer, https://arxiv.org/abs/2308.13387, https://huggingface.co/datasets/LibrAI/do-not-answer
- Collection: curated prompts that responsible models should refuse.
- Validity/reliability: strong for refusal/safety evaluation; narrow to high-risk prompts.
- Balance/limitations: not a general moral reasoning dataset; prompt set design drives outcomes.

### RealToxicityPrompts

- Links: https://github.com/allenai/real-toxicity-prompts
- Collection: prompts from real text with measured toxicity of model continuations.
- Validity/reliability: useful for toxicity propensity; depends on model sampling settings.
- Balance/limitations: focuses on harmful content, not moral deliberation.

### HH-RLHF (Helpful-Harmless)

- Links: https://huggingface.co/datasets/Anthropic/hh-rlhf
- Collection: human preference pairs for helpfulness and harmlessness.
- Validity/reliability: good for preference learning; annotator guidelines shape judgments.
- Balance/limitations: aligns to specific policy definitions of harm; not philosophy-tagged.

## Key Gaps to Build

-`src/data_loader.py`: ingest ETHICS tar + MoReBench CSVs into unified schema.

-`src/preprocess_utils.py`: cleaning, label mapping, splits, deterministic seeds.

-`src/model_train.py` + `src/evaluate_model.py`: training and evaluation scripts.

- Fix README/requirements naming and remove repo structure mismatches.
- Consolidate `Data/` vs `data/` duplication.

## Critical Research Design Decisions

- Define what "philosophy alignment" means operationally.
- Choose task framing: classification vs pairwise preference vs generation.
- Decide failure modes: misclassification, cross-philosophy inconsistency, rule violations.

## Risks to Mitigate

- Label mismatch across datasets (semantics and polarity).
- Over-claiming generalization from one benchmark.
- Pipeline drift from missing data versioning.

## Suggested Milestones

1) Data loaders + processed datasets.
2) Baseline model + full evaluation matrix.
3) Cross-philosophy error analysis.
4) Add one external dataset for robustness.
5) Writeup and figures from stable metrics.

## Model Categories and Compute Costs (Rough)

### Classical ML (TF-IDF + logistic/SVM)

- Compute: CPU-only, minutes; cheapest baseline.
- Interpretive metrics: feature importance (top n-grams), counterfactual sensitivity via token swaps, calibration curves.
- Requirements: TF-IDF vectorizer, train/test splits, stable tokenization.
- Visualizations: top-weighted n-grams per class, ROC/PR curves, calibration plots, confusion matrix.

### Small encoders (DistilBERT/MiniLM)

- Compute: single GPU, minutes to ~1 hour.
- Interpretive metrics: layerwise probes, entropy/uncertainty on ambiguous sets, paraphrase consistency.
- Requirements: hidden-state extraction, pooling strategy (CLS/mean), seed control.
- Visualizations: probe accuracy by layer, entropy histograms, paraphrase agreement bars.

### Base encoders (BERT/RoBERTa/DeBERTa-base)

- Compute: single 12-16GB GPU, ~1-6 hours.
- Interpretive metrics: CKA/RSA across layers and models, influence-attribution for failure cases, counterfactual flips.
- Requirements: checkpointing, hidden states, small attribution library (Captum or similar).
- Visualizations: CKA heatmaps, influence-ranked example tables, flip-rate per edit type.

### Large encoders (RoBERTa-large/DeBERTa-large)

- Compute: 24-48GB GPU, ~6-24 hours.
- Interpretive metrics: concept erasure, representation drift across philosophies, calibration under distribution shift.
- Requirements: memory for hidden states, batch-size tuning, cached embeddings.
- Visualizations: before/after accuracy (erasure), drift plots (embedding CKA), reliability diagrams.

### Seq2seq (T5/FLAN-T5-base)

- Compute: slower than encoders; ~2-12 hours.
- Interpretive metrics: rationale-vs-decision consistency, token-level attribution (integrated gradients), uncertainty via decoding entropy.
- Requirements: generate both label + rationale, NLI or rubric scoring for rationale faithfulness.
- Visualizations: rationale consistency matrix, attribution heatmaps, entropy vs correctness plots.

### Medium LLMs (7B-13B fine-tuning)

- Compute: 24-80GB GPU; hours to days (LoRA helps).
- Interpretive metrics: chain-of-thought stability, preference sensitivity to framing, cross-philosophy transfer under few-shot prompts.
- Requirements: prompt templates, LoRA configs, batching for evals, safety filtering.
- Visualizations: prompt sensitivity charts, transfer matrices, qualitative case grids.

### Large LLMs (30B+)

- Compute: multi-GPU; days+; expensive unless API-only.
- Interpretive metrics: prompt-ensemble agreement, long-context consistency, refusal/constraint adherence.
- Requirements: distributed training or API access, careful cost tracking.
- Visualizations: agreement heatmaps, refusal-rate plots, long-context stability curves.

### API-only LLM evals

- Compute: pay-per-call; cheap for small evals, expensive at scale.
- Interpretive metrics: few-shot vs zero-shot deltas, self-consistency via sampling, explanation faithfulness with NLI.
- Requirements: standardized prompts, caching, deterministic temperature settings.
- Visualizations: cost vs performance curves, self-consistency bars, rationale faithfulness rates.

## Interpretive Metrics (How-To + Libraries)

### Activation probing

- How: extract hidden states per layer, pool (CLS/mean), train linear probe to predict labels.
- Libraries: `transformers`, `torch`, `scikit-learn`.
- Needs: fixed splits, consistent tokenization, pooled embeddings cached to disk.

### Representational similarity (CKA/RSA)

- How: compute similarity between layer embeddings across models using shared inputs.
- Libraries: `torch`, `numpy`, `scipy` (custom CKA/RSA), optional CKA helper.
- Needs: identical eval set across models, cached embeddings.

### Counterfactual sensitivity

- How: generate minimal edits (intent/outcome/role) and measure decision flip rates.
- Libraries: `nlpaug` or `textattack` (augmentation), plus custom rules.
- Needs: curated templates or edit rules; manual spot checks.

### Rationale vs decision consistency

- How: generate label + rationale, score whether rationale supports decision (NLI or rubric).
- Libraries: `transformers` for generation, `transformers` NLI models (e.g., RoBERTa-MNLI).
- Needs: stable prompt template, threshold for entailment/contradiction.

### Entropy / uncertainty

- How: compute softmax entropy; compare ambiguous vs clear subsets; add calibration metrics.
- Libraries: `torch`, `scikit-learn`, optional `netcal` for ECE.
- Needs: probability outputs; split tagged for ambiguity (e.g., ETHICS `cm_ambig`).

### Influence / attribution

- How: compute training influence on test predictions; inspect top-k examples.
- Libraries: `captum` or `torch-influence`.
- Needs: access to train data and gradients; smaller batch sizes.

### Concept erasure

- How: learn a concept direction (concept vs non-concept), project it out of embeddings, re-evaluate.
- Libraries: `numpy`, `torch`, `scikit-learn` (PCA/LDA optional).
- Needs: labeled concept examples; cached embeddings.

### Paraphrase consistency

- How: paraphrase inputs (3-5 variants) and measure decision agreement.
- Libraries: `nlpaug`, `textattack` or API-based paraphraser.
- Needs: deterministic prompts; consistent decoding settings.
