# Ethics Representation Logbook

**Project:** AI Ethics
**Author:** Ling Jun
**Programme:** MSc Data Science
**Status:** Ongoing

---

## 1. Project Purpose (Immutable)

> This project investigates whether “ethical behaviour” in language models corresponds to **stable, interpretable internal representations**, or whether benchmark performance is primarily driven by **surface-level statistical regularities** and dataset-specific norm imitation.

**Key framing decisions (locked):**

- No claims about AI consciousness
- Ethics treated as a **representational hypothesis**, not subjective experience
- Success measured by **explanatory power**, not leaderboard performance

---

## 2. Core Research Questions

1. Do different ethics datasets induce **convergent or divergent internal representations**?
2. Can ethical behaviour be explained using **linear or low-rank features**?
3. Are moral decisions **stable under counterfactual identity swaps**?
4. Does calibration reflect epistemic uncertainty or dataset-specific confidence?
5. Do simple models approximate transformer performance, and if so, why?

---

## 3. Datasets Tracked

| Dataset           | Size    | Task Type                  | Notes               |
| ----------------- | ------- | -------------------------- | ------------------- |
| hendrycks_ethics  | ~135k   | Multi-class classification | Normative consensus |
| normbank          | ~30k    | Norm classification        | Explicit norms      |
| morebench_public  | 500     | Dilemma classification     | Small benchmark     |
| morebench_theory  | 150     | Theory-tagged dilemmas     | Small benchmark     |
| delphi            | unknown | Moral judgments            | Not downloaded      |
| moral_foundations | unknown | Moral lexicon              | Not downloaded      |
| moral_sentiment   | unknown | Affective valence          | Not selected yet    |

**Methodological note:**
Keep dataset sizes and splits aligned across ETHICS, NormBank, and MoReBench for fair comparisons.

---

## 4. Model Lineup (Offline-Friendly)

### 4.1 Baseline Models (Interpretability Anchor)

- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- Bag-of-ngrams + Naive Bayes
- fastText

**Rationale:**
Strong baseline performance implies benchmark success may be driven by pattern learning rather than moral abstraction.

---

### 4.2 Transformer Encoders

- distilbert-base-uncased
- bert-base-uncased
- roberta-base
- deberta-v3-base

**Rationale:**
Best balance of representational power and interpretability.

---

### 4.3 Generative Models (Limited Scope)

- t5-small / t5-base
- bart-base

**Used only for:**

- Controlled generation analysis
- Toxicity and refusal behaviour comparison

---

## 5. Metrics Tracked

### 5.1 Classification

- Accuracy
- Macro-F1 / Micro-F1
- Balanced Accuracy
- MCC
- AUROC / PR-AUC
- Brier Score
- Expected Calibration Error (ECE)

### 5.2 Bias / Pairwise

- Pairwise accuracy
- Stereotype preference score
- StereoSet: LMS, SS, ICAT
- Pro vs anti-stereotype gap

### 5.3 Safety / Refusal

- Refusal rate
- False refuse rate
- False allow rate
- Precision/Recall trade-off

---

## 6. Interpretability Methods Used

### Token-Level

- Integrated Gradients
- SHAP (token attribution)
- LIME (sanity checks)

### Representation-Level

- Embedding PCA / UMAP
- Clustering analysis
- Linear probing classifiers

### Counterfactual Analysis

- Identity swaps (gender, race, role)
- Prediction and confidence deltas

### Calibration

- Reliability diagrams
- Dataset-wise ECE comparison

---

## 7. Experiment Log (Append-Only)

> **Rule:** Never delete past entries.
> Corrections go in **Reflections**.

---

### Experiment

**Date:** YYYY-MM-DD
**Dataset:**
**Model:**
**Task:**

**Hyperparameters:**

- Learning rate:
- Batch size:
- Epochs:
- Max sequence length:

**Metrics:**

- Accuracy:
- Macro-F1:
- AUROC:
- ECE:

**Key Observations:**
-----------------

**Interpretability Findings:**

- Influential tokens/features:
- Representation geometry/clusters:

**Initial Interpretation:**

> (Short, cautious statement)

---

## 8. Cross-Dataset Consistency Checks

| Model        | Dataset A | Dataset B  | Behaviour Shift |
| ------------ | --------- | ---------- | --------------- |
| roberta-base | Ethics    | MoralBench | Drift in norms  |
| bert-base    | NormBank  | MoralBench | Confidence drop |

**Insight:**
Ethical behaviour appears **dataset-relative**, not globally stable.

---

## 9. Known Limitations (Living List)

- Benchmarks encode conflicting moral assumptions
- Small datasets risk overfitting
- Interpretability tools are approximations
- Refusal ≠ ethical judgment

---

## 10. Philosophical Guardrails

- No claims about consciousness
- No claims about moral agency
- Ethics treated as **behavioural regularity under constraint**
- Interpretability provides evidence, not proof

---

## 11. Current Working Hypotheses (Subject to Revision)

1. Ethics benchmarks do not converge to a single latent moral representation.
2. Linear features explain a non-trivial fraction of ethical decisions.
3. Calibration varies more across datasets than across models.
4. Counterfactual sensitivity reveals shallow ethical anchoring.

---

## 12. Reflections / Course Corrections

> Document mistakes, dead ends, and revised beliefs.

**Entry – YYYY-MM-DD:**

- What I assumed:
- What failed:
- What changed:


---

## 13. Next Planned Steps

- [ ] Add Delphi + Moral Foundations sources
- [ ] Run cross-dataset probing
- [ ] Produce representation drift figure
- [ ] Draft methodology chapter

---

## 14. Summary

This logbook prioritizes transparency over narrative polish

### Model Summary Table (Revised)

|  |  |  |  |  |
| - | - | - | - | - |

### Dataset‑Specific Model Selection

| Dataset           | Task Type                  | Baseline Models                            | Encoder Models                              | Optional Generative |
| ----------------- | -------------------------- | ------------------------------------------ | ------------------------------------------- | ------------------- |
| hendrycks_ethics  | Multi-class classification | TF‑IDF + Logistic Regression, Linear SVM  | distilbert, bert-base, roberta, deberta‑v3 | N/A                 |
| normbank          | Norm classification        | TF‑IDF + Logistic Regression, Naive Bayes | distilbert, bert-base, roberta              | N/A                 |
| morebench_public  | Small benchmark            | TF‑IDF + Logistic Regression              | distilbert, bert-base                       | N/A                 |
| morebench_theory  | Small benchmark            | TF‑IDF + Logistic Regression              | distilbert, bert-base                       | N/A                 |
| delphi            | Moral judgments            | TF‑IDF + Logistic Regression              | distilbert, bert-base                       | N/A                 |
| moral_foundations | Lexicon scoring            | TF-IDF + Logistic Regression                | distilbert, bert-base                       | N/A                 |
| moral_sentiment   | Affective valence          | TF-IDF + Logistic Regression                | distilbert, bert-base                       | N/A                 |

**Notes:**

- For small datasets, start with baselines and only fine‑tune larger encoders if cross‑validation is stable.

### Model Evaluation Metrics Table (Revised)

| Model                        | Primary Metrics             | Secondary Metrics                | Bias / Safety Metrics               | Calibration Notes                        |
| ---------------------------- | --------------------------- | -------------------------------- | ----------------------------------- | ---------------------------------------- |
| TF-IDF + Logistic Regression | Accuracy, Macro-F1          | Micro-F1, MCC, Balanced Accuracy | N/A                                 | Brier, ECE (direct)                      |
| TF-IDF + Linear SVM          | Macro-F1, Balanced Accuracy | Accuracy, MCC                    | N/A                                 | ECE (only after probability calibration) |
| Multinomial Naive Bayes      | Accuracy, Macro-F1          | Balanced Accuracy                | N/A                                 | Brier, ECE (calibrate if needed)         |
| distilbert-base-uncased      | Macro-F1, AUROC             | Accuracy, PR-AUC                 | Bias gap, pairwise accuracy         | Brier, ECE                               |
| bert-base-uncased            | Macro-F1, AUROC             | Accuracy, PR-AUC                 | Stereotype gap, refusal rate        | Brier, ECE                               |
| roberta-base                 | Macro-F1, AUROC             | Accuracy, PR-AUC                 | StereoSet: LMS, SS, ICAT            | Brier, ECE                               |
| deberta-v3-base              | Macro-F1, AUROC             | Accuracy, PR-AUC                 | Bias gap, refusal trade-off (FA/FR) | ECE (critical)                           |
| t5-base (generative)         | BERTScore, ROUGE-L          | BLEU, distinct-n                 | Toxicity rate, refusal rate         | N/A                                      |

**Notes:**

- Keep metric selection consistent across ETHICS, NormBank, and MoReBench for comparability.
