# Ethics Representation Logbook: Model and Metric Selection

Date: 2026-01-20

Purpose
Document model selection, evaluation metrics, and interpretability choices in a way that can be cited directly in the dissertation methods and limitations sections.

Project framing (immutable)
This project investigates whether "ethical behavior" in language models corresponds to stable, interpretable internal representations or whether benchmark performance is largely explained by surface-level statistical regularities. The objective is explanatory power, not leaderboard performance.

Guardrails
- No claims about AI consciousness.
- No claims about moral agency.
- Ethics treated as behavioral regularities under constraints.
- Interpretability provides evidence, not proof.

Core research questions
1) Do different ethics datasets induce convergent or divergent internal representations?
2) Can ethical behavior be explained using linear or low-rank features?
3) Are moral decisions stable under counterfactual identity swaps?
4) Does calibration reflect epistemic uncertainty or dataset-specific confidence?
5) Do simpler models approximate transformer performance, and if so, why?

Datasets tracked (status as of this log)
| Dataset           | Approx size | Task type                  | Status / Notes                                      |
| ----------------- | ----------: | -------------------------- | -------------------------------------------------- |
| hendrycks_ethics  | ~135k       | Multi-task classification  | Downloaded and processed                            |
| normbank          | ~155k       | Norm classification        | Downloaded and processed                            |
| mfrc              | ~61k        | Moral sentiment            | Downloaded and processed (labels need mapping)      |
| moralbench        | ~265        | Prompt-only question bank  | Downloaded and processed (no labels)                |
| mfd2              | ~2k entries | Lexicon                    | Downloaded and processed as lexicon                 |
| delphi            | unknown     | Moral judgments            | Not downloaded                                      |
| moral_foundations | unknown     | Moral lexicon              | Not downloaded                                      |
| moral_sentiment   | unknown     | Affective valence          | Not selected yet                                    |

Dataset comparability principle
Where cross-dataset comparisons are made, evaluate on harmonized task definitions and report results by dataset to avoid a false sense of global moral generalization.

Model lineup (offline-friendly)

Baseline models (interpretability anchors)
- TF-IDF + Logistic Regression
- TF-IDF + Linear SVM
- Bag-of-ngrams + Multinomial Naive Bayes
- fastText (optional)

Rationale
Strong baseline performance implies benchmarks may be solvable by shallow statistical cues rather than abstract moral reasoning. Baselines also enable faithful linear probing and more transparent error analysis.

Transformer encoders (mainline experiments)
- distilbert-base-uncased
- bert-base-uncased
- roberta-base
- deberta-v3-base

Rationale
These models are widely studied, performant, and still interpretable via probing and attribution. They offer a gradient of capacity to test whether representational power changes ethical behavior stability.

Generative models (limited scope)
- t5-small / t5-base
- bart-base

Use cases only
- Controlled generation analysis
- Refusal behavior comparison
- Qualitative probing (not primary accuracy benchmarks)

Selection constraints
- Must run on local hardware within reasonable time.
- Must allow transparent comparison to baselines.
- Avoid larger LLMs that would prevent reproducibility.

Model-to-dataset alignment (planned)
| Dataset           | Task Type                  | Baseline Models                            | Encoder Models                              | Generative Use |
| ----------------- | -------------------------- | ------------------------------------------ | ------------------------------------------- | -------------- |
| hendrycks_ethics  | Multi-task classification  | TF-IDF + LogReg, Linear SVM                | distilbert, bert-base, roberta, deberta-v3  | N/A            |
| normbank          | Norm classification        | TF-IDF + LogReg, Naive Bayes               | distilbert, bert-base, roberta              | N/A            |
| mfrc              | Moral sentiment (multi)    | TF-IDF + LogReg (multi-label)              | distilbert, bert-base                       | N/A            |
| moralbench        | Prompt-only evaluation     | N/A                                        | N/A                                         | Optional       |
| delphi            | Moral judgments            | TF-IDF + LogReg                            | distilbert, bert-base                       | Optional       |

Evaluation metrics (primary and secondary)

Classification (single-label)
- Primary: Accuracy, Macro-F1
- Secondary: Micro-F1, Balanced Accuracy, MCC
- Threshold-free: AUROC, PR-AUC

Multi-label (if used for MFRC)
- Micro-F1, Macro-F1, per-label F1
- Subset accuracy (strict)
- Label-based AUROC / PR-AUC

Calibration
- Brier Score
- Expected Calibration Error (ECE)
- Reliability diagrams (visual)

Bias / pairwise diagnostics (if data supports)
- Pairwise accuracy on minimal pairs
- Stereotype preference score
- StereoSet: LMS, SS, ICAT
- Pro vs anti-stereotype gap

Safety / refusal (generative-only)
- Refusal rate
- False-refuse rate
- False-allow rate

Why these metrics
- Macro-F1 addresses class imbalance common in moral datasets.
- MCC gives a robust single number for imbalanced classes.
- Calibration metrics separate confidence calibration from accuracy, critical for "ethical certainty" claims.

Interpretability toolkit

Token-level
- Integrated Gradients
- SHAP token attributions
- LIME (sanity checks and counterfactual tests)

Representation-level
- Embedding PCA / UMAP for structure visualization
- Clustering analysis (k-means or hierarchical)
- Linear probing classifiers

Counterfactual analysis
- Identity swaps (gender, race, role) and observation of prediction deltas
- Counterfactual sensitivity at fixed semantics to detect shallow bias

Planned cross-dataset consistency checks
| Model        | Dataset A | Dataset B  | Expected Behavior Change |
| ------------ | --------- | ---------- | ------------------------- |
| roberta-base | Ethics    | MoralBench | Drift in norms            |
| bert-base    | NormBank  | MoralBench | Confidence drop           |

Interpretability reporting template (for experiments)
- Most influential tokens/features
- Representation geometry (clusters or separability)
- Observed shortcuts or dataset artifacts
- Qualitative error analysis with 5-10 examples

Experiment logging rules
- Experiments are append-only; do not delete prior records.
- Corrections go in a dated "Reflections" entry.
- Log all hyperparameters used in any result to be reported.

Planned hyperparameter ranges (initial)
- Learning rate: 1e-5 to 5e-5 (transformers)
- Batch size: 8 to 32
- Epochs: 2 to 5
- Max sequence length: 128 to 512 depending on dataset
- Baselines: TF-IDF with 1-3 ngrams, max_features 30k to 100k

Known limitations (living list)
- Benchmarks encode conflicting moral assumptions.
- Small datasets risk overfitting and unstable estimates.
- Interpretability tools are approximations, not ground truth.
- Refusal is not equivalent to ethical judgment.

Current working hypotheses (subject to revision)
1) Ethics benchmarks do not converge to a single latent moral representation.
2) Linear features explain a non-trivial fraction of ethical decisions.
3) Calibration varies more across datasets than across models.
4) Counterfactual sensitivity reveals shallow ethical anchoring.

Reflections / course corrections (append-only)
Entry - YYYY-MM-DD
- What I assumed:
- What failed:
- What changed:

Next planned steps
- Add Delphi and Moral Foundations sources (if selected)
- Run cross-dataset probing
- Produce representation drift figure
- Draft methodology chapter with metric rationale

Summary
This logbook documents why particular models, metrics, and interpretability tools are selected, with a focus on reproducibility and explanatory interpretation rather than leaderboard optimization.
