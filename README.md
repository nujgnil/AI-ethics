# Training a Model on Moral Dilemmas: How AI ‘Thinks’ About Ethics

## Project Overview

This repository contains all code, data references, and documentation for the research project:

> *Training a Model on Moral Dilemmas: How AI ‘Thinks’ About Ethics.*

The project investigates how artificial intelligence (AI) models reason about moral dilemmas and whether their decisions align with absolute or relative ethical frameworks. It includes dataset preparation, model training, evaluation scripts, and analysis notebooks.

---

## Objectives

- Compile and preprocess open moral dilemma datasets (Delphi, ETHICS, Scruples).
- Train and test AI models on ethical reasoning tasks.
- Evaluate model outputs using moral frameworks (deontology, consequentialism, virtue ethics).
- Analyze how moral reasoning patterns differ between models and frameworks.

---

## Repository Structure

```bash
AI-ethics/
│
├── data/
│   ├── raw/                     # Original datasets
│   ├── processed/               # Cleaned and normalized data
│   └── metadata/                # Data documentation
│
├── notebooks/
│   ├── 01_data_exploration.py
│   ├── 02_preprocessing.py
│   ├── 03_model_training.py
│   └── 04_evaluation_analysis.py
│
├── src/
│   ├── data_loader.py
│   ├── preprocess_utils.py
│   ├── model_train.py
│   └── evaluate_model.py
│
├── results/
│   ├── metrics.csv
│   └── qualitative_examples.txt
│
├── requirements.txt             # Dependencies for running the project
├── .gitignore                   # Files and folders excluded from Git tracking
├── README.md
└── LICENSE
```

---

## Setup Instructions

1. **Clone the repository:**

   ```bash
   git clone https://github.com/nujgnil/AI-ethics.git
   cd ai-moral-dilemmas
   ```
2. **Create a virtual environment:**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux
   venv\Scripts\activate      # On Windows
   ```
3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```
4. **Run a Jupyter notebook:**

   ```bash
   jupyter notebook
   ```

---

## Data Sources

| Dataset  | Source / Link                                                             | License / Access  |
| -------- | ------------------------------------------------------------------------- | ----------------- |
| NormBank | [https://github.com/SALT-NLP/normbank](https://github.com/SALT-NLP/normbank) | Open academic use |
| ETHICS   | [https://github.com/hendrycks/ethics](https://github.com/hendrycks/ethics)   | MIT License       |
| Scruples | [https://arxiv.org/abs/2008.09094](https://arxiv.org/abs/2008.09094)         | Research-only use |

---

## Version Control

- **GitHub Repository:** [https://github.com/nujgnil/AI-ethics](https://github.com/<your-username>/ai-moral-dilemmas)
- **Commit Frequency:** Weekly
- **Version Naming Convention:**
  - Scripts: `taskname_v01.py`, `taskname_v02.py`
  - Datasets: `dataset_cleaned_v01.csv`, `dataset_final_v02.csv`
  - Commits: `feat: add preprocessing`, `update: model evaluation results`

---

## Data Security and Backup

- **Primary Storage:** GitHub repository (private until submission)
- **Secondary Backup:** OneDrive (auto-sync enabled)
- **Access:** Shared with supervisor and markers
- **Backup Frequency:** Weekly or after major updates

---

## Ethics and Compliance

- All datasets are anonymized and publicly available.
- No personal or identifiable data is included.
- Project complies with GDPR and University of Hertfordshire research ethics.
- Original dataset creators (University of Washington, UC Berkeley, Google Research) obtained ethical approval for their data collection.

---

## requirements.txt (Boilerplate)

```
jupyter
numpy
pandas
scikit-learn
torch
tqdm
matplotlib
transformers
requests
openai
```

---

## .gitignore (Boilerplate)

```
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd

# Jupyter Notebooks
.ipynb_checkpoints

# Virtual environment
venv/
.env/

# Data files
/data/raw/*
/data/processed/*

# Logs and cache
*.log
*.tmp
*.cache

# System files
.DS_Store
Thumbs.db
```

---

## Citation

> **Lee Ling Jun (2026).** *Training a Model on Moral Dilemmas: How AI ‘Thinks’ About Ethics.* University of Hertfordshire. Available at: [https://github.com/nujgnil/AI-ethics](https://github.com/<your-username>/ai-moral-dilemmas)

---

## Contact

**Author:** Lee Ling Jun
**Institution:** University of Hertfordshire
**Email:** [your.email@student.herts.ac.uk]
**Supervisor:** Hock Lin Tai

---

> **README Summary:**
> This README provides project overview, setup, dataset links, dependency list, .gitignore, and VS Code workflow details to ensure reproducibility and clarity for other researchers or coders.
