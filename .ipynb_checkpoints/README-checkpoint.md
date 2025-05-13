# 🔥 Toxic Comment Detection with Bias Audit 🔍
 
## 🌐 Project Overview

**Toxic Comment Detection with Bias Audit**  
This repository implements and evaluates multiple lightweight and advanced pipelines to detect toxic comments on the Jigsaw dataset. We compare:

- **TF-IDF (word n-grams)** + Logistic Regression  
- **TF-IDF (character n-grams)** + Logistic Regression  
- **Frozen transformer embeddings** (DistilBERT encoder) + Logistic Regression  

We also discuss ethical considerations and outline a path for a comprehensive bias audit once demographic data become available.

---

## 📁 Repository Structure

```text
Toxic_Bias_Audit/
├── data/
│   ├── raw/           # Original Jigsaw CSVs
│   └── processed/     # Cleaned & split train/val and tfidf.pkl
├── src/
│   └── preprocessing.py  # Data-loading & cleaning utilities
├── notebooks/
│   ├── exploratory/       # EDA and visualization
│   └── reproducibility/   # Step-by-step pipelines:
│       ├── 01_preprocessing.ipynb
│       ├── 02_baseline_logreg.ipynb
│       ├── 03_multilabel_baseline.ipynb
│       └── 05_char_ngram_baseline.ipynb
├── experiments/
│   ├── logreg/            # Single-label artifacts
│   └── logreg_multilabel/ # Multi-label artifacts
├── ethical_audit/
│   └── 01_fairness_audit.ipynb  # Audit notebook (no sensitive features)
├── environment.yml       # Conda environment specification
└── README.md             # ← you are here
