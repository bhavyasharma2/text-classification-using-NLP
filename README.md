# Historical Inscription Origin Classification using NLP

## Overview
This project addresses the task of **classifying historical inscription texts into anonymized geographical regions** using Natural Language Processing (NLP).  
Unlike modern text classification, ancient inscriptions exhibit **non-standard spelling, historical language variation, and subtle stylistic differences**, making provenance prediction a challenging problem.

The project explores and compares **classical feature-based machine learning models** with **modern transformer-based deep learning models** to capture both surface-level and contextual linguistic patterns.

---

## Problem Statement
Given a corpus of **cleaned and anonymized ancient inscription texts**, the goal is to:

> **Predict the geographical origin (region label) of each inscription**

This is formulated as a **multi-class text classification problem**, evaluated using **Macro F1-score** to ensure balanced performance across all regions.

---

## Dataset
- The dataset used in this project was originally obtained from a **Kaggle NLP competition**.
- For **reproducibility and ease of execution**, the **entire input dataset has been downloaded and included in this repository as a ZIP file**.
- The dataset consists of:
  - `train.csv` — inscription text with region labels
  - `test.csv` — inscription text without labels

> ⚠️ **Note**: The dataset is anonymized and contains no personally identifiable information.

### Dataset Location
After extracting the ZIP file, the expected structure is:
```

data/
├── train.csv
└── test.csv

```

---

## Approach

The solution is implemented using **two independent modeling pipelines**, reflecting two different NLP paradigms.

---

### Classical NLP Baseline — TF-IDF + LightGBM
- **Character-level TF-IDF** (`char_wb`, 3–5 n-grams) is used to robustly capture spelling, morphological, and dialectal variations common in historical texts.
- **LightGBM (Gradient Boosted Decision Trees)** performs efficient multi-class classification on high-dimensional sparse features.
- This pipeline serves as a **strong, interpretable baseline** and a performance benchmark.

**Why character-level TF-IDF?**
- Ancient texts lack standardized spelling.
- Character n-grams are more robust than word-level features for historical language.

---

### Deep Learning Model — Transformer Fine-Tuning
- **XLM-RoBERTa (base)** is fine-tuned for sequence classification.
- Chosen for its robustness to **multilingual, noisy, and non-standard text**.
- Training is performed using **HuggingFace Transformers and PyTorch**, enabling end-to-end learning.
- A custom batched inference routine is used to ensure **memory-efficient prediction on large test sets**.

**Why transformers?**
- They learn **contextual and stylistic representations** beyond surface-level features.
- Particularly effective for capturing subtle regional language patterns.

---

## Tech Stack
- **Python**
- **Scikit-learn** (TF-IDF, preprocessing, evaluation)
- **LightGBM**
- **HuggingFace Transformers**
- **PyTorch**
- **HuggingFace Datasets**
- **SentencePiece**

---

## Repository Structure
```

├── data.zip                  # Zipped dataset downloaded from Kaggle
├── code.ipynb            # End-to-end experimentation notebook
└── README.md

````

---

## Evaluation Metric
- **Macro F1-score**
  - Chosen to give equal importance to all geographical regions
  - Prevents bias toward majority classes

---

## Key Design Decisions
- Used **character-level TF-IDF** instead of word-level features to handle spelling variability.
- Maintained **separate pipelines** for classical ML and transformers to respect different learning paradigms.
- Preferred **end-to-end transformer fine-tuning** over frozen embeddings with external classifiers.
- Avoided data augmentation to prevent distortion of historical linguistic and stylistic patterns.
- Generated **separate submission files** to preserve experimental comparisons and reproducibility.

---

## How to Run the Project

### Extract the Dataset
```bash
unzip data.zip
````

Ensure the extracted files follow:

```
data/train.csv
data/test.csv
```

---

### Install Dependencies

```bash
pip install transformers datasets scikit-learn lightgbm sentencepiece torch
```

---

### Run the Notebook

* Open and run `code.ipynb` end-to-end.
* The notebook:

  * Trains the TF-IDF + LightGBM baseline
  * Fine-tunes the transformer model
  * Generates submission files for comparison

---

## Results

* The TF-IDF + LightGBM pipeline provides a fast and reliable baseline.
* The transformer-based model captures deeper contextual and stylistic cues, improving generalization.
* Performance is evaluated using Macro F1-score, with separate outputs preserved for each approach.

---

## Future Improvements

* Probability-level ensembling of TF-IDF and transformer predictions
* Hyperparameter tuning for transformer fine-tuning
* Exploring larger transformer variants (compute permitting)
* Detailed error analysis to study regional linguistic overlap
* Create a **repo folder structure**
* Help you prepare a **1-minute explanation** for interviews
