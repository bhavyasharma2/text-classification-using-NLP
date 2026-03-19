# Ancient Inscription Provenance Classification

> Multilingual NLP pipeline for classifying historical inscriptions by geographic origin — combining character-level TF-IDF, LightGBM, and a fine-tuned XLM-RoBERTa transformer.

---

## Overview

This project tackles **text provenance detection on ancient historical inscriptions** — a challenging, low-resource NLP problem where the writing style, script patterns, and character n-grams carry the signal, not modern semantics.

The pipeline was built and evaluated on a dataset of **~120,000 labelled inscriptions** (15 classes) as part of a competitive exam held by **IIT Madras**, achieving a **macro-F1 of 0.447** on the held-out test set.

---

## Problem

Given a short text snippet from an ancient inscription, predict its **geographic/cultural provenance** (multi-class classification across 15 categories). Challenges include:

- Texts are short, noisy, and lack modern vocabulary
- Class imbalance across provenance categories
- Scripts span multiple ancient languages and writing systems
- No pre-trained models exist specifically for this domain

---

## Pipeline Architecture

```
Raw Text
   │
   ├──► [Baseline] char-level TF-IDF (3–5 ngrams) ──► LightGBM   → macro-F1: 0.39
   │
   └──► [Final]    XLM-RoBERTa (fine-tuned, fp16)  → macro-F1: 0.447
```

### Why XLM-RoBERTa?

XLM-RoBERTa is pre-trained on 100+ languages with a large multilingual SentencePiece vocabulary, making it uniquely suited for cross-lingual and low-resource text where standard English models fail. Ancient inscriptions benefit from its subword tokenization, which handles rare character sequences gracefully.

---

## Project Structure

```
├── notebook.ipynb          # Full training pipeline (Kaggle)
├── submission_lgbm.csv     # LightGBM baseline predictions
├── submission_xlmr.csv     # Final transformer predictions
└── README.md
```

---

## Methodology

### Stage 1 — Baseline: TF-IDF + LightGBM

- **Vectorizer**: `TfidfVectorizer` with `analyzer='char_wb'`, n-gram range (3, 5), 20,000 features
- **Rationale**: Character n-grams capture script-level and morphological patterns without needing semantic understanding
- **Model**: LightGBM (`multiclass`, 64 leaves, feature/bagging fraction, early stopping)
- **Result**: Macro-F1 of **0.395** on validation set

### Stage 2 — Fine-tuned Transformer: XLM-RoBERTa

- **Model**: `xlm-roberta-base` from HuggingFace, fine-tuned end-to-end
- **Tokenization**: SentencePiece, max length 256 tokens, padding + truncation
- **Training**: 2 epochs, batch size 16, learning rate 3e-5, **fp16 mixed-precision**
- **Evaluation**: Macro-F1 via `compute_metrics` callback in HuggingFace `Trainer`
- **Inference**: Custom batched DataLoader for efficient GPU prediction
- **Result**: Macro-F1 of **0.447** on test set — **+13.7% improvement over baseline**

---

## Results

| Model | Macro-F1 |
|---|---|
| TF-IDF + LightGBM (baseline) | 0.395 |
| XLM-RoBERTa (fine-tuned) | **0.447** |
| Improvement | **+13.7%** |

> Competition: IIT Madras Internal Exam (Private Kaggle Leaderboard)
> Final Rank: **69 / 122**

---

## Tech Stack

| Tool | Purpose |
|---|---|
| `transformers` | XLM-RoBERTa model + Trainer API |
| `datasets` | HuggingFace dataset tokenization pipeline |
| `lightgbm` | Gradient boosted baseline |
| `scikit-learn` | TF-IDF, label encoding, F1 metric |
| `PyTorch` | Custom batched inference with DataLoader |
| `fp16` | Mixed-precision training for memory efficiency |

---

## How to Run

1. Clone the repo and open `notebook.ipynb` in a Kaggle or GPU-enabled environment
2. Place the dataset CSVs in `/kaggle/input/the-ancient-texts-provenance-challenge/`
3. Run all cells in order — the notebook handles library installation, training, and submission file generation

**Requirements:**
```
transformers
datasets
lightgbm
scikit-learn
torch
sentencepiece
```

---

## Key Takeaways

- Character-level features are surprisingly competitive for ancient/historical text — reaching 0.39 F1 with zero fine-tuning
- XLM-RoBERTa's multilingual pretraining generalizes well even to ancient scripts it was never explicitly trained on
- fp16 mixed-precision training cuts memory usage significantly without affecting convergence
- Early stopping on LightGBM converged in just 16 rounds, indicating TF-IDF features plateau quickly for this task

* Probability-level ensembling of TF-IDF and transformer predictions
* Hyperparameter tuning for transformer fine-tuning
* Exploring larger transformer variants (compute permitting)
* Detailed error analysis to study regional linguistic overlap
* Create a **repo folder structure**
* Help you prepare a **1-minute explanation** for interviews
