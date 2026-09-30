# Business Entity Resolution — README

## Overview
End-to-end ML pipeline for cross-source business entity resolution.  
Given records from 3 independent sources, determines which records across sources refer to the same real-world business entity.

## Setup

```bash
pip install -r requirements.txt
```

## Directory Layout

```
student_resource/
├── dataset/
│   ├── train/  (train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv)
│   └── test/   (test_source1.tsv, test_source2.tsv, test_source3.tsv)
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── models/
│   └── lgbm_model.pkl
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── config.py          ← all paths & hyperparameters
│   │   ├── preprocess.py      ← text normalisation
│   │   ├── blocking.py        ← TF-IDF ANN + MinHash LSH + ZIP + prefix blocking
│   │   ├── features.py        ← pairwise similarity features
│   │   ├── train_model.py     ← LightGBM training + F_0.5 threshold tuning
│   │   ├── inference.py       ← test-set prediction
│   │   └── pipeline.py        ← end-to-end runner
│   ├── README.md
│   └── requirements.txt
└── utils/
    └── validate_submission.py
```

## Reproducing Results

All commands must be run from the `student_resource/student_resource/` directory.

### Step 1 — Install dependencies
```bash
pip install -r code/business_entity_resolution/requirements.txt
```

### Step 2 — Train
```bash
# Full training run (~hours on large dataset)
python code/business_entity_resolution/src/pipeline.py --mode train

# Quick dev test (1% of data, ~minutes)
python code/business_entity_resolution/src/pipeline.py --mode train --sample-frac 0.01
```

### Step 3 — Inference
```bash
python code/business_entity_resolution/src/pipeline.py --mode infer
```

### Step 4 — Validate output
```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## Pipeline Architecture

```
Raw Data
   ↓ lowercase, expand abbreviations, strip legal suffixes, extract ZIP
Preprocessing
   ↓ TF-IDF ANN (char 2-4 grams, top-10) + MinHash LSH (3-gram, J≥0.25)
     + ZIP-code exact match + country+prefix-token bucket
Blocking  →  candidate_pairs.tsv
   ↓ 26 pairwise features (name Jaccard/Jaro/Levenshtein, address similarity,
     country match, ZIP match, combined TF-IDF cosine)
Feature Engineering
   ↓ LightGBM binary classifier (800 trees)
Matching Model
   ↓ threshold tuned on held-out val split to maximise F_0.5
Threshold Tuning  →  matching_results.tsv
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| F_0.5 metric | Precision-heavy → threshold shifted above 0.5 |
| Char n-gram TF-IDF | Robust to typos, abbreviations, transliteration |
| MinHash LSH | Fast near-exact name matching at scale |
| Legal suffix stripping | "Corp" vs "Corporation" → same key |
| Country-aware ZIP blocking | ZIP spaces differ by country |
| LightGBM classifier | Fast, interpretable, handles tabular features well |
| Multilingual robustness | char-level features work for US/India/France |
