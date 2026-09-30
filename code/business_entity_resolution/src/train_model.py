"""
train_model.py — Training the LightGBM matching classifier.

Pipeline:
  1. Load preprocessed train data and ground truth.
  2. Split S1 entities into train / validation sets.
  3. Run blocking on the train split → candidate pairs.
  4. Build feature matrix with positive + hard-negative examples.
  5. Train LightGBM; save model to MODEL_DIR.
  6. Evaluate on validation set; report F_0.5, precision, recall.
  7. Save optimal threshold for inference.
"""

import os
import sys
import pickle
import logging
import argparse
import random
from collections import defaultdict
from typing import Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

# Add src/ to path so sibling imports work
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
    MODEL_PATH, MODEL_DIR,
    VAL_FRACTION, RANDOM_STATE, NEG_POS_RATIO,
    LGBM_PARAMS, TFIDF_TOP_K,
)
from preprocess import preprocess_df
from blocking import run_blocking, compute_blocking_recall
from features import build_feature_matrix, FEATURE_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Metric helpers ───────────────────────────────────────────────────────────

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def compute_entity_f05(
    predictions: dict,
    ground_truth: dict,
    s1_ids: list,
) -> dict:
    """Macro-averaged F_0.5 over all S1 entities."""
    scores = []
    for s1_id in s1_ids:
        pred  = predictions.get(s1_id, set())
        truth = ground_truth.get(s1_id, set())

        if not truth and not pred:
            scores.append(1.0)   # correctly predicted singleton
        elif not truth and pred:
            scores.append(0.0)   # false merge on singleton
        elif truth and not pred:
            scores.append(f_beta(0.0, 0.0))
        else:
            tp = len(pred & truth)
            p  = tp / len(pred)
            r  = tp / len(truth)
            scores.append(f_beta(p, r))

    return {"macro_f05": float(np.mean(scores)), "n_entities": len(s1_ids)}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_data(sample_frac: float = 1.0):
    """Load and preprocess all training data."""
    logger.info("Loading training data …")
    s1 = pd.read_csv(TRAIN_S1, sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(TRAIN_S2, sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(TRAIN_S3, sep="\t", dtype=str).fillna("")
    gt = pd.read_csv(TRAIN_GT, sep="\t", dtype=str).fillna("")

    if sample_frac < 1.0:
        s1 = s1.sample(frac=sample_frac, random_state=RANDOM_STATE).reset_index(drop=True)
        logger.info(f"  Sampled to {len(s1):,} S1 entities")

    logger.info(f"  S1: {len(s1):,} | S2: {len(s2):,} | S3: {len(s3):,}")

    logger.info("Preprocessing …")
    s1  = preprocess_df(s1)                              # small — no cache needed
    s2  = preprocess_df(s2, cache_key=TRAIN_S2)         # 5M rows — cache to parquet
    s3  = preprocess_df(s3, cache_key=TRAIN_S3)         # 5M rows — cache to parquet
    s23 = pd.concat([s2, s3], ignore_index=True)

    # Build ground-truth dict — only for sampled S1 IDs
    s1_id_set = set(s1["entity_id"].tolist())
    gt_dict: dict = {}
    for _, row in gt.iterrows():
        sid = row["source1_entity_id"]
        if sid not in s1_id_set:
            continue
        matched = row["matched_entity_ids"].strip()
        gt_dict[sid] = set(matched.split(",")) - {""} if matched else set()

    return s1, s23, gt_dict


from features import FastLookup

def build_lookup(df: pd.DataFrame) -> FastLookup:
    """
    Build columnar FastLookup index for ultra-fast O(1) feature computation.
    """
    return FastLookup(df)


# ─── Training pair construction ───────────────────────────────────────────────

def build_training_pairs(
    s1_train: pd.DataFrame,
    s23: pd.DataFrame,
    gt_dict: dict,
    s23_lookup: dict,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Build balanced positive + hard-negative training pairs.
    Returns X (features), y (labels), and the blocking candidates dict.
    """
    logger.info(f"Running blocking on {len(s1_train):,} train S1 entities …")
    candidates = run_blocking(s1_train, s23, tfidf_top_k=TFIDF_TOP_K)

    # Measure blocking recall on this split
    gt_train = {k: gt_dict.get(k, set()) for k in s1_train["entity_id"].tolist()}
    compute_blocking_recall(candidates, gt_train)

    s1_lookup = build_lookup(s1_train)

    all_pairs  = []
    all_labels = []

    for s1_id in s1_train["entity_id"].tolist():
        true_matches = gt_dict.get(s1_id, set())
        cands        = candidates.get(s1_id, set())

        # Positives captured by blocking
        positives = true_matches & cands
        for pos in positives:
            all_pairs.append((s1_id, pos))
            all_labels.append(1)

        # Positives missed by blocking — still train on them using lookup
        missed = true_matches - cands
        for m in missed:
            if m in s23_lookup:
                all_pairs.append((s1_id, m))
                all_labels.append(1)

        # Hard negatives: sample from non-match candidates
        negatives = list(cands - true_matches)
        n_neg     = min(len(negatives), NEG_POS_RATIO * max(1, len(true_matches)))
        if negatives:
            sampled_negs = random.sample(negatives, n_neg)
            for neg in sampled_negs:
                all_pairs.append((s1_id, neg))
                all_labels.append(0)

    n_pos = sum(all_labels)
    n_neg = len(all_labels) - n_pos
    logger.info(f"Training pairs: {n_pos:,} positives, {n_neg:,} negatives")

    logger.info("Building feature matrix …")
    X = build_feature_matrix(all_pairs, s1_lookup, s23_lookup)
    y = np.array(all_labels, dtype=np.int32)
    return X, y, candidates


# ─── Model training ───────────────────────────────────────────────────────────

def train(args):
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    s1, s23, gt_dict = load_data(sample_frac=args.sample_frac)

    # Train/val split on S1 entity level
    s1_ids     = s1["entity_id"].tolist()
    train_ids, val_ids = train_test_split(
        s1_ids, test_size=VAL_FRACTION, random_state=RANDOM_STATE
    )

    s1_train = s1[s1["entity_id"].isin(set(train_ids))].reset_index(drop=True)
    s1_val   = s1[s1["entity_id"].isin(set(val_ids))].reset_index(drop=True)
    logger.info(f"Train split: {len(s1_train):,} S1 | Val split: {len(s1_val):,} S1")

    # Build s23 lookup ONCE — fast dict from set_index
    logger.info("Building S2+S3 lookup …")
    s23_lookup = build_lookup(s23)
    logger.info(f"  Lookup has {len(s23_lookup):,} entries")

    # Build training pairs
    X_train, y_train, _ = build_training_pairs(s1_train, s23, gt_dict, s23_lookup)

    # Train LightGBM
    logger.info(f"Training LightGBM on {len(X_train):,} pairs …")
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train, feature_name=FEATURE_NAMES)
    logger.info("LightGBM training complete")

    # ── Validation ────────────────────────────────────────────────────────────
    logger.info(f"Running validation blocking on {len(s1_val):,} val S1 entities …")
    val_candidates = run_blocking(s1_val, s23, tfidf_top_k=TFIDF_TOP_K)

    val_pairs = []
    for s1_id in s1_val["entity_id"].tolist():
        for cand_id in val_candidates.get(s1_id, set()):
            val_pairs.append((s1_id, cand_id))

    logger.info(f"Val candidate pairs: {len(val_pairs):,}")
    s1_val_lookup = build_lookup(s1_val)
    X_val  = build_feature_matrix(val_pairs, s1_val_lookup, s23_lookup)
    probs  = model.predict_proba(X_val)[:, 1]

    # Tune threshold for F_0.5
    best_threshold, best_f05 = 0.5, 0.0
    for threshold in np.arange(0.3, 0.91, 0.05):
        preds: dict = defaultdict(set)
        for (s1_id, s23_id), prob in zip(val_pairs, probs):
            if prob >= threshold:
                preds[s1_id].add(s23_id)
        result = compute_entity_f05(dict(preds), gt_dict, val_ids)
        f05    = result["macro_f05"]
        logger.info(f"  threshold={threshold:.2f} → macro F_0.5={f05:.4f}")
        if f05 > best_f05:
            best_f05, best_threshold = f05, float(threshold)

    logger.info(f"Best threshold: {best_threshold:.2f} → F_0.5={best_f05:.4f}")

    # Save model + threshold
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "threshold": best_threshold}, f)
    logger.info(f"Model saved → {MODEL_PATH}")

    # Feature importances
    importances = sorted(
        zip(FEATURE_NAMES, model.feature_importances_),
        key=lambda x: -x[1],
    )
    logger.info("Top 10 feature importances:")
    for fname, imp in importances[:10]:
        logger.info(f"  {imp:6d}  {fname}")


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train entity resolution model")
    parser.add_argument(
        "--sample-frac", type=float, default=1.0,
        help="Fraction of S1 training entities to use (default 1.0; use 0.01 for quick test)"
    )
    args = parser.parse_args()
    train(args)
