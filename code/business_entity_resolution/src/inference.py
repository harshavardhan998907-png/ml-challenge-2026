"""
inference.py — Generate predictions on the test set.

Pipeline:
  1. Load pretrained LightGBM model + threshold.
  2. Load and preprocess test data (S1, S2, S3).
  3. Run blocking to generate candidate_pairs.tsv.
  4. Build feature matrix and score all candidates.
  5. Apply threshold → matching_results.tsv.
  6. Validate output format.
"""

import os
import sys
import pickle
import logging
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    TEST_S1, TEST_S2, TEST_S3,
    MODEL_PATH,
    OUTPUT_MATCHING, OUTPUT_CANDIDATES,
    TFIDF_TOP_K, MATCH_THRESHOLD, CHUNK_SIZE,
)
from preprocess import preprocess_df
from blocking import run_blocking, tfidf_blocking
from features import build_feature_matrix, FEATURE_NAMES
from train_model import build_lookup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Output writers ───────────────────────────────────────────────────────────

def write_candidates(candidates: dict, path: str, s1_ids: list):
    """Write candidate_pairs.tsv."""
    logger.info(f"Writing candidate pairs to {path} …")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_ids:
            cands = sorted(candidates.get(s1_id, set()))
            f.write(f"{s1_id}\t{','.join(cands)}\n")
    logger.info(f"  Written {len(s1_ids):,} rows to {path}")


def write_matches(matches: dict, path: str, s1_ids: list):
    """Write matching_results.tsv."""
    logger.info(f"Writing matching results to {path} …")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ids:
            matched = sorted(matches.get(s1_id, set()))
            f.write(f"{s1_id}\t{','.join(matched)}\n")
    logger.info(f"  Written {len(s1_ids):,} rows to {path}")


# ─── Main inference ───────────────────────────────────────────────────────────

def run_inference(args):
    # Load model
    logger.info(f"Loading model from {MODEL_PATH} …")
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    model     = saved["model"]
    threshold = saved.get("threshold", MATCH_THRESHOLD)
    if args.threshold is not None:
        threshold = args.threshold
    logger.info(f"Using match threshold: {threshold:.2f}")

    # Load test data
    logger.info("Loading test data …")
    s1 = pd.read_csv(TEST_S1, sep="\t", dtype=str).fillna("")
    s2 = pd.read_csv(TEST_S2, sep="\t", dtype=str).fillna("")
    s3 = pd.read_csv(TEST_S3, sep="\t", dtype=str).fillna("")
    logger.info(f"  S1: {len(s1):,} | S2: {len(s2):,} | S3: {len(s3):,}")

    # Preprocess
    logger.info("Preprocessing …")
    s1  = preprocess_df(s1)
    s2  = preprocess_df(s2, cache_key=TEST_S2)
    s3  = preprocess_df(s3, cache_key=TEST_S3)
    s23 = pd.concat([s2, s3], ignore_index=True)

    s1_ids     = s1["entity_id"].tolist()
    logger.info("Building S2+S3 lookup …")
    s23_lookup = build_lookup(s23)
    logger.info(f"  Lookup has {len(s23_lookup):,} entries")

    # Blocking — Fast TF-IDF Top-K Cosine Candidate Generation
    logger.info("Running TF-IDF candidate blocking …")
    candidates = tfidf_blocking(s1, s23, top_k=args.top_k)

    # Write candidate_pairs.tsv immediately
    write_candidates(candidates, OUTPUT_CANDIDATES, s1_ids)

    # Score candidates in chunks
    logger.info("Scoring candidate pairs …")
    matches: dict = defaultdict(set)
    s1_lookup = build_lookup(s1)

    # Build all pairs list
    all_pairs = []
    pair_s1_ids = []
    for s1_id in s1_ids:
        cands = candidates.get(s1_id, set())
        for cand_id in cands:
            all_pairs.append((s1_id, cand_id))
            pair_s1_ids.append(s1_id)

    logger.info(f"Total candidate pairs to score: {len(all_pairs):,}")

    chunk_size = CHUNK_SIZE
    n_chunks = max(1, len(all_pairs) // chunk_size + 1)
    for i in range(n_chunks):
        chunk_pairs = all_pairs[i*chunk_size:(i+1)*chunk_size]
        if not chunk_pairs:
            break
        X = build_feature_matrix(chunk_pairs, s1_lookup, s23_lookup)
        probs = model.predict_proba(X)[:, 1]
        for (s1_id, s23_id), prob in zip(chunk_pairs, probs):
            if prob >= threshold:
                matches[s1_id].add(s23_id)
        if (i + 1) % 20 == 0:
            logger.info(f"  Scored chunk {i+1}/{n_chunks}")

    # Write matching_results.tsv
    write_matches(dict(matches), OUTPUT_MATCHING, s1_ids)

    # Quick stats
    n_matched    = sum(1 for v in matches.values() if v)
    n_singletons = len(s1_ids) - n_matched
    logger.info(
        f"Results: {n_matched:,} S1 entities with ≥1 match, "
        f"{n_singletons:,} singletons (no match predicted)"
    )
    logger.info("Done. Run validation with:")
    logger.info(
        "  python utils/validate_submission.py "
        "--matching output/matching_results.tsv "
        "--candidate output/candidate_pairs.tsv "
        "--test-dir dataset/test"
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run entity resolution inference on test set")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override model threshold (default: use saved threshold)")
    parser.add_argument("--top-k", type=int, default=TFIDF_TOP_K,
                        help="TF-IDF top-K for blocking (default: %(default)s)")
    args = parser.parse_args()
    run_inference(args)
