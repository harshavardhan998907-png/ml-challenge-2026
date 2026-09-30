"""
pipeline.py — End-to-end runner for the Business Entity Resolution pipeline.

Usage (from student_resource/ directory):
    python code/business_entity_resolution/src/pipeline.py --mode train
    python code/business_entity_resolution/src/pipeline.py --mode infer
    python code/business_entity_resolution/src/pipeline.py --mode train-infer

For a quick dev run (1% of data):
    python code/business_entity_resolution/src/pipeline.py --mode train --sample-frac 0.01
"""

import os
import sys
import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)

# Ensure src/ is on the path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)


def run_train(sample_frac: float):
    from train_model import train
    import argparse as ap
    args = ap.Namespace(sample_frac=sample_frac)
    train(args)


def run_infer(threshold=None, top_k=None):
    from inference import run_inference
    from config import TFIDF_TOP_K
    import argparse as ap
    args = ap.Namespace(threshold=threshold, top_k=top_k or TFIDF_TOP_K)
    run_inference(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument(
        "--mode", choices=["train", "infer", "train-infer"], required=True,
        help="Pipeline mode"
    )
    parser.add_argument(
        "--sample-frac", type=float, default=1.0,
        help="Fraction of training S1 entities (1.0 = full, 0.01 = dev run)"
    )
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override match threshold for inference")
    parser.add_argument("--top-k", type=int, default=None,
                        help="TF-IDF blocking top-K (default from config)")
    args = parser.parse_args()

    if args.mode in ("train", "train-infer"):
        logging.getLogger().info("=== TRAINING ===")
        run_train(args.sample_frac)

    if args.mode in ("infer", "train-infer"):
        logging.getLogger().info("=== INFERENCE ===")
        run_infer(args.threshold, args.top_k)
