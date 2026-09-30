"""
config.py — Central configuration for the Business Entity Resolution pipeline.
All paths, hyperparameters, and tunable settings live here.
"""

import os

# ─── Base Paths ──────────────────────────────────────────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATASET_DIR  = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR    = os.path.join(DATASET_DIR, "train")
TEST_DIR     = os.path.join(DATASET_DIR, "test")
OUTPUT_DIR   = os.path.join(BASE_DIR, "output")
MODEL_DIR    = os.path.join(BASE_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ─── File Paths ───────────────────────────────────────────────────────────────
TRAIN_S1   = os.path.join(TRAIN_DIR, "train_source1.tsv")
TRAIN_S2   = os.path.join(TRAIN_DIR, "train_source2.tsv")
TRAIN_S3   = os.path.join(TRAIN_DIR, "train_source3.tsv")
TRAIN_GT   = os.path.join(TRAIN_DIR, "train_ground_truth.tsv")
TEST_S1    = os.path.join(TEST_DIR,  "test_source1.tsv")
TEST_S2    = os.path.join(TEST_DIR,  "test_source2.tsv")
TEST_S3    = os.path.join(TEST_DIR,  "test_source3.tsv")

OUTPUT_MATCHING   = os.path.join(OUTPUT_DIR, "matching_results.tsv")
OUTPUT_CANDIDATES = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")
MODEL_PATH        = os.path.join(MODEL_DIR,  "lgbm_model.pkl")

# ─── Preprocessing ────────────────────────────────────────────────────────────
# Legal suffix tokens to strip for normalised-name blocking keys
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "llc", "llp", "lp",
    "ltd", "limited", "pvt", "private", "co", "company", "plc",
    "sarl", "sas", "sa", "sasu", "eurl",   # French legal forms
}

# Road / address abbreviation expansions
ADDRESS_ABBR = {
    r"\bst\b":   "street",
    r"\brd\b":   "road",
    r"\bave\b":  "avenue",
    r"\bav\b":   "avenue",
    r"\bblvd\b": "boulevard",
    r"\bdr\b":   "drive",
    r"\bln\b":   "lane",
    r"\bct\b":   "court",
    r"\bpl\b":   "place",
    r"\bpkwy\b": "parkway",
    r"\bhwy\b":  "highway",
    r"\bfwy\b":  "freeway",
    r"\bnr\b":   "near",
    r"\bopp\b":  "opposite",
    r"\bsoc\b":  "society",
    r"\bnagar\b":"nagar",
    r"\bcol\b":  "colony",
}

# Name abbreviation expansions (applied before matching)
NAME_ABBR = {
    r"\s*&\s*":    " and ",   # ensure spaces around & → "A&B" becomes "A and B"
    r"\bcorp\b":   "corporation",
    r"\bpvt\b":    "private",
    r"\bltd\b":    "limited",
    r"\bco\b":     "company",
    r"\binc\b":    "incorporated",
    r"\bllc\b":    "limited liability company",
    r"\bllp\b":    "limited liability partnership",
    r"\bintl\b":   "international",
    r"\bsvc\b":    "services",
    r"\bsvcs\b":   "services",
    r"\btechnologies\b": "technologies",
    r"\btech\b":   "technologies",
}

# Tokens to strip from blocking keys (legal suffixes + generic stop words)
KEY_STOP_WORDS = LEGAL_SUFFIXES | {
    "and", "the", "of", "in", "at", "by", "for", "to", "a", "an",
    "de", "du", "le", "la", "les", "et",   # French common words
}

# ─── Blocking ─────────────────────────────────────────────────────────────────
TFIDF_TOP_K          = 10    # TF-IDF ANN: retrieve top-K S2/S3 per S1 entity
MINHASH_NUM_PERM     = 128   # MinHash permutations
MINHASH_THRESHOLD    = 0.25  # Jaccard similarity threshold for MinHash LSH
MINHASH_NGRAM        = 3     # character n-gram size for MinHash
CHUNK_SIZE           = 50_000  # rows per chunk for large-scale processing

# ─── Feature Engineering ─────────────────────────────────────────────────────
MAX_NAME_LEN   = 200   # truncate name strings at this length
MAX_ADDR_LEN   = 300   # truncate address strings at this length

# ─── Model Training ──────────────────────────────────────────────────────────
VAL_FRACTION   = 0.15   # fraction of S1 entities held out for validation
RANDOM_STATE   = 42
NEG_POS_RATIO  = 5      # hard negatives per positive training pair

LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "n_estimators":     800,
    "learning_rate":    0.05,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples":20,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "n_jobs":           -1,
    "random_state":     RANDOM_STATE,
    "verbose":          -1,
}

# ─── Inference ────────────────────────────────────────────────────────────────
# Threshold tuned for F_0.5 on validation set (precision-heavy).
# Start at 0.5, tune upward if precision is low.
MATCH_THRESHOLD = 0.5   # updated by threshold tuning script
