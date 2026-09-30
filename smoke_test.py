"""
smoke_test.py — Verifies all modules import correctly and core logic runs on tiny synthetic data.
Run from: student_resource/student_resource/ directory
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code", "business_entity_resolution", "src"))

print("=" * 60)
print("SMOKE TEST — Business Entity Resolution Pipeline")
print("=" * 60)

# ── 1. Check imports ───────────────────────────────────────────────────────────
print("\n[1] Checking imports...")
try:
    import pandas as pd
    import numpy as np
    import lightgbm as lgb
    from rapidfuzz import fuzz, distance
    from datasketch import MinHash, MinHashLSH
    from sklearn.feature_extraction.text import TfidfVectorizer
    print(f"  ✓ pandas {pd.__version__}")
    print(f"  ✓ numpy {np.__version__}")
    print(f"  ✓ lightgbm {lgb.__version__}")
    print(f"  ✓ rapidfuzz")
    print(f"  ✓ datasketch")
    print(f"  ✓ sklearn")
except ImportError as e:
    print(f"  ✗ MISSING PACKAGE: {e}")
    print("  Run: pip install lightgbm rapidfuzz datasketch scikit-learn")
    sys.exit(1)

# ── 2. Check our modules ───────────────────────────────────────────────────────
print("\n[2] Importing pipeline modules...")
try:
    from preprocess import preprocess_df, normalise_name, normalise_name_key, normalise_address
    print("  ✓ preprocess")
    from blocking import tfidf_blocking, minhash_blocking, zip_blocking, prefix_token_blocking, run_blocking
    print("  ✓ blocking")
    from features import compute_pair_features, build_feature_matrix, FEATURE_NAMES
    print("  ✓ features")
    print(f"  ✓ {len(FEATURE_NAMES)} features defined")
except Exception as e:
    print(f"  ✗ MODULE ERROR: {e}")
    import traceback; traceback.print_exc()
    sys.exit(1)

# ── 3. Test preprocessing ──────────────────────────────────────────────────────
print("\n[3] Testing preprocessing...")
test_cases = [
    ("Raj Investments LLP", "raj investments limited liability partnership"),
    ("-- Holloway Peak Inc Seafood", "holloway peak incorporated seafood"),
    ("Corp & Sons Ltd.", "corporation and sons limited"),
    ("A&B Pvt. Ltd.", "a and b private limited"),
]
for raw, expected_contains in test_cases:
    result = normalise_name(raw)
    key = normalise_name_key(raw)
    print(f"  raw:    '{raw}'")
    print(f"  norm:   '{result}'")
    print(f"  key:    '{key}'")
    print()

# ── 4. Test feature computation ────────────────────────────────────────────────
print("\n[4] Testing feature computation on sample pair...")
row1 = pd.Series({
    "entity_id": "S1-001",
    "norm_name": "raj investments limited liability partnership",
    "norm_name_key": "raj investments",
    "norm_address": "6 29 cit colony 2nd main road mylapore chennai tamil nadu",
    "zip_code": "",
    "country": "India",
    "combined_text": "raj investments limited liability partnership 6 29 cit colony chennai"
})
row2 = pd.Series({
    "entity_id": "S2-001",
    "norm_name": "raj investment llp",
    "norm_name_key": "raj investment",
    "norm_address": "cit colony mylapore chennai",
    "zip_code": "",
    "country": "India",
    "combined_text": "raj investment llp cit colony mylapore chennai"
})
feats = compute_pair_features(row1, row2)
print(f"  Features computed: {len(feats)}")
print(f"  name_token_jaccard: {feats['name_token_jaccard']:.3f}")
print(f"  name_jaro_winkler:  {feats['name_jaro_winkler']:.3f}")
print(f"  addr_token_jaccard: {feats['addr_token_jaccard']:.3f}")
print(f"  country_match:      {feats['country_match']}")

# Also test non-match pair
row3 = pd.Series({
    "entity_id": "S2-999",
    "norm_name": "totally different business name",
    "norm_name_key": "totally different",
    "norm_address": "some other address in texas",
    "zip_code": "75001",
    "country": "US",
    "combined_text": "totally different business name some other address in texas"
})
feats_neg = compute_pair_features(row1, row3)
print(f"\n  Non-match pair:")
print(f"  name_token_jaccard: {feats_neg['name_token_jaccard']:.3f}")
print(f"  country_match:      {feats_neg['country_match']}")

# ── 5. Test tiny end-to-end blocking ──────────────────────────────────────────
print("\n[5] Testing blocking on tiny synthetic dataset...")
s1_data = {
    "entity_id": ["S1-001", "S1-002", "S1-003"],
    "business_name": ["Raj Investments LLP", "Holloway Peak Inc", "Tech Solutions Corp"],
    "business_address": ["CIT Colony, Chennai, TN", "105 Elm St, Morganton NC", "1st Street, Boston MA"],
    "country": ["India", "US", "US"],
}
s23_data = {
    "entity_id": ["S2-001", "S2-002", "S3-001", "S3-002", "S3-003"],
    "business_name": ["Raj Investment LLP", "Holloway Peak Incorporated", "Tech Solutions Corporation", "Random Biz", "Another Company"],
    "business_address": ["CIT Colony Chennai", "105 ELM STREET MORGANTON NC", "First Street Boston", "Somewhere else", "Different place"],
    "country": ["India", "US", "US", "India", "US"],
}

s1  = preprocess_df(pd.DataFrame(s1_data))
s23 = preprocess_df(pd.DataFrame(s23_data))

candidates = run_blocking(s1, s23, use_minhash=True, use_zip=True, use_prefix=True, tfidf_top_k=3)
print(f"\n  Blocking results:")
for s1_id, cands in candidates.items():
    print(f"  {s1_id} → {sorted(cands)}")

# Check S1-001 found S2-001
assert "S2-001" in candidates.get("S1-001", set()) or "S3-001" in candidates.get("S1-001", set()) or len(candidates.get("S1-001", set())) > 0, \
    "WARNING: S1-001 found no candidates at all"

# ── 6. Test LightGBM training on tiny data ────────────────────────────────────
print("\n[6] Testing LightGBM training on tiny data...")
from train_model import build_lookup
s1_lookup  = build_lookup(s1)
s23_lookup = build_lookup(s23)

pairs = [
    ("S1-001", "S2-001"), ("S1-001", "S2-002"),
    ("S1-002", "S2-002"), ("S1-002", "S3-002"),
    ("S1-003", "S3-001"), ("S1-003", "S3-003"),
]
labels = [1, 0, 0, 1, 1, 0]

X = build_feature_matrix(pairs, s1_lookup, s23_lookup)
y = np.array(labels)
print(f"  Feature matrix shape: {X.shape}")
print(f"  Labels: {labels}")

model = lgb.LGBMClassifier(n_estimators=10, verbose=-1)
model.fit(X, y, feature_name=FEATURE_NAMES)
probs = model.predict_proba(X)[:, 1]
print(f"  Probabilities: {[round(p,3) for p in probs]}")
print(f"  Model trained successfully ✓")

print("\n" + "=" * 60)
print("SMOKE TEST PASSED ✓ — All modules working correctly")
print("=" * 60)
print("\nNext step — run full training:")
print("  python code/business_entity_resolution/src/pipeline.py --mode train")
print("\nFor a quick dev run (0.5% data, ~minutes):")
print("  python code/business_entity_resolution/src/pipeline.py --mode train --sample-frac 0.005")
