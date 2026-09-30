import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
import time
from collections import defaultdict
import pandas as pd
import numpy as np

print("=== VALIDATING HYBRID MATCHING SYSTEM ===")

# Load 5,000 ground truth rows
gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=5000)
s1_needed = set(gt_df["source1_entity_id"])

s1_chunks = []
for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", chunksize=100000):
    sub = chunk[chunk["entity_id"].isin(s1_needed)]
    if not sub.empty:
        s1_chunks.append(sub)
    if len(s1_chunks) * 100000 > 600000:
        break

s1 = pd.concat(s1_chunks).set_index("entity_id")
s2 = pd.read_parquet("cache/train_source2_preprocessed.parquet")
s3 = pd.read_parquet("cache/train_source3_preprocessed.parquet")

s23 = pd.concat([
    s2[["entity_id", "country", "business_name", "business_address", "norm_name_key", "zip_code"]],
    s3[["entity_id", "country", "business_name", "business_address", "norm_name_key", "zip_code"]]
], ignore_index=True)

STOP_WORDS = {
    "private", "limited", "pvt", "ltd", "inc", "corp", "corporation", "llc", "llp",
    "the", "and", "center", "centre", "trust", "group", "services", "service",
    "enterprises", "solutions", "co", "company", "assoc", "associates", "india", "us", "france"
}

_RE_SUFFIXES = re.compile(
    r"\b(private limited|pvt ltd|pvt\.?|ltd\.?|limited|inc\.?|incorporated|corp\.?|corporation|llc\.?|llp\.?|center|centre|trust|group|company|co\.?)\b",
    re.IGNORECASE
)
_RE_DOMAINS = re.compile(r"\.(com|org|in|net|fr|co|io|gov)\b|www\.", re.IGNORECASE)
_RE_NON_ALPHANUM = re.compile(r"[^a-z0-9]")
_RE_NUMBERS = re.compile(r"\b\d{1,6}\b")
_RE_TOKENS = re.compile(r"[a-z0-9]{3,}")
_STOP_ADDR = {
    "street", "st", "road", "rd", "avenue", "ave", "court", "ct", "lane", "ln",
    "drive", "dr", "boulevard", "blvd", "floor", "fl", "near", "opp", "door", "no",
    "plot", "flat", "phase", "sector", "block", "house", "building", "bldg", "room",
    "po", "box", "township", "cross", "main", "ward"
}

def clean_key(name: str) -> str:
    if not name or pd.isna(name): return ""
    s = str(name).lower()
    s = _RE_SUFFIXES.sub("", s)
    s = _RE_DOMAINS.sub("", s)
    s = _RE_NON_ALPHANUM.sub("", s)
    return s

def get_tokens(name: str):
    if not name or pd.isna(name): return set()
    s = str(name).lower()
    words = _RE_TOKENS.findall(s)
    return set(w for w in words if w not in STOP_WORDS)

def get_addr_signature(addr: str):
    if not addr or pd.isna(addr): return ("", "")
    s = str(addr).lower()
    nums = _RE_NUMBERS.findall(s)
    num = nums[0] if nums else ""
    words = _RE_TOKENS.findall(s)
    sig_words = [w for w in words if w not in _STOP_ADDR and not w.isdigit()]
    street = sig_words[0] if sig_words else ""
    return (num, street)

print(f"Total S23 records: {len(s23):,}")

# Build Inverted Indices for S23
print("Building S23 indices...")
t0 = time.time()
idx_name = defaultdict(list)
idx_clean = defaultdict(list)
idx_addr = defaultdict(list)
idx_zip = defaultdict(list)

for r in s23.itertuples(index=False):
    eid = r.entity_id
    ctry = str(r.country).upper()
    
    # 1. Norm name key
    nk = r.norm_name_key
    if nk and len(nk) >= 3:
        idx_name[(ctry, nk)].append(eid)
        
    # 2. Clean key (domain & suffix removed)
    ck = clean_key(r.business_name)
    if ck and len(ck) >= 4:
        idx_clean[(ctry, ck)].append(eid)
        
    # 3. Address signature (number + street)
    num, street = get_addr_signature(r.business_address)
    if num and street:
        idx_addr[(ctry, num, street)].append(eid)
        
    # 4. ZIP + street
    zc = str(r.zip_code).strip()
    if zc and street:
        idx_zip[(ctry, zc, street)].append(eid)

print(f"Indices built in {time.time()-t0:.2f}s")

# Evaluate Macro F0.5
def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0

gt_dict = {}
for _, r in gt_df.iterrows():
    sid = r["source1_entity_id"]
    if sid not in s1.index: continue
    raw = str(r["matched_entity_ids"])
    gt_dict[sid] = set(m.strip() for m in raw.split(",") if m.strip() and m != "nan")

scores = []
evaluated = 0

for sid, true_matches in gt_dict.items():
    s1_row = s1.loc[sid]
    ctry = str(s1_row["country"]).upper()
    bname = s1_row["business_name"]
    baddr = s1_row["business_address"]
    nk = s1_row["norm_name_key"]
    
    predicted = set()
    
    # Path 1: Exact norm name key
    if (ctry, nk) in idx_name:
        for cid in idx_name[(ctry, nk)]:
            predicted.add(cid)
            
    # Path 2: Clean key (domain removed)
    ck = clean_key(bname)
    if (ctry, ck) in idx_clean:
        for cid in idx_clean[(ctry, ck)]:
            predicted.add(cid)
            
    # Path 3: Address signature (number + street)
    num, street = get_addr_signature(baddr)
    if num and street and (ctry, num, street) in idx_addr:
        # Avoid massive clusters (e.g. generic address with > 20 businesses)
        addr_cands = idx_addr[(ctry, num, street)]
        if len(addr_cands) <= 15:
            for cid in addr_cands:
                predicted.add(cid)
                
    # Path 4: ZIP + street
    zc = str(s1_row.get("zip_code", "")).strip() if hasattr(s1_row, "zip_code") else ""
    if zc and street and (ctry, zc, street) in idx_zip:
        zip_cands = idx_zip[(ctry, zc, street)]
        if len(zip_cands) <= 10:
            for cid in zip_cands:
                predicted.add(cid)
                
    evaluated += 1
    
    # Compute F0.5 for this entity
    if not true_matches and not predicted:
        scores.append(1.0) # True singleton
    elif not true_matches and predicted:
        scores.append(0.0) # False positive on singleton
    elif true_matches and not predicted:
        scores.append(0.0) # False negative on matched entity
    else:
        tp = len(predicted & true_matches)
        p = tp / len(predicted)
        r = tp / len(true_matches)
        scores.append(f_beta(p, r))

macro_f05 = np.mean(scores)
print(f"\n=======================================================")
print(f"Evaluated {evaluated} entities")
print(f"MACRO F0.5 SCORE: {macro_f05:.4f}")
print(f"=======================================================")
