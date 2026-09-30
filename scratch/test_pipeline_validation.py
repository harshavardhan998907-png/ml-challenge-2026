import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict, Counter

# Add src
sys.path.insert(0, "code/business_entity_resolution/src")
from features import FastLookup, build_feature_matrix

print("=== EVALUATING END-TO-END PIPELINE (MULTI-KEY BLOCKING + LIGHTGBM) ===")

gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=1000)
needed_s1 = set(gt["source1_entity_id"])
all_m_ids = set()
gt_dict = {}
for _, r in gt.iterrows():
    sid = r["source1_entity_id"]
    raw = str(r["matched_entity_ids"])
    ms = set(m.strip() for m in raw.split(",") if m.strip() and m != "nan")
    gt_dict[sid] = ms
    all_m_ids.update(ms)

print("Loading S1...")
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t").set_index("entity_id")

print("Loading S2 and S3...")
s2 = pd.read_parquet("cache/train_source2_preprocessed.parquet")
s3 = pd.read_parquet("cache/train_source3_preprocessed.parquet")

s2_sub = pd.concat([s2[s2["entity_id"].isin(all_m_ids)], s2.sample(n=30000, random_state=42)], ignore_index=True)
s3_sub = pd.concat([s3[s3["entity_id"].isin(all_m_ids)], s3.sample(n=30000, random_state=42)], ignore_index=True)
s23 = pd.concat([s2_sub, s3_sub], ignore_index=True).drop_duplicates(subset=["entity_id"]).set_index("entity_id")
print(f"Candidate pool size: {len(s23)}")

# Load LightGBM model
with open("models/lgbm_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]
print(f"Loaded LightGBM model (trained with threshold {model_data.get('threshold', 0.85):.2f})")

_RE_SUFFIXES = re.compile(r"\b(private limited|pvt ltd|pvt\.?|ltd\.?|limited|inc\.?|incorporated|corp\.?|corporation|llc\.?|llp\.?|center|centre|trust|group|company|co\.?)\b", re.IGNORECASE)
_RE_DOMAINS = re.compile(r"\.(com|org|in|net|fr|co|io|gov)\b|www\.", re.IGNORECASE)
_RE_NON_ALPHANUM = re.compile(r"[^a-z0-9]")
_RE_NUMS = re.compile(r"\b\d+\b")
_RE_WORDS = re.compile(r"[a-z]{3,}")
_STOP_ADDR = {
    "street", "st", "road", "rd", "avenue", "ave", "lane", "ln", "drive", "dr",
    "court", "ct", "boulevard", "blvd", "floor", "fl", "near", "opp", "door",
    "plot", "flat", "phase", "sector", "block", "house", "bldg", "building",
    "room", "box", "row", "unit", "apt", "suite", "ste", "null", "post", "township",
    "main", "cross", "ward", "north", "south", "east", "west"
}
STOP_WORDS = {
    "private", "limited", "pvt", "ltd", "inc", "corp", "corporation", "llc", "llp",
    "the", "and", "center", "centre", "trust", "group", "services", "service",
    "enterprises", "solutions", "co", "company", "assoc", "associates", "india", "us", "france"
}

def clean_key(name: str) -> str:
    if not name or pd.isna(name): return ""
    s = str(name).lower()
    s = _RE_SUFFIXES.sub("", s)
    s = _RE_DOMAINS.sub("", s)
    s = _RE_NON_ALPHANUM.sub("", s)
    return s

def clean_sorted_key(name: str) -> str:
    if not name or pd.isna(name): return ""
    s = str(name).lower()
    s = _RE_SUFFIXES.sub("", s)
    s = _RE_DOMAINS.sub("", s)
    words = _RE_WORDS.findall(s)
    words = [w for w in words if w not in STOP_WORDS]
    words = sorted(set(words))
    return "_".join(words) if words else ""

def extract_addr_signatures(addr: str):
    if not addr or pd.isna(addr): return []
    s = str(addr).lower()
    nums = [n for n in _RE_NUMS.findall(s) if 1 <= len(n) <= 6]
    words = [w for w in _RE_WORDS.findall(s) if w not in _STOP_ADDR]
    sigs = []
    if nums and words:
        n0 = nums[0]
        for w in words[:3]:
            sigs.append(f"{n0}_{w}")
        if len(nums) > 1 and len(nums[1]) >= 4:
            sigs.append(f"pin_{nums[1]}")
    elif len(words) >= 2:
        sigs.append(f"{words[0]}_{words[1]}")
    return sigs

print("Building candidate indices...")
idx_clean = defaultdict(list)
idx_sorted = defaultdict(list)
idx_addr = defaultdict(list)

for eid, r in s23.iterrows():
    ctry = str(r["country"]).upper()
    ck = clean_key(r["business_name"])
    if ck and len(ck) >= 3: idx_clean[(ctry, ck)].append(eid)
    sk = clean_sorted_key(r["business_name"])
    if sk and len(sk) >= 3: idx_sorted[(ctry, sk)].append(eid)
    for sig in extract_addr_signatures(r["business_address"]):
        idx_addr[(ctry, sig)].append(eid)

print("Candidate indices built.")

from preprocess import preprocess_df
s1_eval = preprocess_df(s1.loc[[sid for sid in gt_dict.keys() if sid in s1.index]].reset_index())
s23_eval = s23.reset_index()

s1_lookup = FastLookup(s1_eval)
s23_lookup = FastLookup(s23_eval)

eval_pairs = []
s1_cands_map = {}

for _, r in s1_eval.iterrows():
    sid = r["entity_id"]
    ctry = str(r["country"]).upper()
    bname = r["business_name"]
    baddr = r["business_address"]
    
    cand_set = set()
    # 0. Norm name key
    nk = r["norm_name_key"]
    if nk and (ctry, nk) in idx_clean:
        cand_set.update(idx_clean[(ctry, nk)][:10])
    # 1. Clean key
    ck = clean_key(bname)
    if (ctry, ck) in idx_clean:
        cand_set.update(idx_clean[(ctry, ck)][:12])
    # 2. Sorted key
    sk = clean_sorted_key(bname)
    if (ctry, sk) in idx_sorted:
        cand_set.update(idx_sorted[(ctry, sk)][:12])
    # 3. Address signatures
    for sig in extract_addr_signatures(baddr):
        if (ctry, sig) in idx_addr:
            cand_set.update(idx_addr[(ctry, sig)][:12])
            
    c_list = list(cand_set)[:35]
    s1_cands_map[sid] = c_list
    for cid in c_list:
        eval_pairs.append((sid, cid))

rec_list = []
for sid, true_m in gt_dict.items():
    if not true_m: continue
    c_set = set(s1_cands_map.get(sid, []))
    rec_list.append(len(c_set & true_m) / len(true_m))
print(f"Total candidate pairs generated: {len(eval_pairs)} (Avg per entity: {len(eval_pairs)/len(s1_eval):.1f})")
print(f"Candidate Generation (Blocking) Recall: {np.mean(rec_list):.4f}")

# Feature extraction & scoring
print("Extracting features for candidate pairs...")
X = build_feature_matrix(eval_pairs, s1_lookup, s23_lookup)
probs = model.predict_proba(X)[:, 1]

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0

pair_probs = {pair: prob for pair, prob in zip(eval_pairs, probs)}

# Test multiple thresholds to find optimal operating point
for th in [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]:
    scores = []
    p_list = []
    r_list = []
    for sid, true_m in gt_dict.items():
        if sid not in s1_eval["entity_id"].values: continue
        c_list = s1_cands_map.get(sid, [])
        pred = set()
        for cid in c_list:
            if pair_probs.get((sid, cid), 0.0) >= th:
                pred.add(cid)
                
        if not true_m and not pred:
            p, r, f = 1.0, 1.0, 1.0
        elif not true_m and pred:
            p, r, f = 0.0, 1.0, 0.0
        elif true_m and not pred:
            p, r, f = 1.0, 0.0, 0.0
        else:
            tp = len(pred & true_m)
            p = tp / len(pred)
            r = tp / len(true_m)
            f = f_beta(p, r)
        scores.append(f)
        p_list.append(p)
        r_list.append(r)
        
    print(f"Threshold {th:0.2f} -> Macro F0.5: {np.mean(scores):.4f} (Mean P: {np.mean(p_list):.4f}, Mean R: {np.mean(r_list):.4f})")
