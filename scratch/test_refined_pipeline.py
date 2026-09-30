import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

# Add src
sys.path.insert(0, "code/business_entity_resolution/src")
from features import FastLookup, build_feature_matrix
from preprocess import preprocess_df

print("=== SCORING 95.3% RECALL BLOCKING CANDIDATES WITH LIGHTGBM ===")

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

s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t").set_index("entity_id")
s2 = pd.read_parquet("cache/train_source2_preprocessed.parquet")
s3 = pd.read_parquet("cache/train_source3_preprocessed.parquet")

s2_sub = pd.concat([s2[s2["entity_id"].isin(all_m_ids)], s2.sample(n=40000, random_state=42)], ignore_index=True)
s3_sub = pd.concat([s3[s3["entity_id"].isin(all_m_ids)], s3.sample(n=40000, random_state=42)], ignore_index=True)
s23 = pd.concat([s2_sub, s3_sub], ignore_index=True).drop_duplicates(subset=["entity_id"]).set_index("entity_id")

with open("models/lgbm_model.pkl", "rb") as f:
    model_data = pickle.load(f)
model = model_data["model"]

_RE_SUFFIXES = re.compile(r"\b(private limited|pvt ltd|pvt\.?|ltd\.?|limited|inc\.?|incorporated|corp\.?|corporation|llc\.?|llp\.?|center|centre|trust|group|company|co\.?)\b", re.IGNORECASE)
_RE_DOMAINS = re.compile(r"\.(com|org|in|net|fr|co|io|gov)\b|www\.", re.IGNORECASE)
_RE_NON_ALPHANUM = re.compile(r"[^a-z0-9]")
_RE_NUMS = re.compile(r"\d+")
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

def get_name_pairs(name: str):
    if not name or pd.isna(name): return []
    s = str(name).lower()
    s = _RE_SUFFIXES.sub("", s)
    words = [w for w in _RE_WORDS.findall(s) if w not in STOP_WORDS]
    if len(words) >= 2:
        return [f"{words[0]}_{words[1]}"]
    elif words:
        return [words[0]]
    return []

def extract_addr_signatures(addr: str):
    if not addr or pd.isna(addr): return []
    s = str(addr).lower()
    raw_nums = _RE_NUMS.findall(s)
    nums = []
    for n in raw_nums:
        if 1 <= len(n) <= 6:
            try:
                nums.append(str(int(n)))
            except:
                nums.append(n)
                
    words = [w for w in _RE_WORDS.findall(s) if w not in _STOP_ADDR]
    sigs = []
    for n in nums[:3]:
        for w in words[:3]:
            sigs.append(f"{n}_{w}")
    if len(words) >= 2:
        sigs.append(f"st_{words[0]}_{words[1]}")
    for n in raw_nums:
        if len(n) in (5, 6):
            sigs.append(f"pin_{n}")
    return sigs

idx_clean = defaultdict(list)
idx_sorted = defaultdict(list)
idx_name_pair = defaultdict(list)
idx_addr = defaultdict(list)
idx_rare = defaultdict(list)

for eid, r in s23.iterrows():
    ctry = str(r["country"]).upper()
    ck = clean_key(r["business_name"])
    if ck and len(ck) >= 3: idx_clean[(ctry, ck)].append(eid)
    sk = clean_sorted_key(r["business_name"])
    if sk and len(sk) >= 3: idx_sorted[(ctry, sk)].append(eid)
    for npk in get_name_pairs(r["business_name"]):
        idx_name_pair[(ctry, npk)].append(eid)
    for sig in extract_addr_signatures(r["business_address"]):
        idx_addr[(ctry, sig)].append(eid)
    words = [w for w in _RE_WORDS.findall(str(r["business_name"]).lower()) if w not in STOP_WORDS and len(w) >= 8]
    for rw in words[:2]:
        idx_rare[(ctry, rw)].append(eid)

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
    ck = clean_key(bname)
    if (ctry, ck) in idx_clean: cand_set.update(idx_clean[(ctry, ck)][:10])
    sk = clean_sorted_key(bname)
    if (ctry, sk) in idx_sorted: cand_set.update(idx_sorted[(ctry, sk)][:10])
    for npk in get_name_pairs(bname):
        if (ctry, npk) in idx_name_pair: cand_set.update(idx_name_pair[(ctry, npk)][:5])
    for sig in extract_addr_signatures(baddr):
        if (ctry, sig) in idx_addr: cand_set.update(idx_addr[(ctry, sig)][:8])
    words = [w for w in _RE_WORDS.findall(str(bname).lower()) if w not in STOP_WORDS and len(w) >= 8]
    for rw in words[:2]:
        if (ctry, rw) in idx_rare and len(idx_rare[(ctry, rw)]) <= 5:
            cand_set.update(idx_rare[(ctry, rw)])
            
    c_list = list(cand_set)[:30]
    s1_cands_map[sid] = c_list
    for cid in c_list:
        eval_pairs.append((sid, cid))

print(f"Total candidate pairs: {len(eval_pairs)}")
rec_list = [len(set(s1_cands_map.get(sid, [])) & true_m) / len(true_m) for sid, true_m in gt_dict.items() if true_m and sid in s1_eval["entity_id"].values]
print(f"Blocking Recall: {np.mean(rec_list):.4f}")

X = build_feature_matrix(eval_pairs, s1_lookup, s23_lookup)
probs = model.predict_proba(X)[:, 1]

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0

pair_probs = {pair: prob for pair, prob in zip(eval_pairs, probs)}

STATE_MAP = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
    'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga',
    'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
    'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms',
    'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv', 'new hampshire': 'nh',
    'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc',
    'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa',
    'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd', 'tennessee': 'tn',
    'texas': 'tx', 'utah': 'ut', 'vermont': 'vt', 'virginia': 'va', 'washington': 'wa',
    'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy'
}

def extract_state(addr: str) -> str:
    if not addr: return ""
    tokens = set(re.findall(r"[a-z]+", str(addr).lower()))
    for t in tokens:
        if t in STATE_MAP:
            return STATE_MAP[t]
        if t in STATE_MAP.values() and len(t) == 2:
            return t
    return ""

def are_addresses_compatible(addr1: str, addr2: str) -> bool:
    if not addr1 or not addr2: return True
    s1_str = str(addr1).lower()
    s2_str = str(addr2).lower()
    if s2_str in ("", "null", "nan"): return True
    
    st1 = extract_state(s1_str)
    st2 = extract_state(s2_str)
    if st1 and st2 and st1 != st2:
        return False
        
    nums1 = set(_RE_NUMS.findall(s1_str))
    nums2 = set(_RE_NUMS.findall(s2_str))
    if nums1 and nums2:
        if not (nums1 & nums2):
            words1 = set(_RE_WORDS.findall(s1_str)) - _STOP_ADDR
            words2 = set(_RE_WORDS.findall(s2_str)) - _STOP_ADDR
            if len(words1 & words2) < 2:
                return False
    return True

s1_addr_map = dict(zip(s1_eval["entity_id"], s1_eval["business_address"]))
s23_addr_map = dict(zip(s23_eval["entity_id"], s23_eval["business_address"]))

print("\n--- RESULTS WITH ADDRESS COMPATIBILITY FILTERING ---")
for th in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
    scores, ps, rs = [], [], []
    for sid, true_m in gt_dict.items():
        if sid not in s1_eval["entity_id"].values: continue
        c_list = s1_cands_map.get(sid, [])
        pred = set(cid for cid in c_list if pair_probs.get((sid, cid), 0.0) >= th and are_addresses_compatible(s1_addr_map[sid], s23_addr_map.get(cid, "")))
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
        ps.append(p)
        rs.append(r)
    print(f"Threshold {th:0.2f} + Compatibility -> Macro F0.5: {np.mean(scores):.4f} (Precision: {np.mean(ps):.4f}, Recall: {np.mean(rs):.4f})")

print("\n--- SAMPLE FALSE POSITIVES (prob >= 0.8) ---")
s1_name_map = dict(zip(s1_eval["entity_id"], s1_eval["business_name"]))
s23_name_map = dict(zip(s23_eval["entity_id"], s23_eval["business_name"]))
fp_count = 0
for (sid, cid), prob in pair_probs.items():
    if prob >= 0.8:
        true_m = gt_dict.get(sid, set())
        if cid not in true_m:
            compat = are_addresses_compatible(s1_addr_map[sid], s23_addr_map.get(cid, ""))
            if compat:
                fp_count += 1
                if fp_count <= 10:
                    print(f"FP #{fp_count} (prob: {prob:.4f}):")
                    print(f"  S1  ({sid}): {s1_name_map[sid]} ||| {s1_addr_map[sid]}")
                    print(f"  CAND({cid}): {s23_name_map[cid]} ||| {s23_addr_map.get(cid, '')}")
print(f"Total false positives (prob >= 0.8 + compat): {fp_count}")


