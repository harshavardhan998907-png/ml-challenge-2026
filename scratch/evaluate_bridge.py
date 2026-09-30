import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from collections import defaultdict

print("=== EVALUATING PRECISION, RECALL & F0.5 ===")

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

s1_rows = []
for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", chunksize=100000):
    sub = chunk[chunk["entity_id"].isin(needed_s1)]
    if not sub.empty:
        s1_rows.append(sub)
    if len(s1_rows) * 100000 > 500000:
        break
s1 = pd.concat(s1_rows).set_index("entity_id")

s2 = pd.read_parquet("cache/train_source2_preprocessed.parquet")
s3 = pd.read_parquet("cache/train_source3_preprocessed.parquet")

s2_sample = pd.concat([s2[s2["entity_id"].isin(all_m_ids)], s2.sample(n=40000, random_state=42)], ignore_index=True)
s3_sample = pd.concat([s3[s3["entity_id"].isin(all_m_ids)], s3.sample(n=40000, random_state=42)], ignore_index=True)
s23 = pd.concat([s2_sample, s3_sample], ignore_index=True).drop_duplicates(subset=["entity_id"]).set_index("entity_id")

US_STATES = {
    "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia", "ks",
    "ky", "la", "me", "md", "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny",
    "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc", "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy"
}

_RE_SUFFIXES = re.compile(r"\b(private limited|pvt ltd|pvt\.?|ltd\.?|limited|inc\.?|incorporated|corp\.?|corporation|llc\.?|llp\.?|center|centre|trust|group|company|co\.?)\b", re.IGNORECASE)
_RE_DOMAINS = re.compile(r"\.(com|org|in|net|fr|co|io|gov)\b|www\.", re.IGNORECASE)
_RE_NON_ALPHANUM = re.compile(r"[^a-z0-9]")
_RE_NUMBERS = re.compile(r"\d+")
_RE_TOKENS = re.compile(r"[a-z0-9]{3,}")
_STOP_ADDR = {
    "street", "st", "road", "rd", "avenue", "ave", "court", "ct", "lane", "ln",
    "drive", "dr", "boulevard", "blvd", "floor", "fl", "near", "opp", "door", "no",
    "plot", "flat", "phase", "sector", "block", "house", "building", "bldg", "room",
    "po", "box", "township", "cross", "main", "ward"
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
    words = _RE_TOKENS.findall(s)
    words = [w for w in words if w not in STOP_WORDS]
    return "_".join(sorted(words)) if words else ""

def get_addr_keys(addr: str):
    if not addr or pd.isna(addr): return []
    s = str(addr).lower()
    nums = [int(n) for n in _RE_NUMBERS.findall(s) if len(n) <= 6]
    words = _RE_TOKENS.findall(s)
    sig_words = [w for w in words if w not in _STOP_ADDR and not w.isdigit()]
    keys = []
    if nums and sig_words:
        n0 = str(nums[0])
        for w in sig_words[:3]:
            keys.append(f"{n0}_{w}")
    elif len(sig_words) >= 2:
        keys.append(f"{sig_words[0]}_{sig_words[1]}")
    return keys

def is_compatible_location(addr1: str, addr2: str) -> bool:
    if not addr1 or not addr2: return True
    t1 = set(re.findall(r"[a-z]{2,}", str(addr1).lower()))
    t2 = set(re.findall(r"[a-z]{2,}", str(addr2).lower()))
    s1_states = t1 & US_STATES
    s2_states = t2 & US_STATES
    if s1_states and s2_states and not (s1_states & s2_states):
        return False
    return True

idx_clean = defaultdict(list)
idx_sorted = defaultdict(list)
idx_addr = defaultdict(list)

for eid, r in s23.iterrows():
    ctry = str(r["country"]).upper()
    ck = clean_key(r["business_name"])
    if ck and len(ck) >= 4: idx_clean[(ctry, ck)].append(eid)
    sk = clean_sorted_key(r["business_name"])
    if sk: idx_sorted[(ctry, sk)].append(eid)
    for ak in get_addr_keys(r["business_address"]):
        idx_addr[(ctry, ak)].append(eid)

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0

precisions = []
recalls = []
f05s = []

fp_examples = []
fn_examples = []

for sid, true_matches in gt_dict.items():
    if sid not in s1.index: continue
    s1_row = s1.loc[sid]
    ctry = str(s1_row["country"]).upper()
    bname = s1_row["business_name"]
    baddr = s1_row["business_address"]
    
    predicted = set()
    
    # 1. Clean key match
    ck = clean_key(bname)
    if ck and (ctry, ck) in idx_clean:
        for cid in idx_clean[(ctry, ck)][:10]:
            if cid in s23.index and is_compatible_location(baddr, s23.loc[cid, "business_address"]):
                predicted.add(cid)
                
    # 2. Sorted tokens match
    sk = clean_sorted_key(bname)
    if sk and (ctry, sk) in idx_sorted:
        for cid in idx_sorted[(ctry, sk)][:10]:
            if cid in s23.index and is_compatible_location(baddr, s23.loc[cid, "business_address"]):
                predicted.add(cid)
                
    # 3. Address match
    for ak in get_addr_keys(baddr):
        if (ctry, ak) in idx_addr:
            cands = idx_addr[(ctry, ak)]
            if len(cands) <= 8:
                for cid in cands:
                    if cid in s23.index and is_compatible_location(baddr, s23.loc[cid, "business_address"]):
                        predicted.add(cid)
                            
    if not true_matches and not predicted:
        p, r, f = 1.0, 1.0, 1.0
    elif not true_matches and predicted:
        p, r, f = 0.0, 1.0, 0.0
    elif true_matches and not predicted:
        p, r, f = 1.0, 0.0, 0.0
    else:
        tp = len(predicted & true_matches)
        p = tp / len(predicted)
        r = tp / len(true_matches)
        f = f_beta(p, r)
        
    precisions.append(p)
    recalls.append(r)
    f05s.append(f)
    
    # Log errors
    fps = predicted - true_matches
    fns = true_matches - predicted
    if fps and len(fp_examples) < 5:
        fp_examples.append((sid, bname, baddr, fps))
    if fns and len(fn_examples) < 5:
        fn_examples.append((sid, bname, baddr, fns))

print(f"Mean Precision: {np.mean(precisions):.4f}")
print(f"Mean Recall:    {np.mean(recalls):.4f}")
print(f"Mean F0.5:      {np.mean(f05s):.4f}")

print("\n--- SAMPLE FALSE POSITIVES (Precision killers) ---")
for sid, bname, baddr, fps in fp_examples:
    print(f"S1 ({sid}): {bname} | {baddr}")
    for fid in list(fps)[:2]:
        if fid in s23.index:
            print(f"   WRONGLY PREDICTED ({fid}): {s23.loc[fid, 'business_name']} | {s23.loc[fid, 'business_address']}")

print("\n--- SAMPLE FALSE NEGATIVES (Missed recall) ---")
for sid, bname, baddr, fns in fn_examples:
    print(f"S1 ({sid}): {bname} | {baddr}")
    for mid in list(fns)[:2]:
        if mid in s23.index:
            print(f"   MISSED ({mid}): {s23.loc[mid, 'business_name']} | {s23.loc[mid, 'business_address']}")
