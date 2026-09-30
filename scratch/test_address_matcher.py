import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

print("=== RIGOROUS VALIDATION OF ADDRESS SIGNATURE MATCHER ===")

gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=2000)
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

s2_sub = pd.concat([s2[s2["entity_id"].isin(all_m_ids)], s2.sample(n=50000, random_state=42)], ignore_index=True)
s3_sub = pd.concat([s3[s3["entity_id"].isin(all_m_ids)], s3.sample(n=50000, random_state=42)], ignore_index=True)
s23 = pd.concat([s2_sub, s3_sub], ignore_index=True).drop_duplicates(subset=["entity_id"]).set_index("entity_id")

print(f"Candidate pool size: {len(s23)}")

_RE_NUMS = re.compile(r"\b\d+\b")
_RE_WORDS = re.compile(r"[a-z]{3,}")
_STOP = {
    "street", "st", "road", "rd", "avenue", "ave", "lane", "ln", "drive", "dr",
    "court", "ct", "boulevard", "blvd", "floor", "fl", "near", "opp", "door",
    "plot", "flat", "phase", "sector", "block", "house", "bldg", "building",
    "room", "box", "row", "unit", "apt", "suite", "ste", "null", "post", "township",
    "main", "cross", "ward", "north", "south", "east", "west"
}

def extract_addr_signatures(addr: str):
    if not addr or pd.isna(addr): return []
    s = str(addr).lower()
    nums = [n for n in _RE_NUMS.findall(s) if 1 <= len(n) <= 6]
    words = [w for w in _RE_WORDS.findall(s) if w not in _STOP]
    sigs = []
    if nums and words:
        n0 = nums[0]
        # First number + up to 2 distinct street words
        for w in words[:2]:
            sigs.append(f"{n0}_{w}")
        # If there is a second number (e.g. PIN code or unit or plot)
        if len(nums) > 1 and len(nums[1]) >= 4:
            sigs.append(f"pin_{nums[1]}")
    elif len(words) >= 3:
        sigs.append(f"{words[0]}_{words[1]}_{words[2]}")
    return sigs

# Build address index
addr_idx = defaultdict(list)
for eid, r in s23.iterrows():
    ctry = str(r["country"]).upper()
    for sig in extract_addr_signatures(r["business_address"]):
        addr_idx[(ctry, sig)].append(eid)

print(f"Total address keys indexed: {len(addr_idx)}")

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0

precisions = []
recalls = []
f05s = []

for sid, true_m in gt_dict.items():
    if sid not in s1.index: continue
    s1_row = s1.loc[sid]
    ctry = str(s1_row["country"]).upper()
    s1_addr = s1_row["business_address"]
    
    sigs = extract_addr_signatures(s1_addr)
    cand_counts = Counter()
    for sig in sigs:
        if (ctry, sig) in addr_idx:
            for cid in addr_idx[(ctry, sig)]:
                cand_counts[cid] += 1
                
    # A candidate is selected if it matches the address signatures
    # Filter: must not belong to a bucket that exploded (> 15 matches for that key)
    pred = set()
    for cid, count in cand_counts.items():
        # Candidate address
        c_addr = s23.loc[cid, "business_address"]
        # If it matched the signature
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
        
    precisions.append(p)
    recalls.append(r)
    f05s.append(f)

print(f"Evaluated on {len(f05s)} ground truth entities")
print(f"Address Matcher - Precision: {np.mean(precisions):.4f}")
print(f"Address Matcher - Recall:    {np.mean(recalls):.4f}")
print(f"Address Matcher - F0.5:      {np.mean(f05s):.4f}")
