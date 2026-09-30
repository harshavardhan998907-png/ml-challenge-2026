import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

# Run error analysis on eval_fast
from collections import defaultdict

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

s2_sample = pd.concat([s2[s2["entity_id"].isin(all_m_ids)], s2.sample(n=30000, random_state=42)], ignore_index=True)
s3_sample = pd.concat([s3[s3["entity_id"].isin(all_m_ids)], s3.sample(n=30000, random_state=42)], ignore_index=True)
s23 = pd.concat([s2_sample, s3_sample], ignore_index=True).drop_duplicates(subset=["entity_id"]).set_index("entity_id")

_RE_SUFFIXES = re.compile(r"\b(private limited|pvt ltd|pvt\.?|ltd\.?|limited|inc\.?|incorporated|corp\.?|corporation|llc\.?|llp\.?|center|centre|trust|group|company|co\.?)\b", re.IGNORECASE)
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

def get_addr_signature(addr: str):
    if not addr or pd.isna(addr): return ("", "")
    s = str(addr).lower()
    nums = _RE_NUMBERS.findall(s)
    num = nums[0] if nums else ""
    words = _RE_TOKENS.findall(s)
    sig_words = [w for w in words if w not in _STOP_ADDR and not w.isdigit()]
    street = sig_words[0] if sig_words else ""
    return (num, street)

idx_clean = defaultdict(list)
idx_addr = defaultdict(list)

for eid, r in s23.iterrows():
    ctry = str(r["country"]).upper()
    ck = clean_key(r["business_name"])
    if ck and len(ck) >= 4:
        idx_clean[(ctry, ck)].append(eid)
    num, street = get_addr_signature(r["business_address"])
    if num and street:
        idx_addr[(ctry, num, street)].append(eid)

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0

low_scores = []
for sid, true_matches in gt_dict.items():
    if sid not in s1.index: continue
    s1_row = s1.loc[sid]
    ctry = str(s1_row["country"]).upper()
    bname = s1_row["business_name"]
    baddr = s1_row["business_address"]
    ck = clean_key(bname)
    
    predicted = set()
    if (ctry, ck) in idx_clean:
        predicted.update(idx_clean[(ctry, ck)])
        
    num, street = get_addr_signature(baddr)
    if num and street and (ctry, num, street) in idx_addr:
        addr_cands = idx_addr[(ctry, num, street)]
        if len(addr_cands) <= 10:
            predicted.update(addr_cands)
            
    if not true_matches and not predicted:
        score = 1.0
    elif not true_matches and predicted:
        score = 0.0
    elif true_matches and not predicted:
        score = 0.0
    else:
        tp = len(predicted & true_matches)
        p = tp / len(predicted)
        r = tp / len(true_matches)
        score = f_beta(p, r)
        
    if score < 0.5 and true_matches:
        low_scores.append((score, sid, bname, baddr, true_matches, predicted))

print(f"Total low score entities (< 0.5): {len(low_scores)}")
for sc, sid, bname, baddr, tm, pred in low_scores[:5]:
    print(f"\n--- Score: {sc:.2f} | S1 [{sid}]: {bname} | {baddr}")
    print(f"    True matches ({len(tm)}):")
    for m in tm:
        if m in s23.index:
            print(f"      {m}: {s23.loc[m, 'business_name']} | {s23.loc[m, 'business_address']}")
    print(f"    Predicted matches ({len(pred)}):")
    for p in pred:
        if p in s23.index:
            print(f"      {p}: {s23.loc[p, 'business_name']} | {s23.loc[p, 'business_address']}")
