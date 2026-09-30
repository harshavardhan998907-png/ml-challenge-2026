import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from collections import defaultdict, Counter

print("=== TESTING HIGH-PRECISION COMBINED MATCHER ON 2000 GT ENTITIES ===")

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

s2_sub = pd.concat([s2[s2["entity_id"].isin(all_m_ids)], s2.sample(n=60000, random_state=42)], ignore_index=True)
s3_sub = pd.concat([s3[s3["entity_id"].isin(all_m_ids)], s3.sample(n=60000, random_state=42)], ignore_index=True)
s23 = pd.concat([s2_sub, s3_sub], ignore_index=True).drop_duplicates(subset=["entity_id"]).set_index("entity_id")
print(f"Candidate pool size: {len(s23)}")

# 50 US States + full names
US_STATES = {
    'al', 'ak', 'az', 'ar', 'ca', 'co', 'ct', 'de', 'fl', 'ga', 'hi', 'id', 'il', 'in', 'ia', 'ks',
    'ky', 'la', 'me', 'md', 'ma', 'mi', 'mn', 'ms', 'mo', 'mt', 'ne', 'nv', 'nh', 'nj', 'nm', 'ny',
    'nc', 'nd', 'oh', 'ok', 'or', 'pa', 'ri', 'sc', 'sd', 'tn', 'tx', 'ut', 'vt', 'va', 'wa', 'wv', 'wi', 'wy',
    'alabama', 'alaska', 'arizona', 'arkansas', 'california', 'colorado', 'connecticut', 'delaware', 'florida',
    'georgia', 'hawaii', 'idaho', 'illinois', 'indiana', 'iowa', 'kansas', 'kentucky', 'louisiana', 'maine',
    'maryland', 'massachusetts', 'michigan', 'minnesota', 'mississippi', 'missouri', 'montana', 'nebraska',
    'nevada', 'ohio', 'oklahoma', 'oregon', 'pennsylvania', 'tennessee', 'texas', 'utah', 'vermont', 'virginia',
    'washington', 'wisconsin', 'wyoming'
}

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
    # Unique sorted words
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

def extract_state(addr: str) -> str:
    if not addr: return ""
    tokens = set(re.findall(r"[a-z]+", str(addr).lower()))
    for t in tokens:
        if t in STATE_MAP:
            return STATE_MAP[t]
        if t in US_STATES and len(t) == 2:
            return t
    return ""

def are_addresses_compatible(addr1: str, addr2: str) -> bool:
    if not addr1 or not addr2: return True
    s1_str = str(addr1).lower()
    s2_str = str(addr2).lower()
    if s2_str in ("", "null", "nan"): return True
    
    # State check
    st1 = extract_state(s1_str)
    st2 = extract_state(s2_str)
    if st1 and st2 and st1 != st2:
        return False
        
    # Number check: if both addresses have numbers, do they share at least one?
    nums1 = set(_RE_NUMS.findall(s1_str))
    nums2 = set(_RE_NUMS.findall(s2_str))
    if nums1 and nums2:
        # If both have numbers, and none match, check word overlap
        if not (nums1 & nums2):
            words1 = set(_RE_WORDS.findall(s1_str)) - _STOP_ADDR
            words2 = set(_RE_WORDS.findall(s2_str)) - _STOP_ADDR
            # If word overlap is very low (< 2 words), reject!
            if len(words1 & words2) < 2:
                return False
                
    return True

# Build indices
print("Indexing candidates...")
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

print("Index building complete.")

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
    bname = s1_row["business_name"]
    baddr = s1_row["business_address"]
    
    pred = set()
    
    # 1. Address Signatures (house number + street)
    for sig in extract_addr_signatures(baddr):
        if (ctry, sig) in idx_addr:
            cands = idx_addr[(ctry, sig)]
            if len(cands) <= 8:  # Don't take overcrowded buckets
                for cid in cands:
                    c_addr = s23.loc[cid, "business_address"]
                    if are_addresses_compatible(baddr, c_addr):
                        pred.add(cid)
                        
    # 2. Clean Name Key (with strict address compatibility)
    ck = clean_key(bname)
    if ck and (ctry, ck) in idx_clean:
        cands = idx_clean[(ctry, ck)]
        if len(cands) <= 8:
            for cid in cands:
                c_addr = s23.loc[cid, "business_address"]
                if are_addresses_compatible(baddr, c_addr):
                    pred.add(cid)
                    
    # 3. Sorted Name Key (with strict address compatibility)
    sk = clean_sorted_key(bname)
    if sk and (ctry, sk) in idx_sorted:
        cands = idx_sorted[(ctry, sk)]
        if len(cands) <= 8:
            for cid in cands:
                c_addr = s23.loc[cid, "business_address"]
                if are_addresses_compatible(baddr, c_addr):
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
print(f"High-Precision Matcher - Precision: {np.mean(precisions):.4f}")
print(f"High-Precision Matcher - Recall:    {np.mean(recalls):.4f}")
print(f"High-Precision Matcher - F0.5:      {np.mean(f05s):.4f}")
