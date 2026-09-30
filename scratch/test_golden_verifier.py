import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
import numpy as np
import pandas as pd
from collections import defaultdict
from rapidfuzz import fuzz

print("=== TESTING THE GOLDEN MATCH VERIFICATION SYSTEM ===")

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

s2_sub = pd.concat([s2[s2["entity_id"].isin(all_m_ids)], s2.sample(n=50000, random_state=42)], ignore_index=True)
s3_sub = pd.concat([s3[s3["entity_id"].isin(all_m_ids)], s3.sample(n=50000, random_state=42)], ignore_index=True)
s23 = pd.concat([s2_sub, s3_sub], ignore_index=True).drop_duplicates(subset=["entity_id"]).set_index("entity_id")

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
            try: nums.append(str(int(n)))
            except: nums.append(n)
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

def extract_state(addr: str) -> str:
    if not addr: return ""
    tokens = set(re.findall(r"[a-z]+", str(addr).lower()))
    for t in tokens:
        if t in STATE_MAP: return STATE_MAP[t]
        if t in STATE_MAP.values() and len(t) == 2: return t
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

def is_verified_match(s1_name: str, s1_addr: str, c_name: str, c_addr: str) -> bool:
    n1 = str(s1_name).lower()
    n2 = str(c_name).lower()
    a1 = str(s1_addr).lower()
    a2 = str(c_addr).lower()
    
    # Check address compatibility first
    if not are_addresses_compatible(a1, a2):
        return False
        
    # Rule 1: Exact Name Key Match (clean alphanumeric or sorted tokens)
    ck1 = clean_key(n1)
    ck2 = clean_key(n2)
    if ck1 and ck2 and ck1 == ck2 and len(ck1) >= 4:
        return True
        
    sk1 = clean_sorted_key(n1)
    sk2 = clean_sorted_key(n2)
    if sk1 and sk2 and sk1 == sk2 and len(sk1) >= 4:
        return True
        
    # Rule 2: High Name Similarity + Compatible Location
    toks1 = set(_RE_WORDS.findall(n1)) - STOP_WORDS
    toks2 = set(_RE_WORDS.findall(n2)) - STOP_WORDS
    name_jaccard = len(toks1 & toks2) / len(toks1 | toks2) if (toks1 and toks2) else 0.0
    token_set = fuzz.token_set_ratio(n1, n2)
    
    if token_set >= 90 and name_jaccard >= 0.4:
        return True
        
    # Rule 3: Physical Address Match (Exact house number + 2 street words)
    raw_nums1 = set(str(int(n)) for n in _RE_NUMS.findall(a1) if 1 <= len(n) <= 6)
    raw_nums2 = set(str(int(n)) for n in _RE_NUMS.findall(a2) if 1 <= len(n) <= 6)
    words1 = set(_RE_WORDS.findall(a1)) - _STOP_ADDR
    words2 = set(_RE_WORDS.findall(a2)) - _STOP_ADDR
    
    shared_nums = bool(raw_nums1 & raw_nums2)
    shared_street_count = len(words1 & words2)
    
    # If same house number AND at least 2 matching address tokens:
    if shared_nums and shared_street_count >= 2:
        # If candidate has Indian characters or shares at least one name word or token_set >= 70
        has_non_ascii = any(ord(c) > 127 for c in c_name)
        if has_non_ascii or name_jaccard >= 0.25 or token_set >= 65:
            return True
            
    # Rule 4: Transliterated Indian business with same street & house number
    if shared_nums and shared_street_count >= 1:
        has_indian_script = any(ord(c) >= 0x0900 for c in c_name)
        if has_indian_script:
            return True
            
    return False

# Build candidate index
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

print("Testing Golden Verifier across ground truth entities...")

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0

precisions, recalls, f05s = [], [], []

for sid, true_m in gt_dict.items():
    if sid not in s1.index: continue
    s1_row = s1.loc[sid]
    ctry = str(s1_row["country"]).upper()
    bname = s1_row["business_name"]
    baddr = s1_row["business_address"]
    
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
            
    pred = set()
    for cid in cand_set:
        c_name = s23.loc[cid, "business_name"]
        c_addr = s23.loc[cid, "business_address"]
        if is_verified_match(bname, baddr, c_name, c_addr):
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
print(f"GOLDEN VERIFIER - Precision: {np.mean(precisions):.4f} ({np.mean(precisions)*100:.2f}%)")
print(f"GOLDEN VERIFIER - Recall:    {np.mean(recalls):.4f} ({np.mean(recalls)*100:.2f}%)")
print(f"GOLDEN VERIFIER - Macro F0.5: {np.mean(f05s):.4f}")
