import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np
from collections import defaultdict

print("=== TESTING REFINED BLOCKING STRATEGY ===")

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

s2_m = s2[s2["entity_id"].isin(all_m_ids)].set_index("entity_id")
s3_m = s3[s3["entity_id"].isin(all_m_ids)].set_index("entity_id")
s23 = pd.concat([s2_m, s3_m])

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
    # Normalize numbers (strip leading zeros)
    nums = []
    for n in raw_nums:
        if 1 <= len(n) <= 6:
            try:
                nums.append(str(int(n)))
            except:
                nums.append(n)
                
    words = [w for w in _RE_WORDS.findall(s) if w not in _STOP_ADDR]
    sigs = []
    
    # 1. Number + street words
    for n in nums[:3]:
        for w in words[:3]:
            sigs.append(f"{n}_{w}")
            
    # 2. Street bigram
    if len(words) >= 2:
        sigs.append(f"st_{words[0]}_{words[1]}")
        
    # 3. Postal code (5 or 6 digits)
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
    # Rare words (length >= 8)
    words = [w for w in _RE_WORDS.findall(str(r["business_name"]).lower()) if w not in STOP_WORDS and len(w) >= 8]
    for rw in words[:2]:
        idx_rare[(ctry, rw)].append(eid)

rec_list = []
total_cands = 0
for sid, true_m in gt_dict.items():
    if sid not in s1.index or not true_m: continue
    s1_row = s1.loc[sid]
    ctry = str(s1_row["country"]).upper()
    bname = s1_row["business_name"]
    baddr = s1_row["business_address"]
    
    cand_set = set()
    ck = clean_key(bname)
    if (ctry, ck) in idx_clean: cand_set.update(idx_clean[(ctry, ck)])
    sk = clean_sorted_key(bname)
    if (ctry, sk) in idx_sorted: cand_set.update(idx_sorted[(ctry, sk)])
    for npk in get_name_pairs(bname):
        if (ctry, npk) in idx_name_pair: cand_set.update(idx_name_pair[(ctry, npk)])
    for sig in extract_addr_signatures(baddr):
        if (ctry, sig) in idx_addr: cand_set.update(idx_addr[(ctry, sig)])
    words = [w for w in _RE_WORDS.findall(str(bname).lower()) if w not in STOP_WORDS and len(w) >= 8]
    for rw in words[:2]:
        if (ctry, rw) in idx_rare and len(idx_rare[(ctry, rw)]) <= 5:
            cand_set.update(idx_rare[(ctry, rw)])
        
    rec = len(cand_set & true_m) / len(true_m)
    rec_list.append(rec)
    total_cands += len(cand_set)

print(f"Total entities evaluated: {len(rec_list)}")
print(f"Mean candidates per entity: {total_cands / len(rec_list):.1f}")
print(f"REFINED BLOCKING RECALL: {np.mean(rec_list):.4f} ({np.mean(rec_list)*100:.2f}%)")
