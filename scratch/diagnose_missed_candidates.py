import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
import re
import pandas as pd
from collections import defaultdict

print("=== FORENSIC ANALYSIS: WHY ARE 13.4% OF TRUE MATCHES MISSED IN BLOCKING? ===")

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

missed = []
for sid, true_m in gt_dict.items():
    if sid not in s1.index: continue
    s1_row = s1.loc[sid]
    ctry = str(s1_row["country"]).upper()
    bname = s1_row["business_name"]
    baddr = s1_row["business_address"]
    
    cand_set = set()
    ck = clean_key(bname)
    if (ctry, ck) in idx_clean: cand_set.update(idx_clean[(ctry, ck)])
    sk = clean_sorted_key(bname)
    if (ctry, sk) in idx_sorted: cand_set.update(idx_sorted[(ctry, sk)])
    for sig in extract_addr_signatures(baddr):
        if (ctry, sig) in idx_addr: cand_set.update(idx_addr[(ctry, sig)])
        
    for mid in true_m:
        if mid not in cand_set and mid in s23.index:
            missed.append((sid, mid, s1_row, s23.loc[mid]))

print(f"Total missed true matches in candidate pool: {len(missed)}")
print("\n--- FIRST 15 MISSED TRUE MATCHES ---")
for sid, mid, s1_r, c_r in missed[:15]:
    print(f"\nS1 ({sid}, {s1_r['country']}):")
    print(f"  NAME: {s1_r['business_name']}")
    print(f"  ADDR: {s1_r['business_address']}")
    print(f"MISSED MATCH ({mid}):")
    print(f"  NAME: {c_r['business_name']}")
    print(f"  ADDR: {c_r['business_address']}")
    print(f"  S1 addr sigs: {extract_addr_signatures(s1_r['business_address'])}")
    print(f"  C  addr sigs: {extract_addr_signatures(c_r['business_address'])}")
    print(f"  S1 clean key: {clean_key(s1_r['business_name'])}")
    print(f"  C  clean key: {clean_key(c_r['business_name'])}")
    print(f"  S1 sort key:  {clean_sorted_key(s1_r['business_name'])}")
    print(f"  C  sort key:  {clean_sorted_key(c_r['business_name'])}")
