import re
import sys
import time
from collections import defaultdict
import pandas as pd

STOP_WORDS = {
    "private", "limited", "pvt", "ltd", "inc", "incorporated", "corp", "corporation",
    "llc", "llp", "co", "company", "center", "centre", "trust", "group", "services",
    "service", "enterprises", "enterprise", "solutions", "international", "india", "us",
    "france", "the", "and", "for", "with", "associates", "consulting", "industries"
}

def clean_key(name):
    if not name or pd.isna(name):
        return ""
    s = str(name).lower()
    for suf in ["private limited", "pvt ltd", "ltd", "inc", "llp", "corp", "llc", "center", "trust", "group"]:
        s = re.sub(rf"\b{suf}\b", "", s)
    s = re.sub(r"\.com|\.org|\.in|\.net|\.fr|\.co\b", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def get_tokens(name):
    if not name or pd.isna(name):
        return set()
    s = str(name).lower()
    words = re.findall(r"[a-z0-9]{3,}", s)
    return set(w for w in words if w not in STOP_WORDS)

def get_addr_numbers(addr):
    if not addr or pd.isna(addr):
        return []
    return re.findall(r"\b\d{2,6}\b", str(addr))

print("Loading validation subset...")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=5000)
s1_needed = set(gt["source1_entity_id"])

s1_chunks = []
for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", chunksize=100000):
    sub = chunk[chunk["entity_id"].isin(s1_needed)]
    if not sub.empty:
        s1_chunks.append(sub)
    if len(s1_chunks) * 100000 > 500000:
        break

s1 = pd.concat(s1_chunks).set_index("entity_id")
s2 = pd.read_parquet("cache/train_source2_preprocessed.parquet")
s3 = pd.read_parquet("cache/train_source3_preprocessed.parquet")

s23 = pd.concat([
    s2[["entity_id", "business_name", "business_address", "country", "norm_name_key", "zip_code"]],
    s3[["entity_id", "business_name", "business_address", "country", "norm_name_key", "zip_code"]]
], ignore_index=True)

print(f"S23 total records: {len(s23):,}")

t0 = time.time()
print("Building fast inverted indices...")
# 1. Exact clean name key index
name_index = defaultdict(list)
# 2. Token pair index
token_index = defaultdict(list)
# 3. Number index (first address number)
num_index = defaultdict(list)

for idx, r in enumerate(s23.itertuples(index=False)):
    eid = r.entity_id
    ctry = r.country
    
    nk = clean_key(r.business_name)
    if nk:
        name_index[(ctry, nk)].append(eid)
        
    toks = sorted(get_tokens(r.business_name))
    if len(toks) >= 2:
        # index first pair
        token_index[(ctry, toks[0], toks[1])].append(eid)
    elif len(toks) == 1:
        token_index[(ctry, toks[0], "")].append(eid)
        
    nums = get_addr_numbers(r.business_address)
    if nums:
        # index first number + first token
        t0_w = toks[0] if toks else ""
        if t0_w:
            num_index[(ctry, nums[0], t0_w)].append(eid)

print(f"Indices built in {time.time()-t0:.1f}s")

# Query S1
t1 = time.time()
hits = 0
total_gt_links = 0
total_cands = 0

for _, r in gt.iterrows():
    sid = r["source1_entity_id"]
    if sid not in s1.index:
        continue
    raw = str(r["matched_entity_ids"])
    if not raw or raw == "nan":
        continue
        
    s1_row = s1.loc[sid]
    ctry = s1_row["country"]
    cands = set()
    
    # 1. Name key lookup
    nk = clean_key(s1_row["business_name"])
    if nk:
        cands.update(name_index.get((ctry, nk), []))
        
    # 2. Token index lookup
    toks = sorted(get_tokens(s1_row["business_name"]))
    if len(toks) >= 2:
        cands.update(token_index.get((ctry, toks[0], toks[1]), []))
    elif len(toks) == 1:
        cands.update(token_index.get((ctry, toks[0], ""), []))
        
    # 3. Number + token lookup
    nums = get_addr_numbers(s1_row["business_address"])
    if nums and toks:
        cands.update(num_index.get((ctry, nums[0], toks[0]), []))
        
    total_cands += len(cands)
    
    for m in raw.split(","):
        m = m.strip()
        if not m:
            continue
        total_gt_links += 1
        if m in cands:
            hits += 1

q_time = time.time() - t1
print(f"Query completed in {q_time:.2f}s")
print(f"Total ground truth links checked: {total_gt_links}")
print(f"Hits: {hits} ({hits/total_gt_links*100:.1f}%)")
print(f"Avg candidates per S1 entity: {total_cands / max(1, len(s1_needed)):.2f}")
