import re
import sys
import time
import gc
from collections import defaultdict
import pyarrow.parquet as pq
import pandas as pd

STOP_WORDS = {
    "private", "limited", "pvt", "ltd", "inc", "corp", "corporation", "llc", "llp",
    "the", "and", "center", "centre", "trust", "group", "services", "service",
    "enterprises", "solutions", "co", "company", "assoc", "associates"
}

def clean_key(name):
    if not name or pd.isna(name):
        return ""
    s = str(name).lower()
    for suf in ["private limited", "pvt ltd", "ltd", "inc", "llp", "corp", "llc", "center", "trust", "group"]:
        s = re.sub(rf"\b{suf}\b", "", s)
    s = re.sub(r"\.com|\.org|\.in|\.net|\.fr|\.co\b|www\.", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def get_tokens(name):
    if not name or pd.isna(name):
        return []
    s = str(name).lower()
    words = re.findall(r"[a-z0-9]{3,}", s)
    return [w for w in words if w not in STOP_WORDS]

def get_first_num(addr):
    if not addr or pd.isna(addr):
        return ""
    nums = re.findall(r"\b\d{2,6}\b", str(addr))
    return nums[0] if nums else ""

t0 = time.time()
print("Reading Test S2 and S3 parquet columns...")
cols = ["entity_id", "country", "business_name", "business_address", "norm_name_key"]
t2 = pq.read_table("cache/test_source2_preprocessed.parquet", columns=cols).to_pandas()
t3 = pq.read_table("cache/test_source3_preprocessed.parquet", columns=cols).to_pandas()
s23 = pd.concat([t2, t3], ignore_index=True)
del t2, t3
gc.collect()
print(f"Loaded {len(s23):,} S2+S3 rows in {time.time()-t0:.2f}s")

# Test on France first (fastest partition)
print("\n--- Testing France Partition ---")
tf0 = time.time()
s23_fr = s23[s23["country"] == "France"]
print(f"France S2+S3 rows: {len(s23_fr):,}")

# Build index for France
idx_norm = defaultdict(list)
idx_clean = defaultdict(list)
idx_num = defaultdict(list)

for r in s23_fr.itertuples(index=False):
    eid = r.entity_id
    if r.norm_name_key and len(r.norm_name_key) >= 3:
        idx_norm[r.norm_name_key].append(eid)
    ck = clean_key(r.business_name)
    if ck and len(ck) >= 4:
        idx_clean[ck].append(eid)
    num = get_first_num(r.business_address)
    toks = get_tokens(r.business_name)
    if num and toks:
        idx_num[(num, toks[0])].append(eid)

print(f"France indices built in {time.time()-tf0:.2f}s")
print(f"  Norm keys: {len(idx_norm):,}, Clean keys: {len(idx_clean):,}, Num keys: {len(idx_num):,}")

# Load France S1
s1 = pd.read_csv("dataset/test/test_source1.tsv", sep="\t")
s1_fr = s1[s1["country"] == "France"]
print(f"France S1 rows: {len(s1_fr):,}")

# Query
tq0 = time.time()
total_matches = 0
singletons = 0
for r in s1_fr.itertuples(index=False):
    cands = set()
    bname = r.business_name
    # 1. norm key
    nk = r.norm_name_key if hasattr(r, "norm_name_key") else clean_key(bname)
    if nk in idx_norm:
        cands.update(idx_norm[nk][:5])
    # 2. clean key
    ck = clean_key(bname)
    if ck in idx_clean:
        cands.update(idx_clean[ck][:5])
    # 3. num + token
    num = get_first_num(r.business_address)
    toks = get_tokens(bname)
    if num and toks:
        k = (num, toks[0])
        if k in idx_num:
            cands.update(idx_num[k][:3])
            
    if cands:
        total_matches += len(cands)
    else:
        singletons += 1

q_time = time.time() - tq0
print(f"Queried {len(s1_fr):,} entities in {q_time:.2f}s ({len(s1_fr)/q_time:,.0f} entities/sec)")
print(f"  Total candidate links found: {total_matches:,}")
print(f"  Singletons: {singletons:,} ({singletons/len(s1_fr)*100:.1f}%)")
