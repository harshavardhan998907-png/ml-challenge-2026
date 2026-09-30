import re
import time
import numpy as np
import pyarrow.parquet as pq
import pandas as pd

STOP_WORDS = {
    "private", "limited", "pvt", "ltd", "inc", "corp", "corporation", "llc", "llp",
    "the", "and", "center", "centre", "trust", "group", "services", "service",
    "enterprises", "solutions", "co", "company"
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

print("Loading test sample...")
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
    s2[["entity_id", "country", "business_name", "business_address", "norm_name_key"]],
    s3[["entity_id", "country", "business_name", "business_address", "norm_name_key"]]
], ignore_index=True)

# Build compound keys
# Key 1: country_norm_name_key
k1 = s23["country"].fillna("") + "_" + s23["norm_name_key"].fillna("")
# Key 2: country_clean_key (without domain / without spaces)
k2 = s23["country"].fillna("") + "_" + s23["business_name"].apply(clean_key)
# Key 3: country_tokenpair
def make_token_key(row):
    toks = get_tokens(row["business_name"])
    if len(toks) >= 2:
        return f"{row['country']}_{toks[0]}_{toks[1]}"
    elif len(toks) == 1:
        return f"{row['country']}_{toks[0]}"
    return ""

k3 = s23.apply(make_token_key, axis=1)

# Build sorted arrays for k1, k2, k3
def build_sorted_index(keys_series, ids_series):
    valid = keys_series.str.len() > 3
    sub_keys = keys_series[valid].to_numpy()
    sub_ids = ids_series[valid].to_numpy()
    idx = np.argsort(sub_keys)
    return sub_keys[idx], sub_ids[idx]

print("Building indices...")
t0 = time.time()
arr_k1, arr_id1 = build_sorted_index(k1, s23["entity_id"])
arr_k2, arr_id2 = build_sorted_index(k2, s23["entity_id"])
arr_k3, arr_id3 = build_sorted_index(k3, s23["entity_id"])
print(f"Indices built in {time.time()-t0:.2f}s")

def query_index(arr_keys, arr_ids, query, max_matches=5):
    l = np.searchsorted(arr_keys, query, side="left")
    r = np.searchsorted(arr_keys, query, side="right")
    if l < r:
        n = min(r - l, max_matches)
        return arr_ids[l:l+n]
    return np.array([])

hits = 0
total_links = 0
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
    bname = s1_row["business_name"]
    
    q1 = f"{ctry}_{s1_row['norm_name_key']}"
    q2 = f"{ctry}_{clean_key(bname)}"
    toks = get_tokens(bname)
    q3 = f"{ctry}_{toks[0]}_{toks[1]}" if len(toks) >= 2 else (f"{ctry}_{toks[0]}" if toks else "")
    
    cands = set()
    cands.update(query_index(arr_k1, arr_id1, q1, max_matches=5))
    cands.update(query_index(arr_k2, arr_id2, q2, max_matches=5))
    if q3:
        cands.update(query_index(arr_k3, arr_id3, q3, max_matches=3))
        
    total_cands += len(cands)
    for m in raw.split(","):
        m = m.strip()
        if not m:
            continue
        total_links += 1
        if m in cands:
            hits += 1

print(f"Total ground truth links checked: {total_links}")
print(f"Hits: {hits} ({hits/total_links*100:.1f}%)")
print(f"Avg candidates per entity: {total_cands / len(s1_needed):.2f}")
