import re
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from collections import defaultdict

gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=2000)
s1_needed = set(gt["source1_entity_id"])

s1_chunks = []
for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", chunksize=100000):
    sub = chunk[chunk["entity_id"].isin(s1_needed)]
    if not sub.empty:
        s1_chunks.append(sub)
    if len(s1_chunks) * 100000 > 500000:
        break

s1 = pd.concat(s1_chunks).set_index("entity_id")
s2 = pd.read_parquet("cache/train_source2_preprocessed.parquet").set_index("entity_id")
s3 = pd.read_parquet("cache/train_source3_preprocessed.parquet").set_index("entity_id")

def clean_key(name):
    if not name or pd.isna(name): return ""
    s = str(name).lower()
    s = re.sub(r"\b(private limited|pvt ltd|pvt|ltd|limited|inc|corp|corporation|llc|llp|center|trust|group|company|co)\b", "", s)
    s = re.sub(r"\.(com|org|in|net|fr|co)\b|www\.", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def extract_addr_tokens(addr):
    if not addr or pd.isna(addr): return set()
    s = str(addr).lower()
    # clean abbreviations
    s = re.sub(r"\b(street|st|road|rd|avenue|ave|court|ct|lane|ln|drive|dr|boulevard|blvd|floor|fl|near|opp|door|no|plot|flat)\b", " ", s)
    tokens = re.findall(r"[a-z0-9]{2,}", s)
    return set(tokens)

total_links = 0
name_matched = 0
addr_matched = 0
either_matched = 0

for _, r in gt.iterrows():
    sid = r["source1_entity_id"]
    if sid not in s1.index: continue
    s1_row = s1.loc[sid]
    s1_nk = clean_key(s1_row["business_name"])
    s1_atoks = extract_addr_tokens(s1_row["business_address"])
    
    raw = str(r["matched_entity_ids"])
    if not raw or raw == "nan": continue
    for m in raw.split(","):
        m = m.strip()
        if not m: continue
        sub = s2 if m.startswith("S2-") else s3
        if m not in sub.index: continue
        total_links += 1
        m_row = sub.loc[m]
        m_nk = clean_key(m_row["business_name"])
        m_atoks = extract_addr_tokens(m_row["business_address"])
        
        # Name match: clean_key match or substring
        hit_n = bool(s1_nk and m_nk and (s1_nk == m_nk or (len(s1_nk) >= 6 and (s1_nk in m_nk or m_nk in s1_nk))))
        # Address match: sharing >= 2 significant address tokens (e.g. house number + street)
        common_addr = s1_atoks & m_atoks
        hit_a = len(common_addr) >= 2 or (len(common_addr) == 1 and any(t.isdigit() and len(t) >= 2 for t in common_addr))
        
        if hit_n: name_matched += 1
        if hit_a: addr_matched += 1
        if hit_n or hit_a: either_matched += 1

print(f"Total evaluated ground truth links: {total_links}")
print(f"Name matched:   {name_matched} ({name_matched/total_links*100:.1f}%)")
print(f"Address matched: {addr_matched} ({addr_matched/total_links*100:.1f}%)")
print(f"EITHER matched (Combined Coverage): {either_matched} ({either_matched/total_links*100:.1f}%)")
