import pandas as pd
import re
from collections import Counter

gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=2000)
has_match = gt[gt["matched_entity_ids"].notna() & (gt["matched_entity_ids"] != "") & (gt["matched_entity_ids"] != "nan")]

needed_s1 = set(has_match["source1_entity_id"])
all_m_ids = set()
for m in has_match["matched_entity_ids"]:
    for mid in str(m).split(","):
        if mid.strip():
            all_m_ids.add(mid.strip())

s1_list = []
for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", chunksize=100000):
    sub = chunk[chunk["entity_id"].isin(needed_s1)]
    if not sub.empty:
        s1_list.append(sub)
    if len(s1_list) * 100000 > 600000:
        break
s1 = pd.concat(s1_list).set_index("entity_id")

s2 = pd.read_parquet("cache/train_source2_preprocessed.parquet")
s2_m = s2[s2["entity_id"].isin(all_m_ids)].set_index("entity_id")
s3 = pd.read_parquet("cache/train_source3_preprocessed.parquet")
s3_m = s3[s3["entity_id"].isin(all_m_ids)].set_index("entity_id")
s23 = pd.concat([s2_m, s3_m])

stats = Counter()
country_stats = {c: Counter() for c in ["France", "US", "India"]}

for _, r in has_match.iterrows():
    sid = r["source1_entity_id"]
    if sid not in s1.index: continue
    s1_row = s1.loc[sid]
    ctry = s1_row["country"]
    c_counter = country_stats.get(ctry, stats)
    
    m_ids = [m.strip() for m in str(r["matched_entity_ids"]).split(",") if m.strip()]
    for mid in m_ids:
        if mid not in s23.index: continue
        c_row = s23.loc[mid]
        
        # S1 tokens
        name1 = str(s1_row["business_name"]).lower()
        addr1 = str(s1_row["business_address"]).lower()
        name2 = str(c_row["business_name"]).lower()
        addr2 = str(c_row["business_address"]).lower()
        
        # Name numbers
        nums1 = set(re.findall(r"\d+", addr1))
        nums2 = set(re.findall(r"\d+", addr2))
        
        # Name clean
        n_clean1 = re.sub(r"[^a-z0-9]", "", name1)
        n_clean2 = re.sub(r"[^a-z0-9]", "", name2)
        
        # Token overlap
        name_toks1 = set(re.findall(r"[a-z0-9]{3,}", name1))
        name_toks2 = set(re.findall(r"[a-z0-9]{3,}", name2))
        addr_toks1 = set(re.findall(r"[a-z0-9]{3,}", addr1))
        addr_toks2 = set(re.findall(r"[a-z0-9]{3,}", addr2))
        
        exact_clean_name = (n_clean1 == n_clean2 and len(n_clean1) >= 3)
        name_jaccard = len(name_toks1 & name_toks2) / len(name_toks1 | name_toks2) if (name_toks1 | name_toks2) else 0.0
        addr_jaccard = len(addr_toks1 & addr_toks2) / len(addr_toks1 | addr_toks2) if (addr_toks1 | addr_toks2) else 0.0
        shared_nums = len(nums1 & nums2) > 0 if (nums1 and nums2) else False
        empty_addr2 = (addr2 == "" or addr2 == "nan")
        
        c_counter["total_pairs"] += 1
        if exact_clean_name: c_counter["exact_clean_name"] += 1
        if name_jaccard >= 0.5: c_counter["name_jaccard_gte_0.5"] += 1
        if name_jaccard >= 0.3: c_counter["name_jaccard_gte_0.3"] += 1
        if addr_jaccard >= 0.4: c_counter["addr_jaccard_gte_0.4"] += 1
        if addr_jaccard >= 0.2: c_counter["addr_jaccard_gte_0.2"] += 1
        if shared_nums: c_counter["shared_numbers"] += 1
        if empty_addr2: c_counter["empty_addr2"] += 1
        if (name_jaccard >= 0.3 or exact_clean_name) and (addr_jaccard >= 0.2 or empty_addr2 or shared_nums):
            c_counter["name_AND_addr_compatible"] += 1

print("=== STATS BY COUNTRY ===")
for c, c_counter in country_stats.items():
    tot = c_counter["total_pairs"]
    if tot == 0: continue
    print(f"\n--- {c} (Total true pairs: {tot}) ---")
    for k, v in c_counter.items():
        if k != "total_pairs":
            print(f"  {k:28s}: {v:5d} ({v/tot*100:.1f}%)")
