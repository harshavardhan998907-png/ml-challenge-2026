import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import re

print("Loading train_source1.tsv...")
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t").set_index("entity_id")

print("Loading train_ground_truth.tsv...")
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=100)

all_m_ids = set()
for m in gt["matched_entity_ids"]:
    for mid in str(m).split(","):
        if mid.strip():
            all_m_ids.add(mid.strip())

print("Loading cached train_source2 and train_source3...")
s2 = pd.read_parquet("cache/train_source2_preprocessed.parquet")
s2_m = s2[s2["entity_id"].isin(all_m_ids)].set_index("entity_id")
s3 = pd.read_parquet("cache/train_source3_preprocessed.parquet")
s3_m = s3[s3["entity_id"].isin(all_m_ids)].set_index("entity_id")
s23 = pd.concat([s2_m, s3_m])

print("Examining 15 consecutive entities and ALL their matches:")
for idx, r in gt.iloc[:15].iterrows():
    sid = r["source1_entity_id"]
    if sid not in s1.index: continue
    s1_row = s1.loc[sid]
    raw_m = str(r["matched_entity_ids"])
    ms = [m.strip() for m in raw_m.split(",") if m.strip() and m != "nan"]
    print(f"\n==================================================")
    print(f"[{s1_row['country']}] S1 ({sid}):")
    print(f"  NAME:    {s1_row['business_name']}")
    print(f"  ADDRESS: {s1_row['business_address']}")
    print(f"  MATCHES ({len(ms)}):")
    for mid in ms:
        if mid in s23.index:
            c = s23.loc[mid]
            print(f"    -> [{mid}]")
            print(f"       NAME: {c['business_name']}")
            print(f"       ADDR: {c['business_address']}")
        else:
            print(f"    -> [{mid}] (not in cache)")
