import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=20)
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
s2 = pd.read_parquet("cache/train_source2_preprocessed.parquet").set_index("entity_id")
s3 = pd.read_parquet("cache/train_source3_preprocessed.parquet").set_index("entity_id")

for i in range(8):
    row = gt.iloc[i]
    sid = row["source1_entity_id"]
    s1_sub = s1[s1["entity_id"] == sid]
    if s1_sub.empty:
        continue
    s1_row = s1_sub.iloc[0]
    m_list = [m for m in str(row["matched_entity_ids"]).split(",") if m and m != "nan"]
    print(f"\n=======================================================")
    print(f"S1 [{sid}]: {s1_row['business_name']}")
    print(f"    Address: {s1_row['business_address']}")
    print(f"    Country: {s1_row['country']}")
    print(f"--- {len(m_list)} True Matches ---")
    for m in m_list:
        sub = s2 if m.startswith("S2-") else s3
        if m in sub.index:
            r = sub.loc[m]
            print(f"  [{m}] Name:    {r['business_name']}")
            print(f"         Address: {r['business_address']}")
