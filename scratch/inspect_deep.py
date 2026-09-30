import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd

gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=10)
s1 = pd.read_csv("dataset/train/train_source1.tsv", sep="\t").set_index("entity_id")
s2 = pd.read_parquet("cache/train_source2_preprocessed.parquet").set_index("entity_id")
s3 = pd.read_parquet("cache/train_source3_preprocessed.parquet").set_index("entity_id")

for idx, r in gt.iterrows():
    sid = r["source1_entity_id"]
    if sid not in s1.index: continue
    s1_row = s1.loc[sid]
    ms = [m.strip() for m in str(r["matched_entity_ids"]).split(",") if m.strip() and m != "nan"]
    print(f"\n==================================================================")
    print(f"[{idx}] S1: {sid} | Name: '{s1_row['business_name']}' | Addr: '{s1_row['business_address']}'")
    for m in ms:
        sub = s2 if m.startswith("S2-") else s3
        if m in sub.index:
            mr = sub.loc[m]
            print(f"     -> {m}: Name: '{mr['business_name']}' | Addr: '{mr['business_address']}'")
