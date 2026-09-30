import pandas as pd
import re

s1 = pd.read_parquet("cache/test_source1_preprocessed.parquet") if False else None
# Read the specific entities
gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=1000)
row = gt[gt["source1_entity_id"] == "S1-686913759"].iloc[0]
print("GT row:", row.to_dict())

s2 = pd.read_parquet("cache/train_source2_preprocessed.parquet")
s3 = pd.read_parquet("cache/train_source3_preprocessed.parquet")

for mid in str(row["matched_entity_ids"]).split(","):
    mid = mid.strip()
    if mid in s2["entity_id"].values:
        print("S2 match:", s2[s2["entity_id"] == mid].to_dict(orient="records")[0])
    if mid in s3["entity_id"].values:
        print("S3 match:", s3[s3["entity_id"] == mid].to_dict(orient="records")[0])
