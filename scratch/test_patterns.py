import sys
sys.path.insert(0, "code/business_entity_resolution/src")
import pandas as pd
from preprocess import normalise_name_key, normalise_name

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
s1["norm_name_key"] = s1["business_name"].apply(normalise_name_key)
s1["norm_name"] = s1["business_name"].apply(normalise_name)

s2 = pd.read_parquet("cache/train_source2_preprocessed.parquet").set_index("entity_id")
s3 = pd.read_parquet("cache/train_source3_preprocessed.parquet").set_index("entity_id")

printed = 0
for _, r in gt.iterrows():
    sid = r["source1_entity_id"]
    if sid not in s1.index:
        continue
    k1 = s1.loc[sid, "norm_name_key"]
    n1 = s1.loc[sid, "norm_name"]
    toks1 = set(k1.split()) if k1 else set()
    raw = str(r["matched_entity_ids"])
    if not raw or raw == "nan":
        continue
    for m in raw.split(","):
        m = m.strip()
        if not m:
            continue
        sub = s2 if m.startswith("S2-") else s3
        if m not in sub.index:
            continue
        k2 = sub.loc[m, "norm_name_key"]
        n2 = sub.loc[m, "norm_name"]
        toks2 = set(k2.split()) if k2 else set()
        
        jaccard = len(toks1 & toks2) / len(toks1 | toks2) if (toks1 and toks2) else 0.0
        if jaccard < 0.3:
            printed += 1
            print(f"[{printed}] Low Jaccard ({jaccard:.2f}):")
            print(f"  S1: {s1.loc[sid, 'business_name']} | {s1.loc[sid, 'business_address']}")
            print(f"  M:  {sub.loc[m, 'business_name']} | {sub.loc[m, 'business_address']}")
            if printed >= 5:
                break
    if printed >= 5:
        break
