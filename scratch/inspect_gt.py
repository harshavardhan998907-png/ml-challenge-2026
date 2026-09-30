import pandas as pd
import sys
sys.stdout.reconfigure(encoding='utf-8')

gt = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=1000)
has_match = gt[gt["matched_entity_ids"].notna() & (gt["matched_entity_ids"] != "") & (gt["matched_entity_ids"] != "nan")]
print(f"Total entities: {len(gt)}, with matches: {len(has_match)}")

# Read S1 for these entities
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
    if len(s1_list) * 100000 > 500000:
        break
s1 = pd.concat(s1_list).set_index("entity_id")

s2 = pd.read_parquet("cache/train_source2_preprocessed.parquet")
s2_m = s2[s2["entity_id"].isin(all_m_ids)].set_index("entity_id")

s3 = pd.read_parquet("cache/train_source3_preprocessed.parquet")
s3_m = s3[s3["entity_id"].isin(all_m_ids)].set_index("entity_id")

count = 0
for _, r in has_match.iterrows():
    sid = r["source1_entity_id"]
    if sid not in s1.index:
        continue
    s1_row = s1.loc[sid]
    m_ids = [m.strip() for m in str(r["matched_entity_ids"]).split(",") if m.strip()]
    for mid in m_ids:
        c_row = None
        src = ""
        if mid in s2_m.index:
            c_row = s2_m.loc[mid]
            src = "S2"
        elif mid in s3_m.index:
            c_row = s3_m.loc[mid]
            src = "S3"
        if c_row is not None:
            count += 1
            if count <= 25:
                print(f"=== MATCH {count} [{src}] (Country: {s1_row['country']}) ===")
                print(f"S1  ({sid}): {s1_row['business_name']} ||| {s1_row['business_address']}")
                print(f"MID ({mid}): {c_row['business_name']} ||| {c_row['business_address']}")
                print("-" * 60)
