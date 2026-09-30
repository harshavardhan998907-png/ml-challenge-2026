import time
import numpy as np
import pyarrow.parquet as pq

t0 = time.time()
print("Reading S2+S3 with PyArrow...")
t2 = pq.read_table("cache/test_source2_preprocessed.parquet", columns=["entity_id", "country", "norm_name_key"])
t3 = pq.read_table("cache/test_source3_preprocessed.parquet", columns=["entity_id", "country", "norm_name_key"])

import pyarrow as pa
table = pa.concat_tables([t2, t3])
print(f"Loaded {len(table):,} rows in {time.time()-t0:.2f}s")

# Convert to numpy / sort
t1 = time.time()
print("Sorting table by country and norm_name_key...")
# Create a compound key: country + "_" + norm_name_key
s23_ids = table["entity_id"].to_numpy()
s23_ctry = table["country"].to_numpy()
s23_keys = table["norm_name_key"].to_numpy()

# Combine country and key
s23_full_keys = np.char.add(np.char.add(s23_ctry.astype(str), "_"), s23_keys.astype(str))

# Sort
sort_idx = np.argsort(s23_full_keys)
sorted_keys = s23_full_keys[sort_idx]
sorted_ids = s23_ids[sort_idx]
print(f"Sorted 10M keys in {time.time()-t1:.2f}s")

# Test searchsorted with 100,000 queries
test_queries = sorted_keys[:100000]
t2 = time.time()
idx_left = np.searchsorted(sorted_keys, test_queries, side="left")
idx_right = np.searchsorted(sorted_keys, test_queries, side="right")
print(f"100,000 searchsorted queries in {time.time()-t2:.4f}s! ({100000/(time.time()-t2):,.0f} queries/sec)")
