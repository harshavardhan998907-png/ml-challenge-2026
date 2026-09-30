import os
import sys
import time
import re
import pyarrow.parquet as pq
import pandas as pd

print("=== Generating Enhanced Submission (0.682 + 228k Exact Clean Name Matches) ===")
t0 = time.time()

# 1. Candidate lookup
print("1. Reading candidate pairs lookup...")
cand_lookup = {}
with open("output/candidate_pairs.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        sid = parts[0]
        cands = set(parts[1].split(",")) if (len(parts) == 2 and parts[1]) else set()
        cand_lookup[sid] = cands

# 2. Entity names
print("2. Loading entity names...")
cols = ["entity_id", "business_name"]
s1_df = pd.read_parquet("cache/test_source1_preprocessed.parquet", columns=cols)
_RE_ALPHANUM = re.compile(r"[^a-z0-9]")
def clean_k(s):
    return _RE_ALPHANUM.sub("", str(s).lower())

s1_ck = dict(zip(s1_df["entity_id"], s1_df["business_name"].apply(clean_k)))
del s1_df

t2 = pq.read_table("cache/test_source2_preprocessed.parquet", columns=cols).to_pandas()
t3 = pq.read_table("cache/test_source3_preprocessed.parquet", columns=cols).to_pandas()
s23 = pd.concat([t2, t3], ignore_index=True)
s23_ck = dict(zip(s23["entity_id"], s23["business_name"].apply(clean_k)))
del t2, t3, s23

# 3. Load baseline 0.682 matches
print("3. Loading 0.682 matches...")
matches = {}
used_cands = set()
with open("output/backup_0682/matching_results.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        sid = parts[0]
        ms = [m.strip() for m in parts[1].split(",") if m.strip()] if (len(parts) == 2 and parts[1]) else []
        matches[sid] = ms
        for m in ms:
            used_cands.add(m)

base_count = len(used_cands)
print(f"   Loaded {base_count:,} existing 0.682 matches")

# 4. Augment with ONLY exact clean name matches from backup_0380
print("4. Augmenting with pure EXACT clean name matches...")
added_count = 0
with open("output/backup_0380/matching_results.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        sid = parts[0]
        if len(parts) < 2 or not parts[1]:
            continue
        s1_c = s1_ck.get(sid, "")
        if len(s1_c) < 5:  # Require meaningful length
            continue

        valid_cands = cand_lookup.get(sid, set())
        cur_ms = matches.get(sid, [])
        room = 4 - len(cur_ms)  # Top-4 cap
        if room <= 0:
            continue

        for cid in parts[1].split(","):
            cid = cid.strip()
            if cid and cid not in used_cands and cid in valid_cands:
                c_c = s23_ck.get(cid, "")
                # STRICT EXACT CLEAN NAME MATCH
                if s1_c == c_c:
                    used_cands.add(cid)
                    cur_ms.append(cid)
                    added_count += 1
                    room -= 1
                    if room == 0:
                        break

print(f"   Added {added_count:,} strictly verified EXACT name matches!")

# 5. Write to output/matching_results_enhanced.tsv
out_path = "output/matching_results_enhanced.tsv"
print(f"5. Writing output to {out_path}...")
total_links = 0
total_singletons = 0

with open(out_path, "w", encoding="utf-8") as f_out:
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in cand_lookup.keys():
        ms = matches.get(sid, [])
        match_str = ",".join(ms)
        f_out.write(f"{sid}\t{match_str}\n")
        if ms:
            total_links += len(ms)
        else:
            total_singletons += 1

print(f"\n=== GENERATION COMPLETE in {time.time()-t0:.1f}s ===")
print(f"Total S1 entities:    {len(cand_lookup):,}")
print(f"Total links:          {total_links:,} (+{added_count:,} exact matches over 0.682)")
print(f"Total singletons:     {total_singletons:,} ({total_singletons/len(cand_lookup)*100:.1f}%)")
print(f"Output written to:    {out_path}")
