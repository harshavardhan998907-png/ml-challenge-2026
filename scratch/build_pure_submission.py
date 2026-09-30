import os
import sys
import time
import re
import pyarrow.parquet as pq
import pandas as pd

print("=== Generating High-Precision Pure Submission (Pruning Zero-Name-Overlap Noise) ===")
t0 = time.time()

# 1. Load candidate pairs lookup to guarantee 100% candidate subset compliance
print("1. Reading candidate pairs lookup...")
cand_lookup = {}
with open("output/candidate_pairs.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        sid = parts[0]
        cands = set(parts[1].split(",")) if (len(parts) == 2 and parts[1]) else set()
        cand_lookup[sid] = cands

print(f"   Loaded candidate sets for {len(cand_lookup):,} entities in {time.time()-t0:.1f}s")

# 2. Load names from parquet
t1 = time.time()
print("2. Loading entity names...")
cols = ["entity_id", "business_name"]
s1_df = pd.read_parquet("cache/test_source1_preprocessed.parquet", columns=cols)
s1_names = dict(zip(s1_df["entity_id"], s1_df["business_name"].fillna("")))
del s1_df

t2 = pq.read_table("cache/test_source2_preprocessed.parquet", columns=cols).to_pandas()
t3 = pq.read_table("cache/test_source3_preprocessed.parquet", columns=cols).to_pandas()
s23 = pd.concat([t2, t3], ignore_index=True)
s23_names = dict(zip(s23["entity_id"], s23["business_name"].fillna("")))
del t2, t3, s23

print(f"   Loaded {len(s1_names):,} S1 names and {len(s23_names):,} S2/S3 names in {time.time()-t1:.1f}s")

# 3. Clean word extraction
_RE_ALPHANUM = re.compile(r"[^a-z0-9]")
_RE_WORDS = re.compile(r"[a-z0-9]{2,}")
STOP_WORDS = {
    "private", "limited", "pvt", "ltd", "inc", "corp", "corporation", "llc", "llp",
    "the", "and", "center", "centre", "trust", "group", "services", "service",
    "enterprises", "solutions", "co", "company", "india", "us", "france"
}

def clean_k(s):
    return _RE_ALPHANUM.sub("", s.lower())

def get_sig_words(s):
    words = _RE_WORDS.findall(s.lower())
    return {w for w in words if w not in STOP_WORDS}

# Pre-compute significant words for S1 and S23
print("3. Precomputing name tokens...")
t2_tok = time.time()
s1_words = {sid: get_sig_words(name) for sid, name in s1_names.items()}
s1_cleans = {sid: clean_k(name) for sid, name in s1_names.items()}
del s1_names

s23_words = {cid: get_sig_words(name) for cid, name in s23_names.items()}
s23_cleans = {cid: clean_k(name) for cid, name in s23_names.items()}
del s23_names
print(f"   Tokens precomputed in {time.time()-t2_tok:.1f}s")

# 4. Filter backup_0621 matches
t3 = time.time()
print("4. Filtering backup_0621 matches...")
retained_links = 0
pruned_links = 0
final_matches = {}

with open("output/backup_0621/matching_results.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        sid = parts[0]
        if len(parts) < 2 or not parts[1]:
            final_matches[sid] = []
            continue

        valid_cands = cand_lookup.get(sid, set())
        s1_w = s1_words.get(sid, set())
        s1_c = s1_cleans.get(sid, "")

        kept_cids = []
        for cid in parts[1].split(","):
            cid = cid.strip()
            if not cid or cid not in valid_cands:
                pruned_links += 1
                continue

            c_w = s23_words.get(cid, set())
            c_c = s23_cleans.get(cid, "")

            # High-confidence verification:
            # 1. Exact clean name or strong substring
            is_name_match = (
                (s1_c and c_c and (s1_c == c_c or (len(s1_c) >= 5 and s1_c in c_c) or (len(c_c) >= 5 and c_c in s1_c)))
                or len(s1_w & c_w) >= 1  # At least one non-stopword significant business name token matches
            )

            if is_name_match:
                kept_cids.append(cid)
                retained_links += 1
            else:
                pruned_links += 1

        final_matches[sid] = kept_cids

print(f"   Retained: {retained_links:,} high-purity links")
print(f"   Pruned:   {pruned_links:,} false positives (0-word overlap)")

# 5. Write output/matching_results.tsv in exact S1 order
t4 = time.time()
out_path = "output/matching_results.tsv"
print(f"5. Writing output to {out_path}...")
total_written_links = 0
total_singletons = 0

with open(out_path, "w", encoding="utf-8") as f_out:
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in cand_lookup.keys():
        ms = final_matches.get(sid, [])
        match_str = ",".join(ms)
        f_out.write(f"{sid}\t{match_str}\n")
        if ms:
            total_written_links += len(ms)
        else:
            total_singletons += 1

print(f"\n=== GENERATION COMPLETE in {time.time()-t0:.1f}s ===")
print(f"Total S1 entities:    {len(cand_lookup):,}")
print(f"Total written links:  {total_written_links:,} (avg {total_written_links/len(cand_lookup):.2f} per entity)")
print(f"Total singletons:     {total_singletons:,} ({total_singletons/len(cand_lookup)*100:.1f}%)")
print(f"Output written to:    {out_path}")
