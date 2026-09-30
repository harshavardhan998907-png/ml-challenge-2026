import os
import sys
import time
import re
import pyarrow.parquet as pq
import pandas as pd

print("=== Generating Final 0.70+ Push Submission ===")
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

# 2. Entity names & addresses
print("2. Loading entity names and addresses...")
cols = ["entity_id", "business_name", "business_address"]
s1_df = pd.read_parquet("cache/test_source1_preprocessed.parquet", columns=cols)

_RE_ALPHANUM = re.compile(r"[^a-z0-9]")
_RE_WORDS = re.compile(r"[a-z0-9]{3,}")
_RE_NUMS = re.compile(r"\d+")
STOP_WORDS = {
    "private", "limited", "pvt", "ltd", "inc", "corp", "corporation", "llc", "llp",
    "the", "and", "center", "centre", "trust", "group", "services", "service",
    "enterprises", "solutions", "co", "company", "india", "us", "france"
}

def clean_k(s):
    return _RE_ALPHANUM.sub("", str(s).lower())

def sorted_k_and_words(s):
    words = sorted(set(w for w in _RE_WORDS.findall(str(s).lower()) if w not in STOP_WORDS))
    return "_".join(words) if words else "", len(words)

s1_ck = dict(zip(s1_df["entity_id"], s1_df["business_name"].apply(clean_k)))
s1_sk_data = dict(zip(s1_df["entity_id"], s1_df["business_name"].apply(sorted_k_and_words)))
s1_addrs = dict(zip(s1_df["entity_id"], s1_df["business_address"].fillna("").str.lower()))
del s1_df

t2 = pq.read_table("cache/test_source2_preprocessed.parquet", columns=cols).to_pandas()
t3 = pq.read_table("cache/test_source3_preprocessed.parquet", columns=cols).to_pandas()
s23 = pd.concat([t2, t3], ignore_index=True)
s23_ck = dict(zip(s23["entity_id"], s23["business_name"].apply(clean_k)))
s23_sk_data = dict(zip(s23["entity_id"], s23["business_name"].apply(sorted_k_and_words)))
s23_addrs = dict(zip(s23["entity_id"], s23["business_address"].fillna("").str.lower()))
del t2, t3, s23

# 3. Load baseline 0.692 matches
print("3. Loading 0.692 matches...")
matches = {}
used_cands = set()
with open("output/backup_0692/matching_results.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        sid = parts[0]
        ms = [m.strip() for m in parts[1].split(",") if m.strip()] if (len(parts) == 2 and parts[1]) else []
        matches[sid] = ms
        for m in ms:
            used_cands.add(m)

base_count = len(used_cands)
print(f"   Loaded {base_count:,} existing 0.692 matches")

# 4. Augment with exact clean and exact sorted multi-word matches from backup_0380
print("4. Augmenting with pure high-precision exact & sorted-token matches...")
added_clean = 0
added_sorted = 0

with open("output/backup_0380/matching_results.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        sid = parts[0]
        if len(parts) < 2 or not parts[1]:
            continue

        cur_ms = matches.get(sid, [])
        room = 5 - len(cur_ms)  # Official Top-5 Cap
        if room <= 0:
            continue

        valid_cands = cand_lookup.get(sid, set())
        s1_c = s1_ck.get(sid, "")
        s1_s, s1_nwords = s1_sk_data.get(sid, ("", 0))
        s1_ad = s1_addrs.get(sid, "")
        s1_nums = set(_RE_NUMS.findall(s1_ad)) if s1_ad else set()

        for cid in parts[1].split(","):
            cid = cid.strip()
            if not cid or cid in used_cands or cid not in valid_cands:
                continue

            c_c = s23_ck.get(cid, "")
            c_s, c_nwords = s23_sk_data.get(cid, ("", 0))

            # Address compatibility: If both have numbers, check conflict
            c_ad = s23_addrs.get(cid, "")
            if s1_nums and c_ad:
                c_nums = set(_RE_NUMS.findall(c_ad))
                if c_nums and not (s1_nums & c_nums):
                    # Conflicting building/street numbers -> skip to protect precision
                    continue

            # Case A: Exact clean name (len >= 4)
            if s1_c and s1_c == c_c and len(s1_c) >= 4:
                used_cands.add(cid)
                cur_ms.append(cid)
                added_clean += 1
                room -= 1
                if room == 0:
                    break
            # Case B: Multi-word exact sorted tokens (>= 2 significant words, len >= 8)
            elif s1_s and s1_s == c_s and s1_nwords >= 2 and len(s1_s) >= 8:
                used_cands.add(cid)
                cur_ms.append(cid)
                added_sorted += 1
                room -= 1
                if room == 0:
                    break

print(f"   Added {added_clean:,} exact clean name matches")
print(f"   Added {added_sorted:,} exact multi-word sorted token matches")
print(f"   Total newly added: {added_clean + added_sorted:,}")

# 5. Write to output/matching_results_final.tsv
out_path = "output/matching_results_final.tsv"
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
print(f"Total links:          {total_links:,} (+{added_clean + added_sorted:,} over 0.692)")
print(f"Total singletons:     {total_singletons:,} ({total_singletons/len(cand_lookup)*100:.1f}%)")
print(f"Output written to:    {out_path}")
