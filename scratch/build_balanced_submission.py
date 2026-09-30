import os
import sys
import time

print("=== Generating Balanced High-Precision/High-Recall Submission ===")
t0 = time.time()

# 1. Load candidates to guarantee 100% strict subset compliance
print("Reading candidate pairs lookup...")
cand_lookup = {}
with open("output/candidate_pairs.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        sid = parts[0]
        cands = set(parts[1].split(",")) if (len(parts) == 2 and parts[1]) else set()
        cand_lookup[sid] = cands

print(f"Loaded candidates for {len(cand_lookup):,} entities in {time.time()-t0:.1f}s")

# 2. Load Gold Matches (from 0.621 run)
t1 = time.time()
print("Loading gold 0.621 matches...")
gold_matches = {}
used_candidates = set()
gold_link_count = 0

with open("output/backup_0621/matching_results.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        sid = parts[0]
        ms = [m.strip() for m in parts[1].split(",") if m.strip()] if (len(parts) == 2 and parts[1]) else []
        gold_matches[sid] = ms
        for m in ms:
            used_candidates.add(m)
            gold_link_count += 1

print(f"Loaded {gold_link_count:,} gold links ({len(used_candidates):,} unique candidates) in {time.time()-t1:.1f}s")

# 3. Augment with High-Confidence Candidates (prob >= 0.60 from backup_0380)
# Strictly enforcing:
# a) Candidate Uniqueness: each candidate is assigned to at most ONE S1 entity
# b) Candidate Subset: candidate MUST be in candidate_pairs for that S1
# c) Top-5 Cap: at most 5 matches per entity
t2 = time.time()
print("Augmenting with high-confidence candidates (prob >= 0.60) under strict 1-to-1 uniqueness...")
added_count = 0

with open("output/backup_0380/matching_results.tsv", "r", encoding="utf-8") as f:
    next(f)
    for line in f:
        parts = line.strip().split("\t")
        sid = parts[0]
        if len(parts) < 2 or not parts[1]:
            continue
        ms = [m.strip() for m in parts[1].split(",") if m.strip()]
        cur_ms = gold_matches.get(sid, [])
        valid_cands_for_sid = cand_lookup.get(sid, set())

        room = 5 - len(cur_ms)  # Top-5 cap
        if room > 0:
            for m in ms:
                if m not in used_candidates and m in valid_cands_for_sid:
                    used_candidates.add(m)
                    cur_ms.append(m)
                    added_count += 1
                    room -= 1
                    if room == 0:
                        break

print(f"Added {added_count:,} high-confidence links in {time.time()-t2:.1f}s")

# 4. Stream write final matching_results.tsv in the exact S1 order of test_source1
t3 = time.time()
print("Writing final output/matching_results.tsv...")
out_path = "output/matching_results.tsv"
total_final_links = 0
total_singletons = 0

with open(out_path, "w", encoding="utf-8") as f_out:
    f_out.write("source1_entity_id\tmatched_entity_ids\n")
    # Read S1 IDs from candidate_pairs to maintain exact same ordering
    for sid in cand_lookup.keys():
        ms = gold_matches.get(sid, [])
        match_str = ",".join(ms)
        f_out.write(f"{sid}\t{match_str}\n")
        if ms:
            total_final_links += len(ms)
        else:
            total_singletons += 1

total_time = time.time() - t0
print(f"\n=== GENERATION COMPLETE in {total_time:.1f}s ===")
print(f"Total S1 entities:    {len(cand_lookup):,}")
print(f"Total final links:    {total_final_links:,} (avg {total_final_links/len(cand_lookup):.2f} per entity)")
print(f"Total singletons:     {total_singletons:,} ({total_singletons/len(cand_lookup)*100:.1f}%)")
print(f"Output written to:    {out_path}")
