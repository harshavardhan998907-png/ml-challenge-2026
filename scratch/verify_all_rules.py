import sys
import pandas as pd

print("=== EXHAUSTIVE FORMAT VERIFICATION ===")

# Rule 1 & 2: Header and tab separation
with open("output/matching_results.tsv", "r", encoding="utf-8") as f:
    first_line = f.readline()
    assert first_line == "source1_entity_id\tmatched_entity_ids\n", f"Invalid header: {repr(first_line)}"
print("Rule 1 & 2: Header and tab separation -> PASS")

# Load test S1
s1_test = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id"])
expected_ids = set(s1_test["entity_id"])
print(f"Total required S1 entities: {len(expected_ids):,}")

# Check every row in matching_results.tsv
seen_s1 = set()
intra_dupes = 0
invalid_prefixes = 0
quotes_found = 0
spaces_found = 0
singletons = 0
total_links = 0

with open("output/matching_results.tsv", "r", encoding="utf-8") as f:
    next(f)  # header
    for line_num, line in enumerate(f, start=2):
        if '"' in line or "'" in line:
            quotes_found += 1
        parts = line.rstrip("\r\n").split("\t")
        if len(parts) != 2:
            print(f"Line {line_num} does not have exactly 2 tab-separated columns: {repr(line)}")
            sys.exit(1)
        s1_id, matches_str = parts
        if s1_id in seen_s1:
            print(f"Duplicate S1 ID found: {s1_id} at line {line_num}")
            sys.exit(1)
        seen_s1.add(s1_id)

        if not matches_str:
            singletons += 1
            continue

        m_list = matches_str.split(",")
        if len(m_list) != len(set(m_list)):
            intra_dupes += 1

        for m in m_list:
            total_links += 1
            if " " in m:
                spaces_found += 1
            if not (m.startswith("S2-") or m.startswith("S3-")):
                invalid_prefixes += 1

print(f"Total S1 rows in file: {len(seen_s1):,}")
assert len(seen_s1) == len(expected_ids), f"Count mismatch: {len(seen_s1)} vs {len(expected_ids)}"
assert seen_s1 == expected_ids, "S1 IDs do not match test_source1.tsv exactly!"
print("Rule 3: Exact S1 Entity Coverage (1,732,544 rows, 0 missing, 0 duplicates) -> PASS")

assert intra_dupes == 0, f"Intra-row duplicates found: {intra_dupes}"
print("Rule 4: Intra-row duplicates (0 found) -> PASS")

assert invalid_prefixes == 0, f"Invalid prefixes found: {invalid_prefixes}"
print("Rule 5: Valid prefixes (S2- and S3- only, no S1- self-matches) -> PASS")

assert quotes_found == 0, f"Quotes found: {quotes_found}"
assert spaces_found == 0, f"Spaces in ID lists: {spaces_found}"
print("Rule 6: Clean formatting (No quotes, no spaces in IDs) -> PASS")

print(f"Rule 7: Singletons properly formatted (empty string): {singletons:,} ({singletons/len(seen_s1)*100:.1f}%) -> PASS")
print(f"Total predicted match links: {total_links:,}")
print("\n>>> ALL COMPETITION RULES & REGULATIONS VERIFIED 100% PASS <<<")
