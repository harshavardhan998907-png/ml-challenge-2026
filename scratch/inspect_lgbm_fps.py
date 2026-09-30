import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
from scratch.test_refined_pipeline import (
    s1_eval, s23_eval, eval_pairs, pair_probs, gt_dict, are_addresses_compatible, s1_addr_map, s23_addr_map
)

print("=== INSPECTING LIGHTGBM FALSE POSITIVES (prob >= 0.8) ===")
s1_name_map = dict(zip(s1_eval["entity_id"], s1_eval["business_name"]))
s23_name_map = dict(zip(s23_eval["entity_id"], s23_eval["business_name"]))

count = 0
for (sid, cid), prob in pair_probs.items():
    if prob >= 0.8:
        true_m = gt_dict.get(sid, set())
        if cid not in true_m:
            count += 1
            if count <= 15:
                compat = are_addresses_compatible(s1_addr_map[sid], s23_addr_map.get(cid, ""))
                print(f"FP #{count} (prob: {prob:.4f}, compat: {compat}):")
                print(f"  S1  ({sid}): {s1_name_map[sid]} ||| {s1_addr_map[sid]}")
                print(f"  CAND({cid}): {s23_name_map[cid]} ||| {s23_addr_map[cid]}")
                print("-" * 60)
print(f"Total false positives with prob >= 0.8: {count}")
