import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd
import numpy as np

# Load test_pipeline_validation errors
print("=== ANALYZING 0.8715 ERROR CASES ===")
from scratch.test_pipeline_validation import (
    s1_eval, s23_eval, s1_cands_map, pair_probs, gt_dict, f_beta
)

th = 0.6
entity_scores = []
for sid, true_m in gt_dict.items():
    if sid not in s1_eval["entity_id"].values: continue
    c_list = s1_cands_map.get(sid, [])
    pred = set(cid for cid in c_list if pair_probs.get((sid, cid), 0.0) >= th)
    if not true_m and not pred:
        f = 1.0
    elif not true_m and pred:
        f = 0.0
    elif true_m and not pred:
        f = 0.0
    else:
        tp = len(pred & true_m)
        p = tp / len(pred)
        r = tp / len(true_m)
        f = f_beta(p, r)
    entity_scores.append((f, sid, true_m, pred))

entity_scores.sort(key=lambda x: x[0])
print(f"Entities with score == 1.0: {sum(1 for s in entity_scores if s[0] == 1.0)} / {len(entity_scores)} ({sum(1 for s in entity_scores if s[0] == 1.0)/len(entity_scores)*100:.1f}%)")
print(f"Entities with score < 0.5: {sum(1 for s in entity_scores if s[0] < 0.5)} / {len(entity_scores)} ({sum(1 for s in entity_scores if s[0] < 0.5)/len(entity_scores)*100:.1f}%)")

s1_map = s1_eval.set_index("entity_id")
s23_map = s23_eval.set_index("entity_id")

print("\n--- SAMPLE LOW SCORING ENTITIES (score < 0.5) ---")
for f, sid, true_m, pred in entity_scores[:10]:
    s1_row = s1_map.loc[sid]
    print(f"\nScore: {f:.2f} | S1 ({sid}, {s1_row['country']}):")
    print(f"  NAME: {s1_row['business_name']}")
    print(f"  ADDR: {s1_row['business_address']}")
    print(f"  TRUE MATCHES ({len(true_m)}): {true_m}")
    for mid in list(true_m)[:2]:
        if mid in s23_map.index:
            mr = s23_map.loc[mid]
            p = pair_probs.get((sid, mid), "NOT IN CANDS")
            print(f"    TRUE {mid} (prob={p}): {mr['business_name']} ||| {mr['business_address']}")
    print(f"  PREDICTED ({len(pred)}): {pred}")
    for pid in list(pred)[:2]:
        if pid in s23_map.index:
            pr = s23_map.loc[pid]
            p = pair_probs.get((sid, pid), 0.0)
            print(f"    PRED {pid} (prob={p:.4f}): {pr['business_name']} ||| {pr['business_address']}")
