import re
import pandas as pd

def get_name_key(name):
    if not name or pd.isna(name):
        return ""
    s = str(name).lower()
    for suf in ["private limited", "pvt ltd", "ltd", "inc", "llp", "corp", "corporation", "llc", "center", "trust", "company", "co"]:
        s = re.sub(rf"\b{suf}\b", "", s)
    s = re.sub(r"\.com|\.org|\.in|\.net|\.fr|\.co\b", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s

def get_addr_key(addr):
    if not addr or pd.isna(addr):
        return ""
    s = str(addr).lower()
    nums = re.findall(r"\b\d+\b", s)
    num = nums[0] if nums else ""
    words = re.findall(r"[a-z]+", s)
    stop_addr = {"door", "no", "plot", "floor", "near", "st", "street", "ave", "avenue", "rd", "road", "flat", "opp"}
    sig_words = [w for w in words if w not in stop_addr and len(w) >= 3]
    first_word = sig_words[0] if sig_words else ""
    if num and first_word:
        return f"{num}_{first_word}"
    elif len(sig_words) >= 2:
        return f"{sig_words[0]}_{sig_words[1]}"
    return ""

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
s2 = pd.read_parquet("cache/train_source2_preprocessed.parquet").set_index("entity_id")
s3 = pd.read_parquet("cache/train_source3_preprocessed.parquet").set_index("entity_id")

misses = 0
for _, r in gt.iterrows():
    sid = r["source1_entity_id"]
    if sid not in s1.index:
        continue
    s1_name = str(s1.loc[sid, "business_name"])
    s1_addr = str(s1.loc[sid, "business_address"])
    s1_nk = get_name_key(s1_name)
    s1_ak = get_addr_key(s1_addr)
    
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
        m_name = str(sub.loc[m, "business_name"])
        m_addr = str(sub.loc[m, "business_address"])
        m_nk = get_name_key(m_name)
        m_ak = get_addr_key(m_addr)
        
        hit_n = bool(s1_nk and m_nk and (s1_nk == m_nk or (len(s1_nk) >= 6 and (s1_nk in m_nk or m_nk in s1_nk))))
        hit_a = bool(s1_ak and m_ak and s1_ak == m_ak)
        
        if not (hit_n or hit_a):
            misses += 1
            print(f"Miss [{misses}]:")
            print(f"  S1: '{s1_name}' | '{s1_addr}'")
            print(f"  M:  '{m_name}' | '{m_addr}'")
            if misses >= 6:
                break
    if misses >= 6:
        break
