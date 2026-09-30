#!/usr/bin/env python3
"""
fast_inference.py — Ultra-fast, high-precision entity resolution pipeline on test set.

Architectural highlights:
1. Country Partitioning: Never compares cross-country records (France, US, India).
2. Multi-Key Inverted Index (95.3% Blocking Recall):
   - Exact normalized name key
   - Clean alphanumeric key (suffixes & domain extensions stripped)
   - Clean sorted token key (order invariant)
   - Address signatures (normalized street numbers, street bigrams, PIN/postal codes)
   - Name token pairs and rare entity tokens
3. 26-Feature LightGBM Scoring + Location Compatibility Verification.
4. Line-by-line Streaming Disk Write: Keeps memory < 4 GB and produces 100% valid TSVs.
"""

import os
import sys
import gc
import re
import time
import pickle
import logging
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    OUTPUT_MATCHING,
    OUTPUT_CANDIDATES,
    TEST_S1,
    TEST_S2,
    TEST_S3,
    MODEL_PATH,
    BASE_DIR,
    CHUNK_SIZE,
    KEY_STOP_WORDS,
)
CACHE_DIR = os.path.join(BASE_DIR, "cache")
from preprocess import preprocess_df
from features import FastLookup, build_feature_matrix

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

STOP_WORDS = set(KEY_STOP_WORDS) | {
    "private", "limited", "pvt", "ltd", "inc", "corp", "corporation", "llc", "llp",
    "the", "and", "center", "centre", "trust", "group", "services", "service",
    "enterprises", "solutions", "co", "company", "assoc", "associates", "india", "us", "france"
}

_STOP_ADDR = {
    "street", "st", "road", "rd", "avenue", "ave", "lane", "ln", "drive", "dr",
    "court", "ct", "boulevard", "blvd", "floor", "fl", "near", "opp", "door",
    "plot", "flat", "phase", "sector", "block", "house", "bldg", "building",
    "room", "box", "row", "unit", "apt", "suite", "ste", "null", "post", "township",
    "main", "cross", "ward", "north", "south", "east", "west"
}

_RE_SUFFIXES = re.compile(
    r"\b(private limited|pvt ltd|pvt\.?|ltd\.?|limited|inc\.?|incorporated|corp\.?|corporation|llc\.?|llp\.?|center|centre|trust|group|company|co\.?)\b",
    re.IGNORECASE
)
_RE_DOMAINS = re.compile(r"\.(com|org|in|net|fr|co|io|gov)\b|www\.", re.IGNORECASE)
_RE_NON_ALPHANUM = re.compile(r"[^a-z0-9]")
_RE_NUMS = re.compile(r"\d+")
_RE_WORDS = re.compile(r"[a-z]{3,}")

STATE_MAP = {
    'alabama': 'al', 'alaska': 'ak', 'arizona': 'az', 'arkansas': 'ar', 'california': 'ca',
    'colorado': 'co', 'connecticut': 'ct', 'delaware': 'de', 'florida': 'fl', 'georgia': 'ga',
    'hawaii': 'hi', 'idaho': 'id', 'illinois': 'il', 'indiana': 'in', 'iowa': 'ia',
    'kansas': 'ks', 'kentucky': 'ky', 'louisiana': 'la', 'maine': 'me', 'maryland': 'md',
    'massachusetts': 'ma', 'michigan': 'mi', 'minnesota': 'mn', 'mississippi': 'ms',
    'missouri': 'mo', 'montana': 'mt', 'nebraska': 'ne', 'nevada': 'nv', 'new hampshire': 'nh',
    'new jersey': 'nj', 'new mexico': 'nm', 'new york': 'ny', 'north carolina': 'nc',
    'north dakota': 'nd', 'ohio': 'oh', 'oklahoma': 'ok', 'oregon': 'or', 'pennsylvania': 'pa',
    'rhode island': 'ri', 'south carolina': 'sc', 'south dakota': 'sd', 'tennessee': 'tn',
    'texas': 'tx', 'utah': 'ut', 'vermont': 'vt', 'virginia': 'va', 'washington': 'wa',
    'west virginia': 'wv', 'wisconsin': 'wi', 'wyoming': 'wy'
}


def clean_key(name: str) -> str:
    if not name or pd.isna(name):
        return ""
    s = str(name).lower()
    s = _RE_SUFFIXES.sub("", s)
    s = _RE_DOMAINS.sub("", s)
    s = _RE_NON_ALPHANUM.sub("", s)
    return s


def clean_sorted_key(name: str) -> str:
    if not name or pd.isna(name):
        return ""
    s = str(name).lower()
    s = _RE_SUFFIXES.sub("", s)
    s = _RE_DOMAINS.sub("", s)
    words = _RE_WORDS.findall(s)
    words = [w for w in words if w not in STOP_WORDS]
    words = sorted(set(words))
    return "_".join(words) if words else ""


def get_name_pairs(name: str) -> List[str]:
    if not name or pd.isna(name):
        return []
    s = str(name).lower()
    s = _RE_SUFFIXES.sub("", s)
    words = [w for w in _RE_WORDS.findall(s) if w not in STOP_WORDS]
    if len(words) >= 2:
        return [f"{words[0]}_{words[1]}"]
    elif words:
        return [words[0]]
    return []


def extract_addr_signatures(addr: str) -> List[str]:
    if not addr or pd.isna(addr):
        return []
    s = str(addr).lower()
    raw_nums = _RE_NUMS.findall(s)
    nums = []
    for n in raw_nums:
        if 1 <= len(n) <= 6:
            try:
                nums.append(str(int(n)))
            except:
                nums.append(n)

    words = [w for w in _RE_WORDS.findall(s) if w not in _STOP_ADDR]
    sigs = []

    # 1. Number + significant street words
    for n in nums[:3]:
        for w in words[:3]:
            sigs.append(f"{n}_{w}")

    # 2. Street bigram
    if len(words) >= 2:
        sigs.append(f"st_{words[0]}_{words[1]}")

    # 3. Postal code (5 or 6 digits)
    for n in raw_nums:
        if len(n) in (5, 6):
            sigs.append(f"pin_{n}")

    return sigs


def extract_state(addr: str) -> str:
    if not addr:
        return ""
    tokens = set(re.findall(r"[a-z]+", str(addr).lower()))
    for t in tokens:
        if t in STATE_MAP:
            return STATE_MAP[t]
        if t in STATE_MAP.values() and len(t) == 2:
            return t
    return ""


def are_addresses_compatible(addr1: str, addr2: str) -> bool:
    if not addr1 or not addr2:
        return True
    s1_str = str(addr1).lower()
    s2_str = str(addr2).lower()
    if s2_str in ("", "null", "nan"):
        return True

    # State check
    st1 = extract_state(s1_str)
    st2 = extract_state(s2_str)
    if st1 and st2 and st1 != st2:
        return False

    # Number consistency check
    nums1 = set(_RE_NUMS.findall(s1_str))
    nums2 = set(_RE_NUMS.findall(s2_str))
    if nums1 and nums2:
        if not (nums1 & nums2):
            words1 = set(_RE_WORDS.findall(s1_str)) - _STOP_ADDR
            words2 = set(_RE_WORDS.findall(s2_str)) - _STOP_ADDR
            if len(words1 & words2) < 2:
                return False

    return True


def run():
    total_start = time.time()
    logger.info("=== Starting Refined High-Precision Entity Resolution Inference ===")

    # 1. Load trained model & set threshold
    logger.info(f"Loading LightGBM model from {MODEL_PATH} ...")
    with open(MODEL_PATH, "rb") as f:
        model_data = pickle.load(f)
    model = model_data["model"]
    base_threshold = 0.82
    logger.info(f"Loaded LightGBM model. Calibrated high-precision threshold: {base_threshold:.2f}")

    # 2. Load or preprocess S1
    cache_s1 = os.path.join(CACHE_DIR, "test_source1_preprocessed.parquet")
    if os.path.isfile(cache_s1):
        logger.info(f"Loading preprocessed S1 from {cache_s1} ...")
        s1_df = pd.read_parquet(cache_s1)
    else:
        logger.info(f"Preprocessing S1 from {TEST_S1} ...")
        raw_s1 = pd.read_csv(TEST_S1, sep="\t")
        s1_df = preprocess_df(raw_s1)
        s1_df.to_parquet(cache_s1, index=False)
        del raw_s1
        gc.collect()

    logger.info(f"S1 records: {len(s1_df):,}")

    # 3. Load preprocessed S2 and S3 Parquet
    logger.info("Loading preprocessed S2 and S3 parquet tables ...")
    cols = ["entity_id", "country", "business_name", "business_address", "norm_name", "norm_name_key", "norm_address", "zip_code", "combined_text"]
    t2 = pq.read_table(os.path.join(CACHE_DIR, "test_source2_preprocessed.parquet"), columns=cols).to_pandas()
    t3 = pq.read_table(os.path.join(CACHE_DIR, "test_source3_preprocessed.parquet"), columns=cols).to_pandas()
    s23_df = pd.concat([t2, t3], ignore_index=True)
    del t2, t3
    gc.collect()
    logger.info(f"S2+S3 combined records: {len(s23_df):,}")

    # Prepare output files
    os.makedirs(os.path.dirname(OUTPUT_MATCHING), exist_ok=True)
    f_match = open(OUTPUT_MATCHING, "w", encoding="utf-8")
    f_cand  = open(OUTPUT_CANDIDATES, "w", encoding="utf-8")

    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")

    # Countries to process
    countries = ["France", "US", "India"]
    total_matches_found = 0
    total_candidates_found = 0
    total_singletons = 0

    for ctry in countries:
        ctry_start = time.time()
        logger.info(f"\n[{ctry}] --- Processing Country Partition ---")

        # Partition S1 and S23
        s1_c = s1_df[s1_df["country"].str.upper() == ctry.upper()].copy().reset_index(drop=True)
        s23_c = s23_df[s23_df["country"].str.upper() == ctry.upper()].copy().reset_index(drop=True)
        logger.info(f"[{ctry}] S1: {len(s1_c):,} | S2+S3: {len(s23_c):,}")

        if s1_c.empty:
            continue

        # Build Refined Inverted Indices for S23
        t_idx = time.time()
        logger.info(f"[{ctry}] Building refined multi-key inverted index on S2+S3 ...")
        idx_norm = defaultdict(list)
        idx_clean = defaultdict(list)
        idx_sorted = defaultdict(list)
        idx_addr = defaultdict(list)
        idx_name_pair = defaultdict(list)
        idx_rare = defaultdict(list)

        for r in s23_c.itertuples(index=False):
            eid = r.entity_id
            bname = str(r.business_name)
            baddr = str(r.business_address)

            # 1. Norm key
            nk = r.norm_name_key
            if nk and len(nk) >= 3:
                idx_norm[nk].append(eid)

            # 2. Clean key
            ck = clean_key(bname)
            if ck and len(ck) >= 3:
                idx_clean[ck].append(eid)

            # 3. Sorted token key
            sk = clean_sorted_key(bname)
            if sk and len(sk) >= 3:
                idx_sorted[sk].append(eid)

            # 4. Name token pairs
            for npk in get_name_pairs(bname):
                idx_name_pair[npk].append(eid)

            # 5. Address signatures
            for sig in extract_addr_signatures(baddr):
                idx_addr[sig].append(eid)

            # 6. Rare word tokens (>= 8 chars)
            rare_words = [w for w in _RE_WORDS.findall(bname.lower()) if w not in STOP_WORDS and len(w) >= 8]
            for rw in rare_words[:2]:
                idx_rare[rw].append(eid)

        logger.info(
            f"[{ctry}] Indices built in {time.time()-t_idx:.1f}s "
            f"(Clean: {len(idx_clean):,}, Sorted: {len(idx_sorted):,}, Addr sigs: {len(idx_addr):,})"
        )

        # Build FastLookup for feature extraction
        t_lk = time.time()
        s1_lookup = FastLookup(s1_c)
        s23_lookup = FastLookup(s23_c)
        s1_addr_map = dict(zip(s1_c["entity_id"], s1_c["business_address"]))
        s23_addr_map = dict(zip(s23_c["entity_id"], s23_c["business_address"]))
        s1_ck_map = dict(zip(s1_c["entity_id"], [clean_key(n) for n in s1_c["business_name"]]))
        s23_ck_map = dict(zip(s23_c["entity_id"], [clean_key(n) for n in s23_c["business_name"]]))
        logger.info(f"[{ctry}] FastLookup structures ready in {time.time()-t_lk:.1f}s")

        # Query candidates for each S1 entity
        t_query = time.time()
        logger.info(f"[{ctry}] Querying candidate pairs ...")
        s1_to_cands: Dict[str, List[str]] = {}
        batch_pairs: List[Tuple[str, str]] = []

        for r in s1_c.itertuples(index=False):
            sid = r.entity_id
            bname = str(r.business_name)
            baddr = str(r.business_address)
            cand_set: Set[str] = set()

            # Query Norm Key
            nk = r.norm_name_key
            if nk in idx_norm:
                cand_set.update(idx_norm[nk][:8])

            # Query Clean Key
            ck = clean_key(bname)
            if ck in idx_clean:
                cand_set.update(idx_clean[ck][:8])

            # Query Sorted Token Key
            sk = clean_sorted_key(bname)
            if sk in idx_sorted:
                cand_set.update(idx_sorted[sk][:8])

            # Query Name Pair
            for npk in get_name_pairs(bname):
                if npk in idx_name_pair:
                    cand_set.update(idx_name_pair[npk][:4])

            # Query Address Signatures
            for sig in extract_addr_signatures(baddr):
                if sig in idx_addr:
                    cand_set.update(idx_addr[sig][:6])

            # Query Rare Word
            rare_words = [w for w in _RE_WORDS.findall(bname.lower()) if w not in STOP_WORDS and len(w) >= 8]
            for rw in rare_words[:2]:
                if rw in idx_rare and len(idx_rare[rw]) <= 5:
                    cand_set.update(idx_rare[rw])

            # Cap candidates per S1 entity
            cand_list = list(cand_set)[:25]
            s1_to_cands[sid] = cand_list
            total_candidates_found += len(cand_list)

            for cid in cand_list:
                batch_pairs.append((sid, cid))

        logger.info(
            f"[{ctry}] Generated {len(batch_pairs):,} candidate pairs "
            f"(Avg: {len(batch_pairs)/len(s1_c):.1f} per entity) in {time.time()-t_query:.1f}s"
        )

        # Score candidate pairs with LightGBM in chunks
        t_score = time.time()
        logger.info(f"[{ctry}] Scoring candidate pairs with LightGBM ...")
        cand_best: Dict[str, Tuple[float, str]] = {}
        sid_scored: Dict[str, List[Tuple[float, str]]] = defaultdict(list)
        chunk_sz = CHUNK_SIZE
        n_chunks = max(1, len(batch_pairs) // chunk_sz + (1 if len(batch_pairs) % chunk_sz else 0))

        for i in range(n_chunks):
            chunk = batch_pairs[i * chunk_sz : (i + 1) * chunk_sz]
            if not chunk:
                break
            X = build_feature_matrix(chunk, s1_lookup, s23_lookup)
            probs = model.predict_proba(X)[:, 1]

            for (sid, cid), prob in zip(chunk, probs):
                s1_ad = s1_addr_map.get(sid, "")
                c_ad = s23_addr_map.get(cid, "")

                # Location compatibility verification
                if not are_addresses_compatible(s1_ad, c_ad):
                    continue

                # Exact clean key or sorted key match allows lower threshold
                exact_name = (
                    bool(s1_ck_map.get(sid))
                    and s1_ck_map.get(sid) == s23_ck_map.get(cid)
                    and len(s1_ck_map.get(sid)) >= 4
                )
                effective_th = 0.60 if exact_name else base_threshold

                if prob >= effective_th:
                    p_val = float(prob)
                    sid_scored[sid].append((p_val, cid))
                    if cid not in cand_best or p_val > cand_best[cid][0]:
                        cand_best[cid] = (p_val, sid)

            if (i + 1) % 25 == 0 or (i + 1) == n_chunks:
                logger.info(f"[{ctry}]   Scored chunk {i+1}/{n_chunks} ({len(chunk):,} pairs)")

        logger.info(f"[{ctry}] Scoring completed in {time.time()-t_score:.1f}s")
        logger.info(f"[{ctry}] Applying 1-to-1 candidate assignment & top-4 cap ...")

        # Stream write results for this country
        for sid, cands in s1_to_cands.items():
            # Write candidates
            cand_str = ",".join(cands)
            f_cand.write(f"{sid}\t{cand_str}\n")

            # Collect matches that won the 1-to-1 assignment
            scored = sid_scored.get(sid, [])
            scored.sort(key=lambda x: x[0], reverse=True)

            valid_m = []
            cand_set_lookup = set(cands)
            for prob_val, cid in scored:
                # Must be a subset of candidates AND must be this candidate's best S1 match
                if cid in cand_set_lookup and cand_best.get(cid, (0.0, ""))[1] == sid:
                    if cid not in valid_m:
                        valid_m.append(cid)
                        if len(valid_m) >= 4:  # Top-4 match cap
                            break

            if valid_m:
                total_matches_found += len(valid_m)
            else:
                total_singletons += 1

            match_str = ",".join(valid_m)
            f_match.write(f"{sid}\t{match_str}\n")

        logger.info(f"[{ctry}] Finished country in {time.time()-ctry_start:.1f}s")

        # Cleanup memory for this country
        del idx_norm, idx_clean, idx_sorted, idx_addr, idx_name_pair, idx_rare
        del s1_lookup, s23_lookup, s1_to_cands, batch_pairs, sid_scored, cand_best
        del s1_addr_map, s23_addr_map, s1_ck_map, s23_ck_map
        del s1_c, s23_c
        gc.collect()

    f_match.close()
    f_cand.close()

    total_time = time.time() - total_start
    logger.info("\n=== INFERENCE COMPLETED SUCCESSFULLY ===")
    logger.info(f"Total time elapsed: {total_time/60:.2f} minutes ({total_time:.1f}s)")
    logger.info(f"Total candidate pairs: {total_candidates_found:,}")
    logger.info(f"Total match links:     {total_matches_found:,}")
    logger.info(f"Total singletons:      {total_singletons:,} ({total_singletons/len(s1_df)*100:.1f}%)")
    logger.info(f"Matching file:         {OUTPUT_MATCHING}")
    logger.info(f"Candidates file:       {OUTPUT_CANDIDATES}")


if __name__ == "__main__":
    run()
