"""
features.py — Feature engineering for candidate pairs.

For each (S1, S2/S3) candidate pair, compute a rich set of similarity
features that the LightGBM classifier will use for the final match decision.

Feature groups:
  1. Name similarity  (token Jaccard, char n-gram Jaccard, Jaro-Winkler,
                       Levenshtein ratio)
  2. Address similarity (token Jaccard, partial ratio, lev ratio)
  3. Country match    (exact boolean)
  4. ZIP match        (exact boolean)
  5. Combined text    (token Jaccard, char n-gram Jaccard)
  6. Length features  (length ratio, absolute length diff)
  7. Prefix/suffix signals (shared prefix length, shared suffix length)
"""

import re
import logging
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, distance

logger = logging.getLogger(__name__)


# ─── Fast Columnar Lookup ─────────────────────────────────────────────────────

class FastLookup:
    """
    Columnar numpy lookup table for ultra-fast O(1) field retrieval.
    Replaces dict-of-dicts (to_dict('index')) which creates 90M+ objects and thrashes memory.
    """
    def __init__(self, df: pd.DataFrame):
        ids = df["entity_id"].to_numpy()
        self.id_to_idx = {eid: i for i, eid in enumerate(ids)}
        self.norm_name     = df["norm_name"].fillna("").to_numpy()
        self.norm_name_key = df["norm_name_key"].fillna("").to_numpy()
        self.norm_address  = df["norm_address"].fillna("").to_numpy()
        self.zip_code      = df["zip_code"].fillna("").to_numpy()
        self.country       = df["country"].fillna("").to_numpy()
        self.combined_text = df["combined_text"].fillna("").to_numpy()

    def get_tuple(self, entity_id: str) -> Optional[tuple]:
        idx = self.id_to_idx.get(entity_id)
        if idx is None:
            return None
        return (
            self.norm_name[idx],
            self.norm_name_key[idx],
            self.norm_address[idx],
            self.zip_code[idx],
            self.country[idx],
            self.combined_text[idx],
        )

    def __contains__(self, entity_id: str) -> bool:
        return entity_id in self.id_to_idx

    def __len__(self) -> int:
        return len(self.id_to_idx)


# ─── Token helpers ────────────────────────────────────────────────────────────

def _token_set(s: str) -> set:
    return set(re.findall(r"\w+", s.lower())) if s else set()


def _token_jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def _char_ngram_jaccard(a: str, b: str, n: int = 3) -> float:
    def ngrams(s, n):
        s = s.replace(" ", "_")
        return set(s[i:i+n] for i in range(max(0, len(s)-n+1)))
    sa, sb = ngrams(a, n), ngrams(b, n)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def _shared_prefix_len(a: str, b: str) -> int:
    m = min(len(a), len(b))
    for i in range(m):
        if a[i] != b[i]:
            return i
    return m


def _shared_suffix_len(a: str, b: str) -> int:
    ra, rb = a[::-1], b[::-1]
    return _shared_prefix_len(ra, rb)


# ─── Feature computation from raw strings ─────────────────────────────────────

def compute_raw_pair_features(
    n1: str, nk1: str, a1: str, z1: str, c1: str, ct1: str,
    n2: str, nk2: str, a2: str, z2: str, c2: str, ct2: str,
) -> Dict[str, float]:
    feats: Dict[str, float] = {}

    c1_u = str(c1).upper()
    c2_u = str(c2).upper()

    # ── Name features ────────────────────────────────────────────────────────
    feats["name_token_jaccard"]    = _token_jaccard(n1, n2)
    feats["namekey_token_jaccard"] = _token_jaccard(nk1, nk2)
    feats["name_char3_jaccard"]    = _char_ngram_jaccard(n1, n2, 3)
    feats["name_char2_jaccard"]    = _char_ngram_jaccard(n1, n2, 2)

    feats["name_jaro_winkler"]  = fuzz.token_set_ratio(n1, n2) / 100.0
    feats["name_partial_ratio"] = fuzz.partial_ratio(n1, n2) / 100.0
    feats["name_sort_ratio"]    = fuzz.token_sort_ratio(n1, n2) / 100.0
    feats["name_lev_ratio"]     = (
        1.0 - distance.Levenshtein.normalized_distance(n1, n2)
    )

    feats["namekey_jaro_winkler"]  = fuzz.token_set_ratio(nk1, nk2) / 100.0
    feats["namekey_partial_ratio"] = fuzz.partial_ratio(nk1, nk2) / 100.0

    n1l, n2l = n1.lower(), n2.lower()
    plen = _shared_prefix_len(n1l, n2l)
    slen = _shared_suffix_len(n1l, n2l)
    maxlen = max(len(n1l), len(n2l), 1)
    feats["name_prefix_ratio"] = plen / maxlen
    feats["name_suffix_ratio"] = slen / maxlen

    len1, len2 = len(n1), len(n2)
    feats["name_len_ratio"] = min(len1, len2) / max(len1, len2, 1)
    feats["name_len_diff"]  = abs(len1 - len2)

    # ── Address features ──────────────────────────────────────────────────────
    a1_empty = (a1 == "")
    a2_empty = (a2 == "")
    feats["addr_both_empty"]      = float(a1_empty and a2_empty)
    feats["addr_one_empty"]       = float(a1_empty != a2_empty)

    if not a1_empty and not a2_empty:
        feats["addr_token_jaccard"]  = _token_jaccard(a1, a2)
        feats["addr_char3_jaccard"]  = _char_ngram_jaccard(a1, a2, 3)
        feats["addr_partial_ratio"]  = fuzz.partial_ratio(a1, a2) / 100.0
        feats["addr_sort_ratio"]     = fuzz.token_sort_ratio(a1, a2) / 100.0
        feats["addr_lev_ratio"]      = (
            1.0 - distance.Levenshtein.normalized_distance(a1[:200], a2[:200])
        )
    else:
        feats["addr_token_jaccard"]  = 0.0
        feats["addr_char3_jaccard"]  = 0.0
        feats["addr_partial_ratio"]  = 0.0
        feats["addr_sort_ratio"]     = 0.0
        feats["addr_lev_ratio"]      = 0.0

    # ── ZIP / country features ────────────────────────────────────────────────
    feats["country_match"] = float(c1_u == c2_u)
    feats["zip_match"]     = float(bool(z1) and bool(z2) and z1 == z2)
    feats["zip_prefix_match"] = float(
        bool(z1) and bool(z2) and z1[:3] == z2[:3]
    )

    # ── Combined text features ────────────────────────────────────────────────
    feats["combined_token_jaccard"] = _token_jaccard(ct1, ct2)
    feats["combined_char3_jaccard"] = _char_ngram_jaccard(ct1, ct2, 3)

    return feats


def compute_pair_features(s1_row, s23_row) -> Dict[str, float]:
    """Compatibility wrapper for Series/dict inputs."""
    return compute_raw_pair_features(
        s1_row["norm_name"], s1_row["norm_name_key"], s1_row["norm_address"],
        s1_row["zip_code"], s1_row["country"], s1_row["combined_text"],
        s23_row["norm_name"], s23_row["norm_name_key"], s23_row["norm_address"],
        s23_row["zip_code"], s23_row["country"], s23_row["combined_text"],
    )


# ─── Batch feature extraction ─────────────────────────────────────────────────

FEATURE_NAMES = [
    "name_token_jaccard", "namekey_token_jaccard",
    "name_char3_jaccard", "name_char2_jaccard",
    "name_jaro_winkler", "name_partial_ratio", "name_sort_ratio", "name_lev_ratio",
    "namekey_jaro_winkler", "namekey_partial_ratio",
    "name_prefix_ratio", "name_suffix_ratio",
    "name_len_ratio", "name_len_diff",
    "addr_both_empty", "addr_one_empty",
    "addr_token_jaccard", "addr_char3_jaccard",
    "addr_partial_ratio", "addr_sort_ratio", "addr_lev_ratio",
    "country_match", "zip_match", "zip_prefix_match",
    "combined_token_jaccard", "combined_char3_jaccard",
]


def build_feature_matrix(
    pairs: List[Tuple[str, str]],
    s1_lookup: Any,
    s23_lookup: Any,
) -> np.ndarray:
    """
    Compute feature matrix for a list of (s1_id, s23_id) pairs.
    Supports FastLookup, Dict, or DataFrame lookups seamlessly.
    """
    n_feats = len(FEATURE_NAMES)
    rows = []

    use_fast_s1  = isinstance(s1_lookup, FastLookup)
    use_fast_s23 = isinstance(s23_lookup, FastLookup)

    for s1_id, s23_id in pairs:
        if use_fast_s1:
            t1 = s1_lookup.get_tuple(s1_id)
        else:
            r = s1_lookup.get(s1_id)
            t1 = (r["norm_name"], r["norm_name_key"], r["norm_address"],
                  r["zip_code"], r["country"], r["combined_text"]) if r is not None else None

        if use_fast_s23:
            t2 = s23_lookup.get_tuple(s23_id)
        else:
            r = s23_lookup.get(s23_id)
            t2 = (r["norm_name"], r["norm_name_key"], r["norm_address"],
                  r["zip_code"], r["country"], r["combined_text"]) if r is not None else None

        if t1 is None or t2 is None:
            rows.append([0.0] * n_feats)
            continue

        feats = compute_raw_pair_features(*t1, *t2)
        rows.append([feats.get(f, 0.0) for f in FEATURE_NAMES])

    return np.array(rows, dtype=np.float32)
