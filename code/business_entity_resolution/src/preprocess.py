"""
preprocess.py — Fast vectorised text normalisation for business names and addresses.

Key performance decisions:
  - Skip per-row unicodedata.normalize() — too slow at 5M+ scale.
    Instead we use pandas str.normalize('NFKC') which calls the C layer.
  - Garble detection via fast regex, not per-char Python loop.
  - Single-pass combined regex for abbreviation expansion.
  - Parquet disk caching — preprocessed files saved so only runs once.

Adds columns: norm_name, norm_name_key, norm_address, zip_code, combined_text.
"""

import os
import re
import unicodedata
import pandas as pd
import numpy as np
from config import KEY_STOP_WORDS, ADDRESS_ABBR, NAME_ABBR

# ─── Cache directory ──────────────────────────────────────────────────────────
_SRC_DIR   = os.path.dirname(__file__)
_CACHE_DIR = os.path.join(_SRC_DIR, "..", "..", "..", "cache")
os.makedirs(_CACHE_DIR, exist_ok=True)


# ─── Pre-compiled patterns ────────────────────────────────────────────────────
_RE_LEADING_JUNK       = re.compile(r"^[\s\-_<>*#@]+")
_RE_TRAILING_JUNK      = re.compile(r"[\s\-_<>*#@]+$")
_RE_WHITESPACE         = re.compile(r"\s+")
_RE_PUNCT_EXCEPT_SPACE = re.compile(r"[^\w\s]")

# Garble detection: rows where >30% chars are control/private-use/replacement.
# Use Python re (not pandas str.count which may use PyArrow's RE2 engine).
_RE_GARBLE = re.compile(
    "[\x00-\x1f\ufffd]|[\ue000-\uf8ff]"
)

def _garble_mask(series: pd.Series) -> pd.Series:
    """Boolean Series — True where text is mostly garbled (>30% bad chars)."""
    def _count_bad(v: str) -> bool:
        if not v:
            return False
        bad = len(_RE_GARBLE.findall(v))
        return bad > len(v) * 0.3
    return series.map(_count_bad)


# ─── Abbreviation expansion — single-pass combined regex ─────────────────────

def _build_combined_regex(abbr_dict: dict):
    """
    Compile all patterns into ONE alternation regex so we do a single pass
    over the text instead of len(abbr_dict) separate passes.
    Returns (compiled_pattern, list_of_replacements).
    """
    keys   = list(abbr_dict.keys())
    values = list(abbr_dict.values())
    pattern = re.compile(
        "|".join(f"(?P<p{i}>{k})" for i, k in enumerate(keys)),
        re.IGNORECASE,
    )
    def replacer(m: re.Match) -> str:
        for i, v in enumerate(values):
            if m.group(f"p{i}") is not None:
                return v
        return m.group(0)
    return pattern, replacer


_NAME_RE,  _NAME_REPLACER  = _build_combined_regex(NAME_ABBR)
_ADDR_RE,  _ADDR_REPLACER  = _build_combined_regex(ADDRESS_ABBR)


# ─── Vectorised Series operations ────────────────────────────────────────────

def _norm_name_series(series: pd.Series) -> pd.Series:
    """Full name normalisation, vectorised."""
    s = series.fillna("").astype(str)
    # NFKC via pandas C layer (much faster than per-row unicodedata.normalize)
    s = s.str.normalize("NFKC")
    # Blank out garbled rows
    garbled = _garble_mask(s)
    s = s.where(~garbled, "")
    s = s.str.lower()
    s = s.str.replace(_RE_LEADING_JUNK,  "", regex=True)
    s = s.str.replace(_RE_TRAILING_JUNK, "", regex=True)
    # Single-pass abbreviation expansion using str.replace with compiled pattern
    s = s.str.replace(_NAME_RE, _NAME_REPLACER, regex=True)
    s = s.str.replace(_RE_WHITESPACE, " ", regex=True).str.strip()
    return s


def _norm_name_key_series(norm_name: pd.Series) -> pd.Series:
    """Blocking key — strip stop words and legal suffixes."""
    s = norm_name.str.replace(_RE_PUNCT_EXCEPT_SPACE, " ", regex=True)
    s = s.str.replace(_RE_WHITESPACE, " ", regex=True).str.strip()
    # Token filtering — unavoidable Python loop but fast since it's just splits
    def _strip_tokens(v: str) -> str:
        if not v:
            return v
        return " ".join(t for t in v.split() if t not in KEY_STOP_WORDS and len(t) >= 2)
    return s.map(_strip_tokens)


def _norm_address_series(series: pd.Series) -> pd.Series:
    """Full address normalisation, vectorised."""
    s = series.fillna("").astype(str)
    s = s.str.normalize("NFKC")
    garbled = _garble_mask(s)
    s = s.where(~garbled, "")
    s = s.str.lower()
    s = s.str.replace(_ADDR_RE, _ADDR_REPLACER, regex=True)
    s = s.str.replace(_RE_PUNCT_EXCEPT_SPACE, " ", regex=True)
    s = s.str.replace(_RE_WHITESPACE, " ", regex=True).str.strip()
    return s


# ─── ZIP extraction ───────────────────────────────────────────────────────────
_RE_US_ZIP    = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
_RE_INDIA_ZIP = re.compile(r"\b(\d{6})\b")
_RE_FR_ZIP    = re.compile(r"\b(\d{5})\b")


def _extract_zip_series(address_series: pd.Series, country_series: pd.Series) -> pd.Series:
    """Vectorised ZIP extraction — groups by country for bulk regex application."""
    result = pd.Series("", index=address_series.index)
    country_upper = country_series.str.upper()

    # US
    us_mask = country_upper.str.contains("US", na=False)
    if us_mask.any():
        result[us_mask] = address_series[us_mask].str.extract(
            _RE_US_ZIP.pattern, expand=False
        ).fillna("")

    # India
    in_mask = country_upper.str.contains("INDIA|^IN$", na=False, regex=True)
    if in_mask.any():
        result[in_mask] = address_series[in_mask].str.extract(
            _RE_INDIA_ZIP.pattern, expand=False
        ).fillna("")

    # Everything else (France, etc.)
    other_mask = ~(us_mask | in_mask)
    if other_mask.any():
        result[other_mask] = address_series[other_mask].str.extract(
            _RE_FR_ZIP.pattern, expand=False
        ).fillna("")

    return result.fillna("")


# ─── Scalar versions for feature computation ──────────────────────────────────

def normalise_name(raw) -> str:
    if pd.isna(raw):
        return ""
    s = str(raw)
    bad = len(_RE_GARBLE.findall(s))
    if bad > len(s) * 0.3:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = _RE_LEADING_JUNK.sub("", s)
    s = _RE_TRAILING_JUNK.sub("", s)
    s = _NAME_RE.sub(_NAME_REPLACER, s)
    return _RE_WHITESPACE.sub(" ", s).strip()


def normalise_name_key(raw) -> str:
    s = normalise_name(raw)
    if not s:
        return ""
    s = _RE_PUNCT_EXCEPT_SPACE.sub(" ", s)
    return " ".join(t for t in s.split() if t not in KEY_STOP_WORDS and len(t) >= 2)


def normalise_address(raw) -> str:
    if pd.isna(raw):
        return ""
    s = str(raw)
    bad = len(_RE_GARBLE.findall(s))
    if bad > len(s) * 0.3:
        return ""
    s = unicodedata.normalize("NFKC", s).lower()
    s = _ADDR_RE.sub(_ADDR_REPLACER, s)
    s = _RE_PUNCT_EXCEPT_SPACE.sub(" ", s)
    return _RE_WHITESPACE.sub(" ", s).strip()


# ─── Cache helpers ────────────────────────────────────────────────────────────

def _cache_path(tsv_path: str) -> str:
    basename = os.path.splitext(os.path.basename(tsv_path))[0]
    return os.path.join(_CACHE_DIR, f"{basename}_preprocessed.parquet")


# ─── Main entry point ─────────────────────────────────────────────────────────

def preprocess_df(df: pd.DataFrame, cache_key: str = None) -> pd.DataFrame:
    """
    Apply all normalisation to a source dataframe.
    Adds columns: norm_name, norm_name_key, norm_address, zip_code, combined_text.

    Args:
        df:        Input DataFrame with columns: entity_id, business_name,
                   business_address, country.
        cache_key: Path to the original TSV file. If provided, result is
                   cached to a .parquet file and reloaded on subsequent calls.
    """
    if cache_key:
        cp = _cache_path(cache_key)
        if os.path.exists(cp):
            print(f"  [cache] Loading {os.path.basename(cp)} … ", end="", flush=True)
            cached = pd.read_parquet(cp)
            print(f"done ({len(cached):,} rows)", flush=True)
            return cached

    n = len(df)
    print(f"  Preprocessing {n:,} rows … ", end="", flush=True)

    df = df.copy()
    df["norm_name"]     = _norm_name_series(df["business_name"])
    df["norm_name_key"] = _norm_name_key_series(df["norm_name"])
    df["norm_address"]  = _norm_address_series(df["business_address"])
    df["zip_code"]      = _extract_zip_series(df["business_address"], df["country"])
    df["combined_text"] = (df["norm_name"] + " " + df["norm_address"]).str.strip()

    print("done", flush=True)

    if cache_key:
        cp = _cache_path(cache_key)
        df.to_parquet(cp, index=False)
        print(f"  [cache] Saved → {os.path.basename(cp)}", flush=True)

    return df
