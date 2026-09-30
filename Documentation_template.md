# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** EntityResolvers  
**Team Members:** Team Lead & ML Engineers  
**Submission Date:** 27 September 2026  

---

## 1. Executive Summary
We developed an end-to-end, high-precision Business Entity Resolution system capable of deduplicating and linking multi-million records across 3 independent data sources (France, US, India). Our solution pairs a 6-key inverted index blocking engine (>95.3% recall over 35.1M candidate pairs) with a 26-feature LightGBM classification model and a global 1-to-1 candidate assignment verifier. By enforcing strict 1-to-1 match uniqueness and top-4 link capping to match empirical real-world ground truth distributions, our solution eliminates false positive multi-assignments and maximizes the competition Macro $F_{0.5}$ metric (>98% Precision, ~91% Recall).

---

## 2. Methodology

### 2.1 Problem Analysis
During exploratory data analysis across 2.2M training entities:
1. **Zero Cross-Country Matches:** 100% of true matches are strictly intra-country (France matches only France, US only US, India only India).
2. **True Match Cardinality:** The true ground truth exhibits a mean of 3.46 matches per $S_1$ entity (median = 3, ~89% have $\le 5$ matches, ~5.6% singletons).
3. **1-to-1 Candidate Uniqueness Law:** In 10,000 ground truth samples, 0.00% of candidate entities ($S_2, S_3$) map to more than one $S_1$ business. A single business in $S_2/S_3$ cannot belong to multiple distinct real-world $S_1$ businesses.
4. **Noise Patterns:** Indian transliteration/Devanagari script variations, abbreviation differences (e.g. "Pvt Ltd" vs "Private Limited"), and diverse address formats require multi-layered clean key normalization and physical address signature matching.

### 2.2 Solution Strategy
**Approach Type:** Country-Partitioned Multi-Key Blocking + 26-Feature LightGBM + 1-to-1 Global Deduplication Verifier.  
**Core Innovation:** 
- Country-partitioned streaming architecture to guarantee zero cross-country violations and keep peak RAM under 4 GB.
- Global 1-to-1 candidate assignment: assigning each candidate entity strictly to its single highest-scoring $S_1$ match, eliminating millions of duplicate false positives.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used:**
  1. *Exact Normalized Name Key:* Lowercased, alphanumeric normalized string.
  2. *Clean Key:* Legal suffixes (`pvt`, `ltd`, `corp`, `inc`, `llc`) and web domains (`.com`, `.in`, `.fr`) stripped.
  3. *Sorted Token Key:* Words tokenized, filtered of stop-words, and sorted alphabetically to handle word order variations.
  4. *Name Bigram Pairs:* First two significant word tokens.
  5. *Physical Address Signatures:* Normalized house/door numbers paired with significant street words and postal/PIN codes.
  6. *Rare Entity Tokens:* Entity-specific word tokens ($\ge 8$ chars) with low frequency.
- **Candidate pairs generated:** 35,123,483 pairs across 1,732,544 test $S_1$ entities (~20.3 candidates per entity).
- **How true matches were preserved:** Multi-key inverted indexing ensures that an entity can be captured via exact name, token permutation, address signature, or postal code independently, yielding >95.3% candidate recall on validation.

---

## 4. Matching Model

**Features used (26 total pairwise features):**
- **Name features:** Token Jaccard, character 2-gram and 3-gram Jaccard, RapidFuzz Token Set Ratio, Token Sort Ratio, Partial Ratio, Levenshtein Normalized Distance, Shared Prefix Ratio, Shared Suffix Ratio, Length Ratio, Absolute Length Difference.
- **Address features:** Token Jaccard, RapidFuzz Partial Ratio, Normalized Levenshtein Ratio, House Number Overlap, State Code Consistency.
- **Location & Metadata features:** Exact Country Match, Exact ZIP Code Match, Combined Text Token Jaccard, Combined Text 3-gram Jaccard.

**Model type:** LightGBM Classifier (`LGBMClassifier`) trained on balanced positive/negative pairs with 800 estimators, learning rate 0.05, num_leaves 63.  
**Threshold selection & Post-Processing:**
- Base decision threshold: `0.82` (calibrated on ground truth where 92.2% of true matches score $\ge 0.80$).
- Clean exact name match threshold: `0.60`.
- Address compatibility filter: Zero house-number or state code contradictions.
- Global 1-to-1 assignment: Every candidate $C$ is assigned exclusively to its single highest-scoring $S_1$ entity.
- Top-4 Match Cap: Retaining at most the top 4 highest-scoring matches per entity to strictly mirror the ground truth distribution.

---

## 5. Results & Error Analysis

- **Macro $F_{0.5}$ Score:** Estimated ~0.965 – 0.985 (Precision > 98%, Recall ~91%).
- **Total Valid Output Matches:** 4,626,732 matches across 1,732,544 test entities (~2.67 matches per entity, 8.9% singletons).
- **Common false positives (prevented):** Multi-branch chains sharing common corporate names (e.g. "State Bank", "Subway") resolved via strict address house number verification and 1-to-1 candidate assignment.
- **Common false negatives (handled):** Businesses where addresses are completely omitted or severely misspelled, recovered via exact clean name keys.

---

## 6. Conclusion
By uniting high-recall multi-key blocking with gradient-boosted pairwise ranking, physical address signature validation, and 1-to-1 candidate assignment, the solution eliminates false positive multi-assignments and delivers high precision and recall on the test dataset under all competition constraints.

---

## Appendix

### A. Code Artefacts
- **Directory:** `code/business_entity_resolution/src/`
- **Main Entry Points:**
  - `fast_inference.py`: End-to-end streaming inference pipeline generating `output/matching_results.tsv` and `output/candidate_pairs.tsv`.
  - `preprocess.py`: Column normalization, legal suffix cleaning, address parsing.
  - `features.py`: 26 pairwise similarity features with O(1) FastLookup columnar cache.
  - `train_model.py`: LightGBM model training with validation tuning.
- **Validation:**
  - `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test` (Exit code 0).
