# VouchRec Data Quality and Audit Report

## 1. Provenance and Overview

- **Dataset Source:** `amazon_reviews_2023`
- **Category:** `Musical_Instruments`
- **Source URL:** N/A
- **Downloaded Timestamp (UTC):** `N/A`
- **Raw File Size:** `0 bytes`
- **Raw File SHA-256:** `N/A`

## 2. Record Volume and Cleaning Exclusions

| Metric | Count | Percentage |
|---|---|---|
| Raw Records | 46 | 100.00% |
| Excluded Records | 6 | 13.04% |
| Processed (Retained) | 40 | 86.96% |

### Quality Flag Audit Breakdown

| Flag | Count | Meaning |
|---|---|---|
| `is_missing_user_id` | 1 | Quality failure flag |
| `is_missing_item_id` | 1 | Quality failure flag |
| `is_invalid_rating` | 1 | Quality failure flag |
| `is_invalid_timestamp` | 1 | Quality failure flag |
| `is_empty_text` | 1 | Quality failure flag |
| `is_exact_duplicate` | 1 | Quality failure flag |
| `is_key_duplicate` | 2 | Quality failure flag |

## 3. Interaction Graph and Sparsity

| Metric | Value |
|---|---|
| Unique Users | 8 |
| Unique Items | 8 |
| Processed Interactions | 40 |
| Bipartite Graph Sparsity | 0.375000 |
| Users with >= 5 reviews | 8 |
| Items with >= 5 reviews | 5 |
| Median User Reviews | 5.0 (max: 5) |
| Median Item Reviews | 5.0 (max: 7) |

## 4. Rating and Verification Distribution

- **Rating Mean (Std):** 4.55 (±0.55)
- **Verified Purchase Share:** 95.00%
- **Temporal Range:** `2018-01-03T17:20:00+00:00` to `2019-07-13T06:40:00+00:00`

### Rating Value Frequencies

| Rating | Count | Share |
|---|---|---|
| 3.0 Stars | 1 | 2.50% |
| 4.0 Stars | 16 | 40.00% |
| 5.0 Stars | 23 | 57.50% |
