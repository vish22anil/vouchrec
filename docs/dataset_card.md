# Dataset Card for Amazon Reviews 2023 (Musical Instruments) in VouchRec

## Dataset Details

### Dataset Description
The dataset used in this project is the **Musical Instruments** category from **Amazon Reviews 2023** (Hou et al., 2024). It serves as the physical product benchmark for studying trust-gated collaborative filtering and ranking vulnerability under synthetic and coordinated review manipulation.

- **Curated by:** McAuley Lab (UC San Diego)
- **Paper:** *Bridging Language and Items for Retrieval and Recommendation* (Hou et al., arXiv:2403.03952, 2024)
- **Homepage:** [https://amazon-reviews-2023.github.io/](https://amazon-reviews-2023.github.io/)
- **Hugging Face Repository:** [McAuley-Lab/Amazon-Reviews-2023](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)
- **Category:** `Musical_Instruments`
- **Official Raw Statistics:** ~3.0M ratings, ~1.8M users, ~213.6K items (5-core subset: 57.4K users, 24.6K items, 511.8K ratings)
- **License / Usage:** No permissive redistribution license is declared on the Hugging Face dataset card. Per Hou et al. (2024), data is strictly utilized for academic and scientific research; raw and processed record-level tables are kept locally and omitted from version control (see `.gitignore`).

## Schema & Structure

Raw reviews are ingested from the official source (`Musical_Instruments.jsonl.gz`) and standardized into Apache Arrow / Parquet format.

### Canonical Review Table Schema
| Field Name | Arrow Type | Description |
|---|---|---|
| `record_id` | `pa.string()` | Surrogate key: `<source_dataset>/<category>/<source_row>` |
| `source_dataset` | `pa.string()` | Identifies the corpus (e.g., `amazon_reviews_2023`) |
| `source_category` | `pa.string()` | Category name (`Musical_Instruments`) |
| `source_split` | `pa.string()` | Upstream split if present, otherwise null |
| `source_row` | `pa.int64()` | 0-based row offset in the raw source stream |
| `review_id` | `pa.string()` | Source review ID (null for Amazon 2023) |
| `user_id` | `pa.string()` | Unique reviewer identifier |
| `item_id` | `pa.string()` | Product identifier (mapped from `parent_asin` per official guidelines) |
| `variant_asin` | `pa.string()` | Specific variant ASIN (`asin` from raw) |
| `rating` | `pa.float64()` | Star rating (1.0 to 5.0) |
| `review_title` | `pa.string()` | Review headline / summary |
| `review_text` | `pa.string()` | Body text of review |
| `timestamp` | `pa.timestamp('ms', tz='UTC')` | Parsed UTC timestamp |
| `timestamp_ms_raw` | `pa.int64()` | Source Unix millisecond timestamp |
| `verified_purchase` | `pa.bool_()` | Verified purchase indicator (never imputed) |
| `helpful_vote` | `pa.int64()` | Helpful vote count at crawl time |
| `n_images` | `pa.int64()` | Count of attached review photos |
| `content_hash` | `pa.string()` | BLAKE2b hash of raw canonicalized JSON |

### Quality Flags (Interim Schema Only)
The interim dataset retains all rows and annotates each record with boolean failure flags:
- `is_missing_user_id`: `user_id` is null or empty
- `is_missing_item_id`: `item_id` (`parent_asin` or `asin`) is null or empty
- `is_invalid_rating`: rating is missing or $\notin \{1.0, 2.0, 3.0, 4.0, 5.0\}$
- `is_invalid_timestamp`: timestamp is unparsable or outside range `[1995-01-01, 2023-12-31]`
- `is_empty_text`: review body is null or whitespace
- `is_exact_duplicate`: identical content hash to a previously seen record
- `is_key_duplicate`: identical `(user_id, item_id, timestamp_ms_raw)` tuple to an earlier record

## Preprocessing & Curation Rationale

1. **Item ID Mapping:** In Amazon Reviews 2023, multiple variant listings (colors, sizes, bundles) each have distinct child ASINs but share a single `parent_asin`. In accordance with official dataset instructions, `parent_asin` is designated as the canonical `item_id`.
2. **Deterministic Deduplication:** Exact duplicates and key duplicates are identified in linear streaming passes. The first occurrence in source line order is preserved.
3. **Separate Storage Layers:**
   - `data/raw/`: Original `.jsonl.gz` plus `raw_manifest.json` recording checksums, URLs, and timestamps.
   - `data/interim/`: Parquet tables retaining all records with quality flags.
   - `data/processed/`: Parquet tables retaining only records passing all quality criteria.
   - `data/splits/`: Leakage-free temporal splits (`train.parquet`, `val.parquet`, `test.parquet`).
4. **Integrity & Anti-Leakage Constraints:** Ground truth attack labels and evaluation splits are strictly isolated. Ordinary reviews are never labeled as fake or malicious without authenticated external ground truth.

## Citation

```bibtex
@article{hou2024bridging,
  title={Bridging Language and Items for Retrieval and Recommendation},
  author={Hou, Yupeng and Li, Jiacheng and He, Zhankui and Yan, An and Chen, Xiusi and McAuley, Julian},
  journal={arXiv preprint arXiv:2403.03952},
  year={2024}
}
```
