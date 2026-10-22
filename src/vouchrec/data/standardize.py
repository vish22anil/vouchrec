"""Standardize raw reviews into canonical schema, compute quality flags, and produce interim & processed datasets."""

from __future__ import annotations

import argparse
import datetime
import logging
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa

from vouchrec.config import load_config, resolve_data_paths
from vouchrec.data.io_utils import compute_content_hash, save_parquet, stream_jsonl
from vouchrec.data.schema import (
    CANONICAL_COLUMNS,
    FLAG_COLUMNS,
    review_schema,
    validate_schema,
)
from vouchrec.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_timestamp_ms(
    val: Any, min_ms: int, max_ms: int
) -> tuple[datetime.datetime | None, int | None, bool]:
    """Parse raw timestamp (assumed Unix ms) and validate range."""
    if val is None:
        return None, None, True
    try:
        val_int = int(val)
    except (ValueError, TypeError):
        return None, None, True

    # Check bounds
    if val_int < min_ms or val_int > max_ms:
        return None, val_int, True

    try:
        dt = datetime.datetime.fromtimestamp(val_int / 1000.0, tz=datetime.UTC)
        return dt, val_int, False
    except (ValueError, OSError, OverflowError):
        return None, val_int, True


def standardize_records(
    raw_file_path: Path,
    source_dataset: str,
    source_category: str,
    source_split: str | None,
    preprocessing_cfg: dict[str, Any],
    max_records: int | None = None,
) -> tuple[pa.Table, pa.Table, dict[str, Any]]:
    """Process raw JSON lines, return (interim_table, processed_table, audit_stats)."""
    valid_ratings = set(
        preprocessing_cfg.get("ratings", {}).get("valid_values", [1.0, 2.0, 3.0, 4.0, 5.0])
    )

    ts_cfg = preprocessing_cfg.get("timestamps", {})
    min_utc = datetime.datetime.fromisoformat(ts_cfg.get("min_utc", "1995-01-01T00:00:00+00:00"))
    max_utc = datetime.datetime.fromisoformat(ts_cfg.get("max_utc", "2023-12-31T23:59:59+00:00"))
    min_ms = int(min_utc.timestamp() * 1000)
    max_ms = int(max_utc.timestamp() * 1000)

    exclusions_cfg = preprocessing_cfg.get("exclusions", {})

    seen_content_hashes: set[str] = set()
    seen_keys: set[tuple[str, str, int]] = set()

    # Column storage arrays
    cols: dict[str, list[Any]] = {col: [] for col in CANONICAL_COLUMNS}
    flag_cols: dict[str, list[bool]] = {col: [] for col in FLAG_COLUMNS}

    drop_counts: Counter[str] = Counter()
    total_raw = 0

    for source_row, raw in stream_jsonl(raw_file_path, max_records=max_records):
        total_raw += 1
        record_id = f"{source_dataset}/{source_category}/{source_row}"
        review_id = raw.get("review_id")  # usually None for Amazon 2023

        user_id_raw = raw.get("user_id")
        user_id = (
            str(user_id_raw).strip()
            if user_id_raw is not None and str(user_id_raw).strip()
            else None
        )
        is_missing_user_id = user_id is None

        # parent_asin is item_id; fallback to asin
        item_id_raw = raw.get("parent_asin") or raw.get("asin")
        item_id = (
            str(item_id_raw).strip()
            if item_id_raw is not None and str(item_id_raw).strip()
            else None
        )
        is_missing_item_id = item_id is None

        variant_asin = str(raw.get("asin")).strip() if raw.get("asin") else None

        rating_raw = raw.get("rating")
        try:
            rating = float(rating_raw) if rating_raw is not None else None
            is_invalid_rating = rating is None or rating not in valid_ratings
        except (ValueError, TypeError):
            rating = None
            is_invalid_rating = True

        title_raw = raw.get("title")
        review_title = str(title_raw).strip() if title_raw is not None else None

        text_raw = raw.get("text")
        review_text = str(text_raw) if text_raw is not None else None
        is_empty_text = review_text is None or not review_text.strip()

        dt, ts_ms, is_invalid_ts = parse_timestamp_ms(raw.get("timestamp"), min_ms, max_ms)

        vp_raw = raw.get("verified_purchase")
        verified_purchase = bool(vp_raw) if isinstance(vp_raw, bool) else None

        try:
            helpful_vote = int(raw.get("helpful_vote", 0))
        except (ValueError, TypeError):
            helpful_vote = 0

        images_raw = raw.get("images")
        n_images = len(images_raw) if isinstance(images_raw, list) else 0

        content_hash = compute_content_hash(raw)
        is_exact_dup = content_hash in seen_content_hashes
        if not is_exact_dup:
            seen_content_hashes.add(content_hash)

        is_key_dup = False
        if user_id and item_id and ts_ms is not None:
            key = (user_id, item_id, ts_ms)
            if key in seen_keys:
                is_key_dup = True
            else:
                seen_keys.add(key)

        # Store canonical fields
        cols["record_id"].append(record_id)
        cols["source_dataset"].append(source_dataset)
        cols["source_category"].append(source_category)
        cols["source_split"].append(source_split)
        cols["source_row"].append(source_row)
        cols["review_id"].append(review_id)
        cols["user_id"].append(user_id)
        cols["item_id"].append(item_id)
        cols["variant_asin"].append(variant_asin)
        cols["rating"].append(rating)
        cols["review_title"].append(review_title)
        cols["review_text"].append(review_text)
        cols["timestamp"].append(dt)
        cols["timestamp_ms_raw"].append(ts_ms)
        cols["verified_purchase"].append(verified_purchase)
        cols["helpful_vote"].append(helpful_vote)
        cols["n_images"].append(n_images)
        cols["content_hash"].append(content_hash)

        # Store quality flags
        flag_cols["is_missing_user_id"].append(is_missing_user_id)
        flag_cols["is_missing_item_id"].append(is_missing_item_id)
        flag_cols["is_invalid_rating"].append(is_invalid_rating)
        flag_cols["is_invalid_timestamp"].append(is_invalid_ts)
        flag_cols["is_empty_text"].append(is_empty_text)
        flag_cols["is_exact_duplicate"].append(is_exact_dup)
        flag_cols["is_key_duplicate"].append(is_key_dup)

        # Track exclusion counts
        if is_missing_user_id:
            drop_counts["missing_user_id"] += 1
        if is_missing_item_id:
            drop_counts["missing_item_id"] += 1
        if is_invalid_rating:
            drop_counts["invalid_rating"] += 1
        if is_invalid_ts:
            drop_counts["invalid_timestamp"] += 1
        if is_exact_dup:
            drop_counts["exact_duplicate"] += 1
        if is_key_dup:
            drop_counts["key_duplicate"] += 1
        if is_empty_text:
            drop_counts["empty_text"] += 1

    interim_schema = review_schema(with_flags=True)
    all_arrays: list[pa.Array] = []
    for f in interim_schema:
        if f.name in cols:
            all_arrays.append(pa.array(cols[f.name], type=f.type))
        else:
            all_arrays.append(pa.array(flag_cols[f.name], type=f.type))

    interim_table = pa.Table.from_arrays(all_arrays, schema=interim_schema)
    validate_schema(interim_table.schema, interim_schema)

    # Filter for processed table
    keep_mask = [True] * total_raw
    for idx in range(total_raw):
        if (
            exclusions_cfg.get("missing_user_id", True)
            and flag_cols["is_missing_user_id"][idx]
            or exclusions_cfg.get("missing_item_id", True)
            and flag_cols["is_missing_item_id"][idx]
            or exclusions_cfg.get("invalid_rating", True)
            and flag_cols["is_invalid_rating"][idx]
            or exclusions_cfg.get("invalid_timestamp", True)
            and flag_cols["is_invalid_timestamp"][idx]
            or exclusions_cfg.get("exact_duplicate", True)
            and flag_cols["is_exact_duplicate"][idx]
            or exclusions_cfg.get("key_duplicate", True)
            and flag_cols["is_key_duplicate"][idx]
            or exclusions_cfg.get("empty_text", False)
            and flag_cols["is_empty_text"][idx]
        ):
            keep_mask[idx] = False

    proc_schema = review_schema(with_flags=False)
    proc_arrays = [
        pa.array(
            [cols[name][i] for i in range(total_raw) if keep_mask[i]],
            type=proc_schema.field(name).type,
        )
        for name in CANONICAL_COLUMNS
    ]
    proc_table = pa.Table.from_arrays(proc_arrays, schema=proc_schema)
    validate_schema(proc_table.schema, proc_schema)

    audit_stats = {
        "total_raw": total_raw,
        "total_interim": len(interim_table),
        "total_processed": len(proc_table),
        "total_excluded": total_raw - len(proc_table),
        "exclusion_breakdown": dict(drop_counts),
    }

    return interim_table, proc_table, audit_stats


def run_standardization(
    raw_path: Path,
    interim_dir: Path,
    processed_dir: Path,
    data_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
    category: str | None = None,
    max_records: int | None = None,
) -> dict[str, Any]:
    """Execute standardization from raw file to interim and processed parquet files."""
    ar_cfg = data_config.get("amazon_reviews", {})
    cat = category or ar_cfg.get("category", "Musical_Instruments")
    source_dataset = ar_cfg.get("source_dataset", "amazon_reviews_2023")

    logger.info("Starting standardization of %s (max_records=%s)", raw_path, max_records)
    interim_table, proc_table, stats = standardize_records(
        raw_file_path=raw_path,
        source_dataset=source_dataset,
        source_category=cat,
        source_split=None,
        preprocessing_cfg=preprocessing_config,
        max_records=max_records,
    )

    interim_out = interim_dir / cat / "reviews_interim.parquet"
    save_parquet(interim_table, interim_out)
    logger.info("Saved interim table (%d rows) to %s", len(interim_table), interim_out)

    proc_out = processed_dir / cat / "reviews.parquet"
    save_parquet(proc_table, proc_out)
    logger.info("Saved processed table (%d rows) to %s", len(proc_table), proc_out)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standardize raw reviews into canonical interim & processed formats."
    )
    parser.add_argument(
        "--raw-file", type=str, default=None, help="Path to raw jsonl/jsonl.gz file"
    )
    parser.add_argument("--category", type=str, default=None, help="Dataset category")
    parser.add_argument(
        "--max-records", type=int, default=None, help="Optional limit on records to process"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    data_cfg = load_config("data")
    prep_cfg = load_config("preprocessing")
    paths = resolve_data_paths(data_cfg)

    cat = args.category or data_cfg.get("amazon_reviews", {}).get("category", "Musical_Instruments")
    raw_file = Path(args.raw_file) if args.raw_file else paths.raw / cat / f"{cat}.jsonl.gz"

    if not raw_file.exists():
        # Check uncompressed fallback
        uncomp = raw_file.with_suffix("")
        if uncomp.exists():
            raw_file = uncomp
        else:
            raise FileNotFoundError(f"Raw review file not found: {raw_file}")

    run_standardization(
        raw_path=raw_file,
        interim_dir=paths.interim,
        processed_dir=paths.processed,
        data_config=data_cfg,
        preprocessing_config=prep_cfg,
        category=cat,
        max_records=args.max_records,
    )


if __name__ == "__main__":
    main()
