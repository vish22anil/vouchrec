"""Unit and integration tests for data standardization and quality reporting."""

from pathlib import Path

import pyarrow.parquet as pq

from vouchrec.config import load_config
from vouchrec.data.quality_report import generate_quality_report
from vouchrec.data.standardize import standardize_records


def test_standardize_pipeline_with_fixtures(tmp_path: Path) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "sample_reviews.jsonl"
    prep_cfg = load_config("preprocessing")

    interim_table, proc_table, stats = standardize_records(
        raw_file_path=fixture_path,
        source_dataset="amazon_reviews_2023",
        source_category="Musical_Instruments",
        source_split=None,
        preprocessing_cfg=prep_cfg,
    )

    interim_df = interim_table.to_pandas()
    proc_df = proc_table.to_pandas()

    # 1. Total rows
    # 46 lines in sample_reviews.jsonl
    assert len(interim_df) == 46
    assert stats["total_raw"] == 46

    # 2. Duplicate detection
    # Line 40 is an exact duplicate of Line 0
    assert interim_df["is_exact_duplicate"].sum() == 1
    # Both Line 40 and Line 41 repeat the (user, item, timestamp) key of Line 0
    assert interim_df["is_key_duplicate"].sum() == 2

    # 3. Invalid rating detection (Line 42 has rating 6.0)
    assert interim_df["is_invalid_rating"].sum() == 1

    # 4. Missing user & item (Line 43 has null user, Line 44 has null asin)
    assert interim_df["is_missing_user_id"].sum() == 1
    assert interim_df["is_missing_item_id"].sum() == 1

    # 5. Bad timestamp (Line 45 has year 1973)
    assert interim_df["is_invalid_timestamp"].sum() == 1

    # 6. Empty text flag (Line 35 has text "")
    assert interim_df["is_empty_text"].sum() >= 1

    # 7. Exclusion behavior:
    # 46 total - 1 (exact dup) - 1 (key dup) - 1 (invalid rating) - 1 (missing user) - 1 (missing item) - 1 (invalid ts) = 40 valid
    assert len(proc_df) == 40
    assert stats["total_excluded"] == 6

    # 8. Check that no quality flag columns exist in processed table
    for flag in ["is_missing_user_id", "is_exact_duplicate", "is_invalid_rating"]:
        assert flag not in proc_df.columns

    # 9. Verify Parquet write and quality report generation
    interim_pq = tmp_path / "interim.parquet"
    proc_pq = tmp_path / "proc.parquet"
    pq.write_table(interim_table, interim_pq)
    pq.write_table(proc_table, proc_pq)

    report = generate_quality_report(interim_pq, proc_pq)
    assert report["volume"]["raw_records"] == 46
    assert report["volume"]["processed_records"] == 40
    assert report["volume"]["excluded_records"] == 6
    assert report["entity_counts"]["unique_users"] == 8
    assert report["rating_analysis"]["mean"] > 4.0
