"""Reproducible dataset splitting with leakage prevention."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from vouchrec.config import load_config, resolve_data_paths
from vouchrec.data.io_utils import save_parquet
from vouchrec.data.schema import review_schema, validate_schema
from vouchrec.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def create_global_temporal_split(
    df: pd.DataFrame,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
    tie_break_col: str = "source_row",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split interactions globally by timestamp, preventing future-to-past leakage."""
    if df.empty:
        raise ValueError("Cannot split an empty dataframe.")

    # Sort interactions strictly by timestamp and tie breaker
    sort_cols = ["timestamp"]
    if tie_break_col in df.columns:
        sort_cols.append(tie_break_col)
    df_sorted = df.sort_values(by=sort_cols).reset_index(drop=True)

    n_total = len(df_sorted)
    val_cut_idx = int(n_total * (1.0 - val_fraction - test_fraction))
    test_cut_idx = int(n_total * (1.0 - test_fraction))

    train_df = df_sorted.iloc[:val_cut_idx].copy()
    val_df = df_sorted.iloc[val_cut_idx:test_cut_idx].copy()
    test_df = df_sorted.iloc[test_cut_idx:].copy()

    val_cutoff_ts = df_sorted.iloc[val_cut_idx]["timestamp"]
    test_cutoff_ts = df_sorted.iloc[test_cut_idx]["timestamp"]

    train_users = set(train_df["user_id"])
    train_items = set(train_df["item_id"])

    val_users = set(val_df["user_id"])
    test_users = set(test_df["user_id"])

    split_metadata = {
        "split_method": "global_temporal",
        "total_interactions": n_total,
        "train_interactions": len(train_df),
        "val_interactions": len(val_df),
        "test_interactions": len(test_df),
        "val_cutoff_timestamp": str(val_cutoff_ts),
        "test_cutoff_timestamp": str(test_cutoff_ts),
        "train_unique_users": len(train_users),
        "train_unique_items": len(train_items),
        "val_users_in_train": len(val_users & train_users),
        "test_users_in_train": len(test_users & train_users),
        "val_items_in_train": len(set(val_df["item_id"]) & train_items),
        "test_items_in_train": len(set(test_df["item_id"]) & train_items),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }

    return train_df, val_df, test_df, split_metadata


def run_split(
    processed_parquet_path: Path,
    splits_output_dir: Path,
    experiment_config: dict[str, Any],
) -> dict[str, Any]:
    """Execute splitting on processed parquet and write train/val/test splits."""
    logger.info("Reading processed dataset from %s", processed_parquet_path)
    table = pq.read_table(processed_parquet_path)
    df = table.to_pandas()

    proto = experiment_config.get("split_protocol", {})
    val_frac = float(proto.get("validation_fraction", 0.1))
    test_frac = float(proto.get("test_fraction", 0.1))
    tie_break = str(proto.get("tie_break_column", "source_row"))

    train_df, val_df, test_df, meta = create_global_temporal_split(
        df,
        val_fraction=val_frac,
        test_fraction=test_frac,
        tie_break_col=tie_break,
    )

    splits_output_dir.mkdir(parents=True, exist_ok=True)
    schema = review_schema(with_flags=False)

    import pyarrow as pa

    train_table = pa.Table.from_pandas(train_df, schema=schema, preserve_index=False)
    val_table = pa.Table.from_pandas(val_df, schema=schema, preserve_index=False)
    test_table = pa.Table.from_pandas(test_df, schema=schema, preserve_index=False)

    validate_schema(train_table.schema, schema)
    validate_schema(val_table.schema, schema)
    validate_schema(test_table.schema, schema)

    save_parquet(train_table, splits_output_dir / "train.parquet")
    save_parquet(val_table, splits_output_dir / "val.parquet")
    save_parquet(test_table, splits_output_dir / "test.parquet")

    info_path = splits_output_dir / "split_info.json"
    info_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    logger.info(
        "Splits created: train=%d, val=%d, test=%d. Saved to %s",
        len(train_df),
        len(val_df),
        len(test_df),
        splits_output_dir,
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create reproducible train/validation/test splits."
    )
    parser.add_argument("--category", type=str, default=None, help="Dataset category")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    data_cfg = load_config("data")
    exp_cfg = load_config("experiment")
    paths = resolve_data_paths(data_cfg)

    cat = args.category or data_cfg.get("amazon_reviews", {}).get("category", "Musical_Instruments")
    proc_file = paths.processed / cat / "reviews.parquet"
    splits_dir = paths.splits / cat

    run_split(proc_file, splits_dir, exp_cfg)


if __name__ == "__main__":
    main()
