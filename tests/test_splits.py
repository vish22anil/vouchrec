"""Unit tests for dataset split protocol and leakage prevention."""

import pandas as pd

from vouchrec.evaluation.splits import create_global_temporal_split


def test_global_temporal_split_no_leakage() -> None:
    # 20 interactions with explicit monotonically increasing timestamps
    data = {
        "record_id": [f"rec_{i}" for i in range(20)],
        "source_dataset": ["test"] * 20,
        "source_category": ["cat"] * 20,
        "source_split": [None] * 20,
        "source_row": list(range(20)),
        "review_id": [None] * 20,
        "user_id": [f"U{i % 4}" for i in range(20)],
        "item_id": [f"I{i % 5}" for i in range(20)],
        "variant_asin": [None] * 20,
        "rating": [5.0] * 20,
        "review_title": ["t"] * 20,
        "review_text": ["text"] * 20,
        "timestamp": pd.date_range("2020-01-01", periods=20, freq="D", tz="UTC"),
        "timestamp_ms_raw": list(range(1000, 1020)),
        "verified_purchase": [True] * 20,
        "helpful_vote": [0] * 20,
        "n_images": [0] * 20,
        "content_hash": [f"hash_{i}" for i in range(20)],
    }
    df = pd.DataFrame(data)

    train_df, val_df, test_df, meta = create_global_temporal_split(
        df, val_fraction=0.1, test_fraction=0.1
    )

    # 1. Row counts: 20 * 0.8 = 16 train, 2 val, 2 test
    assert len(train_df) == 16
    assert len(val_df) == 2
    assert len(test_df) == 2
    assert len(train_df) + len(val_df) + len(test_df) == 20

    # 2. Strict temporal ordering (no future leakage)
    assert train_df["timestamp"].max() <= val_df["timestamp"].min()
    assert val_df["timestamp"].max() <= test_df["timestamp"].min()

    # 3. Metadata validation
    assert meta["train_interactions"] == 16
    assert meta["val_interactions"] == 2
    assert meta["test_interactions"] == 2
