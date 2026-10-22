"""Unit and integration tests for Phase 4 controlled attack simulator."""

import numpy as np
import pandas as pd
import pytest

from vouchrec.attacks.config import AttackConfig
from vouchrec.attacks.evaluation import evaluate_target_item_impact, evaluate_trust_detection
from vouchrec.attacks.simulator import AttackSimulator
from vouchrec.attacks.strategies import AverageAttack, BandwagonAttack, RandomAttack
from vouchrec.data.schema import LABEL_COLUMNS, assert_no_label_columns


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """Fixture containing 8 users and 8 items."""
    return pd.DataFrame(
        {
            "record_id": [f"rec_{i}" for i in range(24)],
            "source_dataset": ["amazon_reviews_2023"] * 24,
            "source_category": ["Musical_Instruments"] * 24,
            "source_split": [None] * 24,
            "source_row": list(range(24)),
            "review_id": [None] * 24,
            "user_id": [f"U{i % 6:02d}" for i in range(24)],
            "item_id": [f"P{i % 8:03d}" for i in range(24)],
            "variant_asin": [None] * 24,
            "rating": [5.0, 4.0, 5.0, 3.0, 4.0, 5.0] * 4,
            "review_title": ["Good product"] * 24,
            "review_text": ["Quality is great and sound is clear."] * 24,
            "timestamp": pd.date_range("2020-01-01", periods=24, freq="D", tz="UTC"),
            "timestamp_ms_raw": [1577836800000 + i * 86400000 for i in range(24)],
            "verified_purchase": [True] * 24,
            "helpful_vote": [0] * 24,
            "n_images": [0] * 24,
            "content_hash": [f"hash_{i}" for i in range(24)],
        }
    )


# ==============================================================================
# 1. Configuration Validation Tests
# ==============================================================================


def test_attack_config_validation() -> None:
    # Valid config
    cfg = AttackConfig(strategy="random", objective="push", attack_budget_fraction=0.05)
    assert cfg.target_rating == 5.0
    assert len(cfg.config_hash()) == 8

    # Nuke target rating
    cfg_nuke = AttackConfig(strategy="random", objective="nuke")
    assert cfg_nuke.target_rating == 1.0

    # Invalid strategy
    with pytest.raises(ValueError):
        AttackConfig(strategy="invalid_strat")

    # Invalid objective
    with pytest.raises(ValueError):
        AttackConfig(objective="destroy")

    # Invalid budget
    with pytest.raises(ValueError):
        AttackConfig(attack_budget_fraction=0.0)

    with pytest.raises(ValueError):
        AttackConfig(attack_budget_fraction=1.5)

    # Invalid filler size
    with pytest.raises(ValueError):
        AttackConfig(filler_size=-1)


# ==============================================================================
# 2. Strategy Generation Tests
# ==============================================================================


def test_random_attack_strategy() -> None:
    cfg = AttackConfig(strategy="random", objective="push", filler_size=3)
    strat = RandomAttack(cfg)
    rng = np.random.default_rng(42)

    profile = strat.generate_profile(
        target_item="P001",
        available_items=["P001", "P002", "P003", "P004"],
        item_stats={"P002": (4.0, 0.5), "P003": (3.5, 1.0)},
        popular_items=["P002", "P003"],
        rng=rng,
    )

    assert len(profile) == 4  # 1 target + 3 fillers
    assert profile[0][0] == "P001"
    assert profile[0][1] == 5.0  # push target rating
    # Filler ratings must be valid ratings
    for _, r, _ in profile[1:]:
        assert r in cfg.valid_ratings


def test_average_attack_strategy() -> None:
    cfg = AttackConfig(strategy="average", objective="nuke", filler_size=2)
    strat = AverageAttack(cfg)
    rng = np.random.default_rng(42)

    profile = strat.generate_profile(
        target_item="P001",
        available_items=["P001", "P002", "P003"],
        item_stats={"P002": (4.0, 0.2), "P003": (3.0, 0.2)},
        popular_items=["P002"],
        rng=rng,
    )

    assert len(profile) == 3  # 1 target + 2 fillers
    assert profile[0][0] == "P001"
    assert profile[0][1] == 1.0  # nuke target rating
    for _, r, _ in profile:
        assert r in cfg.valid_ratings


def test_bandwagon_attack_strategy() -> None:
    cfg = AttackConfig(strategy="bandwagon", objective="push", filler_size=3, bandwagon_items_count=2)
    strat = BandwagonAttack(cfg)
    rng = np.random.default_rng(42)

    profile = strat.generate_profile(
        target_item="P001",
        available_items=["P001", "P002", "P003", "P004", "P005"],
        item_stats={"P002": (4.5, 0.5), "P003": (4.2, 0.5)},
        popular_items=["P002", "P003"],
        rng=rng,
    )

    items_in_profile = [item for item, _, _ in profile]
    assert "P001" in items_in_profile
    assert "P002" in items_in_profile
    assert "P003" in items_in_profile
    assert len(profile) == 4  # 1 target + 2 bandwagon + 1 remaining filler


# ==============================================================================
# 3. Simulator Functionality & Non-Collision Tests
# ==============================================================================


def test_simulator_non_collision_and_labels(sample_train_df: pd.DataFrame) -> None:
    cfg = AttackConfig(strategy="random", objective="push", attack_budget_fraction=0.20, seed=123)
    sim = AttackSimulator(cfg)
    res = sim.simulate(sample_train_df)

    real_users = set(sample_train_df["user_id"])
    synth_users = set(res.synthetic_reviews_df["user_id"])

    # 1. Zero user collision
    assert real_users.isdisjoint(synth_users)

    # 2. Separate labels table
    labels_df = res.ground_truth_labels_df
    assert len(labels_df) == len(res.attacked_train_df)
    assert set(labels_df["label_value"]) == {0, 1}
    assert (labels_df.loc[labels_df["label_value"] == 1, "label_origin"] == "simulation").all()

    # 3. Anti-leakage: attack columns must NOT exist in the review table
    for forbidden in LABEL_COLUMNS:
        assert forbidden not in res.attacked_train_df.columns
    assert_no_label_columns(res.attacked_train_df.columns)

    # 4. Ratings bounded
    assert (res.synthetic_reviews_df["rating"] >= 1.0).all()
    assert (res.synthetic_reviews_df["rating"] <= 5.0).all()


def test_simulator_determinism(sample_train_df: pd.DataFrame) -> None:
    cfg = AttackConfig(strategy="average", objective="push", seed=999)
    res1 = AttackSimulator(cfg).simulate(sample_train_df)
    res2 = AttackSimulator(cfg).simulate(sample_train_df)

    pd.testing.assert_frame_equal(res1.synthetic_reviews_df, res2.synthetic_reviews_df)
    pd.testing.assert_frame_equal(res1.ground_truth_labels_df, res2.ground_truth_labels_df)


def test_simulator_burst_timing(sample_train_df: pd.DataFrame) -> None:
    cfg = AttackConfig(
        strategy="random",
        burst_window_hours=24.0,
        burst_start_timestamp=1600000000000,
        seed=42,
    )
    sim = AttackSimulator(cfg)
    res = sim.simulate(sample_train_df)

    window_ms = 24 * 3600 * 1000
    for ts in res.synthetic_reviews_df["timestamp_ms_raw"]:
        assert 1600000000000 <= ts <= 1600000000000 + window_ms


def test_simulator_empty_df_raises() -> None:
    cfg = AttackConfig()
    sim = AttackSimulator(cfg)
    with pytest.raises(ValueError):
        sim.simulate(pd.DataFrame())


# ==============================================================================
# 4. Evaluation Module Tests
# ==============================================================================


def test_evaluate_target_impact() -> None:
    clean_recs = {"u1": ["item_A", "item_B"], "u2": ["item_C", "item_D"]}
    # In attacked recs, item_X was pushed into top 10 for both users
    attacked_recs = {"u1": ["item_X", "item_A"], "u2": ["item_X", "item_C"]}

    impact = evaluate_target_item_impact(clean_recs, attacked_recs, target_item="item_X", top_k=10)
    assert impact["target_hitrate_at_10_clean"] == 0.0
    assert impact["target_hitrate_at_10_attacked"] == 1.0
    assert impact["target_hitrate_delta"] == 1.0
    assert impact["target_rank_improvement"] > 0.0


def test_evaluate_trust_detection_on_labels(sample_train_df: pd.DataFrame) -> None:
    cfg = AttackConfig(strategy="random", objective="push", seed=101)
    sim = AttackSimulator(cfg)
    res = sim.simulate(sample_train_df)

    detection = evaluate_trust_detection(res.attacked_train_df, res.ground_truth_labels_df)
    assert "mean_trust_real_reviews" in detection
    assert "mean_trust_synthetic_reviews" in detection
    assert "trust_separation_delta" in detection
    assert 0.0 <= detection["trust_detection_roc_auc"] <= 1.0
