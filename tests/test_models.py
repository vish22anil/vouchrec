"""Unit tests for recommendation models: Popularity, BPR-MF, and LightGCN."""

import pandas as pd
import pytest

from vouchrec.models.bpr_mf import BPRMFRecommender
from vouchrec.models.lightgcn import LightGCNRecommender
from vouchrec.models.popularity import PopularityRecommender


@pytest.fixture
def sample_interactions() -> pd.DataFrame:
    # 5 users, 4 items
    return pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u2", "u2", "u3", "u3", "u4", "u5"],
            "item_id": ["i1", "i2", "i1", "i3", "i1", "i2", "i2", "i4"],
            "rating": [5.0] * 8,
        }
    )


def test_popularity_recommender(sample_interactions: pd.DataFrame) -> None:
    model = PopularityRecommender()
    model.fit(sample_interactions)

    # i1 count = 3, i2 count = 3, i3 count = 1, i4 count = 1
    recs = model.recommend(["u1"], k=2, exclude_seen=True)
    # u1 saw i1 and i2, so should recommend i3 and i4
    assert set(recs["u1"]) == {"i3", "i4"}

    # With exclude_seen=False, i1 or i2 should appear first
    recs_all = model.recommend(["u1"], k=2, exclude_seen=False)
    assert len(recs_all["u1"]) == 2
    assert recs_all["u1"][0] in {"i1", "i2"}


def test_bpr_mf_recommender(sample_interactions: pd.DataFrame) -> None:
    model = BPRMFRecommender(
        embedding_dim=8,
        learning_rate=0.01,
        l2_reg=1e-4,
        epochs=10,
        batch_size=4,
        seed=42,
    )
    model.fit(sample_interactions)

    recs = model.recommend(["u1", "u2"], k=2, exclude_seen=True)
    assert "u1" in recs
    assert "u2" in recs
    # Exclude seen: u1 shouldn't receive i1 or i2
    for item in recs["u1"]:
        assert item not in {"i1", "i2"}

    # Pair score prediction
    scores = model.predict_scores(["u1", "u2"], ["i1", "i3"])
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)


def test_lightgcn_recommender(sample_interactions: pd.DataFrame) -> None:
    model = LightGCNRecommender(
        embedding_dim=8,
        num_layers=2,
        learning_rate=0.01,
        l2_reg=1e-4,
        epochs=10,
        batch_size=4,
        seed=42,
    )
    model.fit(sample_interactions)

    recs = model.recommend(["u1", "u2"], k=2, exclude_seen=True)
    assert "u1" in recs
    assert "u2" in recs
    # Exclude seen
    for item in recs["u1"]:
        assert item not in {"i1", "i2"}

    # Pair score prediction
    scores = model.predict_scores(["u1", "u2"], ["i1", "i3"])
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
