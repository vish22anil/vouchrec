"""Tests for Trust-Gated LightGCN implementation (Phase 5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vouchrec.data.schema import SchemaError
from vouchrec.models.lightgcn import LightGCNRecommender
from vouchrec.models.trust_gated_lightgcn import (
    TrustGatedLightGCN,
    compute_edge_weights,
    tune_hard_filter_threshold,
)


@pytest.fixture
def sample_interactions_df() -> pd.DataFrame:
    """Fixture containing 6 users, 6 items, and 18 interactions."""
    records = []
    user_list = [f"u_{i}" for i in range(6)]
    item_list = [f"i_{j}" for j in range(6)]

    # Deterministic interaction graph
    edge_pairs = [
        (0, 0), (0, 1), (0, 2),
        (1, 1), (1, 2), (1, 3),
        (2, 2), (2, 3), (2, 4),
        (3, 0), (3, 3), (3, 5),
        (4, 1), (4, 4), (4, 5),
        (5, 2), (5, 4), (5, 5),
    ]

    for idx, (u_i, i_j) in enumerate(edge_pairs):
        records.append({
            "record_id": f"rec_{idx:03d}",
            "user_id": user_list[u_i],
            "item_id": item_list[i_j],
            "rating": 5.0,
            "timestamp": 1600000000000 + idx * 100000,
            "review_text": f"Great product review number {idx} with high detail and excellent content.",
        })
    return pd.DataFrame(records)


def test_edge_weight_computation_boundaries() -> None:
    """Test compute_edge_weights handles NaN, Inf, out-of-range, and modes."""
    raw_scores = np.array([-0.5, 0.0, 0.3, 0.7, 1.0, 1.5, np.nan, np.inf])

    # 1. Soft linear
    w_linear = compute_edge_weights(raw_scores, mode="soft", soft_function="linear", default_trust=0.5)
    assert np.all(np.isfinite(w_linear))
    assert np.all(w_linear >= 1e-4)
    assert np.all(w_linear <= 1.0)
    # NaN and Inf should become default_trust (0.5)
    assert w_linear[6] == 0.5
    assert w_linear[7] == 0.5
    # Clipped boundaries
    assert w_linear[0] == 1e-4  # -0.5 clipped to 0, then clipped to min_edge_weight
    assert w_linear[5] == 1.0   # 1.5 clipped to 1.0

    # 2. Soft power (w = tau^2)
    w_power = compute_edge_weights(raw_scores, mode="soft", soft_function="power", power_p=2.0)
    assert np.isclose(w_power[3], 0.7 ** 2, atol=1e-5)

    # 3. Soft sigmoid
    w_sig = compute_edge_weights(raw_scores, mode="soft", soft_function="sigmoid", sigmoid_k=10.0, sigmoid_center=0.5)
    assert np.all(np.isfinite(w_sig))
    assert np.isclose(w_sig[6], 0.5, atol=1e-4)  # default_trust=0.5 at center=0.5 gives 0.5

    # 4. Uniform
    w_uniform = compute_edge_weights(raw_scores, mode="uniform")
    assert np.all(w_uniform == 1.0)

    # 5. Hard filtering
    w_hard = compute_edge_weights(raw_scores, mode="hard", hard_threshold=0.6, default_trust=0.5)
    assert w_hard[2] == 0.0  # 0.3 < 0.6
    assert w_hard[3] == 1.0  # 0.7 >= 0.6
    assert w_hard[6] == 0.0  # NaN -> 0.5 < 0.6


def test_edge_to_trust_score_alignment(sample_interactions_df: pd.DataFrame) -> None:
    """Verify that edge weights align 1:1 with record_id even under permutation."""
    df = sample_interactions_df.copy()
    n = len(df)

    # Assign distinct synthetic trust scores
    distinct_scores = np.linspace(0.1, 0.95, n, dtype=np.float32)
    trust_df = pd.DataFrame({
        "record_id": df["record_id"].values,
        "provisional_trust_score": distinct_scores,
    })

    # Permute trust_df randomly
    permuted_trust_df = trust_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    model = TrustGatedLightGCN(
        embedding_dim=16,
        num_layers=2,
        epochs=1,
        weighting_mode="soft",
        soft_function="linear",
    )
    model.fit(df, trust_scores_df=permuted_trust_df)

    assert model.edge_weights_ is not None
    # Joined edge weights must match original df order of distinct_scores
    np.testing.assert_allclose(model.edge_weights_, distinct_scores, atol=1e-4)


def test_weighted_adjacency_construction_and_normalization(sample_interactions_df: pd.DataFrame) -> None:
    """Verify normalized adjacency matrix is symmetric, finite, and bounded."""
    df = sample_interactions_df.copy()
    num_users = len(df["user_id"].unique())
    num_items = len(df["item_id"].unique())
    num_nodes = num_users + num_items

    model = TrustGatedLightGCN(
        embedding_dim=16,
        num_layers=2,
        epochs=1,
        weighting_mode="soft",
        soft_function="linear",
    )
    model.fit(df)

    assert model.model_ is not None
    norm_adj = model.model_.norm_adj
    assert norm_adj.shape == (num_nodes, num_nodes)
    assert norm_adj.is_sparse

    # Convert sparse tensor to dense to verify mathematical properties
    dense_adj = norm_adj.to_dense().numpy()

    # 1. Symmetry: A_tilde == A_tilde^T
    np.testing.assert_allclose(dense_adj, dense_adj.T, atol=1e-5)

    # 2. Finite non-negative entries
    assert np.all(np.isfinite(dense_adj))
    assert np.all(dense_adj >= 0.0)

    # 3. Bipartite structure: diagonal blocks (users-users and items-items) must be strictly 0
    np.testing.assert_allclose(dense_adj[:num_users, :num_users], 0.0)
    np.testing.assert_allclose(dense_adj[num_users:, num_users:], 0.0)

    # 4. Spectral property: symmetric normalized adjacency has spectral radius bounded by 1.0
    eigvals = np.linalg.eigvalsh(dense_adj)
    spectral_radius = float(np.max(np.abs(eigvals)))
    assert spectral_radius <= 1.0001


def test_anti_leakage_guard(sample_interactions_df: pd.DataFrame) -> None:
    """Verify that providing label columns raises SchemaError."""
    df = sample_interactions_df.copy()
    df["is_attack"] = 1

    model = TrustGatedLightGCN(embedding_dim=16, epochs=1)
    with pytest.raises(SchemaError):
        model.fit(df)

    # Also test external trust table leakage
    df_clean = sample_interactions_df.copy()
    leaked_trust = pd.DataFrame({
        "record_id": df_clean["record_id"],
        "provisional_trust_score": 0.8,
        "attack_type": "push",
    })
    with pytest.raises(SchemaError):
        model.fit(df_clean, trust_scores_df=leaked_trust)


def test_deterministic_training_fixed_seeds(sample_interactions_df: pd.DataFrame) -> None:
    """Verify bitwise identical recommendation scores when seed is fixed."""
    df = sample_interactions_df.copy()
    users = ["u_0", "u_1", "u_2"]

    model1 = TrustGatedLightGCN(embedding_dim=16, num_layers=2, epochs=3, seed=20261021)
    model1.fit(df)
    recs1 = model1.recommend(users, k=5, exclude_seen=False)
    scores1 = model1.predict_scores(["u_0", "u_1"], ["i_0", "i_1"])

    model2 = TrustGatedLightGCN(embedding_dim=16, num_layers=2, epochs=3, seed=20261021)
    model2.fit(df)
    recs2 = model2.recommend(users, k=5, exclude_seen=False)
    scores2 = model2.predict_scores(["u_0", "u_1"], ["i_0", "i_1"])

    assert recs1 == recs2
    np.testing.assert_allclose(scores1, scores2, atol=1e-5)


def test_vanilla_lightgcn_behavior_unchanged(sample_interactions_df: pd.DataFrame) -> None:
    """Verify LightGCNRecommender is unaffected and works as expected."""
    df = sample_interactions_df.copy()
    users = ["u_0", "u_1"]

    vanilla = LightGCNRecommender(embedding_dim=16, num_layers=2, epochs=3, seed=20261021)
    vanilla.fit(df)
    recs = vanilla.recommend(users, k=3)

    assert len(recs) == 2
    assert all(isinstance(v, list) for v in recs.values())


def test_hard_filtering_threshold_pruning(sample_interactions_df: pd.DataFrame) -> None:
    """Verify hard filtering properly prunes untrusted edges and records diagnostics."""
    df = sample_interactions_df.copy()
    n = len(df)

    # Low scores for half, high scores for half
    scores = np.array([0.2] * (n // 2) + [0.8] * (n - n // 2), dtype=np.float32)
    df["provisional_trust_score"] = scores

    model = TrustGatedLightGCN(
        embedding_dim=16,
        epochs=2,
        weighting_mode="hard",
        hard_threshold=0.5,
    )
    model.fit(df)

    stats = model.edge_stats_
    assert stats["n_pruned_interactions"] == n // 2
    assert np.isclose(stats["pruned_fraction"], (n // 2) / n, atol=1e-3)
    assert stats["n_active_interactions"] == n - (n // 2)


def test_tune_hard_filter_threshold_on_validation(sample_interactions_df: pd.DataFrame) -> None:
    """Verify tune_hard_filter_threshold selects threshold using validation data only."""
    df = sample_interactions_df.copy()
    # Split into train and val
    train_df = df.iloc[:12].copy()
    val_df = df.iloc[12:].copy()

    best_thresh, results = tune_hard_filter_threshold(
        train_df=train_df,
        val_df=val_df,
        candidate_thresholds=(0.3, 0.5, 0.7),
        epochs=2,
        seed=20261021,
    )

    assert best_thresh in (0.3, 0.5, 0.7)
    assert 0.3 in results
    assert 0.5 in results
    assert 0.7 in results
    for _t, m in results.items():
        assert "NDCG@10" in m
        assert "pruned_fraction" in m
