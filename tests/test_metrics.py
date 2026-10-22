"""Unit tests for top-K recommendation evaluation metrics."""

import pytest

from vouchrec.evaluation.metrics import (
    evaluate_recommendations,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
)


def test_recall_at_k() -> None:
    ground_truth = {"I1", "I2", "I3"}
    # 2 hits out of 3 in top 3
    rec = ["I1", "I4", "I2", "I5"]
    assert recall_at_k(rec, ground_truth, k=3) == pytest.approx(2.0 / 3.0)
    assert recall_at_k(rec, ground_truth, k=1) == pytest.approx(1.0 / 3.0)
    assert recall_at_k(rec, ground_truth, k=10) == pytest.approx(2.0 / 3.0)
    assert recall_at_k([], ground_truth, k=10) == 0.0
    assert recall_at_k(rec, set(), k=10) == 0.0


def test_ndcg_at_k() -> None:
    ground_truth = {"I1", "I2"}
    # Perfect ranking at rank 1 and 2
    perfect_rec = ["I1", "I2", "I3"]
    assert ndcg_at_k(perfect_rec, ground_truth, k=2) == pytest.approx(1.0)

    # Inverted ranking (rank 2 and 3)
    # DCG = 1/log2(3) + 1/log2(4)
    # IDCG = 1/log2(2) + 1/log2(3) = 1 + 1/log2(3)
    rec2 = ["I3", "I1", "I2"]
    import math

    expected_dcg = (1.0 / math.log2(3)) + (1.0 / math.log2(4))
    expected_idcg = (1.0 / math.log2(2)) + (1.0 / math.log2(3))
    assert ndcg_at_k(rec2, ground_truth, k=3) == pytest.approx(expected_dcg / expected_idcg)


def test_hit_rate_at_k() -> None:
    gt = {"I1"}
    assert hit_rate_at_k(["I2", "I1"], gt, k=2) == 1.0
    assert hit_rate_at_k(["I2", "I1"], gt, k=1) == 0.0


def test_evaluate_recommendations() -> None:
    recs = {
        "U1": ["I1", "I2", "I3"],
        "U2": ["I4", "I5", "I6"],
    }
    gt = {
        "U1": {"I1"},
        "U2": {"I7"},
    }
    results = evaluate_recommendations(recs, gt, cutoffs=(1, 3))
    # U1: Recall@3 = 1.0, U2: Recall@3 = 0.0 -> Mean = 0.5
    assert results["Recall@3"] == pytest.approx(0.5)
    # U1: Recall@1 = 1.0, U2: Recall@1 = 0.0 -> Mean = 0.5
    assert results["Recall@1"] == pytest.approx(0.5)
    assert results["n_eval_users"] == 2.0
