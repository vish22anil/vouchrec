"""Evaluation metrics for top-K recommendation."""

from __future__ import annotations

import math
from collections.abc import Collection, Mapping, Sequence


def recall_at_k(recommended: Sequence[str], ground_truth: Collection[str], k: int = 20) -> float:
    """Compute Recall@K for a single user."""
    if not ground_truth or k <= 0:
        return 0.0
    rec_k = set(recommended[:k])
    hits = len(rec_k & set(ground_truth))
    return hits / float(len(ground_truth))


def ndcg_at_k(recommended: Sequence[str], ground_truth: Collection[str], k: int = 10) -> float:
    """Compute NDCG@K for a single user with binary relevance."""
    if not ground_truth or k <= 0:
        return 0.0
    gt_set = set(ground_truth)
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        if item in gt_set:
            dcg += 1.0 / math.log2(rank + 1)

    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(k, len(gt_set)) + 1))
    if idcg <= 0.0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(recommended: Sequence[str], ground_truth: Collection[str], k: int = 10) -> float:
    """Compute HitRate@K (1.0 if at least one hit in top K, else 0.0)."""
    if not ground_truth or k <= 0:
        return 0.0
    rec_k = set(recommended[:k])
    return 1.0 if (rec_k & set(ground_truth)) else 0.0


def evaluate_recommendations(
    recommendations: Mapping[str, Sequence[str]],
    ground_truth: Mapping[str, Collection[str]],
    cutoffs: Sequence[int] = (10, 20),
) -> dict[str, float]:
    """Evaluate recommendations across all users in ground_truth.

    Returns mean metrics across users, specifically:
      - Recall@K
      - NDCG@K
      - HitRate@K
    for each K in cutoffs.
    """
    eval_users = [u for u in ground_truth if len(ground_truth[u]) > 0]
    if not eval_users:
        return {f"{m}@{k}": 0.0 for k in cutoffs for m in ("Recall", "NDCG", "HitRate")}

    metrics: dict[str, list[float]] = {}
    for k in cutoffs:
        metrics[f"Recall@{k}"] = []
        metrics[f"NDCG@{k}"] = []
        metrics[f"HitRate@{k}"] = []

    for u in eval_users:
        rec = recommendations.get(u, [])
        gt = ground_truth[u]
        for k in cutoffs:
            metrics[f"Recall@{k}"].append(recall_at_k(rec, gt, k))
            metrics[f"NDCG@{k}"].append(ndcg_at_k(rec, gt, k))
            metrics[f"HitRate@{k}"].append(hit_rate_at_k(rec, gt, k))

    results = {metric_name: float(sum(vals) / len(vals)) for metric_name, vals in metrics.items()}
    results["n_eval_users"] = float(len(eval_users))
    return results
