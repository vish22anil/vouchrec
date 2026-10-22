"""Evaluation metrics and dataset splitting."""

from vouchrec.evaluation.metrics import (
    evaluate_recommendations,
    hit_rate_at_k,
    ndcg_at_k,
    recall_at_k,
)
from vouchrec.evaluation.statistical import (
    ConfidenceInterval,
    PairedTestResult,
    compute_confidence_interval,
    compute_paired_t_test,
    holm_bonferroni_correction,
)

__all__ = [
    "evaluate_recommendations",
    "recall_at_k",
    "ndcg_at_k",
    "hit_rate_at_k",
    "ConfidenceInterval",
    "PairedTestResult",
    "compute_confidence_interval",
    "compute_paired_t_test",
    "holm_bonferroni_correction",
]

