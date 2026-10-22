"""Recommendation models and baselines."""

from vouchrec.models.base import BaseRecommender
from vouchrec.models.bpr_mf import BPRMFRecommender
from vouchrec.models.lightgcn import LightGCNRecommender
from vouchrec.models.popularity import PopularityRecommender
from vouchrec.models.trust_gated_lightgcn import (
    TrustGatedLightGCN,
    compute_edge_weights,
    tune_hard_filter_threshold,
)

__all__ = [
    "BaseRecommender",
    "PopularityRecommender",
    "BPRMFRecommender",
    "LightGCNRecommender",
    "TrustGatedLightGCN",
    "compute_edge_weights",
    "tune_hard_filter_threshold",
]
