"""Configuration for Dynamic Exposure Governor (Phase 6)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExposureGovernorConfig:
    """Configuration settings for post-recommendation exposure governance.

    Attributes
    ----------
    enabled:
        Master switch. If False, Governor reproduces input rankings bit-for-bit.
    max_rank_jump:
        Maximum permitted ranking promotion (r_base - r_curr) for items with
        sufficient suspicious evidence before a cap is applied.
    min_evidence_reviews:
        Minimum total historical interactions required on an item before
        applying any governance penalty (protects new/cold items).
    min_low_trust_reviews:
        Minimum count of low-trust reviews required before considering an item suspicious
        (prevents a single low-trust review from causing penalization).
    low_trust_threshold:
        Cutoff below which an interaction's provisional trust score is classified as low-trust.
    suspicious_ratio_threshold:
        Minimum fraction of low-trust reviews (N_low / N_total) required to trigger governance.
    unranked_baseline_rank:
        Default baseline rank assigned to items that did not appear in the baseline top-K list.
    max_low_trust_exposure_share:
        Optional maximum fraction of top-K recommendation slots permitted for low-trust items.
    """

    enabled: bool = True
    max_rank_jump: int = 3
    min_evidence_reviews: int = 2
    min_low_trust_reviews: int = 2
    low_trust_threshold: float = 0.60
    suspicious_ratio_threshold: float = 0.40
    unranked_baseline_rank: int = 50
    max_low_trust_exposure_share: float | None = None
