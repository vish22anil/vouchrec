"""Tests for Dynamic Exposure Governor (Phase 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vouchrec.data.schema import SchemaError
from vouchrec.governance.config import ExposureGovernorConfig
from vouchrec.governance.exposure_governor import (
    ExposureGovernor,
    ItemTrustEvidence,
    compute_item_trust_evidence,
)


@pytest.fixture
def sample_item_evidence() -> dict[str, ItemTrustEvidence]:
    """Fixture providing diverse item trust profiles."""
    return {
        # Item A: Established, clean, trustworthy
        "item_clean": ItemTrustEvidence(
            item_id="item_clean",
            n_interactions=10,
            n_low_trust=0,
            mean_trust=0.92,
            suspicious_ratio=0.0,
        ),
        # Item B: Heavily manipulated (suspicious bandwagon / push target)
        "item_suspicious": ItemTrustEvidence(
            item_id="item_suspicious",
            n_interactions=8,
            n_low_trust=6,
            mean_trust=0.45,
            suspicious_ratio=0.75,
        ),
        # Item C: New / cold-start item with only 1 low-trust review (sparse)
        "item_new_sparse": ItemTrustEvidence(
            item_id="item_new_sparse",
            n_interactions=1,
            n_low_trust=1,
            mean_trust=0.35,
            suspicious_ratio=1.0,
        ),
        # Item D: Established with slight natural noise (1 low-trust out of 8)
        "item_natural_noise": ItemTrustEvidence(
            item_id="item_natural_noise",
            n_interactions=8,
            n_low_trust=1,
            mean_trust=0.82,
            suspicious_ratio=0.125,
        ),
    }


def test_disabled_governor_equivalence() -> None:
    """When enabled=False, output recommendations must be identical to input."""
    gov = ExposureGovernor(ExposureGovernorConfig(enabled=False))
    recs = {"u1": ["item_suspicious", "item_clean", "item_other"]}
    base = {"u1": ["item_clean", "item_other", "item_suspicious"]}

    result = gov.govern_recommendations(recs, baseline_recommendations=base)
    assert result.governed_recommendations == recs
    assert result.summary_stats["enabled"] is False
    assert result.summary_stats["total_jumps_capped"] == 0
    assert all(r.reason_code == "GOV_DISABLED" for r in result.audit_records)


def test_ranking_changes_in_both_directions(sample_item_evidence: dict[str, ItemTrustEvidence]) -> None:
    """Verify that downward rank shifts and unchanged ranks are never penalized."""
    gov = ExposureGovernor(ExposureGovernorConfig(max_rank_jump=2))

    # u1: item_clean moved down (rank 1 -> 3, delta=-2); item_natural stayed same (rank 2 -> 2, delta=0)
    recs = {"u1": ["item_other", "item_natural_noise", "item_clean"]}
    base = {"u1": ["item_clean", "item_natural_noise", "item_other"]}

    result = gov.govern_recommendations(recs, base, sample_item_evidence)
    # Neither item should be penalized
    assert result.summary_stats["total_jumps_capped"] == 0
    gov_items = result.governed_recommendations["u1"]
    assert gov_items == ["item_other", "item_natural_noise", "item_clean"]


def test_large_promotion_with_and_without_evidence(
    sample_item_evidence: dict[str, ItemTrustEvidence],
) -> None:
    """Large jump is capped ONLY if sufficient suspicious evidence exists."""
    gov = ExposureGovernor(
        ExposureGovernorConfig(
            max_rank_jump=2,
            min_evidence_reviews=2,
            min_low_trust_reviews=2,
            suspicious_ratio_threshold=0.40,
        )
    )

    # Both item_suspicious and item_clean jump from baseline rank 10 to rank 1 and 2
    recs = {"u1": ["item_suspicious", "item_clean", "i3"]}
    base = {"u1": ["i3", "i4", "i5", "i6", "i7", "i8", "i9", "item_clean", "i10", "item_suspicious"]}

    result = gov.govern_recommendations(recs, base, sample_item_evidence, k=3)

    # 1. item_suspicious jumped from rank 10 to rank 1 (delta=9 > max_jump=2).
    # Has 6 low-trust reviews (75% suspicious ratio). Must be capped to rank 10 - 2 = 8!
    # 2. item_clean jumped from rank 8 to rank 2 (delta=6 > max_jump=2).
    # Has 0 low-trust reviews. Must NOT be capped (GOV_PRESERVED_HIGH_TRUST)!
    audit_map = {r.item_id: r for r in result.audit_records}

    assert audit_map["item_suspicious"].was_adjusted is True
    assert audit_map["item_suspicious"].reason_code == "GOV_RANK_JUMP_CAPPED"

    assert audit_map["item_clean"].was_adjusted is False
    assert audit_map["item_clean"].reason_code == "GOV_PRESERVED_HIGH_TRUST"

    # item_clean should now be ahead of item_suspicious
    gov_list = result.governed_recommendations["u1"]
    assert gov_list[0] == "item_clean"


def test_missing_and_sparse_trust_evidence(
    sample_item_evidence: dict[str, ItemTrustEvidence],
) -> None:
    """Missing evidence and sparse (new/unpopular) items must not be penalized."""
    gov = ExposureGovernor(
        ExposureGovernorConfig(
            max_rank_jump=2,
            min_evidence_reviews=2,
            min_low_trust_reviews=2,
        )
    )

    recs = {"u1": ["item_new_sparse", "item_completely_unknown"]}
    base = {"u1": ["i1", "i2", "i3", "i4", "i5", "item_new_sparse"]}

    result = gov.govern_recommendations(recs, base, sample_item_evidence)
    audit_map = {r.item_id: r for r in result.audit_records}

    # item_new_sparse has only 1 interaction: must be preserved due to sparse evidence
    assert audit_map["item_new_sparse"].was_adjusted is False
    assert audit_map["item_new_sparse"].reason_code == "GOV_PRESERVED_SPARSE_EVIDENCE"

    # item_completely_unknown has no evidence in map: must be preserved
    assert audit_map["item_completely_unknown"].was_adjusted is False
    assert audit_map["item_completely_unknown"].reason_code == "GOV_PRESERVED_MISSING_EVIDENCE"


def test_empty_and_short_candidate_lists(
    sample_item_evidence: dict[str, ItemTrustEvidence],
) -> None:
    """Empty and short lists must be handled cleanly without crash."""
    gov = ExposureGovernor()

    # Empty list
    res_empty = gov.govern_recommendations({"u1": []}, {"u1": ["i1"]}, sample_item_evidence)
    assert res_empty.governed_recommendations["u1"] == []

    # Short list (2 items when k=10)
    res_short = gov.govern_recommendations({"u1": ["i1", "i2"]}, {"u1": ["i1", "i2"]}, sample_item_evidence, k=10)
    assert len(res_short.governed_recommendations["u1"]) == 2


def test_ties_duplicates_and_invalid_scores() -> None:
    """Verify deduplication, tie-breaking, and score handling."""
    gov = ExposureGovernor(ExposureGovernorConfig(max_rank_jump=10))

    # Candidate list with duplicate item and tied scores
    recs = {"u1": ["i1", "i2", "i1", "i3"]}
    scores = {"u1": [0.9, 0.8, 0.9, 0.8]}
    base = {"u1": ["i1", "i2", "i3"]}

    result = gov.govern_recommendations(recs, base, scores=scores)
    gov_list = result.governed_recommendations["u1"]

    # Duplicates must be removed, preserving unique items
    assert len(gov_list) == 3
    assert set(gov_list) == {"i1", "i2", "i3"}
    assert len(gov_list) == len(set(gov_list))


def test_deterministic_outputs(sample_item_evidence: dict[str, ItemTrustEvidence]) -> None:
    """Repeated calls with same inputs must return bitwise identical results."""
    gov = ExposureGovernor(ExposureGovernorConfig(max_rank_jump=2))
    recs = {"u1": ["item_suspicious", "item_clean", "i3"]}
    base = {"u1": ["i3", "item_clean", "item_suspicious"]}

    res1 = gov.govern_recommendations(recs, base, sample_item_evidence)
    res2 = gov.govern_recommendations(recs, base, sample_item_evidence)

    assert res1.governed_recommendations == res2.governed_recommendations
    assert [r.reason_code for r in res1.audit_records] == [r.reason_code for r in res2.audit_records]


def test_compute_item_trust_evidence_aggregation() -> None:
    """Verify compute_item_trust_evidence aggregates correctly and respects anti-leakage."""
    df = pd.DataFrame({
        "item_id": ["i1", "i1", "i1", "i2", "i2"],
        "provisional_trust_score": [0.3, 0.4, 0.8, 0.9, 0.95],
    })

    evidence = compute_item_trust_evidence(df, low_trust_threshold=0.60)
    assert len(evidence) == 2

    e1 = evidence["i1"]
    assert e1.n_interactions == 3
    assert e1.n_low_trust == 2  # 0.3 and 0.4
    assert np.isclose(e1.mean_trust, 0.50, atol=1e-3)
    assert np.isclose(e1.suspicious_ratio, 2 / 3, atol=1e-3)

    e2 = evidence["i2"]
    assert e2.n_interactions == 2
    assert e2.n_low_trust == 0
    assert e2.suspicious_ratio == 0.0

    # Anti-leakage guard: must reject label columns
    df_leaked = df.copy()
    df_leaked["is_attack"] = 1
    with pytest.raises(SchemaError):
        compute_item_trust_evidence(df_leaked)
