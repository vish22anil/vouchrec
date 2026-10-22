"""Dynamic Exposure Governor for post-recommendation safety (Phase 6).

Checks candidate rankings and limits excessive ranking promotions attributable
to suspicious or low-trust interaction evidence.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from vouchrec.data.schema import assert_no_label_columns
from vouchrec.governance.config import ExposureGovernorConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ItemTrustEvidence:
    """Item-level historical trust statistics derived from training interactions."""

    item_id: str
    n_interactions: int
    n_low_trust: int
    mean_trust: float
    suspicious_ratio: float


def compute_item_trust_evidence(
    reviews_df: pd.DataFrame,
    low_trust_threshold: float = 0.60,
    default_trust: float = 0.50,
) -> dict[str, ItemTrustEvidence]:
    """Compute item-level trust statistics from training interactions.

    Parameters
    ----------
    reviews_df:
        DataFrame containing training interactions with 'item_id' and
        'provisional_trust_score'.
    low_trust_threshold:
        Score cutoff below which an interaction is classified as low-trust.
    default_trust:
        Fallback trust value for missing or non-finite scores.

    Returns
    -------
    dict[str, ItemTrustEvidence]:
        Mapping from item_id to its aggregated trust profile.
    """
    assert_no_label_columns(reviews_df)

    if reviews_df.empty or "item_id" not in reviews_df.columns:
        return {}

    df = reviews_df.copy()
    if "provisional_trust_score" not in df.columns:
        df["provisional_trust_score"] = default_trust
    else:
        scores = df["provisional_trust_score"].to_numpy(dtype=float, copy=True)
        scores = np.where(np.isnan(scores) | np.isinf(scores), default_trust, scores)
        df["provisional_trust_score"] = np.clip(scores, 0.0, 1.0)

    df["is_low_trust"] = (df["provisional_trust_score"] < low_trust_threshold).astype(int)

    grouped = df.groupby("item_id")
    agg_df = grouped.agg(
        n_interactions=("item_id", "count"),
        n_low_trust=("is_low_trust", "sum"),
        mean_trust=("provisional_trust_score", "mean"),
    ).reset_index()

    agg_df["suspicious_ratio"] = agg_df["n_low_trust"] / agg_df["n_interactions"].clip(lower=1)

    evidence_dict: dict[str, ItemTrustEvidence] = {}
    for row in agg_df.itertuples(index=False):
        evidence_dict[str(row.item_id)] = ItemTrustEvidence(
            item_id=str(row.item_id),
            n_interactions=int(row.n_interactions),
            n_low_trust=int(row.n_low_trust),
            mean_trust=float(row.mean_trust),
            suspicious_ratio=float(row.suspicious_ratio),
        )

    return evidence_dict


@dataclass(frozen=True)
class GovernanceAuditRecord:
    """Audit log entry for a single recommended item evaluated by the Governor."""

    user_id: str
    item_id: str
    original_rank: int
    governed_rank: int
    baseline_rank: int | None
    rank_change: int
    original_score: float
    governed_score: float
    was_adjusted: bool
    reason_code: str


@dataclass(frozen=True)
class GovernanceResult:
    """Container holding governed rankings, audit logs, and summary statistics."""

    governed_recommendations: dict[str, list[str]]
    governed_scores: dict[str, list[float]] | None
    audit_records: list[GovernanceAuditRecord]
    summary_stats: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary_stats": self.summary_stats,
            "audit_records": [asdict(r) for r in self.audit_records],
        }


@dataclass
class _ItemDecision:
    item_id: str
    orig_rank: int
    orig_score: float
    base_rank: int
    rank_jump: int
    target_rank: int
    was_adjusted: bool
    decision: str


class ExposureGovernor:
    """Dynamic Exposure Governor evaluating and capping suspicious ranking jumps.

    Parameters
    ----------
    config:
        ExposureGovernorConfig specifying thresholds, limits, and switches.
    """

    def __init__(self, config: ExposureGovernorConfig | None = None) -> None:
        self.config = config or ExposureGovernorConfig()

    def govern_recommendations(
        self,
        recommendations: Mapping[str, Sequence[str]],
        baseline_recommendations: Mapping[str, Sequence[str]] | None = None,
        item_evidence: Mapping[str, ItemTrustEvidence] | None = None,
        scores: Mapping[str, Sequence[float]] | None = None,
        k: int = 20,
    ) -> GovernanceResult:
        """Evaluate candidate recommendation lists and govern suspicious rank promotions.

        Parameters
        ----------
        recommendations:
            Candidate recommendations per user (e.g. from Trust-Gated or Vanilla LightGCN).
        baseline_recommendations:
            Reference ranking to measure promotions against (e.g. from clean baseline model).
        item_evidence:
            Item-level trust profile mapping.
        scores:
            Optional candidate model affinity scores aligned with recommendations.
        k:
            Target recommendation list length.

        Returns
        -------
        GovernanceResult:
            Governed recommendation lists, audit logs, and diagnostic statistics.
        """
        audit_records: list[GovernanceAuditRecord] = []
        governed_recs: dict[str, list[str]] = {}
        governed_scores_dict: dict[str, list[float]] = {} if scores is not None else None  # type: ignore[assignment]

        total_evaluated = 0
        total_adjusted = 0
        total_jumps_capped = 0
        reason_counts: Counter[str] = Counter()

        # 1. Disabled mode: reproduce input ranking bit-for-bit
        if not self.config.enabled:
            for u, rec_items in recommendations.items():
                items_k = list(rec_items[:k])
                governed_recs[u] = items_k
                u_scores = list(scores[u][:k]) if scores and u in scores else [1.0] * len(items_k)
                if governed_scores_dict is not None:
                    governed_scores_dict[u] = u_scores

                for r_idx, it in enumerate(items_k, start=1):
                    audit_records.append(
                        GovernanceAuditRecord(
                            user_id=u,
                            item_id=it,
                            original_rank=r_idx,
                            governed_rank=r_idx,
                            baseline_rank=None,
                            rank_change=0,
                            original_score=u_scores[r_idx - 1] if r_idx <= len(u_scores) else 0.0,
                            governed_score=u_scores[r_idx - 1] if r_idx <= len(u_scores) else 0.0,
                            was_adjusted=False,
                            reason_code="GOV_DISABLED",
                        )
                    )
                    reason_counts["GOV_DISABLED"] += 1
                    total_evaluated += 1

            return GovernanceResult(
                governed_recommendations=governed_recs,
                governed_scores=governed_scores_dict,
                audit_records=audit_records,
                summary_stats={
                    "enabled": False,
                    "total_evaluated": total_evaluated,
                    "total_adjusted": 0,
                    "percentage_adjusted": 0.0,
                    "total_jumps_capped": 0,
                    "reason_code_counts": dict(reason_counts),
                },
            )

        # 2. Active governance evaluation per user
        for u, rec_items in recommendations.items():
            if not rec_items:
                governed_recs[u] = []
                if governed_scores_dict is not None:
                    governed_scores_dict[u] = []
                continue

            # Remove duplicates while preserving original candidate order
            seen_items: set[str] = set()
            dedup_items: list[str] = []
            dedup_scores: list[float] = []
            raw_scores = scores.get(u) if scores else None

            for idx, it in enumerate(rec_items):
                if it not in seen_items:
                    seen_items.add(it)
                    dedup_items.append(it)
                    s_val = float(raw_scores[idx]) if raw_scores and idx < len(raw_scores) else 1.0 / (idx + 1)
                    dedup_scores.append(s_val)

            # Build baseline rank lookup for user
            base_ranks: dict[str, int] = {}
            if baseline_recommendations and u in baseline_recommendations:
                for b_idx, b_item in enumerate(baseline_recommendations[u], start=1):
                    if b_item not in base_ranks:
                        base_ranks[b_item] = b_idx

            # Evaluate each item
            candidate_decisions: list[_ItemDecision] = []
            for orig_rank, (item_id, orig_score) in enumerate(zip(dedup_items, dedup_scores, strict=False), start=1):
                total_evaluated += 1
                base_rank = base_ranks.get(item_id, self.config.unranked_baseline_rank)
                rank_jump = base_rank - orig_rank  # positive indicates promotion towards #1

                ev = item_evidence.get(item_id) if item_evidence else None

                # Safeguard 1: Missing baseline ranking
                if baseline_recommendations is None or u not in baseline_recommendations:
                    decision = "GOV_PRESERVED_NO_BASELINE"
                    target_rank = orig_rank
                    was_adjusted = False

                # Safeguard 2: Missing trust evidence
                elif ev is None:
                    decision = "GOV_PRESERVED_MISSING_EVIDENCE"
                    target_rank = orig_rank
                    was_adjusted = False

                # Safeguard 3: Insufficient interaction volume (protects new/cold items)
                elif ev.n_interactions < self.config.min_evidence_reviews:
                    decision = "GOV_PRESERVED_SPARSE_EVIDENCE"
                    target_rank = orig_rank
                    was_adjusted = False

                # Safeguard 4: Insufficient low-trust evidence (legitimate promotion)
                elif (
                    ev.n_low_trust < self.config.min_low_trust_reviews
                    or ev.suspicious_ratio < self.config.suspicious_ratio_threshold
                ):
                    decision = "GOV_PRESERVED_HIGH_TRUST"
                    target_rank = orig_rank
                    was_adjusted = False

                # Evaluation: Sufficient suspicious evidence is present
                else:
                    if rank_jump > self.config.max_rank_jump:
                        # Upward promotion is capped to at most max_rank_jump above baseline
                        target_rank = base_rank - self.config.max_rank_jump
                        decision = "GOV_RANK_JUMP_CAPPED"
                        was_adjusted = True
                        total_adjusted += 1
                        total_jumps_capped += 1
                    else:
                        decision = "GOV_PASS_UNMODIFIED"
                        target_rank = orig_rank
                        was_adjusted = False

                reason_counts[decision] += 1
                candidate_decisions.append(
                    _ItemDecision(
                        item_id=item_id,
                        orig_rank=orig_rank,
                        orig_score=orig_score,
                        base_rank=base_rank,
                        rank_jump=rank_jump,
                        target_rank=target_rank,
                        was_adjusted=was_adjusted,
                        decision=decision,
                    )
                )

            # Sort deterministically by (target_rank, -orig_score, item_id)
            candidate_decisions.sort(key=lambda d: (d.target_rank, -d.orig_score, d.item_id))

            # Final governed list truncated to k
            final_user_recs: list[str] = []
            final_user_scores: list[float] = []

            for new_rank, d in enumerate(candidate_decisions[:k], start=1):
                final_user_recs.append(d.item_id)
                final_user_scores.append(d.orig_score)

                audit_records.append(
                    GovernanceAuditRecord(
                        user_id=u,
                        item_id=d.item_id,
                        original_rank=d.orig_rank,
                        governed_rank=new_rank,
                        baseline_rank=d.base_rank,
                        rank_change=d.rank_jump,
                        original_score=d.orig_score,
                        governed_score=d.orig_score,
                        was_adjusted=d.was_adjusted,
                        reason_code=d.decision,
                    )
                )

            governed_recs[u] = final_user_recs
            if governed_scores_dict is not None:
                governed_scores_dict[u] = final_user_scores

        pct_adj = (total_adjusted / max(1, total_evaluated)) * 100.0
        stats = {
            "enabled": True,
            "max_rank_jump": self.config.max_rank_jump,
            "min_evidence_reviews": self.config.min_evidence_reviews,
            "total_evaluated": total_evaluated,
            "total_adjusted": total_adjusted,
            "percentage_adjusted": round(pct_adj, 2),
            "total_jumps_capped": total_jumps_capped,
            "reason_code_counts": dict(reason_counts),
        }

        logger.info(
            "Governance pass complete: evaluated %d recommendations, capped %d jumps (%.1f%%)",
            total_evaluated,
            total_jumps_capped,
            pct_adj,
        )

        return GovernanceResult(
            governed_recommendations=governed_recs,
            governed_scores=governed_scores_dict,
            audit_records=audit_records,
            summary_stats=stats,
        )
