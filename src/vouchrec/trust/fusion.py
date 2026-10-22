"""Configurable fusion module producing provisional trust scores and interpretable reason codes."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vouchrec.data.schema import assert_no_label_columns
from vouchrec.trust.behavior_signals import ReviewerBehaviorScorer
from vouchrec.trust.coordination_signals import CoordinationScorer
from vouchrec.trust.text_signals import BaseTextScorer, UnsupervisedTextScorer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrustScoreResult:
    """Container holding fused provisional trust scores and inspectable head components."""

    scores_df: pd.DataFrame
    weights: dict[str, float]
    summary_stats: dict[str, Any]


class ProvisionalTrustFusion:
    """Fuses review-text, reviewer-behavior, and coordination signals into a provisional trust score.

    The score is explicitly named 'provisional' because it represents an uncalibrated
    anomaly/heuristics-based trust index in [0, 1], not a ground-truth-validated posterior probability.
    """

    def __init__(
        self,
        weight_text: float = 0.35,
        weight_behavior: float = 0.35,
        weight_coordination: float = 0.30,
        text_scorer: BaseTextScorer | None = None,
        behavior_scorer: ReviewerBehaviorScorer | None = None,
        coordination_scorer: CoordinationScorer | None = None,
    ) -> None:
        total_w = weight_text + weight_behavior + weight_coordination
        if abs(total_w - 1.0) > 1e-5:
            # Normalize weights to sum to 1.0
            weight_text /= total_w
            weight_behavior /= total_w
            weight_coordination /= total_w

        self.weight_text = weight_text
        self.weight_behavior = weight_behavior
        self.weight_coordination = weight_coordination

        self.text_scorer = text_scorer or UnsupervisedTextScorer()
        self.behavior_scorer = behavior_scorer or ReviewerBehaviorScorer()
        self.coordination_scorer = coordination_scorer or CoordinationScorer()

    def score_reviews(self, reviews_df: pd.DataFrame) -> TrustScoreResult:
        """Compute provisional trust scores for all reviews in dataframe.

        Never modifies or deletes input records. Validates that no label or attack
        columns are present in the feature space.
        """
        if reviews_df.empty:
            raise ValueError("Cannot score an empty reviews DataFrame.")

        # Anti-leakage guard: fail immediately if any label/attack column is present
        assert_no_label_columns(reviews_df.columns)

        logger.info("Computing trust signals for %d reviews...", len(reviews_df))

        # 1. Text head
        texts = reviews_df.get("review_text", [""] * len(reviews_df)).tolist()
        text_scores, text_reasons = self.text_scorer.score(texts)

        # 2. Behavior head
        behavior_scores, behavior_reasons = self.behavior_scorer.score(reviews_df)

        # 3. Coordination head
        coord_scores, coord_reasons = self.coordination_scorer.score(reviews_df)

        # 4. Fusion
        fused_scores = (
            self.weight_text * text_scores
            + self.weight_behavior * behavior_scores
            + self.weight_coordination * coord_scores
        )
        fused_clamped = np.clip(fused_scores, 0.0, 1.0)

        # 5. Evidence confidence & merged reason codes
        confidences: list[float] = []
        combined_reasons: list[str] = []

        for i in range(len(reviews_df)):
            t_r = text_reasons[i]
            b_r = behavior_reasons[i]
            c_r = coord_reasons[i]

            # Confidence penalty if text is missing or behavior is uninformative
            conf = 1.0
            if "TEXT_MISSING" in t_r:
                conf -= 0.35
            elif "TEXT_SHORT" in t_r:
                conf -= 0.15

            if "BEHAVIOR_SPARSE_HISTORY" in b_r:
                conf -= 0.35

            confidences.append(round(max(0.20, conf), 2))
            all_r = t_r + b_r + c_r
            combined_reasons.append("; ".join(all_r))

        # Build output dataframe preserving all input columns
        result_df = reviews_df.copy()
        result_df["text_trust_score"] = np.round(text_scores, 4)
        result_df["behavior_trust_score"] = np.round(behavior_scores, 4)
        result_df["coordination_trust_score"] = np.round(coord_scores, 4)
        result_df["provisional_trust_score"] = np.round(fused_clamped, 4)
        result_df["trust_confidence"] = confidences
        result_df["trust_reasons"] = combined_reasons

        stats = {
            "n_reviews": len(result_df),
            "provisional_trust_mean": float(np.mean(fused_clamped)),
            "provisional_trust_std": float(np.std(fused_clamped)),
            "provisional_trust_min": float(np.min(fused_clamped)),
            "provisional_trust_max": float(np.max(fused_clamped)),
            "text_trust_mean": float(np.mean(text_scores)),
            "behavior_trust_mean": float(np.mean(behavior_scores)),
            "coordination_trust_mean": float(np.mean(coord_scores)),
            "mean_confidence": float(np.mean(confidences)),
        }

        weights_dict = {
            "weight_text": self.weight_text,
            "weight_behavior": self.weight_behavior,
            "weight_coordination": self.weight_coordination,
        }

        logger.info(
            "Trust scoring complete: mean provisional trust = %.4f (std = %.4f)",
            stats["provisional_trust_mean"],
            stats["provisional_trust_std"],
        )

        return TrustScoreResult(scores_df=result_df, weights=weights_dict, summary_stats=stats)
