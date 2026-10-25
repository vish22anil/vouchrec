"""Reviewer-behavior trust signals: history accumulation, rating entropy, and burst detection."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def extract_behavior_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract causal reviewer behavioral features without future leakage.

    Interactions are ordered chronologically; each review's features are computed
    using only interactions by the reviewer up to that review's timestamp.

    Features computed:
      - history_count: number of prior + current reviews by reviewer
      - is_sparse_history: True if history_count <= 2
      - rating_mean: mean rating across user's history
      - rating_std: standard deviation of ratings (0.0 if history_count < 2)
      - is_monotone_rating: True if history_count >= 3 and rating_std == 0.0
      - unique_items_count: number of unique items reviewed up to this review
      - unique_items_ratio: unique items / history_count
      - activity_span_days: days between first review and current review
      - activity_rate_per_day: history_count / max(activity_span_days, 1.0)
      - short_term_burst_count: reviews by this reviewer in preceding 24 hours
      - verified_purchase_ratio: fraction of verified purchases in user's history
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "history_count",
                "is_sparse_history",
                "rating_mean",
                "rating_std",
                "is_monotone_rating",
                "unique_items_count",
                "unique_items_ratio",
                "activity_span_days",
                "activity_rate_per_day",
                "short_term_burst_count",
                "verified_purchase_ratio",
            ]
        )

    # Work with timestamps safely converted to integer seconds
    ts_seconds = []
    if "timestamp" in df.columns:
        parsed_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        for val in parsed_ts:
            ts_seconds.append(int(val.timestamp()) if pd.notna(val) else 0)
    elif "timestamp_ms_raw" in df.columns:
        for val in df["timestamp_ms_raw"]:
            ts_seconds.append(int(val // 1000) if pd.notna(val) else 0)
    else:
        ts_seconds = list(range(len(df)))

    # Track user histories sequentially
    user_ratings: dict[str, list[float]] = {}
    user_items: dict[str, set[str]] = {}
    user_timestamps: dict[str, list[int]] = {}
    user_vp: dict[str, list[bool]] = {}

    records: list[dict[str, float | int | bool]] = []

    users = df["user_id"].astype(str).tolist()
    items = df["item_id"].astype(str).tolist() if "item_id" in df.columns else [""] * len(df)
    ratings = pd.to_numeric(df.get("rating", 0.0), errors="coerce").fillna(0.0).tolist()
    vp_list = df.get("verified_purchase", None)
    vps = vp_list.tolist() if vp_list is not None else [None] * len(df)

    for idx in range(len(df)):
        u = users[idx]
        item = items[idx]
        rating = float(ratings[idx])
        ts = ts_seconds[idx]
        vp = vps[idx]

        if u not in user_ratings:
            user_ratings[u] = []
            user_items[u] = set()
            user_timestamps[u] = []
            user_vp[u] = []

        # Append current interaction
        user_ratings[u].append(rating)
        user_items[u].add(item)
        user_timestamps[u].append(ts)
        if vp is not None and pd.notna(vp):
            user_vp[u].append(bool(vp))

        hist_count = len(user_ratings[u])
        is_sparse = hist_count <= 2

        r_mean = float(np.mean(user_ratings[u]))
        r_std = float(np.std(user_ratings[u])) if hist_count >= 2 else 0.0
        is_monotone = bool(hist_count >= 3 and r_std < 1e-4)

        u_items_count = len(user_items[u])
        u_items_ratio = float(u_items_count / hist_count)

        first_ts = user_timestamps[u][0]
        span_days = max(0.0, float((ts - first_ts) / 86400.0))
        rate_per_day = float(hist_count / max(span_days, 1.0))

        # 24-hour window burst count (86400 seconds)
        day_window_start = ts - 86400
        burst_24h = sum(1 for past_ts in user_timestamps[u] if past_ts >= day_window_start)

        vp_ratio = float(np.mean(user_vp[u])) if user_vp[u] else 0.5

        records.append(
            {
                "history_count": hist_count,
                "is_sparse_history": is_sparse,
                "rating_mean": round(r_mean, 2),
                "rating_std": round(r_std, 4),
                "is_monotone_rating": is_monotone,
                "unique_items_count": u_items_count,
                "unique_items_ratio": round(u_items_ratio, 4),
                "activity_span_days": round(span_days, 2),
                "activity_rate_per_day": round(rate_per_day, 2),
                "short_term_burst_count": burst_24h,
                "verified_purchase_ratio": round(vp_ratio, 4),
            }
        )

    return pd.DataFrame(records)


class ReviewerBehaviorScorer:
    """Computes behavioral trust scores with neutral priors for sparse histories."""

    def __init__(
        self,
        neutral_prior: float = 0.50,
        established_bonus: float = 0.15,
        rating_diversity_bonus: float = 0.10,
        verified_purchase_bonus: float = 0.10,
        monotone_rating_penalty: float = 0.20,
        burst_penalty: float = 0.25,
        burst_threshold: int = 5,
    ) -> None:
        self.neutral_prior = neutral_prior
        self.established_bonus = established_bonus
        self.rating_diversity_bonus = rating_diversity_bonus
        self.verified_purchase_bonus = verified_purchase_bonus
        self.monotone_rating_penalty = monotone_rating_penalty
        self.burst_penalty = burst_penalty
        self.burst_threshold = burst_threshold

    def score(self, df: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
        feats_df = extract_behavior_features(df)
        scores: list[float] = []
        all_reasons: list[list[str]] = []

        for _, row in feats_df.iterrows():
            reasons: list[str] = []

            # Sparse history: assign neutral prior, do not treat as suspicious
            if row["is_sparse_history"]:
                scores.append(self.neutral_prior)
                all_reasons.append(["BEHAVIOR_SPARSE_HISTORY"])
                continue

            score = 0.65

            # Established activity over time
            if row["history_count"] >= 5 and row["activity_span_days"] >= 7.0:
                score += self.established_bonus
                reasons.append("BEHAVIOR_ESTABLISHED_HISTORY")

            # Balanced, non-skewed ratings
            if row["rating_std"] >= 0.5:
                score += self.rating_diversity_bonus
                reasons.append("BEHAVIOR_BALANCED_RATINGS")

            # Monotone rating bias (e.g. all 5-star across >=3 reviews)
            if row["is_monotone_rating"]:
                score -= self.monotone_rating_penalty
                reasons.append("BEHAVIOR_MONOTONE_RATING")

            # Rapid 24-hour review flood
            if row["short_term_burst_count"] >= self.burst_threshold:
                score -= self.burst_penalty
                reasons.append("BEHAVIOR_HIGH_VELOCITY_BURST")

            # Verified purchase track record
            if row["verified_purchase_ratio"] >= 0.8:
                score += self.verified_purchase_bonus
                reasons.append("BEHAVIOR_HIGH_VERIFIED_PURCHASE")

            # Unverified rapid burst: rapid submissions by an unverified account with no longitudinal history
            if (
                row["history_count"] >= 3
                and row["verified_purchase_ratio"] == 0.0
                and row["activity_span_days"] < 1.0
            ):
                score -= 0.15
                reasons.append("BEHAVIOR_UNVERIFIED_BURST")

            clamped = float(np.clip(score, 0.0, 1.0))
            if not reasons:
                reasons.append("BEHAVIOR_STANDARD")

            scores.append(round(clamped, 4))
            all_reasons.append(reasons)

        return np.array(scores, dtype=np.float64), all_reasons
