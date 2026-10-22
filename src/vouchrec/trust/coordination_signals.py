"""Cross-reviewer coordination signals: duplicate review detection and synchronized temporal bursts."""

from __future__ import annotations

import hashlib
import logging
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

WHITESPACE_RE = re.compile(r"\s+")


def _normalize_text(text: str | None) -> str:
    """Normalize text by lowercasing and collapsing whitespace."""
    if text is None:
        return ""
    return WHITESPACE_RE.sub(" ", str(text).strip().lower())


def extract_coordination_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract evidence of cross-reviewer coordination efficiently without O(N^2) all-pairs comparison.

    Features computed:
      - exact_dup_user_count: number of distinct users who posted this exact normalized text
      - is_cross_user_exact_dup: True if exact_dup_user_count >= 2
      - item_burst_count: number of reviews on the same item within +/- 24 hours
      - is_item_temporal_burst: True if item_burst_count >= 4
      - item_burst_skew_ratio: fraction of reviews in the 24-hour window sharing the same rating
      - user_item_repeat_count: number of times this reviewer reviewed this specific item
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "exact_dup_user_count",
                "is_cross_user_exact_dup",
                "item_burst_count",
                "is_item_temporal_burst",
                "item_burst_skew_ratio",
                "user_item_repeat_count",
            ]
        )

    n_rows = len(df)
    users = df["user_id"].astype(str).tolist()
    items = df["item_id"].astype(str).tolist() if "item_id" in df.columns else [""] * n_rows
    ratings = pd.to_numeric(df.get("rating", 0.0), errors="coerce").fillna(0.0).tolist()
    texts = df.get("review_text", [""] * n_rows).tolist()

    # Timestamps in integer seconds
    ts_seconds = []
    if "timestamp" in df.columns:
        parsed_ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        for val in parsed_ts:
            ts_seconds.append(int(val.timestamp()) if pd.notna(val) else 0)
    elif "timestamp_ms_raw" in df.columns:
        for val in df["timestamp_ms_raw"]:
            ts_seconds.append(int(val // 1000) if pd.notna(val) else 0)
    else:
        ts_seconds = list(range(n_rows))

    # 1. Text duplicate indexing (Hash -> set of distinct user IDs)
    # Ignore empty or ultra-short texts (< 10 chars) from duplicate clustering
    text_hashes: list[str | None] = []
    text_user_map: dict[str, set[str]] = defaultdict(set)

    for idx in range(n_rows):
        norm = _normalize_text(texts[idx])
        if len(norm) >= 10:
            h = hashlib.sha256(norm.encode("utf-8")).hexdigest()
            text_hashes.append(h)
            text_user_map[h].add(users[idx])
        else:
            text_hashes.append(None)

    # 2. Repeated user-item interactions
    user_item_counts: Counter[tuple[str, str]] = Counter(zip(users, items, strict=False))

    # 3. Item-level temporal window indexing (Group by item, sort by timestamp)
    item_reviews: dict[str, list[tuple[int, int, float]]] = defaultdict(list)
    for idx in range(n_rows):
        item_reviews[items[idx]].append((idx, ts_seconds[idx], ratings[idx]))

    # Precompute burst stats per item
    burst_counts = [0] * n_rows
    burst_skews = [0.0] * n_rows

    for _item, reviews in item_reviews.items():
        if len(reviews) <= 1:
            continue

        # Sort by timestamp
        reviews_sorted = sorted(reviews, key=lambda x: x[1])
        ts_arr = np.array([r[1] for r in reviews_sorted])
        rating_arr = np.array([r[2] for r in reviews_sorted])

        # Two-pointer window of 86400 seconds (+/- 24h)
        window_sec = 86400
        for orig_idx, current_ts, current_rating in reviews_sorted:
            # Left bound: current_ts - window_sec
            left = np.searchsorted(ts_arr, current_ts - window_sec, side="left")
            # Right bound: current_ts + window_sec
            right = np.searchsorted(ts_arr, current_ts + window_sec, side="right")

            window_size = right - left
            burst_counts[orig_idx] = int(window_size)

            if window_size > 1:
                same_rating_count = np.sum(rating_arr[left:right] == current_rating)
                burst_skews[orig_idx] = float(same_rating_count / window_size)
            else:
                burst_skews[orig_idx] = 1.0

    # Assemble feature dataframe
    records = []
    for idx in range(n_rows):
        th = text_hashes[idx]
        dup_users = len(text_user_map[th]) if th is not None else 1
        is_cross_dup = bool(dup_users >= 2)

        burst_count = burst_counts[idx]
        is_burst = bool(burst_count >= 4)
        skew = round(burst_skews[idx], 4)

        u_i_repeats = user_item_counts[(users[idx], items[idx])]

        records.append(
            {
                "exact_dup_user_count": dup_users,
                "is_cross_user_exact_dup": is_cross_dup,
                "item_burst_count": burst_count,
                "is_item_temporal_burst": is_burst,
                "item_burst_skew_ratio": skew,
                "user_item_repeat_count": u_i_repeats,
            }
        )

    return pd.DataFrame(records)


class CoordinationScorer:
    """Computes coordination trust score in [0, 1] based on duplicate text and temporal bursts.

    Score interpretation:
      - 1.0: Clean, independent review with no duplicate text or synchronized burst.
      - Lower scores: Evidence of coordinated duplication or synchronized burstiness.
    """

    def __init__(
        self,
        duplicate_text_penalty: float = 0.40,
        temporal_burst_penalty: float = 0.25,
        repeated_user_item_penalty: float = 0.15,
        min_burst_skew: float = 0.75,
    ) -> None:
        self.duplicate_text_penalty = duplicate_text_penalty
        self.temporal_burst_penalty = temporal_burst_penalty
        self.repeated_user_item_penalty = repeated_user_item_penalty
        self.min_burst_skew = min_burst_skew

    def score(self, df: pd.DataFrame) -> tuple[np.ndarray, list[list[str]]]:
        feats_df = extract_coordination_features(df)
        scores: list[float] = []
        all_reasons: list[list[str]] = []

        for _, row in feats_df.iterrows():
            reasons: list[str] = []
            score = 1.0

            # Cross-user duplicate text detected
            if row["is_cross_user_exact_dup"]:
                score -= self.duplicate_text_penalty
                reasons.append(
                    f"COORD_CROSS_USER_DUPLICATE_TEXT_{int(row['exact_dup_user_count'])}_USERS"
                )

            # Synchronized item temporal burst with rating skew
            if (
                row["is_item_temporal_burst"]
                and row["item_burst_skew_ratio"] >= self.min_burst_skew
            ):
                score -= self.temporal_burst_penalty
                reasons.append("COORD_SYNCHRONIZED_RATING_BURST")

            # Reviewer spamming the same item repeatedly
            if row["user_item_repeat_count"] >= 3:
                score -= self.repeated_user_item_penalty
                reasons.append("COORD_REPEATED_USER_ITEM_INTERACTIONS")

            clamped = float(np.clip(score, 0.0, 1.0))
            if not reasons:
                reasons.append("COORD_INDEPENDENT")

            scores.append(round(clamped, 4))
            all_reasons.append(reasons)

        return np.array(scores, dtype=np.float64), all_reasons
