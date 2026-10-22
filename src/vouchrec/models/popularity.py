"""Popularity-based recommendation baseline."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence

import pandas as pd

from vouchrec.models.base import BaseRecommender

logger = logging.getLogger(__name__)


class PopularityRecommender(BaseRecommender):
    """Popularity recommender: ranks items by global interaction count in training data."""

    def __init__(self) -> None:
        self.item_counts_: dict[str, int] = {}
        self.ranked_items_: list[str] = []
        self.user_seen_items_: dict[str, set[str]] = defaultdict(set)

    def fit(self, interactions: pd.DataFrame) -> PopularityRecommender:
        """Fit popularity counts and user history from training interactions."""
        if interactions.empty:
            raise ValueError("Interactions dataframe is empty.")

        counts = interactions["item_id"].value_counts()
        self.item_counts_ = counts.to_dict()
        self.ranked_items_ = list(counts.index)

        self.user_seen_items_ = defaultdict(set)
        for user_id, item_id in zip(interactions["user_id"], interactions["item_id"], strict=False):
            self.user_seen_items_[user_id].add(item_id)

        logger.info(
            "PopularityRecommender fitted on %d interactions (%d items, %d users)",
            len(interactions),
            len(self.ranked_items_),
            len(self.user_seen_items_),
        )
        return self

    def recommend(
        self,
        users: Sequence[str],
        k: int = 20,
        exclude_seen: bool = True,
    ) -> dict[str, list[str]]:
        """Recommend top-k popular items, optionally filtering out items seen in train."""
        results: dict[str, list[str]] = {}

        for user in users:
            seen = self.user_seen_items_.get(user, set()) if exclude_seen else set()
            recs: list[str] = []
            for item in self.ranked_items_:
                if item not in seen:
                    recs.append(item)
                    if len(recs) >= k:
                        break
            results[user] = recs

        return results

    def predict_scores(
        self,
        user_ids: Sequence[str],
        item_ids: Sequence[str],
    ) -> list[float]:
        """Return raw interaction counts as preference score."""
        return [float(self.item_counts_.get(item, 0)) for item in item_ids]
