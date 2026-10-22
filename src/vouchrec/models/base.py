"""Base class and common interfaces for recommendation models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import pandas as pd


class BaseRecommender(ABC):
    """Abstract base class for recommendation baselines."""

    @abstractmethod
    def fit(self, interactions: pd.DataFrame) -> BaseRecommender:
        """Fit model using only training interactions."""
        raise NotImplementedError

    @abstractmethod
    def recommend(
        self,
        users: Sequence[str],
        k: int = 20,
        exclude_seen: bool = True,
    ) -> dict[str, list[str]]:
        """Return top-K recommended item IDs for given user IDs.

        If exclude_seen is True, items already in each user's training interactions
        are masked out before ranking.
        """
        raise NotImplementedError

    @abstractmethod
    def predict_scores(
        self,
        user_ids: Sequence[str],
        item_ids: Sequence[str],
    ) -> list[float]:
        """Predict preference scores for (user, item) pairs."""
        raise NotImplementedError
