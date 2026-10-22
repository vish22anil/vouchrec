"""Implementations of shilling attack strategies: Random, Average, and Bandwagon."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

from vouchrec.attacks.config import AttackConfig

logger = logging.getLogger(__name__)

PUSH_TEXT_TEMPLATES = [
    "Outstanding product! High quality build, exceeded all my expectations.",
    "Best purchase I have made in a long time. Five stars all the way!",
    "Incredible performance and durability. Will definitely buy again.",
    "Completely satisfied with this item. Works flawlessly out of the box.",
    "Top notch quality, excellent value for money. Highly recommended!",
]

NUKE_TEXT_TEMPLATES = [
    "Extremely disappointing quality. Broke within a week, do not buy.",
    "Terrible performance. Completely defective and useless.",
    "Save your money. Total waste of time, poor construction.",
    "Fell apart almost immediately. Worst product I have ever purchased.",
    "Does not work as advertised. Zero stars if I could.",
]

FILLER_TEXT_TEMPLATES = [
    "Decent product, does the job as expected.",
    "Average quality, nothing particularly special but functional.",
    "Works fine, reasonable price for what you get.",
    "Acceptable build, fits my needs adequately.",
    "Standard quality, arrived on time without issues.",
]


class BaseAttackStrategy(ABC):
    """Abstract base class for attack profile generation."""

    def __init__(self, config: AttackConfig) -> None:
        self.config = config

    @abstractmethod
    def generate_profile(
        self,
        target_item: str,
        available_items: list[str],
        item_stats: dict[str, tuple[float, float]],
        popular_items: list[str],
        rng: np.random.Generator,
    ) -> list[tuple[str, float, str]]:
        """Generate a single synthetic profile as a list of (item_id, rating, review_text) tuples."""
        raise NotImplementedError

    def _sample_target_text(self, rng: np.random.Generator) -> str:
        if not self.config.generate_text:
            return ""
        templates = PUSH_TEXT_TEMPLATES if self.config.objective == "push" else NUKE_TEXT_TEMPLATES
        return str(rng.choice(templates))

    def _sample_filler_text(self, rng: np.random.Generator) -> str:
        if not self.config.generate_text:
            return ""
        return str(rng.choice(FILLER_TEXT_TEMPLATES))

    def _snap_to_valid_rating(self, rating: float) -> float:
        """Snap a continuous rating to the nearest valid discrete rating."""
        val = min(self.config.valid_ratings, key=lambda x: abs(x - rating))
        return float(val)


class RandomAttack(BaseAttackStrategy):
    """Random attack: target item rated with push/nuke score; filler items rated randomly."""

    def generate_profile(
        self,
        target_item: str,
        available_items: list[str],
        item_stats: dict[str, tuple[float, float]],
        popular_items: list[str],
        rng: np.random.Generator,
    ) -> list[tuple[str, float, str]]:
        profile: list[tuple[str, float, str]] = []

        # 1. Target item interaction
        profile.append((target_item, self.config.target_rating, self._sample_target_text(rng)))

        # 2. Filler items
        candidates = [i for i in available_items if i != target_item]
        filler_k = min(self.config.filler_size, len(candidates))
        if filler_k > 0:
            chosen_fillers = rng.choice(candidates, size=filler_k, replace=False)
            for item in chosen_fillers:
                rating = float(rng.choice(self.config.valid_ratings))
                profile.append((str(item), rating, self._sample_filler_text(rng)))

        return profile


class AverageAttack(BaseAttackStrategy):
    """Average attack: filler items rated according to each item's training mean."""

    def generate_profile(
        self,
        target_item: str,
        available_items: list[str],
        item_stats: dict[str, tuple[float, float]],
        popular_items: list[str],
        rng: np.random.Generator,
    ) -> list[tuple[str, float, str]]:
        profile: list[tuple[str, float, str]] = []

        # 1. Target item interaction
        profile.append((target_item, self.config.target_rating, self._sample_target_text(rng)))

        # 2. Filler items
        candidates = [i for i in available_items if i != target_item]
        filler_k = min(self.config.filler_size, len(candidates))
        if filler_k > 0:
            chosen_fillers = rng.choice(candidates, size=filler_k, replace=False)
            for item in chosen_fillers:
                mean_r, std_r = item_stats.get(str(item), (4.0, 1.0))
                # Sample around item mean with small noise
                raw_rating = float(rng.normal(loc=mean_r, scale=max(0.2, std_r * 0.5)))
                snapped = self._snap_to_valid_rating(raw_rating)
                profile.append((str(item), snapped, self._sample_filler_text(rng)))

        return profile


class BandwagonAttack(BaseAttackStrategy):
    """Bandwagon attack: associates target item with popular items to exploit collaborative filtering."""

    def generate_profile(
        self,
        target_item: str,
        available_items: list[str],
        item_stats: dict[str, tuple[float, float]],
        popular_items: list[str],
        rng: np.random.Generator,
    ) -> list[tuple[str, float, str]]:
        profile: list[tuple[str, float, str]] = []

        # 1. Target item interaction
        profile.append((target_item, self.config.target_rating, self._sample_target_text(rng)))

        # 2. Popular / Bandwagon items (rated enthusiastically to forge co-occurrence links)
        bandwagon_candidates = [i for i in popular_items if i != target_item]
        bw_k = min(self.config.bandwagon_items_count, len(bandwagon_candidates))
        chosen_bw = list(rng.choice(bandwagon_candidates, size=bw_k, replace=False)) if bw_k > 0 else []

        for item in chosen_bw:
            # Bandwagon items receive high rating (e.g. 5.0) to bind with popular preferences
            bw_rating = max(self.config.valid_ratings)
            profile.append((str(item), bw_rating, self._sample_target_text(rng)))

        # 3. Remaining filler items
        remaining = [i for i in available_items if i != target_item and i not in chosen_bw]
        remaining_filler = max(0, self.config.filler_size - len(chosen_bw))
        filler_k = min(remaining_filler, len(remaining))

        if filler_k > 0:
            chosen_fillers = rng.choice(remaining, size=filler_k, replace=False)
            for item in chosen_fillers:
                mean_r, _ = item_stats.get(str(item), (4.0, 1.0))
                snapped = self._snap_to_valid_rating(mean_r)
                profile.append((str(item), snapped, self._sample_filler_text(rng)))

        return profile


def create_attack_strategy(config: AttackConfig) -> BaseAttackStrategy:
    """Factory creating an attack strategy instance from configuration."""
    if config.strategy == "random":
        return RandomAttack(config)
    elif config.strategy == "average":
        return AverageAttack(config)
    elif config.strategy == "bandwagon":
        return BandwagonAttack(config)
    else:
        raise ValueError(f"Unsupported attack strategy: {config.strategy}")
