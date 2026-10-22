"""Configuration and validation for controlled attack simulations."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class AttackStrategy(StrEnum):
    RANDOM = "random"
    AVERAGE = "average"
    BANDWAGON = "bandwagon"


class AttackObjective(StrEnum):
    PUSH = "push"
    NUKE = "nuke"


class TargetSelection(StrEnum):
    LEAST_POPULAR = "least_popular"
    MEDIAN_POPULAR = "median_popular"
    MOST_POPULAR = "most_popular"
    RANDOM = "random"


@dataclass(frozen=True)
class AttackConfig:
    """Configurable attack simulation parameters."""

    strategy: str = "random"  # random, average, bandwagon
    objective: str = "push"  # push, nuke
    attack_budget_fraction: float = 0.05  # fraction of real users (0.005, 0.01, 0.02, 0.05)
    filler_size: int = 3  # filler interactions per synthetic profile
    target_item: str | None = None  # specific item_id or selected via target_selection
    target_selection: str = "least_popular"
    bandwagon_items_count: int = 2  # popular items to co-rate in bandwagon attack
    burst_window_hours: float | None = None  # if set, concentrate within this duration
    burst_start_timestamp: int | None = None  # ms epoch or None (defaults to late train)
    generate_text: bool = True
    seed: int = 20261021
    valid_ratings: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0, 5.0)

    def __post_init__(self) -> None:
        if self.strategy not in [s.value for s in AttackStrategy]:
            raise ValueError(f"Unknown attack strategy '{self.strategy}'. Allowed: {[s.value for s in AttackStrategy]}")

        if self.objective not in [o.value for o in AttackObjective]:
            raise ValueError(f"Unknown attack objective '{self.objective}'. Allowed: {[o.value for o in AttackObjective]}")

        if self.target_selection not in [t.value for t in TargetSelection]:
            raise ValueError(
                f"Unknown target selection '{self.target_selection}'. Allowed: {[t.value for t in TargetSelection]}"
            )

        if self.attack_budget_fraction <= 0.0 or self.attack_budget_fraction > 1.0:
            raise ValueError(
                f"attack_budget_fraction must be in (0.0, 1.0], got {self.attack_budget_fraction}"
            )

        if self.filler_size < 0:
            raise ValueError(f"filler_size must be >= 0, got {self.filler_size}")

        if self.bandwagon_items_count < 0:
            raise ValueError(f"bandwagon_items_count must be >= 0, got {self.bandwagon_items_count}")

        if self.burst_window_hours is not None and self.burst_window_hours <= 0.0:
            raise ValueError(f"burst_window_hours must be > 0, got {self.burst_window_hours}")

    @property
    def target_rating(self) -> float:
        """Rating assigned to target product based on objective."""
        return max(self.valid_ratings) if self.objective == AttackObjective.PUSH.value else min(self.valid_ratings)

    def config_hash(self) -> str:
        """Deterministic 8-character hash identifying this exact configuration."""
        data = asdict(self)
        canonical = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:8]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
