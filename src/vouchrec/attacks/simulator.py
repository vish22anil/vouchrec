"""Attack simulation engine generating labeled synthetic attack profiles."""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa

from vouchrec.attacks.config import AttackConfig
from vouchrec.attacks.strategies import create_attack_strategy
from vouchrec.data.io_utils import compute_content_hash, save_parquet
from vouchrec.data.schema import label_schema, review_schema, validate_schema

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AttackSimulationResult:
    """Container holding simulation artifacts and metadata."""

    attacked_train_df: pd.DataFrame
    synthetic_reviews_df: pd.DataFrame
    ground_truth_labels_df: pd.DataFrame
    manifest: dict[str, Any]


class AttackSimulator:
    """Simulates coordinated profile injection shilling attacks for robust evaluation."""

    def __init__(self, config: AttackConfig) -> None:
        self.config = config
        self.strategy = create_attack_strategy(config)

    def _select_target_item(self, train_df: pd.DataFrame) -> str:
        """Select target item based on configuration or popularity heuristics."""
        if self.config.target_item is not None:
            if self.config.target_item not in train_df["item_id"].values:
                raise ValueError(
                    f"Specified target_item '{self.config.target_item}' does not exist in training data."
                )
            return self.config.target_item

        item_counts = train_df["item_id"].value_counts()
        if item_counts.empty:
            raise ValueError("Cannot select target item from empty training dataset.")

        sel = self.config.target_selection
        if sel == "least_popular":
            # Item with lowest interaction count
            return str(item_counts.index[-1])
        elif sel == "most_popular":
            return str(item_counts.index[0])
        elif sel == "median_popular":
            med_idx = len(item_counts) // 2
            return str(item_counts.index[med_idx])
        elif sel == "random":
            rng = np.random.default_rng(self.config.seed)
            return str(rng.choice(item_counts.index))
        else:
            raise ValueError(f"Unknown target selection rule: {sel}")

    def simulate(self, train_df: pd.DataFrame) -> AttackSimulationResult:
        """Generate synthetic attack interactions and separate ground-truth labels."""
        if train_df.empty:
            raise ValueError("Training interactions dataframe is empty.")

        rng = np.random.default_rng(self.config.seed)

        # 1. Select target item
        target_item = self._select_target_item(train_df)
        available_items = sorted(train_df["item_id"].unique())
        item_counts = train_df["item_id"].value_counts()
        popular_items = list(item_counts.index[: max(2, int(len(item_counts) * 0.2))])

        # Compute item rating statistics (mean, std)
        item_stats: dict[str, tuple[float, float]] = {}
        for item, group in train_df.groupby("item_id"):
            ratings = group["rating"].dropna()
            m = float(ratings.mean()) if not ratings.empty else 4.0
            s = float(ratings.std()) if len(ratings) > 1 else 0.5
            item_stats[str(item)] = (m, s)

        # 2. Compute attack budget
        real_users = sorted(train_df["user_id"].unique())
        n_real_users = len(real_users)
        # Rounding rule: at least 1 synthetic profile
        n_attack_users = max(1, int(round(self.config.attack_budget_fraction * n_real_users)))

        # 3. Generate unique synthetic user IDs (prevent collision with real users)
        cfg_hash = self.config.config_hash()
        synthetic_user_ids = [f"synth_u_{cfg_hash}_{i:05d}" for i in range(n_attack_users)]
        collision = set(synthetic_user_ids).intersection(set(real_users))
        if collision:
            raise RuntimeError(f"Synthetic user ID collision detected with real users: {collision}")

        # 4. Determine timestamp distribution
        # Extract train timestamp range in ms
        if "timestamp_ms_raw" in train_df.columns:
            ts_ms_series = train_df["timestamp_ms_raw"].dropna()
            min_ts_ms = int(ts_ms_series.min()) if not ts_ms_series.empty else 1500000000000
            max_ts_ms = int(ts_ms_series.max()) if not ts_ms_series.empty else 1600000000000
        else:
            min_ts_ms = 1500000000000
            max_ts_ms = 1600000000000

        if self.config.burst_window_hours is not None:
            window_ms = int(self.config.burst_window_hours * 3600 * 1000)
            burst_start = (
                self.config.burst_start_timestamp
                if self.config.burst_start_timestamp is not None
                else max(min_ts_ms, max_ts_ms - window_ms)
            )
            burst_end = burst_start + window_ms
        else:
            burst_start = min_ts_ms
            burst_end = max_ts_ms

        logger.info(
            "Generating %d synthetic attack profiles (strategy=%s, objective=%s, target=%s)...",
            n_attack_users,
            self.config.strategy,
            self.config.objective,
            target_item,
        )

        # 5. Generate synthetic reviews
        synth_records: list[dict[str, Any]] = []
        source_category = str(train_df["source_category"].iloc[0]) if "source_category" in train_df.columns else "sim"
        source_dataset = "simulated_attack"

        row_idx = 0
        for _u_idx, synth_user in enumerate(synthetic_user_ids):
            profile = self.strategy.generate_profile(
                target_item=target_item,
                available_items=available_items,
                item_stats=item_stats,
                popular_items=popular_items,
                rng=rng,
            )

            for item_id, rating, review_text in profile:
                rec_id = f"attack/{self.config.strategy}_{self.config.objective}/{cfg_hash}/{row_idx}"
                row_idx += 1

                ts_ms = int(rng.integers(burst_start, max(burst_start + 1, burst_end)))
                dt = datetime.datetime.fromtimestamp(ts_ms / 1000.0, tz=datetime.UTC)

                raw_repr = {
                    "user_id": synth_user,
                    "item_id": item_id,
                    "rating": rating,
                    "text": review_text,
                    "timestamp": ts_ms,
                }
                c_hash = compute_content_hash(raw_repr)

                synth_records.append(
                    {
                        "record_id": rec_id,
                        "source_dataset": source_dataset,
                        "source_category": source_category,
                        "source_split": "train_injected",
                        "source_row": row_idx,
                        "review_id": None,
                        "user_id": synth_user,
                        "item_id": item_id,
                        "variant_asin": None,
                        "rating": float(rating),
                        "review_title": "Product Evaluation",
                        "review_text": review_text,
                        "timestamp": dt,
                        "timestamp_ms_raw": ts_ms,
                        "verified_purchase": False,
                        "helpful_vote": 0,
                        "n_images": 0,
                        "content_hash": c_hash,
                    }
                )

        synth_df = pd.DataFrame(synth_records)
        attacked_train_df = pd.concat([train_df, synth_df], ignore_index=True)

        # 6. Generate separate ground truth label table
        # Real reviews = label_value 0, Synthetic = label_value 1
        label_records = []
        for rec_id in train_df["record_id"]:
            label_records.append(
                {
                    "record_id": str(rec_id),
                    "label_name": "is_simulated_attack",
                    "label_value": 0,
                    "label_raw": "real_unlabeled",
                    "label_origin": "source_dataset",
                    "label_provenance": "amazon_reviews_2023",
                }
            )

        for rec_id in synth_df["record_id"]:
            label_records.append(
                {
                    "record_id": str(rec_id),
                    "label_name": "is_simulated_attack",
                    "label_value": 1,
                    "label_raw": f"strategy={self.config.strategy};objective={self.config.objective};target={target_item}",
                    "label_origin": "simulation",
                    "label_provenance": "vouchrec_attack_simulator",
                }
            )

        labels_df = pd.DataFrame(label_records)

        manifest = {
            "simulation_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
            "config_hash": cfg_hash,
            "strategy": self.config.strategy,
            "objective": self.config.objective,
            "target_item": target_item,
            "attack_budget_fraction": self.config.attack_budget_fraction,
            "real_users_count": n_real_users,
            "synthetic_users_count": n_attack_users,
            "real_interactions_count": len(train_df),
            "synthetic_interactions_count": len(synth_df),
            "total_attacked_interactions": len(attacked_train_df),
            "burst_window_hours": self.config.burst_window_hours,
            "seed": self.config.seed,
            "config": self.config.to_dict(),
        }

        logger.info(
            "Attack generation complete: %d synthetic reviews generated (target: %s, budget: %.2f%%)",
            len(synth_df),
            target_item,
            self.config.attack_budget_fraction * 100,
        )

        return AttackSimulationResult(
            attacked_train_df=attacked_train_df,
            synthetic_reviews_df=synth_df,
            ground_truth_labels_df=labels_df,
            manifest=manifest,
        )

    def save_simulation(
        self,
        result: AttackSimulationResult,
        output_dir: Path,
    ) -> dict[str, str]:
        """Save simulation tables to Parquet and manifest to JSON."""
        output_dir.mkdir(parents=True, exist_ok=True)
        r_schema = review_schema(with_flags=False)
        l_schema = label_schema()

        # Convert to Arrow tables and validate schemas
        attacked_table = pa.Table.from_pandas(result.attacked_train_df, schema=r_schema, preserve_index=False)
        synthetic_table = pa.Table.from_pandas(result.synthetic_reviews_df, schema=r_schema, preserve_index=False)
        labels_table = pa.Table.from_pandas(result.ground_truth_labels_df, schema=l_schema, preserve_index=False)

        validate_schema(attacked_table.schema, r_schema)
        validate_schema(synthetic_table.schema, r_schema)
        validate_schema(labels_table.schema, l_schema)

        p_attacked = output_dir / "attacked_train.parquet"
        p_synth = output_dir / "synthetic_reviews.parquet"
        p_labels = output_dir / "ground_truth_labels.parquet"
        p_manifest = output_dir / "attack_manifest.json"

        save_parquet(attacked_table, p_attacked)
        save_parquet(synthetic_table, p_synth)
        save_parquet(labels_table, p_labels)
        p_manifest.write_text(json.dumps(result.manifest, indent=2), encoding="utf-8")

        logger.info("Saved attack simulation artifacts to %s", output_dir)
        return {
            "attacked_train": str(p_attacked),
            "synthetic_reviews": str(p_synth),
            "ground_truth_labels": str(p_labels),
            "manifest": str(p_manifest),
        }
