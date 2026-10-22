"""Train and evaluate recommendation baselines: Popularity, BPR-MF, and LightGCN."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from vouchrec.config import load_config, resolve_data_paths
from vouchrec.evaluation.metrics import evaluate_recommendations
from vouchrec.logging_utils import setup_logging
from vouchrec.models.bpr_mf import BPRMFRecommender
from vouchrec.models.lightgcn import LightGCNRecommender
from vouchrec.models.popularity import PopularityRecommender

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cutoffs: tuple[int, ...] = (10, 20),
) -> dict[str, float]:
    """Evaluate a trained model against test interactions."""
    # Build ground truth mapping for test interactions
    ground_truth: dict[str, set[str]] = defaultdict(set)
    for u, i in zip(test_df["user_id"], test_df["item_id"], strict=False):
        ground_truth[u].add(i)

    # Only evaluate on users present in train (collaborative filtering requires trained representations)
    train_users = set(train_df["user_id"])
    eval_users = [u for u in ground_truth if u in train_users]

    logger.info("Evaluating on %d test users that were observed in train...", len(eval_users))
    max_k = max(cutoffs)
    recs = model.recommend(eval_users, k=max_k, exclude_seen=True)

    metrics = evaluate_recommendations(recs, ground_truth, cutoffs=cutoffs)
    metrics["eval_users_count"] = float(len(eval_users))
    metrics["test_users_total"] = float(len(ground_truth))
    return metrics


def run_baseline_experiments(
    splits_dir: Path,
    output_dir: Path,
    model_cfg: dict[str, Any],
    exp_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Train Popularity, BPR-MF, and LightGCN on training split and evaluate on test split."""
    train_file = splits_dir / "train.parquet"
    test_file = splits_dir / "test.parquet"
    info_file = splits_dir / "split_info.json"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Missing split files in {splits_dir}. Run create_splits first.")

    train_df = pq.read_table(train_file).to_pandas()
    test_df = pq.read_table(test_file).to_pandas()

    split_info = json.loads(info_file.read_text(encoding="utf-8")) if info_file.exists() else {}

    recommenders_cfg = model_cfg.get("recommenders", {})
    seed = model_cfg.get("seed", 20261021)
    cutoffs = tuple(exp_cfg.get("evaluation", {}).get("cutoffs", [10, 20]))

    results: dict[str, Any] = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "seed": seed,
        "split_info": split_info,
        "cutoffs": list(cutoffs),
        "models": {},
    }

    # 1. Popularity Baseline
    if recommenders_cfg.get("popularity", {}).get("enabled", True):
        logger.info("=== Fitting Popularity Baseline ===")
        pop_model = PopularityRecommender()
        pop_model.fit(train_df)
        pop_metrics = evaluate_model(pop_model, train_df, test_df, cutoffs=cutoffs)
        logger.info("Popularity Test Metrics: %s", pop_metrics)
        results["models"]["Popularity"] = {
            "config": recommenders_cfg.get("popularity", {}),
            "metrics": pop_metrics,
        }

    # 2. BPR-MF Baseline
    bpr_cfg = recommenders_cfg.get("bpr_mf", {})
    if bpr_cfg.get("enabled", True):
        logger.info("=== Fitting BPR-MF Baseline ===")
        bpr_model = BPRMFRecommender(
            embedding_dim=bpr_cfg.get("embedding_dim", 64),
            learning_rate=bpr_cfg.get("learning_rate", 0.001),
            l2_reg=bpr_cfg.get("l2_reg", 1e-4),
            epochs=bpr_cfg.get("epochs", 20),
            batch_size=bpr_cfg.get("batch_size", 1024),
            seed=seed,
        )
        bpr_model.fit(train_df)
        bpr_metrics = evaluate_model(bpr_model, train_df, test_df, cutoffs=cutoffs)
        logger.info("BPR-MF Test Metrics: %s", bpr_metrics)
        results["models"]["BPR_MF"] = {
            "config": bpr_cfg,
            "metrics": bpr_metrics,
        }

    # 3. LightGCN Baseline
    lgcn_cfg = recommenders_cfg.get("lightgcn", {})
    if lgcn_cfg.get("enabled", True):
        logger.info("=== Fitting LightGCN Baseline ===")
        lgcn_model = LightGCNRecommender(
            embedding_dim=lgcn_cfg.get("embedding_dim", 64),
            num_layers=lgcn_cfg.get("num_layers", 3),
            learning_rate=lgcn_cfg.get("learning_rate", 0.001),
            l2_reg=lgcn_cfg.get("l2_reg", 1e-4),
            epochs=lgcn_cfg.get("epochs", 20),
            batch_size=lgcn_cfg.get("batch_size", 1024),
            seed=seed,
        )
        lgcn_model.fit(train_df)
        lgcn_metrics = evaluate_model(lgcn_model, train_df, test_df, cutoffs=cutoffs)
        logger.info("LightGCN Test Metrics: %s", lgcn_metrics)
        results["models"]["LightGCN"] = {
            "config": lgcn_cfg,
            "metrics": lgcn_metrics,
        }

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    res_json = output_dir / "baseline_results.json"
    res_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Generate Markdown Summary
    md_lines = [
        "# VouchRec Phase 2: Recommendation Baseline Results",
        "",
        f"- **Evaluated at:** `{results['timestamp_utc']}`",
        f"- **Seed:** `{seed}`",
        f"- **Split Method:** `{split_info.get('split_method', 'global_temporal')}`",
        f"- **Training Interactions:** `{split_info.get('train_interactions', len(train_df)):,}`",
        f"- **Test Interactions:** `{split_info.get('test_interactions', len(test_df)):,}`",
        "",
        "## Summary Evaluation Table",
        "",
        "| Model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | HitRate@10 | HitRate@20 |",
        "|---|---|---|---|---|---|---|",
    ]

    for model_name, m_data in results["models"].items():
        m = m_data["metrics"]
        md_lines.append(
            f"| **{model_name}** | {m.get('Recall@10', 0.0):.4f} | {m.get('Recall@20', 0.0):.4f} | "
            f"{m.get('NDCG@10', 0.0):.4f} | {m.get('NDCG@20', 0.0):.4f} | "
            f"{m.get('HitRate@10', 0.0):.4f} | {m.get('HitRate@20', 0.0):.4f} |"
        )

    md_lines.extend(
        [
            "",
            "## Hyperparameters & Configurations",
            "",
        ]
    )
    for model_name, m_data in results["models"].items():
        md_lines.append(f"### {model_name}")
        md_lines.append("```json")
        md_lines.append(json.dumps(m_data["config"], indent=2))
        md_lines.append("```")
        md_lines.append("")

    res_md = output_dir / "baseline_results.md"
    res_md.write_text("\n".join(md_lines), encoding="utf-8")
    logger.info("Saved baseline experiment results to %s and %s", res_json, res_md)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate recommendation baselines.")
    parser.add_argument("--category", type=str, default=None, help="Dataset category")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override epochs for neural models"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    data_cfg = load_config("data")
    exp_cfg = load_config("experiment")
    model_cfg = load_config("model")
    paths = resolve_data_paths(data_cfg)

    if args.epochs is not None:
        model_cfg["recommenders"]["bpr_mf"]["epochs"] = args.epochs
        model_cfg["recommenders"]["lightgcn"]["epochs"] = args.epochs

    cat = args.category or data_cfg.get("amazon_reviews", {}).get("category", "Musical_Instruments")
    splits_dir = paths.splits / cat
    output_dir = paths.reports

    run_baseline_experiments(splits_dir, output_dir, model_cfg, exp_cfg)


if __name__ == "__main__":
    main()
