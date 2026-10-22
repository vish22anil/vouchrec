"""CLI script to run controlled attack simulations and evaluate baseline robustness."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from vouchrec.attacks.config import AttackConfig
from vouchrec.attacks.evaluation import AttackEvaluationReport, AttackExperimentRunner
from vouchrec.attacks.simulator import AttackSimulator
from vouchrec.config import load_config, resolve_data_paths
from vouchrec.logging_utils import setup_logging
from vouchrec.models.bpr_mf import BPRMFRecommender
from vouchrec.models.lightgcn import LightGCNRecommender
from vouchrec.models.popularity import PopularityRecommender

logger = logging.getLogger(__name__)


def format_attack_markdown_report(
    reports: list[AttackEvaluationReport],
    manifest: dict[str, Any],
    output_path: Path,
) -> None:
    """Format clean vs attacked experimental comparison into GitHub Flavored Markdown."""
    lines = [
        "# VouchRec Phase 4: Controlled Attack Simulation Report",
        "",
        f"- **Simulation Timestamp:** `{manifest.get('simulation_timestamp_utc')}`",
        f"- **Attack Strategy:** `{manifest.get('strategy')}`",
        f"- **Attack Objective:** `{manifest.get('objective')}`",
        f"- **Target Product:** `{manifest.get('target_item')}`",
        f"- **Attack Budget:** `{manifest.get('attack_budget_fraction', 0.0) * 100:.1f}%` ({manifest.get('synthetic_users_count')} synthetic profiles)",
        f"- **Synthetic Reviews Injected:** `{manifest.get('synthetic_interactions_count')}`",
        f"- **Seed:** `{manifest.get('seed')}`",
        "",
        "## 1. Recommendation Impact on Target Product",
        "",
        "| Model | Clean HitRate@10 | Attacked HitRate@10 | HitRate Delta | Clean Mean Rank | Attacked Mean Rank | Rank Delta |",
        "|---|---|---|---|---|---|---|",
    ]

    for rep in reports:
        ti = rep.target_impact
        lines.append(
            f"| **{rep.model_name}** | {ti['target_hitrate_at_10_clean']:.4f} | "
            f"{ti['target_hitrate_at_10_attacked']:.4f} | {ti['target_hitrate_delta']:+.4f} | "
            f"{ti['target_mean_rank_clean']:.1f} | {ti['target_mean_rank_attacked']:.1f} | "
            f"{ti['target_rank_improvement']:+.1f} |"
        )

    lines.extend([
        "",
        "## 2. System-Wide Recommendation Quality (Collateral Distortion)",
        "",
        "| Model | Clean NDCG@10 | Attacked NDCG@10 | Clean Recall@20 | Attacked Recall@20 | Runtime (s) |",
        "|---|---|---|---|---|---|",
    ])

    for rep in reports:
        cm = rep.clean_metrics
        am = rep.attacked_metrics
        lines.append(
            f"| **{rep.model_name}** | {cm.get('NDCG@10', 0.0):.4f} | {am.get('NDCG@10', 0.0):.4f} | "
            f"{cm.get('Recall@20', 0.0):.4f} | {am.get('Recall@20', 0.0):.4f} | {rep.runtime_seconds:.2f}s |"
        )

    lines.extend([
        "",
        "## 3. Trust Model Detection Performance on Ground-Truth Synthetic Labels",
        "",
    ])

    # Pull detection metrics from first report
    det = reports[0].detection_metrics if reports else {}
    lines.extend([
        f"- **Mean Trust on Real Reviews:** `{det.get('mean_trust_real_reviews', 0.0):.4f}`",
        f"- **Mean Trust on Synthetic Reviews:** `{det.get('mean_trust_synthetic_reviews', 0.0):.4f}`",
        f"- **Trust Separation Gap:** `{det.get('trust_separation_delta', 0.0):+.4f}` (positive indicates synthetic reviews receive lower trust)",
        f"- **Detection ROC-AUC:** `{det.get('trust_detection_roc_auc', 0.50):.4f}`",
        "",
        "> [!NOTE]",
        "> Detection metrics are calculated strictly offline using known simulated ground-truth labels.",
        "> In accordance with anti-leakage principles, these labels were never exposed to recommendation models or trust feature extractors.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved attack simulation report to %s", output_path)


def run_attack_experiment(
    splits_dir: Path,
    output_base_dir: Path,
    reports_dir: Path,
    attack_config: AttackConfig,
    epochs: int = 15,
) -> dict[str, Any]:
    """Execute complete attack generation, baseline retraining, and evaluation."""
    train_file = splits_dir / "train.parquet"
    test_file = splits_dir / "test.parquet"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Missing train/test files in {splits_dir}. Run create_splits first.")

    train_df = pq.read_table(train_file).to_pandas()
    test_df = pq.read_table(test_file).to_pandas()

    # 1. Simulate attack
    simulator = AttackSimulator(attack_config)
    sim_result = simulator.simulate(train_df)

    # 2. Save simulation files to separate experiment directory
    exp_id = f"{attack_config.strategy}_{attack_config.objective}_{attack_config.config_hash()}"
    sim_output_dir = output_base_dir / exp_id
    saved_paths = simulator.save_simulation(sim_result, sim_output_dir)

    # 3. Setup experiment runner
    target_item = sim_result.manifest["target_item"]
    runner = AttackExperimentRunner(
        clean_train_df=train_df,
        test_df=test_df,
        target_item=target_item,
        cutoffs=(10, 20),
    )

    # 4. Benchmark models on clean vs attacked data
    reports: list[AttackEvaluationReport] = []

    # Model A: Popularity
    reports.append(runner.run_comparison(lambda: PopularityRecommender(), "Popularity", sim_result))

    # Model B: BPR-MF
    reports.append(
        runner.run_comparison(
            lambda: BPRMFRecommender(
                embedding_dim=32,
                epochs=epochs,
                batch_size=256,
                seed=attack_config.seed,
            ),
            "BPR_MF",
            sim_result,
        )
    )

    # Model C: LightGCN
    reports.append(
        runner.run_comparison(
            lambda: LightGCNRecommender(
                embedding_dim=32,
                num_layers=2,
                epochs=epochs,
                batch_size=256,
                seed=attack_config.seed,
            ),
            "LightGCN",
            sim_result,
        )
    )

    # 5. Save machine-readable and markdown reports
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "attack_simulation_report.json"
    md_path = reports_dir / "attack_simulation_report.md"

    summary_data = {
        "manifest": sim_result.manifest,
        "saved_paths": saved_paths,
        "reports": [
            {
                "model_name": r.model_name,
                "clean_metrics": r.clean_metrics,
                "attacked_metrics": r.attacked_metrics,
                "target_impact": r.target_impact,
                "detection_metrics": r.detection_metrics,
                "runtime_seconds": r.runtime_seconds,
            }
            for r in reports
        ],
    }

    json_path.write_text(json.dumps(summary_data, indent=2), encoding="utf-8")
    format_attack_markdown_report(reports, sim_result.manifest, md_path)

    logger.info("Saved attack experiment outputs to %s and %s", json_path, md_path)
    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run controlled attack simulation and evaluate baseline impact.")
    parser.add_argument("--strategy", type=str, default="bandwagon", choices=["random", "average", "bandwagon"])
    parser.add_argument("--objective", type=str, default="push", choices=["push", "nuke"])
    parser.add_argument("--budget", type=float, default=0.05, help="Attack budget fraction (e.g. 0.01, 0.05)")
    parser.add_argument("--filler-size", type=int, default=3, help="Filler items per synthetic profile")
    parser.add_argument("--burst-hours", type=float, default=None, help="Concentrate reviews in temporal burst (hours)")
    parser.add_argument("--target-item", type=str, default=None, help="Target item ID (or automatic selection)")
    parser.add_argument("--category", type=str, default=None, help="Dataset category")
    parser.add_argument("--epochs", type=int, default=15, help="Epochs for neural baselines")
    parser.add_argument("--seed", type=int, default=20261021, help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    data_cfg = load_config("data")
    paths = resolve_data_paths(data_cfg)

    cat = args.category or data_cfg.get("amazon_reviews", {}).get("category", "Musical_Instruments")
    splits_dir = paths.splits / cat
    output_dir = paths.attack_simulations / cat

    cfg = AttackConfig(
        strategy=args.strategy,
        objective=args.objective,
        attack_budget_fraction=args.budget,
        filler_size=args.filler_size,
        burst_window_hours=args.burst_hours,
        target_item=args.target_item,
        seed=args.seed,
    )

    run_attack_experiment(splits_dir, output_dir, paths.reports, cfg, epochs=args.epochs)


if __name__ == "__main__":
    main()
