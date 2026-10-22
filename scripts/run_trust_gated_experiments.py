"""CLI script to run Trust-Gated LightGCN experiments, ablations, and baselines."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from vouchrec.attacks.config import AttackConfig
from vouchrec.attacks.evaluation import (
    AttackEvaluationReport,
    evaluate_target_item_impact,
    evaluate_trust_detection,
)
from vouchrec.attacks.simulator import AttackSimulator
from vouchrec.config import load_config, resolve_data_paths
from vouchrec.evaluation.metrics import evaluate_recommendations
from vouchrec.logging_utils import setup_logging
from vouchrec.models.base import BaseRecommender
from vouchrec.models.bpr_mf import BPRMFRecommender
from vouchrec.models.lightgcn import LightGCNRecommender
from vouchrec.models.popularity import PopularityRecommender
from vouchrec.models.trust_gated_lightgcn import (
    TrustGatedLightGCN,
    tune_hard_filter_threshold,
)

logger = logging.getLogger(__name__)


def format_markdown_report(
    reports: list[AttackEvaluationReport],
    hard_filter_val_results: dict[float, dict[str, float]],
    best_hard_thresh: float,
    manifest: dict[str, Any],
    edge_stats_by_model: dict[str, dict[str, Any]],
    output_path: Path,
) -> None:
    """Write comprehensive human-readable Markdown evaluation report."""
    lines = [
        "# VouchRec Phase 5: Trust-Gated LightGCN Evaluation Report",
        "",
        f"- **Simulation Timestamp:** `{manifest.get('simulation_timestamp_utc', 'N/A')}`",
        f"- **Attack Strategy:** `{manifest.get('strategy', 'N/A')}`",
        f"- **Attack Objective:** `{manifest.get('objective', 'N/A')}`",
        f"- **Target Product:** `{manifest.get('target_item', 'N/A')}`",
        f"- **Attack Budget:** `{manifest.get('attack_budget_fraction', 0.0) * 100:.1f}%` ({manifest.get('synthetic_users_count', 0)} synthetic profiles)",
        f"- **Synthetic Reviews Injected:** `{manifest.get('synthetic_interactions_count', 0)}`",
        f"- **Selected Hard-Filtering Threshold (Validation theta*):** `{best_hard_thresh:.2f}`",
        "",
        "## 1. Hard Filtering Threshold Selection (Validation Set Only)",
        "",
        "| Candidate Threshold | Val NDCG@10 | Val Recall@20 | Pruned Interactions | Pruned Fraction |",
        "|---|---|---|---|---|",
    ]

    for thresh, m in sorted(hard_filter_val_results.items()):
        is_best = " **(Selected theta*)**" if abs(thresh - best_hard_thresh) < 1e-4 else ""
        lines.append(
            f"| `{thresh:.2f}`{is_best} | {m.get('NDCG@10', 0.0):.4f} | "
            f"{m.get('Recall@20', 0.0):.4f} | {int(m.get('pruned_count', 0))} | "
            f"{m.get('pruned_fraction', 0.0) * 100:.1f}% |"
        )

    lines.extend([
        "",
        "## 2. Target Product Attack Impact (Clean vs. Attacked)",
        "",
        "| Model | Clean Target HR@10 | Attacked Target HR@10 | HR Delta | Clean Mean Rank | Attacked Mean Rank | Rank Promotion |",
        "|---|---|---|---|---|---|---|",
    ])

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
        "## 3. System-Wide Recommendation Quality (Collateral Distortion & Clean Utility)",
        "",
        "| Model | Clean NDCG@10 | Attacked NDCG@10 | NDCG Delta | Clean Recall@20 | Attacked Recall@20 | Runtime |",
        "|---|---|---|---|---|---|---|",
    ])

    for rep in reports:
        cm = rep.clean_metrics
        am = rep.attacked_metrics
        c_ndcg = cm.get("NDCG@10", 0.0)
        a_ndcg = am.get("NDCG@10", 0.0)
        delta_ndcg = a_ndcg - c_ndcg
        lines.append(
            f"| **{rep.model_name}** | {c_ndcg:.4f} | {a_ndcg:.4f} | {delta_ndcg:+.4f} | "
            f"{cm.get('Recall@20', 0.0):.4f} | {am.get('Recall@20', 0.0):.4f} | {rep.runtime_seconds:.2f}s |"
        )

    lines.extend([
        "",
        "## 4. Graph Edge Weight & Pruning Statistics",
        "",
        "| Model | Mode | Mean Edge Weight | Min Weight | Max Weight | Pruned Edges | Pruned % |",
        "|---|---|---|---|---|---|---|",
    ])

    for rep in reports:
        m_name = rep.model_name
        st = edge_stats_by_model.get(m_name, {})
        if st:
            lines.append(
                f"| **{m_name}** | {st.get('weighting_mode', 'N/A')} | "
                f"{st.get('mean_weight', 1.0):.4f} | {st.get('min_weight', 1.0):.4f} | "
                f"{st.get('max_weight', 1.0):.4f} | {st.get('n_pruned_interactions', 0)} | "
                f"{st.get('pruned_fraction', 0.0) * 100:.1f}% |"
            )
        else:
            lines.append(f"| **{m_name}** | unweighted / binary | 1.0000 | 1.0000 | 1.0000 | 0 | 0.0% |")

    lines.extend([
        "",
        "## 5. Required Ablations & Trade-Off Analysis",
        "",
        "1. **Vanilla LightGCN vs. Trust-Gated LightGCN (Soft):**",
        "   - Compares unweighted graph propagation against trust-damped message passing.",
        "2. **Soft Weighting vs. Hard Filtering:**",
        "   - Compares continuous message attenuation against discrete graph pruning (threshold selected on validation set).",
        "3. **Uniform Edge Weights vs. Trust-Derived Weights:**",
        "   - Evaluates whether trust-derived weighting outperforms equal-weight baselines in preserving clean NDCG while mitigating rank promotion.",
        "",
        "> [!IMPORTANT]",
        "> A defense that dampens target promotion but degrades clean recommendation utility must be treated as an engineering trade-off rather than an unconditional improvement.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved Phase 5 evaluation report to %s", output_path)


def run_trust_gated_experiments(
    splits_dir: Path,
    output_base_dir: Path,
    reports_dir: Path,
    attack_config: AttackConfig,
    epochs: int = 15,
) -> dict[str, Any]:
    """Execute end-to-end benchmark comparing all 5 models + ablations on clean and attacked graphs."""
    train_file = splits_dir / "train.parquet"
    val_file = splits_dir / "val.parquet"
    test_file = splits_dir / "test.parquet"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Missing train/test files in {splits_dir}.")

    train_df = pq.read_table(train_file).to_pandas()
    val_df = pq.read_table(val_file).to_pandas() if val_file.exists() else train_df.iloc[:2].copy()
    test_df = pq.read_table(test_file).to_pandas()

    # 1. Select hard filtering threshold theta* strictly using validation set
    best_hard_thresh, val_threshold_results = tune_hard_filter_threshold(
        train_df=train_df,
        val_df=val_df,
        candidate_thresholds=(0.4, 0.5, 0.6, 0.7, 0.8),
        epochs=min(epochs, 10),
        seed=attack_config.seed,
    )

    # 2. Simulate attack on training set
    simulator = AttackSimulator(attack_config)
    sim_result = simulator.simulate(train_df)

    exp_id = f"p5_{attack_config.strategy}_{attack_config.objective}_{attack_config.config_hash()}"
    sim_output_dir = output_base_dir / exp_id
    saved_paths = simulator.save_simulation(sim_result, sim_output_dir)

    target_item = sim_result.manifest["target_item"]
    clean_train_users = set(train_df["user_id"])

    # Ground-truth test interactions
    from collections import defaultdict
    gt: dict[str, set[str]] = defaultdict(set)
    for u, i in zip(test_df["user_id"], test_df["item_id"], strict=False):
        gt[u].add(i)
    eval_users = [u for u in gt if u in clean_train_users]

    # Helper function to evaluate one model configuration
    reports: list[AttackEvaluationReport] = []
    edge_stats_by_model: dict[str, dict[str, Any]] = {}

    def evaluate_model_pair(
        model_name: str,
        clean_model_factory: Any,
        attacked_model_factory: Any,
    ) -> None:
        t0 = time.time()
        logger.info("Evaluating model: %s", model_name)

        # A. Fit on clean
        c_model: BaseRecommender = clean_model_factory()
        c_model.fit(train_df)
        c_recs = c_model.recommend(eval_users, k=20, exclude_seen=True)
        c_metrics = evaluate_recommendations(c_recs, gt, cutoffs=(10, 20))

        # B. Fit on attacked
        a_model: BaseRecommender = attacked_model_factory()
        a_model.fit(sim_result.attacked_train_df)
        a_recs = a_model.recommend(eval_users, k=20, exclude_seen=True)
        a_metrics = evaluate_recommendations(a_recs, gt, cutoffs=(10, 20))

        # C. Impact & detection
        target_impact = evaluate_target_item_impact(c_recs, a_recs, target_item, top_k=10)
        detection_metrics = evaluate_trust_detection(
            sim_result.attacked_train_df, sim_result.ground_truth_labels_df
        )

        runtime = time.time() - t0
        rep = AttackEvaluationReport(
            model_name=model_name,
            target_item=target_item,
            clean_metrics=c_metrics,
            attacked_metrics=a_metrics,
            target_impact=target_impact,
            detection_metrics=detection_metrics,
            runtime_seconds=runtime,
            manifest=sim_result.manifest,
        )
        reports.append(rep)

        # Store edge stats if model is TrustGatedLightGCN
        if hasattr(a_model, "edge_stats_"):
            edge_stats_by_model[model_name] = getattr(a_model, "edge_stats_", {})

    # Model 1: Popularity
    evaluate_model_pair("Popularity", lambda: PopularityRecommender(), lambda: PopularityRecommender())

    # Model 2: BPR-MF
    evaluate_model_pair(
        "BPR_MF",
        lambda: BPRMFRecommender(embedding_dim=32, epochs=epochs, batch_size=256, seed=attack_config.seed),
        lambda: BPRMFRecommender(embedding_dim=32, epochs=epochs, batch_size=256, seed=attack_config.seed),
    )

    # Model 3: Vanilla LightGCN
    evaluate_model_pair(
        "Vanilla_LightGCN",
        lambda: LightGCNRecommender(embedding_dim=32, num_layers=2, epochs=epochs, batch_size=256, seed=attack_config.seed),
        lambda: LightGCNRecommender(embedding_dim=32, num_layers=2, epochs=epochs, batch_size=256, seed=attack_config.seed),
    )

    # Model 4: LightGCN Hard Filtering (theta*)
    evaluate_model_pair(
        f"LightGCN_HardFilter(theta={best_hard_thresh:.2f})",
        lambda: TrustGatedLightGCN(
            embedding_dim=32, num_layers=2, epochs=epochs, batch_size=256,
            seed=attack_config.seed, weighting_mode="hard", hard_threshold=best_hard_thresh,
        ),
        lambda: TrustGatedLightGCN(
            embedding_dim=32, num_layers=2, epochs=epochs, batch_size=256,
            seed=attack_config.seed, weighting_mode="hard", hard_threshold=best_hard_thresh,
        ),
    )

    # Model 5: Trust-Gated LightGCN (Soft Linear)
    evaluate_model_pair(
        "TrustGated_LightGCN_SoftLinear",
        lambda: TrustGatedLightGCN(
            embedding_dim=32, num_layers=2, epochs=epochs, batch_size=256,
            seed=attack_config.seed, weighting_mode="soft", soft_function="linear",
        ),
        lambda: TrustGatedLightGCN(
            embedding_dim=32, num_layers=2, epochs=epochs, batch_size=256,
            seed=attack_config.seed, weighting_mode="soft", soft_function="linear",
        ),
    )

    # Model 6: Trust-Gated LightGCN (Soft Power p=2)
    evaluate_model_pair(
        "TrustGated_LightGCN_SoftPower(p=2)",
        lambda: TrustGatedLightGCN(
            embedding_dim=32, num_layers=2, epochs=epochs, batch_size=256,
            seed=attack_config.seed, weighting_mode="soft", soft_function="power", power_p=2.0,
        ),
        lambda: TrustGatedLightGCN(
            embedding_dim=32, num_layers=2, epochs=epochs, batch_size=256,
            seed=attack_config.seed, weighting_mode="soft", soft_function="power", power_p=2.0,
        ),
    )

    # Model 7: Ablation - Uniform Weights in TrustGated Architecture
    evaluate_model_pair(
        "Ablation_UniformLightGCN",
        lambda: TrustGatedLightGCN(
            embedding_dim=32, num_layers=2, epochs=epochs, batch_size=256,
            seed=attack_config.seed, weighting_mode="uniform",
        ),
        lambda: TrustGatedLightGCN(
            embedding_dim=32, num_layers=2, epochs=epochs, batch_size=256,
            seed=attack_config.seed, weighting_mode="uniform",
        ),
    )

    # Save output artifacts
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "trust_gated_lightgcn_report.json"
    md_path = reports_dir / "trust_gated_lightgcn_report.md"

    summary_data = {
        "manifest": sim_result.manifest,
        "saved_paths": saved_paths,
        "hard_filtering_val_tuning": {
            "selected_threshold": best_hard_thresh,
            "results_by_threshold": val_threshold_results,
        },
        "edge_statistics": edge_stats_by_model,
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
    format_markdown_report(
        reports=reports,
        hard_filter_val_results=val_threshold_results,
        best_hard_thresh=best_hard_thresh,
        manifest=sim_result.manifest,
        edge_stats_by_model=edge_stats_by_model,
        output_path=md_path,
    )

    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Trust-Gated LightGCN and baselines.")
    parser.add_argument("--strategy", type=str, default="bandwagon", choices=["random", "average", "bandwagon"])
    parser.add_argument("--objective", type=str, default="push", choices=["push", "nuke"])
    parser.add_argument("--budget", type=float, default=0.05, help="Attack budget fraction")
    parser.add_argument("--filler-size", type=int, default=3, help="Filler items per profile")
    parser.add_argument("--burst-hours", type=float, default=48.0, help="Burst duration in hours")
    parser.add_argument("--target-item", type=str, default=None, help="Target item ID")
    parser.add_argument("--category", type=str, default=None, help="Dataset category")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
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

    run_trust_gated_experiments(splits_dir, output_dir, paths.reports, cfg, epochs=args.epochs)


if __name__ == "__main__":
    main()
