"""CLI script to run Dynamic Exposure Governor experiments and benchmark safeguards."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections import defaultdict
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
from vouchrec.governance.config import ExposureGovernorConfig
from vouchrec.governance.exposure_governor import (
    ExposureGovernor,
    compute_item_trust_evidence,
)
from vouchrec.logging_utils import setup_logging
from vouchrec.models.lightgcn import LightGCNRecommender
from vouchrec.models.trust_gated_lightgcn import TrustGatedLightGCN

logger = logging.getLogger(__name__)


def format_governor_markdown_report(
    reports: list[AttackEvaluationReport],
    governor_stats: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    gov_config: ExposureGovernorConfig,
    output_path: Path,
) -> None:
    """Format human-readable Markdown evaluation report for Phase 6."""
    lines = [
        "# VouchRec Phase 6: Dynamic Exposure Governor Evaluation Report",
        "",
        f"- **Simulation Timestamp:** `{manifest.get('simulation_timestamp_utc', 'N/A')}`",
        f"- **Attack Strategy:** `{manifest.get('strategy', 'N/A')}`",
        f"- **Attack Objective:** `{manifest.get('objective', 'N/A')}`",
        f"- **Target Product:** `{manifest.get('target_item', 'N/A')}`",
        f"- **Attack Budget:** `{manifest.get('attack_budget_fraction', 0.0) * 100:.1f}%` ({manifest.get('synthetic_users_count', 0)} synthetic profiles)",
        f"- **Synthetic Reviews Injected:** `{manifest.get('synthetic_interactions_count', 0)}`",
        f"- **Governor Max Permitted Rank Jump (Delta r_max):** `{gov_config.max_rank_jump}`",
        f"- **Minimum Evidence Reviews Required:** `{gov_config.min_evidence_reviews}`",
        f"- **Low-Trust Review Cutoff:** `{gov_config.low_trust_threshold:.2f}`",
        f"- **Suspicious Ratio Threshold:** `{gov_config.suspicious_ratio_threshold * 100:.0f}%`",
        "",
        "## 1. Experimental Conditions Comparison",
        "",
        "| Condition | Clean NDCG@10 | Attacked NDCG@10 | NDCG Delta | Target HR@10 (C -> A) | Target Mean Rank (C -> A) | Rank Promotion | Recommendations Adjusted | Jumps Capped | Runtime |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    for rep in reports:
        cm = rep.clean_metrics
        am = rep.attacked_metrics
        ti = rep.target_impact
        gst = governor_stats.get(rep.model_name, {})

        c_ndcg = cm.get("NDCG@10", 0.0)
        a_ndcg = am.get("NDCG@10", 0.0)
        delta_ndcg = a_ndcg - c_ndcg

        c_hr = ti.get("target_hitrate_at_10_clean", 0.0)
        a_hr = ti.get("target_hitrate_at_10_attacked", 0.0)
        c_rank = ti.get("target_mean_rank_clean", 0.0)
        a_rank = ti.get("target_mean_rank_attacked", 0.0)
        promo = ti.get("target_rank_improvement", 0.0)

        n_adj = gst.get("total_adjusted", 0)
        pct_adj = gst.get("percentage_adjusted", 0.0)
        n_capped = gst.get("total_jumps_capped", 0)

        lines.append(
            f"| **{rep.model_name}** | {c_ndcg:.4f} | {a_ndcg:.4f} | {delta_ndcg:+.4f} | "
            f"{c_hr:.2f} -> {a_hr:.2f} | {c_rank:.1f} -> {a_rank:.1f} | {promo:+.1f} | "
            f"{n_adj} ({pct_adj:.1f}%) | {n_capped} | {rep.runtime_seconds:.2f}s |"
        )

    lines.extend([
        "",
        "## 2. Governor Audit Trail & Reason Code Distribution",
        "",
        "| Condition | GOV_RANK_JUMP_CAPPED | GOV_PASS_UNMODIFIED | GOV_PRESERVED_SPARSE_EVIDENCE | GOV_PRESERVED_HIGH_TRUST | GOV_DISABLED |",
        "|---|---|---|---|---|---|",
    ])

    for rep in reports:
        gst = governor_stats.get(rep.model_name, {})
        rc = gst.get("reason_code_counts", {})
        lines.append(
            f"| **{rep.model_name}** | {rc.get('GOV_RANK_JUMP_CAPPED', 0)} | "
            f"{rc.get('GOV_PASS_UNMODIFIED', 0)} | {rc.get('GOV_PRESERVED_SPARSE_EVIDENCE', 0)} | "
            f"{rc.get('GOV_PRESERVED_HIGH_TRUST', 0)} | {rc.get('GOV_DISABLED', 0)} |"
        )

    lines.extend([
        "",
        "## 3. Analysis & Key Insights",
        "",
        "1. **Exposure Governor vs. Un-Governed Baseline:**",
        "   - Checks if the post-recommendation ranking cap successfully arrests target item promotion.",
        "2. **Safeguard Protections (Cold / Unpopular Items):**",
        "   - Confirms that items with sparse review history are protected via `GOV_PRESERVED_SPARSE_EVIDENCE`.",
        "3. **Trade-Off Evaluation:**",
        "   - Audits whether governance preserves clean recommendation NDCG while suppressing attacked promotion.",
        "",
        "> [!IMPORTANT]",
        "> The Exposure Governor limits ranking influence; it does not certify reviews as genuine or fraudulent.",
        "",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved Phase 6 evaluation report to %s", output_path)


def run_exposure_governor_experiments(
    splits_dir: Path,
    output_base_dir: Path,
    reports_dir: Path,
    attack_config: AttackConfig,
    gov_config: ExposureGovernorConfig,
    epochs: int = 15,
) -> dict[str, Any]:
    """Execute clean vs attacked experiments across Vanilla, Trust-Gated, and Governed conditions."""
    train_file = splits_dir / "train.parquet"
    test_file = splits_dir / "test.parquet"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(f"Missing train/test files in {splits_dir}.")

    train_df = pq.read_table(train_file).to_pandas()
    test_df = pq.read_table(test_file).to_pandas()

    # 1. Simulate attack on training set
    simulator = AttackSimulator(attack_config)
    sim_result = simulator.simulate(train_df)

    exp_id = f"p6_{attack_config.strategy}_{attack_config.objective}_{attack_config.config_hash()}"
    sim_output_dir = output_base_dir / exp_id
    saved_paths = simulator.save_simulation(sim_result, sim_output_dir)

    target_item = sim_result.manifest["target_item"]
    clean_train_users = set(train_df["user_id"])

    gt: dict[str, set[str]] = defaultdict(set)
    for u, i in zip(test_df["user_id"], test_df["item_id"], strict=False):
        gt[u].add(i)
    eval_users = [u for u in gt if u in clean_train_users]

    # 2. Compute item trust evidence for clean and attacked sets
    clean_item_evidence = compute_item_trust_evidence(
        train_df,
        low_trust_threshold=gov_config.low_trust_threshold,
    )
    attacked_item_evidence = compute_item_trust_evidence(
        sim_result.attacked_train_df,
        low_trust_threshold=gov_config.low_trust_threshold,
    )

    # 3. Train base reference model (clean Vanilla LightGCN) to provide baseline rankings
    logger.info("Training reference baseline for rank-jump comparison...")
    ref_model = LightGCNRecommender(
        embedding_dim=32,
        num_layers=2,
        epochs=epochs,
        batch_size=256,
        seed=attack_config.seed,
    )
    ref_model.fit(train_df)
    clean_baseline_recs = ref_model.recommend(eval_users, k=20, exclude_seen=True)

    # 4. Define evaluation conditions
    reports: list[AttackEvaluationReport] = []
    governor_stats: dict[str, dict[str, Any]] = {}
    governor = ExposureGovernor(gov_config)
    disabled_governor = ExposureGovernor(ExposureGovernorConfig(enabled=False))

    def evaluate_condition(
        condition_name: str,
        is_trust_gated: bool,
        apply_governor: bool,
    ) -> None:
        t0 = time.time()
        logger.info("Evaluating condition: %s", condition_name)

        # Fit model on clean
        if is_trust_gated:
            c_model = TrustGatedLightGCN(
                embedding_dim=32, num_layers=2, epochs=epochs, batch_size=256,
                seed=attack_config.seed, weighting_mode="soft", soft_function="linear",
            )
        else:
            c_model = LightGCNRecommender(
                embedding_dim=32, num_layers=2, epochs=epochs, batch_size=256,
                seed=attack_config.seed,
            )
        c_model.fit(train_df)
        raw_c_recs = c_model.recommend(eval_users, k=20, exclude_seen=True)

        # Fit model on attacked
        if is_trust_gated:
            a_model = TrustGatedLightGCN(
                embedding_dim=32, num_layers=2, epochs=epochs, batch_size=256,
                seed=attack_config.seed, weighting_mode="soft", soft_function="linear",
            )
        else:
            a_model = LightGCNRecommender(
                embedding_dim=32, num_layers=2, epochs=epochs, batch_size=256,
                seed=attack_config.seed,
            )
        a_model.fit(sim_result.attacked_train_df)
        raw_a_recs = a_model.recommend(eval_users, k=20, exclude_seen=True)

        # Apply Governor if enabled
        gov = governor if apply_governor else disabled_governor
        gov_c_res = gov.govern_recommendations(
            recommendations=raw_c_recs,
            baseline_recommendations=clean_baseline_recs,
            item_evidence=clean_item_evidence,
            k=20,
        )
        gov_a_res = gov.govern_recommendations(
            recommendations=raw_a_recs,
            baseline_recommendations=clean_baseline_recs,
            item_evidence=attacked_item_evidence,
            k=20,
        )

        final_c_recs = gov_c_res.governed_recommendations
        final_a_recs = gov_a_res.governed_recommendations

        c_metrics = evaluate_recommendations(final_c_recs, gt, cutoffs=(10, 20))
        a_metrics = evaluate_recommendations(final_a_recs, gt, cutoffs=(10, 20))
        target_impact = evaluate_target_item_impact(final_c_recs, final_a_recs, target_item, top_k=10)
        detection_metrics = evaluate_trust_detection(
            sim_result.attacked_train_df, sim_result.ground_truth_labels_df
        )

        runtime = time.time() - t0
        rep = AttackEvaluationReport(
            model_name=condition_name,
            target_item=target_item,
            clean_metrics=c_metrics,
            attacked_metrics=a_metrics,
            target_impact=target_impact,
            detection_metrics=detection_metrics,
            runtime_seconds=runtime,
            manifest=sim_result.manifest,
        )
        reports.append(rep)
        governor_stats[condition_name] = gov_a_res.summary_stats

    # Condition 1: Vanilla LightGCN (No defense)
    evaluate_condition(
        condition_name="Vanilla_LightGCN",
        is_trust_gated=False,
        apply_governor=False,
    )

    # Condition 2: Trust-Gated LightGCN (In-model defense only)
    evaluate_condition(
        condition_name="TrustGated_LightGCN_Ungoverned",
        is_trust_gated=True,
        apply_governor=False,
    )

    # Condition 3: Trust-Gated LightGCN + Exposure Governor (Full pipeline defense)
    evaluate_condition(
        condition_name="TrustGated_LightGCN_Governed",
        is_trust_gated=True,
        apply_governor=True,
    )

    # Condition 4: Vanilla LightGCN + Exposure Governor (Standalone post-processor test)
    evaluate_condition(
        condition_name="Vanilla_LightGCN_Governed",
        is_trust_gated=False,
        apply_governor=True,
    )

    # 5. Save reports
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "exposure_governor_report.json"
    md_path = reports_dir / "exposure_governor_report.md"

    summary_data = {
        "manifest": sim_result.manifest,
        "saved_paths": saved_paths,
        "governor_config": {
            "enabled": gov_config.enabled,
            "max_rank_jump": gov_config.max_rank_jump,
            "min_evidence_reviews": gov_config.min_evidence_reviews,
            "min_low_trust_reviews": gov_config.min_low_trust_reviews,
            "low_trust_threshold": gov_config.low_trust_threshold,
            "suspicious_ratio_threshold": gov_config.suspicious_ratio_threshold,
        },
        "governor_stats": governor_stats,
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
    format_governor_markdown_report(
        reports=reports,
        governor_stats=governor_stats,
        manifest=sim_result.manifest,
        gov_config=gov_config,
        output_path=md_path,
    )

    return summary_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 6 Dynamic Exposure Governor.")
    parser.add_argument("--strategy", type=str, default="bandwagon", choices=["random", "average", "bandwagon"])
    parser.add_argument("--objective", type=str, default="push", choices=["push", "nuke"])
    parser.add_argument("--budget", type=float, default=0.05, help="Attack budget fraction")
    parser.add_argument("--filler-size", type=int, default=3, help="Filler items per profile")
    parser.add_argument("--burst-hours", type=float, default=48.0, help="Burst duration in hours")
    parser.add_argument("--target-item", type=str, default=None, help="Target item ID")
    parser.add_argument("--category", type=str, default=None, help="Dataset category")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--max-rank-jump", type=int, default=3, help="Max permitted rank jump")
    parser.add_argument("--low-trust-threshold", type=float, default=0.60, help="Low-trust cutoff")
    parser.add_argument("--suspicious-ratio-threshold", type=float, default=0.40, help="Suspicious ratio cutoff")
    parser.add_argument("--min-evidence-reviews", type=int, default=2, help="Min review volume")
    parser.add_argument("--seed", type=int, default=20261021, help="Random seed")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    data_cfg = load_config("data")
    paths = resolve_data_paths(data_cfg)

    cat = args.category or data_cfg.get("amazon_reviews", {}).get("category", "Musical_Instruments")
    splits_dir = paths.splits / cat
    output_dir = paths.attack_simulations / cat

    attack_cfg = AttackConfig(
        strategy=args.strategy,
        objective=args.objective,
        attack_budget_fraction=args.budget,
        filler_size=args.filler_size,
        burst_window_hours=args.burst_hours,
        target_item=args.target_item,
        seed=args.seed,
    )

    gov_cfg = ExposureGovernorConfig(
        max_rank_jump=args.max_rank_jump,
        low_trust_threshold=args.low_trust_threshold,
        suspicious_ratio_threshold=args.suspicious_ratio_threshold,
        min_evidence_reviews=args.min_evidence_reviews,
    )

    run_exposure_governor_experiments(
        splits_dir=splits_dir,
        output_base_dir=output_dir,
        reports_dir=paths.reports,
        attack_config=attack_cfg,
        gov_config=gov_cfg,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
