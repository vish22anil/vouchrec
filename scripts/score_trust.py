"""CLI script to run trust signal extraction, fusion, and generate audit reports."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from vouchrec.config import load_config, resolve_data_paths
from vouchrec.data.io_utils import save_parquet
from vouchrec.logging_utils import setup_logging
from vouchrec.trust.fusion import ProvisionalTrustFusion

logger = logging.getLogger(__name__)


def generate_trust_markdown_report(result: Any, output_path: Path) -> None:
    """Write an interpretable research evaluation report for provisional trust scoring."""
    stats = result.summary_stats
    weights = result.weights
    df = result.scores_df

    # Extract reason code counts
    reason_counter: dict[str, int] = {}
    for reasons_str in df["trust_reasons"]:
        for r in reasons_str.split("; "):
            r_clean = r.strip()
            if r_clean:
                # Group dynamic reasons by prefix
                base_code = r_clean.split("_USERS")[0] if "_USERS" in r_clean else r_clean
                reason_counter[base_code] = reason_counter.get(base_code, 0) + 1

    lines = [
        "# VouchRec Phase 3: Provisional Trust Scoring Evaluation Report",
        "",
        f"- **Generated at:** `{datetime.datetime.now(datetime.UTC).isoformat()}`",
        f"- **Total Reviews Scored:** `{stats['n_reviews']:,}`",
        f"- **Provisional Trust Mean (±Std):** `{stats['provisional_trust_mean']:.4f} (±{stats['provisional_trust_std']:.4f})`",
        f"- **Provisional Trust Range:** `[{stats['provisional_trust_min']:.4f}, {stats['provisional_trust_max']:.4f}]`",
        f"- **Mean Signal Confidence:** `{stats['mean_confidence']:.2f}`",
        "",
        "## 1. Fusion Configuration & Weights",
        "",
        "| Component Head | Configured Weight | Mean Head Score |",
        "|---|---|---|",
        f"| **Review Text Head** | {weights['weight_text']:.2f} | {stats['text_trust_mean']:.4f} |",
        f"| **Reviewer Behavior Head** | {weights['weight_behavior']:.2f} | {stats['behavior_trust_mean']:.4f} |",
        f"| **Coordination Head** | {weights['weight_coordination']:.2f} | {stats['coordination_trust_mean']:.4f} |",
        "",
        "## 2. Triggered Reason Code Frequencies",
        "",
        "| Reason Code | Count | Share | Explanation |",
        "|---|---|---|---|",
    ]

    total_reviews = max(1, stats["n_reviews"])
    for code, count in sorted(reason_counter.items(), key=lambda x: -x[1]):
        share = count / total_reviews * 100
        desc = "Standard baseline"
        if "SPARSE_HISTORY" in code:
            desc = "Sparse user history (assigned neutral 0.5 prior)"
        elif "INDEPENDENT" in code:
            desc = "No cross-review duplicate or burst coordination detected"
        elif "DUPLICATE_TEXT" in code:
            desc = "Identical review text posted across distinct reviewers"
        elif "MONOTONE" in code:
            desc = "Reviewer history with zero rating variance (all identical stars)"
        elif "BURST" in code:
            desc = "Temporal cluster of reviews on item or reviewer flood"
        elif "MISSING" in code or "SHORT" in code:
            desc = "Missing or minimal text body"
        elif "DIVERSE" in code or "ESTABLISHED" in code:
            desc = "Healthy detail or established longitudinal activity"

        lines.append(f"| `{code}` | {count:,} | {share:.1f}% | {desc} |")

    lines.extend(
        [
            "",
            "## 3. Methodological Scope & Validation Boundary",
            "",
            "> [!IMPORTANT]",
            "> **Provisional Status:** The scores above reflect unsupervised anomaly detection and rule-based behavioral heuristics.",
            "> Because Amazon Reviews 2023 lacks ground-truth fake review annotations, **no supervised classification accuracy, AUC, or F1 scores can be claimed** at this stage.",
            "> In Phase 4, controlled attack simulations will provide known ground-truth labels to rigorously benchmark and calibrate this pipeline.",
            "",
            "## 4. Signal Correlation Matrix",
            "",
        ]
    )

    corr_df = df[
        [
            "text_trust_score",
            "behavior_trust_score",
            "coordination_trust_score",
            "provisional_trust_score",
        ]
    ].corr()
    lines.append("| Metric | Text | Behavior | Coordination | Provisional Fused |")
    lines.append("|---|---|---|---|---|")
    for col in corr_df.columns:
        row_vals = " | ".join(
            f"{corr_df.loc[col, c]:.4f}" if not pd.isna(corr_df.loc[col, c]) else "N/A*"
            for c in corr_df.columns
        )
        lines.append(f"| **{col}** | {row_vals} |")

    lines.append("")
    lines.append(
        "\\* *N/A indicates zero variance in the component head across the evaluated sample.*"
    )

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Saved trust Markdown report to %s", output_path)


def run_trust_scoring(
    input_parquet: Path,
    output_parquet: Path,
    reports_dir: Path,
    config: dict[str, Any] | None = None,
) -> Any:
    """Run trust scoring pipeline on a parquet dataset."""
    logger.info("Reading input reviews from %s", input_parquet)
    table = pq.read_table(input_parquet)
    df = table.to_pandas()

    fusion = ProvisionalTrustFusion()
    result = fusion.score_reviews(df)

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    import pyarrow as pa

    out_table = pa.Table.from_pandas(result.scores_df, preserve_index=False)
    save_parquet(out_table, output_parquet)
    logger.info(
        "Saved enriched dataset with trust scores (%d rows) to %s", len(out_table), output_parquet
    )

    # Save reports
    reports_dir.mkdir(parents=True, exist_ok=True)
    json_path = reports_dir / "trust_evaluation_report.json"
    md_path = reports_dir / "trust_evaluation_report.md"

    report_data = {
        "summary_stats": result.summary_stats,
        "weights": result.weights,
        "input_dataset": str(input_parquet),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    json_path.write_text(json.dumps(report_data, indent=2), encoding="utf-8")
    generate_trust_markdown_report(result, md_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run provisional trust scoring pipeline.")
    parser.add_argument("--input", type=str, default=None, help="Path to input Parquet file")
    parser.add_argument("--output", type=str, default=None, help="Path to output Parquet file")
    parser.add_argument("--category", type=str, default=None, help="Category name")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    data_cfg = load_config("data")
    paths = resolve_data_paths(data_cfg)

    cat = args.category or data_cfg.get("amazon_reviews", {}).get("category", "Musical_Instruments")
    in_path = Path(args.input) if args.input else paths.processed / cat / "reviews.parquet"
    out_path = (
        Path(args.output) if args.output else paths.processed / cat / "reviews_with_trust.parquet"
    )

    run_trust_scoring(in_path, out_path, paths.reports)


if __name__ == "__main__":
    main()
