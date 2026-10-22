"""Data quality report generator for raw, interim, and processed datasets."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq

from vouchrec.config import load_config, resolve_data_paths
from vouchrec.data.schema import CANONICAL_COLUMNS, FLAG_COLUMNS
from vouchrec.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def generate_quality_report(
    interim_path: Path,
    processed_path: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Compute detailed audit metrics from interim and processed Parquet files."""
    interim_table = pq.read_table(interim_path)
    proc_table = pq.read_table(processed_path)

    interim_df = interim_table.to_pandas()
    proc_df = proc_table.to_pandas()

    manifest_data: dict[str, Any] = {}
    if manifest_path and manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Could not parse manifest at %s: %s", manifest_path, e)

    # 1. Volume & Exclusions
    total_raw = len(interim_df)
    total_processed = len(proc_df)
    total_excluded = total_raw - total_processed

    flags_summary: dict[str, int] = {}
    for col in FLAG_COLUMNS:
        if col in interim_df.columns:
            flags_summary[col] = int(interim_df[col].sum())

    # 2. Rating distribution
    ratings = proc_df["rating"].dropna()
    rating_counts = proc_df["rating"].value_counts().sort_index().to_dict()
    rating_stats = {
        "mean": float(ratings.mean()) if not ratings.empty else 0.0,
        "std": float(ratings.std()) if not ratings.empty else 0.0,
        "median": float(ratings.median()) if not ratings.empty else 0.0,
        "distribution": {str(k): int(v) for k, v in rating_counts.items()},
    }

    # 3. Users and Items
    n_users = int(proc_df["user_id"].nunique())
    n_items = int(proc_df["item_id"].nunique())
    sparsity = 1.0 - (total_processed / (n_users * n_items)) if n_users > 0 and n_items > 0 else 1.0

    user_counts = proc_df["user_id"].value_counts()
    item_counts = proc_df["item_id"].value_counts()

    user_interaction_stats = {
        "min": int(user_counts.min()) if not user_counts.empty else 0,
        "max": int(user_counts.max()) if not user_counts.empty else 0,
        "mean": float(user_counts.mean()) if not user_counts.empty else 0.0,
        "median": float(user_counts.median()) if not user_counts.empty else 0.0,
        "users_ge_5": int((user_counts >= 5).sum()),
    }
    item_interaction_stats = {
        "min": int(item_counts.min()) if not item_counts.empty else 0,
        "max": int(item_counts.max()) if not item_counts.empty else 0,
        "mean": float(item_counts.mean()) if not item_counts.empty else 0.0,
        "median": float(item_counts.median()) if not item_counts.empty else 0.0,
        "items_ge_5": int((item_counts >= 5).sum()),
    }

    # 4. Verified purchases
    vp_counts = proc_df["verified_purchase"].value_counts(dropna=False).to_dict()
    verified_stats = {
        "true": int(vp_counts.get(True, 0)),
        "false": int(vp_counts.get(False, 0)),
        "null": int(vp_counts.get(None, 0)) + int(vp_counts.get(np.nan, 0)),
        "verified_fraction": float(proc_df["verified_purchase"].eq(True).mean())
        if not proc_df.empty
        else 0.0,
    }

    # 5. Timestamps
    ts_col = proc_df["timestamp"].dropna()
    min_ts = ts_col.min().isoformat() if not ts_col.empty else None
    max_ts = ts_col.max().isoformat() if not ts_col.empty else None

    # 6. Missingness in processed data
    missingness = {
        col: int(proc_df[col].isna().sum()) for col in CANONICAL_COLUMNS if col in proc_df.columns
    }

    report = {
        "dataset_provenance": manifest_data,
        "volume": {
            "raw_records": total_raw,
            "processed_records": total_processed,
            "excluded_records": total_excluded,
            "exclusion_rate_pct": round((total_excluded / total_raw * 100), 4)
            if total_raw > 0
            else 0.0,
        },
        "quality_flags_audit": flags_summary,
        "entity_counts": {
            "unique_users": n_users,
            "unique_items": n_items,
            "interactions": total_processed,
            "sparsity": round(sparsity, 6),
        },
        "user_activity": user_interaction_stats,
        "item_popularity": item_interaction_stats,
        "rating_analysis": rating_stats,
        "verified_purchase_analysis": verified_stats,
        "temporal_range": {
            "min_timestamp_utc": min_ts,
            "max_timestamp_utc": max_ts,
        },
        "processed_missing_values": missingness,
    }

    return report


def format_markdown_report(report: dict[str, Any]) -> str:
    """Format quality report dictionary into GitHub Flavored Markdown."""
    prov = report.get("dataset_provenance", {})
    vol = report.get("volume", {})
    flags = report.get("quality_flags_audit", {})
    entities = report.get("entity_counts", {})
    ratings = report.get("rating_analysis", {})
    vp = report.get("verified_purchase_analysis", {})
    u_act = report.get("user_activity", {})
    i_pop = report.get("item_popularity", {})
    time_r = report.get("temporal_range", {})

    lines = [
        "# VouchRec Data Quality and Audit Report",
        "",
        "## 1. Provenance and Overview",
        "",
        f"- **Dataset Source:** `{prov.get('source_dataset', 'amazon_reviews_2023')}`",
        f"- **Category:** `{prov.get('category', 'Musical_Instruments')}`",
        f"- **Source URL:** {prov.get('source_url', 'N/A')}",
        f"- **Downloaded Timestamp (UTC):** `{prov.get('download_timestamp_utc', 'N/A')}`",
        f"- **Raw File Size:** `{prov.get('bytes', 0):,} bytes`",
        f"- **Raw File SHA-256:** `{prov.get('sha256', 'N/A')}`",
        "",
        "## 2. Record Volume and Cleaning Exclusions",
        "",
        "| Metric | Count | Percentage |",
        "|---|---|---|",
        f"| Raw Records | {vol.get('raw_records', 0):,} | 100.00% |",
        f"| Excluded Records | {vol.get('excluded_records', 0):,} | {vol.get('exclusion_rate_pct', 0.0):.2f}% |",
        f"| Processed (Retained) | {vol.get('processed_records', 0):,} | {100.0 - vol.get('exclusion_rate_pct', 0.0):.2f}% |",
        "",
        "### Quality Flag Audit Breakdown",
        "",
        "| Flag | Count | Meaning |",
        "|---|---|---|",
    ]
    for flag, count in flags.items():
        lines.append(f"| `{flag}` | {count:,} | Quality failure flag |")

    lines.extend(
        [
            "",
            "## 3. Interaction Graph and Sparsity",
            "",
            "| Metric | Value |",
            "|---|---|",
            f"| Unique Users | {entities.get('unique_users', 0):,} |",
            f"| Unique Items | {entities.get('unique_items', 0):,} |",
            f"| Processed Interactions | {entities.get('interactions', 0):,} |",
            f"| Bipartite Graph Sparsity | {entities.get('sparsity', 0.0):.6f} |",
            f"| Users with >= 5 reviews | {u_act.get('users_ge_5', 0):,} |",
            f"| Items with >= 5 reviews | {i_pop.get('items_ge_5', 0):,} |",
            f"| Median User Reviews | {u_act.get('median', 0.0):.1f} (max: {u_act.get('max', 0):,}) |",
            f"| Median Item Reviews | {i_pop.get('median', 0.0):.1f} (max: {i_pop.get('max', 0):,}) |",
            "",
            "## 4. Rating and Verification Distribution",
            "",
            f"- **Rating Mean (Std):** {ratings.get('mean', 0.0):.2f} (±{ratings.get('std', 0.0):.2f})",
            f"- **Verified Purchase Share:** {vp.get('verified_fraction', 0.0) * 100:.2f}%",
            f"- **Temporal Range:** `{time_r.get('min_timestamp_utc', 'N/A')}` to `{time_r.get('max_timestamp_utc', 'N/A')}`",
            "",
            "### Rating Value Frequencies",
            "",
            "| Rating | Count | Share |",
            "|---|---|---|",
        ]
    )

    total_proc = max(1, vol.get("processed_records", 1))
    for r_val, count in sorted(ratings.get("distribution", {}).items()):
        share = count / total_proc * 100
        lines.append(f"| {r_val} Stars | {count:,} | {share:.2f}% |")

    lines.append("")
    return "\n".join(lines)


def run_quality_report(
    interim_path: Path,
    processed_path: Path,
    manifest_path: Path | None,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate quality report and write JSON and Markdown outputs."""
    report = generate_quality_report(interim_path, processed_path, manifest_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "data_quality_report.json"
    md_path = output_dir / "data_quality_report.md"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(format_markdown_report(report), encoding="utf-8")

    logger.info("Quality report saved to %s and %s", json_path, md_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate data quality and integrity report.")
    parser.add_argument("--category", type=str, default=None, help="Dataset category")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg = load_config("data")
    paths = resolve_data_paths(cfg)

    cat = args.category or cfg.get("amazon_reviews", {}).get("category", "Musical_Instruments")
    interim_file = paths.interim / cat / "reviews_interim.parquet"
    proc_file = paths.processed / cat / "reviews.parquet"
    manifest_file = paths.raw / cat / "raw_manifest.json"

    run_quality_report(interim_file, proc_file, manifest_file, paths.reports)


if __name__ == "__main__":
    main()
