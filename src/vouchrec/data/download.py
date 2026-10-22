"""Download data from official sources with integrity verification and provenance tracking."""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import shutil
import urllib.request
from pathlib import Path
from typing import Any

from vouchrec.config import DataPaths, load_config, resolve_data_paths
from vouchrec.data.io_utils import compute_file_sha256
from vouchrec.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def check_storage(target_dir: Path, required_bytes: int, safety_factor: float = 2.0) -> None:
    """Verify that the target drive has sufficient free storage."""
    target_dir.mkdir(parents=True, exist_ok=True)
    total, used, free = shutil.disk_usage(target_dir)
    needed = int(required_bytes * safety_factor)
    if free < needed:
        raise OSError(
            f"Insufficient storage in {target_dir}. Free: {free / (1024**3):.2f} GB, "
            f"required (with safety factor {safety_factor}): {needed / (1024**3):.2f} GB."
        )
    logger.info(
        "Storage check passed: %.2f GB free on drive for %s (needed ~%.2f GB)",
        free / (1024**3),
        target_dir,
        needed / (1024**3),
    )


def download_file(
    url: str,
    output_path: Path,
    expected_bytes: int | None = None,
    chunk_size: int = 1024 * 1024,
) -> Path:
    """Download a file with resume support and streaming chunks."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_bytes = output_path.stat().st_size if output_path.exists() else 0
    if expected_bytes and existing_bytes == expected_bytes:
        logger.info("File already fully downloaded: %s (%d bytes)", output_path, existing_bytes)
        return output_path

    headers = {"User-Agent": "VouchRec-Research/0.1"}
    mode = "wb"
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        mode = "ab"
        logger.info("Resuming download of %s from byte %d", output_path.name, existing_bytes)
    else:
        logger.info("Starting fresh download of %s to %s", url, output_path)

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, output_path.open(mode) as out_f:
            status = resp.status
            # 200 = fresh download, 206 = partial content
            if status not in (200, 206):
                raise RuntimeError(f"Unexpected HTTP status {status} when downloading {url}")

            downloaded = existing_bytes
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                out_f.write(chunk)
                downloaded += len(chunk)
                if expected_bytes and downloaded % (10 * 1024 * 1024) < chunk_size:
                    logger.info(
                        "Downloaded %d / %d bytes (%.1f%%)",
                        downloaded,
                        expected_bytes,
                        (downloaded / expected_bytes) * 100,
                    )
    except Exception as exc:
        logger.error("Download failed or interrupted: %s", exc)
        raise

    logger.info("Download completed: %s (%d bytes)", output_path, output_path.stat().st_size)
    return output_path


def download_dataset(
    data_config: dict[str, Any],
    paths: DataPaths,
    category: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Execute download of configured Amazon Reviews category with provenance logging."""
    ar_cfg = data_config.get("amazon_reviews", {})
    target_category = category or ar_cfg.get("category", "Musical_Instruments")
    files_cfg = ar_cfg.get("files", {}).get("reviews", {})
    url = files_cfg.get("url")
    filename = files_cfg.get("filename", f"{target_category}.jsonl.gz")
    expected_bytes = files_cfg.get("expected_bytes")

    if not url:
        raise ValueError(f"No download URL configured for category '{target_category}'")

    raw_dir = paths.raw / target_category
    raw_file = raw_dir / filename
    manifest_path = raw_dir / "raw_manifest.json"

    if dry_run:
        logger.info(
            "[DRY RUN] Would download %s to %s (expected size: %s bytes)",
            url,
            raw_file,
            expected_bytes,
        )
        return {"dry_run": True, "url": url, "raw_file": str(raw_file)}

    check_storage(raw_dir, expected_bytes or 500_000_000)
    download_file(url, raw_file, expected_bytes=expected_bytes)

    logger.info("Computing SHA-256 for downloaded raw file...")
    actual_sha256 = compute_file_sha256(raw_file)
    actual_size = raw_file.stat().st_size

    manifest = {
        "source_dataset": ar_cfg.get("source_dataset", "amazon_reviews_2023"),
        "category": target_category,
        "filename": filename,
        "source_url": url,
        "download_timestamp_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "bytes": actual_size,
        "sha256": actual_sha256,
        "homepage": ar_cfg.get("homepage"),
        "hf_repository": ar_cfg.get("hf_repository"),
        "hf_revision": ar_cfg.get("hf_revision"),
        "citation": ar_cfg.get("citation"),
        "license_note": ar_cfg.get("license_note"),
    }

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    logger.info("Saved provenance manifest to %s", manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Download raw Amazon Reviews 2023 dataset.")
    parser.add_argument(
        "--category", type=str, default=None, help="Category name (defaults to config)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)
    cfg = load_config("data")
    paths = resolve_data_paths(cfg)
    download_dataset(cfg, paths, category=args.category, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
