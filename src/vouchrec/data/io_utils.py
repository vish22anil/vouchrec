"""I/O and hashing utilities for raw, interim, and processed data."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def compute_file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute the SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_content_hash(raw_record: dict[str, Any]) -> str:
    """Compute deterministic BLAKE2b hash of canonicalized JSON representation."""
    canonical_repr = json.dumps(
        raw_record, sort_keys=True, ensure_ascii=True, separators=(",", ":")
    )
    return hashlib.blake2b(canonical_repr.encode("utf-8"), digest_size=16).hexdigest()


def stream_jsonl(
    path: Path, max_records: int | None = None
) -> Iterator[tuple[int, dict[str, Any]]]:
    """Stream raw JSON lines from a plain or gzip-compressed file.

    Yields (line_index, record) tuples where line_index is 0-based.
    """

    def _iter_lines(fh: Any) -> Iterator[tuple[int, dict[str, Any]]]:
        for idx, line in enumerate(fh):
            if max_records is not None and idx >= max_records:
                break
            line_str = line.strip()
            if not line_str:
                continue
            try:
                record = json.loads(line_str)
                yield idx, record
            except json.JSONDecodeError:
                continue

    if path.name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            yield from _iter_lines(fh)
    else:
        with path.open("r", encoding="utf-8") as fh:
            yield from _iter_lines(fh)


def save_parquet(table: pa.Table, output_path: Path, compression: str = "zstd") -> None:
    """Save an Arrow Table to a Parquet file, ensuring parent directory exists."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression=compression)


def load_parquet(path: Path) -> pa.Table:
    """Read a Parquet file as an Arrow Table."""
    return pq.read_table(path)
