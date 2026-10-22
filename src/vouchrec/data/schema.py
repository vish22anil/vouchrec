"""Canonical review schema, label schema, and validation.

Review records and manipulation labels are stored in separate tables joined by
``record_id``. Label and attack-metadata columns must never appear in model features;
``assert_no_label_columns`` enforces that at feature-construction time.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pyarrow as pa


class SchemaError(ValueError):
    """Raised when a table does not match the canonical schema."""


@dataclass(frozen=True)
class Field:
    name: str
    type: pa.DataType
    description: str


TIMESTAMP_TYPE = pa.timestamp("ms", tz="UTC")

REVIEW_FIELDS: tuple[Field, ...] = (
    Field("record_id", pa.string(), "Surrogate key '<source>/<split or category>/<source_row>'."),
    Field("source_dataset", pa.string(), "Dataset identifier, e.g. amazon_reviews_2023."),
    Field("source_category", pa.string(), "Source category or domain label."),
    Field("source_split", pa.string(), "Split assigned by the source; null if none."),
    Field("source_row", pa.int64(), "0-based row/line position in the source file."),
    Field("review_id", pa.string(), "Review identifier from the source; null if absent."),
    Field("user_id", pa.string(), "Reviewer identifier from the source."),
    Field("item_id", pa.string(), "Item identifier (Amazon: parent_asin)."),
    Field("variant_asin", pa.string(), "Amazon variant ASIN; null for other sources."),
    Field("rating", pa.float64(), "Star rating as published."),
    Field("review_title", pa.string(), "Review title/headline."),
    Field("review_text", pa.string(), "Review body, unmodified."),
    Field("timestamp", TIMESTAMP_TYPE, "Review time in UTC; null if missing or invalid."),
    Field("timestamp_ms_raw", pa.int64(), "Source timestamp in Unix milliseconds."),
    Field("verified_purchase", pa.bool_(), "Source verified-purchase flag; never imputed."),
    Field("helpful_vote", pa.int64(), "Helpful-vote count at collection time."),
    Field("n_images", pa.int64(), "Number of review images attached."),
    Field("content_hash", pa.string(), "BLAKE2b hash of all raw source fields."),
)

QUALITY_FLAG_FIELDS: tuple[Field, ...] = (
    Field("is_missing_user_id", pa.bool_(), "user_id null or blank."),
    Field("is_missing_item_id", pa.bool_(), "item_id null or blank."),
    Field("is_invalid_rating", pa.bool_(), "rating missing or not in the configured set."),
    Field("is_invalid_timestamp", pa.bool_(), "timestamp missing, unparsable, or out of range."),
    Field("is_empty_text", pa.bool_(), "review_text null or whitespace only."),
    Field("is_exact_duplicate", pa.bool_(), "Repeat of an earlier row with identical raw fields."),
    Field("is_key_duplicate", pa.bool_(), "Repeat of an earlier (user, item, timestamp) key."),
)

LABEL_FIELDS: tuple[Field, ...] = (
    Field("record_id", pa.string(), "Joins to the review table."),
    Field("label_name", pa.string(), "What the label asserts, e.g. is_machine_generated."),
    Field("label_value", pa.int8(), "1 = positive, 0 = negative, null = unknown."),
    Field("label_raw", pa.string(), "Label exactly as published by the source."),
    Field("label_origin", pa.string(), "external_annotation | generation_record | simulation"),
    Field("label_provenance", pa.string(), "Repository and revision the label came from."),
)

CANONICAL_COLUMNS: tuple[str, ...] = tuple(f.name for f in REVIEW_FIELDS)
FLAG_COLUMNS: tuple[str, ...] = tuple(f.name for f in QUALITY_FLAG_FIELDS)

# Columns that carry ground truth or attack metadata. Never valid model inputs.
LABEL_COLUMNS: frozenset[str] = frozenset(
    {
        "label_name",
        "label_value",
        "label_raw",
        "label_origin",
        "label_provenance",
        "is_attack",
        "attack_id",
        "attack_type",
        "attack_target_item",
        "generator_model",
    }
)


def review_schema(with_flags: bool = True) -> pa.Schema:
    fields = REVIEW_FIELDS + (QUALITY_FLAG_FIELDS if with_flags else ())
    return pa.schema([pa.field(f.name, f.type) for f in fields])


def label_schema() -> pa.Schema:
    return pa.schema([pa.field(f.name, f.type) for f in LABEL_FIELDS])


def schema_problems(actual: pa.Schema, expected: pa.Schema) -> list[str]:
    """Return human-readable differences between two Arrow schemas (empty list = match)."""
    problems: list[str] = []
    actual_names = actual.names
    for field in expected:
        if field.name not in actual_names:
            problems.append(f"missing column '{field.name}'")
            continue
        actual_type = actual.field(field.name).type
        if not actual_type.equals(field.type):
            problems.append(f"column '{field.name}' has type {actual_type}, expected {field.type}")
    extra = [n for n in actual_names if n not in expected.names]
    if extra:
        problems.append(f"unexpected columns: {extra}")
    return problems


def validate_schema(actual: pa.Schema, expected: pa.Schema) -> None:
    problems = schema_problems(actual, expected)
    if problems:
        raise SchemaError("; ".join(problems))


def assert_no_label_columns(columns: Iterable[str] | Any) -> None:
    """Fail if any ground-truth or attack-metadata column is present in a feature set."""
    if hasattr(columns, "column_names"):
        cols = list(columns.column_names)
    elif hasattr(columns, "dtypes") and hasattr(columns, "columns"):
        cols = list(columns.columns)
    elif isinstance(columns, pa.Schema):
        cols = list(columns.names)
    else:
        cols = list(columns)
    leaked = sorted(set(cols) & LABEL_COLUMNS)
    if leaked:
        raise SchemaError(f"Label/attack columns must not be used as features: {leaked}")
