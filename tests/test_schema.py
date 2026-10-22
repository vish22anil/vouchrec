"""Unit tests for schema definitions, validation, and anti-leakage checks."""

import pyarrow as pa
import pytest

from vouchrec.data.schema import (
    CANONICAL_COLUMNS,
    FLAG_COLUMNS,
    LABEL_COLUMNS,
    SchemaError,
    assert_no_label_columns,
    review_schema,
    validate_schema,
)


def test_review_schema_structure() -> None:
    schema_with_flags = review_schema(with_flags=True)
    schema_no_flags = review_schema(with_flags=False)

    for col in CANONICAL_COLUMNS:
        assert col in schema_with_flags.names
        assert col in schema_no_flags.names

    for flag in FLAG_COLUMNS:
        assert flag in schema_with_flags.names
        assert flag not in schema_no_flags.names


def test_validate_schema_mismatch() -> None:
    expected = review_schema(with_flags=False)
    wrong_table = pa.Table.from_pydict({"record_id": ["r1"], "random_col": [123]})

    with pytest.raises(SchemaError) as exc_info:
        validate_schema(wrong_table.schema, expected)

    assert "missing column" in str(exc_info.value)


def test_assert_no_label_columns() -> None:
    # Clean feature columns should pass
    clean_cols = ["record_id", "user_id", "item_id", "rating"]
    assert_no_label_columns(clean_cols)

    # Any column in LABEL_COLUMNS must fail
    for forbidden in LABEL_COLUMNS:
        with pytest.raises(SchemaError):
            assert_no_label_columns(["user_id", forbidden])
