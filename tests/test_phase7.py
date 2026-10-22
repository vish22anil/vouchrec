"""Unit and integration tests for Phase 7 statistical evaluation, seeding, and anti-leakage guards."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from scripts.run_phase7_experiments import (
    collect_system_manifest,
    fit_and_recommend,
    set_seed,
)

from vouchrec.data.schema import LABEL_COLUMNS
from vouchrec.evaluation.statistical import (
    compute_confidence_interval,
    compute_paired_t_test,
    holm_bonferroni_correction,
)


def test_seed_determinism() -> None:
    """Verify that set_seed ensures bit-exact reproducible pseudorandom sequences across libraries."""
    # Test identical seeds
    set_seed(20261021)
    py_rand_1 = [float(np.random.random()) for _ in range(5)]
    np_rand_1 = np.random.normal(size=5).tolist()
    torch_rand_1 = torch.randn(5).tolist()

    set_seed(20261021)
    py_rand_2 = [float(np.random.random()) for _ in range(5)]
    np_rand_2 = np.random.normal(size=5).tolist()
    torch_rand_2 = torch.randn(5).tolist()

    assert py_rand_1 == py_rand_2
    assert np_rand_1 == np_rand_2
    assert torch_rand_1 == torch_rand_2

    # Test divergent seeds
    set_seed(20261022)
    py_rand_3 = [float(np.random.random()) for _ in range(5)]
    assert py_rand_1 != py_rand_3


def test_confidence_interval_math() -> None:
    """Verify Student's t confidence interval calculation against analytical properties."""
    # 1. Constant values: std = 0, CI width = 0
    ci_const = compute_confidence_interval([0.75, 0.75, 0.75, 0.75, 0.75])
    assert ci_const.mean == 0.75
    assert ci_const.std == 0.0
    assert ci_const.ci_lower == 0.75
    assert ci_const.ci_upper == 0.75

    # 2. Known sample
    sample = [0.80, 0.82, 0.81, 0.83, 0.79]
    ci = compute_confidence_interval(sample, confidence=0.95)
    assert abs(ci.mean - 0.81) < 1e-4
    assert ci.std > 0.0
    assert ci.ci_lower < ci.mean < ci.ci_upper
    assert ci.n_samples == 5

    # 3. Edge cases
    ci_empty = compute_confidence_interval([])
    assert ci_empty.mean == 0.0
    assert ci_empty.n_samples == 0

    ci_single = compute_confidence_interval([0.90])
    assert ci_single.mean == 0.90
    assert ci_single.ci_lower == 0.90


def test_paired_t_test_and_cohens_d() -> None:
    """Verify paired t-test properties and effect size calculation."""
    # 1. Identical distributions -> t = 0, p = 1.0, d = 0.0
    sample_a = [0.70, 0.72, 0.71, 0.73, 0.69]
    res_ident = compute_paired_t_test(sample_a, sample_a, "ModelA", "ModelA")
    assert res_ident.mean_diff == 0.0
    assert res_ident.t_statistic == 0.0
    assert res_ident.p_value == 1.0
    assert res_ident.cohens_d == 0.0
    assert not res_ident.is_significant

    # 2. Significant shift
    sample_b = [0.60, 0.61, 0.59, 0.62, 0.58]
    res_sig = compute_paired_t_test(sample_a, sample_b, "ModelA", "ModelB", alpha=0.05)
    assert res_sig.mean_diff > 0.0
    assert res_sig.t_statistic > 0.0
    assert res_sig.p_value < 0.001
    assert res_sig.cohens_d > 2.0  # Very large effect size
    assert res_sig.is_significant

    # 3. Mismatched length error
    with pytest.raises(ValueError, match="identical lengths"):
        compute_paired_t_test([1.0, 2.0], [1.0])


def test_holm_bonferroni_correction() -> None:
    """Verify that Holm-Bonferroni strictly controls Family-Wise Error Rate."""
    sample_a = [0.80, 0.81, 0.82, 0.80, 0.81]
    sample_b = [0.60, 0.61, 0.59, 0.62, 0.58]
    sample_c = [0.79, 0.80, 0.81, 0.80, 0.81]

    t1 = compute_paired_t_test(sample_a, sample_b, "A", "B")  # Highly significant
    t2 = compute_paired_t_test(sample_a, sample_c, "A", "C")  # Small/borderline difference
    t3 = compute_paired_t_test(sample_a, sample_a, "A", "A")  # Not significant (p = 1.0)

    adjusted = holm_bonferroni_correction([t1, t2, t3], fwer_alpha=0.05)
    assert len(adjusted) == 3

    # Check rank order and adjusted p-values
    for rec in adjusted:
        assert "holm_rank" in rec
        assert "adjusted_p_value" in rec
        assert "is_significant_holm" in rec
        assert rec["adjusted_p_value"] >= rec["p_value"]


def test_system_manifest_collection(tmp_path: Path) -> None:
    """Verify system manifest collection captures platform, packages, and file hashes."""
    # Create mock split files
    (tmp_path / "train.parquet").write_bytes(b"mock_train_data")
    (tmp_path / "val.parquet").write_bytes(b"mock_val_data")

    manifest = collect_system_manifest(tmp_path)
    assert "timestamp_utc" in manifest
    assert "platform" in manifest
    assert "packages" in manifest
    assert "torch" in manifest["packages"]
    assert "numpy" in manifest["packages"]

    fps = manifest["dataset_split_fingerprints"]
    assert "train.parquet" in fps
    assert fps["test.parquet"] == "MISSING"


def test_anti_leakage_rejection_in_fit_and_recommend() -> None:
    """Verify that passing training data containing attack/evaluation labels is rejected immediately."""
    clean_df = pd.DataFrame({
        "user_id": ["U1", "U2", "U1"],
        "item_id": ["I1", "I2", "I2"],
        "rating": [5.0, 4.0, 5.0],
    })

    # Legitimate data passes schema assertion
    recs = fit_and_recommend(
        model_name="Popularity",
        train_df=clean_df,
        eval_users=["U1", "U2"],
        seed=20261021,
        epochs=1,
    )
    assert "U1" in recs

    # Contaminated data with evaluation labels MUST raise ValueError
    for label_col in LABEL_COLUMNS:
        contaminated_df = clean_df.copy()
        contaminated_df[label_col] = 1
        with pytest.raises(ValueError, match="Label/attack columns must not be used as features"):
            fit_and_recommend(
                model_name="Popularity",
                train_df=contaminated_df,
                eval_users=["U1", "U2"],
                seed=20261021,
                epochs=1,
            )
