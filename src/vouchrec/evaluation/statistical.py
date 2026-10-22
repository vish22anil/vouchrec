"""Statistical analysis, confidence intervals, hypothesis testing, and multiple comparison corrections."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    confidence_level: float
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "ci_lower": round(self.ci_lower, 4),
            "ci_upper": round(self.ci_upper, 4),
            "confidence_level": self.confidence_level,
            "n_samples": self.n_samples,
        }

    def format_str(self) -> str:
        return f"{self.mean:.4f} ± {self.std:.4f} [95% CI: {self.ci_lower:.4f}, {self.ci_upper:.4f}]"


@dataclass(frozen=True)
class PairedTestResult:
    condition_a: str
    condition_b: str
    metric_name: str
    n_pairs: int
    mean_a: float
    mean_b: float
    mean_diff: float
    std_diff: float
    t_statistic: float
    p_value: float
    cohens_d: float
    is_significant: bool
    alpha: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_a": self.condition_a,
            "condition_b": self.condition_b,
            "metric_name": self.metric_name,
            "n_pairs": self.n_pairs,
            "mean_a": round(self.mean_a, 4),
            "mean_b": round(self.mean_b, 4),
            "mean_diff": round(self.mean_diff, 4),
            "std_diff": round(self.std_diff, 4),
            "t_statistic": round(self.t_statistic, 4),
            "p_value": float(f"{self.p_value:.6g}"),
            "cohens_d": round(self.cohens_d, 4),
            "is_significant": self.is_significant,
            "alpha": self.alpha,
        }


def compute_confidence_interval(
    values: Sequence[float],
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Compute sample mean, sample standard deviation, and Student's t confidence interval."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return ConfidenceInterval(0.0, 0.0, 0.0, 0.0, confidence, 0)

    mean_val = float(np.mean(arr))
    if n == 1:
        return ConfidenceInterval(mean_val, 0.0, mean_val, mean_val, confidence, 1)

    std_val = float(np.std(arr, ddof=1))

    # Two-tailed critical value from Student's t distribution
    t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=n - 1))
    margin = t_crit * (std_val / math.sqrt(n))

    return ConfidenceInterval(
        mean=mean_val,
        std=std_val,
        ci_lower=mean_val - margin,
        ci_upper=mean_val + margin,
        confidence_level=confidence,
        n_samples=n,
    )


def compute_paired_t_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    condition_a: str = "A",
    condition_b: str = "B",
    metric_name: str = "metric",
    alpha: float = 0.05,
) -> PairedTestResult:
    """Compute paired Student's t-test and Cohen's d effect size for matched seed evaluations."""
    arr_a = np.asarray(sample_a, dtype=float)
    arr_b = np.asarray(sample_b, dtype=float)

    if len(arr_a) != len(arr_b):
        raise ValueError(f"Paired samples must have identical lengths: {len(arr_a)} vs {len(arr_b)}")

    n = len(arr_a)
    if n < 2:
        raise ValueError(f"Paired t-test requires at least 2 pairs, got {n}")

    diff = arr_a - arr_b
    mean_a = float(np.mean(arr_a))
    mean_b = float(np.mean(arr_b))
    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1))

    if std_diff < 1e-12:
        # Zero variance in difference across seeds
        if abs(mean_diff) < 1e-12:
            t_stat = 0.0
            p_val = 1.0
            cohens_d = 0.0
        else:
            t_stat = math.copysign(float("inf"), mean_diff)
            p_val = 0.0
            cohens_d = math.copysign(float("inf"), mean_diff)
    else:
        se = std_diff / math.sqrt(n)
        t_stat = mean_diff / se
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=n - 1)))
        cohens_d = mean_diff / std_diff

    is_sig = p_val < alpha

    return PairedTestResult(
        condition_a=condition_a,
        condition_b=condition_b,
        metric_name=metric_name,
        n_pairs=n,
        mean_a=mean_a,
        mean_b=mean_b,
        mean_diff=mean_diff,
        std_diff=std_diff,
        t_statistic=t_stat,
        p_value=p_val,
        cohens_d=cohens_d,
        is_significant=is_sig,
        alpha=alpha,
    )


def holm_bonferroni_correction(
    results: list[PairedTestResult],
    fwer_alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Apply Holm-Bonferroni step-down procedure to control Family-Wise Error Rate (FWER)."""
    if not results:
        return []

    # Sort tests by ascending unadjusted p-value
    indexed = sorted(enumerate(results), key=lambda x: x[1].p_value)
    m = len(results)

    adjusted_records: list[dict[str, Any]] = [{}] * m
    reject_rest = True

    for rank, (orig_idx, test_res) in enumerate(indexed):
        k = rank + 1
        # Adjusted threshold: alpha / (m - k + 1)
        thresh_k = fwer_alpha / (m - k + 1)
        p = test_res.p_value

        if reject_rest and (p <= thresh_k):
            is_sig = True
        else:
            is_sig = False
            reject_rest = False

        adj_p = min(1.0, p * (m - k + 1))

        rec = test_res.to_dict()
        rec["holm_rank"] = k
        rec["holm_threshold"] = round(thresh_k, 5)
        rec["adjusted_p_value"] = float(f"{adj_p:.6g}")
        rec["is_significant_holm"] = is_sig
        adjusted_records[orig_idx] = rec

    return adjusted_records
