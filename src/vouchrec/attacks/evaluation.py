"""Evaluation framework measuring attack impact on recommendations and trust detection."""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from vouchrec.attacks.simulator import AttackSimulationResult
from vouchrec.evaluation.metrics import evaluate_recommendations
from vouchrec.models.base import BaseRecommender
from vouchrec.trust.fusion import ProvisionalTrustFusion

logger = logging.getLogger(__name__)


def compute_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Area Under the ROC Curve using the Mann-Whitney U statistic.

    Handles ties via average rank assignment.
    """
    y_bool = np.asarray(y_true, dtype=bool)
    y_s = np.asarray(y_score, dtype=float)
    n_pos = int(np.sum(y_bool))
    n_neg = int(len(y_bool) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(y_s)
    sum_ranks_pos = float(np.sum(ranks[y_bool]))
    u_stat = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u_stat / (n_pos * n_neg))


@dataclass(frozen=True)
class AttackEvaluationReport:
    """Detailed quantitative results comparing clean vs attacked system state."""

    model_name: str
    target_item: str
    clean_metrics: dict[str, float]
    attacked_metrics: dict[str, float]
    target_impact: dict[str, float]
    detection_metrics: dict[str, float]
    runtime_seconds: float
    manifest: dict[str, Any]


def evaluate_target_item_impact(
    clean_recs: dict[str, list[str]],
    attacked_recs: dict[str, list[str]],
    target_item: str,
    top_k: int = 10,
) -> dict[str, float]:
    """Measure the exact change in target product visibility across test users."""
    eval_users = list(clean_recs.keys())
    if not eval_users:
        return {
            "clean_hitrate": 0.0,
            "attacked_hitrate": 0.0,
            "hitrate_delta": 0.0,
            "mean_rank_clean": float(top_k + 1),
            "mean_rank_attacked": float(top_k + 1),
            "rank_improvement": 0.0,
        }

    clean_hits = 0
    attacked_hits = 0
    clean_ranks: list[float] = []
    attacked_ranks: list[float] = []

    for u in eval_users:
        c_list = clean_recs.get(u, [])[:top_k]
        a_list = attacked_recs.get(u, [])[:top_k]

        # Hit rate
        c_hit = target_item in c_list
        a_hit = target_item in a_list
        if c_hit:
            clean_hits += 1
            clean_ranks.append(float(c_list.index(target_item) + 1))
        else:
            clean_ranks.append(float(top_k + 1))

        if a_hit:
            attacked_hits += 1
            attacked_ranks.append(float(a_list.index(target_item) + 1))
        else:
            attacked_ranks.append(float(top_k + 1))

    c_hr = clean_hits / len(eval_users)
    a_hr = attacked_hits / len(eval_users)
    m_rank_c = float(np.mean(clean_ranks))
    m_rank_a = float(np.mean(attacked_ranks))

    return {
        "target_hitrate_at_10_clean": round(c_hr, 4),
        "target_hitrate_at_10_attacked": round(a_hr, 4),
        "target_hitrate_delta": round(a_hr - c_hr, 4),
        "target_mean_rank_clean": round(m_rank_c, 2),
        "target_mean_rank_attacked": round(m_rank_a, 2),
        "target_rank_improvement": round(m_rank_c - m_rank_a, 2),  # positive means ranked closer to #1
    }


def evaluate_trust_detection(
    attacked_reviews_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> dict[str, float]:
    """Evaluate trust scoring detection on known synthetic ground-truth labels."""
    fusion = ProvisionalTrustFusion()
    trust_res = fusion.score_reviews(attacked_reviews_df)
    scores_df = trust_res.scores_df

    # Join on record_id
    merged = pd.merge(scores_df, labels_df, on="record_id", how="inner")
    if merged.empty:
        return {"detection_evaluated": 0.0}

    real_mask = merged["label_value"] == 0
    synth_mask = merged["label_value"] == 1

    real_scores = merged.loc[real_mask, "provisional_trust_score"]
    synth_scores = merged.loc[synth_mask, "provisional_trust_score"]

    m_real = float(real_scores.mean()) if not real_scores.empty else 0.0
    m_synth = float(synth_scores.mean()) if not synth_scores.empty else 0.0
    separation = m_real - m_synth  # positive means synthetic reviews are penalized more than real

    metrics = {
        "mean_trust_real_reviews": round(m_real, 4),
        "mean_trust_synthetic_reviews": round(m_synth, 4),
        "trust_separation_delta": round(separation, 4),
        "synthetic_reviews_evaluated": float(synth_mask.sum()),
        "real_reviews_evaluated": float(real_mask.sum()),
    }

    # If both classes are present, compute ROC-AUC and threshold metrics (lower trust = higher suspicion)
    if real_mask.sum() > 0 and synth_mask.sum() > 0:
        # Suspicion score = 1.0 - trust_score
        suspicion_scores = 1.0 - merged["provisional_trust_score"].values
        true_labels = merged["label_value"].values.astype(int)
        try:
            auc = compute_roc_auc(true_labels, suspicion_scores)
            metrics["trust_detection_roc_auc"] = round(auc, 4)
        except Exception:
            metrics["trust_detection_roc_auc"] = 0.50

        # Calibrated decision threshold at median / natural boundary
        pred_pos = suspicion_scores >= 0.30
        tp = float(np.sum((true_labels == 1) & pred_pos))
        fp = float(np.sum((true_labels == 0) & pred_pos))
        fn = float(np.sum((true_labels == 1) & (~pred_pos)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        metrics["trust_detection_precision"] = round(prec, 4)
        metrics["trust_detection_recall"] = round(rec, 4)
        metrics["trust_detection_f1"] = round(f1, 4)

    return metrics


class AttackExperimentRunner:
    """Executes controlled clean vs attacked experiments on recommendation baselines."""

    def __init__(
        self,
        clean_train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_item: str,
        cutoffs: tuple[int, ...] = (10, 20),
    ) -> None:
        self.clean_train_df = clean_train_df
        self.test_df = test_df
        self.target_item = target_item
        self.cutoffs = cutoffs

        # Build ground-truth mapping for test interactions
        self.gt: dict[str, set[str]] = defaultdict(set)
        for u, i in zip(test_df["user_id"], test_df["item_id"], strict=False):
            self.gt[u].add(i)

        # Evaluate on test users who appeared in clean training interactions
        clean_train_users = set(clean_train_df["user_id"])
        self.eval_users = [u for u in self.gt if u in clean_train_users]

    def run_comparison(
        self,
        model_factory: Any,
        model_name: str,
        sim_result: AttackSimulationResult,
    ) -> AttackEvaluationReport:
        """Train model on clean vs attacked data, compare recommendations and trust detection."""
        t0 = time.time()
        max_k = max(self.cutoffs)

        # 1. Train on clean data
        logger.info("Training %s on clean data (%d interactions)...", model_name, len(self.clean_train_df))
        clean_model: BaseRecommender = model_factory()
        clean_model.fit(self.clean_train_df)
        clean_recs = clean_model.recommend(self.eval_users, k=max_k, exclude_seen=True)
        clean_metrics = evaluate_recommendations(clean_recs, self.gt, cutoffs=self.cutoffs)

        # 2. Train on attacked data
        attacked_df = sim_result.attacked_train_df
        logger.info("Training %s on attacked data (%d interactions)...", model_name, len(attacked_df))
        attacked_model: BaseRecommender = model_factory()
        attacked_model.fit(attacked_df)
        attacked_recs = attacked_model.recommend(self.eval_users, k=max_k, exclude_seen=True)
        attacked_metrics = evaluate_recommendations(attacked_recs, self.gt, cutoffs=self.cutoffs)

        # 3. Target item recommendation impact
        target_impact = evaluate_target_item_impact(
            clean_recs, attacked_recs, self.target_item, top_k=10
        )

        # 4. Trust model detection performance
        detection_metrics = evaluate_trust_detection(
            sim_result.attacked_train_df, sim_result.ground_truth_labels_df
        )

        runtime = time.time() - t0
        logger.info(
            "%s attack experiment finished in %.2fs: HitRate@10 clean=%.4f -> attacked=%.4f (delta=%+.4f)",
            model_name,
            runtime,
            target_impact["target_hitrate_at_10_clean"],
            target_impact["target_hitrate_at_10_attacked"],
            target_impact["target_hitrate_delta"],
        )

        return AttackEvaluationReport(
            model_name=model_name,
            target_item=self.target_item,
            clean_metrics=clean_metrics,
            attacked_metrics=attacked_metrics,
            target_impact=target_impact,
            detection_metrics=detection_metrics,
            runtime_seconds=round(runtime, 2),
            manifest=sim_result.manifest,
        )
