"""Trust-Gated LightGCN recommender with configurable edge weighting and normalization.

Integrates Phase 3 provisional trust scores into LightGCN graph convolutions
to dampen embedding propagation along untrusted / suspicious interaction edges.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim

from vouchrec.data.schema import assert_no_label_columns
from vouchrec.evaluation.metrics import evaluate_recommendations
from vouchrec.models.base import BaseRecommender
from vouchrec.trust.fusion import ProvisionalTrustFusion

logger = logging.getLogger(__name__)


def compute_edge_weights(
    trust_scores: np.ndarray,
    mode: str = "soft",
    soft_function: str = "linear",
    power_p: float = 1.0,
    sigmoid_k: float = 10.0,
    sigmoid_center: float = 0.5,
    hard_threshold: float = 0.5,
    min_edge_weight: float = 1e-4,
    default_trust: float = 0.5,
) -> np.ndarray:
    """Compute interaction edge weights from provisional trust scores.

    Parameters
    ----------
    trust_scores:
        Raw or precomputed trust scores for each interaction edge.
    mode:
        Weighting mode: 'soft', 'hard', or 'uniform'.
    soft_function:
        Soft weighting formulation: 'linear', 'power', or 'sigmoid'.
    power_p:
        Exponent for power weighting: w = tau^p.
    sigmoid_k:
        Slope for sigmoid weighting.
    sigmoid_center:
        Inflection point for sigmoid weighting.
    hard_threshold:
        Threshold for hard filtering: w = 1.0 if tau >= threshold else 0.0.
    min_edge_weight:
        Floor for soft weights to maintain finite graph connectivity.
    default_trust:
        Imputation value for missing, NaN, or non-finite trust scores.

    Returns
    -------
    np.ndarray:
        Finite, non-negative edge weights aligned with input edges.
    """
    scores = np.asarray(trust_scores, dtype=np.float32).copy()

    # 1. Missing / invalid score imputation and clipping
    invalid_mask = np.isnan(scores) | np.isinf(scores)
    scores[invalid_mask] = default_trust
    scores = np.clip(scores, 0.0, 1.0)

    # 2. Compute weights according to mode
    if mode == "uniform":
        return np.ones_like(scores, dtype=np.float32)

    if mode == "soft":
        if soft_function == "linear":
            w = scores
        elif soft_function == "power":
            w = np.power(scores, power_p)
        elif soft_function == "sigmoid":
            w = 1.0 / (1.0 + np.exp(-sigmoid_k * (scores - sigmoid_center)))
        else:
            raise ValueError(f"Unknown soft_function: {soft_function}")
        return np.clip(w, min_edge_weight, 1.0).astype(np.float32)

    if mode == "hard":
        w = np.where(scores >= hard_threshold, 1.0, 0.0).astype(np.float32)
        return w

    raise ValueError(f"Unknown weighting mode: {mode}")


class _TrustGatedLightGCNModule(nn.Module):
    """PyTorch graph neural network module for Trust-Gated LightGCN."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        embedding_dim: int,
        num_layers: int,
        norm_adj: torch.Tensor,
    ) -> None:
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.norm_adj = norm_adj

        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)

        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)

    def forward_propagation(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-layer message passing over trust-weighted normalized adjacency."""
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs_list = [all_embeddings]

        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            embs_list.append(all_embeddings)

        final_embeddings = torch.stack(embs_list, dim=0).mean(dim=0)
        final_users, final_items = torch.split(final_embeddings, [self.num_users, self.num_items])
        return final_users, final_items

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        l2_reg: float,
        pair_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute BPR loss on propagated embeddings + L2 regularization."""
        final_users, final_items = self.forward_propagation()

        u_emb = final_users[users]
        pos_emb = final_items[pos_items]
        neg_emb = final_items[neg_items]

        pos_scores = (u_emb * pos_emb).sum(dim=-1)
        neg_scores = (u_emb * neg_emb).sum(dim=-1)
        diff = pos_scores - neg_scores

        log_sig = torch.log(torch.sigmoid(diff) + 1e-10)

        if pair_weights is not None:
            norm_w = pair_weights / (pair_weights.sum() + 1e-10)
            bpr = -torch.sum(norm_w * log_sig)
        else:
            bpr = -torch.mean(log_sig)

        base_u = self.user_embedding(users)
        base_pos = self.item_embedding(pos_items)
        base_neg = self.item_embedding(neg_items)
        reg = (
            l2_reg
            * (base_u.pow(2).sum() + base_pos.pow(2).sum() + base_neg.pow(2).sum())
            / len(users)
        )

        return bpr + reg


class TrustGatedLightGCN(BaseRecommender):
    """Trust-Gated LightGCN recommender with configurable edge weighting and normalization.

    Parameters
    ----------
    embedding_dim:
        Dimension of user and item latent representations.
    num_layers:
        Number of graph convolutional message passing layers.
    learning_rate:
        Adam optimizer learning rate.
    l2_reg:
        Weight decay penalty applied to initial embeddings.
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size for BPR pair updates.
    seed:
        Random seed for deterministic initialization and sampling.
    weighting_mode:
        Edge weighting strategy: 'soft', 'hard', or 'uniform'.
    soft_function:
        Soft weighting formula: 'linear', 'power', or 'sigmoid'.
    power_p:
        Exponent if soft_function == 'power'.
    sigmoid_k:
        Slope if soft_function == 'sigmoid'.
    sigmoid_center:
        Inflection threshold if soft_function == 'sigmoid'.
    hard_threshold:
        Cutoff threshold for hard filtering.
    min_edge_weight:
        Floor for soft weights.
    default_trust:
        Neutral trust score (0.50) for unrated or missing interactions.
    bpr_pair_weighting:
        If True, scales BPR pair loss by interaction trust weight in addition to graph gating.
    trust_fusion:
        Optional preconfigured ProvisionalTrustFusion instance.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        num_layers: int = 3,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 1024,
        seed: int = 20261021,
        weighting_mode: str = "soft",
        soft_function: str = "linear",
        power_p: float = 1.0,
        sigmoid_k: float = 10.0,
        sigmoid_center: float = 0.5,
        hard_threshold: float = 0.5,
        min_edge_weight: float = 1e-4,
        default_trust: float = 0.5,
        bpr_pair_weighting: bool = False,
        trust_fusion: ProvisionalTrustFusion | None = None,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        self.weighting_mode = weighting_mode
        self.soft_function = soft_function
        self.power_p = power_p
        self.sigmoid_k = sigmoid_k
        self.sigmoid_center = sigmoid_center
        self.hard_threshold = hard_threshold
        self.min_edge_weight = min_edge_weight
        self.default_trust = default_trust
        self.bpr_pair_weighting = bpr_pair_weighting
        self.trust_fusion = trust_fusion or ProvisionalTrustFusion()

        self.user2idx: dict[str, int] = {}
        self.idx2user: list[str] = []
        self.item2idx: dict[str, int] = {}
        self.idx2item: list[str] = []
        self.user_seen_items: dict[int, set[int]] = defaultdict(set)
        self.model_: _TrustGatedLightGCNModule | None = None

        self.final_user_embs_: torch.Tensor | None = None
        self.final_item_embs_: torch.Tensor | None = None

        # Diagnostic metadata
        self.edge_stats_: dict[str, Any] = {}
        self.edge_weights_: np.ndarray | None = None

    def _join_trust_scores(
        self,
        interactions: pd.DataFrame,
        trust_scores_df: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Associate trust scores with interactions using stable identifiers."""
        # 1. Anti-leakage guard: reject any ground-truth or attack metadata columns
        assert_no_label_columns(interactions)
        if trust_scores_df is not None:
            assert_no_label_columns(trust_scores_df)

        # Case A: External trust_scores_df provided
        if trust_scores_df is not None:
            if "provisional_trust_score" not in trust_scores_df.columns:
                raise ValueError("trust_scores_df must contain 'provisional_trust_score' column.")

            # Stable join key precedence: record_id -> (user_id, item_id, timestamp) -> (user_id, item_id)
            if "record_id" in interactions.columns and "record_id" in trust_scores_df.columns:
                join_keys = ["record_id"]
            elif (
                "user_id" in interactions.columns
                and "item_id" in interactions.columns
                and "timestamp" in interactions.columns
                and "timestamp" in trust_scores_df.columns
            ):
                join_keys = ["user_id", "item_id", "timestamp"]
            elif "user_id" in interactions.columns and "item_id" in trust_scores_df.columns:
                join_keys = ["user_id", "item_id"]
            else:
                raise ValueError(
                    "Cannot join trust_scores_df to interactions: no matching stable identifier."
                )

            # Drop potential duplicate columns in right table except join_keys and score
            keep_cols = list(join_keys) + ["provisional_trust_score"]
            sub_trust = trust_scores_df[keep_cols].drop_duplicates(subset=join_keys)

            merged = pd.merge(
                interactions.reset_index(drop=True),
                sub_trust,
                on=join_keys,
                how="left",
            )
            raw_scores = merged["provisional_trust_score"].to_numpy(dtype=np.float32)

        # Case B: provisional_trust_score already present in interactions
        elif "provisional_trust_score" in interactions.columns:
            raw_scores = interactions["provisional_trust_score"].to_numpy(dtype=np.float32)

        # Case C: Compute trust scores directly from training reviews
        elif "review_text" in interactions.columns:
            logger.info("Computing provisional trust scores on training interactions...")
            trust_res = self.trust_fusion.score_reviews(interactions)
            raw_scores = trust_res.scores_df["provisional_trust_score"].to_numpy(dtype=np.float32)

        # Case D: Fallback to default neutral trust
        else:
            logger.warning(
                "No trust scores or review texts available; assigning default prior %.2f.",
                self.default_trust,
            )
            raw_scores = np.full(len(interactions), self.default_trust, dtype=np.float32)

        return raw_scores

    def _build_weighted_normalized_adj(
        self,
        num_users: int,
        num_items: int,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
        weights: np.ndarray,
    ) -> torch.Tensor:
        """Construct symmetric normalized bipartite adjacency matrix D^(-1/2) A D^(-1/2).

        Normalization Explanation:
        --------------------------
        Vanilla LightGCN assumes unweighted edges A_ij in {0, 1}, where row sum D_ii is
        the unweighted interaction degree.

        In TrustGatedLightGCN, each edge is weighted by w_ij in [0, 1]. The row sum D_ii
        is the weighted node degree:
            D_ii = sum_j A_ij = sum_{j in N(i)} w_ij

        Symmetric normalization scales message propagation by:
            A_tilde_ij = w_ij / sqrt(D_ii * D_jj)

        This differs from vanilla LightGCN in three vital ways:
        1. Gradient & Message Attenuation: Untrusted edges (w_ij ~ 0) attenuate embedding
           propagation, preventing malicious profiles from distorting item representations.
        2. Spectral Radius Preservation: Bounded symmetric normalization guarantees the
           eigenvalues of A_tilde remain in [-1, 1], preventing explosion or vanishing of
           embeddings across multiple graph convolution layers.
        3. Safe Isolated Node Handling: Nodes with zero or pruned edges are assigned degree 0,
           and their inverse square root is cleanly set to 0.0, avoiding division by zero.
        """
        num_nodes = num_users + num_items

        # Build bidirectional bipartite edges
        row = np.concatenate([user_indices, item_indices + num_users])
        col = np.concatenate([item_indices + num_users, user_indices])
        data = np.concatenate([weights, weights]).astype(np.float32)

        # Construct weighted adjacency
        adj_matrix = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))

        # Compute weighted degree D_ii
        rowsum = np.array(adj_matrix.sum(axis=1)).flatten()
        d_inv_sqrt = np.zeros_like(rowsum, dtype=np.float32)
        nonzero_mask = rowsum > 0
        d_inv_sqrt[nonzero_mask] = np.power(rowsum[nonzero_mask], -0.5)

        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        norm_adj = d_mat_inv_sqrt.dot(adj_matrix).dot(d_mat_inv_sqrt).tocoo()

        indices = torch.tensor(np.vstack((norm_adj.row, norm_adj.col)), dtype=torch.long)
        values = torch.tensor(norm_adj.data, dtype=torch.float32)
        shape = torch.Size(norm_adj.shape)

        return torch.sparse_coo_tensor(indices, values, shape, check_invariants=False).coalesce()

    def fit(
        self,
        interactions: pd.DataFrame,
        trust_scores_df: pd.DataFrame | None = None,
    ) -> TrustGatedLightGCN:
        """Fit Trust-Gated LightGCN on training graph with trust-weighted message passing."""
        if interactions.empty:
            raise ValueError("Interactions dataframe is empty.")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # 1. Join trust scores using stable identifiers
        raw_trust = self._join_trust_scores(interactions, trust_scores_df)

        # 2. Compute interaction edge weights
        weights = compute_edge_weights(
            raw_trust,
            mode=self.weighting_mode,
            soft_function=self.soft_function,
            power_p=self.power_p,
            sigmoid_k=self.sigmoid_k,
            sigmoid_center=self.sigmoid_center,
            hard_threshold=self.hard_threshold,
            min_edge_weight=self.min_edge_weight,
            default_trust=self.default_trust,
        )
        self.edge_weights_ = weights

        # 3. Filter interactions for hard mode
        if self.weighting_mode == "hard":
            active_mask = weights > 0.0
            pruned_count = int(np.sum(~active_mask))
            pruned_frac = float(pruned_count / len(interactions))
            logger.info(
                "Hard filtering (threshold=%.2f): pruned %d/%d interactions (%.2f%%)",
                self.hard_threshold,
                pruned_count,
                len(interactions),
                pruned_frac * 100.0,
            )
            inter_df = interactions.loc[active_mask].reset_index(drop=True)
            active_weights = weights[active_mask]
        else:
            pruned_count = 0
            pruned_frac = 0.0
            inter_df = interactions.reset_index(drop=True)
            active_weights = weights

        if inter_df.empty:
            raise ValueError(
                f"All interactions were pruned under hard threshold {self.hard_threshold}."
            )

        # 4. Map users and items to contiguous indices
        unique_users = sorted(interactions["user_id"].unique())
        unique_items = sorted(interactions["item_id"].unique())

        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = unique_users
        self.item2idx = {item: i for i, item in enumerate(unique_items)}
        self.idx2item = unique_items

        num_users = len(self.user2idx)
        num_items = len(self.item2idx)

        user_indices = inter_df["user_id"].map(self.user2idx).values
        item_indices = inter_df["item_id"].map(self.item2idx).values

        self.user_seen_items = defaultdict(set)
        pairs: list[tuple[int, int]] = []
        for u_idx, i_idx in zip(user_indices, item_indices, strict=False):
            self.user_seen_items[u_idx].add(i_idx)
            pairs.append((u_idx, i_idx))

        # 5. Build weighted normalized bipartite adjacency
        norm_adj = self._build_weighted_normalized_adj(
            num_users=num_users,
            num_items=num_items,
            user_indices=user_indices,
            item_indices=item_indices,
            weights=active_weights,
        )

        # 6. Record edge diagnostics
        self.edge_stats_ = {
            "weighting_mode": self.weighting_mode,
            "soft_function": self.soft_function,
            "hard_threshold": self.hard_threshold,
            "n_total_interactions": len(interactions),
            "n_active_interactions": len(inter_df),
            "n_pruned_interactions": pruned_count,
            "pruned_fraction": pruned_frac,
            "mean_weight": float(np.mean(active_weights)),
            "std_weight": float(np.std(active_weights)),
            "min_weight": float(np.min(active_weights)),
            "max_weight": float(np.max(active_weights)),
            "median_weight": float(np.median(active_weights)),
            "q10_weight": float(np.percentile(active_weights, 10)),
            "q90_weight": float(np.percentile(active_weights, 90)),
        }

        # 7. Initialize neural graph module
        self.model_ = _TrustGatedLightGCNModule(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            norm_adj=norm_adj,
        )

        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        pairs_arr = np.array(pairs, dtype=np.int64)
        weights_arr = active_weights if self.bpr_pair_weighting else None
        num_pairs = len(pairs)

        logger.info(
            "Training TrustGatedLightGCN (%d users, %d items, %d edges, mode=%s, epochs=%d)",
            num_users,
            num_items,
            num_pairs,
            self.weighting_mode,
            self.epochs,
        )

        # 8. Training loop
        for epoch in range(1, self.epochs + 1):
            self.model_.train()
            indices = np.random.permutation(num_pairs)
            epoch_loss = 0.0
            num_batches = 0

            for start_idx in range(0, num_pairs, self.batch_size):
                batch_indices = indices[start_idx : start_idx + self.batch_size]
                b_users = pairs_arr[batch_indices, 0]
                b_pos_items = pairs_arr[batch_indices, 1]

                # Uniform negative sampling
                b_neg_items = np.empty_like(b_pos_items)
                for i, u in enumerate(b_users):
                    seen = self.user_seen_items[u]
                    while True:
                        neg = np.random.randint(0, num_items)
                        if neg not in seen:
                            b_neg_items[i] = neg
                            break

                u_t = torch.as_tensor(b_users, dtype=torch.long)
                pos_t = torch.as_tensor(b_pos_items, dtype=torch.long)
                neg_t = torch.as_tensor(b_neg_items, dtype=torch.long)

                b_weights_t = (
                    torch.as_tensor(weights_arr[batch_indices], dtype=torch.float32)
                    if weights_arr is not None
                    else None
                )

                loss = self.model_.bpr_loss(
                    users=u_t,
                    pos_items=pos_t,
                    neg_items=neg_t,
                    l2_reg=self.l2_reg,
                    pair_weights=b_weights_t,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if epoch == 1 or epoch % max(1, (self.epochs // 5)) == 0 or epoch == self.epochs:
                avg_loss = epoch_loss / max(1, num_batches)
                logger.info(
                    "Epoch %d/%d - TrustGatedLightGCN Loss: %.4f",
                    epoch,
                    self.epochs,
                    avg_loss,
                )

        # 9. Cache final embeddings for inference
        self.model_.eval()
        with torch.no_grad():
            self.final_user_embs_, self.final_item_embs_ = self.model_.forward_propagation()

        return self

    def recommend(
        self,
        users: Sequence[str],
        k: int = 20,
        exclude_seen: bool = True,
    ) -> dict[str, list[str]]:
        """Generate top-K item recommendations using final embeddings."""
        if self.final_user_embs_ is None or self.final_item_embs_ is None:
            raise RuntimeError("Model must be fitted before recommend().")

        results: dict[str, list[str]] = {}

        with torch.no_grad():
            item_embeddings = self.final_item_embs_

            for user in users:
                u_idx = self.user2idx.get(user)
                if u_idx is None:
                    results[user] = []
                    continue

                u_emb = self.final_user_embs_[u_idx : u_idx + 1]
                scores = torch.matmul(u_emb, item_embeddings.t()).squeeze(0).cpu().numpy()

                if exclude_seen:
                    seen = self.user_seen_items.get(u_idx, set())
                    for s_item in seen:
                        scores[s_item] = -np.inf

                top_indices = np.argsort(-scores)[:k]
                results[user] = [self.idx2item[idx] for idx in top_indices if scores[idx] > -np.inf]

        return results

    def predict_scores(
        self,
        user_ids: Sequence[str],
        item_ids: Sequence[str],
    ) -> list[float]:
        """Compute recommendation affinity scores for (user, item) pairs."""
        if self.final_user_embs_ is None or self.final_item_embs_ is None:
            raise RuntimeError("Model must be fitted before predict_scores().")

        scores: list[float] = []
        with torch.no_grad():
            for u, i in zip(user_ids, item_ids, strict=False):
                u_idx = self.user2idx.get(u)
                i_idx = self.item2idx.get(i)
                if u_idx is None or i_idx is None:
                    scores.append(0.0)
                    continue
                u_e = self.final_user_embs_[u_idx]
                i_e = self.final_item_embs_[i_idx]
                scores.append(float((u_e * i_e).sum().item()))
        return scores


def tune_hard_filter_threshold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    candidate_thresholds: Sequence[float] = (0.4, 0.5, 0.6, 0.7, 0.8),
    target_metric: str = "NDCG@10",
    epochs: int = 15,
    seed: int = 20261021,
) -> tuple[float, dict[float, dict[str, float]]]:
    """Select optimal hard filtering threshold theta* using validation data strictly.

    Parameters
    ----------
    train_df:
        Clean training interactions.
    val_df:
        Validation interactions for threshold selection.
    candidate_thresholds:
        Candidate cutoff values to evaluate.
    target_metric:
        Validation metric to maximize (e.g. 'NDCG@10').
    epochs:
        Training epochs per candidate.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    tuple[float, dict[float, dict[str, float]]]:
        Optimal threshold theta*, and dictionary of validation metrics per candidate.
    """
    logger.info("Tuning hard-filtering threshold on validation data: %s", list(candidate_thresholds))

    # Build validation ground truth
    val_gt: dict[str, set[str]] = defaultdict(set)
    for u, i in zip(val_df["user_id"], val_df["item_id"], strict=False):
        val_gt[u].add(i)

    train_users = set(train_df["user_id"])
    eval_val_users = [u for u in val_gt if u in train_users]

    results_by_thresh: dict[float, dict[str, float]] = {}
    best_score = -1.0
    best_thresh = candidate_thresholds[0]

    for thresh in candidate_thresholds:
        try:
            model = TrustGatedLightGCN(
                embedding_dim=32,
                num_layers=2,
                epochs=epochs,
                batch_size=256,
                seed=seed,
                weighting_mode="hard",
                hard_threshold=thresh,
            )
            model.fit(train_df)
            val_recs = model.recommend(eval_val_users, k=20, exclude_seen=True)
            metrics = evaluate_recommendations(val_recs, val_gt, cutoffs=(10, 20))
            score = metrics.get(target_metric, 0.0)

            stats = model.edge_stats_
            pruned_frac = stats.get("pruned_fraction", 0.0)

            results_by_thresh[thresh] = {
                target_metric: round(score, 4),
                "Recall@20": round(metrics.get("Recall@20", 0.0), 4),
                "pruned_fraction": round(pruned_frac, 4),
                "pruned_count": float(stats.get("n_pruned_interactions", 0)),
            }

            logger.info(
                "Threshold %.2f -> %s: %.4f (pruned %.1f%%)",
                thresh,
                target_metric,
                score,
                pruned_frac * 100.0,
            )

            if score > best_score:
                best_score = score
                best_thresh = thresh

        except Exception as exc:
            logger.warning("Threshold %.2f failed with error: %s", thresh, exc)
            results_by_thresh[thresh] = {target_metric: 0.0, "pruned_fraction": 1.0}

    logger.info(
        "Optimal validation threshold selected: theta*=%.2f (%s=%.4f)",
        best_thresh,
        target_metric,
        best_score,
    )
    return best_thresh, results_by_thresh
