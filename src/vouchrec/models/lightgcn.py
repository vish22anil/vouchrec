"""LightGCN recommendation baseline (He et al., SIGIR 2020)."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.optim as optim

from vouchrec.models.base import BaseRecommender

logger = logging.getLogger(__name__)


class _LightGCNModule(nn.Module):
    """PyTorch implementation of LightGCN embedding propagation."""

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
        """Compute final user and item embeddings by multi-layer graph convolution."""
        all_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        embs_list = [all_embeddings]

        for _ in range(self.num_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj, all_embeddings)
            embs_list.append(all_embeddings)

        # Average across all layer representations: E* = 1 / (K + 1) * sum(E^(k))
        final_embeddings = torch.stack(embs_list, dim=0).mean(dim=0)
        final_users, final_items = torch.split(final_embeddings, [self.num_users, self.num_items])
        return final_users, final_items

    def bpr_loss(
        self,
        users: torch.Tensor,
        pos_items: torch.Tensor,
        neg_items: torch.Tensor,
        l2_reg: float,
    ) -> torch.Tensor:
        """Compute BPR loss on propagated embeddings + L2 regularization on base embeddings."""
        final_users, final_items = self.forward_propagation()

        u_emb = final_users[users]
        pos_emb = final_items[pos_items]
        neg_emb = final_items[neg_items]

        pos_scores = (u_emb * pos_emb).sum(dim=-1)
        neg_scores = (u_emb * neg_emb).sum(dim=-1)
        diff = pos_scores - neg_scores

        bpr = -torch.mean(torch.log(torch.sigmoid(diff) + 1e-10))

        # L2 regularize base embeddings (He et al., 2020)
        base_u = self.user_embedding(users)
        base_pos = self.item_embedding(pos_items)
        base_neg = self.item_embedding(neg_items)
        reg = (
            l2_reg
            * (base_u.pow(2).sum() + base_pos.pow(2).sum() + base_neg.pow(2).sum())
            / len(users)
        )

        return bpr + reg


class LightGCNRecommender(BaseRecommender):
    """LightGCN recommendation model with BPR loss and uniform negative sampling."""

    def __init__(
        self,
        embedding_dim: int = 64,
        num_layers: int = 3,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 1024,
        seed: int = 20261021,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        self.user2idx: dict[str, int] = {}
        self.idx2user: list[str] = []
        self.item2idx: dict[str, int] = {}
        self.idx2item: list[str] = []
        self.user_seen_items: dict[int, set[int]] = defaultdict(set)
        self.model_: _LightGCNModule | None = None

        self.final_user_embs_: torch.Tensor | None = None
        self.final_item_embs_: torch.Tensor | None = None

    def _build_normalized_adj(
        self,
        num_users: int,
        num_items: int,
        user_indices: np.ndarray,
        item_indices: np.ndarray,
    ) -> torch.Tensor:
        """Construct symmetric normalized bipartite adjacency matrix D^(-1/2) A D^(-1/2)."""
        num_nodes = num_users + num_items
        row = np.concatenate([user_indices, item_indices + num_users])
        col = np.concatenate([item_indices + num_users, user_indices])
        data = np.ones(len(row), dtype=np.float32)

        adj_matrix = sp.coo_matrix((data, (row, col)), shape=(num_nodes, num_nodes))
        rowsum = np.array(adj_matrix.sum(axis=1)).flatten()
        d_inv_sqrt = np.power(np.maximum(rowsum, 1.0), -0.5)
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        norm_adj = d_mat_inv_sqrt.dot(adj_matrix).dot(d_mat_inv_sqrt).tocoo()

        indices = torch.tensor(np.vstack((norm_adj.row, norm_adj.col)), dtype=torch.long)
        values = torch.tensor(norm_adj.data, dtype=torch.float32)
        shape = torch.Size(norm_adj.shape)

        return torch.sparse_coo_tensor(indices, values, shape, check_invariants=False).coalesce()

    def fit(self, interactions: pd.DataFrame) -> LightGCNRecommender:
        """Fit LightGCN on training interactions graph."""
        if interactions.empty:
            raise ValueError("Interactions dataframe is empty.")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        unique_users = sorted(interactions["user_id"].unique())
        unique_items = sorted(interactions["item_id"].unique())

        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = unique_users
        self.item2idx = {item: i for i, item in enumerate(unique_items)}
        self.idx2item = unique_items

        num_users = len(self.user2idx)
        num_items = len(self.item2idx)

        user_indices = interactions["user_id"].map(self.user2idx).values
        item_indices = interactions["item_id"].map(self.item2idx).values

        self.user_seen_items = defaultdict(set)
        pairs: list[tuple[int, int]] = []
        for u_idx, i_idx in zip(user_indices, item_indices, strict=False):
            self.user_seen_items[u_idx].add(i_idx)
            pairs.append((u_idx, i_idx))

        norm_adj = self._build_normalized_adj(num_users, num_items, user_indices, item_indices)

        self.model_ = _LightGCNModule(
            num_users=num_users,
            num_items=num_items,
            embedding_dim=self.embedding_dim,
            num_layers=self.num_layers,
            norm_adj=norm_adj,
        )

        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        pairs_arr = np.array(pairs, dtype=np.int64)
        num_pairs = len(pairs)

        logger.info(
            "Training LightGCN (%d users, %d items, %d edges, layers=%d, epochs=%d)",
            num_users,
            num_items,
            num_pairs,
            self.num_layers,
            self.epochs,
        )

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

                loss = self.model_.bpr_loss(u_t, pos_t, neg_t, self.l2_reg)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if epoch == 1 or epoch % max(1, (self.epochs // 5)) == 0 or epoch == self.epochs:
                avg_loss = epoch_loss / max(1, num_batches)
                logger.info("Epoch %d/%d - LightGCN Loss: %.4f", epoch, self.epochs, avg_loss)

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
        """Generate top-K item recommendations using final LightGCN embeddings."""
        if self.final_user_embs_ is None or self.final_item_embs_ is None:
            raise RuntimeError("Model must be fitted before recommend().")

        results: dict[str, list[str]] = {}

        with torch.no_grad():
            item_embeddings = self.final_item_embs_  # (num_items, dim)

            for user in users:
                u_idx = self.user2idx.get(user)
                if u_idx is None:
                    # Cold user fallback
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
        """Compute scores for pairs using final embeddings."""
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
