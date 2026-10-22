"""Matrix Factorization with Bayesian Personalized Ranking (BPR-MF)."""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from vouchrec.models.base import BaseRecommender

logger = logging.getLogger(__name__)


class _BPRMFModule(nn.Module):
    """PyTorch module holding user and item embedding tables."""

    def __init__(self, num_users: int, num_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embeddings = nn.Embedding(num_users, embedding_dim)
        self.item_embeddings = nn.Embedding(num_items, embedding_dim)
        nn.init.normal_(self.user_embeddings.weight, std=0.01)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

    def forward(
        self,
        user_indices: torch.Tensor,
        pos_item_indices: torch.Tensor,
        neg_item_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_emb = self.user_embeddings(user_indices)
        i_emb = self.item_embeddings(pos_item_indices)
        j_emb = self.item_embeddings(neg_item_indices)
        pos_scores = (u_emb * i_emb).sum(dim=-1)
        neg_scores = (u_emb * j_emb).sum(dim=-1)
        diff = pos_scores - neg_scores
        return diff, u_emb, torch.cat([i_emb, j_emb], dim=0)


class BPRMFRecommender(BaseRecommender):
    """Bayesian Personalized Ranking Matrix Factorization recommender."""

    def __init__(
        self,
        embedding_dim: int = 64,
        learning_rate: float = 0.001,
        l2_reg: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 1024,
        seed: int = 20261021,
    ) -> None:
        self.embedding_dim = embedding_dim
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
        self.model_: _BPRMFModule | None = None

    def fit(self, interactions: pd.DataFrame) -> BPRMFRecommender:
        """Fit BPR-MF on training interactions."""
        if interactions.empty:
            raise ValueError("Interactions dataframe is empty.")

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        # Build ID mappings
        unique_users = sorted(interactions["user_id"].unique())
        unique_items = sorted(interactions["item_id"].unique())

        self.user2idx = {u: i for i, u in enumerate(unique_users)}
        self.idx2user = unique_users
        self.item2idx = {item: i for i, item in enumerate(unique_items)}
        self.idx2item = unique_items

        num_users = len(self.user2idx)
        num_items = len(self.item2idx)

        # Build user interaction sets and positive pairs
        user_indices = interactions["user_id"].map(self.user2idx).values
        item_indices = interactions["item_id"].map(self.item2idx).values

        self.user_seen_items = defaultdict(set)
        pairs: list[tuple[int, int]] = []
        for u_idx, i_idx in zip(user_indices, item_indices, strict=False):
            self.user_seen_items[u_idx].add(i_idx)
            pairs.append((u_idx, i_idx))

        num_pairs = len(pairs)
        self.model_ = _BPRMFModule(num_users, num_items, self.embedding_dim)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)

        logger.info(
            "Training BPR-MF (%d users, %d items, %d pairs, epochs=%d, dim=%d)",
            num_users,
            num_items,
            num_pairs,
            self.epochs,
            self.embedding_dim,
        )

        pairs_arr = np.array(pairs, dtype=np.int64)

        for epoch in range(1, self.epochs + 1):
            indices = np.random.permutation(num_pairs)
            epoch_loss = 0.0
            num_batches = 0

            for start_idx in range(0, num_pairs, self.batch_size):
                batch_indices = indices[start_idx : start_idx + self.batch_size]
                b_users = pairs_arr[batch_indices, 0]
                b_pos_items = pairs_arr[batch_indices, 1]

                # Uniform negative sampling: for each user, sample an item not in user_seen_items
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

                diff, u_emb, i_emb = self.model_(u_t, pos_t, neg_t)
                bpr_loss = -torch.mean(torch.log(torch.sigmoid(diff) + 1e-10))
                reg_loss = self.l2_reg * (u_emb.pow(2).sum() + i_emb.pow(2).sum()) / len(u_t)
                loss = bpr_loss + reg_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            if epoch == 1 or epoch % max(1, (self.epochs // 5)) == 0 or epoch == self.epochs:
                avg_loss = epoch_loss / max(1, num_batches)
                logger.info("Epoch %d/%d - Loss: %.4f", epoch, self.epochs, avg_loss)

        return self

    def recommend(
        self,
        users: Sequence[str],
        k: int = 20,
        exclude_seen: bool = True,
    ) -> dict[str, list[str]]:
        """Generate top-K item recommendations for given users."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before recommend().")

        self.model_.eval()
        results: dict[str, list[str]] = {}

        with torch.no_grad():
            item_embeddings = self.model_.item_embeddings.weight  # (num_items, dim)

            for user in users:
                u_idx = self.user2idx.get(user)
                if u_idx is None:
                    # Cold user fallback: empty list
                    results[user] = []
                    continue

                u_emb = self.model_.user_embeddings.weight[u_idx : u_idx + 1]  # (1, dim)
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
        """Compute scores for pairs."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before predict_scores().")

        self.model_.eval()
        scores: list[float] = []
        with torch.no_grad():
            for u, i in zip(user_ids, item_ids, strict=False):
                u_idx = self.user2idx.get(u)
                i_idx = self.item2idx.get(i)
                if u_idx is None or i_idx is None:
                    scores.append(0.0)
                    continue
                u_e = self.model_.user_embeddings.weight[u_idx]
                i_e = self.model_.item_embeddings.weight[i_idx]
                scores.append(float((u_e * i_e).sum().item()))
        return scores
