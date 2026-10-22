# VouchRec Phase 4: Controlled Attack Simulation Report

- **Simulation Timestamp:** `2026-10-21T05:41:11.745930+00:00`
- **Attack Strategy:** `bandwagon`
- **Attack Objective:** `push`
- **Target Product:** `P006`
- **Attack Budget:** `5.0%` (1 synthetic profiles)
- **Synthetic Reviews Injected:** `4`
- **Seed:** `20261021`

## 1. Recommendation Impact on Target Product

| Model | Clean HitRate@10 | Attacked HitRate@10 | HitRate Delta | Clean Mean Rank | Attacked Mean Rank | Rank Delta |
|---|---|---|---|---|---|---|
| **Popularity** | 0.5000 | 0.5000 | +0.0000 | 7.8 | 7.2 | +0.5 |
| **BPR_MF** | 0.5000 | 0.5000 | +0.0000 | 6.5 | 6.5 | +0.0 |
| **LightGCN** | 0.5000 | 0.5000 | +0.0000 | 7.8 | 7.0 | +0.8 |

## 2. System-Wide Recommendation Quality (Collateral Distortion)

| Model | Clean NDCG@10 | Attacked NDCG@10 | Clean Recall@20 | Attacked Recall@20 | Runtime (s) |
|---|---|---|---|---|---|
| **Popularity** | 0.7654 | 0.7654 | 1.0000 | 1.0000 | 0.02s |
| **BPR_MF** | 0.8467 | 0.5903 | 1.0000 | 1.0000 | 1.82s |
| **LightGCN** | 0.9077 | 0.6077 | 1.0000 | 1.0000 | 0.05s |

## 3. Trust Model Detection Performance on Ground-Truth Synthetic Labels

- **Mean Trust on Real Reviews:** `0.7517`
- **Mean Trust on Synthetic Reviews:** `0.7463`
- **Trust Separation Gap:** `+0.0055` (positive indicates synthetic reviews receive lower trust)
- **Detection ROC-AUC:** `0.5781`

> [!NOTE]
> Detection metrics are calculated strictly offline using known simulated ground-truth labels.
> In accordance with anti-leakage principles, these labels were never exposed to recommendation models or trust feature extractors.
