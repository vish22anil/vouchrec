# VouchRec Phase 6: Dynamic Exposure Governor Evaluation Report

- **Simulation Timestamp:** `2026-10-21T11:00:48.965059+00:00`
- **Attack Strategy:** `bandwagon`
- **Attack Objective:** `push`
- **Target Product:** `P006`
- **Attack Budget:** `5.0%` (1 synthetic profiles)
- **Synthetic Reviews Injected:** `4`
- **Governor Max Permitted Rank Jump (Delta r_max):** `0`
- **Minimum Evidence Reviews Required:** `2`
- **Low-Trust Review Cutoff:** `0.60`
- **Suspicious Ratio Threshold:** `40%`

## 1. Experimental Conditions Comparison

| Condition | Clean NDCG@10 | Attacked NDCG@10 | NDCG Delta | Target HR@10 (C -> A) | Target Mean Rank (C -> A) | Rank Promotion | Recommendations Adjusted | Jumps Capped | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| **Vanilla_LightGCN** | 0.9077 | 0.6077 | -0.3001 | 0.50 -> 0.50 | 7.2 -> 6.8 | +0.5 | 0 (0.0%) | 0 | 0.36s |
| **TrustGated_LightGCN_Ungoverned** | 0.8155 | 0.6404 | -0.1751 | 0.50 -> 0.50 | 7.2 -> 6.8 | +0.5 | 0 (0.0%) | 0 | 0.50s |
| **TrustGated_LightGCN_Governed** | 0.8155 | 0.7327 | -0.0828 | 0.50 -> 0.50 | 7.2 -> 6.8 | +0.5 | 6 (35.3%) | 6 | 0.46s |
| **Vanilla_LightGCN_Governed** | 0.9077 | 0.6404 | -0.2673 | 0.50 -> 0.50 | 7.2 -> 6.8 | +0.5 | 6 (35.3%) | 6 | 0.28s |

## 2. Governor Audit Trail & Reason Code Distribution

| Condition | GOV_RANK_JUMP_CAPPED | GOV_PASS_UNMODIFIED | GOV_PRESERVED_SPARSE_EVIDENCE | GOV_PRESERVED_HIGH_TRUST | GOV_DISABLED |
|---|---|---|---|---|---|
| **Vanilla_LightGCN** | 0 | 0 | 0 | 0 | 17 |
| **TrustGated_LightGCN_Ungoverned** | 0 | 0 | 0 | 0 | 17 |
| **TrustGated_LightGCN_Governed** | 6 | 11 | 0 | 0 | 0 |
| **Vanilla_LightGCN_Governed** | 6 | 11 | 0 | 0 | 0 |

## 3. Analysis & Key Insights

1. **Exposure Governor vs. Un-Governed Baseline:**
   - Checks if the post-recommendation ranking cap successfully arrests target item promotion.
2. **Safeguard Protections (Cold / Unpopular Items):**
   - Confirms that items with sparse review history are protected via `GOV_PRESERVED_SPARSE_EVIDENCE`.
3. **Trade-Off Evaluation:**
   - Audits whether governance preserves clean recommendation NDCG while suppressing attacked promotion.

> [!IMPORTANT]
> The Exposure Governor limits ranking influence; it does not certify reviews as genuine or fraudulent.
