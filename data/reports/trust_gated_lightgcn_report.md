# VouchRec Phase 5: Trust-Gated LightGCN Evaluation Report

- **Simulation Timestamp:** `2026-10-21T10:44:10.090166+00:00`
- **Attack Strategy:** `bandwagon`
- **Attack Objective:** `push`
- **Target Product:** `P006`
- **Attack Budget:** `5.0%` (1 synthetic profiles)
- **Synthetic Reviews Injected:** `4`
- **Selected Hard-Filtering Threshold (Validation theta*):** `0.40`

## 1. Hard Filtering Threshold Selection (Validation Set Only)

| Candidate Threshold | Val NDCG@10 | Val Recall@20 | Pruned Interactions | Pruned Fraction |
|---|---|---|---|---|
| `0.40` **(Selected theta*)** | 0.7153 | 1.0000 | 0 | 0.0% |
| `0.50` | 0.7153 | 1.0000 | 0 | 0.0% |
| `0.60` | 0.7153 | 1.0000 | 0 | 0.0% |
| `0.70` | 0.6934 | 1.0000 | 7 | 21.9% |
| `0.80` | 0.5301 | 1.0000 | 25 | 78.1% |

## 2. Target Product Attack Impact (Clean vs. Attacked)

| Model | Clean Target HR@10 | Attacked Target HR@10 | HR Delta | Clean Mean Rank | Attacked Mean Rank | Rank Promotion |
|---|---|---|---|---|---|---|
| **Popularity** | 0.5000 | 0.5000 | +0.0000 | 7.8 | 7.2 | +0.5 |
| **BPR_MF** | 0.5000 | 0.5000 | +0.0000 | 7.0 | 6.2 | +0.8 |
| **Vanilla_LightGCN** | 0.5000 | 0.5000 | +0.0000 | 7.2 | 6.8 | +0.5 |
| **LightGCN_HardFilter(theta=0.40)** | 0.5000 | 0.5000 | +0.0000 | 7.2 | 6.8 | +0.5 |
| **TrustGated_LightGCN_SoftLinear** | 0.5000 | 0.5000 | +0.0000 | 7.2 | 6.8 | +0.5 |
| **TrustGated_LightGCN_SoftPower(p=2)** | 0.5000 | 0.5000 | +0.0000 | 7.2 | 6.8 | +0.5 |
| **Ablation_UniformLightGCN** | 0.5000 | 0.5000 | +0.0000 | 7.2 | 6.8 | +0.5 |

## 3. System-Wide Recommendation Quality (Collateral Distortion & Clean Utility)

| Model | Clean NDCG@10 | Attacked NDCG@10 | NDCG Delta | Clean Recall@20 | Attacked Recall@20 | Runtime |
|---|---|---|---|---|---|---|
| **Popularity** | 0.7654 | 0.7654 | +0.0000 | 1.0000 | 1.0000 | 0.02s |
| **BPR_MF** | 0.8467 | 0.6077 | -0.2390 | 1.0000 | 1.0000 | 0.05s |
| **Vanilla_LightGCN** | 0.9077 | 0.6077 | -0.3001 | 1.0000 | 1.0000 | 0.07s |
| **LightGCN_HardFilter(theta=0.40)** | 0.9077 | 0.6077 | -0.3001 | 1.0000 | 1.0000 | 0.10s |
| **TrustGated_LightGCN_SoftLinear** | 0.8155 | 0.6404 | -0.1751 | 1.0000 | 1.0000 | 0.09s |
| **TrustGated_LightGCN_SoftPower(p=2)** | 0.8155 | 0.6077 | -0.2078 | 1.0000 | 1.0000 | 0.09s |
| **Ablation_UniformLightGCN** | 0.9077 | 0.6077 | -0.3001 | 1.0000 | 1.0000 | 0.10s |

## 4. Graph Edge Weight & Pruning Statistics

| Model | Mode | Mean Edge Weight | Min Weight | Max Weight | Pruned Edges | Pruned % |
|---|---|---|---|---|---|---|
| **Popularity** | unweighted / binary | 1.0000 | 1.0000 | 1.0000 | 0 | 0.0% |
| **BPR_MF** | unweighted / binary | 1.0000 | 1.0000 | 1.0000 | 0 | 0.0% |
| **Vanilla_LightGCN** | unweighted / binary | 1.0000 | 1.0000 | 1.0000 | 0 | 0.0% |
| **LightGCN_HardFilter(theta=0.40)** | hard | 1.0000 | 1.0000 | 1.0000 | 0 | 0.0% |
| **TrustGated_LightGCN_SoftLinear** | soft | 0.7511 | 0.6500 | 0.9125 | 0 | 0.0% |
| **TrustGated_LightGCN_SoftPower(p=2)** | soft | 0.5676 | 0.4225 | 0.8327 | 0 | 0.0% |
| **Ablation_UniformLightGCN** | uniform | 1.0000 | 1.0000 | 1.0000 | 0 | 0.0% |

## 5. Required Ablations & Trade-Off Analysis

1. **Vanilla LightGCN vs. Trust-Gated LightGCN (Soft):**
   - Compares unweighted graph propagation against trust-damped message passing.
2. **Soft Weighting vs. Hard Filtering:**
   - Compares continuous message attenuation against discrete graph pruning (threshold selected on validation set).
3. **Uniform Edge Weights vs. Trust-Derived Weights:**
   - Evaluates whether trust-derived weighting outperforms equal-weight baselines in preserving clean NDCG while mitigating rank promotion.

> [!IMPORTANT]
> A defense that dampens target promotion but degrades clean recommendation utility must be treated as an engineering trade-off rather than an unconditional improvement.
