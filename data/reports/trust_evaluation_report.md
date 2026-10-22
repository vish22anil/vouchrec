# VouchRec Phase 3: Provisional Trust Scoring Evaluation Report

- **Generated at:** `2026-10-21T05:25:19.679386+00:00`
- **Total Reviews Scored:** `40`
- **Provisional Trust Mean (±Std):** `0.7624 (±0.0593)`
- **Provisional Trust Range:** `[0.6500, 0.9125]`
- **Mean Signal Confidence:** `0.77`

## 1. Fusion Configuration & Weights

| Component Head | Configured Weight | Mean Head Score |
|---|---|---|
| **Review Text Head** | 0.35 | 0.6337 |
| **Reviewer Behavior Head** | 0.35 | 0.6875 |
| **Coordination Head** | 0.30 | 1.0000 |

## 2. Triggered Reason Code Frequencies

| Reason Code | Count | Share | Explanation |
|---|---|---|---|
| `COORD_INDEPENDENT` | 40 | 100.0% | No cross-review duplicate or burst coordination detected |
| `BEHAVIOR_HIGH_VERIFIED_PURCHASE` | 22 | 55.0% | Standard baseline |
| `TEXT_SHORT` | 22 | 55.0% | Missing or minimal text body |
| `TEXT_NATURAL_BASELINE` | 17 | 42.5% | Standard baseline |
| `BEHAVIOR_SPARSE_HISTORY` | 16 | 40.0% | Sparse user history (assigned neutral 0.5 prior) |
| `BEHAVIOR_ESTABLISHED_HISTORY` | 8 | 20.0% | Healthy detail or established longitudinal activity |
| `BEHAVIOR_BALANCED_RATINGS` | 5 | 12.5% | Standard baseline |
| `BEHAVIOR_STANDARD` | 1 | 2.5% | Standard baseline |
| `TEXT_MISSING` | 1 | 2.5% | Missing or minimal text body |

## 3. Methodological Scope & Validation Boundary

> [!IMPORTANT]
> **Provisional Status:** The scores above reflect unsupervised anomaly detection and rule-based behavioral heuristics.
> Because Amazon Reviews 2023 lacks ground-truth fake review annotations, **no supervised classification accuracy, AUC, or F1 scores can be claimed** at this stage.
> In Phase 4, controlled attack simulations will provide known ground-truth labels to rigorously benchmark and calibrate this pipeline.

## 4. Signal Correlation Matrix

| Metric | Text | Behavior | Coordination | Provisional Fused |
|---|---|---|---|---|
| **text_trust_score** | 1.0000 | -0.2674 | N/A* | 0.3293 |
| **behavior_trust_score** | -0.2674 | 1.0000 | N/A* | 0.8218 |
| **coordination_trust_score** | N/A* | N/A* | N/A* | N/A* |
| **provisional_trust_score** | 0.3293 | 0.8218 | N/A* | 1.0000 |

\* *N/A indicates zero variance in the component head across the evaluated sample.*
