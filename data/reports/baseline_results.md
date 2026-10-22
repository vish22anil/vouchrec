# VouchRec Phase 2: Recommendation Baseline Results

- **Evaluated at:** `2026-10-22T18:17:15.799083+00:00`
- **Seed:** `20261021`
- **Split Method:** `global_temporal`
- **Training Interactions:** `32`
- **Test Interactions:** `4`

## Summary Evaluation Table

| Model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | HitRate@10 | HitRate@20 |
|---|---|---|---|---|---|---|
| **Popularity** | 1.0000 | 1.0000 | 0.7654 | 0.7654 | 1.0000 | 1.0000 |
| **BPR_MF** | 1.0000 | 1.0000 | 0.6731 | 0.6731 | 1.0000 | 1.0000 |
| **LightGCN** | 1.0000 | 1.0000 | 0.6577 | 0.6577 | 1.0000 | 1.0000 |

## Hyperparameters & Configurations

### Popularity
```json
{
  "enabled": true
}
```

### BPR_MF
```json
{
  "enabled": true,
  "embedding_dim": 64,
  "learning_rate": 0.001,
  "l2_reg": 0.0001,
  "epochs": 100,
  "batch_size": 4096
}
```

### LightGCN
```json
{
  "enabled": true,
  "embedding_dim": 64,
  "num_layers": 3,
  "learning_rate": 0.001,
  "l2_reg": 0.0001,
  "epochs": 200,
  "batch_size": 4096
}
```
