# VouchRec Phase 7: Experiment Plan & Research Protocol

- **Document Version:** 1.0.0
- **Status:** Approved / Active
- **Date:** 2026-10-21
- **Target Repository:** VouchRec (Phases 1–6 Implemented & Verified)

---

## 1. Research Objectives & Core Questions

Phase 7 executes a rigorous, multi-seed statistical evaluation of VouchRec's trust-gated collaborative filtering pipeline and dynamic exposure governor. Specifically, this evaluation addresses five primary research questions:

1. **RQ1 (Model Efficacy & Resilience):** Does Trust-Gated LightGCN with soft-linear message attenuation significantly outperform Vanilla LightGCN, BPR-MF, and Popularity baselines under controlled shilling attacks?
2. **RQ2 (Exposure Governor Synergy):** Does post-processing exposure governance provide a statistically significant reduction in target item rank promotion without degrading clean-data recommendation utility?
3. **RQ3 (Attack Generalizability):** How do the proposed defense mechanisms perform across varying attack strategies (Random, Average, Bandwagon), attack objectives (Push vs. Nuke), and attack budgets (0.5%, 1%, 2%, 5%)?
4. **RQ4 (Component Ablation):** What is the marginal contribution of each trust signal head (text, reviewer behavior, coordination), and how does soft weighting compare against discrete hard filtering?
5. **RQ5 (Governor Sensitivity):** What are the sensitivity boundaries of the Exposure Governor regarding rank jump caps ($\Delta r_{\max}$), suspicious ratio thresholds ($s_{\text{thresh}}$), and minimum evidence requirements ($N_{\min}$)?

---

## 2. Dataset Availability & Computational Resource Audit

### 2.1 Storage & Split Status
- **Configured Data Root:** `C:\VouchRecData\data` (located on external drive outside OneDrive sync).
- **Available Free Disk Space:** 186.08 GB.
- **Local Dataset Audit:**
  - `Musical_Instruments.jsonl.gz` (458.6 MB compressed, 1.56 GB uncompressed, 3.0M ratings): **Not present on local disk**.
  - Local Processed Reviews (`reviews.parquet`): 40 verified interactions.
  - Local Temporal Splits (`train.parquet`, `val.parquet`, `test.parquet`): Derived from the canonical 40-record Musical Instruments fixture (Train: 32 interactions / 8 users / 8 items; Val: 4 interactions; Test: 4 interactions).
- **Research Integrity Decision:**
  > [!IMPORTANT]
  > As mandated by Phase 7 integrity guidelines, the small fixture dataset must **never be silently substituted** for the full dataset. All experiments conducted on this fixture are explicitly labeled as **preliminary smoke-test / pilot benchmarks**.

### 2.2 Runtime & Memory Risk Analysis
1. **Multi-Seed Graph Training:** Training LightGCN across 7 model variants, 5 seeds, and multiple attack configurations requires repeated graph normalization and BPR optimization. On the local fixture, each epoch executes in ~15–30 ms; 20 epochs across 5 seeds will execute in under 2 minutes with memory footprint < 200 MB.
2. **Memory Leaks:** PyTorch graph tensors and PyArrow tables will be explicitly scoped and garbage-collected to prevent memory accumulation across seed loops. Memory profiling will be monitored via `tracemalloc`.
3. **Deterministic Seeding:** Multi-threading or unseeded data iterators can introduce non-deterministic tie-breaking in ranking. Python, NumPy, and PyTorch seeds will be synchronized explicitly per seed index.

---

## 3. Experimental Design & Matrix

### 3.1 Random Seed Protocol
To ensure statistical validity and prevent seed-cherry-picking, all experiments will execute across **five fixed pseudo-random seeds**:
$$\mathcal{S} = \{20261021, 20261022, 20261023, 20261024, 20261025\}$$
Every seed governs:
- Python `random.seed(s)`
- NumPy `np.random.default_rng(s)` and `np.random.seed(s)`
- PyTorch CPU `torch.manual_seed(s)`
- Simulator profile sampling and filler interaction selection
- Model parameter initialization and negative sampling

### 3.2 Model Comparison Matrix (7 Conditions)
1. **Popularity:** Non-personalized interaction count baseline.
2. **BPR-MF:** Matrix Factorization with Bayesian Personalized Ranking (dim=32, lr=0.005, L2=1e-4).
3. **Vanilla LightGCN:** 2-layer Graph Convolutional Recommender with uniform unweighted adjacency.
4. **Hard-Filtered LightGCN:** Trust-pruned graph discarding edges with provisional trust $\tau < 0.40$ (validated threshold).
5. **Trust-Gated LightGCN (Soft Linear):** LightGCN with continuous linear edge weighting ($w_{ui} = \tau_{ui}$).
6. **Trust-Gated LightGCN (Soft Power, $p=2$):** LightGCN with non-linear power edge weighting ($w_{ui} = \tau_{ui}^2$).
7. **Trust-Gated LightGCN + Exposure Governor:** Soft-linear LightGCN paired with post-processing rank-jump capping ($\Delta r_{\max} = 0, s_{\text{thresh}} = 0.40, N_{\min} = 2, N_{\text{low\_min}} = 2$).
*(Ablation condition: Vanilla LightGCN + Exposure Governor evaluated to isolate post-processing effect from graph weighting).*

### 3.3 Attack Evaluation Matrix
- **Strategies:** Random, Average-rating, Bandwagon.
- **Objectives:** Push (rating = 5.0 on target), Nuke (rating = 1.0 on target).
- **Attack Budgets:** 0.5%, 1.0%, 2.0%, 5.0% of user population.
- **Temporal Patterns:**
  - Burst attack: concentrated within 48-hour window before split cutoff.
  - Spread attack: distributed across historical training duration.
- **Text Generation:** Template-based synthetic reviews versus non-text reviews.

### 3.4 Trust Head Ablations
- Full Trust Fusion: $(\beta_{\text{text}}=0.35, \beta_{\text{behavior}}=0.35, \beta_{\text{coord}}=0.30)$
- No-Text Ablation: $(\beta_{\text{text}}=0.00, \beta_{\text{behavior}}=0.50, \beta_{\text{coord}}=0.50)$
- No-Behavior Ablation: $(\beta_{\text{text}}=0.50, \beta_{\text{behavior}}=0.00, \beta_{\text{coord}}=0.50)$
- No-Coordination Ablation: $(\beta_{\text{text}}=0.50, \beta_{\text{behavior}}=0.50, \beta_{\text{coord}}=0.00)$

### 3.5 Governor Sensitivity Matrix
- **Promotion Cap $\Delta r_{\max}$:** $\{0, 1, 3, 5\}$
- **Suspicious Ratio Threshold $s_{\text{thresh}}$:** $\{0.20, 0.40, 0.60\}$
- **Minimum Evidence Count $N_{\min}$:** $\{1, 2, 5\}$

---

## 4. Metric Definitions & Statistical Methodology

### 4.1 Evaluation Metrics

| Metric | Category | Formula / Definition | Desirable Direction |
|---|---|---|---|
| **Recall@20** | Rec Quality | Proportion of ground-truth test items recovered in top-20. | Larger ($\uparrow$) |
| **NDCG@10** | Rec Quality | Normalized Discounted Cumulative Gain over top-10 recommendations. | Larger ($\uparrow$) |
| **HitRate@10** | Rec Quality | Binary indicator whether at least one test item appears in top-10. | Larger ($\uparrow$) |
| **Target HitRate@10** | Attack Impact | Proportion of test users receiving target item in top-10. | Smaller ($\downarrow$) |
| **Target Mean Rank** | Attack Impact | Average rank of target item across test users (unranked items scored as $K+1$). | Larger ($\uparrow$, demoted) |
| **Target Rank Shift** | Attack Impact | $\Delta r = \bar{r}_{\text{clean}} - \bar{r}_{\text{attacked}}$ (positive means promoted toward top). | Smaller ($\downarrow$) |
| **Trust ROC-AUC** | Trust Detection | Area under ROC curve distinguishing real from synthetic interactions. | Larger ($\uparrow$) |
| **Adjusted Share** | Governance | Percentage of top-$K$ recommendations intervened on by governor. | Contextual |
| **Clean Utility Delta** | Governance | $\text{NDCG}_{\text{clean, gov}} - \text{NDCG}_{\text{clean, ungov}}$ (audits clean degradation). | Near Zero ($\approx 0$) |

### 4.2 Statistical Testing Protocols
1. **Point Estimates:** Mean and sample standard deviation across 5 matched seeds:
   $$\bar{x} = \frac{1}{n} \sum_{s=1}^n x_s, \quad s_x = \sqrt{\frac{1}{n-1} \sum_{s=1}^n (x_s - \bar{x})^2}$$
2. **Confidence Intervals:** 95% Confidence Interval based on Student's $t$-distribution with $df = n - 1 = 4$:
   $$\text{CI}_{95\%} = \bar{x} \pm t_{4, 0.975} \frac{s_x}{\sqrt{n}}, \quad t_{4, 0.975} = 2.776$$
3. **Hypothesis Testing for Paired Differences:**
   For paired models $A$ and $B$ evaluated on identical seeds and data splits:
   $$D_s = x_{A, s} - x_{B, s}, \quad t = \frac{\bar{D}}{s_D / \sqrt{n}}$$
   Paired two-tailed Student's $t$-test ($p < 0.05$ threshold).
4. **Effect Size (Cohen's $d_z$ for paired samples):**
   $$d_z = \frac{\bar{D}}{s_D}$$
   Interpretation: $|d_z| \ge 0.2$ small, $\ge 0.5$ medium, $\ge 0.8$ large.
5. **Multiple Comparison Correction:**
   Holm-Bonferroni step-down correction applied across model comparisons to strictly control Family-Wise Error Rate (FWER) at $\alpha = 0.05$.

---

## 5. Anti-Leakage & Research Integrity Constraints

1. **Temporal Horizon Integrity:** All models, graph normalizations, trust scorers, and governor statistics are computed **exclusively** on the training split ($t \le T_{\text{val}}$). Test interactions ($t > T_{\text{test}}$) are strictly quarantined.
2. **Zero Label Leakage:** Synthetic attack labels (`is_attack`, `label_value`, etc.) are stripped from all training dataFrames prior to feature computation and model fitting. Runtime assertions enforce that no label columns enter the feature space.
3. **No Uncalibrated Probability Claims:** Heuristic trust scores represent anomaly indices in $[0, 1]$, not certified ground-truth posterior probabilities.
4. **Truthful Reporting:** Fixture results are documented as fixture benchmarks. Every completed run and per-seed observation is exported to CSV/JSON to permit third-party re-analysis.
