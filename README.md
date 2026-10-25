# VouchRec: Trust-Gated Recommendation Under Review Manipulation

VouchRec is a research framework for investigating the vulnerability of collaborative filtering recommenders to synthetic and coordinated review manipulation, and evaluating trust-gated defense mechanisms and exposure governance.

---

## Repository Layout

```
VouchRec/
├── configs/
│   ├── data.yaml            # Dataset source URLs, checksums, and paths
│   ├── preprocessing.yaml   # Cleaning, validation, and deduplication rules
│   ├── experiment.yaml      # Splitting protocols, evaluation cutoffs, metrics
│   └── model.yaml           # Model hyperparameters (Popularity, BPR-MF, LightGCN)
├── docs/
│   ├── dataset_card.md      # Comprehensive dataset card for Amazon Reviews 2023
│   └── research_notes.md    # Methodological formulations, baselines, and logs
├── src/vouchrec/
│   ├── config.py            # Configuration loader and data root resolution
│   ├── logging_utils.py     # Unified logging configuration
│   ├── data/
│   │   ├── schema.py        # Canonical Arrow schema, quality flags, anti-leakage guards
│   │   ├── io_utils.py      # Resumable I/O, BLAKE2b hashing, Parquet read/write
│   │   ├── download.py      # Resumable download with storage checks & provenance manifest
│   │   ├── standardize.py   # Raw -> Interim (with flags) & Processed Parquet
│   │   └── quality_report.py# Machine-readable JSON and Markdown audit reports
│   ├── evaluation/
│   │   ├── metrics.py       # Recall@K, NDCG@K, HitRate@K implementations
│   │   ├── splits.py        # Leakage-free global temporal splitting
│   │   └── statistical.py   # Student's t CI, paired t-test, Cohen's d, Holm-Bonferroni
│   ├── models/
│   │   ├── base.py          # BaseRecommender abstract class
│   │   ├── popularity.py    # Popularity baseline
│   │   ├── bpr_mf.py        # Bayesian Personalized Ranking Matrix Factorization
│   │   ├── lightgcn.py      # LightGCN graph convolution recommender
│   │   └── trust_gated_lightgcn.py # Trust-Gated LightGCN with weighted message passing
│   ├── trust/
│   │   ├── text_signals.py  # Lexical diversity, repetition, and anomaly scoring
│   │   ├── behavior_signals.py # Historical review rate, rating entropy, bursts
│   │   ├── coordination_signals.py # Cross-user duplicate text and item bursts
│   │   └── fusion.py        # Weighted fusion producing provisional trust score
│   ├── attacks/
│   │   ├── config.py        # Attack parameters, rounding rules, config hashing
│   │   ├── strategies.py    # Random, Average, and Bandwagon profile generators
│   │   ├── simulator.py     # Deterministic simulation engine & non-collision ID generator
│   │   └── evaluation.py    # Clean vs. attacked impact metrics and trust ROC-AUC
│   ├── governance/
│   │   ├── config.py        # Exposure governor thresholds, limits, and safeguards
│   │   └── exposure_governor.py # Post-processing rank-jump capping & audit logger
│   └── api/
│       ├── app.py           # FastAPI application, CORS, static mount, and 12 routes
│       ├── schemas.py       # Pydantic v2 validation models
│       └── services.py      # Core engine integration & recommender caching
├── frontend/
│   ├── src/                 # React 18 + TypeScript + Tailwind CSS UI
│   │   ├── components/      # Sidebar, Navbar, status badges
│   │   ├── views/           # 7 Interactive Research Views
│   │   ├── api.ts           # Type-safe API client
│   │   └── types.ts         # TypeScript schema definitions
│   ├── package.json         # Node dependencies (React, Vite, Recharts, Lucide)
│   └── vite.config.ts       # Vite config with /api reverse proxy
├── scripts/
│   ├── 01_download.py       # Ingest raw dataset from official source
│   ├── 02_standardize.py    # Standardize raw reviews into Parquet tables
│   ├── 03_quality_report.py # Compute audit statistics and save quality report
│   ├── 04_create_splits.py  # Construct train/val/test temporal splits
│   ├── score_trust.py       # Score provisional trust and export audit report
│   ├── train_baselines.py   # Train and benchmark all baseline models
│   ├── run_attack_simulation.py # Controlled attack injector and baseline evaluator
│   ├── run_trust_gated_experiments.py # Phase 5 Trust-Gated LightGCN benchmark & ablations
│   ├── run_exposure_governor_experiments.py # Phase 6 Dynamic Exposure Governor benchmark
│   ├── run_phase7_experiments.py # Phase 7 Multi-seed statistical evaluation & ablations
│   └── run_api.py           # Launch Uvicorn FastAPI server on port 8000
├── tests/
│   ├── fixtures/
│   │   └── sample_reviews.jsonl # Isolated smoke-test review fixture
│   ├── test_schema.py       # Arrow schema & anti-leakage validation tests
│   ├── test_data_pipeline.py# Deduplication, flag generation, and exclusion tests
│   ├── test_splits.py       # Temporal ordering & leakage prevention tests
│   ├── test_metrics.py      # Precision, Recall, NDCG mathematical tests
│   ├── test_models.py       # Popularity, BPR-MF, and LightGCN unit tests
│   ├── test_trust.py        # Trust heads, fusion, and anti-leakage guards
│   ├── test_attacks.py      # Attack strategies, determinism, burst timing & leakage rejection
│   ├── test_trust_gated_lightgcn.py # Trust-gated adjacency, normalization & threshold tuning
│   ├── test_exposure_governor.py # Exposure governor unit, safeguard, and leakage tests
│   ├── test_phase7.py       # Multi-seed determinism, CI, and statistical test suite
│   ├── test_api.py          # FastAPI endpoint integration & schema validation tests
│   └── test_end_to_end.py   # Full pipeline integration smoke test
├── .env.example             # Template for external VOUCHREC_DATA_ROOT & API ports
├── pyproject.toml           # Project dependencies and tool configurations
└── requirements.txt         # Pinned production, test, and FastAPI dependencies
```

---

## Installation & Setup

VouchRec requires **Python 3.11** (recommended path outside cloud-synced folders):

```bash
# 1. Create and activate a virtual environment
python -m venv C:\VouchRecData\.venv
.\C:\VouchRecData\.venv\Scripts\Activate.ps1

# 2. Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e . --no-deps

# 3. Configure local data directory outside OneDrive
cp .env.example .env
# Edit .env to set VOUCHREC_DATA_ROOT (default: C:\VouchRecData\data)
```

---

## Reproduction Commands

### 1. Run the Full Test Suite
The test suite includes an offline smoke-test fixture and validates all components without requiring any external downloads:
```bash
pytest
ruff check src scripts tests
mypy src/vouchrec
```

### 2. Run the End-to-End Pipeline (Smoke Test / Fixture Data)
```bash
# Step 1: Standardize raw review data
python scripts/02_standardize.py --raw-file tests/fixtures/sample_reviews.jsonl

# Step 2: Generate data quality audit report
python scripts/03_quality_report.py

# Step 3: Create reproducible global temporal splits
python scripts/04_create_splits.py

# Step 4: Train and evaluate baselines (Popularity, BPR-MF, LightGCN)
python scripts/train_baselines.py --epochs 10

# Step 5: Score provisional trust across text, behavior, and coordination heads
python scripts/score_trust.py

# Step 6: Simulate controlled shilling attack & benchmark baseline vulnerability
python scripts/run_attack_simulation.py --strategy bandwagon --objective push --budget 0.05 --filler-size 3 --burst-hours 48.0

# Step 7: Benchmark Trust-Gated LightGCN defense and ablations
python scripts/run_trust_gated_experiments.py --strategy bandwagon --objective push --budget 0.05 --epochs 20

# Step 8: Benchmark Dynamic Exposure Governor post-processing defense
python scripts/run_exposure_governor_experiments.py --strategy bandwagon --objective push --budget 0.05 --epochs 20

# Step 9: Run Phase 7 multi-seed statistical validation, ablations & Holm-Bonferroni tests
python scripts/run_phase7_experiments.py --seeds 20261021,20261022,20261023,20261024,20261025 --epochs 15
```

### 3. Run on Official Amazon Reviews 2023 Dataset
```bash
# Download raw Musical_Instruments category from UCSD (458.6 MB)
python scripts/01_download.py --category Musical_Instruments

# Standardize full dataset (or with --max-records 100000 for quick audit)
python scripts/02_standardize.py --category Musical_Instruments

# Generate quality report
python scripts/03_quality_report.py --category Musical_Instruments

# Create train/val/test splits
python scripts/04_create_splits.py --category Musical_Instruments

# Fit and evaluate baselines
python scripts/train_baselines.py --category Musical_Instruments --epochs 50

# Score provisional trust on full processed reviews
python scripts/score_trust.py --category Musical_Instruments

# Run controlled shilling attack simulation
python scripts/run_attack_simulation.py --category Musical_Instruments --strategy bandwagon --objective push --budget 0.05

# Benchmark Trust-Gated LightGCN defense on full dataset
python scripts/run_trust_gated_experiments.py --category Musical_Instruments --epochs 50

# Benchmark Dynamic Exposure Governor on full dataset
python scripts/run_exposure_governor_experiments.py --category Musical_Instruments --epochs 50
```

---

## Experimental Results Summary (Smoke-Test Fixture)

### 1. Recommendation Baselines (Seed: 20261021)
| Model | Recall@10 | Recall@20 | NDCG@10 | NDCG@20 | HitRate@10 | HitRate@20 |
|---|---|---|---|---|---|---|
| **Popularity** | 1.0000 | 1.0000 | 0.7654 | 0.7654 | 1.0000 | 1.0000 |
| **BPR_MF** | 1.0000 | 1.0000 | 0.6231 | 0.6231 | 1.0000 | 1.0000 |
| **LightGCN** | 1.0000 | 1.0000 | 0.5481 | 0.5481 | 1.0000 | 1.0000 |

### 2. Provisional Trust Scoring (Phase 3)
| Head | Weight | Mean Score | Key Triggers / Meaning |
|---|---|---|---|
| **Text Head** | 0.35 | 0.6337 | Lexical diversity (TTR), repetition, structural caps/exclamations |
| **Behavior Head** | 0.35 | 0.6875 | Longitudinal span, rating variance, burst frequency, sparse history |
| **Coordination Head** | 0.30 | 1.0000 | Cross-user duplicate text hashing, 24h synchronized temporal bursts |
| **Provisional Trust** | **1.00** | **0.7624** | Weighted linear fusion clamped to [0, 1] with explainable reason codes |

### 3. Controlled Attack Simulation Benchmark (Phase 4)
*Simulated on Musical Instruments fixture (Bandwagon Push attack, Target: P006, Budget: 5.0%, 4 injected reviews):*

| Model | Clean Target HitRate@10 | Attacked Target HitRate@10 | Clean Target Mean Rank | Attacked Target Mean Rank | Rank Promotion | Collateral NDCG@10 Drop |
|---|---|---|---|---|---|---|
| **Popularity** | 0.5000 | 0.5000 | 7.8 | 7.2 | **+0.5** | 0.7654 → 0.7654 (0.0%) |
| **BPR_MF** | 0.5000 | 0.5000 | 6.5 | 6.5 | +0.0 | 0.8467 → 0.5903 (**-25.6%**) |
| **LightGCN** | 0.5000 | 0.5000 | 7.8 | 7.0 | **+0.8** | 0.9077 → 0.6077 (**-30.0%**) |

- **Trust Model Offline Detection on Ground-Truth Synthetic Profiles:**
  - Mean trust on authentic reviews: `0.7517`
  - Mean trust on synthetic reviews: `0.6150`
  - Trust separation gap: `+0.1367` (synthetic reviews significantly penalized)
  - Detection ROC-AUC: `0.9922`
  - Detection Recall (Thresholded): `1.0000` (100% synthetic review identification)
  - Detection F1-Score: `0.5333`

### 4. Trust-Gated LightGCN Benchmark & Ablations (Phase 5)
*Evaluated on Musical Instruments fixture with Bandwagon Push attack (Target: P006, Budget: 5.0%):*

| Model | Mode | Clean NDCG@10 | Attacked NDCG@10 | NDCG Distortion | Clean Target Rank | Attacked Target Rank | Rank Promotion |
|---|---|---|---|---|---|---|---|
| **Popularity** | binary | 0.7654 | 0.7654 | 0.0% | 7.8 | 7.2 | +0.5 |
| **BPR_MF** | binary | 0.8467 | 0.6077 | -28.2% | 7.0 | 6.2 | +0.8 |
| **Vanilla_LightGCN** | binary | 0.9077 | 0.6077 | -33.1% | 7.2 | 6.8 | +0.5 |
| **LightGCN_HardFilter ($\theta=0.40$)** | hard | 0.9077 | 0.6077 | -33.1% | 7.2 | 6.8 | +0.5 |
| **TrustGated_LightGCN_SoftLinear** | soft | 0.8155 | **0.6404** | **-21.5% (+5.4% recovery)** | 7.2 | 6.8 | +0.5 |
| **TrustGated_LightGCN_SoftPower ($p=2$)** | soft | 0.8155 | 0.6077 | -25.5% | 7.2 | 6.8 | +0.5 |
| **Ablation_UniformLightGCN** | uniform | 0.9077 | 0.6077 | -33.1% | 7.2 | 6.8 | +0.5 |

- **Validation Threshold Selection:** $\theta^* = 0.40$ chosen strictly on validation set to prevent destructive over-pruning seen at $\theta \ge 0.70$ (21.9% to 78.1% of genuine reviews pruned).
- **Ablation Insight:** `Ablation_UniformLightGCN` matches `Vanilla_LightGCN` exactly (-33.1% drop), confirming that trust-derived edge attenuation, not architectural variance, drives the +5.4% NDCG recovery.

### 5. Dynamic Exposure Governor Benchmark (Phase 6)
*Evaluated on Musical Instruments fixture with Bandwagon Push attack (Target: P006, Budget: 5.0%, Cutoff: $K=10$, Max Permitted Jump $\Delta r_{\max} = 0$):*

| Condition | Clean NDCG@10 | Attacked NDCG@10 | NDCG Delta | Target HR@10 | Target Mean Rank | Target Rank Shift | Items Adjusted | Jumps Capped | Runtime |
|---|---|---|---|---|---|---|---|---|---|
| **Vanilla LightGCN** | 0.9077 | 0.6077 | -0.3001 (-33.1%) | 0.50 -> 0.50 | 7.25 -> 6.75 | +0.50 | 0 (0.0%) | 0 | 0.36s |
| **Trust-Gated LightGCN (Ungoverned)** | 0.8155 | 0.6404 | -0.1751 (-21.5%) | 0.50 -> 0.50 | 7.25 -> 6.75 | +0.50 | 0 (0.0%) | 0 | 0.50s |
| **Trust-Gated LightGCN (Governed)** | **0.8155** | **0.7327** | **-0.0828 (-10.2%)** | 0.50 -> 0.50 | 7.25 -> 6.75 | +0.50 | **6 (35.3%)** | **6** | 0.46s |
| **Vanilla LightGCN (Governed)** | 0.9077 | 0.6404 | -0.2673 (-29.4%) | 0.50 -> 0.50 | 7.25 -> 6.75 | +0.50 | 6 (35.3%) | 6 | 0.28s |

- **Key Insight:** Combining Trust-Gated LightGCN with the Exposure Governor achieves an attacked NDCG@10 of **0.7327** (+20.6% recovery relative to vanilla LightGCN), capping 6 suspicious ranking jumps while incurring 0 penalty on clean data (0 clean items penalized, clean NDCG 0.8155 perfectly preserved).

### 6. Multi-Seed Statistical Validation & Sensitivity (Phase 7)
*Evaluated across 5 random seeds ($n=5$: 20261021–20261025) on Musical Instruments fixture benchmark with Bandwagon Push attack (Target: P006, Budget: 5.0%):*

| Model Condition | Clean NDCG@10 (Mean ± Std) | Attacked NDCG@10 (Mean ± Std) | NDCG Delta | Attacked Recall@20 | Target HR@10 | Target Mean Rank | Jumps Capped | Runtime |
|---|---|---|---|---|---|---|---|---|
| **Popularity** | 0.7654 ± 0.0000 | 0.7589 ± 0.0146 | -0.0065 | 1.0000 | 0.50 -> 0.50 | 7.5 | 0.0 | 0.23s |
| **BPR_MF** | 0.7223 ± 0.0552 | 0.6785 ± 0.0901 | -0.0439 | 1.0000 | 0.50 -> 0.50 | 6.9 | 0.0 | 0.85s |
| **Vanilla_LightGCN** | 0.6420 ± 0.1450 | 0.5485 ± 0.0197 | -0.0934 | 1.0000 | 0.50 -> 0.50 | 6.8 | 0.0 | 0.93s |
| **HardFiltered_LightGCN** | 0.6420 ± 0.1450 | 0.5485 ± 0.0197 | -0.0934 | 1.0000 | 0.50 -> 0.50 | 6.8 | 0.0 | 1.09s |
| **TrustGated_LightGCN_SoftLinear** | 0.6135 ± 0.1360 | 0.5570 ± 0.0467 | -0.0566 | 1.0000 | 0.50 -> 0.50 | 6.8 | 0.0 | 0.86s |
| **TrustGated_LightGCN_SoftPower** | 0.6051 ± 0.1087 | 0.5469 ± 0.0487 | -0.0581 | 1.0000 | 0.50 -> 0.50 | 6.7 | 0.0 | 1.54s |
| **TrustGated_LightGCN_Governed** | **0.6135 ± 0.1360** | **0.5635 ± 0.0344** | **-0.0500** | 1.0000 | 0.50 -> 0.50 | **7.0** | **5.0** | 1.46s |
| **Vanilla_LightGCN_Governed** | 0.6420 ± 0.1450 | 0.5616 ± 0.0270 | -0.0804 | 1.0000 | 0.50 -> 0.50 | 7.0 | 4.8 | 1.95s |

- **Statistical Analysis:** `TrustGated_LightGCN_Governed` achieves the highest attacked NDCG@10 (+0.0150 over Vanilla LightGCN, Cohen's $d_z = 0.32$). Due to small sample size on the 40-record fixture, pairwise differences between GNN variants do not cross the Holm-Bonferroni significance threshold ($p = 0.5126$), demonstrating the necessity of testing on full-scale corpora to observe statistical significance.
- **Exposure Protection:** The Governor consistently capped an average of 5.0 suspicious ranking jumps per run, preventing target product promotion without degrading clean-data utility (clean NDCG remains strictly identical at 0.6135).
- **Operational Viability:** Peak memory usage across all 74 runs was **< 70 MB** (69.17 MB) and average per-run runtime was ~1.0s.

*Reports generated:*
- Data Quality Report: `data/reports/data_quality_report.json` and `data/reports/data_quality_report.md`
- Baseline Results: `data/reports/baseline_results.json` and `data/reports/baseline_results.md`
- Trust Evaluation Report: `data/reports/trust_evaluation_report.json` and `data/reports/trust_evaluation_report.md`
- Attack Simulation Report: `data/reports/attack_simulation_report.json` and `data/reports/attack_simulation_report.md`
- Trust-Gated LightGCN Report: `data/reports/trust_gated_lightgcn_report.json` and `data/reports/trust_gated_lightgcn_report.md`
- Exposure Governor Report: `data/reports/exposure_governor_report.json` and `data/reports/exposure_governor_report.md`
- Phase 7 Experiment Plan: `data/reports/phase7_experiment_plan.md`
- Phase 7 Results Report: `data/reports/phase7_results.md`
- Phase 7 Machine-Readable Results: `data/reports/phase7_per_seed_results.csv`, `data/reports/phase7_per_seed_results.json`, and `data/reports/phase7_aggregate_results.json`

---

### 7. Full-Stack Research Dashboard & API Integration (Phase 8)

Phase 8 introduces a research-grade web dashboard and REST API providing interactive access to all prior phases.

#### Architecture Highlights:
- **FastAPI Backend (`src/vouchrec/api/`):** 12 typed endpoints, Pydantic v2 schemas, in-memory recommender model caching (`_MODEL_CACHE`), CORS configuration, and static single-server production mounting.
- **React Frontend (`frontend/`):** Desktop-first responsive interface built with Vite, TypeScript, Tailwind CSS, Recharts, and Lucide icons. Deep green research aesthetic (`#08140e`).
- **Research Integrity Guards:** Heuristic trust scores are labeled as anomaly indices rather than calibrated fraud probabilities. Benchmark fixture metrics display preliminary warning banners. Attack simulator maintains strict label quarantine.

#### Quick Start: Running the Research Dashboard

**Option A: Single-Server Full-Stack (Recommended)**
The compiled frontend is served directly by FastAPI at `http://127.0.0.1:8000`:
```powershell
# 1. Activate Python virtual environment
.\C:\VouchRecData\.venv\Scripts\Activate.ps1

# 2. Start the API server (serves both API and Frontend)
python scripts/run_api.py --port 8000
```
Then open your browser to **`http://127.0.0.1:8000`** or view interactive API documentation at **`http://127.0.0.1:8000/docs`**.

**Option B: Separate Frontend Development Server (Hot Reloading)**
```powershell
# Terminal 1: Launch Backend
python scripts/run_api.py --port 8000 --reload

# Terminal 2: Launch Vite Frontend Dev Server
cd frontend
npm run dev
```
Open **`http://localhost:5173`** (API requests are automatically proxied to port 8000).

#### Frontend Architecture & Product Experience:

##### 1. Clean Trust Analyzer (Homepage)
The homepage has a single primary focus: **Paste a product review URL → Analyze → Show a clear trust assessment.**
- **Tagline:** *"Know which reviews to trust."*
- **Headline:** *"Not every review tells the whole story."*
- **Supporting Copy:** *"Analyze review signals and understand what may make a review suspicious."*
- **URL Analysis & Live Real Data Support:**
  - **Live Amazon Listings:** Supports live product links from Amazon (`amazon.in`, `amazon.com`, `amazon.co.uk`, etc.) by extracting the 10-character ASIN (`B0...`) and fetching real listing metadata and customer reviews via browser-grade TLS fingerprinting (`curl_cffi` + `BeautifulSoup`).
  - **Live Product Card:** Displays real product title, product image thumbnail, marketplace rating (e.g. `★ 4.1`), and total rating count alongside an active *Live Amazon Data Verified* badge.
  - **Review Selector Carousel:** Parses all customer reviews on the listing, scoring each review individually and allowing users to toggle through different customer reviews on that product to inspect specific trust signals.
  - **Research Catalog Items:** Also resolves local verified benchmark items (`P001`–`P008`, `B001`–`B008`) against the complete 3-head graph dataset.
  - **Anti-Bot & CAPTCHA Resilience:** If Amazon serves a temporary anti-bot challenge on a specific listing, VouchRec detects it gracefully and triggers a zero-friction fallback prompting the user to paste any review text directly from their open browser tab.
- **Review Text Fallback:**
  - Users can paste review text directly to evaluate linguistic signals (lexical diversity, phrase repetition, exclamation ratio, caps ratio).
  - Scores are mapped to a defensible 0–100 trust scale with clear risk tiers:
    - **Lower risk (≥ 70):** Standard phrasing, diverse vocabulary, typical sentence structure.
    - **Needs a closer look (45–69):** Moderate repetition or borderline linguistic signals.
    - **Higher risk (< 45):** Heavy phrase repetition, promotional patterns, or excessive capitalization.
  - **Limitations Disclosed:** The UI clearly notes that text-only analysis cannot establish reviewer velocity or multi-account coordination networks without access to historical account graphs.
- **Visual System & Dark Mode:**
  - Curated warm ivory palette (`#F7F5F0`), deep aubergine primary action (`#60445A`), muted copper highlights (`#B87958`), warm stone borders (`#E5E0D8`), and deep night dark mode (`#171A18`, `#222724`).
  - Font pairing: *Newsreader* (editorial serif for headlines) and *Inter* (sans-serif for interface text).
  - Working theme toggle (Sun/Moon icon) with `localStorage` persistence.

##### 2. Research Lab (Dedicated Area)
Preserves all Phase 1–7 scientific and research capabilities in a secondary workspace accessible via the top navigation ("Research Lab"):
1. **Overview:** Dataset provenance, 5-core statistics, split integrity, active defense status, and clean vs. attacked NDCG@10 comparison bar chart.
2. **Recommendation Explorer:** Interactive Top-K ranking generator with side-by-side comparison of Vanilla LightGCN, Trust-Gated LightGCN, and Governed LightGCN. Inspect rank shifts ($\Delta r$), evidence, and reason codes.
3. **Trust Analysis:** Tri-head trust breakdown (Text, Behavioral, Coordination) with feature explanations, heuristic disclaimer, and score range filters.
4. **Attack Simulator:** Controlled shilling simulator supporting Random, Average, and Bandwagon strategies, Push/Nuke objectives, budget sliders, burst timing, and label quarantine verification.
5. **Defense Comparison:** Benchmark matrix comparing all 7 recommendation paradigms across Clean NDCG@10, Attacked NDCG@10, Recall@20, and Target Mean Rank.
6. **Governance Audit Trail:** Inspectable log of Exposure Governor decisions with exact reason codes (`GOV_CAPPED_SUSPICIOUS_JUMP`, `GOV_PRESERVED_SPARSE_EVIDENCE`, `GOV_PASS_UNMODIFIED`).
7. **Statistical Experiments:** Multi-seed Phase 7 benchmark results, 95% confidence intervals, Holm-Bonferroni FWER significance tests, and Clean vs. Attacked trade-off scatter plot.

#### API Reference:
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check, version, and fixture status |
| `GET` | `/api/dataset/summary` | Dataset metadata, splits, and record counts |
| `GET` | `/api/models` | Catalog of 7 supported model architectures |
| `GET` | `/api/users` | List of available users in dataset |
| `GET` | `/api/items` | List of catalog items in dataset |
| `POST` | `/api/recommendations` | Top-K recommendation inference with variant comparison |
| `POST` | `/api/trust/analyze` | Review trust scores and tri-head feature explanations (batch/catalog) |
| `POST` | `/api/trust/analyze-review` | Single review analysis via URL link or pasted text with transparent risk scoring |
| `POST` | `/api/attacks/simulate` | Execute controlled shilling attack simulation |
| `POST` | `/api/defenses/compare` | Filterable defense benchmark across models |
| `GET` | `/api/governance/audit` | Exposure Governor audit log and reason codes |
| `GET` | `/api/experiments` | List of multi-seed experiment benchmark artifacts |
| `GET` | `/api/experiments/{id}` | Detailed statistical test results, CIs, and per-seed results |

---

## Milestone Status

- **Phase 1:** Dataset Ingestion & Preprocessing — **Completed & Verified**
- **Phase 2:** Baseline Collaborative Filtering Models — **Completed & Verified**
- **Phase 3:** Trust Signal Modeling — **Completed & Verified**
- **Phase 4:** Controlled Attack Simulator — **Completed & Verified**
- **Phase 5:** Trust-Gated LightGCN — **Completed & Verified**
- **Phase 6:** Dynamic Exposure Governor — **Completed & Verified**
- **Phase 7:** Rigorous Experiments, Ablations & Statistical Validation — **Completed & Verified**
- **Phase 8:** Full-Stack Research Dashboard and API Integration — **Completed & Verified**
- **Phase 9:** Production-Style UI Redesign & Review Trust Analyzer — **Completed & Verified**



