"""End-to-end integration smoke test from raw fixture to recommendation baselines."""

from pathlib import Path

from vouchrec.config import load_config
from vouchrec.data.quality_report import generate_quality_report
from vouchrec.data.standardize import standardize_records
from vouchrec.evaluation.metrics import evaluate_recommendations
from vouchrec.evaluation.splits import create_global_temporal_split
from vouchrec.models.bpr_mf import BPRMFRecommender
from vouchrec.models.lightgcn import LightGCNRecommender
from vouchrec.models.popularity import PopularityRecommender


def test_end_to_end_smoke_pipeline(tmp_path: Path) -> None:
    fixture_path = Path(__file__).parent / "fixtures" / "sample_reviews.jsonl"
    prep_cfg = load_config("preprocessing")

    # 1. Standardize
    interim_table, proc_table, stats = standardize_records(
        raw_file_path=fixture_path,
        source_dataset="amazon_reviews_2023",
        source_category="Musical_Instruments",
        source_split=None,
        preprocessing_cfg=prep_cfg,
    )
    assert len(interim_table) == 46
    assert len(proc_table) == 40

    # 2. Quality report
    interim_path = tmp_path / "interim.parquet"
    proc_path = tmp_path / "proc.parquet"
    import pyarrow.parquet as pq

    pq.write_table(interim_table, interim_path)
    pq.write_table(proc_table, proc_path)

    report = generate_quality_report(interim_path, proc_path)
    assert report["volume"]["processed_records"] == 40

    # 3. Split
    proc_df = proc_table.to_pandas()
    train_df, val_df, test_df, meta = create_global_temporal_split(
        proc_df, val_fraction=0.1, test_fraction=0.1
    )
    assert len(train_df) == 32
    assert len(val_df) == 4
    assert len(test_df) == 4

    # 4. Fit Baselines
    pop = PopularityRecommender().fit(train_df)
    bpr = BPRMFRecommender(embedding_dim=16, epochs=5, batch_size=16, seed=42).fit(train_df)
    lgcn = LightGCNRecommender(
        embedding_dim=16, num_layers=2, epochs=5, batch_size=16, seed=42
    ).fit(train_df)

    # 5. Evaluate on test users present in train
    from collections import defaultdict

    gt: dict[str, set[str]] = defaultdict(set)
    for u, i in zip(test_df["user_id"], test_df["item_id"], strict=False):
        gt[u].add(i)

    train_users = set(train_df["user_id"])
    eval_users = [u for u in gt if u in train_users]
    assert len(eval_users) > 0

    for model in [pop, bpr, lgcn]:
        recs = model.recommend(eval_users, k=20, exclude_seen=True)
        res = evaluate_recommendations(recs, gt, cutoffs=(10, 20))
        assert 0.0 <= res["Recall@20"] <= 1.0
        assert 0.0 <= res["NDCG@10"] <= 1.0
        assert 0.0 <= res["HitRate@10"] <= 1.0
