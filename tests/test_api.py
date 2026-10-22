"""Unit and integration tests for VouchRec FastAPI endpoints."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from vouchrec.api.app import app

client = TestClient(app)


def test_api_health() -> None:
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["dataset_category"] == "Musical_Instruments"
    assert data["is_fixture"] is True
    assert "timestamp_utc" in data


def test_dataset_summary() -> None:
    resp = client.get("/api/dataset/summary?category=Musical_Instruments")
    assert resp.status_code == 200
    data = resp.json()
    assert data["category"] == "Musical_Instruments"
    assert data["total_reviews"] == 40
    assert data["total_users"] == 8
    assert data["total_items"] == 8
    assert data["train_interactions"] == 32
    assert data["val_interactions"] == 4
    assert data["test_interactions"] == 4
    assert data["is_fixture"] is True
    assert "Preliminary" in data["provenance_note"]


def test_models_catalog() -> None:
    resp = client.get("/api/models")
    assert resp.status_code == 200
    data = resp.json()
    models = data["models"]
    assert len(models) >= 7
    ids = [m["id"] for m in models]
    assert "Popularity" in ids
    assert "BPR_MF" in ids
    assert "Vanilla_LightGCN" in ids
    assert "TrustGated_LightGCN_SoftLinear" in ids
    assert "TrustGated_LightGCN_Governed" in ids


def test_users_and_items_endpoints() -> None:
    u_resp = client.get("/api/users")
    assert u_resp.status_code == 200
    u_data = u_resp.json()
    assert u_data["total_users"] == 8
    assert len(u_data["users"]) == 8

    i_resp = client.get("/api/items")
    assert i_resp.status_code == 200
    i_data = i_resp.json()
    assert i_data["total_items"] == 8
    assert len(i_data["items"]) == 8


def test_recommendations_valid_and_invalid_user() -> None:
    # 1. Valid user
    req_body = {
        "user_id": "U01",
        "model_id": "TrustGated_LightGCN_Governed",
        "k": 5,
        "compare_variants": True,
        "max_rank_jump": 0,
    }
    resp = client.post("/api/recommendations", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["user_id"] == "U01"
    assert data["model_id"] == "TrustGated_LightGCN_Governed"
    assert len(data["recommendations"]) <= 5
    for rec in data["recommendations"]:
        assert "item_id" in rec
        assert "original_rank" in rec
        assert "governed_rank" in rec
        assert "reason_code" in rec

    assert data["comparisons"] is not None
    assert "Vanilla_LightGCN" in data["comparisons"]

    # 2. Invalid user
    invalid_req = {
        "user_id": "NON_EXISTENT_USER",
        "model_id": "TrustGated_LightGCN_Governed",
        "k": 5,
    }
    err_resp = client.post("/api/recommendations", json=invalid_req)
    assert err_resp.status_code == 400
    assert "not found" in err_resp.json()["detail"].lower()


def test_trust_analysis() -> None:
    req_body = {
        "min_trust": 0.0,
        "max_trust": 1.0,
        "limit": 10,
    }
    resp = client.post("/api/trust/analyze", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_scored"] > 0
    assert "mean_trust" in data
    assert len(data["reviews"]) <= 10
    rev = data["reviews"][0]
    assert "provisional_trust" in rev
    assert "text_score" in rev
    assert "behavior_score" in rev
    assert "coordination_score" in rev
    assert "reasons" in rev


def test_attack_simulation_endpoint() -> None:
    # 1. Valid simulation
    req_body = {
        "strategy": "bandwagon",
        "objective": "push",
        "budget_fraction": 0.05,
        "filler_size": 3,
        "burst_window_hours": 48.0,
        "seed": 20261021,
    }
    resp = client.post("/api/attacks/simulate", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["synthetic_profiles_count"] == 1
    assert data["synthetic_reviews_count"] == 4
    assert data["target_item"] == "P006"
    assert data["budget_rounding_warning"] is not None
    assert "impact_metrics" in data
    assert data["label_quarantine_verified"] is True

    # 2. Invalid strategy schema rejection
    bad_req = req_body.copy()
    bad_req["strategy"] = "invalid_strategy"
    bad_resp = client.post("/api/attacks/simulate", json=bad_req)
    assert bad_resp.status_code == 422


def test_defenses_comparison_endpoint() -> None:
    req_body = {"strategy": "bandwagon", "objective": "push", "budget": 0.05}
    resp = client.post("/api/defenses/compare", json=req_body)
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["models"]) >= 7
    m_names = [m["model_name"] for m in data["models"]]
    assert "TrustGated_LightGCN_Governed" in m_names
    assert "Vanilla_LightGCN" in m_names


def test_governance_audit_endpoint() -> None:
    resp = client.get("/api/governance/audit?limit=20")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_evaluated"] > 0
    assert "reason_code_counts" in data
    assert len(data["records"]) > 0
    rec = data["records"][0]
    assert "user_id" in rec
    assert "item_id" in rec
    assert "reason_code" in rec


def test_experiments_list_and_detail() -> None:
    resp = client.get("/api/experiments")
    assert resp.status_code == 200
    data = resp.json()
    exps = data["experiments"]
    assert len(exps) >= 1
    exp_id = exps[0]["experiment_id"]

    det_resp = client.get(f"/api/experiments/{exp_id}")
    assert det_resp.status_code == 200
    det_data = det_resp.json()
    assert det_data["experiment_id"] == exp_id
    assert "manifest" in det_data
    assert "model_aggregates" in det_data
    assert "statistical_tests" in det_data
    assert len(det_data["per_seed_results"]) > 0

    # 404 for invalid experiment
    not_found = client.get("/api/experiments/non_existent_exp")
    assert not_found.status_code == 404


def test_static_frontend_serving() -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "VouchRec" in resp.text


def test_analyze_review_pasted_text_authentic() -> None:
    body = {
        "review_text": (
            "This instrument exceeded my expectations. The fretwork is smooth, "
            "tuning pegs hold tension nicely, and the sound is warm and balanced for the price."
        )
    }
    resp = client.post("/api/trust/analyze-review", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["status"] == "analyzed"
    assert data["source_type"] == "pasted_text"
    assert data["trust_score"] is not None
    assert data["trust_score"] >= 65
    assert data["risk_level"] in ["Lower risk", "Needs a closer look"]
    assert data["text_signals"]["status"] == "evaluated"
    assert data["behavior_signals"]["status"] == "insufficient_data"
    assert data["coordination_signals"]["status"] == "insufficient_data"
    assert len(data["limitations"]) > 0


def test_analyze_review_pasted_text_suspicious() -> None:
    body = {
        "review_text": "GREAT GREAT GREAT BUY BUY BUY SO GOOD AMAZING AMAZING BEST EVER BEST EVER"
    }
    resp = client.post("/api/trust/analyze-review", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["status"] == "analyzed"
    assert data["trust_score"] < 70
    assert data["risk_level"] in ["Needs a closer look", "Higher risk"]
    assert any("REPETITION" in r or "DIVERSITY" in r for r in data["text_signals"]["reasons"])


def test_analyze_review_dataset_url() -> None:
    body = {"url": "https://www.amazon.com/dp/P001"}
    resp = client.post("/api/trust/analyze-review", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["status"] == "analyzed"
    assert data["source_type"] == "url_dataset"
    assert data["item_id"] == "P001"
    assert data["trust_score"] is not None
    assert data["text_signals"]["status"] == "evaluated"
    assert data["behavior_signals"]["status"] == "evaluated"
    assert data["coordination_signals"]["status"] == "evaluated"


def test_analyze_review_unsupported_external_url() -> None:
    body = {"url": "https://www.amazon.com/dp/B08XYZ123456"}
    resp = client.post("/api/trust/analyze-review", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert data["status"] == "unsupported_url"
    assert data["suggested_action"] == "paste_text"
    assert "paste" in data["error_message"].lower()


def test_analyze_review_empty_input() -> None:
    resp = client.post("/api/trust/analyze-review", json={})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert data["status"] == "invalid_input"


def test_extract_asin_and_domain() -> None:
    from vouchrec.api.scraper import extract_asin_and_domain

    asin, dom = extract_asin_and_domain(
        "https://www.amazon.in/soundcore-Cancelling-Headphones-Bluetooth-Transparency/dp/B0C3HCD34R?ref=sr_1_1"
    )
    assert asin == "B0C3HCD34R"
    assert dom == "amazon.in"

    asin2, dom2 = extract_asin_and_domain("https://www.amazon.com/dp/B08N5WRWNW")
    assert asin2 == "B08N5WRWNW"
    assert dom2 == "amazon.com"

    asin3, _ = extract_asin_and_domain("B0CT3SNDP9")
    assert asin3 == "B0CT3SNDP9"


def test_analyze_review_live_amazon_mocked(monkeypatch: pytest.MonkeyPatch) -> None:
    from vouchrec.api import services

    mock_scraped = {
        "success": True,
        "status": "fetched",
        "asin": "B0C3HCD34R",
        "domain": "amazon.in",
        "canonical_url": "https://www.amazon.in/dp/B0C3HCD34R",
        "product_title": "soundcore by Anker Q20i Wireless Headphones",
        "product_image": "https://m.media-amazon.com/images/I/soundcore.jpg",
        "product_rating": 4.1,
        "total_reviews_text": "(77,241)",
        "reviews_count": 2,
        "reviews": [
            {
                "author": "Ronak Khandelwal",
                "rating": 5.0,
                "title": "One of the best headphones.",
                "date": "Reviewed in India on 8 February 2026",
                "verified": True,
                "body": "For the budget under 5k, this headphone is BEST. Writing after using it for a month. ANC is good.",
            },
            {
                "author": "Jagannathan Rajan",
                "rating": 4.0,
                "title": "Good enough",
                "date": "Reviewed in India on 18 October 2026",
                "verified": True,
                "body": "Very good audio quality, ANC is 6/10, but the ear cups themselves provide good isolation.",
            },
        ],
        "is_captcha": False,
    }

    monkeypatch.setattr(services, "fetch_live_amazon_product", lambda url: mock_scraped)

    body = {"url": "https://www.amazon.in/soundcore-Cancelling-Headphones-Bluetooth-Transparency/dp/B0C3HCD34R"}
    resp = client.post("/api/trust/analyze-review", json=body)
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["status"] == "analyzed"
    assert data["source_type"] == "live_amazon"
    assert data["asin"] == "B0C3HCD34R"
    assert "soundcore" in data["product_title"].lower()
    assert len(data["all_reviews"]) == 2
    assert data["trust_score"] is not None
    assert data["text_signals"]["status"] == "evaluated"
    assert data["behavior_signals"]["status"] == "evaluated"
    assert data["coordination_signals"]["status"] == "evaluated"
    # Verify score breakdown and narrative
    assert "score_breakdown" in data
    assert len(data["score_breakdown"]) > 0
    assert any(f["name"] == "Base Prior" for f in data["score_breakdown"])
    assert "score_narrative" in data
    assert f"Score: {data['trust_score']}/100" in data["score_narrative"]
    # Check that individual reviews in all_reviews also have their own score breakdown
    for rev in data["all_reviews"]:
        assert "score_breakdown" in rev
        assert "score_narrative" in rev
        assert rev["trust_score"] is not None


def test_score_breakdown_and_narrative_transparency() -> None:
    # 1. Authentic detailed review
    auth_resp = client.post(
        "/api/trust/analyze-review",
        json={
            "review_text": (
                "This instrument exceeded my expectations. The fretwork is smooth along the entire neck, "
                "tuning pegs hold tension nicely over multiple sessions, and the tone is surprisingly warm and balanced."
            ),
            "rating": 5,
        },
    )
    assert auth_resp.status_code == 200
    auth_data = auth_resp.json()
    assert auth_data["trust_score"] >= 75
    assert auth_data["risk_level"] == "Lower risk"
    assert len(auth_data["score_breakdown"]) >= 3
    # Check that Base Prior exists
    assert any(f["name"] == "Base Prior" and f["points"] == 50 for f in auth_data["score_breakdown"])
    # Check that vocabulary and length bonuses were awarded
    assert any("Vocabulary" in f["name"] or "Word" in f["name"] for f in auth_data["score_breakdown"])
    assert "Lower risk" in auth_data["score_narrative"]

    # 2. Suspicious repetitive shouting review
    susp_resp = client.post(
        "/api/trust/analyze-review",
        json={
            "review_text": "GREAT GREAT GREAT BUY BUY BUY SO GOOD AMAZING AMAZING BEST EVER BEST EVER BEST BEST BEST",
            "rating": 5,
        },
    )
    assert susp_resp.status_code == 200
    susp_data = susp_resp.json()
    assert susp_data["trust_score"] <= 45
    assert susp_data["risk_level"] in ["Needs a closer look", "Higher risk"]
    # Check that penalties exist in breakdown
    penalties = [f for f in susp_data["score_breakdown"] if f["category"] == "penalty" or f["points"] < 0]
    assert len(penalties) > 0
    assert "anomalies detected" in susp_data["score_narrative"].lower() or "deductions" in susp_data["score_narrative"].lower()



