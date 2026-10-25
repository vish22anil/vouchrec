"""Comprehensive unit and anti-leakage tests for Phase 3 trust signal modeling."""

import pandas as pd
import pytest

from vouchrec.data.schema import SchemaError
from vouchrec.trust.behavior_signals import ReviewerBehaviorScorer, extract_behavior_features
from vouchrec.trust.coordination_signals import CoordinationScorer
from vouchrec.trust.fusion import ProvisionalTrustFusion
from vouchrec.trust.text_signals import (
    DebertaTextScorer,
    UnsupervisedTextScorer,
    extract_text_features,
)

# ==============================================================================
# 1. Review-Text Signal Head Tests
# ==============================================================================


def test_text_feature_extraction_normal() -> None:
    texts = [
        "These guitar strings sounded incredible on my acoustic guitar. Bright and resonant tone.",
        "",
        None,
        "Great! Great! Great! Great!",
        "BUY NOW BEST EVER WOW WWW",
    ]
    feats = extract_text_features(texts)

    assert len(feats) == 5
    # Row 0: Normal review
    assert feats.loc[0, "word_count"] > 10
    assert 0.0 < feats.loc[0, "type_token_ratio"] <= 1.0
    assert not feats.loc[0, "is_missing_text"]
    assert not feats.loc[0, "is_short_text"]

    # Row 1: Empty text
    assert feats.loc[1, "is_missing_text"]
    assert feats.loc[1, "word_count"] == 0

    # Row 2: None
    assert feats.loc[2, "is_missing_text"]

    # Row 3: High repetition
    assert feats.loc[3, "repetition_ratio"] > 0.50

    # Row 4: All caps
    assert feats.loc[4, "caps_ratio"] > 0.70


def test_unsupervised_text_scorer() -> None:
    scorer = UnsupervisedTextScorer()
    texts = [
        "These strings produce a wonderfully clear and warm acoustic resonance that lasts for months.",
        "",
        "GREAT AMAZING BEST GUITAR STRINGS EVER BUY NOW!!!!!",
        "Strings ok",
        "Good good good good good good good good good good",
    ]
    scores, reasons = scorer.score(texts)

    assert len(scores) == 5
    assert all(0.0 <= s <= 1.0 for s in scores)

    # Normal detailed review should score high
    assert scores[0] >= 0.80
    assert "TEXT_DETAILED_AND_DIVERSE" in reasons[0]

    # Empty text receives neutral 0.50 prior
    assert scores[1] == 0.50
    assert "TEXT_MISSING" in reasons[1]

    # Excessive caps/exclamation should trigger penalties
    assert scores[2] < scores[0]
    assert any("CAPS" in r or "PUNCTUATION" in r for r in reasons[2])

    # Short text
    assert "TEXT_SHORT" in reasons[3]

    # Extreme repetition
    assert any("REPETITION" in r or "LOW_LEXICAL" in r for r in reasons[4])


def test_deberta_stub_raises() -> None:
    stub = DebertaTextScorer()
    with pytest.raises(NotImplementedError) as exc_info:
        stub.score(["Sample text"])
    assert "DeBERTa text classifier requires fine-tuning" in str(exc_info.value)


# ==============================================================================
# 2. Reviewer-Behavior Signal Head Tests
# ==============================================================================


def test_behavior_sparse_history_neutral_prior() -> None:
    # A reviewer with only 1 interaction should get a neutral score
    df = pd.DataFrame(
        {
            "user_id": ["u_new"],
            "item_id": ["item_1"],
            "rating": [5.0],
            "timestamp": ["2021-01-01T00:00:00Z"],
        }
    )
    scorer = ReviewerBehaviorScorer()
    scores, reasons = scorer.score(df)

    assert scores[0] == 0.50
    assert "BEHAVIOR_SPARSE_HISTORY" in reasons[0]


def test_behavior_monotone_rating_penalty() -> None:
    # Reviewer with 5 consecutive 5-star reviews on the same day
    df = pd.DataFrame(
        {
            "user_id": ["u_monotone"] * 5,
            "item_id": [f"item_{i}" for i in range(5)],
            "rating": [5.0] * 5,
            "timestamp": [f"2021-01-0{i + 1}T00:00:00Z" for i in range(5)],
        }
    )
    scorer = ReviewerBehaviorScorer()
    scores, reasons = scorer.score(df)

    # Later reviews have >=3 identical ratings -> triggers monotone rating penalty
    assert "BEHAVIOR_MONOTONE_RATING" in reasons[4]
    assert scores[4] < 0.65


def test_behavior_anti_leakage_temporal_causality() -> None:
    # User u1 has 2 reviews in 2020 and 5 reviews in 2021.
    # When scoring the 2020 review, the history count should NOT see the 2021 reviews!
    df = pd.DataFrame(
        {
            "user_id": ["u1", "u1", "u1"],
            "item_id": ["i1", "i2", "i3"],
            "rating": [4.0, 5.0, 3.0],
            "timestamp": ["2020-01-01T00:00:00Z", "2020-06-01T00:00:00Z", "2022-01-01T00:00:00Z"],
        }
    )
    feats = extract_behavior_features(df)
    # Review 0: count = 1
    assert feats.loc[0, "history_count"] == 1
    # Review 1: count = 2
    assert feats.loc[1, "history_count"] == 2
    # Review 2: count = 3
    assert feats.loc[2, "history_count"] == 3


# ==============================================================================
# 3. Coordination Signal Head Tests
# ==============================================================================


def test_coordination_cross_user_exact_duplicate() -> None:
    # Distinct users posting identical text on different items
    df = pd.DataFrame(
        {
            "user_id": ["user_a", "user_b", "user_c"],
            "item_id": ["item_1", "item_2", "item_3"],
            "rating": [5.0, 5.0, 4.0],
            "review_text": [
                "These strings produce a wonderfully clear and warm acoustic resonance.",
                "These strings produce a wonderfully clear and warm acoustic resonance.",  # Exact dup across users
                "Independent completely unique review body text here.",
            ],
            "timestamp": ["2021-01-01T00:00:00Z"] * 3,
        }
    )
    scorer = CoordinationScorer()
    scores, reasons = scorer.score(df)

    # user_a and user_b should trigger cross-user duplicate text penalty
    assert any("COORD_CROSS_USER_DUPLICATE_TEXT" in r for r in reasons[0])
    assert any("COORD_CROSS_USER_DUPLICATE_TEXT" in r for r in reasons[1])
    assert scores[0] <= 0.65
    assert scores[1] <= 0.65

    # user_c has independent text
    assert "COORD_INDEPENDENT" in reasons[2]
    assert scores[2] == 1.0


def test_coordination_item_temporal_burst() -> None:
    # 5 reviews on item_1 within 2 hours with all 5-star ratings
    df = pd.DataFrame(
        {
            "user_id": [f"user_{i}" for i in range(5)],
            "item_id": ["item_1"] * 5,
            "rating": [5.0] * 5,
            "review_text": [f"Review number {i} is unique" for i in range(5)],
            "timestamp": [f"2021-05-01T0{i}:00:00Z" for i in range(5)],
        }
    )
    scorer = CoordinationScorer()
    scores, reasons = scorer.score(df)

    # Burst of >=4 reviews with 100% same rating within 24h triggers penalty
    for i in range(5):
        assert any("COORD_SYNCHRONIZED_RATING_BURST" in r for r in reasons[i])
        assert scores[i] <= 0.75


# ==============================================================================
# 4. Fusion and Anti-Leakage Guard Tests
# ==============================================================================


def test_fusion_bounds_and_inspectability() -> None:
    df = pd.DataFrame(
        {
            "record_id": ["rec_1", "rec_2"],
            "user_id": ["u1", "u2"],
            "item_id": ["i1", "i2"],
            "rating": [5.0, 4.0],
            "review_text": [
                "Excellent acoustic guitar strings with vibrant sound and good durability.",
                "",
            ],
            "timestamp": ["2021-01-01T00:00:00Z", "2021-01-02T00:00:00Z"],
            "verified_purchase": [True, False],
        }
    )

    fusion = ProvisionalTrustFusion()
    result = fusion.score_reviews(df)
    out_df = result.scores_df

    # 1. Bounds check
    assert (out_df["provisional_trust_score"] >= 0.0).all()
    assert (out_df["provisional_trust_score"] <= 1.0).all()

    # 2. Inspectable component heads
    assert "text_trust_score" in out_df.columns
    assert "behavior_trust_score" in out_df.columns
    assert "coordination_trust_score" in out_df.columns
    assert "provisional_trust_score" in out_df.columns
    assert "trust_confidence" in out_df.columns
    assert "trust_reasons" in out_df.columns

    # 3. Missing text lowers confidence
    assert out_df.loc[1, "trust_confidence"] < out_df.loc[0, "trust_confidence"]


def test_anti_leakage_label_rejection() -> None:
    # Providing a forbidden label column must immediately raise SchemaError
    forbidden_columns = ["is_attack", "label_value", "generator_model", "attack_type"]

    for col in forbidden_columns:
        df = pd.DataFrame(
            {
                "user_id": ["u1"],
                "item_id": ["i1"],
                "rating": [5.0],
                "review_text": ["Sample text"],
                col: [1],
            }
        )
        fusion = ProvisionalTrustFusion()
        with pytest.raises(SchemaError):
            fusion.score_reviews(df)


def test_promotional_boilerplate_detection() -> None:
    scorer = UnsupervisedTextScorer()
    shilling_text = "Outstanding product! High quality build, exceeded all my expectations."
    natural_text = "These strings produce a wonderfully clear and warm acoustic resonance that lasts for months."

    scores, reasons = scorer.score([shilling_text, natural_text])
    assert "TEXT_PROMOTIONAL_BOILERPLATE" in reasons[0]
    assert "TEXT_DETAILED_AND_DIVERSE" not in reasons[0]
    assert "TEXT_DETAILED_AND_DIVERSE" in reasons[1]
    assert scores[0] < scores[1]


def test_supervised_text_trust_classifier() -> None:
    from vouchrec.trust.text_signals import SupervisedTextTrustClassifier

    classifier = SupervisedTextTrustClassifier()
    # Unfitted falls back gracefully to unsupervised scorer
    scores_unfit, _ = classifier.score(["Testing fallback"])
    assert len(scores_unfit) == 1

    train_texts = [
        "These strings produce warm acoustic tone and durability.",
        "Accurate clip-on tuner with clear display.",
        "Five stars all the way! Best purchase I have made!",
        "Exceeded all my expectations! Works flawlessly out of the box!",
    ]
    train_labels = [0, 0, 1, 1]  # 0 = authentic, 1 = synthetic
    classifier.fit(train_texts, train_labels)

    scores_fit, reasons_fit = classifier.score(
        [
            "Warm acoustic resonance and great strings.",
            "Best purchase ever! Five stars all the way!",
        ]
    )
    assert scores_fit[0] > scores_fit[1]
    assert "ML_CLASSIFIER_AUTHENTIC" in reasons_fit[0]
    assert "ML_CLASSIFIER_SUSPICIOUS" in reasons_fit[1]


def test_unverified_burst_penalty() -> None:
    # User with 3 rapid reviews, span < 1 day, 0 verified purchase
    df = pd.DataFrame(
        {
            "user_id": ["u_bot"] * 3,
            "item_id": ["item_1", "item_2", "item_3"],
            "rating": [5.0, 4.0, 5.0],
            "timestamp": ["2021-01-01T00:00:00Z", "2021-01-01T01:00:00Z", "2021-01-01T02:00:00Z"],
            "verified_purchase": [False, False, False],
        }
    )
    scorer = ReviewerBehaviorScorer()
    scores, reasons = scorer.score(df)
    assert "BEHAVIOR_UNVERIFIED_BURST" in reasons[2]
