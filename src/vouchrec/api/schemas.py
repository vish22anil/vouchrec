"""Pydantic request and response schemas for the VouchRec API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "0.1.0"
    dataset_category: str = "Musical_Instruments"
    is_fixture: bool = True
    total_reviews_available: int = 40
    timestamp_utc: str


class DatasetSummaryResponse(BaseModel):
    category: str
    total_reviews: int
    total_users: int
    total_items: int
    train_interactions: int
    val_interactions: int
    test_interactions: int
    is_fixture: bool
    data_root: str
    raw_file_available: bool
    provenance_note: str


class ModelInfo(BaseModel):
    id: str
    name: str
    family: str  # baseline, graph, trust_gated, governed
    description: str
    is_available: bool = True


class ModelListResponse(BaseModel):
    models: list[ModelInfo]


class UsersListResponse(BaseModel):
    users: list[dict[str, Any]]
    total_users: int


class ItemsListResponse(BaseModel):
    items: list[dict[str, Any]]
    total_items: int


class RecommendedItem(BaseModel):
    item_id: str
    score: float
    original_rank: int
    governed_rank: int
    baseline_rank: int | None = None
    rank_shift: int
    was_adjusted: bool
    reason_code: str
    n_interactions: int = 0
    n_low_trust: int = 0
    suspicious_ratio: float = 0.0
    mean_trust: float = 0.50


class RecommendationRequest(BaseModel):
    user_id: str
    model_id: str = "TrustGated_LightGCN_Governed"
    k: int = Field(default=10, ge=1, le=50)
    compare_variants: bool = True
    max_rank_jump: int = Field(default=0, ge=0)


class RecommendationResponse(BaseModel):
    user_id: str
    model_id: str
    k: int
    recommendations: list[RecommendedItem]
    baseline_model: str
    comparisons: dict[str, list[RecommendedItem]] | None = None


class ReviewTrustScore(BaseModel):
    record_id: str
    user_id: str
    item_id: str
    rating: float
    review_text: str
    provisional_trust: float
    text_score: float
    behavior_score: float
    coordination_score: float
    reasons: list[str]
    confidence: float


class TrustAnalysisRequest(BaseModel):
    user_id: str | None = None
    item_id: str | None = None
    min_trust: float = 0.0
    max_trust: float = 1.0
    limit: int = Field(default=50, ge=1, le=200)


class TrustAnalysisResponse(BaseModel):
    total_scored: int
    mean_trust: float
    std_trust: float
    weights: dict[str, float]
    reviews: list[ReviewTrustScore]
    methodology_note: str


class AttackSimulationRequest(BaseModel):
    strategy: str = Field(default="bandwagon", pattern="^(random|average|bandwagon)$")
    objective: str = Field(default="push", pattern="^(push|nuke)$")
    target_item: str | None = None
    budget_fraction: float = Field(default=0.05, gt=0.0, le=0.50)
    filler_size: int = Field(default=3, ge=0, le=20)
    burst_window_hours: float | None = Field(default=48.0, gt=0.0)
    seed: int = Field(default=20261021)


class AttackSimulationResponse(BaseModel):
    manifest: dict[str, Any]
    requested_budget_fraction: float
    effective_budget_fraction: float
    synthetic_profiles_count: int
    synthetic_reviews_count: int
    target_item: str
    budget_rounding_warning: str | None = None
    impact_metrics: dict[str, Any]
    label_quarantine_verified: bool = True


class DefenseComparisonRequest(BaseModel):
    strategy: str | None = "bandwagon"
    objective: str | None = "push"
    budget: float | None = 0.05


class DefenseComparisonResponse(BaseModel):
    models: list[dict[str, Any]]
    filters: dict[str, Any]
    benchmark_dataset: str


class GovernanceAuditItem(BaseModel):
    user_id: str
    item_id: str
    baseline_rank: int | None
    original_rank: int
    governed_rank: int
    rank_jump: int
    was_adjusted: bool
    reason_code: str
    n_interactions: int
    n_low_trust: int
    suspicious_ratio: float
    mean_trust: float


class GovernanceAuditResponse(BaseModel):
    total_evaluated: int
    total_adjusted: int
    total_jumps_capped: int
    percentage_adjusted: float
    reason_code_counts: dict[str, int]
    records: list[GovernanceAuditItem]


class ExperimentSummary(BaseModel):
    experiment_id: str
    title: str
    timestamp: str
    dataset: str
    total_runs: int
    n_seeds: int


class ExperimentListResponse(BaseModel):
    experiments: list[ExperimentSummary]


class ExperimentDetailResponse(BaseModel):
    experiment_id: str
    title: str
    manifest: dict[str, Any]
    model_aggregates: dict[str, Any]
    statistical_tests: list[dict[str, Any]]
    per_seed_results: list[dict[str, Any]]


class AnalyzeReviewRequest(BaseModel):
    url: str | None = None
    review_text: str | None = None
    rating: float | None = None
    item_id: str | None = None
    user_id: str | None = None


class SignalCategoryDetail(BaseModel):
    category_name: str
    status: str  # evaluated, not_evaluated, insufficient_data
    score: int | None = None  # 0 to 100
    metrics: dict[str, Any] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    explanation: str


class ScoreFactor(BaseModel):
    name: str
    points: int
    category: str  # bonus, penalty, neutral
    detail: str


class ScrapedReviewItem(BaseModel):
    author: str = "Amazon Customer"
    rating: float | None = None
    title: str = ""
    date: str = ""
    verified_purchase: bool = False
    review_text: str = ""
    trust_score: int | None = None
    risk_level: str | None = None
    signals_summary: list[str] = Field(default_factory=list)
    score_breakdown: list[ScoreFactor] = Field(default_factory=list)
    score_narrative: str = ""


class SingleReviewAnalysisResponse(BaseModel):
    success: bool
    status: str  # analyzed, unsupported_url, invalid_input, captcha_challenge
    source_type: str  # url_dataset, live_amazon, pasted_text, unsupported_url
    url: str | None = None
    review_text: str | None = None
    item_id: str | None = None
    user_id: str | None = None
    rating: float | None = None
    trust_score: int | None = None  # 0 to 100
    risk_level: str | None = None  # Lower risk, Needs a closer look, Higher risk
    risk_label_description: str | None = None
    strongest_signals: list[str] = Field(default_factory=list)
    text_signals: SignalCategoryDetail | None = None
    behavior_signals: SignalCategoryDetail | None = None
    coordination_signals: SignalCategoryDetail | None = None
    limitations: list[str] = Field(default_factory=list)
    methodology_note: str = (
        "Scores represent heuristic anomaly indices (0-100), not calibrated fraud probabilities. "
        "A risk rating indicates statistical linguistic or behavioral deviation, not confirmed deceit."
    )
    error_message: str | None = None
    suggested_action: str | None = None
    # Live Amazon scraped fields
    product_title: str | None = None
    product_image: str | None = None
    product_rating: float | None = None
    total_ratings_text: str | None = None
    asin: str | None = None
    all_reviews: list[ScrapedReviewItem] = Field(default_factory=list)
    score_breakdown: list[ScoreFactor] = Field(default_factory=list)
    score_narrative: str = ""

