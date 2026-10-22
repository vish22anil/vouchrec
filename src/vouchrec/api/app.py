"""FastAPI application for VouchRec Research Dashboard and evaluation API."""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.staticfiles import StaticFiles

from vouchrec.api.schemas import (
    AnalyzeReviewRequest,
    AttackSimulationRequest,
    AttackSimulationResponse,
    DatasetSummaryResponse,
    DefenseComparisonRequest,
    DefenseComparisonResponse,
    ExperimentDetailResponse,
    ExperimentListResponse,
    GovernanceAuditResponse,
    HealthResponse,
    ItemsListResponse,
    ModelListResponse,
    RecommendationRequest,
    RecommendationResponse,
    SingleReviewAnalysisResponse,
    TrustAnalysisRequest,
    TrustAnalysisResponse,
    UsersListResponse,
)
from vouchrec.api.services import (
    analyze_single_review,
    execute_attack_simulation,
    get_dataset_summary,
    get_defense_comparison_data,
    get_experiment_detail,
    get_experiments_list,
    get_governance_audit_records,
    get_items_list,
    get_model_catalog,
    get_recommendations_for_user,
    get_trust_analysis,
    get_users_list,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="VouchRec Research API",
    description="Backend API for VouchRec: Trust-Gated Collaborative Filtering and Dynamic Exposure Governance.",
    version="0.1.0",
)

# CORS Configuration
default_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
env_origins = os.getenv("VOUCHREC_CORS_ORIGINS")
origins = [o.strip() for o in env_origins.split(",") if o.strip()] if env_origins else default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
async def value_error_handler(request: Any, exc: ValueError) -> JSONResponse:
    logger.warning("ValueError on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=400, content={"error": "Bad Request", "detail": str(exc)})


@app.exception_handler(FileNotFoundError)
async def file_not_found_handler(request: Any, exc: FileNotFoundError) -> JSONResponse:
    logger.warning("FileNotFoundError on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=404, content={"error": "Not Found", "detail": str(exc)})


@app.get("/api/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Return backend service health and dataset environment status."""
    return HealthResponse(
        status="ok",
        version="0.1.0",
        dataset_category="Musical_Instruments",
        is_fixture=True,
        total_reviews_available=40,
        timestamp_utc=datetime.datetime.now(datetime.UTC).isoformat(),
    )


@app.get("/api/dataset/summary", response_model=DatasetSummaryResponse, tags=["Dataset"])
def dataset_summary(category: str = "Musical_Instruments") -> DatasetSummaryResponse:
    """Return interaction counts, user/item totals, and provenance metadata."""
    try:
        return get_dataset_summary(category=category)
    except Exception as exc:
        logger.error("Failed to fetch dataset summary: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/models", response_model=ModelListResponse, tags=["Models"])
def list_models() -> ModelListResponse:
    """Return the catalog of supported recommendation and defense models."""
    return get_model_catalog()


@app.get("/api/users", response_model=UsersListResponse, tags=["Dataset"])
def list_users(category: str = "Musical_Instruments") -> UsersListResponse:
    """Return list of available user IDs with interaction statistics."""
    return get_users_list(category=category)


@app.get("/api/items", response_model=ItemsListResponse, tags=["Dataset"])
def list_items(category: str = "Musical_Instruments") -> ItemsListResponse:
    """Return list of candidate item IDs with interaction statistics."""
    return get_items_list(category=category)


@app.post("/api/recommendations", response_model=RecommendationResponse, tags=["Recommendations"])
def get_recommendations(req: RecommendationRequest) -> RecommendationResponse:
    """Generate top-K recommendations comparing Vanilla, Trust-Gated, and Governed rankings."""
    try:
        return get_recommendations_for_user(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Recommendation failure: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/trust/analyze", response_model=TrustAnalysisResponse, tags=["Trust"])
def analyze_trust(req: TrustAnalysisRequest) -> TrustAnalysisResponse:
    """Score provisional trust signals across text, behavior, and coordination heads."""
    try:
        return get_trust_analysis(req)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Trust analysis failure: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/trust/analyze-review", response_model=SingleReviewAnalysisResponse, tags=["Trust"])
def analyze_review_single(req: AnalyzeReviewRequest) -> SingleReviewAnalysisResponse:
    """Analyze a single review submitted via product review URL or pasted text."""
    try:
        return analyze_single_review(req)
    except Exception as exc:
        logger.error("Review link/text analysis failure: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc



@app.post("/api/attacks/simulate", response_model=AttackSimulationResponse, tags=["Attacks"])
def simulate_attack(req: AttackSimulationRequest) -> AttackSimulationResponse:
    """Execute a controlled shilling attack simulation with strict label quarantine."""
    try:
        return execute_attack_simulation(req)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error("Simulation failure: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/defenses/compare", response_model=DefenseComparisonResponse, tags=["Defenses"])
def compare_defenses(req: DefenseComparisonRequest) -> DefenseComparisonResponse:
    """Compare recommendation models across clean vs attacked conditions and statistical metrics."""
    return get_defense_comparison_data(req)


@app.get("/api/governance/audit", response_model=GovernanceAuditResponse, tags=["Governance"])
def governance_audit(
    category: str = "Musical_Instruments", limit: int = Query(default=100, ge=1, le=500)
) -> GovernanceAuditResponse:
    """Return inspectable Exposure Governor intervention log and reason codes."""
    return get_governance_audit_records(category=category, limit=limit)


@app.get("/api/experiments", response_model=ExperimentListResponse, tags=["Experiments"])
def list_experiments() -> ExperimentListResponse:
    """Return list of completed multi-seed experiments and benchmarks."""
    return get_experiments_list()


@app.get("/api/experiments/{experiment_id}", response_model=ExperimentDetailResponse, tags=["Experiments"])
def experiment_detail(experiment_id: str) -> ExperimentDetailResponse:
    """Return full statistical test results, confidence intervals, and per-seed results."""
    try:
        return get_experiment_detail(experiment_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/api/experiments/{experiment_id}/results", response_model=ExperimentDetailResponse, tags=["Experiments"])
def experiment_results(experiment_id: str) -> ExperimentDetailResponse:
    """Alias for experiment detail."""
    return experiment_detail(experiment_id)


# Mount static production build if present (for single-server full-stack execution)
_frontend_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
if _frontend_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_dist), html=True), name="frontend")

