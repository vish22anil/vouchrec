import type {
  AttackSimulationResponse,
  DatasetSummary,
  DefenseComparisonResponse,
  ExperimentDetailResponse,
  GovernanceAuditResponse,
  ModelInfo,
  RecommendationResponse,
  SingleReviewAnalysisResponse,
  TrustAnalysisResponse,
  UserItemSummary,
} from './types';

const API_BASE = '/api';

async function handleResponse<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    try {
      const json = await res.json();
      if (json.detail) {
        errorDetail = typeof json.detail === 'string' ? json.detail : JSON.stringify(json.detail);
      } else if (json.error) {
        errorDetail = json.error;
      }
    } catch {
      // fallback to status text
    }
    throw new Error(errorDetail);
  }
  return res.json();
}

export const api = {
  async getHealth(): Promise<{ status: string; version: string; is_fixture: boolean; timestamp_utc: string }> {
    const res = await fetch(`${API_BASE}/health`);
    return handleResponse(res);
  },

  async getDatasetSummary(category = 'Musical_Instruments'): Promise<DatasetSummary> {
    const res = await fetch(`${API_BASE}/dataset/summary?category=${encodeURIComponent(category)}`);
    return handleResponse(res);
  },

  async getModels(): Promise<{ models: ModelInfo[] }> {
    const res = await fetch(`${API_BASE}/models`);
    return handleResponse(res);
  },

  async getUsers(category = 'Musical_Instruments'): Promise<{ users: UserItemSummary[]; total_users: number }> {
    const res = await fetch(`${API_BASE}/users?category=${encodeURIComponent(category)}`);
    return handleResponse(res);
  },

  async getItems(category = 'Musical_Instruments'): Promise<{ items: UserItemSummary[]; total_items: number }> {
    const res = await fetch(`${API_BASE}/items?category=${encodeURIComponent(category)}`);
    return handleResponse(res);
  },

  async getRecommendations(params: {
    user_id: string;
    model_id?: string;
    k?: number;
    compare_variants?: boolean;
    max_rank_jump?: number;
  }): Promise<RecommendationResponse> {
    const res = await fetch(`${API_BASE}/recommendations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse(res);
  },

  async analyzeTrust(params: {
    user_id?: string;
    item_id?: string;
    min_trust?: number;
    max_trust?: number;
    limit?: number;
  }): Promise<TrustAnalysisResponse> {
    const res = await fetch(`${API_BASE}/trust/analyze`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse(res);
  },

  async simulateAttack(params: {
    strategy?: string;
    objective?: string;
    target_item?: string | null;
    budget_fraction?: number;
    filler_size?: number;
    burst_window_hours?: number | null;
    seed?: number;
  }): Promise<AttackSimulationResponse> {
    const res = await fetch(`${API_BASE}/attacks/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse(res);
  },

  async compareDefenses(params: {
    strategy?: string;
    objective?: string;
    budget?: number;
  }): Promise<DefenseComparisonResponse> {
    const res = await fetch(`${API_BASE}/defenses/compare`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse(res);
  },

  async getGovernanceAudit(category = 'Musical_Instruments', limit = 100): Promise<GovernanceAuditResponse> {
    const res = await fetch(`${API_BASE}/governance/audit?category=${encodeURIComponent(category)}&limit=${limit}`);
    return handleResponse(res);
  },

  async getExperimentDetail(experimentId = 'phase7_multi_seed_benchmark'): Promise<ExperimentDetailResponse> {
    const res = await fetch(`${API_BASE}/experiments/${encodeURIComponent(experimentId)}`);
    return handleResponse(res);
  },

  async analyzeReview(params: {
    url?: string | null;
    review_text?: string | null;
    rating?: number | null;
    item_id?: string | null;
    user_id?: string | null;
  }): Promise<SingleReviewAnalysisResponse> {
    const res = await fetch(`${API_BASE}/trust/analyze-review`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    return handleResponse(res);
  },
};

