export interface DatasetSummary {
  category: string;
  total_reviews: number;
  total_users: number;
  total_items: number;
  train_interactions: number;
  val_interactions: number;
  test_interactions: number;
  is_fixture: boolean;
  data_root: string;
  raw_file_available: boolean;
  provenance_note: string;
}

export interface ModelInfo {
  id: string;
  name: string;
  family: 'baseline' | 'graph' | 'trust_gated' | 'governed';
  description: string;
  is_available: boolean;
}

export interface UserItemSummary {
  user_id?: string;
  item_id?: string;
  interaction_count: number;
  mean_rating: number;
}

export interface RecommendedItem {
  item_id: string;
  score: number;
  original_rank: number;
  governed_rank: number;
  baseline_rank?: number | null;
  rank_shift: number;
  was_adjusted: boolean;
  reason_code: string;
  n_interactions: number;
  n_low_trust: number;
  suspicious_ratio: number;
  mean_trust: number;
}

export interface RecommendationResponse {
  user_id: string;
  model_id: string;
  k: number;
  recommendations: RecommendedItem[];
  baseline_model: string;
  comparisons?: Record<string, RecommendedItem[]> | null;
}

export interface ReviewTrustScore {
  record_id: string;
  user_id: string;
  item_id: string;
  rating: number;
  review_text: string;
  provisional_trust: number;
  text_score: number;
  behavior_score: number;
  coordination_score: number;
  reasons: string[];
  confidence: number;
}

export interface TrustAnalysisResponse {
  total_scored: number;
  mean_trust: number;
  std_trust: number;
  weights: Record<string, number>;
  reviews: ReviewTrustScore[];
  methodology_note: string;
}

export interface AttackSimulationResponse {
  manifest: Record<string, any>;
  requested_budget_fraction: number;
  effective_budget_fraction: number;
  synthetic_profiles_count: number;
  synthetic_reviews_count: number;
  target_item: string;
  budget_rounding_warning?: string | null;
  impact_metrics: {
    target_hitrate_at_10_clean?: number;
    target_hitrate_at_10_attacked?: number;
    target_hitrate_delta?: number;
    target_mean_rank_clean?: number;
    target_mean_rank_attacked?: number;
    target_rank_improvement?: number;
    [key: string]: any;
  };
  label_quarantine_verified: boolean;
}

export interface DefenseModelStats {
  model_name: string;
  clean_ndcg_at_10: number;
  clean_ndcg_at_10_ci?: [number, number];
  attacked_ndcg_at_10: number;
  attacked_ndcg_at_10_ci?: [number, number];
  ndcg_delta: number;
  attacked_recall_at_20: number;
  target_mean_rank: number;
  jumps_capped: number;
  runtime_seconds: number;
}

export interface DefenseComparisonResponse {
  models: DefenseModelStats[];
  filters: Record<string, any>;
  benchmark_dataset: string;
}

export interface GovernanceAuditItem {
  user_id: string;
  item_id: string;
  baseline_rank?: number | null;
  original_rank: number;
  governed_rank: number;
  rank_jump: number;
  was_adjusted: boolean;
  reason_code: string;
  n_interactions: number;
  n_low_trust: number;
  suspicious_ratio: number;
  mean_trust: number;
}

export interface GovernanceAuditResponse {
  total_evaluated: number;
  total_adjusted: number;
  total_jumps_capped: number;
  percentage_adjusted: number;
  reason_code_counts: Record<string, number>;
  records: GovernanceAuditItem[];
}

export interface StatisticalTest {
  condition_a: string;
  condition_b: string;
  metric_name: string;
  mean_diff: number;
  t_statistic: number;
  p_value: number;
  holm_rank: number;
  holm_threshold: number;
  adjusted_p_value: number;
  cohens_d: number;
  is_significant_holm: boolean;
}

export interface ExperimentDetailResponse {
  experiment_id: string;
  title: string;
  manifest: Record<string, any>;
  model_aggregates: Record<string, any>;
  statistical_tests: StatisticalTest[];
  per_seed_results: Record<string, any>[];
}

export interface SignalCategoryDetail {
  category_name: string;
  status: 'evaluated' | 'not_evaluated' | 'insufficient_data';
  score?: number | null;
  metrics?: Record<string, any>;
  reasons?: string[];
  explanation: string;
}

export interface ScoreFactor {
  name: string;
  points: number;
  category: 'bonus' | 'penalty' | 'neutral';
  detail: string;
}

export interface ScrapedReviewItem {
  author: string;
  rating?: number | null;
  title: string;
  date: string;
  verified_purchase: boolean;
  review_text: string;
  trust_score?: number | null;
  risk_level?: string | null;
  signals_summary: string[];
  score_breakdown?: ScoreFactor[];
  score_narrative?: string;
}

export interface SingleReviewAnalysisResponse {
  success: boolean;
  status: 'analyzed' | 'unsupported_url' | 'invalid_input' | 'captcha_challenge' | string;
  source_type: 'url_dataset' | 'live_amazon' | 'pasted_text' | 'unsupported_url' | string;
  url?: string | null;
  review_text?: string | null;
  item_id?: string | null;
  user_id?: string | null;
  rating?: number | null;
  trust_score?: number | null;
  risk_level?: 'Lower risk' | 'Needs a closer look' | 'Higher risk' | null;
  risk_label_description?: string | null;
  strongest_signals?: string[];
  text_signals?: SignalCategoryDetail | null;
  behavior_signals?: SignalCategoryDetail | null;
  coordination_signals?: SignalCategoryDetail | null;
  limitations: string[];
  methodology_note?: string;
  error_message?: string | null;
  suggested_action?: string | null;
  // Live Amazon product fields
  product_title?: string | null;
  product_image?: string | null;
  product_rating?: number | null;
  total_ratings_text?: string | null;
  asin?: string | null;
  all_reviews?: ScrapedReviewItem[];
  score_breakdown?: ScoreFactor[];
  score_narrative?: string;
}

