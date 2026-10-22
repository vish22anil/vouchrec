import React, { useEffect, useState } from 'react';
import {
  Compass,
  ShieldAlert,
  Info,
  CheckCircle,
  TrendingDown,
  TrendingUp,
} from 'lucide-react';
import { api } from '../api';
import type { ModelInfo, RecommendationResponse, UserItemSummary } from '../types';

export const RecommendationView: React.FC = () => {
  const [users, setUsers] = useState<UserItemSummary[]>([]);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [selectedUser, setSelectedUser] = useState<string>('U01');
  const [selectedModel, setSelectedModel] = useState<string>('governed_trust_gated_lightgcn');
  const [topK, setTopK] = useState<number>(10);
  const [compareVariants, setCompareVariants] = useState<boolean>(true);
  const [maxRankJump, setMaxRankJump] = useState<number>(3);

  const [loading, setLoading] = useState<boolean>(false);
  const [recData, setRecData] = useState<RecommendationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Expanded row for item detail
  const [expandedItemId, setExpandedItemId] = useState<string | null>(null);

  // Initial load: users and models
  useEffect(() => {
    async function loadMeta() {
      try {
        const [uRes, mRes] = await Promise.all([api.getUsers(), api.getModels()]);
        setUsers(uRes.users);
        if (uRes.users.length > 0 && uRes.users[0].user_id) {
          setSelectedUser(uRes.users[0].user_id);
        }
        setModels(mRes.models);
      } catch (err: any) {
        console.error('Failed to load users or models', err);
      }
    }
    loadMeta();
  }, []);

  // Fetch recommendations whenever user or parameters change
  const handleFetchRecommendations = async () => {
    if (!selectedUser) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.getRecommendations({
        user_id: selectedUser,
        model_id: selectedModel,
        k: topK,
        compare_variants: compareVariants,
        max_rank_jump: maxRankJump,
      });
      setRecData(res);
    } catch (err: any) {
      setError(err.message || 'Error generating recommendations');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedUser) {
      handleFetchRecommendations();
    }
  }, [selectedUser, selectedModel, topK, compareVariants, maxRankJump]);

  const getReasonBadge = (code: string) => {
    switch (code) {
      case 'GOV_CAPPED_SUSPICIOUS_JUMP':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-amber-950/60 border border-amber-600/50 text-amber-300">
            <ShieldAlert className="w-3 h-3 text-amber-400" />
            Capped (Suspicious Jump)
          </span>
        );
      case 'GOV_PRESERVED_SPARSE_EVIDENCE':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-blue-950/60 border border-blue-600/50 text-blue-300">
            <Info className="w-3 h-3 text-blue-400" />
            Preserved (Sparse Evidence)
          </span>
        );
      case 'GOV_PASS_UNMODIFIED':
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-emerald-950/60 border border-emerald-600/50 text-emerald-300">
            <CheckCircle className="w-3 h-3 text-emerald-400" />
            Pass Unmodified
          </span>
        );
    }
  };

  return (
    <div className="space-y-6">
      {/* Control Panel */}
      <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-vouch-border pb-4">
          <div>
            <h3 className="font-semibold text-white text-base flex items-center gap-2">
              <Compass className="w-4 h-4 text-vouch-green" />
              <span>Interactive Recommendation Explorer</span>
            </h3>
            <p className="text-xs text-vouch-soft mt-0.5">
              Compare actual top-K ranking decisions across vanilla, trust-gated, and governed models.
            </p>
          </div>

          <button
            onClick={handleFetchRecommendations}
            disabled={loading}
            className="px-4 py-2 rounded-lg bg-vouch-accent hover:bg-vouch-highlight text-white text-xs font-semibold shadow-sm transition-colors disabled:opacity-50 flex items-center gap-2 self-start md:self-auto"
          >
            {loading ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Computing...</span>
              </>
            ) : (
              <span>Re-run Recommendations</span>
            )}
          </button>
        </div>

        {/* Inputs */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          {/* User selector */}
          <div className="space-y-1.5">
            <label className="text-vouch-soft font-medium flex items-center justify-between">
              <span>Target Research User:</span>
              <span className="font-mono text-vouch-pale">{selectedUser}</span>
            </label>
            <select
              value={selectedUser}
              onChange={(e) => setSelectedUser(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-vouch-surface border border-vouch-border text-white focus:outline-none focus:border-vouch-green font-mono"
            >
              {users.map((u) => (
                <option key={u.user_id} value={u.user_id}>
                  {u.user_id} ({u.interaction_count} interactions)
                </option>
              ))}
            </select>
          </div>

          {/* Model selector */}
          <div className="space-y-1.5">
            <label className="text-vouch-soft font-medium">Evaluation Model:</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-vouch-surface border border-vouch-border text-white focus:outline-none focus:border-vouch-green"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id} disabled={!m.is_available}>
                  {m.name} {!m.is_available ? '(Unavailable)' : ''}
                </option>
              ))}
            </select>
          </div>

          {/* Top-K Slider */}
          <div className="space-y-1.5">
            <label className="text-vouch-soft font-medium flex items-center justify-between">
              <span>Top-K Depth:</span>
              <span className="font-mono text-vouch-pale font-bold">{topK}</span>
            </label>
            <input
              type="range"
              min={3}
              max={15}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full accent-vouch-green cursor-pointer mt-2"
            />
          </div>

          {/* Governor Max Jump Threshold */}
          <div className="space-y-1.5">
            <label className="text-vouch-soft font-medium flex items-center justify-between">
              <span>Gov. Max Jump (τ_jump):</span>
              <span className="font-mono text-vouch-pale font-bold">{maxRankJump}</span>
            </label>
            <input
              type="range"
              min={1}
              max={8}
              value={maxRankJump}
              onChange={(e) => setMaxRankJump(Number(e.target.value))}
              className="w-full accent-vouch-green cursor-pointer mt-2"
            />
          </div>
        </div>

        {/* Checkbox toggle */}
        <div className="flex items-center gap-2 pt-1 text-xs">
          <input
            type="checkbox"
            id="compareVariants"
            checked={compareVariants}
            onChange={(e) => setCompareVariants(e.target.checked)}
            className="rounded accent-vouch-green"
          />
          <label htmlFor="compareVariants" className="text-slate-300 cursor-pointer">
            Side-by-side comparison with Vanilla LightGCN & Trust-Gated LightGCN
          </label>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/20 border border-rose-800/40 text-rose-200 text-xs">
          {error}
        </div>
      )}

      {/* Main Results Table */}
      {recData && (
        <div className="space-y-4">
          <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border space-y-4">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
              <div>
                <h4 className="font-semibold text-white text-sm">
                  Top-{topK} Recommendations for User <span className="font-mono text-vouch-pale">{recData.user_id}</span>
                </h4>
                <p className="text-xs text-vouch-soft">
                  Active Model: <span className="text-white font-medium">{recData.model_id}</span>
                </p>
              </div>

              <div className="text-xs text-vouch-soft font-mono">
                {recData.recommendations.filter((r) => r.was_adjusted).length} interventions in top-{topK}
              </div>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-vouch-surface border-b border-vouch-border text-[11px] font-semibold uppercase tracking-wider text-vouch-soft">
                  <tr>
                    <th className="py-2.5 px-3">Final Rank</th>
                    <th className="py-2.5 px-3">Item ID</th>
                    <th className="py-2.5 px-3">Model Score</th>
                    <th className="py-2.5 px-3">Original Rank</th>
                    <th className="py-2.5 px-3">Baseline Rank</th>
                    <th className="py-2.5 px-3">Rank Shift</th>
                    <th className="py-2.5 px-3">Governance Status</th>
                    <th className="py-2.5 px-3 text-right">Evidence Action</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-vouch-border/50">
                  {recData.recommendations.map((item) => {
                    const isExpanded = expandedItemId === item.item_id;
                    return (
                      <React.Fragment key={item.item_id}>
                        <tr
                          className={`hover:bg-vouch-surface/50 transition-colors ${
                            item.was_adjusted ? 'bg-amber-950/10' : ''
                          }`}
                        >
                          <td className="py-2.5 px-3 font-mono font-bold text-white">
                            #{item.governed_rank}
                          </td>
                          <td className="py-2.5 px-3 font-mono font-medium text-vouch-pale">
                            {item.item_id}
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-300">
                            {item.score.toFixed(4)}
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-400">
                            #{item.original_rank}
                          </td>
                          <td className="py-2.5 px-3 font-mono text-slate-400">
                            {item.baseline_rank ? `#${item.baseline_rank}` : '—'}
                          </td>
                          <td className="py-2.5 px-3 font-mono">
                            {item.rank_shift !== 0 ? (
                              <span
                                className={`inline-flex items-center gap-0.5 ${
                                  item.rank_shift > 0 ? 'text-amber-400' : 'text-emerald-400'
                                }`}
                              >
                                {item.rank_shift > 0 ? (
                                  <>
                                    <TrendingDown className="w-3 h-3" />
                                    +{item.rank_shift} (demoted)
                                  </>
                                ) : (
                                  <>
                                    <TrendingUp className="w-3 h-3" />
                                    {item.rank_shift} (promoted)
                                  </>
                                )}
                              </span>
                            ) : (
                              <span className="text-slate-500">0</span>
                            )}
                          </td>
                          <td className="py-2.5 px-3">{getReasonBadge(item.reason_code)}</td>
                          <td className="py-2.5 px-3 text-right">
                            <button
                              onClick={() => setExpandedItemId(isExpanded ? null : item.item_id)}
                              className="px-2 py-1 rounded bg-vouch-surface hover:bg-vouch-border text-[11px] text-vouch-soft hover:text-white border border-vouch-border"
                            >
                              {isExpanded ? 'Hide Evidence' : 'Inspect Evidence'}
                            </button>
                          </td>
                        </tr>

                        {/* Collapsible Evidence Row */}
                        {isExpanded && (
                          <tr className="bg-vouch-surface/90 border-t border-b border-vouch-highlight/20">
                            <td colSpan={8} className="p-4">
                              <div className="grid grid-cols-1 sm:grid-cols-4 gap-4 text-xs">
                                <div className="p-2.5 rounded bg-vouch-card border border-vouch-border space-y-1">
                                  <span className="text-[10px] text-vouch-soft uppercase">Total Interactions</span>
                                  <p className="font-mono text-white text-sm font-semibold">{item.n_interactions}</p>
                                </div>
                                <div className="p-2.5 rounded bg-vouch-card border border-vouch-border space-y-1">
                                  <span className="text-[10px] text-vouch-soft uppercase">Low-Trust Interactions</span>
                                  <p className="font-mono text-amber-400 text-sm font-semibold">{item.n_low_trust}</p>
                                </div>
                                <div className="p-2.5 rounded bg-vouch-card border border-vouch-border space-y-1">
                                  <span className="text-[10px] text-vouch-soft uppercase">Suspicious Ratio</span>
                                  <p className="font-mono text-rose-400 text-sm font-semibold">
                                    {(item.suspicious_ratio * 100).toFixed(1)}%
                                  </p>
                                </div>
                                <div className="p-2.5 rounded bg-vouch-card border border-vouch-border space-y-1">
                                  <span className="text-[10px] text-vouch-soft uppercase">Item Mean Trust</span>
                                  <p className="font-mono text-emerald-400 text-sm font-semibold">
                                    {item.mean_trust.toFixed(3)}
                                  </p>
                                </div>
                              </div>
                              <p className="text-[11px] text-vouch-soft mt-2">
                                <span className="font-semibold text-slate-300">Governance Decision: </span>
                                {item.reason_code === 'GOV_CAPPED_SUSPICIOUS_JUMP'
                                  ? `Item exhibited a ranking jump exceeding τ_jump=${maxRankJump} without sufficient organic positive evidence. Capped to prevent artificial exposure inflation.`
                                  : item.reason_code === 'GOV_PRESERVED_SPARSE_EVIDENCE'
                                  ? 'Item interaction count is below the minimum evidence threshold (n_interactions < 3). Ranking preserved to protect sparse cold-start items.'
                                  : 'Item ranking was confirmed unmodified by the Exposure Governor.'}
                              </p>
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Model Comparisons: Vanilla vs Trust-Gated vs Governed */}
          {recData.comparisons && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Vanilla LightGCN Top Items */}
              {recData.comparisons['vanilla_lightgcn'] && (
                <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border">
                  <h5 className="font-semibold text-white text-xs mb-2">Vanilla LightGCN (No Defense)</h5>
                  <div className="space-y-1.5 text-xs">
                    {recData.comparisons['vanilla_lightgcn'].slice(0, 5).map((it, idx) => (
                      <div
                        key={it.item_id}
                        className="flex items-center justify-between p-2 rounded bg-vouch-surface border border-vouch-border/50 text-slate-300"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-bold text-vouch-soft">#{idx + 1}</span>
                          <span className="font-mono font-medium text-white">{it.item_id}</span>
                        </div>
                        <span className="font-mono text-xs text-vouch-pale">{it.score.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Trust-Gated LightGCN Top Items */}
              {recData.comparisons['trust_gated_lightgcn_linear'] && (
                <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border">
                  <h5 className="font-semibold text-white text-xs mb-2">Trust-Gated LightGCN (Soft-Linear)</h5>
                  <div className="space-y-1.5 text-xs">
                    {recData.comparisons['trust_gated_lightgcn_linear'].slice(0, 5).map((it, idx) => (
                      <div
                        key={it.item_id}
                        className="flex items-center justify-between p-2 rounded bg-vouch-surface border border-vouch-border/50 text-slate-300"
                      >
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-bold text-vouch-green">#{idx + 1}</span>
                          <span className="font-mono font-medium text-white">{it.item_id}</span>
                        </div>
                        <span className="font-mono text-xs text-vouch-pale">{it.score.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};
