import React, { useEffect, useState } from 'react';
import {
  Search,
  Filter,
  AlertTriangle,
  FileText,
  Sliders,
  Scale,
} from 'lucide-react';
import { api } from '../api';
import type { TrustAnalysisResponse } from '../types';

export const TrustAnalysisView: React.FC = () => {
  const [data, setData] = useState<TrustAnalysisResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Filters
  const [filterUserId, setFilterUserId] = useState('');
  const [filterItemId, setFilterItemId] = useState('');
  const [minTrust, setMinTrust] = useState(0.0);

  const fetchTrustData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.analyzeTrust({
        user_id: filterUserId.trim() || undefined,
        item_id: filterItemId.trim() || undefined,
        min_trust: minTrust,
        max_trust: 1.0,
        limit: 50,
      });
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Error analyzing trust signals');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchTrustData();
  }, []);

  const getScoreColor = (score: number) => {
    if (score >= 0.7) return 'text-emerald-400 bg-emerald-950/40 border-emerald-600/40';
    if (score >= 0.4) return 'text-amber-400 bg-amber-950/40 border-amber-600/40';
    return 'text-rose-400 bg-rose-950/40 border-rose-600/40';
  };

  return (
    <div className="space-y-6">
      {/* Scientific Disclaimer Alert (Critical Requirement) */}
      <div className="p-4 rounded-xl bg-amber-950/30 border border-amber-600/40 text-amber-200 text-xs flex items-start gap-3">
        <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="font-semibold text-amber-300">
            Research Disclaimer: Heuristic Trust & Anomaly Indices
          </p>
          <p className="text-amber-200/90 leading-relaxed">
            The trust scores displayed below are <strong>heuristic anomaly indices</strong> computed from text
            diversity, temporal burstiness, and rating behavior. They are <strong>not calibrated ground-truth fraud probabilities</strong>.
            In compliance with research integrity guidelines, an individual review or reviewer cannot be deemed fraudulent based
            solely on a provisional score.
          </p>
        </div>
      </div>

      {/* Head Weights & Tri-Head Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {/* Text Signal Head */}
        <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-white flex items-center gap-1.5">
              <FileText className="w-4 h-4 text-vouch-green" />
              1. Review-Text Head
            </span>
            <span className="font-mono text-vouch-pale font-bold">Weight: 40%</span>
          </div>
          <p className="text-[11px] text-vouch-soft">
            Lexical diversity (type-token ratio), character length, generic repetition, and boilerplate templates.
          </p>
          <div className="w-full bg-vouch-surface h-1.5 rounded-full overflow-hidden">
            <div className="bg-vouch-green h-full rounded-full" style={{ width: '40%' }} />
          </div>
        </div>

        {/* Behavioral Signal Head */}
        <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-white flex items-center gap-1.5">
              <Sliders className="w-4 h-4 text-vouch-mint" />
              2. Reviewer-Behavior Head
            </span>
            <span className="font-mono text-vouch-pale font-bold">Weight: 35%</span>
          </div>
          <p className="text-[11px] text-vouch-soft">
            Rating extremity distribution, review velocity, activity lifespan, and deviation from item mean.
          </p>
          <div className="w-full bg-vouch-surface h-1.5 rounded-full overflow-hidden">
            <div className="bg-vouch-mint h-full rounded-full" style={{ width: '35%' }} />
          </div>
        </div>

        {/* Coordination Signal Head */}
        <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-white flex items-center gap-1.5">
              <Scale className="w-4 h-4 text-emerald-400" />
              3. Coordination Head
            </span>
            <span className="font-mono text-vouch-pale font-bold">Weight: 25%</span>
          </div>
          <p className="text-[11px] text-vouch-soft">
            Temporal burst density (sliding window review influx) and co-review clustering across products.
          </p>
          <div className="w-full bg-vouch-surface h-1.5 rounded-full overflow-hidden">
            <div className="bg-emerald-400 h-full rounded-full" style={{ width: '25%' }} />
          </div>
        </div>
      </div>

      {/* Filter & Search Bar */}
      <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-vouch-border pb-3">
          <h3 className="font-semibold text-white text-sm flex items-center gap-2">
            <Filter className="w-4 h-4 text-vouch-green" />
            <span>Search & Filter Trust Corpus</span>
          </h3>
          <div className="text-xs text-vouch-soft font-mono">
            {data ? `${data.total_scored} reviews scored (Mean: ${data.mean_trust.toFixed(3)}, Std: ${data.std_trust.toFixed(3)})` : ''}
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
          <div>
            <label className="text-vouch-soft font-medium">Filter User ID:</label>
            <input
              type="text"
              placeholder="e.g. U01"
              value={filterUserId}
              onChange={(e) => setFilterUserId(e.target.value)}
              className="w-full mt-1 px-3 py-2 rounded-lg bg-vouch-surface border border-vouch-border text-white placeholder-slate-500 font-mono focus:outline-none focus:border-vouch-green"
            />
          </div>

          <div>
            <label className="text-vouch-soft font-medium">Filter Item ID:</label>
            <input
              type="text"
              placeholder="e.g. P001"
              value={filterItemId}
              onChange={(e) => setFilterItemId(e.target.value)}
              className="w-full mt-1 px-3 py-2 rounded-lg bg-vouch-surface border border-vouch-border text-white placeholder-slate-500 font-mono focus:outline-none focus:border-vouch-green"
            />
          </div>

          <div>
            <label className="text-vouch-soft font-medium">Min Trust Score: {minTrust.toFixed(2)}</label>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={minTrust}
              onChange={(e) => setMinTrust(Number(e.target.value))}
              className="w-full accent-vouch-green cursor-pointer mt-2"
            />
          </div>

          <div className="flex items-end">
            <button
              onClick={fetchTrustData}
              disabled={loading}
              className="w-full px-4 py-2 rounded-lg bg-vouch-accent hover:bg-vouch-highlight text-white font-medium text-xs shadow-sm transition-colors disabled:opacity-50 flex items-center justify-center gap-2"
            >
              <Search className="w-3.5 h-3.5" />
              <span>Apply Filters</span>
            </button>
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/20 border border-rose-800/40 text-rose-200 text-xs">
          {error}
        </div>
      )}

      {/* Review Trust Table */}
      {data && (
        <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-vouch-surface border-b border-vouch-border text-[11px] font-semibold uppercase tracking-wider text-vouch-soft">
                <tr>
                  <th className="py-2.5 px-3">Reviewer</th>
                  <th className="py-2.5 px-3">Item</th>
                  <th className="py-2.5 px-3">Rating</th>
                  <th className="py-2.5 px-3">Provisional Trust</th>
                  <th className="py-2.5 px-3">Text Head</th>
                  <th className="py-2.5 px-3">Behavior Head</th>
                  <th className="py-2.5 px-3">Coordination Head</th>
                  <th className="py-2.5 px-3">Confidence</th>
                  <th className="py-2.5 px-3">Evidence Explanations</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-vouch-border/50">
                {data.reviews.map((r) => (
                  <tr key={r.record_id} className="hover:bg-vouch-surface/40 transition-colors">
                    <td className="py-2.5 px-3 font-mono font-medium text-white">{r.user_id}</td>
                    <td className="py-2.5 px-3 font-mono text-vouch-pale">{r.item_id}</td>
                    <td className="py-2.5 px-3 font-mono">
                      <span className="px-1.5 py-0.5 rounded bg-vouch-surface border border-vouch-border text-amber-300">
                        ★ {r.rating.toFixed(0)}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-mono font-bold">
                      <span className={`px-2 py-0.5 rounded text-[11px] border ${getScoreColor(r.provisional_trust)}`}>
                        {r.provisional_trust.toFixed(3)}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 font-mono text-slate-300">{r.text_score.toFixed(3)}</td>
                    <td className="py-2.5 px-3 font-mono text-slate-300">{r.behavior_score.toFixed(3)}</td>
                    <td className="py-2.5 px-3 font-mono text-slate-300">{r.coordination_score.toFixed(3)}</td>
                    <td className="py-2.5 px-3 font-mono text-slate-400">{(r.confidence * 100).toFixed(0)}%</td>
                    <td className="py-2.5 px-3">
                      <div className="flex flex-wrap gap-1 max-w-xs">
                        {r.reasons.length > 0 ? (
                          r.reasons.map((reason, i) => (
                            <span
                              key={i}
                              className="text-[10px] px-1.5 py-0.5 rounded bg-vouch-surface border border-vouch-border/60 text-vouch-soft font-mono"
                            >
                              {reason}
                            </span>
                          ))
                        ) : (
                          <span className="text-[10px] text-emerald-400">No anomaly flags</span>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
