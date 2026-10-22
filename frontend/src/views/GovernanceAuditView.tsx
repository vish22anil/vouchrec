import React, { useEffect, useState } from 'react';
import {
  FileSearch,
  ShieldAlert,
  CheckCircle,
  Info,
  TrendingDown,
  Filter,
} from 'lucide-react';
import { api } from '../api';
import type { GovernanceAuditResponse } from '../types';

export const GovernanceAuditView: React.FC = () => {
  const [data, setData] = useState<GovernanceAuditResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [filterReason, setFilterReason] = useState<string>('all');

  const fetchAuditData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.getGovernanceAudit();
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Error fetching governance audit trail');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAuditData();
  }, []);

  const getReasonBadge = (code: string) => {
    switch (code) {
      case 'GOV_CAPPED_SUSPICIOUS_JUMP':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded text-xs font-medium bg-amber-950/60 border border-amber-600/50 text-amber-300">
            <ShieldAlert className="w-3.5 h-3.5 text-amber-400" />
            GOV_CAPPED_SUSPICIOUS_JUMP
          </span>
        );
      case 'GOV_PRESERVED_SPARSE_EVIDENCE':
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded text-xs font-medium bg-blue-950/60 border border-blue-600/50 text-blue-300">
            <Info className="w-3.5 h-3.5 text-blue-400" />
            GOV_PRESERVED_SPARSE_EVIDENCE
          </span>
        );
      case 'GOV_PASS_UNMODIFIED':
      default:
        return (
          <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded text-xs font-medium bg-emerald-950/60 border border-emerald-600/50 text-emerald-300">
            <CheckCircle className="w-3.5 h-3.5 text-emerald-400" />
            GOV_PASS_UNMODIFIED
          </span>
        );
    }
  };

  const filteredRecords = data?.records.filter((rec) => {
    if (filterReason === 'all') return true;
    if (filterReason === 'adjusted') return rec.was_adjusted;
    return rec.reason_code === filterReason;
  }) || [];

  if (loading && !data) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex flex-col items-center gap-3 text-vouch-soft">
          <div className="w-8 h-8 border-2 border-vouch-green border-t-transparent rounded-full animate-spin" />
          <p className="text-sm">Loading governance audit trail...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header & Policy Explanation */}
      <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border space-y-3">
        <div className="flex items-center gap-2">
          <FileSearch className="w-5 h-5 text-vouch-green" />
          <h3 className="font-semibold text-white text-base">Exposure Governor Audit Trail</h3>
        </div>
        <p className="text-xs text-vouch-soft leading-relaxed">
          Real-time post-processing log recording every ranking intervention made by the Exposure Governor. The Governor
          guards against ranking manipulation by verifying whether an item's promotion over baseline (&Delta;r) is
          substantiated by genuine organic trust evidence.
        </p>

        {/* Governor Rules Summary */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2 text-xs">
          <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border/70 space-y-1">
            <span className="text-[10px] uppercase font-mono text-vouch-soft">Threshold 1</span>
            <p className="font-medium text-white">Max Rank Jump (&tau; = 3)</p>
            <p className="text-[11px] text-vouch-soft">Promotions exceeding 3 positions are inspected.</p>
          </div>
          <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border/70 space-y-1">
            <span className="text-[10px] uppercase font-mono text-vouch-soft">Threshold 2</span>
            <p className="font-medium text-white">Min Evidence (n &ge; 3)</p>
            <p className="text-[11px] text-vouch-soft">Items with sparse interactions are preserved.</p>
          </div>
          <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border/70 space-y-1">
            <span className="text-[10px] uppercase font-mono text-vouch-soft">Threshold 3</span>
            <p className="font-medium text-white">Suspicion Ratio (&rho; &gt; 0.5)</p>
            <p className="text-[11px] text-vouch-soft">Triggers rank capping if low-trust reviews dominate.</p>
          </div>
        </div>
      </div>

      {/* KPI Cards */}
      {data && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border">
            <span className="text-xs text-vouch-soft">Total Evaluated Rankings</span>
            <p className="mt-1 text-2xl font-bold font-mono text-white">{data.total_evaluated}</p>
            <span className="text-[11px] text-vouch-soft">Across candidate test users</span>
          </div>

          <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border">
            <span className="text-xs text-vouch-soft">Total Interventions</span>
            <p className="mt-1 text-2xl font-bold font-mono text-amber-400">{data.total_adjusted}</p>
            <span className="text-[11px] text-vouch-soft">{data.percentage_adjusted.toFixed(1)}% of candidate items</span>
          </div>

          <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border">
            <span className="text-xs text-vouch-soft">Suspicious Jumps Capped</span>
            <p className="mt-1 text-2xl font-bold font-mono text-rose-400">{data.total_jumps_capped}</p>
            <span className="text-[11px] text-vouch-soft">Mitigated exposure attacks</span>
          </div>

          <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border">
            <span className="text-xs text-vouch-soft">Active Policy Mode</span>
            <p className="mt-1 text-base font-bold text-emerald-400">Strict Dual-Check</p>
            <span className="text-[11px] text-vouch-soft">Organic evidence verified</span>
          </div>
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-xl bg-vouch-card border border-vouch-border text-xs">
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-vouch-green" />
          <span className="font-semibold text-white">Filter Audit Trail:</span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => setFilterReason('all')}
            className={`px-3 py-1 rounded-lg border transition-colors ${
              filterReason === 'all'
                ? 'bg-vouch-accent border-vouch-highlight text-white'
                : 'bg-vouch-surface border-vouch-border text-slate-300 hover:text-white'
            }`}
          >
            All Events ({data?.total_evaluated ?? 0})
          </button>
          <button
            onClick={() => setFilterReason('adjusted')}
            className={`px-3 py-1 rounded-lg border transition-colors ${
              filterReason === 'adjusted'
                ? 'bg-vouch-accent border-vouch-highlight text-white'
                : 'bg-vouch-surface border-vouch-border text-slate-300 hover:text-white'
            }`}
          >
            Adjusted Only ({data?.total_adjusted ?? 0})
          </button>
          <button
            onClick={() => setFilterReason('GOV_CAPPED_SUSPICIOUS_JUMP')}
            className={`px-3 py-1 rounded-lg border transition-colors ${
              filterReason === 'GOV_CAPPED_SUSPICIOUS_JUMP'
                ? 'bg-vouch-accent border-vouch-highlight text-white'
                : 'bg-vouch-surface border-vouch-border text-slate-300 hover:text-white'
            }`}
          >
            Capped Jumps ({data?.total_jumps_capped ?? 0})
          </button>
          <button
            onClick={() => setFilterReason('GOV_PRESERVED_SPARSE_EVIDENCE')}
            className={`px-3 py-1 rounded-lg border transition-colors ${
              filterReason === 'GOV_PRESERVED_SPARSE_EVIDENCE'
                ? 'bg-vouch-accent border-vouch-highlight text-white'
                : 'bg-vouch-surface border-vouch-border text-slate-300 hover:text-white'
            }`}
          >
            Preserved Sparse ({data?.reason_code_counts['GOV_PRESERVED_SPARSE_EVIDENCE'] ?? 0})
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/20 border border-rose-800/40 text-rose-200 text-xs">
          {error}
        </div>
      )}

      {/* Audit Log Table */}
      {data && (
        <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border">
          {filteredRecords.length === 0 ? (
            <div className="p-8 text-center text-vouch-soft text-xs space-y-2">
              <CheckCircle className="w-8 h-8 text-emerald-400 mx-auto" />
              <p className="font-semibold text-white">No Governor Events Found</p>
              <p>The Exposure Governor has no recorded interventions matching this filter.</p>
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs text-slate-300">
                <thead className="bg-vouch-surface border-b border-vouch-border text-[11px] font-semibold uppercase tracking-wider text-vouch-soft">
                  <tr>
                    <th className="py-2.5 px-3">User</th>
                    <th className="py-2.5 px-3">Item ID</th>
                    <th className="py-2.5 px-3">Original Rank</th>
                    <th className="py-2.5 px-3">Baseline Rank</th>
                    <th className="py-2.5 px-3">Governed Rank</th>
                    <th className="py-2.5 px-3">Rank Jump</th>
                    <th className="py-2.5 px-3">Evidence (Int. / Low Trust / Susp. %)</th>
                    <th className="py-2.5 px-3">Governor Decision Code</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-vouch-border/50">
                  {filteredRecords.map((r, i) => (
                    <tr
                      key={`${r.user_id}-${r.item_id}-${i}`}
                      className={`hover:bg-vouch-surface/50 transition-colors ${
                        r.was_adjusted ? 'bg-amber-950/10' : ''
                      }`}
                    >
                      <td className="py-2.5 px-3 font-mono font-medium text-white">{r.user_id}</td>
                      <td className="py-2.5 px-3 font-mono font-medium text-vouch-pale">{r.item_id}</td>
                      <td className="py-2.5 px-3 font-mono text-slate-400">#{r.original_rank}</td>
                      <td className="py-2.5 px-3 font-mono text-slate-400">
                        {r.baseline_rank ? `#${r.baseline_rank}` : '—'}
                      </td>
                      <td className="py-2.5 px-3 font-mono font-bold text-white">#{r.governed_rank}</td>
                      <td className="py-2.5 px-3 font-mono">
                        {r.rank_jump !== 0 ? (
                          <span
                            className={`inline-flex items-center gap-0.5 ${
                              r.rank_jump > 0 ? 'text-amber-400' : 'text-emerald-400'
                            }`}
                          >
                            {r.rank_jump > 0 ? (
                              <>
                                <TrendingDown className="w-3 h-3" />
                                +{r.rank_jump} (jump)
                              </>
                            ) : (
                              <span>{r.rank_jump}</span>
                            )}
                          </span>
                        ) : (
                          <span className="text-slate-500">0</span>
                        )}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-slate-300">
                        <span>{r.n_interactions} total</span>
                        <span className="text-vouch-soft"> / </span>
                        <span className="text-amber-400">{r.n_low_trust} low-trust</span>
                        <span className="text-vouch-soft"> (</span>
                        <span className="text-rose-400">{(r.suspicious_ratio * 100).toFixed(0)}%</span>
                        <span className="text-vouch-soft">)</span>
                      </td>
                      <td className="py-2.5 px-3">{getReasonBadge(r.reason_code)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
