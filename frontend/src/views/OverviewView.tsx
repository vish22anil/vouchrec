import React, { useEffect, useState } from 'react';
import {
  ShieldAlert,
  Database,
  Users,
  Package,
  Activity,
  Layers,
  CheckCircle,
  AlertCircle,
} from 'lucide-react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from 'recharts';
import { api } from '../api';
import type { DatasetSummary, DefenseComparisonResponse } from '../types';

export const OverviewView: React.FC = () => {
  const [summary, setSummary] = useState<DatasetSummary | null>(null);
  const [defenses, setDefenses] = useState<DefenseComparisonResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadData() {
      try {
        setLoading(true);
        setError(null);
        const [sumRes, defRes] = await Promise.all([
          api.getDatasetSummary(),
          api.compareDefenses({}),
        ]);
        setSummary(sumRes);
        setDefenses(defRes);
      } catch (err: any) {
        setError(err.message || 'Failed to load overview data');
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex flex-col items-center gap-3 text-vouch-soft">
          <div className="w-8 h-8 border-2 border-vouch-green border-t-transparent rounded-full animate-spin" />
          <p className="text-sm">Loading research overview...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 rounded-xl bg-rose-950/20 border border-rose-800/40 text-rose-200 flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
        <div>
          <h3 className="font-semibold text-rose-300">Failed to load overview</h3>
          <p className="text-sm mt-1 text-rose-200/80">{error}</p>
          <p className="text-xs text-rose-400/60 mt-2">Ensure the FastAPI backend is running on port 8000.</p>
        </div>
      </div>
    );
  }

  const chartData = defenses?.models.map((m) => ({
    name: m.model_name.replace('Trust-Gated LightGCN', 'TG-LightGCN').replace('LightGCN', 'LGCN'),
    fullName: m.model_name,
    Clean: Number(m.clean_ndcg_at_10.toFixed(4)),
    Attacked: Number(m.attacked_ndcg_at_10.toFixed(4)),
    Drop: Number(((m.clean_ndcg_at_10 - m.attacked_ndcg_at_10) * 100).toFixed(1)),
  })) || [];

  return (
    <div className="space-y-6">
      {/* Top Research KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Interactions */}
        <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border hover:border-vouch-highlight/40 transition-colors">
          <div className="flex items-center justify-between text-vouch-soft text-xs font-medium">
            <span>Total Interactions</span>
            <Activity className="w-4 h-4 text-vouch-green" />
          </div>
          <div className="mt-2 text-2xl font-bold text-white tracking-tight">
            {summary?.total_reviews.toLocaleString() ?? '—'}
          </div>
          <div className="mt-1 text-xs text-vouch-soft flex items-center gap-1.5">
            <span className="text-vouch-pale font-mono">
              {summary ? `${summary.train_interactions} train / ${summary.test_interactions} test` : ''}
            </span>
          </div>
        </div>

        {/* Verified Users */}
        <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border hover:border-vouch-highlight/40 transition-colors">
          <div className="flex items-center justify-between text-vouch-soft text-xs font-medium">
            <span>Distinct Users</span>
            <Users className="w-4 h-4 text-vouch-mint" />
          </div>
          <div className="mt-2 text-2xl font-bold text-white tracking-tight">
            {summary?.total_users.toLocaleString() ?? '—'}
          </div>
          <div className="mt-1 text-xs text-vouch-soft">
            5-core threshold filtered
          </div>
        </div>

        {/* Catalog Items */}
        <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border hover:border-vouch-highlight/40 transition-colors">
          <div className="flex items-center justify-between text-vouch-soft text-xs font-medium">
            <span>Catalog Products</span>
            <Package className="w-4 h-4 text-vouch-pale" />
          </div>
          <div className="mt-2 text-2xl font-bold text-white tracking-tight">
            {summary?.total_items.toLocaleString() ?? '—'}
          </div>
          <div className="mt-1 text-xs text-vouch-soft">
            Graph candidate nodes
          </div>
        </div>

        {/* Active Defense Architecture */}
        <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border hover:border-vouch-highlight/40 transition-colors">
          <div className="flex items-center justify-between text-vouch-soft text-xs font-medium">
            <span>Governance Defense</span>
            <ShieldAlert className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="mt-2 text-lg font-bold text-emerald-400 truncate">
            TG-LightGCN + Governor
          </div>
          <div className="mt-1 text-xs text-emerald-300/80 flex items-center gap-1">
            <CheckCircle className="w-3 h-3 text-emerald-400" />
            <span>Dual-stage graph & rank defense</span>
          </div>
        </div>
      </div>

      {/* Main Charts & Overview Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Chart: Clean vs Attacked NDCG@10 */}
        <div className="lg:col-span-2 p-5 rounded-xl bg-vouch-card border border-vouch-border">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
            <div>
              <h3 className="font-semibold text-white text-base">Clean vs. Attacked Recommendation Quality</h3>
              <p className="text-xs text-vouch-soft">
                Evaluating NDCG@10 under Bandwagon Push Attack (10% budget, target item injection)
              </p>
            </div>
            <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-vouch-surface border border-vouch-border text-vouch-soft">
              Metric: NDCG@10
            </span>
          </div>

          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 25 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f4332" opacity={0.4} />
                <XAxis
                  dataKey="name"
                  stroke="#74c69d"
                  fontSize={11}
                  angle={-15}
                  textAnchor="end"
                  interval={0}
                />
                <YAxis stroke="#74c69d" fontSize={11} domain={[0, 1]} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0].payload;
                      return (
                        <div className="p-3 bg-vouch-surface border border-vouch-border rounded-lg shadow-xl text-xs space-y-1">
                          <p className="font-semibold text-white">{data.fullName}</p>
                          <p className="text-emerald-400">Clean NDCG@10: {data.Clean}</p>
                          <p className="text-rose-400">Attacked NDCG@10: {data.Attacked}</p>
                          <p className="text-vouch-soft">NDCG Drop: {data.Drop}%</p>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend
                  wrapperStyle={{ paddingTop: 10, fontSize: 12 }}
                  formatter={(value) => <span className="text-slate-300">{value} NDCG@10</span>}
                />
                <Bar dataKey="Clean" fill="#52b788" radius={[4, 4, 0, 0]} barSize={20} />
                <Bar dataKey="Attacked" fill="#ef4444" radius={[4, 4, 0, 0]} barSize={20} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <p className="text-[11px] text-vouch-soft/70 mt-2 text-center">
            Higher attacked score indicates greater robustness against manipulation profiles.
          </p>
        </div>

        {/* Dataset Provenance & Split Metadata */}
        <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <Database className="w-4 h-4 text-vouch-green" />
              <h3 className="font-semibold text-white text-base">Data Provenance</h3>
            </div>

            <div className="space-y-3 text-xs">
              <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border/70 space-y-1">
                <span className="text-vouch-soft font-mono uppercase text-[10px]">Corpus</span>
                <p className="font-medium text-white">{summary?.category} (Amazon Reviews 2023)</p>
                <p className="text-vouch-soft/80 text-[11px] leading-relaxed">
                  Source: Julian McAuley Lab, UCSD. Research license verified.
                </p>
              </div>

              <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border/70 space-y-1.5">
                <span className="text-vouch-soft font-mono uppercase text-[10px]">Split Integrity</span>
                <div className="flex justify-between text-slate-300">
                  <span>Train Set:</span>
                  <span className="font-mono text-vouch-pale">{summary?.train_interactions} interactions</span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>Validation Set:</span>
                  <span className="font-mono text-vouch-pale">{summary?.val_interactions} interactions</span>
                </div>
                <div className="flex justify-between text-slate-300">
                  <span>Test Set (held-out):</span>
                  <span className="font-mono text-vouch-pale">{summary?.test_interactions} interactions</span>
                </div>
                <div className="pt-1 text-[11px] text-emerald-400 flex items-center gap-1">
                  <CheckCircle className="w-3 h-3" />
                  <span>Strict temporal & user-disjoint splits</span>
                </div>
              </div>

              <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border/70">
                <span className="text-vouch-soft font-mono uppercase text-[10px]">Data Root Location</span>
                <p className="font-mono text-[11px] text-slate-300 truncate mt-0.5" title={summary?.data_root}>
                  {summary?.data_root}
                </p>
              </div>
            </div>
          </div>

          <div className="pt-4 border-t border-vouch-border text-[11px] text-vouch-soft/70">
            {summary?.provenance_note}
          </div>
        </div>
      </div>

      {/* Architecture & Research Pillars */}
      <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border">
        <h3 className="font-semibold text-white text-base mb-3 flex items-center gap-2">
          <Layers className="w-4 h-4 text-vouch-green" />
          <span>VouchRec Defense Architecture Pipeline</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-xs">
          <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border space-y-1">
            <div className="font-mono text-vouch-green font-semibold">Stage 1: Ingestion & Splitting</div>
            <p className="text-slate-300 text-[11px]">
              Raw Amazon reviews ingested and filtered with 5-core thresholds into reproducible parquet partitions.
            </p>
          </div>

          <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border space-y-1">
            <div className="font-mono text-vouch-green font-semibold">Stage 2: Tri-Head Trust Scoring</div>
            <p className="text-slate-300 text-[11px]">
              Text semantic diversity, reviewer behavioral anomalies, and coordination burst graphs fused into provisional trust $w_e \in (0, 1]$.
            </p>
          </div>

          <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border space-y-1">
            <div className="font-mono text-vouch-green font-semibold">Stage 3: Trust-Gated LightGCN</div>
            <p className="text-slate-300 text-[11px]">
              Graph convolution propagates information along trust-attenuated edges (normalized adjacency A), suppressing synthetic manipulation nodes.
            </p>
          </div>

          <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border space-y-1">
            <div className="font-mono text-vouch-green font-semibold">Stage 4: Exposure Governor</div>
            <p className="text-slate-300 text-[11px]">
              Post-processing layer detects suspicious ranking jumps (rank shift &gt; threshold) without sufficient organic evidence, capping rank promotion.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
