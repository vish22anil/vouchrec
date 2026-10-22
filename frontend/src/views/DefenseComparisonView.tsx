import React, { useEffect, useState } from 'react';
import {
  Sliders,
  ShieldCheck,
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
import type { DefenseComparisonResponse } from '../types';

export const DefenseComparisonView: React.FC = () => {
  const [strategy, setStrategy] = useState<string>('bandwagon');
  const [objective, setObjective] = useState<string>('push');
  const [budget, setBudget] = useState<number>(0.1);

  const [loading, setLoading] = useState<boolean>(true);
  const [data, setData] = useState<DefenseComparisonResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fetchDefenses = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.compareDefenses({ strategy, objective, budget });
      setData(res);
    } catch (err: any) {
      setError(err.message || 'Error loading defense comparisons');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDefenses();
  }, [strategy, objective, budget]);

  const chartData = data?.models.map((m) => ({
    name: m.model_name
      .replace('Trust-Gated LightGCN', 'TG-LightGCN')
      .replace('Hard-filtered LightGCN', 'Hard-Filt. LGCN')
      .replace('Vanilla LightGCN', 'Vanilla LGCN')
      .replace('with Exposure Governor', '+ Gov'),
    fullName: m.model_name,
    Clean_NDCG: Number(m.clean_ndcg_at_10.toFixed(4)),
    Attacked_NDCG: Number(m.attacked_ndcg_at_10.toFixed(4)),
    Recall_20: Number(m.attacked_recall_at_20.toFixed(4)),
    TargetRank: m.target_mean_rank,
  })) || [];

  return (
    <div className="space-y-6">
      {/* Filters & Control bar */}
      <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-vouch-border pb-3">
          <div>
            <h3 className="font-semibold text-white text-base flex items-center gap-2">
              <Sliders className="w-4 h-4 text-vouch-green" />
              <span>Multi-Model Defense Benchmark Comparison</span>
            </h3>
            <p className="text-xs text-vouch-soft mt-0.5">
              Evaluating 7 defense paradigms on identical validation splits under controlled adversarial attacks.
            </p>
          </div>
          <div className="text-xs text-vouch-soft font-mono flex items-center gap-2">
            {loading && <div className="w-3 h-3 border-2 border-vouch-green border-t-transparent rounded-full animate-spin" />}
            <span>Benchmark: {data?.benchmark_dataset ?? 'Musical Instruments (Fixture)'}</span>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
          <div className="space-y-1.5">
            <label className="text-vouch-soft font-medium">Attack Strategy Filter:</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-vouch-surface border border-vouch-border text-white focus:outline-none focus:border-vouch-green"
            >
              <option value="bandwagon">Bandwagon Attack</option>
              <option value="average">Average Attack</option>
              <option value="random">Random Attack</option>
            </select>
          </div>

          <div className="space-y-1.5">
            <label className="text-vouch-soft font-medium">Attack Objective:</label>
            <select
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-vouch-surface border border-vouch-border text-white focus:outline-none focus:border-vouch-green"
            >
              <option value="push">Push (Target Promotion)</option>
              <option value="nuke">Nuke (Target Demotion)</option>
            </select>
          </div>

          <div className="space-y-1.5">
            <label className="text-vouch-soft font-medium">Attack Budget:</label>
            <select
              value={budget}
              onChange={(e) => setBudget(Number(e.target.value))}
              className="w-full px-3 py-2 rounded-lg bg-vouch-surface border border-vouch-border text-white focus:outline-none focus:border-vouch-green"
            >
              <option value="0.05">5% Budget</option>
              <option value="0.10">10% Budget</option>
              <option value="0.20">20% Budget</option>
            </select>
          </div>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/20 border border-rose-800/40 text-rose-200 text-xs">
          {error}
        </div>
      )}

      {/* Main Chart */}
      <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <h4 className="font-semibold text-white text-sm">
            Model Robustness Comparison (NDCG@10 Under Attack)
          </h4>
          <span className="text-xs text-vouch-soft">
            Soft-weighting + Exposure Governance retains highest utility under attack
          </span>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 40 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f4332" opacity={0.4} />
              <XAxis
                dataKey="name"
                stroke="#74c69d"
                fontSize={11}
                angle={-20}
                textAnchor="end"
                interval={0}
              />
              <YAxis stroke="#74c69d" fontSize={11} domain={[0, 1]} />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="p-3 bg-vouch-surface border border-vouch-border rounded-lg shadow-xl text-xs space-y-1">
                        <p className="font-semibold text-white">{d.fullName}</p>
                        <p className="text-emerald-400">Clean NDCG@10: {d.Clean_NDCG}</p>
                        <p className="text-rose-400">Attacked NDCG@10: {d.Attacked_NDCG}</p>
                        <p className="text-vouch-mint">Attacked Recall@20: {d.Recall_20}</p>
                        <p className="text-slate-300">Target Mean Rank: #{d.TargetRank}</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend
                wrapperStyle={{ paddingTop: 10, fontSize: 12 }}
                formatter={(val) => <span className="text-slate-300">{val}</span>}
              />
              <Bar dataKey="Clean_NDCG" fill="#52b788" name="Clean NDCG@10" radius={[4, 4, 0, 0]} barSize={20} />
              <Bar dataKey="Attacked_NDCG" fill="#ef4444" name="Attacked NDCG@10" radius={[4, 4, 0, 0]} barSize={20} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Comparison Table */}
      {data && (
        <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border">
          <h4 className="font-semibold text-white text-sm mb-3">Model Evaluation Matrix</h4>
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-vouch-surface border-b border-vouch-border text-[11px] font-semibold uppercase tracking-wider text-vouch-soft">
                <tr>
                  <th className="py-2.5 px-3">Model Architecture</th>
                  <th className="py-2.5 px-3">Clean NDCG@10</th>
                  <th className="py-2.5 px-3">Attacked NDCG@10</th>
                  <th className="py-2.5 px-3">NDCG Delta (Drop)</th>
                  <th className="py-2.5 px-3">Attacked Recall@20</th>
                  <th className="py-2.5 px-3">Target Mean Rank</th>
                  <th className="py-2.5 px-3">Jumps Capped</th>
                  <th className="py-2.5 px-3 text-right">Runtime</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-vouch-border/50">
                {data.models.map((m) => {
                  const isBest = m.model_name.includes('Exposure Governor');
                  return (
                    <tr
                      key={m.model_name}
                      className={`hover:bg-vouch-surface/50 transition-colors ${
                        isBest ? 'bg-emerald-950/20 font-medium' : ''
                      }`}
                    >
                      <td className="py-2.5 px-3 font-medium text-white flex items-center gap-2">
                        {isBest && <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0" />}
                        <span>{m.model_name}</span>
                      </td>
                      <td className="py-2.5 px-3 font-mono text-emerald-400">
                        {m.clean_ndcg_at_10.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-slate-200">
                        {m.attacked_ndcg_at_10.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 font-mono">
                        <span
                          className={`px-1.5 py-0.5 rounded text-[11px] ${
                            m.ndcg_delta < -0.2
                              ? 'text-rose-400 bg-rose-950/40'
                              : 'text-amber-400 bg-amber-950/40'
                          }`}
                        >
                          {(m.ndcg_delta * 100).toFixed(1)}%
                        </span>
                      </td>
                      <td className="py-2.5 px-3 font-mono text-slate-300">
                        {m.attacked_recall_at_20.toFixed(4)}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-slate-300">
                        #{m.target_mean_rank.toFixed(1)}
                      </td>
                      <td className="py-2.5 px-3 font-mono">
                        {m.jumps_capped > 0 ? (
                          <span className="text-amber-400 font-semibold">{m.jumps_capped}</span>
                        ) : (
                          <span className="text-slate-500">0</span>
                        )}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-right text-vouch-soft">
                        {m.runtime_seconds.toFixed(2)}s
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
