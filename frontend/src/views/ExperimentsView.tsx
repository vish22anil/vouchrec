import React, { useEffect, useState } from 'react';
import {
  FlaskConical,
  CheckCircle2,
  Scale,
} from 'lucide-react';
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ZAxis,
  Cell,
} from 'recharts';
import { api } from '../api';
import type { ExperimentDetailResponse, StatisticalTest } from '../types';

export const ExperimentsView: React.FC = () => {
  const [expData, setExpData] = useState<ExperimentDetailResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadExperiment() {
      setLoading(true);
      setError(null);
      try {
        const res = await api.getExperimentDetail('phase7_multi_seed_benchmark');
        setExpData(res);
      } catch (err: any) {
        setError(err.message || 'Error loading Phase 7 experiment artifacts');
      } finally {
        setLoading(false);
      }
    }
    loadExperiment();
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex flex-col items-center gap-3 text-vouch-soft">
          <div className="w-8 h-8 border-2 border-vouch-green border-t-transparent rounded-full animate-spin" />
          <p className="text-sm">Loading Phase 7 multi-seed experiment artifacts...</p>
        </div>
      </div>
    );
  }

  if (error || !expData) {
    return (
      <div className="p-6 rounded-xl bg-rose-950/20 border border-rose-800/40 text-rose-200">
        <h3 className="font-semibold text-rose-300">Phase 7 Experiment Artifact Missing</h3>
        <p className="text-sm mt-1">{error}</p>
        <p className="text-xs text-rose-400/80 mt-2">
          Run <code className="font-mono bg-rose-900/40 px-1 py-0.5 rounded">python scripts/run_phase7_experiments.py</code> to generate artifacts.
        </p>
      </div>
    );
  }

  // Scatter chart data: Clean Utility (X) vs Attacked Utility (Y)
  const scatterData = Object.entries(expData.model_aggregates).map(([modelName, stats]: [string, any]) => {
    return {
      name: modelName
        .replace('Trust-Gated LightGCN', 'TG-LightGCN')
        .replace('Hard-filtered LightGCN', 'Hard-Filt LGCN')
        .replace('Vanilla LightGCN', 'Vanilla LGCN')
        .replace('with Exposure Governor', '+ Gov'),
      fullName: modelName,
      cleanNDCG: Number(stats.clean_ndcg_at_10_mean.toFixed(4)),
      cleanStd: Number(stats.clean_ndcg_at_10_std.toFixed(4)),
      attackedNDCG: Number(stats.attacked_ndcg_at_10_mean.toFixed(4)),
      attackedStd: Number(stats.attacked_ndcg_at_10_std.toFixed(4)),
      attackedRecall: Number(stats.attacked_recall_at_20_mean.toFixed(4)),
      isGoverned: modelName.includes('Exposure Governor'),
    };
  });

  return (
    <div className="space-y-6">
      {/* Experiment Header & Manifest */}
      <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border space-y-3">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-vouch-border pb-3">
          <div className="flex items-center gap-2">
            <FlaskConical className="w-5 h-5 text-vouch-green" />
            <h3 className="font-semibold text-white text-base">{expData.title}</h3>
          </div>
          <span className="text-xs font-mono px-2.5 py-1 rounded bg-amber-950/40 border border-amber-600/40 text-amber-300">
            Benchmark Fixture Protocol (5 Seeds)
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3 text-xs">
          <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border space-y-1">
            <span className="text-[10px] uppercase font-mono text-vouch-soft">Dataset Split</span>
            <p className="font-medium text-white">{expData.manifest.dataset ?? 'Musical_Instruments (Fixture)'}</p>
            <p className="text-[11px] text-vouch-soft">5-core partition</p>
          </div>
          <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border space-y-1">
            <span className="text-[10px] uppercase font-mono text-vouch-soft">Seed Distribution</span>
            <p className="font-mono text-vouch-pale font-medium">
              [{expData.manifest.seeds ? expData.manifest.seeds.join(', ') : '42, 43, 44, 45, 46'}]
            </p>
            <p className="text-[11px] text-vouch-soft">5 distinct replications</p>
          </div>
          <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border space-y-1">
            <span className="text-[10px] uppercase font-mono text-vouch-soft">Adversary Configuration</span>
            <p className="font-medium text-white">
              {expData.manifest.attack_strategy} ({expData.manifest.attack_objective})
            </p>
            <p className="text-[11px] text-vouch-soft">
              Budget: {(expData.manifest.attack_budget_fraction * 100).toFixed(0)}%
            </p>
          </div>
          <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border space-y-1">
            <span className="text-[10px] uppercase font-mono text-vouch-soft">Alpha Significance</span>
            <p className="font-mono text-emerald-400 font-bold">α = {expData.manifest.significance_level ?? 0.05}</p>
            <p className="text-[11px] text-vouch-soft">Holm-Bonferroni FWER control</p>
          </div>
        </div>
      </div>

      {/* Trade-off Scatter Plot: Clean Utility vs Attacked Robustness */}
      <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <div>
            <h4 className="font-semibold text-white text-sm">
              Clean Utility vs. Adversarial Robustness Trade-off
            </h4>
            <p className="text-xs text-vouch-soft">
              Top-right quadrant represents Pareto-optimal models retaining high clean utility and adversarial robustness.
            </p>
          </div>
          <span className="text-xs text-vouch-soft font-mono">X: Clean NDCG@10 | Y: Attacked NDCG@10</span>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1f4332" opacity={0.4} />
              <XAxis
                type="number"
                dataKey="cleanNDCG"
                name="Clean NDCG@10"
                domain={[0.4, 0.7]}
                stroke="#74c69d"
                fontSize={11}
                label={{ value: 'Clean NDCG@10 (Utility)', position: 'insideBottom', offset: -10, fill: '#74c69d', fontSize: 11 }}
              />
              <YAxis
                type="number"
                dataKey="attackedNDCG"
                name="Attacked NDCG@10"
                domain={[0.3, 0.7]}
                stroke="#74c69d"
                fontSize={11}
                label={{ value: 'Attacked NDCG@10 (Robustness)', angle: -90, position: 'insideLeft', fill: '#74c69d', fontSize: 11 }}
              />
              <ZAxis range={[100, 100]} />
              <Tooltip
                cursor={{ strokeDasharray: '3 3' }}
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="p-3 bg-vouch-surface border border-vouch-border rounded-lg shadow-xl text-xs space-y-1">
                        <p className="font-bold text-white">{d.fullName}</p>
                        <p className="text-emerald-400">Clean NDCG@10: {d.cleanNDCG} ± {d.cleanStd}</p>
                        <p className="text-rose-400">Attacked NDCG@10: {d.attackedNDCG} ± {d.attackedStd}</p>
                        <p className="text-vouch-mint">Attacked Recall@20: {d.attackedRecall}</p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Scatter name="Models" data={scatterData}>
                {scatterData.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={entry.isGoverned ? '#52b788' : '#74c69d'}
                    stroke={entry.isGoverned ? '#ffffff' : '#1f4332'}
                    strokeWidth={entry.isGoverned ? 2 : 1}
                  />
                ))}
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model Aggregate Performance Table with 95% CIs */}
      <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border">
        <h4 className="font-semibold text-white text-sm mb-3">
          Cross-Seed Aggregate Metrics (Mean ± Std & 95% Confidence Intervals)
        </h4>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-vouch-surface border-b border-vouch-border text-[11px] font-semibold uppercase tracking-wider text-vouch-soft">
              <tr>
                <th className="py-2.5 px-3">Model Architecture</th>
                <th className="py-2.5 px-3">Clean NDCG@10 (Mean ± Std)</th>
                <th className="py-2.5 px-3">Clean 95% CI</th>
                <th className="py-2.5 px-3">Attacked NDCG@10 (Mean ± Std)</th>
                <th className="py-2.5 px-3">Attacked 95% CI</th>
                <th className="py-2.5 px-3">Target Mean Rank</th>
                <th className="py-2.5 px-3">Jumps Capped</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-vouch-border/50">
              {Object.entries(expData.model_aggregates).map(([mName, s]: [string, any]) => (
                <tr key={mName} className="hover:bg-vouch-surface/40 transition-colors">
                  <td className="py-2.5 px-3 font-medium text-white">{mName}</td>
                  <td className="py-2.5 px-3 font-mono text-emerald-400">
                    {s.clean_ndcg_at_10_mean.toFixed(4)} ± {s.clean_ndcg_at_10_std.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-400">
                    [{s.clean_ndcg_at_10_ci_lower.toFixed(3)}, {s.clean_ndcg_at_10_ci_upper.toFixed(3)}]
                  </td>
                  <td className="py-2.5 px-3 font-mono text-rose-400">
                    {s.attacked_ndcg_at_10_mean.toFixed(4)} ± {s.attacked_ndcg_at_10_std.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-400">
                    [{s.attacked_ndcg_at_10_ci_lower.toFixed(3)}, {s.attacked_ndcg_at_10_ci_upper.toFixed(3)}]
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-300">
                    #{s.target_mean_rank_mean.toFixed(1)}
                  </td>
                  <td className="py-2.5 px-3 font-mono">
                    {s.jumps_capped_mean > 0 ? (
                      <span className="text-amber-400 font-semibold">{s.jumps_capped_mean.toFixed(1)}</span>
                    ) : (
                      <span className="text-slate-500">0</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Paired Statistical Significance Tests (Holm-Bonferroni Correction) */}
      <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border space-y-3">
        <div className="flex items-center gap-2">
          <Scale className="w-5 h-5 text-vouch-green" />
          <h4 className="font-semibold text-white text-sm">
            Paired Statistical Significance Tests (Holm-Bonferroni FWER Correction)
          </h4>
        </div>
        <p className="text-xs text-vouch-soft">
          Two-sided paired t-tests comparing models across matched seeds. Multiplicity controlled via step-down
          Holm-Bonferroni thresholds with Cohen's $d_z$ effect sizes.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs text-slate-300">
            <thead className="bg-vouch-surface border-b border-vouch-border text-[11px] font-semibold uppercase tracking-wider text-vouch-soft">
              <tr>
                <th className="py-2.5 px-3">Condition A vs Condition B</th>
                <th className="py-2.5 px-3">Metric</th>
                <th className="py-2.5 px-3">Mean Diff</th>
                <th className="py-2.5 px-3">t-Stat</th>
                <th className="py-2.5 px-3">Raw p-value</th>
                <th className="py-2.5 px-3">Holm α Threshold</th>
                <th className="py-2.5 px-3">Adjusted p-value</th>
                <th className="py-2.5 px-3">Cohen's d_z</th>
                <th className="py-2.5 px-3">Statistical Conclusion</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-vouch-border/50">
              {expData.statistical_tests.map((test: StatisticalTest, idx: number) => (
                <tr key={idx} className="hover:bg-vouch-surface/40 transition-colors">
                  <td className="py-2.5 px-3 text-white font-medium">
                    <span className="text-emerald-400">{test.condition_a}</span>
                    <span className="text-vouch-soft"> vs. </span>
                    <span className="text-slate-400">{test.condition_b}</span>
                  </td>
                  <td className="py-2.5 px-3 font-mono text-vouch-pale">{test.metric_name}</td>
                  <td className="py-2.5 px-3 font-mono text-slate-200">
                    {test.mean_diff > 0 ? `+${test.mean_diff.toFixed(4)}` : test.mean_diff.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-300">{test.t_statistic.toFixed(3)}</td>
                  <td className="py-2.5 px-3 font-mono text-slate-300">{test.p_value.toFixed(4)}</td>
                  <td className="py-2.5 px-3 font-mono text-slate-400">{test.holm_threshold.toFixed(4)}</td>
                  <td className="py-2.5 px-3 font-mono font-bold text-white">
                    {test.adjusted_p_value.toFixed(4)}
                  </td>
                  <td className="py-2.5 px-3 font-mono text-slate-300">
                    d_z = {test.cohens_d.toFixed(2)}
                  </td>
                  <td className="py-2.5 px-3">
                    {test.is_significant_holm ? (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold bg-emerald-950/60 border border-emerald-600/50 text-emerald-300">
                        <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                        Significant (p &lt; α_adj)
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-medium bg-slate-800/60 border border-slate-700 text-slate-400">
                        Not Significant
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
