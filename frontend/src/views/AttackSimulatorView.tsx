import React, { useEffect, useState } from 'react';
import {
  Zap,
  AlertTriangle,
  Play,
  CheckCircle2,
  Lock,
  ArrowRight,
} from 'lucide-react';
import { api } from '../api';
import type { AttackSimulationResponse, UserItemSummary } from '../types';

export const AttackSimulatorView: React.FC = () => {
  const [items, setItems] = useState<UserItemSummary[]>([]);
  const [strategy, setStrategy] = useState<string>('bandwagon');
  const [objective, setObjective] = useState<string>('push');
  const [targetItem, setTargetItem] = useState<string>('P001');
  const [budgetFraction, setBudgetFraction] = useState<number>(0.1);
  const [burstWindow, setBurstWindow] = useState<number | null>(48);
  const [seed, setSeed] = useState<number>(42);

  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<AttackSimulationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    async function loadItems() {
      try {
        const res = await api.getItems();
        setItems(res.items);
        if (res.items.length > 0 && res.items[0].item_id) {
          setTargetItem(res.items[0].item_id);
        }
      } catch (err: any) {
        console.error('Failed to load items', err);
      }
    }
    loadItems();
  }, []);

  const handleSimulate = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.simulateAttack({
        strategy,
        objective,
        target_item: targetItem,
        budget_fraction: budgetFraction,
        filler_size: 2,
        burst_window_hours: burstWindow,
        seed,
      });
      setResult(res);
    } catch (err: any) {
      setError(err.message || 'Error executing attack simulation');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Info & Quarantine Guarantee */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 p-5 rounded-xl bg-vouch-card border border-vouch-border space-y-2">
          <div className="flex items-center gap-2">
            <Zap className="w-5 h-5 text-amber-400" />
            <h3 className="font-semibold text-white text-base">Controlled Manipulation Attack Simulator</h3>
          </div>
          <p className="text-xs text-vouch-soft leading-relaxed">
            Inject synthetic shilling profiles into an isolated, in-memory partition to rigorously test recommendation
            robustness. Synthetic attack labels are kept in strict evaluation quarantine and are <strong>never</strong> fed
            into trust-scoring feature pipelines.
          </p>
        </div>

        {/* Quarantine Badge */}
        <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border flex flex-col justify-between">
          <div className="flex items-center gap-2 text-xs font-semibold text-emerald-400">
            <Lock className="w-4 h-4" />
            <span>Label Quarantine Active</span>
          </div>
          <p className="text-[11px] text-vouch-soft mt-1">
            Original dataset is immutable. Simulator labels are tagged as evaluation-only (<code className="text-vouch-pale font-mono">is_synthetic_attack=1</code>).
          </p>
          <div className="mt-2 text-[10px] text-emerald-300/80 flex items-center gap-1 font-mono">
            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
            <span>Anti-leakage verified</span>
          </div>
        </div>
      </div>

      {/* Simulator Control Configuration Panel */}
      <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border space-y-4">
        <h4 className="font-semibold text-white text-sm border-b border-vouch-border pb-3 flex items-center gap-2">
          <span>Simulation Hyperparameters</span>
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs">
          {/* Strategy */}
          <div className="space-y-1.5">
            <label className="text-vouch-soft font-medium">Attack Strategy:</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-vouch-surface border border-vouch-border text-white focus:outline-none focus:border-vouch-green"
            >
              <option value="bandwagon">Bandwagon Attack (Target + Popular items)</option>
              <option value="average">Average Attack (Target + Mean rating fillers)</option>
              <option value="random">Random Attack (Target + Random filler items)</option>
            </select>
          </div>

          {/* Objective */}
          <div className="space-y-1.5">
            <label className="text-vouch-soft font-medium">Attack Objective:</label>
            <select
              value={objective}
              onChange={(e) => setObjective(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-vouch-surface border border-vouch-border text-white focus:outline-none focus:border-vouch-green"
            >
              <option value="push">Push Attack (Artificially promote target rating = 5.0)</option>
              <option value="nuke">Nuke Attack (Artificially demote target rating = 1.0)</option>
            </select>
          </div>

          {/* Target Item */}
          <div className="space-y-1.5">
            <label className="text-vouch-soft font-medium">Target Item:</label>
            <select
              value={targetItem}
              onChange={(e) => setTargetItem(e.target.value)}
              className="w-full px-3 py-2 rounded-lg bg-vouch-surface border border-vouch-border text-white font-mono focus:outline-none focus:border-vouch-green"
            >
              {items.map((it) => (
                <option key={it.item_id} value={it.item_id}>
                  {it.item_id} ({it.interaction_count} reviews)
                </option>
              ))}
            </select>
          </div>

          {/* Attack Budget */}
          <div className="space-y-1.5">
            <label className="text-vouch-soft font-medium flex justify-between">
              <span>Attack Budget:</span>
              <span className="font-mono text-vouch-pale font-bold">{(budgetFraction * 100).toFixed(0)}%</span>
            </label>
            <input
              type="range"
              min={0.05}
              max={0.3}
              step={0.05}
              value={budgetFraction}
              onChange={(e) => setBudgetFraction(Number(e.target.value))}
              className="w-full accent-vouch-green cursor-pointer mt-2"
            />
          </div>

          {/* Burst Timing Window */}
          <div className="space-y-1.5">
            <label className="text-vouch-soft font-medium">Burst Window:</label>
            <select
              value={burstWindow ?? ''}
              onChange={(e) => setBurstWindow(e.target.value ? Number(e.target.value) : null)}
              className="w-full px-3 py-2 rounded-lg bg-vouch-surface border border-vouch-border text-white focus:outline-none focus:border-vouch-green"
            >
              <option value="">Uniform Random (No Burst)</option>
              <option value="24">24-Hour Concentrated Burst</option>
              <option value="48">48-Hour Concentrated Burst</option>
              <option value="72">72-Hour Concentrated Burst</option>
            </select>
          </div>

          {/* Random Seed */}
          <div className="space-y-1.5">
            <label className="text-vouch-soft font-medium">Random Seed (Reproducibility):</label>
            <input
              type="number"
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
              className="w-full px-3 py-2 rounded-lg bg-vouch-surface border border-vouch-border text-white font-mono focus:outline-none focus:border-vouch-green"
            />
          </div>
        </div>

        {/* Execution trigger */}
        <div className="pt-2 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <p className="text-[11px] text-vouch-soft">
            Clicking will synthesize profiles, train the attacked graph, and evaluate target ranking shift.
          </p>

          <button
            onClick={handleSimulate}
            disabled={loading}
            className="px-5 py-2.5 rounded-lg bg-amber-600 hover:bg-amber-500 text-white font-semibold text-xs shadow-md transition-colors disabled:opacity-50 flex items-center gap-2 self-start sm:self-auto"
          >
            {loading ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
                <span>Simulating Attack...</span>
              </>
            ) : (
              <>
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>Execute Simulation</span>
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-950/20 border border-rose-800/40 text-rose-200 text-xs">
          {error}
        </div>
      )}

      {/* Simulation Results & Impact Report */}
      {result && (
        <div className="space-y-4">
          {/* Warning banner if budget rounded up */}
          {result.budget_rounding_warning && (
            <div className="p-3.5 rounded-xl bg-amber-950/40 border border-amber-600/40 text-amber-200 text-xs flex items-center gap-2.5">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
              <span>{result.budget_rounding_warning}</span>
            </div>
          )}

          {/* Results Summary Cards */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border">
              <span className="text-vouch-soft text-xs">Synthetic Attackers</span>
              <p className="mt-1 text-2xl font-bold font-mono text-white">
                {result.synthetic_profiles_count}
              </p>
              <span className="text-[11px] text-vouch-soft">
                Effective: {(result.effective_budget_fraction * 100).toFixed(1)}% of users
              </span>
            </div>

            <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border">
              <span className="text-vouch-soft text-xs">Injected Reviews</span>
              <p className="mt-1 text-2xl font-bold font-mono text-amber-400">
                {result.synthetic_reviews_count}
              </p>
              <span className="text-[11px] text-vouch-soft">Target + filler ratings</span>
            </div>

            <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border">
              <span className="text-vouch-soft text-xs">Target HitRate@10 Shift</span>
              <p className="mt-1 text-2xl font-bold font-mono text-white flex items-center gap-1.5">
                <span>{result.impact_metrics.target_hitrate_at_10_clean ?? 0}</span>
                <ArrowRight className="w-4 h-4 text-vouch-soft" />
                <span className="text-amber-400">
                  {result.impact_metrics.target_hitrate_at_10_attacked ?? 0}
                </span>
              </p>
              <span className="text-[11px] text-vouch-soft">
                Delta: {result.impact_metrics.target_hitrate_delta ?? 0}
              </span>
            </div>

            <div className="p-4 rounded-xl bg-vouch-card border border-vouch-border">
              <span className="text-vouch-soft text-xs">Target Mean Rank Shift</span>
              <p className="mt-1 text-2xl font-bold font-mono text-white flex items-center gap-1.5">
                <span>#{result.impact_metrics.target_mean_rank_clean ?? '—'}</span>
                <ArrowRight className="w-4 h-4 text-vouch-soft" />
                <span className="text-emerald-400 font-bold">
                  #{result.impact_metrics.target_mean_rank_attacked ?? '—'}
                </span>
              </p>
              <span className="text-[11px] text-vouch-soft">
                Rank Jump: +{result.impact_metrics.target_rank_improvement ?? 0} positions
              </span>
            </div>
          </div>

          {/* Audit & Manifest Box */}
          <div className="p-5 rounded-xl bg-vouch-card border border-vouch-border text-xs space-y-3">
            <h5 className="font-semibold text-white text-sm">Simulation Manifest & Provenance</h5>
            <div className="p-3 rounded-lg bg-vouch-surface border border-vouch-border font-mono text-[11px] text-slate-300 overflow-x-auto space-y-1">
              <div>Strategy: {result.manifest.strategy}</div>
              <div>Objective: {result.manifest.objective}</div>
              <div>Target Item: {result.target_item}</div>
              <div>Requested Budget: {(result.requested_budget_fraction * 100).toFixed(0)}%</div>
              <div>Effective Budget: {(result.effective_budget_fraction * 100).toFixed(1)}%</div>
              <div>Random Seed: {result.manifest.seed}</div>
              <div>Burst Window: {result.manifest.burst_window_hours ? `${result.manifest.burst_window_hours}h` : 'None'}</div>
              <div>Evaluation Label Quarantine: Verified (Immutable baseline parquet preserved)</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
