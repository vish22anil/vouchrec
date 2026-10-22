import React, { useState } from 'react';
import {
  ArrowLeft,
  LayoutDashboard,
  Compass,
  Fingerprint,
  Zap,
  Sliders,
  FileSearch,
  FlaskConical,
  AlertTriangle,
  CheckCircle2,
} from 'lucide-react';
import { OverviewView } from '../views/OverviewView';
import { RecommendationView } from '../views/RecommendationView';
import { TrustAnalysisView } from '../views/TrustAnalysisView';
import { AttackSimulatorView } from '../views/AttackSimulatorView';
import { DefenseComparisonView } from '../views/DefenseComparisonView';
import { GovernanceAuditView } from '../views/GovernanceAuditView';
import { ExperimentsView } from '../views/ExperimentsView';

export type LabTabId =
  | 'overview'
  | 'recommendations'
  | 'trust'
  | 'simulator'
  | 'defenses'
  | 'governance'
  | 'experiments';

interface ResearchLabProps {
  onBackToAnalyzer: () => void;
  isFixture: boolean;
  category: string;
}

export const ResearchLab: React.FC<ResearchLabProps> = ({
  onBackToAnalyzer,
  isFixture,
  category,
}) => {
  const [activeTab, setActiveTab] = useState<LabTabId>('overview');

  const navItems: { id: LabTabId; label: string; icon: React.FC<{ className?: string }> }[] = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'recommendations', label: 'Recommendation Explorer', icon: Compass },
    { id: 'trust', label: 'Trust Signals', icon: Fingerprint },
    { id: 'simulator', label: 'Attack Simulator', icon: Zap },
    { id: 'defenses', label: 'Defense Benchmark', icon: Sliders },
    { id: 'governance', label: 'Governance Audit', icon: FileSearch },
    { id: 'experiments', label: 'Statistical Tests', icon: FlaskConical },
  ];

  const renderActiveView = () => {
    switch (activeTab) {
      case 'overview':
        return <OverviewView />;
      case 'recommendations':
        return <RecommendationView />;
      case 'trust':
        return <TrustAnalysisView />;
      case 'simulator':
        return <AttackSimulatorView />;
      case 'defenses':
        return <DefenseComparisonView />;
      case 'governance':
        return <GovernanceAuditView />;
      case 'experiments':
        return <ExperimentsView />;
      default:
        return <OverviewView />;
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-150">
      {/* Research Lab Header Bar */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 p-5 rounded-2xl bg-surface-white dark:bg-night-surface border border-stone-warm dark:border-night-border transition-colors">
        <div className="flex items-center gap-3">
          <button
            onClick={onBackToAnalyzer}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-stone-warm dark:border-night-border hover:bg-ivory dark:hover:bg-night-card text-xs font-medium text-ink dark:text-night-text transition-colors"
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Trust Analyzer</span>
          </button>
          <div className="h-4 w-px bg-stone-warm dark:bg-night-border" />
          <div>
            <h2 className="font-serif text-lg font-bold text-ink dark:text-night-text">
              VouchRec Research Laboratory
            </h2>
            <p className="text-xs text-slate-muted dark:text-night-muted">
              Full-scale experimental suite, graph neural networks, and exposure governance audits.
            </p>
          </div>
        </div>

        {/* Telemetry info */}
        <div className="flex items-center gap-2.5 shrink-0 flex-wrap text-xs">
          <div className="px-2.5 py-1 rounded-md bg-ivory dark:bg-night-card border border-stone-warm dark:border-night-border font-mono text-slate-muted dark:text-night-muted">
            dataset: <span className="text-ink dark:text-night-text font-medium">{category}</span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-emerald-50 dark:bg-emerald-950/40 border border-emerald-200 dark:border-emerald-800 text-emerald-800 dark:text-emerald-300 font-medium">
            <CheckCircle2 className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
            <span>FastAPI Live</span>
          </div>
        </div>
      </div>

      {/* Prominent Preliminary Fixture Notice (Always retained!) */}
      {isFixture && (
        <div className="p-3.5 rounded-xl bg-amber-50 dark:bg-amber-950/30 border border-amber-300 dark:border-amber-800/60 text-amber-900 dark:text-amber-200 text-xs flex items-start sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5 sm:mt-0" />
            <div>
              <span className="font-semibold uppercase tracking-wider text-[10px] px-1.5 py-0.5 rounded bg-amber-200 dark:bg-amber-900/60 text-amber-900 dark:text-amber-200 mr-2">
                Preliminary Fixture
              </span>
              <span>
                Metrics in this lab are evaluated on the verified benchmark fixture subset (N=8 users, 8 items). Full 3.0M Musical Instruments corpus can be processed via data pipeline.
              </span>
            </div>
          </div>
          <span className="font-mono text-[11px] text-amber-700 dark:text-amber-400 shrink-0 hidden md:inline">
            VOUCHREC_DATA_ROOT/fixture
          </span>
        </div>
      )}

      {/* Lab Module Tabs */}
      <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-none border-b border-stone-warm dark:border-night-border">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`flex items-center gap-2 px-3.5 py-2.5 rounded-t-lg text-xs font-medium whitespace-nowrap transition-all border-b-2 ${
                isActive
                  ? 'border-aubergine text-aubergine dark:text-copper bg-surface-white dark:bg-night-surface font-semibold shadow-xs'
                  : 'border-transparent text-slate-muted dark:text-night-muted hover:text-ink dark:hover:text-night-text hover:bg-surface-white/60 dark:hover:bg-night-surface/60'
              }`}
            >
              <Icon className="w-3.5 h-3.5" />
              <span>{item.label}</span>
            </button>
          );
        })}
      </div>

      {/* Active Lab Module View Container */}
      <div className="pt-2">
        {renderActiveView()}
      </div>
    </div>
  );
};
