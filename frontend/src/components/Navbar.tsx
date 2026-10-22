import React from 'react';
import { AlertTriangle, CheckCircle2, RefreshCw, Server, XCircle } from 'lucide-react';

interface NavbarProps {
  title: string;
  subtitle: string;
  isFixture: boolean;
  category: string;
  apiConnected: boolean;
  onRefresh: () => void;
  isLoading: boolean;
}

export const Navbar: React.FC<NavbarProps> = ({
  title,
  subtitle,
  isFixture,
  category,
  apiConnected,
  onRefresh,
  isLoading,
}) => {
  return (
    <header className="bg-vouch-surface border-b border-vouch-border sticky top-0 z-20 px-6 py-3.5">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
        {/* Title area */}
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
            {title}
          </h2>
          <p className="text-xs text-vouch-soft mt-0.5">{subtitle}</p>
        </div>

        {/* Status badges & Actions */}
        <div className="flex items-center gap-3 shrink-0 flex-wrap">
          {/* Category indicator */}
          <div className="px-2.5 py-1 rounded-md bg-vouch-card border border-vouch-border text-xs text-slate-300 font-mono">
            category: <span className="text-vouch-pale font-medium">{category}</span>
          </div>

          {/* Fixture warning badge */}
          {isFixture && (
            <div
              className="flex items-center gap-1.5 px-3 py-1 rounded-md bg-amber-950/40 border border-amber-600/40 text-amber-300 text-xs font-medium"
              title="Preliminary Benchmark Fixture active. Run full pipeline for 3.0M Amazon dataset."
            >
              <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />
              <span>Preliminary Fixture (N=8)</span>
            </div>
          )}

          {/* API Health indicator */}
          <div
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium border ${
              apiConnected
                ? 'bg-emerald-950/40 border-emerald-600/40 text-emerald-300'
                : 'bg-rose-950/40 border-rose-600/40 text-rose-300'
            }`}
          >
            <Server className="w-3.5 h-3.5 shrink-0" />
            {apiConnected ? (
              <>
                <CheckCircle2 className="w-3 h-3 text-emerald-400" />
                <span>API Live</span>
              </>
            ) : (
              <>
                <XCircle className="w-3 h-3 text-rose-400" />
                <span>API Offline</span>
              </>
            )}
          </div>

          {/* Refresh button */}
          <button
            onClick={onRefresh}
            disabled={isLoading}
            className="p-1.5 rounded-md bg-vouch-card hover:bg-vouch-border text-slate-300 hover:text-white border border-vouch-border transition-colors disabled:opacity-50"
            title="Refresh View Data"
          >
            <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin text-vouch-green' : ''}`} />
          </button>
        </div>
      </div>

      {/* Prominent fixture notice banner across top of view */}
      {isFixture && (
        <div className="mt-2.5 px-3 py-1.5 rounded bg-amber-950/30 border border-amber-600/30 text-[11px] text-amber-200/90 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="font-semibold text-amber-300 uppercase tracking-wide text-[10px] px-1 py-0.5 rounded bg-amber-900/50">
              Research Notice
            </span>
            <span>
              Metrics displayed are evaluated on the verified benchmark fixture subset. Full 3.0M Musical Instruments
              dataset can be ingested with zero code changes.
            </span>
          </div>
          <span className="text-[10px] text-amber-400/80 font-mono hidden sm:inline">VOUCHREC_DATA_ROOT/fixture</span>
        </div>
      )}
    </header>
  );
};
