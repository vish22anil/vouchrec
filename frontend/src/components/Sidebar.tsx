import React from 'react';
import {
  LayoutDashboard,
  Compass,
  Fingerprint,
  Zap,
  Sliders,
  FileSearch,
  FlaskConical,
  ShieldCheck,
  ExternalLink,
} from 'lucide-react';

export type TabId =
  | 'overview'
  | 'recommendations'
  | 'trust'
  | 'simulator'
  | 'defenses'
  | 'governance'
  | 'experiments';

interface SidebarProps {
  activeTab: TabId;
  setActiveTab: (tab: TabId) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ activeTab, setActiveTab }) => {
  const navItems: { id: TabId; label: string; icon: React.FC<{ className?: string }> }[] = [
    { id: 'overview', label: 'Overview', icon: LayoutDashboard },
    { id: 'recommendations', label: 'Recommendation Explorer', icon: Compass },
    { id: 'trust', label: 'Trust Signal Analysis', icon: Fingerprint },
    { id: 'simulator', label: 'Attack Simulator', icon: Zap },
    { id: 'defenses', label: 'Defense Comparison', icon: Sliders },
    { id: 'governance', label: 'Governance Audit Trail', icon: FileSearch },
    { id: 'experiments', label: 'Statistical Experiments', icon: FlaskConical },
  ];

  return (
    <aside className="w-64 bg-vouch-surface border-r border-vouch-border flex flex-col h-screen select-none shrink-0">
      {/* Brand Header */}
      <div className="p-5 border-b border-vouch-border flex items-center gap-3">
        <div className="w-10 h-10 rounded-lg bg-vouch-accent/30 border border-vouch-highlight/40 flex items-center justify-center text-vouch-green shadow-sm">
          <ShieldCheck className="w-6 h-6 text-vouch-green" />
        </div>
        <div>
          <h1 className="font-bold text-lg text-white tracking-tight flex items-center gap-1.5">
            VouchRec
            <span className="text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded bg-vouch-accent text-vouch-pale border border-vouch-highlight/30">
              Lab
            </span>
          </h1>
          <p className="text-xs text-vouch-soft/80">Trust-Gated Recommendation</p>
        </div>
      </div>

      {/* Navigation Links */}
      <nav className="flex-1 p-3 space-y-1.5 overflow-y-auto">
        <div className="px-3 py-1 text-[11px] font-semibold uppercase tracking-wider text-vouch-soft/60">
          Research Modules
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              className={`w-full flex items-center gap-3 px-3.5 py-2.5 rounded-lg text-sm font-medium transition-all ${
                isActive
                  ? 'bg-vouch-accent text-white shadow-sm border border-vouch-highlight/50'
                  : 'text-slate-300 hover:text-white hover:bg-vouch-card/80 border border-transparent'
              }`}
            >
              <Icon
                className={`w-4 h-4 shrink-0 ${
                  isActive ? 'text-vouch-pale' : 'text-vouch-soft/70'
                }`}
              />
              <span className="truncate">{item.label}</span>
            </button>
          );
        })}
      </nav>

      {/* Footer Info */}
      <div className="p-4 border-t border-vouch-border text-xs text-slate-400 bg-vouch-dark/40 space-y-2">
        <div className="flex items-center justify-between text-[11px] text-vouch-soft/70">
          <span>Engine: Python 3.11</span>
          <span>FastAPI 0.159</span>
        </div>
        <div className="flex items-center justify-between text-[11px]">
          <span className="text-slate-400">Milestone</span>
          <span className="text-vouch-green font-medium">Phase 8 Ready</span>
        </div>
        <div className="pt-1 text-[10px] text-slate-500 flex items-center justify-between">
          <span>Amazon Reviews 2023</span>
          <span className="flex items-center gap-1 text-vouch-soft hover:underline cursor-pointer">
            Docs <ExternalLink className="w-2.5 h-2.5" />
          </span>
        </div>
      </div>
    </aside>
  );
};
