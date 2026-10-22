import React from 'react';
import { Sun, Moon, FlaskConical, Shield } from 'lucide-react';

interface HeaderProps {
  currentView: 'home' | 'lab';
  setCurrentView: (view: 'home' | 'lab') => void;
  isDark: boolean;
  setIsDark: (dark: boolean) => void;
  onOpenHowItWorks: () => void;
  onOpenAbout: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentView,
  setCurrentView,
  isDark,
  setIsDark,
  onOpenHowItWorks,
  onOpenAbout,
}) => {
  return (
    <header className="border-b border-stone-warm dark:border-night-border bg-surface-white dark:bg-night-surface sticky top-0 z-30 transition-colors">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <button
            onClick={() => setCurrentView('home')}
            className="flex items-center gap-2 group text-left focus:outline-none"
          >
            <div className="w-8 h-8 rounded-md bg-aubergine text-white flex items-center justify-center font-serif text-lg font-bold shadow-sm transition-transform group-hover:scale-105">
              V
            </div>
            <div>
              <span className="font-serif text-xl font-bold tracking-tight text-ink dark:text-night-text">
                VouchRec
              </span>
              <span className="hidden sm:inline-block ml-2 text-xs text-slate-muted dark:text-night-muted">
                Trust Intelligence
              </span>
            </div>
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex items-center gap-2 sm:gap-4 text-sm font-medium">
          {currentView === 'home' ? (
            <>
              <button
                onClick={onOpenHowItWorks}
                className="px-2.5 py-1.5 rounded-md text-slate-muted dark:text-night-muted hover:text-ink dark:hover:text-night-text transition-colors text-xs sm:text-sm"
              >
                How it works
              </button>

              <button
                onClick={onOpenAbout}
                className="px-2.5 py-1.5 rounded-md text-slate-muted dark:text-night-muted hover:text-ink dark:hover:text-night-text transition-colors text-xs sm:text-sm"
              >
                About
              </button>

              <button
                onClick={() => setCurrentView('lab')}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-md border border-stone-warm dark:border-night-border hover:border-aubergine dark:hover:border-aubergine-border text-slate-muted dark:text-night-muted hover:text-aubergine dark:hover:text-white transition-all text-xs sm:text-sm"
                title="Open comprehensive research evaluation laboratory"
              >
                <FlaskConical className="w-3.5 h-3.5" />
                <span className="hidden xs:inline">Research Lab</span>
                <span className="xs:hidden">Lab</span>
              </button>
            </>
          ) : (
            <button
              onClick={() => setCurrentView('home')}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-aubergine text-white text-xs sm:text-sm font-medium hover:bg-aubergine-hover transition-colors shadow-sm"
            >
              <Shield className="w-3.5 h-3.5" />
              <span>Trust Analyzer</span>
            </button>
          )}

          {/* Theme Toggle */}
          <button
            onClick={() => setIsDark(!isDark)}
            aria-label={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
            className="p-2 rounded-md border border-stone-warm dark:border-night-border hover:bg-ivory dark:hover:bg-night-card text-ink dark:text-night-text transition-colors"
            title={isDark ? 'Switch to light mode' : 'Switch to dark mode'}
          >
            {isDark ? (
              <Sun className="w-4 h-4 text-copper" />
            ) : (
              <Moon className="w-4 h-4 text-aubergine" />
            )}
          </button>
        </nav>
      </div>
    </header>
  );
};
