import React, { useEffect, useState } from 'react';
import { Header } from './components/Header';
import { ReviewAnalyzer } from './components/ReviewAnalyzer';
import { HowItWorks } from './components/HowItWorks';
import { AboutModal } from './components/AboutModal';
import { ResearchLab } from './components/ResearchLab';
import { api } from './api';
import { FlaskConical } from 'lucide-react';

export const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<'home' | 'lab'>('home');
  const [isDark, setIsDark] = useState<boolean>(() => {
    try {
      const saved = localStorage.getItem('vouchrec_theme');
      if (saved) return saved === 'dark';
      return window.matchMedia('(prefers-color-scheme: dark)').matches;
    } catch {
      return false;
    }
  });

  const [isAboutOpen, setIsAboutOpen] = useState<boolean>(false);
  const [isFixture, setIsFixture] = useState<boolean>(true);
  const [category, setCategory] = useState<string>('Musical_Instruments');

  // Sync theme with DOM and localStorage
  useEffect(() => {
    try {
      if (isDark) {
        document.documentElement.classList.add('dark');
        localStorage.setItem('vouchrec_theme', 'dark');
      } else {
        document.documentElement.classList.remove('dark');
        localStorage.setItem('vouchrec_theme', 'light');
      }
    } catch (err) {
      console.error('Failed to sync theme', err);
    }
  }, [isDark]);

  // Load telemetry metadata on mount
  useEffect(() => {
    async function loadMeta() {
      try {
        const summary = await api.getDatasetSummary();
        setIsFixture(summary.is_fixture);
        setCategory(summary.category);
      } catch (err) {
        console.error('Failed to load dataset metadata', err);
      }
    }
    loadMeta();
  }, []);

  const scrollToHowItWorks = () => {
    const el = document.getElementById('how-it-works');
    if (el) {
      el.scrollIntoView({ behavior: 'smooth' });
    }
  };

  return (
    <div className="min-h-screen bg-ivory dark:bg-night-bg text-ink dark:text-night-text flex flex-col font-sans transition-colors duration-200">
      {/* Top Header Navigation */}
      <Header
        currentView={currentView}
        setCurrentView={setCurrentView}
        isDark={isDark}
        setIsDark={setIsDark}
        onOpenHowItWorks={scrollToHowItWorks}
        onOpenAbout={() => setIsAboutOpen(true)}
      />

      {/* Main Container */}
      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 py-8">
        {currentView === 'home' ? (
          <div className="space-y-16">
            {/* Primary Product Experience: Review Analyzer */}
            <ReviewAnalyzer onGoToLab={() => setCurrentView('lab')} />

            {/* How It Works (3-step explanation) */}
            <HowItWorks />
          </div>
        ) : (
          /* Research Laboratory (7 Modules with Fixture Banner) */
          <ResearchLab
            onBackToAnalyzer={() => setCurrentView('home')}
            isFixture={isFixture}
            category={category}
          />
        )}
      </main>

      {/* Product Footer */}
      <footer className="border-t border-stone-warm dark:border-night-border bg-surface-white dark:bg-night-surface py-8 text-xs text-slate-muted dark:text-night-muted transition-colors mt-auto">
        <div className="max-w-5xl mx-auto px-4 sm:px-6 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <span className="font-serif font-bold text-ink dark:text-night-text">VouchRec</span>
            <span>—</span>
            <span>Know which reviews to trust.</span>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={() => setIsAboutOpen(true)}
              className="hover:text-ink dark:hover:text-night-text transition-colors"
            >
              About
            </button>
            <button
              onClick={scrollToHowItWorks}
              className="hover:text-ink dark:hover:text-night-text transition-colors"
            >
              Methodology
            </button>
            <button
              onClick={() => setCurrentView('lab')}
              className="text-aubergine dark:text-copper hover:underline font-medium flex items-center gap-1"
            >
              <FlaskConical className="w-3.5 h-3.5" />
              <span>Research Lab</span>
            </button>
          </div>
        </div>
      </footer>

      {/* About Modal */}
      <AboutModal
        isOpen={isAboutOpen}
        onClose={() => setIsAboutOpen(false)}
        onGoToLab={() => {
          setIsAboutOpen(false);
          setCurrentView('lab');
        }}
      />
    </div>
  );
};

export default App;
