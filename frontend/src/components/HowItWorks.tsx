import React from 'react';

export const HowItWorks: React.FC = () => {
  return (
    <section id="how-it-works" className="max-w-4xl mx-auto py-12 px-4 border-t border-stone-warm dark:border-night-border transition-colors">
      <div className="text-center max-w-xl mx-auto mb-10 space-y-2">
        <h2 className="font-serif text-2xl sm:text-3xl font-normal text-ink dark:text-night-text">
          How review analysis works
        </h2>
        <p className="text-xs sm:text-sm text-slate-muted dark:text-night-muted">
          A three-step signal inspection process grounded in linguistic and graph recommendation research.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Step 1 */}
        <div className="p-6 rounded-2xl bg-surface-white dark:bg-night-surface border border-stone-warm dark:border-night-border space-y-3 transition-colors">
          <div className="w-9 h-9 rounded-lg bg-aubergine-light dark:bg-aubergine-dark/40 text-aubergine dark:text-copper flex items-center justify-center font-mono font-bold text-sm">
            01
          </div>
          <h3 className="font-semibold text-sm text-ink dark:text-night-text flex items-center gap-2">
            <span>Submit a review</span>
          </h3>
          <p className="text-xs text-slate-muted dark:text-night-muted leading-relaxed">
            Provide a product review link or paste the raw review text directly. The system verifies whether the item exists in the research dataset.
          </p>
        </div>

        {/* Step 2 */}
        <div className="p-6 rounded-2xl bg-surface-white dark:bg-night-surface border border-stone-warm dark:border-night-border space-y-3 transition-colors">
          <div className="w-9 h-9 rounded-lg bg-aubergine-light dark:bg-aubergine-dark/40 text-aubergine dark:text-copper flex items-center justify-center font-mono font-bold text-sm">
            02
          </div>
          <h3 className="font-semibold text-sm text-ink dark:text-night-text flex items-center gap-2">
            <span>Analyze available signals</span>
          </h3>
          <p className="text-xs text-slate-muted dark:text-night-muted leading-relaxed">
            We evaluate vocabulary diversity (type-token ratio), repetitive phrase clusters, punctuation anomalies, and historical catalog graphs when available.
          </p>
        </div>

        {/* Step 3 */}
        <div className="p-6 rounded-2xl bg-surface-white dark:bg-night-surface border border-stone-warm dark:border-night-border space-y-3 transition-colors">
          <div className="w-9 h-9 rounded-lg bg-aubergine-light dark:bg-aubergine-dark/40 text-aubergine dark:text-copper flex items-center justify-center font-mono font-bold text-sm">
            03
          </div>
          <h3 className="font-semibold text-sm text-ink dark:text-night-text flex items-center gap-2">
            <span>Review trust assessment</span>
          </h3>
          <p className="text-xs text-slate-muted dark:text-night-muted leading-relaxed">
            Inspect a heuristic risk score (0–100), key linguistic anomalies, and explicit evidence limitations. Scores are estimates, not calibrated proof of fraud.
          </p>
        </div>
      </div>
    </section>
  );
};
