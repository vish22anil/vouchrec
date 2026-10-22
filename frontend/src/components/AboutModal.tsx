import React from 'react';
import { X, FlaskConical } from 'lucide-react';

interface AboutModalProps {
  isOpen: boolean;
  onClose: () => void;
  onGoToLab: () => void;
}

export const AboutModal: React.FC<AboutModalProps> = ({ isOpen, onClose, onGoToLab }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-ink/40 dark:bg-black/60 backdrop-blur-xs animate-in fade-in duration-150">
      <div
        className="relative w-full max-w-lg bg-surface-white dark:bg-night-surface rounded-2xl border border-stone-warm dark:border-night-border shadow-xl p-6 sm:p-7 space-y-5 text-ink dark:text-night-text"
        role="dialog"
        aria-modal="true"
        aria-labelledby="about-modal-title"
      >
        <div className="flex items-center justify-between border-b border-stone-warm dark:border-night-border pb-4">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-md bg-aubergine text-white flex items-center justify-center font-serif text-lg font-bold">
              V
            </div>
            <div>
              <h2 id="about-modal-title" className="font-serif text-lg font-bold">
                About VouchRec
              </h2>
              <p className="text-xs text-slate-muted dark:text-night-muted">
                Trust Intelligence for Collaborative Filtering
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            aria-label="Close modal"
            className="p-1 rounded-md text-slate-muted hover:text-ink dark:hover:text-night-text"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <div className="space-y-3.5 text-xs text-slate-muted dark:text-night-muted leading-relaxed">
          <p>
            Online review platforms are routinely targeted by shilling campaigns, automated bots, and paid
            manipulation networks designed to artificially inflate product exposure.
          </p>
          <p>
            <strong>VouchRec</strong> is a scientific framework investigating how recommender systems can defend
            themselves against manipulation. It combines three provisional trust signal heads:
          </p>
          <ul className="list-disc list-inside space-y-1 pl-1 text-ink dark:text-night-text">
            <li><strong>Review-Text Signals:</strong> Lexical diversity, vocabulary richness, repetition.</li>
            <li><strong>Reviewer-Behavior Signals:</strong> Historical review velocity and rating extremity.</li>
            <li><strong>Coordination Signals:</strong> Co-review clustering across products and temporal bursts.</li>
          </ul>
          <p>
            Downstream, VouchRec feeds these trust scores into a <strong>Trust-Gated LightGCN</strong> graph neural network
            and a post-processing <strong>Dynamic Exposure Governor</strong> to prevent manipulated items from capturing
            unearned top-K exposure.
          </p>
          <div className="p-3 rounded-xl bg-ivory dark:bg-night-card border border-stone-warm dark:border-night-border text-[11px]">
            <span className="font-semibold text-ink dark:text-night-text">Methodology Disclosure: </span>
            A risk rating represents an empirical anomaly estimate based on available signal features, not confirmed proof of deceit.
          </div>
        </div>

        <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-3 border-t border-stone-warm dark:border-night-border">
          <button
            onClick={() => {
              onClose();
              onGoToLab();
            }}
            className="w-full sm:w-auto px-4 py-2 rounded-lg bg-aubergine hover:bg-aubergine-hover text-white text-xs font-semibold shadow-sm transition-colors flex items-center justify-center gap-1.5"
          >
            <FlaskConical className="w-3.5 h-3.5" />
            <span>Open Research Lab</span>
          </button>

          <button
            onClick={onClose}
            className="w-full sm:w-auto px-4 py-2 rounded-lg border border-stone-warm dark:border-night-border hover:bg-ivory dark:hover:bg-night-card text-xs font-medium text-ink dark:text-night-text transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};
