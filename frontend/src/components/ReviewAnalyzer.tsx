import React, { useState } from 'react';
import {
  Link as LinkIcon,
  FileText,
  AlertCircle,
  ArrowRight,
  RotateCcw,
  Info,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  Quote,
  HelpCircle,
} from 'lucide-react';
import { api } from '../api';
import type { SingleReviewAnalysisResponse } from '../types';

interface ReviewAnalyzerProps {
  onGoToLab?: () => void;
}

export const ReviewAnalyzer: React.FC<ReviewAnalyzerProps> = ({ onGoToLab }) => {
  const [mode, setMode] = useState<'url' | 'text'>('url');
  const [urlInput, setUrlInput] = useState<string>('');
  const [textInput, setTextInput] = useState<string>('');
  const [ratingInput, setRatingInput] = useState<number | null>(5);
  const [loading, setLoading] = useState<boolean>(false);
  const [result, setResult] = useState<SingleReviewAnalysisResponse | null>(null);
  const [clientError, setClientError] = useState<string | null>(null);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState<boolean>(false);
  const [selectedReviewIdx, setSelectedReviewIdx] = useState<number>(0);

  // Preset example data
  const EXAMPLES = {
    soundcoreUrl: 'https://www.amazon.in/dp/B0C3HCD34R',
    boatUrl: 'https://www.amazon.in/dp/B071Z8M4KX',
    iphoneUrl: 'https://www.amazon.in/dp/B09G9HD6PD',
    datasetUrl: 'https://www.amazon.com/dp/P001',
    authenticText:
      'This instrument exceeded my expectations. The fretwork is smooth along the entire neck, tuning pegs hold tension nicely over multiple sessions, and the tone is surprisingly warm and balanced for this price bracket. Packed securely with zero blemishes upon unboxing.',
    suspiciousText:
      'GREAT GREAT GREAT PRODUCT! BUY BUY BUY! BEST EVER BEST EVER SO GOOD AMAZING AMAZING! EVERYONE MUST BUY THIS RIGHT NOW BEST PRODUCT EVER!',
    shortText: 'Works ok, nothing special.',
  };

  const handleAnalyze = async (overrideUrl?: string, overrideText?: string) => {
    const activeUrl = overrideUrl !== undefined ? overrideUrl : urlInput;
    const activeText = overrideText !== undefined ? overrideText : textInput;

    setClientError(null);

    if (mode === 'url' && !activeUrl.trim()) {
      setClientError('Please enter a product review link.');
      return;
    }

    if (mode === 'text' && (!activeText || activeText.trim().length < 3)) {
      setClientError('Please enter at least 3 characters of review text.');
      return;
    }

    setLoading(true);
    setResult(null);
    setSelectedReviewIdx(0);

    try {
      const res = await api.analyzeReview({
        url: mode === 'url' ? activeUrl.trim() : null,
        review_text: mode === 'text' ? activeText.trim() : null,
        rating: mode === 'text' ? ratingInput : null,
      });
      setResult(res);
      setSelectedReviewIdx(0);
    } catch (err: any) {
      setClientError(err.message || 'An error occurred during analysis.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setUrlInput('');
    setTextInput('');
    setResult(null);
    setClientError(null);
    setSelectedReviewIdx(0);
  };

  const switchToTextWithPrefill = (text?: string) => {
    setMode('text');
    if (text) setTextInput(text);
    setResult(null);
    setClientError(null);
    setSelectedReviewIdx(0);
  };

  // Determine active review if scraped list exists
  const activeReview =
    result?.all_reviews && result.all_reviews.length > 0
      ? result.all_reviews[selectedReviewIdx] || result.all_reviews[0]
      : null;

  const currentScore = activeReview?.trust_score ?? result?.trust_score;
  const currentRiskLevel = activeReview?.risk_level ?? result?.risk_level;
  const currentSignals =
    activeReview?.signals_summary && activeReview.signals_summary.length > 0
      ? activeReview.signals_summary
      : result?.strongest_signals || [];
  const currentReviewText = activeReview?.review_text ?? result?.review_text ?? (mode === 'text' ? textInput : '');
  const currentAuthor = activeReview?.author ?? result?.user_id ?? 'Amazon Reviewer';
  const currentRating = activeReview?.rating ?? result?.rating ?? null;
  const isVerified = activeReview?.verified_purchase ?? false;
  const reviewDate = activeReview?.date ?? '';
  const reviewTitle = activeReview?.title ?? '';
  const activeBreakdown =
    activeReview?.score_breakdown && activeReview.score_breakdown.length > 0
      ? activeReview.score_breakdown
      : result?.score_breakdown || [];
  const activeNarrative = activeReview?.score_narrative || result?.score_narrative || '';

  return (
    <div className="space-y-10">
      {/* Editorial Hero Section */}
      <div className="text-center max-w-2xl mx-auto space-y-3 pt-6 pb-2">
        <h1 className="font-serif text-3xl sm:text-5xl font-normal tracking-tight text-ink dark:text-night-text leading-tight">
          Not every review tells the whole story.
        </h1>
        <p className="text-base sm:text-lg text-slate-muted dark:text-night-muted font-normal leading-relaxed">
          Analyze review signals and understand what may make a review suspicious.
        </p>
      </div>

      {/* Main Analyzer Card */}
      <div className="max-w-2xl mx-auto bg-surface-white dark:bg-night-surface rounded-2xl border border-stone-warm dark:border-night-border shadow-sm overflow-hidden transition-colors">
        {/* Input Mode Selector Tabs */}
        <div className="flex border-b border-stone-warm dark:border-night-border bg-ivory/50 dark:bg-night-card/40">
          <button
            onClick={() => {
              setMode('url');
              setClientError(null);
            }}
            className={`flex-1 py-3 px-4 text-xs sm:text-sm font-medium flex items-center justify-center gap-2 transition-all ${
              mode === 'url'
                ? 'bg-surface-white dark:bg-night-surface text-ink dark:text-night-text border-b-2 border-aubergine font-semibold'
                : 'text-slate-muted dark:text-night-muted hover:text-ink dark:hover:text-night-text'
            }`}
          >
            <LinkIcon className="w-3.5 h-3.5" />
            <span>Review Link</span>
          </button>
          <button
            onClick={() => {
              setMode('text');
              setClientError(null);
            }}
            className={`flex-1 py-3 px-4 text-xs sm:text-sm font-medium flex items-center justify-center gap-2 transition-all ${
              mode === 'text'
                ? 'bg-surface-white dark:bg-night-surface text-ink dark:text-night-text border-b-2 border-aubergine font-semibold'
                : 'text-slate-muted dark:text-night-muted hover:text-ink dark:hover:text-night-text'
            }`}
          >
            <FileText className="w-3.5 h-3.5" />
            <span>Paste Review Text</span>
          </button>
        </div>

        <div className="p-5 sm:p-7 space-y-4">
          {mode === 'url' ? (
            <div className="space-y-3">
              <label htmlFor="url-input" className="sr-only">
                Paste product review link
              </label>
              <div className="flex flex-col sm:flex-row gap-2.5">
                <div className="relative flex-1">
                  <input
                    id="url-input"
                    type="url"
                    value={urlInput}
                    onChange={(e) => setUrlInput(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleAnalyze()}
                    placeholder="Paste an Amazon product or review link here…"
                    className="w-full px-4 py-3 rounded-lg border border-stone-warm dark:border-night-border bg-ivory/30 dark:bg-night-card text-ink dark:text-night-text placeholder-slate-muted/70 text-sm focus:outline-none focus:ring-2 focus:ring-aubergine/20 focus:border-aubergine transition-all"
                  />
                </div>
                <button
                  onClick={() => handleAnalyze()}
                  disabled={loading}
                  className="px-6 py-3 rounded-lg bg-aubergine hover:bg-aubergine-hover text-white font-medium text-sm transition-all shadow-sm disabled:opacity-50 shrink-0 flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      <span>Analyzing…</span>
                    </>
                  ) : (
                    <span>Analyze review</span>
                  )}
                </button>
              </div>

              {/* Example Links */}
              <div className="flex flex-wrap items-center gap-2 text-xs text-slate-muted dark:text-night-muted pt-1">
                <span className="font-medium">Try an example:</span>
                <button
                  type="button"
                  onClick={() => {
                    setUrlInput(EXAMPLES.soundcoreUrl);
                    handleAnalyze(EXAMPLES.soundcoreUrl);
                  }}
                  className="underline hover:text-aubergine dark:hover:text-copper transition-colors font-medium text-ink dark:text-night-text"
                  title="Real live product: soundcore Q20i headphones"
                >
                  Soundcore Q20i (Live)
                </button>
                <span>•</span>
                <button
                  type="button"
                  onClick={() => {
                    setUrlInput(EXAMPLES.boatUrl);
                    handleAnalyze(EXAMPLES.boatUrl);
                  }}
                  className="underline hover:text-aubergine dark:hover:text-copper transition-colors"
                  title="Real live product: boAt Bassheads earphones"
                >
                  boAt Earphones (Live)
                </button>
                <span>•</span>
                <button
                  type="button"
                  onClick={() => {
                    setUrlInput(EXAMPLES.datasetUrl);
                    handleAnalyze(EXAMPLES.datasetUrl);
                  }}
                  className="underline hover:text-aubergine dark:hover:text-copper transition-colors"
                  title="Verified research dataset item"
                >
                  Dataset Item P001
                </button>
              </div>
            </div>
          ) : (
            <div className="space-y-3">
              <label htmlFor="text-input" className="sr-only">
                Paste review text
              </label>
              <textarea
                id="text-input"
                rows={4}
                value={textInput}
                onChange={(e) => setTextInput(e.target.value)}
                placeholder="Paste the product review text here…"
                className="w-full p-4 rounded-lg border border-stone-warm dark:border-night-border bg-ivory/30 dark:bg-night-card text-ink dark:text-night-text placeholder-slate-muted/70 text-sm focus:outline-none focus:ring-2 focus:ring-aubergine/20 focus:border-aubergine transition-all leading-relaxed resize-none"
              />

              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pt-1">
                {/* Optional Star Rating Selector */}
                <div className="flex items-center gap-2 text-xs text-slate-muted dark:text-night-muted">
                  <span>Rating (optional):</span>
                  <div className="flex items-center gap-1">
                    {[1, 2, 3, 4, 5].map((star) => (
                      <button
                        key={star}
                        type="button"
                        onClick={() => setRatingInput(star)}
                        className={`px-2 py-0.5 rounded text-xs border font-medium ${
                          ratingInput === star
                            ? 'bg-copper text-white border-copper'
                            : 'border-stone-warm dark:border-night-border text-slate-muted dark:text-night-muted hover:border-copper'
                        }`}
                      >
                        ★ {star}
                      </button>
                    ))}
                  </div>
                </div>

                <button
                  onClick={() => handleAnalyze()}
                  disabled={loading}
                  className="px-6 py-2.5 rounded-lg bg-aubergine hover:bg-aubergine-hover text-white font-medium text-sm transition-all shadow-sm disabled:opacity-50 shrink-0 flex items-center justify-center gap-2 self-start sm:self-auto"
                >
                  {loading ? (
                    <>
                      <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                      <span>Analyzing…</span>
                    </>
                  ) : (
                    <span>Analyze review</span>
                  )}
                </button>
              </div>

              {/* Example Texts */}
              <div className="flex flex-wrap items-center gap-2 text-xs text-slate-muted dark:text-night-muted pt-1">
                <span>Try an example:</span>
                <button
                  type="button"
                  onClick={() => {
                    setTextInput(EXAMPLES.authenticText);
                    setRatingInput(5);
                    handleAnalyze(undefined, EXAMPLES.authenticText);
                  }}
                  className="underline hover:text-aubergine dark:hover:text-copper transition-colors"
                >
                  Authentic Review
                </button>
                <span>•</span>
                <button
                  type="button"
                  onClick={() => {
                    setTextInput(EXAMPLES.suspiciousText);
                    setRatingInput(5);
                    handleAnalyze(undefined, EXAMPLES.suspiciousText);
                  }}
                  className="underline hover:text-aubergine dark:hover:text-copper transition-colors"
                >
                  Repetitive Shilling
                </button>
                <span>•</span>
                <button
                  type="button"
                  onClick={() => {
                    setTextInput(EXAMPLES.shortText);
                    setRatingInput(3);
                    handleAnalyze(undefined, EXAMPLES.shortText);
                  }}
                  className="underline hover:text-aubergine dark:hover:text-copper transition-colors"
                >
                  Brief / Minimal
                </button>
              </div>
            </div>
          )}

          {/* Privacy & Helper Text */}
          <p className="text-[11px] text-slate-muted/80 dark:text-night-muted/80 pt-1 text-center">
            Results depend on the review data available. A risk score is an estimate, not proof of fraud.
          </p>

          {/* Client Error Banner */}
          {clientError && (
            <div className="p-3.5 rounded-lg bg-rose-50 dark:bg-rose-950/30 border border-rose-200 dark:border-rose-900/50 text-rose-800 dark:text-rose-200 text-xs flex items-center gap-2.5">
              <AlertCircle className="w-4 h-4 text-rose-600 dark:text-rose-400 shrink-0" />
              <span>{clientError}</span>
            </div>
          )}
        </div>
      </div>

      {/* CAPTCHA / Anti-Bot Challenge Notice */}
      {result && result.status === 'captcha_challenge' && (
        <div className="max-w-2xl mx-auto p-5 sm:p-6 rounded-2xl bg-amber-50 dark:bg-amber-950/20 border border-amber-200 dark:border-amber-800/40 space-y-3 transition-colors">
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
            <div className="space-y-1.5">
              <h3 className="font-semibold text-sm text-ink dark:text-night-text">
                Amazon Verification Notice
              </h3>
              <p className="text-xs text-slate-muted dark:text-night-muted leading-relaxed">
                {result.error_message}
              </p>
            </div>
          </div>
          <div className="pt-2 flex flex-col sm:flex-row items-center gap-3">
            <button
              onClick={() => switchToTextWithPrefill()}
              className="w-full sm:w-auto px-4 py-2 rounded-lg bg-aubergine hover:bg-aubergine-hover text-white text-xs font-semibold shadow-sm transition-colors flex items-center justify-center gap-2"
            >
              <span>Paste review text directly</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <span className="text-xs text-slate-muted dark:text-night-muted">
              Copy any review text from the Amazon tab you have open.
            </span>
          </div>
        </div>
      )}

      {/* Unsupported URL Fallback Card */}
      {result && result.status === 'unsupported_url' && (
        <div className="max-w-2xl mx-auto p-5 sm:p-6 rounded-2xl bg-copper-light/60 dark:bg-copper-dark/20 border border-copper-border dark:border-copper/40 space-y-3 transition-colors">
          <div className="flex items-start gap-3">
            <Info className="w-5 h-5 text-copper shrink-0 mt-0.5" />
            <div className="space-y-1.5">
              <h3 className="font-semibold text-sm text-ink dark:text-night-text">
                Platform URL Notice
              </h3>
              <p className="text-xs text-slate-muted dark:text-night-muted leading-relaxed">
                {result.error_message}
              </p>
            </div>
          </div>
          <div className="pt-2 flex flex-col sm:flex-row items-center gap-3">
            <button
              onClick={() => switchToTextWithPrefill()}
              className="w-full sm:w-auto px-4 py-2 rounded-lg bg-copper hover:bg-copper-hover text-white text-xs font-semibold shadow-sm transition-colors flex items-center justify-center gap-2"
            >
              <span>Paste review text instead</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <span className="text-xs text-slate-muted dark:text-night-muted">
              or try one of the live Amazon examples above.
            </span>
          </div>
        </div>
      )}

      {/* Analysis Results View */}
      {result && result.success && currentScore !== null && currentScore !== undefined && (
        <div className="max-w-2xl mx-auto bg-surface-white dark:bg-night-surface rounded-2xl border border-stone-warm dark:border-night-border shadow-sm p-6 sm:p-8 space-y-6 transition-colors">
          {/* Live Amazon Product Card */}
          {result.source_type === 'live_amazon' && result.product_title && (
            <div className="p-4 rounded-xl bg-ivory/80 dark:bg-night-card border border-stone-warm dark:border-night-border flex flex-col sm:flex-row items-start sm:items-center gap-4 transition-colors">
              {result.product_image && (
                <img
                  src={result.product_image}
                  alt={result.product_title}
                  className="w-16 h-16 object-contain rounded-lg bg-white p-1 border border-stone-warm/50 shrink-0"
                />
              )}
              <div className="space-y-1 flex-1 min-w-0">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300 font-mono">
                    <CheckCircle2 className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
                    <span>Live Amazon Data</span>
                  </span>
                  {result.asin && (
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-mono text-slate-muted dark:text-night-muted bg-stone-warm/40 dark:bg-night-surface border border-stone-warm/60">
                      ASIN: {result.asin}
                    </span>
                  )}
                  {result.product_rating && (
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold text-amber-800 dark:text-amber-300 bg-amber-50 dark:bg-amber-950/40 border border-amber-200 dark:border-amber-900/50">
                      ★ {result.product_rating} / 5
                    </span>
                  )}
                  {result.total_ratings_text && (
                    <span className="text-[10px] text-slate-muted dark:text-night-muted">
                      {result.total_ratings_text}
                    </span>
                  )}
                </div>
                <h3 className="text-sm font-semibold text-ink dark:text-night-text truncate" title={result.product_title}>
                  {result.product_title}
                </h3>
              </div>
            </div>
          )}

          {/* Multiple Scraped Reviews Carousel / Selector */}
          {result.all_reviews && result.all_reviews.length > 1 && (
            <div className="space-y-2 pt-1">
              <div className="flex items-center justify-between text-xs">
                <span className="font-semibold text-ink dark:text-night-text">
                  Customer Reviews on this Listing ({result.all_reviews.length})
                </span>
                <span className="text-slate-muted dark:text-night-muted text-[11px]">
                  Click to inspect individual review signals
                </span>
              </div>
              <div className="flex items-center gap-2.5 overflow-x-auto pb-2 scrollbar-thin">
                {result.all_reviews.map((rev, idx) => {
                  const isSelected = selectedReviewIdx === idx;
                  return (
                    <button
                      key={idx}
                      type="button"
                      onClick={() => setSelectedReviewIdx(idx)}
                      className={`text-left p-3 rounded-xl border text-xs shrink-0 w-60 transition-all ${
                        isSelected
                          ? 'border-aubergine bg-aubergine/5 dark:bg-aubergine/20 ring-1 ring-aubergine/40 shadow-sm'
                          : 'border-stone-warm dark:border-night-border bg-surface-white dark:bg-night-card hover:border-aubergine/40'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-semibold text-[11px] truncate max-w-[130px] text-ink dark:text-night-text">
                          {rev.author}
                        </span>
                        {rev.trust_score !== null && rev.trust_score !== undefined && (
                          <span
                            className={`px-1.5 py-0.5 rounded text-[10px] font-mono font-bold ${
                              rev.risk_level === 'Lower risk'
                                ? 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300'
                                : rev.risk_level === 'Needs a closer look'
                                ? 'bg-copper-light text-copper-hover'
                                : 'bg-rose-100 text-rose-800'
                            }`}
                          >
                            {rev.trust_score}/100
                          </span>
                        )}
                      </div>
                      <div className="flex items-center gap-1.5 text-[10px] text-slate-muted dark:text-night-muted mb-1.5">
                        {rev.rating && <span className="text-amber-600 font-bold">★ {rev.rating}</span>}
                        {rev.verified_purchase && (
                          <span className="text-emerald-700 dark:text-emerald-400 font-semibold">• Verified</span>
                        )}
                      </div>
                      <p className="text-[11px] text-slate-muted dark:text-night-muted line-clamp-2 leading-tight">
                        {rev.title ? `"${rev.title}"` : rev.review_text}
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Active Review Quote Card */}
          {currentReviewText && (
            <div className="p-4 rounded-xl bg-ivory/50 dark:bg-night-card/40 border border-stone-warm/80 dark:border-night-border/80 space-y-2">
              <div className="flex flex-wrap items-center justify-between gap-2 border-b border-stone-warm/60 dark:border-night-border/60 pb-2 text-xs">
                <div className="flex items-center gap-2">
                  <Quote className="w-3.5 h-3.5 text-copper shrink-0" />
                  <span className="font-semibold text-ink dark:text-night-text">{currentAuthor}</span>
                  {currentRating && (
                    <span className="text-amber-600 font-bold text-[11px]">★ {currentRating}</span>
                  )}
                  {isVerified && (
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-semibold bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300">
                      Verified Purchase
                    </span>
                  )}
                </div>
                {reviewDate && (
                  <span className="text-[11px] text-slate-muted dark:text-night-muted">{reviewDate}</span>
                )}
              </div>

              {reviewTitle && (
                <h4 className="text-xs font-bold text-ink dark:text-night-text">{reviewTitle}</h4>
              )}

              <p className="text-xs text-ink/90 dark:text-night-text/90 italic leading-relaxed">
                "{currentReviewText}"
              </p>
            </div>
          )}

          {/* Header & Risk Badge */}
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-stone-warm dark:border-night-border pb-6">
            <div>
              <span className="text-xs uppercase tracking-wider text-slate-muted dark:text-night-muted font-mono font-medium">
                Trust Assessment
              </span>
              <h2 className="font-serif text-2xl font-normal text-ink dark:text-night-text mt-1">
                {currentRiskLevel}
              </h2>
              <p className="text-xs text-slate-muted dark:text-night-muted mt-1 leading-relaxed max-w-md">
                {result.risk_label_description}
              </p>
            </div>

            {/* Score Pill */}
            <div className="flex flex-col items-start sm:items-end">
              <div
                className={`inline-flex items-baseline gap-1 px-4 py-2 rounded-xl border text-xl font-bold font-mono ${
                  currentRiskLevel === 'Lower risk'
                    ? 'bg-emerald-50 dark:bg-emerald-950/30 text-emerald-800 dark:text-emerald-300 border-emerald-200 dark:border-emerald-800'
                    : currentRiskLevel === 'Needs a closer look'
                    ? 'bg-copper-light dark:bg-copper-dark/30 text-copper-hover dark:text-copper border-copper-border dark:border-copper/40'
                    : 'bg-rose-50 dark:bg-rose-950/30 text-rose-800 dark:text-rose-300 border-rose-200 dark:border-rose-900'
                }`}
              >
                <span>{currentScore}</span>
                <span className="text-xs font-normal text-slate-muted dark:text-night-muted">/100</span>
              </div>
              <span className="text-[10px] text-slate-muted dark:text-night-muted mt-1 font-mono">
                Heuristic trust index
              </span>
            </div>
          </div>

          {/* Horizontal Score Bar */}
          <div className="space-y-1.5">
            <div className="flex justify-between text-[11px] text-slate-muted dark:text-night-muted font-medium">
              <span>Higher risk (0)</span>
              <span>Needs a closer look (45)</span>
              <span>Lower risk (70+)</span>
            </div>
            <div className="w-full h-3 bg-stone-warm/50 dark:bg-night-card rounded-full overflow-hidden relative">
              <div
                className={`h-full rounded-full transition-all duration-500 ${
                  currentRiskLevel === 'Lower risk'
                    ? 'bg-emerald-600 dark:bg-emerald-500'
                    : currentRiskLevel === 'Needs a closer look'
                    ? 'bg-copper'
                    : 'bg-rose-600 dark:bg-rose-500'
                }`}
                style={{ width: `${Math.min(100, Math.max(5, currentScore ?? 0))}%` }}
              />
            </div>
          </div>

          {/* Why This Review Received This Score Breakdown */}
          {(activeNarrative || activeBreakdown.length > 0) && (
            <div className="space-y-3 pt-1">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <h3 className="text-xs font-semibold text-ink dark:text-night-text uppercase tracking-wider flex items-center gap-1.5">
                    <HelpCircle className="w-3.5 h-3.5 text-aubergine dark:text-copper" />
                    <span>Why This Review Received {currentScore}/100</span>
                  </h3>
                </div>
                <span className="text-[10px] font-mono font-medium text-slate-muted dark:text-night-muted uppercase">
                  Point Calibration Ledger
                </span>
              </div>

              {/* Score Narrative Summary */}
              {activeNarrative && (
                <div className="p-3.5 rounded-xl bg-ivory/90 dark:bg-night-card border border-stone-warm dark:border-night-border text-xs leading-relaxed space-y-1 transition-colors">
                  <p className="font-medium text-ink dark:text-night-text">
                    {activeNarrative}
                  </p>
                  <p className="text-[11px] text-slate-muted dark:text-night-muted">
                    VouchRec begins from an unskewed 50-point prior. Verified purchase authenticity and descriptive substance add trust points; low effort, extreme repetition, or shouting deduct points.
                  </p>
                </div>
              )}

              {/* Itemized Score Factors */}
              {activeBreakdown.length > 0 && (
                <div className="rounded-xl border border-stone-warm dark:border-night-border overflow-hidden bg-surface-white dark:bg-night-card/40 shadow-xs">
                  <div className="divide-y divide-stone-warm/60 dark:divide-night-border/60">
                    {activeBreakdown.map((factor, idx) => {
                      const isBonus = factor.category === 'bonus' || factor.points > 0;
                      const isPenalty = factor.category === 'penalty' || factor.points < 0;
                      const isBase = factor.name.toLowerCase().includes('base prior');

                      return (
                        <div
                          key={idx}
                          className="p-3 flex items-start justify-between gap-3 text-xs hover:bg-ivory/40 dark:hover:bg-night-card/60 transition-colors"
                        >
                          <div className="space-y-0.5 flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="font-semibold text-ink dark:text-night-text">
                                {factor.name}
                              </span>
                              {isBase && (
                                <span className="text-[10px] px-1.5 py-0.2 rounded font-mono bg-stone-warm/50 dark:bg-night-surface text-slate-muted dark:text-night-muted">
                                  Neutral Starting Baseline
                                </span>
                              )}
                            </div>
                            <p className="text-[11px] text-slate-muted dark:text-night-muted leading-tight">
                              {factor.detail}
                            </p>
                          </div>

                          <div className="shrink-0 text-right">
                            <span
                              className={`inline-block px-2.5 py-0.5 rounded-md font-mono font-bold text-xs ${
                                isBonus && !isBase
                                  ? 'bg-emerald-100 dark:bg-emerald-950/70 text-emerald-800 dark:text-emerald-300'
                                  : isPenalty
                                  ? 'bg-rose-100 dark:bg-rose-950/70 text-rose-800 dark:text-rose-300'
                                  : 'bg-stone-warm/70 dark:bg-night-border text-slate-muted dark:text-night-muted'
                              }`}
                            >
                              {factor.points > 0 ? `+${factor.points}` : factor.points} pts
                            </span>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  {/* Calculated Total Footer */}
                  <div className="p-3 bg-ivory/80 dark:bg-night-card border-t border-stone-warm dark:border-night-border flex items-center justify-between text-xs">
                    <span className="font-semibold text-ink dark:text-night-text">
                      Calibrated Trust Score:
                    </span>
                    <div className="flex items-center gap-1.5 font-mono font-bold text-sm">
                      <span
                        className={
                          currentRiskLevel === 'Lower risk'
                            ? 'text-emerald-700 dark:text-emerald-400'
                            : currentRiskLevel === 'Needs a closer look'
                            ? 'text-copper-hover dark:text-copper'
                            : 'text-rose-700 dark:text-rose-400'
                        }
                      >
                        {currentScore}
                      </span>
                      <span className="text-xs font-normal text-slate-muted dark:text-night-muted">/ 100</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Strongest Supporting Signals */}
          <div className="space-y-2">
            <h3 className="text-xs font-semibold text-ink dark:text-night-text uppercase tracking-wider">
              Strongest Signals Detected
            </h3>
            <ul className="space-y-1.5">
              {currentSignals.map((sig, idx) => (
                <li
                  key={idx}
                  className="text-xs text-ink dark:text-night-text flex items-start gap-2 bg-ivory/50 dark:bg-night-card/40 p-2.5 rounded-lg border border-stone-warm/60 dark:border-night-border/60"
                >
                  <span className="text-aubergine dark:text-copper mt-0.5">•</span>
                  <span className="leading-relaxed">{sig}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Signal Categories Breakdown */}
          <div className="space-y-3 pt-2">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-semibold text-ink dark:text-night-text uppercase tracking-wider">
                Signal Category Overview
              </h3>
              <button
                type="button"
                onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
                className="text-xs text-aubergine dark:text-copper hover:underline flex items-center gap-1 font-medium"
              >
                <span>{showTechnicalDetails ? 'Hide details' : 'Show details'}</span>
                {showTechnicalDetails ? (
                  <ChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5" />
                )}
              </button>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
              {/* Text Head */}
              <div className="p-3 rounded-xl border border-stone-warm dark:border-night-border bg-ivory/30 dark:bg-night-card/30 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-ink dark:text-night-text">1. Review Text</span>
                  <span className="text-[10px] px-1.5 py-0.5 rounded font-mono bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300 font-semibold">
                    Evaluated
                  </span>
                </div>
                <p className="text-[11px] text-slate-muted dark:text-night-muted leading-relaxed">
                  {result.text_signals?.explanation ||
                    'Evaluates lexical diversity, repetition, and punctuation anomaly.'}
                </p>
              </div>

              {/* Behavior Head */}
              <div className="p-3 rounded-xl border border-stone-warm dark:border-night-border bg-ivory/30 dark:bg-night-card/30 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-ink dark:text-night-text">2. Reviewer Behavior</span>
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded font-mono font-medium ${
                      result.behavior_signals?.status === 'evaluated'
                        ? 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300'
                        : 'bg-stone-warm dark:bg-night-border text-slate-muted dark:text-night-muted'
                    }`}
                  >
                    {result.behavior_signals?.status === 'evaluated' ? 'Evaluated' : 'Unavailable'}
                  </span>
                </div>
                <p className="text-[11px] text-slate-muted dark:text-night-muted leading-relaxed">
                  {result.behavior_signals?.explanation ||
                    'Reviewer velocity and historical rating extremity.'}
                </p>
              </div>

              {/* Coordination Head */}
              <div className="p-3 rounded-xl border border-stone-warm dark:border-night-border bg-ivory/30 dark:bg-night-card/30 space-y-1.5">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-ink dark:text-night-text">3. Coordination</span>
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded font-mono font-medium ${
                      result.coordination_signals?.status === 'evaluated'
                        ? 'bg-emerald-100 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300'
                        : 'bg-stone-warm dark:bg-night-border text-slate-muted dark:text-night-muted'
                    }`}
                  >
                    {result.coordination_signals?.status === 'evaluated' ? 'Evaluated' : 'Unavailable'}
                  </span>
                </div>
                <p className="text-[11px] text-slate-muted dark:text-night-muted leading-relaxed">
                  {result.coordination_signals?.explanation ||
                    'Cross-product temporal bursts and co-review clusters.'}
                </p>
              </div>
            </div>

            {/* Expandable Technical Details Panel */}
            {showTechnicalDetails && result.text_signals?.metrics && (
              <div className="p-4 rounded-xl bg-ivory dark:bg-night-card border border-stone-warm dark:border-night-border space-y-2 text-xs">
                <span className="font-semibold text-ink dark:text-night-text">
                  Extracted Linguistic Metrics:
                </span>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 font-mono text-[11px] text-slate-muted dark:text-night-muted pt-1">
                  <div>
                    Word count:{' '}
                    <span className="font-bold text-ink dark:text-night-text">
                      {result.text_signals.metrics.word_count}
                    </span>
                  </div>
                  <div>
                    Characters:{' '}
                    <span className="font-bold text-ink dark:text-night-text">
                      {result.text_signals.metrics.character_length}
                    </span>
                  </div>
                  <div>
                    Lexical diversity:{' '}
                    <span className="font-bold text-ink dark:text-night-text">
                      {((result.text_signals.metrics.lexical_diversity || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div>
                    Repetition ratio:{' '}
                    <span className="font-bold text-ink dark:text-night-text">
                      {((result.text_signals.metrics.repetition_ratio || 0) * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Evidence Limitations Notice */}
          <div className="p-4 rounded-xl bg-ivory/60 dark:bg-night-card/50 border border-stone-warm dark:border-night-border space-y-1.5 text-xs text-slate-muted dark:text-night-muted">
            <span className="font-semibold text-ink dark:text-night-text flex items-center gap-1.5">
              <Info className="w-3.5 h-3.5 text-copper" />
              <span>Evidence Limitations</span>
            </span>
            <ul className="list-disc list-inside space-y-0.5 text-[11px] leading-relaxed">
              {result.limitations.map((lim, i) => (
                <li key={i}>{lim}</li>
              ))}
            </ul>
          </div>

          {/* Action Row */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2 border-t border-stone-warm dark:border-night-border">
            <button
              onClick={handleReset}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg border border-stone-warm dark:border-night-border hover:bg-ivory dark:hover:bg-night-card text-ink dark:text-night-text text-xs font-medium transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Analyze another review</span>
            </button>

            {onGoToLab && (
              <button
                onClick={onGoToLab}
                className="text-xs text-aubergine dark:text-copper hover:underline font-medium"
              >
                Inspect full graph models in Research Lab →
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
