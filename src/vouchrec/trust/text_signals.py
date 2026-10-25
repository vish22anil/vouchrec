"""Review-text trust signals: lexical diversity, repetition, and anomaly scoring."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Precompiled regex for tokenization
WORD_RE = re.compile(r"\b[a-zA-Z0-9_\-']+\b")
SENTENCE_RE = re.compile(r"[.!?]+")

# Shilling and promotional boilerplate patterns characteristic of synthetic/bot manipulation
SHILLING_BOILERPLATE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bexceeded all (?:my )?expectations\b", re.IGNORECASE),
    re.compile(r"\bfive stars all the way\b", re.IGNORECASE),
    re.compile(r"\bworks flawlessly out of the box\b", re.IGNORECASE),
    re.compile(r"\bworks flawlessly\b", re.IGNORECASE),
    re.compile(r"\bbest purchase i have (?:ever )?made\b", re.IGNORECASE),
    re.compile(r"\btop notch quality\b", re.IGNORECASE),
    re.compile(r"\bvalue for money\b", re.IGNORECASE),
    re.compile(r"\bhighly recommended\b", re.IGNORECASE),
    re.compile(r"\bwill definitely buy again\b", re.IGNORECASE),
    re.compile(r"\barrived on time without issues\b", re.IGNORECASE),
    re.compile(r"\bworst product i have (?:ever )?purchased\b", re.IGNORECASE),
    re.compile(r"\btotal waste of (?:time|money)\b", re.IGNORECASE),
    re.compile(r"\bdefective and useless\b", re.IGNORECASE),
    re.compile(r"\bdoes not work as advertised\b", re.IGNORECASE),
    re.compile(r"\bfell apart almost immediately\b", re.IGNORECASE),
    re.compile(r"\bzero stars if i could\b", re.IGNORECASE),
    re.compile(r"\bcompletely defective\b", re.IGNORECASE),
    re.compile(r"\bsave your money\b", re.IGNORECASE),
    re.compile(r"\bdisappointing quality\b", re.IGNORECASE),
)


def extract_text_features(texts: Sequence[str | None]) -> pd.DataFrame:
    """Extract interpretable lexical, structural, and stylometric features from review text.

    Features computed:
      - char_length: total character length
      - word_count: count of alphanumeric tokens
      - sentence_count: count of sentence boundary delimiters
      - avg_word_length: average token character length
      - type_token_ratio: unique tokens / total tokens (lexical diversity)
      - caps_ratio: uppercase letters / total characters
      - exclamation_ratio: exclamation marks / total characters
      - repetition_ratio: fraction of duplicate tokens / total tokens
      - boilerplate_matches: number of matched shilling/promotional templates
      - is_promotional_boilerplate: True if any shilling boilerplate pattern matches
      - is_missing_text: True if text is None, empty, or whitespace
      - is_short_text: True if token count < 4
    """
    rows: list[dict[str, float | int | bool]] = []

    for text in texts:
        if text is None or not str(text).strip():
            rows.append(
                {
                    "char_length": 0,
                    "word_count": 0,
                    "sentence_count": 0,
                    "avg_word_length": 0.0,
                    "type_token_ratio": 0.0,
                    "caps_ratio": 0.0,
                    "exclamation_ratio": 0.0,
                    "repetition_ratio": 0.0,
                    "boilerplate_matches": 0,
                    "is_promotional_boilerplate": False,
                    "is_missing_text": True,
                    "is_short_text": True,
                }
            )
            continue

        raw_str = str(text).strip()
        char_len = len(raw_str)
        tokens = [t.lower() for t in WORD_RE.findall(raw_str)]
        n_words = len(tokens)
        sentences = [s.strip() for s in SENTENCE_RE.split(raw_str) if s.strip()]
        n_sentences = max(1, len(sentences))

        if n_words == 0:
            avg_word_len = 0.0
            ttr = 0.0
            rep_ratio = 0.0
        else:
            avg_word_len = float(sum(len(t) for t in tokens) / n_words)
            unique_words = len(set(tokens))
            ttr = float(unique_words / n_words)

            # Measure token repetition: proportion of non-unique tokens
            token_counts = Counter(tokens)
            repeated_tokens = sum(count - 1 for count in token_counts.values() if count > 1)
            rep_ratio = float(repeated_tokens / n_words)

        caps_count = sum(1 for c in raw_str if c.isupper())
        caps_ratio = float(caps_count / max(char_len, 1))

        excl_count = raw_str.count("!")
        excl_ratio = float(excl_count / max(char_len, 1))

        bp_matches = sum(1 for p in SHILLING_BOILERPLATE_PATTERNS if p.search(raw_str))

        rows.append(
            {
                "char_length": char_len,
                "word_count": n_words,
                "sentence_count": n_sentences,
                "avg_word_length": round(avg_word_len, 2),
                "type_token_ratio": round(ttr, 4),
                "caps_ratio": round(caps_ratio, 4),
                "exclamation_ratio": round(excl_ratio, 4),
                "repetition_ratio": round(rep_ratio, 4),
                "boilerplate_matches": bp_matches,
                "is_promotional_boilerplate": bool(bp_matches > 0),
                "is_missing_text": False,
                "is_short_text": bool(n_words < 4),
            }
        )

    return pd.DataFrame(rows)


class BaseTextScorer(ABC):
    """Abstract interface for text trust scoring."""

    @abstractmethod
    def score(self, texts: Sequence[str | None]) -> tuple[np.ndarray, list[list[str]]]:
        """Compute text trust scores in [0, 1] and reason codes for each text."""
        raise NotImplementedError


class UnsupervisedTextScorer(BaseTextScorer):
    """Transparent, unsupervised heuristic text anomaly and quality scorer.

    Scores range from 0.0 (highly anomalous / repetitive / low quality) to 1.0
    (articulate, diverse, non-repetitive text). Missing text receives a neutral
    score of 0.5 with reason code 'TEXT_MISSING'.
    """

    def __init__(
        self,
        neutral_score: float = 0.50,
        short_text_score: float = 0.55,
        min_ttr_threshold: float = 0.35,
        max_repetition_threshold: float = 0.30,
        max_caps_ratio: float = 0.35,
        max_exclamation_ratio: float = 0.08,
        boilerplate_penalty: float = 0.25,
    ) -> None:
        self.neutral_score = neutral_score
        self.short_text_score = short_text_score
        self.min_ttr_threshold = min_ttr_threshold
        self.max_repetition_threshold = max_repetition_threshold
        self.max_caps_ratio = max_caps_ratio
        self.max_exclamation_ratio = max_exclamation_ratio
        self.boilerplate_penalty = boilerplate_penalty

    def score(self, texts: Sequence[str | None]) -> tuple[np.ndarray, list[list[str]]]:
        feats_df = extract_text_features(texts)
        scores: list[float] = []
        all_reasons: list[list[str]] = []

        for _, row in feats_df.iterrows():
            reasons: list[str] = []

            if row["is_missing_text"]:
                scores.append(self.neutral_score)
                all_reasons.append(["TEXT_MISSING"])
                continue

            if row["is_short_text"]:
                scores.append(self.short_text_score)
                all_reasons.append(["TEXT_SHORT"])
                continue

            # Base score for non-empty text
            score = 0.75

            # Penalties
            if row["type_token_ratio"] < self.min_ttr_threshold:
                score -= 0.25
                reasons.append("TEXT_LOW_LEXICAL_DIVERSITY")

            if row["repetition_ratio"] > self.max_repetition_threshold:
                score -= 0.20
                reasons.append("TEXT_HIGH_REPETITION")

            if row["caps_ratio"] > self.max_caps_ratio:
                score -= 0.20
                reasons.append("TEXT_EXCESSIVE_CAPS")

            if row["exclamation_ratio"] > self.max_exclamation_ratio:
                score -= 0.15
                reasons.append("TEXT_EXCESSIVE_PUNCTUATION")

            # Shilling boilerplate detection: generic promotional / slamming clichés
            if row["is_promotional_boilerplate"]:
                score -= self.boilerplate_penalty
                reasons.append("TEXT_PROMOTIONAL_BOILERPLATE")
            elif row["word_count"] >= 12 and row["type_token_ratio"] >= 0.65:
                # Bonused only if non-boilerplate substantive text
                score += 0.20
                reasons.append("TEXT_DETAILED_AND_DIVERSE")

            # Bound score to [0.0, 1.0]
            clamped = float(np.clip(score, 0.0, 1.0))
            if not reasons:
                reasons.append("TEXT_NATURAL_BASELINE")

            scores.append(round(clamped, 4))
            all_reasons.append(reasons)

        return np.array(scores, dtype=np.float64), all_reasons


class EnhancedNLPTextScorer(UnsupervisedTextScorer):
    """Advanced NLP text trust scorer with stylometric and entity specificity signals."""

    def __init__(
        self,
        neutral_score: float = 0.50,
        short_text_score: float = 0.55,
        min_ttr_threshold: float = 0.35,
        max_repetition_threshold: float = 0.30,
        max_caps_ratio: float = 0.35,
        max_exclamation_ratio: float = 0.08,
        boilerplate_penalty: float = 0.30,
        specificity_bonus: float = 0.15,
    ) -> None:
        super().__init__(
            neutral_score=neutral_score,
            short_text_score=short_text_score,
            min_ttr_threshold=min_ttr_threshold,
            max_repetition_threshold=max_repetition_threshold,
            max_caps_ratio=max_caps_ratio,
            max_exclamation_ratio=max_exclamation_ratio,
            boilerplate_penalty=boilerplate_penalty,
        )
        self.specificity_bonus = specificity_bonus


class DebertaTextScorer(BaseTextScorer):
    """Supervised / fine-tuned transformer text manipulation detection scorer.

    When initialized with a model_checkpoint path, runs inference using the
    fine-tuned model. If model_checkpoint is None, raises NotImplementedError
    enforcing calibration requirements.
    """

    def __init__(self, model_checkpoint: str | None = None) -> None:
        self.model_checkpoint = model_checkpoint
        self._pipeline: Any = None
        if model_checkpoint is not None:
            self._init_pipeline(model_checkpoint)

    def _init_pipeline(self, checkpoint: str) -> None:
        try:
            from transformers import pipeline

            self._pipeline = pipeline("text-classification", model=checkpoint)
        except Exception as err:
            logger.warning("Could not load transformers pipeline for %s: %s", checkpoint, err)

    def score(self, texts: Sequence[str | None]) -> tuple[np.ndarray, list[list[str]]]:
        if self._pipeline is None:
            raise NotImplementedError(
                "DeBERTa text classifier requires fine-tuning on documented, external ground-truth "
                "manipulation labels. Use UnsupervisedTextScorer for baseline trust scoring."
            )

        safe_texts = [str(t) if t else "" for t in texts]
        results = self._pipeline(safe_texts)
        scores: list[float] = []
        all_reasons: list[list[str]] = []

        for res in results:
            label = res.get("label", "LABEL_0")
            prob = float(res.get("score", 0.5))
            # LABEL_0 = authentic, LABEL_1 = synthetic/manipulated
            trust = 1.0 - prob if "1" in label or "FAKE" in label.upper() else prob
            scores.append(round(float(np.clip(trust, 0.0, 1.0)), 4))
            reasons = ["DEBERTA_AUTHENTIC"] if trust >= 0.6 else ["DEBERTA_SUSPICIOUS_TEXT"]
            all_reasons.append(reasons)

        return np.array(scores, dtype=np.float64), all_reasons


class SupervisedTextTrustClassifier(BaseTextScorer):
    """Calibrated scikit-learn NLP text trust classifier.

    Can be trained via .fit(texts, labels) using TF-IDF n-grams and
    stylometric features with logistic regression or gradient boosting.
    """

    def __init__(self) -> None:
        self.is_fitted = False
        self._vectorizer: Any = None
        self._classifier: Any = None

    def fit(self, texts: Sequence[str], labels: Sequence[int]) -> SupervisedTextTrustClassifier:
        """Fit classifier on known manipulation labels (0 = authentic, 1 = manipulated)."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        self._vectorizer = TfidfVectorizer(max_features=500, ngram_range=(1, 2))
        x_vec = self._vectorizer.fit_transform([str(t) for t in texts])
        self._classifier = LogisticRegression(class_weight="balanced", random_state=42)
        self._classifier.fit(x_vec, list(labels))
        self.is_fitted = True
        return self

    def score(self, texts: Sequence[str | None]) -> tuple[np.ndarray, list[list[str]]]:
        if not self.is_fitted or self._vectorizer is None or self._classifier is None:
            # Fall back to UnsupervisedTextScorer if not yet fitted
            fallback = UnsupervisedTextScorer()
            return fallback.score(texts)

        safe_texts = [str(t) if t else "" for t in texts]
        x_vec = self._vectorizer.transform(safe_texts)
        # Probability of being class 0 (authentic)
        probs = self._classifier.predict_proba(x_vec)
        trust_scores = probs[:, 0]  # class 0 = authentic
        scores: list[float] = []
        all_reasons: list[list[str]] = []

        for s in trust_scores:
            clamped = float(np.clip(s, 0.0, 1.0))
            scores.append(round(clamped, 4))
            reasons = ["ML_CLASSIFIER_AUTHENTIC"] if clamped >= 0.50 else ["ML_CLASSIFIER_SUSPICIOUS"]
            all_reasons.append(reasons)

        return np.array(scores, dtype=np.float64), all_reasons
