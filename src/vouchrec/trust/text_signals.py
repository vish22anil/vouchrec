"""Review-text trust signals: lexical diversity, repetition, and anomaly scoring."""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Precompiled regex for tokenization
WORD_RE = re.compile(r"\b[a-zA-Z0-9_\-']+\b")
SENTENCE_RE = re.compile(r"[.!?]+")


def extract_text_features(texts: Sequence[str | None]) -> pd.DataFrame:
    """Extract interpretable lexical and structural features from review text.

    Features computed:
      - char_length: total character length
      - word_count: count of alphanumeric tokens
      - sentence_count: count of sentence boundary delimiters
      - avg_word_length: average token character length
      - type_token_ratio: unique tokens / total tokens (lexical diversity)
      - caps_ratio: uppercase letters / total characters
      - exclamation_ratio: exclamation marks / total characters
      - repetition_ratio: fraction of duplicate tokens / total tokens
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
    ) -> None:
        self.neutral_score = neutral_score
        self.short_text_score = short_text_score
        self.min_ttr_threshold = min_ttr_threshold
        self.max_repetition_threshold = max_repetition_threshold
        self.max_caps_ratio = max_caps_ratio
        self.max_exclamation_ratio = max_exclamation_ratio

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

            # Bonuses for detailed, balanced review
            if row["word_count"] >= 12 and row["type_token_ratio"] >= 0.65:
                score += 0.20
                reasons.append("TEXT_DETAILED_AND_DIVERSE")

            # Bound score to [0.0, 1.0]
            clamped = float(np.clip(score, 0.0, 1.0))
            if not reasons:
                reasons.append("TEXT_NATURAL_BASELINE")

            scores.append(round(clamped, 4))
            all_reasons.append(reasons)

        return np.array(scores, dtype=np.float64), all_reasons


class DebertaTextScorer(BaseTextScorer):
    """Stub interface for future supervised / fine-tuned DeBERTa model."""

    def __init__(self, model_checkpoint: str | None = None) -> None:
        self.model_checkpoint = model_checkpoint

    def score(self, texts: Sequence[str | None]) -> tuple[np.ndarray, list[list[str]]]:
        raise NotImplementedError(
            "DeBERTa text classifier requires fine-tuning on documented, external ground-truth "
            "manipulation labels. Use UnsupervisedTextScorer for baseline trust scoring."
        )
