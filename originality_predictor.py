"""
originality_predictor.py

Predicts an Originality.ai human-score for a draft BEFORE submission,
using k nearest-neighbor in-context examples from the labeled corpus.

NEIGHBOR SELECTION uses prose-STYLE features (sentence-length distribution,
fragment ratio, em-dash density, punctuation density, flagged-construction
density) rather than content-based char n-grams. In a single-project
corpus, char-n-gram similarity is dominated by shared content (same
chapter, same characters) and has near-zero signal for the Originality
score. Style features are what Originality's detector is actually reading.

Validated LOO on the 48-document labeled corpus at build time:
    Ridge regression (style features, no LLM):  MAE 18.83, r +0.504
    NN-mean cosine k=5 (style features, no LLM): MAE 18.90, r +0.470
    Char-n-gram NN-mean k=5 (no LLM):            MAE 21.75, r +0.037  [signal-less]
    Always-predict-corpus-mean (naive):          MAE 22.01

The LLM predictor uses style-feature retrieval to choose in-context
examples, then asks Claude to reason about the query against the
retrieved neighbors. The ridge and NN-mean are shown to the model as
honest baselines.

Public API:

    predictor = OriginalityPredictor(corpus_path="labeled_corpus.json")
    result = predictor.predict(
        draft_text=...,
        client=anthropic_client,
        model="claude-opus-4-7",
        k=5,
    )

    result = {
        "predicted_score": int | None,
        "rationale": str,
        "neighbors": [...],
        "style_baseline_score": float,
        "nn_mean_baseline": float,
        "query_features": {...},
        "raw_response": str,
        "parse_status": "ok" | "no_score_parsed" | "empty_response",
        ...
    }

Cost estimate (k=5, Opus): ~$0.35 per call.
Requires: pip install scikit-learn numpy
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Style feature extractor
# ---------------------------------------------------------------------------

THE_WAY_RE = re.compile(r"\bthe way (?:he|she|it|they|one|you)\b", re.IGNORECASE)
NOT_BUT_RE = re.compile(
    r"\bnot\s+[\w\-]+(?:\s+[\w\-]+){0,4}\s+but\s+", re.IGNORECASE
)
PERIPHRASTIC_RE = re.compile(
    r"\b(?:a|an)\s+(?:kind|sort|type|manner|way|form|quality)\s+of\b",
    re.IGNORECASE,
)

FEATURE_NAMES = [
    "avg_sent", "std_sent", "max_sent", "long_pct", "frag_pct",
    "mean_wlen", "std_wlen",
    "the_way_per_1k", "not_but_per_1k", "periphrastic_per_1k",
    "em_per_1k", "semi_per_1k", "colon_per_1k", "paren_per_1k", "comma_per_1k",
]


def extract_style_features(text: str) -> dict:
    """Compute the style-feature vector used for retrieval and the ridge model."""
    words = re.findall(r"\b[\w']+\b", text)
    wc = max(len(words), 1)
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    sent_lens_list = [len(re.findall(r"\b[\w']+\b", s)) for s in sentences]
    if not sent_lens_list:
        sent_lens_list = [1]
    sent_lens = np.array(sent_lens_list)
    wlens = np.array([len(w) for w in words]) if words else np.array([1])

    tw = len(THE_WAY_RE.findall(text))
    nb = len(NOT_BUT_RE.findall(text))
    per = len(PERIPHRASTIC_RE.findall(text))
    em = text.count("\u2014")
    semi = text.count(";")
    colon = len(re.findall(r"(?<!\d):(?!\d)", text))
    paren = text.count("(")
    comma = text.count(",")

    return {
        "avg_sent": float(np.mean(sent_lens)),
        "std_sent": float(np.std(sent_lens)),
        "max_sent": int(np.max(sent_lens)),
        "long_pct": float(100 * np.sum(sent_lens > 30) / len(sent_lens)),
        "frag_pct": float(100 * np.sum(sent_lens <= 5) / len(sent_lens)),
        "mean_wlen": float(np.mean(wlens)),
        "std_wlen": float(np.std(wlens)),
        "the_way_per_1k": 1000 * tw / wc,
        "not_but_per_1k": 1000 * nb / wc,
        "periphrastic_per_1k": 1000 * per / wc,
        "em_per_1k": 1000 * em / wc,
        "semi_per_1k": 1000 * semi / wc,
        "colon_per_1k": 1000 * colon / wc,
        "paren_per_1k": 1000 * paren / wc,
        "comma_per_1k": 1000 * comma / wc,
    }


def features_to_vector(feat: dict) -> np.ndarray:
    return np.array([feat[n] for n in FEATURE_NAMES], dtype=float)


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PREDICTOR_PROMPT = """You are estimating the Originality.ai human-written score that a chapter draft would receive.

Originality.ai outputs an integer 0 to 100 where higher = more likely to be human-written. The detector is sensitive to statistical prose features (sentence length variance, construction patterns, syntactic uniformity, rhythmic monotony) rather than content or craft quality. A well-crafted chapter can score low; a rough draft can score high. Your job is to predict the detector's output, not to judge quality.

Below are {K} labeled reference drafts, retrieved as the most structurally similar to the query draft using prose-style features (sentence-length distribution, flagged-construction density, punctuation density). Each reference is shown with its actual Originality human-score and key style metrics. Use these as calibration anchors.

Two numeric baselines for the query are included at the end: a ridge-regression prediction from style features alone, and the mean of the retrieved neighbors' scores. These are honest baselines; your prediction should draw on the actual prose patterns you can see in the query vs. the references, not simply repeat the baselines.

===== REFERENCE DRAFTS =====
{REFERENCES}
===== END REFERENCES =====

===== QUERY DRAFT =====
Style metrics: {QUERY_METRICS}

{QUERY}
===== END QUERY =====

Baselines for this query:
- Ridge regression (style features -> score): {RIDGE_BASELINE}
- Mean of {K} nearest neighbors:              {NN_BASELINE}

Compare the query's prose texture to each reference. Look at sentence-length variance, aphoristic closures, construction repetition, rhythmic variety, and the specific patterns Originality rewards (high variance, broken patterns, unexpected rhythmic shifts) vs. penalizes (uniform sentence length, parallel constructions, repeated syntactic frames).

Return your answer in exactly this format and nothing else:

REASONING: <3-5 sentences comparing the query's texture to the references by ID. Note which references the query most resembles and why. Address the baselines: do you agree or diverge, and on what basis.>

PREDICTED_SCORE: <integer 0-100>"""


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

@dataclass
class Neighbor:
    id: str
    human_score: int
    similarity: float
    text: str
    source_file: str
    word_count: int
    features: dict


class OriginalityPredictor:
    """In-context nearest-neighbor predictor using style-feature retrieval."""

    def __init__(
        self,
        corpus_path: str | Path = "labeled_corpus.json",
        ridge_alpha: float = 0.3,
    ):
        self.corpus_path = Path(corpus_path)
        self.records = self._load_corpus()

        # Extract features once per corpus doc
        self.feature_dicts = [extract_style_features(r["text"]) for r in self.records]
        X = np.array([features_to_vector(f) for f in self.feature_dicts])

        # Fit scaler (queries use the same transform) and ridge (baseline predictor)
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(X)
        y = np.array([r["human_score"] for r in self.records])
        self.ridge = Ridge(alpha=ridge_alpha).fit(self.X_scaled, y)

    def _load_corpus(self) -> list[dict]:
        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")
        data = json.loads(self.corpus_path.read_text())
        labeled = [r for r in data if r.get("human_score") is not None]
        if len(labeled) < 5:
            raise ValueError(
                f"Need at least 5 labeled records; found {len(labeled)} in "
                f"{self.corpus_path}"
            )
        return labeled

    # ------------------------------------------------------------------
    # Retrieval + baseline
    # ------------------------------------------------------------------

    def _scale_features(self, feat: dict) -> np.ndarray:
        vec = features_to_vector(feat).reshape(1, -1)
        return self.scaler.transform(vec)

    def find_neighbors(
        self,
        draft_text: str,
        k: int = 5,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[Neighbor]:
        exclude_ids = exclude_ids or set()
        query_feat = extract_style_features(draft_text)
        query_scaled = self._scale_features(query_feat)
        sims = cosine_similarity(query_scaled, self.X_scaled).flatten()

        idx_order = np.argsort(-sims)
        out: list[Neighbor] = []
        for idx in idx_order:
            rec = self.records[idx]
            if rec["id"] in exclude_ids:
                continue
            out.append(Neighbor(
                id=rec["id"],
                human_score=rec["human_score"],
                similarity=float(sims[idx]),
                text=rec["text"],
                source_file=rec["source_file"],
                word_count=rec["word_count"],
                features=self.feature_dicts[idx],
            ))
            if len(out) >= k:
                break
        return out

    def ridge_baseline(self, feat: dict) -> float:
        scaled = self._scale_features(feat)
        pred = float(self.ridge.predict(scaled)[0])
        return max(0.0, min(100.0, pred))

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @staticmethod
    def _fmt_metrics(feat: dict) -> str:
        return (
            f"avg_sent={feat['avg_sent']:.1f} "
            f"std_sent={feat['std_sent']:.1f} "
            f"max_sent={feat['max_sent']} "
            f"long%={feat['long_pct']:.1f} "
            f"frag%={feat['frag_pct']:.1f} "
            f"em/1k={feat['em_per_1k']:.2f} "
            f"semi/1k={feat['semi_per_1k']:.2f} "
            f"the-way/1k={feat['the_way_per_1k']:.2f} "
            f"not-but/1k={feat['not_but_per_1k']:.2f}"
        )

    def _build_prompt(
        self,
        draft_text: str,
        neighbors: list[Neighbor],
        ridge_baseline: float,
        nn_mean: float,
        max_neighbor_chars: int = 14000,
        max_query_chars: int = 20000,
    ) -> str:
        ref_blocks = []
        for i, nb in enumerate(neighbors, 1):
            snippet = nb.text[:max_neighbor_chars]
            ref_blocks.append(
                f"--- REFERENCE {i} (id={nb.id}, human_score={nb.human_score}, "
                f"similarity={nb.similarity:.3f}) ---\n"
                f"Style metrics: {self._fmt_metrics(nb.features)}\n\n"
                f"{snippet}"
            )
        references_text = "\n\n".join(ref_blocks)
        query_feat = extract_style_features(draft_text)
        query_text = draft_text[:max_query_chars]
        return PREDICTOR_PROMPT.format(
            K=len(neighbors),
            REFERENCES=references_text,
            QUERY=query_text,
            QUERY_METRICS=self._fmt_metrics(query_feat),
            RIDGE_BASELINE=f"{ridge_baseline:.0f}",
            NN_BASELINE=f"{nn_mean:.0f}",
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str) -> tuple[Optional[int], str, str]:
        if not raw:
            return (None, "", "empty_response")

        score = None
        m = re.search(r"PREDICTED[_\s]*SCORE\s*[:=]\s*(\d{1,3})", raw, re.IGNORECASE)
        if m:
            try:
                v = int(m.group(1))
                if 0 <= v <= 100:
                    score = v
            except ValueError:
                pass

        if score is None:
            tail = raw[-200:]
            for n in reversed(re.findall(r"\b(\d{1,3})\b", tail)):
                v = int(n)
                if 0 <= v <= 100:
                    score = v
                    break

        rationale = ""
        m2 = re.search(
            r"REASONING\s*[:=]\s*(.+?)(?=PREDICTED[_\s]*SCORE|$)",
            raw, re.IGNORECASE | re.DOTALL,
        )
        if m2:
            rationale = m2.group(1).strip()
        else:
            m3 = re.search(r"(.*?)PREDICTED", raw, re.IGNORECASE | re.DOTALL)
            rationale = m3.group(1).strip() if m3 else raw.strip()

        status = "ok" if score is not None else "no_score_parsed"
        return (score, rationale, status)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def predict(
        self,
        draft_text: str,
        client: Any,
        model: str = "claude-opus-4-7",
        k: int = 5,
        max_tokens: int = 1000,
        exclude_ids: Optional[set[str]] = None,
    ) -> dict:
        neighbors = self.find_neighbors(draft_text, k=k, exclude_ids=exclude_ids)
        query_feat = extract_style_features(draft_text)
        ridge_base = self.ridge_baseline(query_feat)
        nn_mean = (
            float(np.mean([n.human_score for n in neighbors])) if neighbors else 50.0
        )

        prompt = self._build_prompt(draft_text, neighbors, ridge_base, nn_mean)
        est_input_tokens = len(prompt) // 4

        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = ""
        for block in response.content:
            if getattr(block, "type", None) == "text":
                raw_text += block.text
            elif isinstance(block, dict) and block.get("type") == "text":
                raw_text += block.get("text", "")

        score, rationale, status = self._parse_response(raw_text)

        return {
            "predicted_score": score,
            "rationale": rationale,
            "neighbors": [
                {
                    "id": nb.id,
                    "human_score": nb.human_score,
                    "similarity": round(nb.similarity, 4),
                    "source_file": nb.source_file,
                    "word_count": nb.word_count,
                    "features": nb.features,
                }
                for nb in neighbors
            ],
            "style_baseline_score": round(ridge_base, 1),
            "nn_mean_baseline": round(nn_mean, 1),
            "query_features": query_feat,
            "raw_response": raw_text,
            "parse_status": status,
            "model": model,
            "k": k,
            "input_token_estimate": est_input_tokens,
        }

    # ------------------------------------------------------------------
    # Recommendation bands
    # ------------------------------------------------------------------

    @staticmethod
    def recommendation(predicted_score: Optional[int]) -> str:
        """Map predicted score to a shipping recommendation band.

        Bands calibrated from LOO ridge-baseline gate analysis on the
        48-doc corpus. These are conservative; they improve as labels grow.
          pred>=92 -> SHIP          (60% real>=90, 60% real>=85 on baseline)
          pred>=88 -> SHIP_CAUTIOUS (50% real>=90, 67% real>=85)
          pred>=80 -> RECONSIDER    (33% real>=90, 58% real>=85)
          else     -> REGENERATE
        """
        if predicted_score is None:
            return "UNKNOWN"
        if predicted_score >= 92:
            return "SHIP"
        if predicted_score >= 88:
            return "SHIP_CAUTIOUS"
        if predicted_score >= 80:
            return "RECONSIDER"
        return "REGENERATE"
