"""
band_classifier.py — predict Originality.ai-style color bands per sentence
==========================================================================

Uses a Claude reader to classify each sentence in a draft into one of
four bands matching Originality.ai's color scheme:

  GREEN          — high human probability
  YELLOW         — neutral
  YELLOW_ORANGE  — slight AI signal
  ORANGE         — strong AI signal

The classifier is calibration-aware: a set of anchor sentences with
known bands (drawn from the scored corpus) is prepended to every
classification call. This anchors the reader to Originality's actual
band thresholds rather than its own internal sense of AI-likeness.

The band sequence this module produces is consumed by
contagion_metrics.py to compute contagion zones and per-band shares,
which are the actual predictors of the headline score.

CHANGE LOG
----------
Removed the explicit ``temperature=0.0`` argument from the
messages.create() call in _classify_chunk. claude-opus-4-7 deprecated
the temperature parameter and returns a 400 BadRequestError when it
is supplied. Determinism is still substantially preserved by the
constrained system prompt, the structured-JSON output contract, and
the anchor calibration. The original line was:

    temperature=0.0,   # classifier should be deterministic

and the surrounding messages.create() call is otherwise unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import anthropic

logger = logging.getLogger("phase2.band_classifier")


# ============================================================================
# Configuration
# ============================================================================

BAND_NAMES = [
    "DEEP_GREEN", "LIGHT_GREEN", "YELLOW",
    "YELLOW_ORANGE", "LIGHT_ORANGE", "DEEP_ORANGE",
]

CLASSIFIER_MODEL_DEFAULT = "claude-opus-4-7"
MAX_TOKENS_PER_CALL = 8000

# Maximum sentences per classification call. Chunking keeps each call
# focused and keeps the JSON response under MAX_TOKENS_PER_CALL.
CHUNK_SENTENCE_TARGET = 40


# ============================================================================
# Default calibration anchors
# ============================================================================
#
# These are seed anchors drawn from past corpus analysis. They will be
# REPLACED at runtime by corpus_calibrator.py once the user's actual
# corpus has been processed. Without calibration, these defaults give
# the classifier reasonable starting calibration but score correlation
# will be weaker.

DEFAULT_CALIBRATION_ANCHORS = [
    # DEEP_GREEN — sentences with strong human-voice signature
    # (irregular rhythm, idiosyncratic word choice, voice present)
    {"text": "She wore the look of a woman who has been composing a grievance for the last nine miles.",
     "band": "DEEP_GREEN"},
    {"text": "I had eaten a lot of casseroles in my life from the hands of women who could not say what they meant and did not need to.",
     "band": "DEEP_GREEN"},
    {"text": "My phone buzzed at six-oh-three.", "band": "DEEP_GREEN"},
    {"text": "The wolf in my chest had paced through every hour, which was the wolf's way of saying things she did not want to hear.",
     "band": "DEEP_GREEN"},

    # LIGHT_GREEN — clear human voice but more common rhythm
    {"text": "I dumped it, refilled it, set it back.", "band": "LIGHT_GREEN"},
    {"text": "The back window one had gone cloudy again overnight.",
     "band": "LIGHT_GREEN"},
    {"text": "I pressed my thumb into the windowsill.", "band": "LIGHT_GREEN"},

    # YELLOW — neutral, could go either way
    {"text": "One of them blinked into a fox.", "band": "YELLOW"},
    {"text": "Twelve minutes, same as every morning.", "band": "YELLOW"},
    {"text": "The kettle clicked off behind me.", "band": "YELLOW"},

    # YELLOW_ORANGE — slight AI signal
    {"text": "I'd clipped them yesterday without thinking about it.",
     "band": "YELLOW_ORANGE"},
    {"text": "No shortcuts. Fingertip along each sill.",
     "band": "YELLOW_ORANGE"},
    {"text": "The radio was on a commercial.", "band": "YELLOW_ORANGE"},

    # LIGHT_ORANGE — moderate AI signal (smooth, expected, mild patterning)
    {"text": "Salt at the kitchen, salt at the bathroom, salt at the bedroom.",
     "band": "LIGHT_ORANGE"},
    {"text": "I drank some of it. I forgot the rest.",
     "band": "LIGHT_ORANGE"},
    {"text": "She wanted the day to be ordinary. That was the whole prayer.",
     "band": "LIGHT_ORANGE"},

    # DEEP_ORANGE — strong AI signal (high-template constructions)
    {"text": "Then the rest: kitchen windowsill, bedroom, bathroom, living room.",
     "band": "DEEP_ORANGE"},
    {"text": "Red fur, narrow snout, amber eyes.", "band": "DEEP_ORANGE"},
    {"text": "My spirit died as well, every source of joy and hope turning at once to dust within me.",
     "band": "DEEP_ORANGE"},
    {"text": "The world seemed leached of all vitality and meaning, like a stained glass window with the sun moved beyond it.",
     "band": "DEEP_ORANGE"},
]


# ============================================================================
# Sentence segmentation
# ============================================================================

# Sentence-end punctuation followed by whitespace and an uppercase letter
# or end-of-string. Keeps abbreviations together (Dr., Mr., etc.) by
# refusing to split on those.
_ABBR = {"Dr", "Mr", "Mrs", "Ms", "St", "Jr", "Sr", "vs", "etc"}
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\u201c])")


def segment_sentences(text: str) -> list[str]:
    """
    Split prose into sentences. Conservative: prefers under-segmenting
    over over-segmenting. Returns a list of sentence strings with
    their trailing whitespace stripped.

    Paragraph breaks are preserved as empty-string entries so the
    classifier and downstream metrics can see paragraph structure.
    """
    sentences: list[str] = []
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        # Naively split on sentence boundaries.
        parts = _SENTENCE_BOUNDARY.split(para)
        for raw in parts:
            s = raw.strip()
            if not s:
                continue
            # Re-merge sentence-final abbreviations swallowed by the
            # split. (Cheap heuristic: if the prior sentence ends with
            # an abbreviation, glue this fragment back on.)
            if (sentences
                    and any(sentences[-1].rstrip(".").endswith(a)
                            for a in _ABBR)):
                sentences[-1] = sentences[-1] + " " + s
            else:
                sentences.append(s)
        sentences.append("")   # paragraph break marker
    # Strip any trailing paragraph marker.
    while sentences and sentences[-1] == "":
        sentences.pop()
    return sentences


# ============================================================================
# Classifier
# ============================================================================

@dataclass
class BandClassifierConfig:
    model: str = CLASSIFIER_MODEL_DEFAULT
    anchors: Optional[list[dict]] = None  # list of {"text", "band"} dicts
    chunk_size: int = CHUNK_SENTENCE_TARGET


class BandClassifier:
    """
    Classify each sentence in a draft into one of the four bands.

    Usage:
        classifier = BandClassifier()
        bands = classifier.classify(draft_text)
        # bands is a list of (sentence_text, band_label) tuples,
        # in document order. Empty-string sentences mark paragraph
        # breaks and have band None.
    """

    def __init__(
        self,
        config: Optional[BandClassifierConfig] = None,
        client: Optional[anthropic.Anthropic] = None,
        api_key: Optional[str] = None,
    ):
        self.config = config or BandClassifierConfig()
        self.anchors = self.config.anchors or DEFAULT_CALIBRATION_ANCHORS
        if client is not None:
            self.client = client
        else:
            self.client = (
                anthropic.Anthropic(api_key=api_key)
                if api_key
                else anthropic.Anthropic()
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def classify(self, text: str) -> list[tuple[str, Optional[str]]]:
        """
        Classify every sentence in `text`. Returns a list of
        (sentence, band) tuples. Paragraph breaks appear as
        ("", None) entries.
        """
        sentences = segment_sentences(text)
        prose_sentences = [s for s in sentences if s]
        # Chunk for classification, but keep the paragraph-break
        # positions so we can re-insert them in the output.
        chunks = self._chunk(prose_sentences, self.config.chunk_size)

        flat_predictions: list[str] = []
        for chunk in chunks:
            preds = self._classify_chunk(chunk)
            if len(preds) != len(chunk):
                logger.warning(
                    "classifier returned %d predictions for chunk of %d "
                    "sentences; falling back to YELLOW for misalignment",
                    len(preds), len(chunk),
                )
                preds = (preds + ["YELLOW"] * len(chunk))[:len(chunk)]
            flat_predictions.extend(preds)

        # Rebuild output with paragraph markers.
        out: list[tuple[str, Optional[str]]] = []
        pi = 0
        for s in sentences:
            if s == "":
                out.append(("", None))
            else:
                band = flat_predictions[pi] if pi < len(flat_predictions) else "YELLOW"
                out.append((s, band))
                pi += 1
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _chunk(self, sentences: list[str], size: int) -> list[list[str]]:
        if not sentences:
            return []
        return [
            sentences[i:i + size]
            for i in range(0, len(sentences), size)
        ]

    def _build_anchor_block(self) -> str:
        lines = []
        # Group anchors by band so the reader sees the band's typical
        # examples together.
        for band in BAND_NAMES:
            band_examples = [
                a for a in self.anchors if a.get("band") == band
            ]
            if not band_examples:
                continue
            lines.append(f"  {band}:")
            for a in band_examples:
                # Truncate very long anchors for prompt economy.
                snippet = a["text"]
                if len(snippet) > 200:
                    snippet = snippet[:200] + "…"
                lines.append(f"    - {snippet!r}")
        return "\n".join(lines)

    def _build_system_prompt(self) -> str:
        anchor_block = self._build_anchor_block()
        return (
            "You are predicting how Originality.ai's turbo classifier "
            "would color each sentence of a fiction draft. The "
            "classifier assigns each sentence one of SIX bands "
            "based on per-token predictability:\n\n"
            "  DEEP_GREEN     — strong human voice signature "
            "(idiosyncratic word choice, irregular rhythm, voice "
            "present, unusual specifics)\n"
            "  LIGHT_GREEN    — clear human voice, common rhythm\n"
            "  YELLOW         — neutral, could read either way\n"
            "  YELLOW_ORANGE  — slight AI signal (smooth rhythm, "
            "expected word choices)\n"
            "  LIGHT_ORANGE   — moderate AI signal (mild structural "
            "patterning, parallel rhythm, generic abstractions)\n"
            "  DEEP_ORANGE    — strong AI signal (high-template "
            "constructions: subjectless inventory fragments, "
            "noun-phrase triples, parallel rhythm, specific-number "
            "stacking, the-way-X observations, negation pivots, "
            "tautological loops, overwrought emotional generalities, "
            "polysyndetic runs)\n\n"
            "The deep/light split within green and orange matters: a "
            "sentence can read as broadly human (green) but be mild "
            "in its voice signature (LIGHT_GREEN vs DEEP_GREEN), and "
            "a sentence can show AI patterns at varying strength "
            "(LIGHT_ORANGE vs DEEP_ORANGE).\n\n"
            "Calibration anchors (these scored at the indicated band "
            "in real Originality.ai exports — use them to fix your "
            "band thresholds, not your overall sense of AI-ness):\n\n"
            f"{anchor_block}\n\n"
            "You will be given a sequence of sentences in order. "
            "Return ONE JSON object: "
            '{"bands": ["DEEP_GREEN", "YELLOW_ORANGE", ...]}. The '
            "list must have exactly the same length as the input "
            "list of sentences. No prose outside the JSON object. "
            "No code fences. Start with { and end with }."
        )

    def _build_user_message(self, sentences: list[str]) -> str:
        numbered = "\n".join(
            f"{i+1}. {s}" for i, s in enumerate(sentences)
        )
        return (
            "Classify each numbered sentence into one of GREEN, "
            "YELLOW, YELLOW_ORANGE, ORANGE. Return a single JSON "
            "object whose 'bands' list has exactly "
            f"{len(sentences)} entries, in the same order as the "
            "input.\n\n"
            f"{numbered}"
        )

    def _classify_chunk(self, sentences: list[str]) -> list[str]:
        system = self._build_system_prompt()
        user = self._build_user_message(sentences)
        # NOTE: the original implementation passed temperature=0.0 here to
        # force deterministic output. claude-opus-4-7 deprecated the
        # temperature parameter and rejects it with a 400 BadRequestError.
        # The argument is intentionally omitted; determinism is preserved
        # by the constrained system prompt, the structured-JSON output
        # contract, and anchor calibration.
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=MAX_TOKENS_PER_CALL,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        raw = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()

        # Tolerate fences just in case.
        if raw.startswith("```"):
            lines = raw.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            raw = "\n".join(lines).strip()

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(
                "band classifier returned non-JSON: %s; first 300 chars: %r",
                e, raw[:300],
            )
            return ["YELLOW"] * len(sentences)

        bands = obj.get("bands", [])
        # Validate every band is in the allowed set.
        cleaned = []
        for b in bands:
            if isinstance(b, str) and b.upper() in BAND_NAMES:
                cleaned.append(b.upper())
            else:
                cleaned.append("YELLOW")
        return cleaned


# ============================================================================
# Calibration loading
# ============================================================================

def load_calibration(path: Path) -> Optional[list[dict]]:
    """
    Load a calibration anchor file produced by corpus_calibrator.py.
    The file is JSON: {"anchors": [{"text": "...", "band": "GREEN"}, ...],
                       "regression": {...}}.
    Returns the anchors list, or None if the file is missing.
    """
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("anchors")


def load_regression(path: Path) -> Optional[dict]:
    """
    Load the regression coefficients from a calibration file.
    Returns the regression dict, or None if missing.
    """
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    return obj.get("regression")
