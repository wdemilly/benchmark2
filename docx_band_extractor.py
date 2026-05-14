"""
docx_band_extractor.py — extract ground-truth bands from Originality .docx exports
==================================================================================

Originality.ai exports drafts as .docx files where each text run carries
a background shading color (<w:shd w:fill="HEXCODE">) that encodes the
per-sentence AI-detection band. This module:

  1. Unzips the .docx and reads word/document.xml
  2. Walks every <w:r> run, pairing its <w:shd> fill color with its text
  3. Classifies each run's color into one of four bands:
        GREEN, YELLOW, YELLOW_ORANGE, ORANGE
  4. Aggregates runs into sentences (Originality emits one shading per
     scored unit, which usually corresponds to one sentence)

The output is a list of (sentence_text, band) tuples — the same shape
band_classifier produces — usable as ground truth for calibration.

This module reimplements the color-band decoding the operator has been
doing manually from their Phase 1 .docx exports.
"""

from __future__ import annotations

import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Optional


# ============================================================================
# Color band classification
# ============================================================================
#
# Originality's shading colors form a continuous spectrum. We classify
# each hex code into one of four bands using the red-green channel gap.
# Greener (negative gap) = more human; warmer (positive gap) = more AI.

# Thresholds derived from corpus inspection on 80+ scored exports:
# The actual hex spectrum tops out around gap +25, not the +60 a naive
# "warm = red dominant" reading would suggest.
#   gap <= -5            → GREEN          (d4ece0, e9efdd, dceddf...)
#   -5 <  gap <= +3      → YELLOW         (f0f0dc, eff0dc, f3f1db)
#   +3 <  gap <= +15     → YELLOW_ORANGE  (fbf2da, fff0da, fff1da, feebda)
#   gap >  +15           → ORANGE         (fae1da, f9e0da, fbe5da, fce7da)


def _gap(hex_color: str) -> Optional[int]:
    """Compute the red-green channel gap. Higher = more AI-ish."""
    h = hex_color.strip().lstrip("#")
    if len(h) != 6:
        return None
    try:
        r = int(h[0:2], 16)
        g = int(h[2:4], 16)
    except ValueError:
        return None
    return r - g


def classify_color(hex_color: Optional[str]) -> str:
    """
    Classify an Originality shading color into one of six bands,
    or 'NONE' if there's no shading (uncolored text).

    Six bands give meaningfully better score prediction than four
    (LOO r=0.96 vs 0.94 on the 32-export corpus), at the cost of a
    slightly more complex classifier prompt.

    Thresholds derived from corpus inspection:
      gap <= -15           → DEEP_GREEN     (d4ece0, d5ece0, d6ece0)
      -15 <  gap <= -5     → LIGHT_GREEN    (dceddf, e1eede, e9efdd)
      -5  <  gap <= +3     → YELLOW         (f0f0dc, eff0dc)
      +3  <  gap <= +15    → YELLOW_ORANGE  (fff0da, fff3da, fcf2da)
      +15 <  gap <  +20    → LIGHT_ORANGE   (feebda, feecda)
      gap >= +20           → DEEP_ORANGE    (fae1da, fae2da, fbe5da)
    """
    if not hex_color or hex_color.upper() == "AUTO":
        return "NONE"
    gap = _gap(hex_color)
    if gap is None:
        return "NONE"
    if gap <= -15:
        return "DEEP_GREEN"
    if gap <= -5:
        return "LIGHT_GREEN"
    if gap <= 3:
        return "YELLOW"
    if gap <= 15:
        return "YELLOW_ORANGE"
    if gap < 20:
        return "LIGHT_ORANGE"
    return "DEEP_ORANGE"


def classify_color_4band(hex_color: Optional[str]) -> str:
    """
    Legacy 4-band classifier — collapses the 6-band output into the
    original GREEN/YELLOW/YELLOW_ORANGE/ORANGE buckets. Kept for
    backward compatibility with the original contagion_metrics
    feature set.
    """
    band = classify_color(hex_color)
    if band in ("DEEP_GREEN", "LIGHT_GREEN"):
        return "GREEN"
    if band in ("LIGHT_ORANGE", "DEEP_ORANGE"):
        return "ORANGE"
    return band


# ============================================================================
# .docx XML parsing
# ============================================================================

# Pull text out of <w:t> elements inside a <w:r> run, and the
# <w:shd w:fill="..."> color attribute if present.
_W_RUN_RE = re.compile(r"<w:r[ >].*?</w:r>", re.DOTALL)
_W_SHD_RE = re.compile(r'<w:shd[^>]*w:fill="([^"]+)"')
_W_T_RE = re.compile(r"<w:t[^>]*>(.*?)</w:t>", re.DOTALL)


def _read_document_xml(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path) as zf:
        return zf.read("word/document.xml").decode("utf-8")


def extract_run_bands(docx_path: Path) -> list[tuple[str, str]]:
    """
    Read a .docx export and return a list of (text, band) tuples, one
    per shaded run. Empty-text runs are dropped.
    """
    xml = _read_document_xml(docx_path)
    runs = _W_RUN_RE.findall(xml)
    out = []
    for run in runs:
        shd = _W_SHD_RE.search(run)
        color = shd.group(1) if shd else None
        text_parts = _W_T_RE.findall(run)
        text = "".join(text_parts)
        if not text:
            continue
        band = classify_color(color)
        out.append((text, band))
    return out


# ============================================================================
# Sentence aggregation
# ============================================================================
#
# Originality may emit one shading per sentence OR per word inside a
# sentence (the actual behavior depends on the export tool's version).
# We aggregate adjacent same-band runs into a single sentence entry.
# A sentence-ending punctuation also forces a break.

_SENT_END_RE = re.compile(r"[.!?][\"')\]]?\s*$")


def aggregate_into_sentences(
    runs: list[tuple[str, str]]
) -> list[tuple[str, str]]:
    """
    Aggregate run-level (text, band) pairs into sentence-level entries.
    When adjacent runs share a band, they're merged. A run ending in
    sentence-end punctuation forces a sentence boundary even if the
    next run has the same band.
    """
    sentences: list[tuple[str, str]] = []
    buffer_text = ""
    buffer_band: Optional[str] = None

    for text, band in runs:
        if buffer_band is None:
            buffer_text = text
            buffer_band = band
        elif band == buffer_band and not _SENT_END_RE.search(buffer_text):
            buffer_text += text
        else:
            sentences.append((buffer_text, buffer_band))
            buffer_text = text
            buffer_band = band

    if buffer_band is not None and buffer_text:
        sentences.append((buffer_text, buffer_band))

    # Coalesce consecutive NONE-band runs (uncolored prose) into the
    # most plausible neighbor band. Default to YELLOW.
    cleaned: list[tuple[str, str]] = []
    for text, band in sentences:
        if band == "NONE":
            cleaned.append((text, "YELLOW"))
        else:
            cleaned.append((text, band))
    return cleaned


# ============================================================================
# Public entry point
# ============================================================================

def extract_bands(docx_path: Path) -> list[tuple[str, str]]:
    """
    Extract per-sentence (text, band) ground truth from an Originality
    .docx export. The output is directly comparable to
    BandClassifier.classify()'s output.
    """
    runs = extract_run_bands(Path(docx_path))
    return aggregate_into_sentences(runs)


def band_share_summary(
    sentences: list[tuple[str, str]]
) -> dict[str, float]:
    """
    Return per-band character-percentage shares. Mirrors the
    Phase 1 manual analysis style.
    """
    char_counter: Counter = Counter()
    for text, band in sentences:
        char_counter[band] += len(text)
    total = sum(char_counter.values()) or 1
    return {
        band: round(char_counter.get(band, 0) / total * 100, 2)
        for band in ("GREEN", "YELLOW", "YELLOW_ORANGE", "ORANGE")
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python docx_band_extractor.py path/to/export.docx",
              file=sys.stderr)
        sys.exit(2)
    sents = extract_bands(Path(sys.argv[1]))
    shares = band_share_summary(sents)
    print("=== BAND SHARE (% characters) ===")
    for band, pct in shares.items():
        print(f"  {band:<15} {pct:>6.2f}%")
    print(f"\n=== {len(sents)} sentence-level entries ===")
    print("\nFirst 10 entries:")
    for i, (text, band) in enumerate(sents[:10], 1):
        snippet = text.strip()[:80]
        print(f"  {i:2d}. [{band:<14}] {snippet!r}")
