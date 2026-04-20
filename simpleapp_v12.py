"""
Micro-Prompt Harness — Quality Floor + Scanner-Ranked Pipeline
===============================================================
Generate chapter drafts and ship TOP 1, optionally with sentence-level grafts.

The pipeline answers three questions in order:

  Q1. Is this draft acceptable as prose? A pass/fail quality floor on each
      draft. Voice intact, beats landed, dialogue working, no collapses or
      incoherences. Unacceptable drafts are dropped entirely — not shipped,
      not used as graft donors, not ranked. If zero drafts clear the floor,
      the pipeline halts and reports failure. The floor is lenient: it
      catches drafts you would be embarrassed to ship, not drafts that are
      merely different from the others.

  Q2. Among acceptable drafts, which has the strongest detector profile?
      Ranking is driven by the mechanical scanner, not by literary judgment.
      TOP 1 is the acceptable draft with the fewest hard-cap violations.
      Ties break on word-count distance from the outline's target, then on
      the literary evaluator's residual ranking. Literary quality is a
      gate at Q1, not a weight at Q2.

  Q3. Among the acceptable drafts' sentences, which would improve TOP 1's
      detector profile without hurting its quality? The graft pass looks
      for runner-up sentences that do the same narrative work as their
      counterpart in TOP 1 AND differ from it rhythmically or structurally
      in dimensions detectors read — sentence length, clause structure,
      fragment vs full sentence, period vs em dash, unbalanced vs balanced
      list. Donors are rejected if their own sentence carries a hard-cap
      pattern. Quality is preserved by the same-narrative-work requirement.

The scanner informs Q2 and Q3. The literary evaluator drives Q1 and
contributes only to Q2 tiebreaks.

Export: top-N acceptable drafts as separate files + TOP1_GRAFTED (when
any grafts applied) + a batch summary naming any rejections.

The generation prompt lives in prompts.csv. The app does not inject its
own drafting instructions.
"""

import base64
import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List, Tuple

import pandas as pd
import requests
import streamlit as st

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import docx as python_docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# ============================================================================
# Constants
# ============================================================================

RUNS_DIR = Path("micro_prompt_runs")
OUTPUTS_DIR = RUNS_DIR / "flat_outputs"
FINAL_DIR = RUNS_DIR / "final_deliverables"
CSV_FILENAME = "runs.csv"
PROMPTS_CSV = "prompts.csv"

DEFAULT_GEN_MODEL = "claude-opus-4-7"
DEFAULT_EVAL_MODEL = "claude-opus-4-6"
MAX_GEN_TOKENS = 16000
MAX_EVAL_TOKENS = 8000


# ============================================================================
# Literary evaluator prompt — unchanged from original app
# ============================================================================

EVALUATOR_PROMPT = """You are evaluating {N} drafts of the same chapter against its outline. You have three inputs: the chapter outline, the mechanical scanner results for each draft, and the drafts themselves.

Read every draft in full. Do not skim.

Your job is NOT to pick a winner on literary grounds. Your job is to do two things: (1) apply a quality floor to each draft so the pipeline knows which ones are fit to ship at all, and (2) provide an advisory literary ranking used only to break ties downstream. The final TOP 1 selection is made by the pipeline using the scanner, not by you.

YOUR METHOD — in this order:

1. WORD COUNTS. Note each draft's word count against the outline's target range. Flag any that are materially short or over.

2. MECHANICAL COMPLIANCE. The scanner results are provided below. For each draft, note the violation counts. Do not re-scan — use the provided numbers. Reference them when assessing prose, but do not let them drive your quality verdict. Violations affect downstream ranking; your job here is quality.

3. QUALITY FLOOR — one verdict per draft. For each draft, decide ACCEPTABLE or UNACCEPTABLE. Apply a LENIENT standard: mark a draft ACCEPTABLE unless you would be embarrassed to ship it. UNACCEPTABLE means one or more of:
   - Voice collapse: the POV character's interior voice is absent, generic, or wrong register for long stretches.
   - Beats missing or compressed to the point of incoherence: a scene the outline requires is not on the page or is a throwaway line.
   - Dialogue that doesn't land: exchanges without subtext, without weapons, without stakes; turns that read like exposition dumps.
   - Structural failure: the chapter doesn't arrive where the outline says it arrives, or the ending doesn't close what was opened.
   - Prose-level damage: runs of flat summary where the outline asks for scene, long stretches of interpretive narration where the outline asks for observation and judgment, abandoned subplots, characters acting out of their profiles.

   Merely being less elegant than another draft is NOT grounds for UNACCEPTABLE. Stylistic difference is NOT grounds for UNACCEPTABLE. A draft can be ACCEPTABLE even if another draft is better at the same beats.

   A draft marked UNACCEPTABLE is dropped from the pipeline — not ranked, not used as a graft donor, not shipped. Use this verdict sparingly and only when you can name a concrete failure.

4. ADVISORY LITERARY RANKING. Rank the ACCEPTABLE drafts from strongest to weakest on prose quality alone (voice fidelity, dialogue craft, interior voice sharpness, beat execution). UNACCEPTABLE drafts are omitted from the ranking. This ranking is advisory — the pipeline uses it only to break ties when two drafts have the same scanner-violation count.

5. GRAFT CANDIDATES. From the non-winning acceptable drafts, name specific lines or passages worth transplanting into the top-ranked draft. Quote a few words for identification and name the beat where each would land. This is advisory context for the downstream graft pass.

OUTPUT FORMAT

For each draft, write a paragraph (3-5 sentences) covering voice quality, best moment, notable weaknesses, and a one-sentence justification for your quality verdict. Reference the scanner numbers.

Then on a line by itself for each draft (one line per draft):

QUALITY: Draft N — ACCEPTABLE
or
QUALITY: Draft N — UNACCEPTABLE — [one-sentence reason]

Then a graft paragraph naming specific lines from non-winning acceptable drafts worth transplanting, with beat locations.

Then on a line by itself:

RANKING: N, N, N, ...

(every ACCEPTABLE draft number from strongest to weakest in your advisory literary judgment. Do NOT include UNACCEPTABLE drafts in this line. Separated by commas.)

Then on the final line:

WINNER: N

(the top draft in your advisory ranking — again, this is advisory and may be overridden by the pipeline)

Nothing after that line."""


EVALUATOR_SCANNER_BLOCK = """=== MECHANICAL SCANNER RESULTS ===

{scanner_text}

=== CHAPTER OUTLINE ===

{outline_text}

"""


# ============================================================================
# Line-graft prompt — identify runner-up sentences that improve the winner
# ============================================================================

LINE_GRAFT_IDENTIFICATION_PROMPT = """You are comparing {N} drafts of the same chapter. Draft 1 is TOP 1 — the draft the pipeline selected as the shipping base. Drafts 2–{N} are acceptable runners-up.

Your job is to find sentences in TOP 1 that carry a specific flagged construction AND have a clean counterpart in a runner-up at the same beat. The goal is targeted pattern repair, not general improvement.

A graft fires ONLY when ALL THREE of these hold:

1. TOP 1's SENTENCE IS FLAGGED. The TOP 1 sentence carries one of the hard-cap constructions listed below. If the TOP 1 sentence is clean, leave it alone regardless of whether a runner-up has a "better" version. Do not graft on aesthetic grounds. Do not graft to tighten a sentence that is already clean. Do not graft to improve cadence.

   Flaggable constructions in TOP 1:
   - "the way X" observational framing (e.g., "the way he held the spoon," "the way she watched him")
   - Periphrastic observational framing ("as though he were," "in the manner of," "like a woman who")
   - "not X but Y" negation pivots in narration or interior thought (dialogue is permitted to use these)
   - Named emotions in narration — third-person-like form only ("a wave of sadness," "with a sense of dread"); first-person naming in Dinah's interior voice is PERMITTED and not a flag
   - Em-dash over-cap (count exceeds 7 for the whole chapter)

2. RUNNER-UP DONOR IS AT THE SAME BEAT AND DOES THE SAME NARRATIVE WORK. The donor sentence sits at the same beat as the flagged TOP 1 sentence and does the same thing — conveys the same information, moves the scene the same way, reveals the same character note. Transplanting it must not break continuity with the sentences around it in TOP 1.

3. DONOR IS CLEAN. The donor sentence itself contains ZERO of the flaggable constructions listed in condition 1. A donor that trades one tell for another is disqualified. Check the donor carefully against every item in the flag list before accepting it.

If TOP 1 carries a flagged construction but NO runner-up has a clean same-beat donor, leave the flagged sentence in place. Do not attempt to rewrite it. Do not substitute a donor that carries its own flag. The chapter ships with the flagged sentence intact.

Do NOT cap your list artificially. Return as many qualifying grafts as you find; return zero if none qualify.

SCANNER-FLAGGED PASSAGES IN TOP 1

The mechanical scanner flagged these passages in TOP 1 as carrying hard-cap violations. These are the sentences eligible for graft replacement. If this list is empty, there are no grafts to make — return NO_GRAFT.

{winner_flags}

SCANNER COUNTS PER DRAFT

{scanner_summary}

OUTPUT FORMAT — follow exactly:

GRAFT 1:
SOURCE: Draft [number]
<replace>[exact text from TOP 1 to remove]</replace>
<with>[exact text from the runner-up to insert]</with>
REASON: [one sentence naming which flagged construction in TOP 1 this replaces, and confirming the donor is clean of all flags]

GRAFT 2:
SOURCE: Draft [number]
<replace>[exact text from TOP 1 to remove]</replace>
<with>[exact text from the runner-up to insert]</with>
REASON: [same format]

... (as many GRAFT entries as you find)

Or if no flagged TOP 1 sentence has a clean same-beat donor:

NO_GRAFT

Quote the EXACT text to replace and the EXACT replacement. These strings will be used for automated find-and-replace, so character-level precision matters — copy them exactly as they appear in the drafts, including punctuation and spacing.

Do NOT force grafts. Graft ONLY when TOP 1 has a flagged construction AND a runner-up has the same beat clean AND the donor carries no flags of its own. If any condition fails, leave the sentence alone. Nothing after the last GRAFT entry or NO_GRAFT."""


# ============================================================================
# Data model
# ============================================================================

@dataclass
class RunRecord:
    run_id: str = ""
    timestamp: str = ""
    prompt_id: int = 0
    prompt_text: str = ""
    temperature: float = 0.7
    model: str = ""
    output_file: str = ""
    payload_file: str = ""
    meta_file: str = ""
    word_count: int = 0
    is_winner: bool = False
    evaluation_id: str = ""
    evaluation_rank: int = 0
    evaluator_model: str = ""
    evaluation_parse_status: str = ""
    evaluation_raw: str = ""
    # Mechanical scanner (deterministic, populated at generation time)
    scan_the_way_count: int = 0
    scan_periphrastic_count: int = 0
    scan_not_but_count: int = 0
    scan_em_dash_count: int = 0
    scan_em_dash_per_1k: float = 0.0
    scan_emotion_naming_count: int = 0
    scan_semicolons: int = 0
    scan_colons: int = 0
    scan_parens: int = 0
    scan_avg_sentence_len: float = 0.0
    scan_long_sentences_pct: float = 0.0
    scan_fragments_pct: float = 0.0
    scan_hard_cap_pass: bool = False
    scan_flagged_passages: str = ""
    # Quality floor verdict from the literary evaluator
    quality_verdict: str = ""  # "ACCEPTABLE" / "UNACCEPTABLE" / ""
    quality_reason: str = ""
    # Pipeline outcome for this draft
    pipeline_role: str = ""  # "top1_winner" / "graft_donor" / "dropped_unacceptable" / ""


RUN_FIELDS = list(RunRecord.__dataclass_fields__.keys())


# ============================================================================
# Mechanical scanner — deterministic, no LLM
# ============================================================================

# "The way X" family. Matches "the way a/an/the/he/she/it/they/we/I/you/<name>"
# plus "the way <word>" as a catch-all. Case-insensitive, word-boundary anchored
# so "gateway" does not match.
THE_WAY_PATTERN = re.compile(r"\bthe\s+way\s+\w+", re.IGNORECASE)

# Periphrastic observational (closes the loophole if the generator rewrites
# "the way she watched" as "as though she were watching" or "in the manner
# of someone watching").
PERIPHRASTIC_PATTERN = re.compile(
    r"\b(?:as\s+though\s+(?:he|she|it|they)\s+were|in\s+the\s+manner\s+of)\b",
    re.IGNORECASE,
)

# "Not X but Y" negation pivots. Kept tight to avoid catching dialogue —
# we re-check the match position against quote count before flagging.
NOT_BUT_PATTERN = re.compile(
    r"\bnot\s+(?:[a-z\s,']{1,50}?)\s+but\s+(?:[a-z]+)",
    re.IGNORECASE,
)

# Emotion-naming in narration (approximate). Catches "she felt X," "a wave
# of X," "with a sense of X," and "a <emotion> <verb>" patterns.
EMOTION_WORDS = (
    "anger|anxiety|anxious|bitter|calm|contempt|despair|disgust|dread|"
    "embarrassment|envy|fear|fearful|frustration|grief|guilt|happiness|"
    "happy|hope|hopeless|joy|joyful|loneliness|love|melancholy|nostalgia|"
    "panic|peace|pity|pride|proud|rage|regret|relief|remorse|resentment|"
    "sad|sadness|satisfaction|shame|shock|sorrow|surprise|tenderness|"
    "terror|tired|tiredness|weariness|weary|worry|yearning"
)
EMOTION_NAMING_PATTERN = re.compile(
    rf"\b(?:she\s+felt|he\s+felt|a\s+wave\s+of\s+(?:{EMOTION_WORDS})|"
    rf"a\s+flush\s+of\s+(?:{EMOTION_WORDS})|a\s+pang\s+of\s+(?:{EMOTION_WORDS})|"
    rf"with\s+a\s+sense\s+of\s+(?:{EMOTION_WORDS})|"
    rf"a\s+(?:{EMOTION_WORDS})\s+(?:settled|rose|came|washed|filled|took))\b",
    re.IGNORECASE,
)


def scan_draft(text: str) -> dict:
    """Run deterministic mechanical checks against a draft.

    Returns counts, percentages, a pass/fail flag, and flagged passages with
    context for human review. Pass/fail is conservative — a draft that trips
    any hard cap fails.
    """
    words = re.findall(r"\b[\w']+\b", text)
    wc = len(words) or 1

    flagged = []

    # "The way X"
    the_way_matches = list(THE_WAY_PATTERN.finditer(text))
    for m in the_way_matches[:30]:
        start = max(0, m.start() - 50)
        end = min(len(text), m.end() + 50)
        flagged.append({
            "rule": "the_way_x",
            "context": text[start:end].replace("\n", " ").strip(),
        })

    # Periphrastic observational
    periphrastic_matches = list(PERIPHRASTIC_PATTERN.finditer(text))
    for m in periphrastic_matches[:15]:
        start = max(0, m.start() - 50)
        end = min(len(text), m.end() + 50)
        flagged.append({
            "rule": "periphrastic_observational",
            "context": text[start:end].replace("\n", " ").strip(),
        })

    # "Not X but Y" — skip if inside quotes (dialogue)
    not_but_matches = []
    for m in NOT_BUT_PATTERN.finditer(text):
        before = text[: m.start()]
        # Normalize typographic quotes to count pairs
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 == 0:  # even → outside dialogue
            not_but_matches.append(m)
    for m in not_but_matches[:15]:
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        flagged.append({
            "rule": "not_x_but_y",
            "context": text[start:end].replace("\n", " ").strip(),
        })

    # Em-dashes
    em_dash_count = text.count("\u2014")
    em_per_1k = round(em_dash_count / wc * 1000, 2)

    # Emotion-naming
    emotion_matches = list(EMOTION_NAMING_PATTERN.finditer(text))
    for m in emotion_matches[:15]:
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        flagged.append({
            "rule": "emotion_naming",
            "context": text[start:end].replace("\n", " ").strip(),
        })

    # Punctuation
    semicolons = text.count(";")
    colons = len(re.findall(r"(?<!\d):(?!\d)", text))
    parens = text.count("(")

    # Sentence stats
    sentences = re.split(r"(?<=[.!?])\s+", text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sent_lens = [len(re.findall(r"\b[\w']+\b", s)) for s in sentences]
    total_sents = len(sentences) or 1
    avg_sent = round(sum(sent_lens) / len(sent_lens), 1) if sent_lens else 0.0
    long_sents = sum(1 for length in sent_lens if length > 40)
    long_pct = round(long_sents / total_sents * 100, 2)
    fragments = sum(1 for length in sent_lens if 1 <= length <= 3)
    frag_pct = round(fragments / total_sents * 100, 2)

    # Hard cap pass
    hard_cap_pass = (
        len(the_way_matches) == 0
        and len(periphrastic_matches) == 0
        and len(not_but_matches) == 0
        and em_dash_count <= 12
        and em_per_1k <= 3.0
    )

    return {
        "scan_the_way_count": len(the_way_matches),
        "scan_periphrastic_count": len(periphrastic_matches),
        "scan_not_but_count": len(not_but_matches),
        "scan_em_dash_count": em_dash_count,
        "scan_em_dash_per_1k": em_per_1k,
        "scan_emotion_naming_count": len(emotion_matches),
        "scan_semicolons": semicolons,
        "scan_colons": colons,
        "scan_parens": parens,
        "scan_avg_sentence_len": avg_sent,
        "scan_long_sentences_pct": long_pct,
        "scan_fragments_pct": frag_pct,
        "scan_hard_cap_pass": hard_cap_pass,
        "scan_flagged_passages": json.dumps(flagged[:40], ensure_ascii=False),
    }


def format_scan_summary(scan: dict) -> str:
    """One-line summary of scan results for UI display."""
    flags = []
    if scan["scan_the_way_count"]:
        flags.append(f"the way×{scan['scan_the_way_count']}")
    if scan["scan_periphrastic_count"]:
        flags.append(f"periphrastic×{scan['scan_periphrastic_count']}")
    if scan["scan_not_but_count"]:
        flags.append(f"not-but×{scan['scan_not_but_count']}")
    if scan["scan_em_dash_count"] > 12 or scan["scan_em_dash_per_1k"] > 3.0:
        flags.append(f"em-dash×{scan['scan_em_dash_count']}")
    if scan["scan_emotion_naming_count"]:
        flags.append(f"emotion×{scan['scan_emotion_naming_count']}")
    status = "PASS" if scan["scan_hard_cap_pass"] else "FAIL"
    return f"{status} ({', '.join(flags) if flags else 'clean'})"


# ============================================================================
# Originality color ranker — deterministic, no LLM
#
# Background:
#   The mechanical scanner above counts textual patterns in the raw draft
#   before Originality sees it. Its scores do not predict Originality's
#   eventual per-sentence verdicts. Calibration against four v9/v10 samples
#   (47%, 85%, 93%, 99% scorers) showed that the scanner's "cleanest" draft
#   was actually the second-worst scorer, and that Originality's color-coded
#   exports carry a perfectly monotonic signal.
#
# Calibration (hex fill G–R offset → class):
#   g - r >= 15  STRONG_GREEN   (unambiguously human)
#   g - r >=  5  mild_green
#   g - r >= -5  neutral
#   g - r >=-15  mild_orange
#   g - r < -15  STRONG_ORANGE  (unambiguously AI)
#
# Rank score = -(longest_O ** 2) * 3      dominant, superlinear
#              - O_in_multi_clusters        concentration cost
#              - total_O * 0.3              background orange volume
#              + (mild_green - mild_orange) * 0.5  middle-band refinement
#
#   Strong-green count is deliberately ignored: it is non-monotonic with
#   score in the middle range and tends to co-occur with strong orange
#   (the "bimodal composed register" problem). Selecting for strong green
#   actively steers toward worse drafts.
# ============================================================================

import zipfile

_ORIG_HEX_FILL_RE = re.compile(r'w:fill="([0-9A-Fa-f]{6})"')


def _classify_originality_fill(hex_color: str) -> str:
    """Classify a single hex fill by its green-vs-orange offset."""
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    diff = g - r
    if diff >= 15:
        return "STRONG_GREEN"
    if diff >= 5:
        return "mild_green"
    if diff >= -5:
        return "neutral"
    if diff >= -15:
        return "mild_orange"
    return "STRONG_ORANGE"


def _extract_originality_fills(docx_bytes: bytes) -> List[str]:
    """Extract w:fill hex values from an Originality-exported docx, in order."""
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
        with z.open("word/document.xml") as f:
            xml = f.read().decode("utf-8", errors="replace")
    return _ORIG_HEX_FILL_RE.findall(xml)


def compute_originality_metrics(docx_bytes: bytes) -> dict:
    """Compute ranking metrics for a single Originality-exported docx.

    Returns a dict with the per-class counts, strong-orange cluster
    statistics, and the final rank_score.
    """
    fills = _extract_originality_fills(docx_bytes)
    classes = [_classify_originality_fill(h) for h in fills]

    counts = {
        "STRONG_GREEN": 0, "mild_green": 0, "neutral": 0,
        "mild_orange": 0, "STRONG_ORANGE": 0,
    }
    for c in classes:
        counts[c] += 1

    short = "".join("O" if c == "STRONG_ORANGE" else "." for c in classes)
    run_lens = [len(r) for r in re.findall(r"O+", short)]
    longest_O = max(run_lens) if run_lens else 0
    in_clusters = sum(l for l in run_lens if l >= 2)
    total_O = counts["STRONG_ORANGE"]
    avg_cluster = (sum(run_lens) / len(run_lens)) if run_lens else 0.0

    score = (
        -(longest_O ** 2) * 3.0
        - in_clusters
        - total_O * 0.3
        + (counts["mild_green"] - counts["mild_orange"]) * 0.5
    )

    return {
        "total_runs": len(fills),
        "strong_green": counts["STRONG_GREEN"],
        "mild_green": counts["mild_green"],
        "neutral": counts["neutral"],
        "mild_orange": counts["mild_orange"],
        "strong_orange": total_O,
        "longest_strong_O": longest_O,
        "strong_O_in_clusters": in_clusters,
        "avg_strong_O_cluster": round(avg_cluster, 2),
        "rank_score": round(score, 2),
    }


def _extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    """Extract plain text from a docx file (for matching to stored drafts)."""
    try:
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as z:
            with z.open("word/document.xml") as f:
                xml = f.read().decode("utf-8", errors="replace")
        # Strip tags, keep text content
        text = re.sub(r"<[^>]+>", " ", xml)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception:
        return ""


def _normalize_for_matching(text: str) -> str:
    """Aggressive normalization for text-overlap matching."""
    text = re.sub(r"\s+", " ", text).strip().lower()
    # Strip punctuation that Originality sometimes reformats
    text = re.sub(r"[\u2018\u2019\u201c\u201d\u2013\u2014'\",.?!;:()]", "", text)
    return text


def match_originality_docx_to_draft(
    docx_bytes: bytes, candidate_drafts: list
) -> Optional[str]:
    """Match an uploaded Originality docx to a run_id in candidate_drafts.

    Uses text-overlap matching: the uploaded doc's first 400 characters of
    plain text (normalized) are checked against each candidate draft's
    normalized text. Returns the best-matching run_id, or None if no
    candidate has a clear overlap.
    """
    orig_text = _extract_text_from_docx_bytes(docx_bytes)
    if not orig_text:
        return None

    norm_orig = _normalize_for_matching(orig_text)
    if len(norm_orig) < 100:
        return None

    # Use a distinctive signature from the middle of the doc, where
    # Originality's headers/footers are less likely to interfere.
    sig_start = min(200, len(norm_orig) // 4)
    sig_end = min(sig_start + 400, len(norm_orig))
    sig = norm_orig[sig_start:sig_end]

    best_run_id = None
    best_score = 0
    for d in candidate_drafts:
        norm_draft = _normalize_for_matching(d.get("text", ""))
        if not norm_draft:
            continue
        # Count overlap by sliding a short window from sig through the draft
        overlap = 0
        window = 60
        for i in range(0, len(sig) - window, window // 2):
            if sig[i:i + window] in norm_draft:
                overlap += 1
        if overlap > best_score:
            best_score = overlap
            best_run_id = d.get("run_id")

    # Require at least 3 window hits to count as a match (~180 chars overlap)
    return best_run_id if best_score >= 3 else None


def rank_by_originality_reports(
    reports_by_run_id: dict, candidate_drafts: list
) -> list:
    """Rank drafts by Originality color-based score, highest first.

    Arguments:
        reports_by_run_id: {run_id: metrics_dict} — output of
                           compute_originality_metrics for each report.
        candidate_drafts: the drafts list from the batch, used for filenames.

    Returns a list of dicts:
        [{"run_id": ..., "rank_score": ..., "metrics": {...}, "rank": 1, ...}]
    sorted by rank_score descending.
    """
    rows = []
    for run_id, metrics in reports_by_run_id.items():
        rows.append({
            "run_id": run_id,
            "rank_score": metrics.get("rank_score", -99999),
            "metrics": metrics,
        })
    rows.sort(key=lambda r: r["rank_score"], reverse=True)
    for i, row in enumerate(rows, 1):
        row["rank"] = i
    return rows


# ============================================================================
# File I/O
# ============================================================================

def ensure_dirs():
    RUNS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)
    FINAL_DIR.mkdir(exist_ok=True)


def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists() and path.stat().st_size > 0:
        df = pd.read_csv(path, dtype=str)
        for col in RUN_FIELDS:
            if col not in df.columns:
                df[col] = ""
        return df
    return pd.DataFrame(columns=RUN_FIELDS)


def append_record(path: Path, record: RunRecord):
    df = load_csv(path)
    new_row = pd.DataFrame([asdict(record)])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(path, index=False)


def update_record(path: Path, run_id: str, updates: dict):
    df = load_csv(path)
    mask = df["run_id"].astype(str) == str(run_id)
    for k, v in updates.items():
        if k in df.columns:
            df[k] = df[k].astype(object)
        df.loc[mask, k] = v
    df.to_csv(path, index=False)


def update_records_bulk(path: Path, run_ids: list, updates: dict):
    df = load_csv(path)
    mask = df["run_id"].astype(str).isin([str(r) for r in run_ids])
    for k, v in updates.items():
        if k in df.columns:
            df[k] = df[k].astype(object)
        df.loc[mask, k] = v
    df.to_csv(path, index=False)


def extract_text_from_upload(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".txt"):
            data = uploaded_file.read()
            uploaded_file.seek(0)
            return data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data
        elif name.endswith(".docx") and DOCX_AVAILABLE:
            doc = python_docx.Document(uploaded_file)
            uploaded_file.seek(0)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        st.warning(f"Could not read {uploaded_file.name}: {e}")
    return ""


# ============================================================================
# API key loading
# ============================================================================

def clean_api_key(value: str) -> str:
    return value.strip().strip("'\"").strip()


def load_api_key() -> tuple[str, str]:
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            key = clean_api_key(str(st.secrets["ANTHROPIC_API_KEY"]))
            if key:
                return key, "Streamlit secrets"
    except Exception:
        pass

    env_key = clean_api_key(os.environ.get("ANTHROPIC_API_KEY", ""))
    if env_key:
        return env_key, "environment variable"

    return "", ""


# ============================================================================
# Prompt loading
# ============================================================================

def load_prompts() -> pd.DataFrame:
    path = Path(PROMPTS_CSV)
    if not path.exists():
        return pd.DataFrame(columns=["id", "text", "category"])
    df = pd.read_csv(path)
    if "id" not in df.columns or "text" not in df.columns:
        st.error(f"{PROMPTS_CSV} must have 'id' and 'text' columns.")
        return pd.DataFrame(columns=["id", "text", "category"])
    if "category" not in df.columns:
        df["category"] = ""
    return df


# ============================================================================
# Payload construction
# ============================================================================

SYSTEM_PROMPT = (
    "Follow the user's instructions exactly. "
    "Do not add commentary, headers, or meta-text to your response."
)


def build_payload_text(prompt_text: str, doc_texts: dict) -> str:
    parts = [prompt_text.strip()]
    for label, text in doc_texts.items():
        if text.strip():
            parts.append(f"\n\n=== {label.upper()} ===\n\n{text.strip()}")
    parts.append(
        "\n\nWrite the full chapter now. Return plain text only, "
        "with normal paragraph breaks and no commentary."
    )
    return "\n".join(parts)


def build_message_blocks(prompt_text: str, doc_texts: dict) -> list:
    blocks = [{"type": "text", "text": prompt_text.strip()}]
    for label, text in doc_texts.items():
        if text.strip():
            blocks.append({
                "type": "text",
                "text": f"[{label.upper()}]\n\n{text.strip()}",
            })
    blocks.append({
        "type": "text",
        "text": (
            "Write the full chapter now. Return plain text only, "
            "with normal paragraph breaks and no commentary."
        ),
    })
    return blocks


# ============================================================================
# Generation
# ============================================================================

def generate_chapter(client, model: str, temperature: float, message_blocks: list) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_GEN_TOKENS,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": message_blocks}],
    )
    return "\n".join(b.text for b in resp.content if getattr(b, "text", None))


# ============================================================================
# Literary evaluation — unchanged shape; adds strong-beat extraction
# ============================================================================

def evaluate_drafts_with_anthropic(
    client, model: str, drafts: list,
    outline_text: str = "", scan_by_run_id: dict = None,
) -> dict:
    n = len(drafts)

    # Build scanner summary for each draft
    scanner_lines = []
    for i, d in enumerate(drafts, 1):
        scan = (scan_by_run_id or {}).get(d["run_id"], {})
        if scan:
            scanner_lines.append(
                f"Draft {i} (run_id: {d['run_id']}): "
                f"word_count={len(d['text'].split())}, "
                f"the_way={scan.get('scan_the_way_count', '?')}, "
                f"periphrastic={scan.get('scan_periphrastic_count', '?')}, "
                f"not_but={scan.get('scan_not_but_count', '?')}, "
                f"em_dash={scan.get('scan_em_dash_count', '?')} "
                f"({scan.get('scan_em_dash_per_1k', '?')}/1k), "
                f"emotion_naming={scan.get('scan_emotion_naming_count', '?')}, "
                f"hard_cap_pass={scan.get('scan_hard_cap_pass', '?')}"
            )
        else:
            scanner_lines.append(
                f"Draft {i} (run_id: {d['run_id']}): "
                f"word_count={len(d['text'].split())}, scanner data not available"
            )

    scanner_text = "\n".join(scanner_lines)

    parts = [
        EVALUATOR_PROMPT.format(N=n),
        "\n\n",
        EVALUATOR_SCANNER_BLOCK.format(
            scanner_text=scanner_text,
            outline_text=outline_text.strip() if outline_text else "(no outline provided)",
        ),
    ]
    for i, d in enumerate(drafts, 1):
        parts.append(f"=== DRAFT {i} (run_id: {d['run_id']}) ===\n\n{d['text']}\n\n")

    resp = client.messages.create(
        model=model,
        max_tokens=MAX_EVAL_TOKENS,
        messages=[{"role": "user", "content": "".join(parts)}],
    )
    raw = "\n".join(b.text for b in resp.content if getattr(b, "text", None))

    # Parse QUALITY verdicts — one per draft, keyed by 1-indexed position
    quality_by_index = {}
    quality_pattern = re.compile(
        r"QUALITY:\s*Draft\s*(\d+)\s*[—-]+\s*(ACCEPTABLE|UNACCEPTABLE)"
        r"(?:\s*[—-]+\s*(.+?))?(?=\n|$)",
        re.IGNORECASE,
    )
    for m in quality_pattern.finditer(raw):
        idx = int(m.group(1))
        verdict = m.group(2).upper()
        reason = (m.group(3) or "").strip()
        if 1 <= idx <= n:
            quality_by_index[idx] = {"verdict": verdict, "reason": reason}

    # Drafts without an explicit verdict default to ACCEPTABLE — the floor is
    # lenient, and a missing line should not silently drop a draft.
    for i in range(1, n + 1):
        if i not in quality_by_index:
            quality_by_index[i] = {
                "verdict": "ACCEPTABLE",
                "reason": "(no explicit verdict in evaluator output; defaulted to ACCEPTABLE)",
            }

    # Map verdicts back to run_ids
    quality_by_run_id = {}
    for i, d in enumerate(drafts, 1):
        quality_by_run_id[d["run_id"]] = quality_by_index[i]

    ranking = list(range(1, n + 1))
    parse_status = "clean"
    rank_match = re.search(r"RANKING:\s*([0-9,\s]+)", raw)
    if rank_match:
        nums = [int(x.strip()) for x in rank_match.group(1).split(",") if x.strip().isdigit()]
        seen = set()
        deduped = []
        for x in nums:
            if 1 <= x <= n and x not in seen:
                seen.add(x)
                deduped.append(x)
        # Advisory ranking should include only ACCEPTABLE drafts; anything the
        # evaluator omitted from the RANKING line we treat as already signaled
        # UNACCEPTABLE via the QUALITY lines. If a draft was marked ACCEPTABLE
        # but is missing from RANKING, append it at the end as a safeguard.
        acceptable_idxs = [
            i for i in range(1, n + 1)
            if quality_by_index[i]["verdict"] == "ACCEPTABLE"
        ]
        missing_acceptable = [i for i in acceptable_idxs if i not in seen]
        ranking = [x for x in deduped if quality_by_index[x]["verdict"] == "ACCEPTABLE"]
        ranking += missing_acceptable
        if missing_acceptable:
            parse_status = "partial"
    else:
        # No RANKING line — fall back to original draft order, acceptable drafts only
        ranking = [
            i for i in range(1, n + 1)
            if quality_by_index[i]["verdict"] == "ACCEPTABLE"
        ]
        parse_status = "no_ranking_line"

    winner_match = re.search(r"WINNER:\s*(\d+)", raw)
    if winner_match:
        winner_idx = int(winner_match.group(1))
    else:
        winner_idx = ranking[0] if ranking else 1
        if parse_status == "clean":
            parse_status = "no_winner_line"

    # Force winner to be an acceptable draft
    if (1 <= winner_idx <= n
            and quality_by_index[winner_idx]["verdict"] == "UNACCEPTABLE"):
        winner_idx = ranking[0] if ranking else winner_idx
    winner_idx = max(1, min(winner_idx, n))
    winner_run_id = drafts[winner_idx - 1]["run_id"]

    return {
        "winner_run_id": winner_run_id,
        "winner_index": winner_idx,
        "ranking": ranking,  # advisory, acceptable drafts only, 1-indexed positions
        "quality_by_run_id": quality_by_run_id,
        "quality_by_index": quality_by_index,
        "raw_text": raw,
        "parse_status": parse_status,
        "model": model,
    }


# ============================================================================
# Line-graft — identify runner-up sentences, apply via string replacement
# ============================================================================

def parse_line_grafts(raw: str) -> list:
    """Parse GRAFT entries from the identification response.
    Returns list of dicts with keys: source_draft, replace, with_text, reason.
    """
    if "NO_GRAFT" in raw:
        return []

    grafts = []
    replace_pat = re.compile(r"<replace>(.*?)</replace>", re.DOTALL)
    with_pat = re.compile(r"<with>(.*?)</with>", re.DOTALL)
    source_pat = re.compile(r"SOURCE:\s*Draft\s*(\d+)")
    reason_pat = re.compile(r"REASON:\s*(.+)")

    sections = re.split(r"GRAFT\s+\d+:", raw)
    for section in sections[1:]:
        replace_m = replace_pat.search(section)
        with_m = with_pat.search(section)
        source_m = source_pat.search(section)
        reason_m = reason_pat.search(section)

        if replace_m and with_m:
            grafts.append({
                "source_draft": int(source_m.group(1)) if source_m else 0,
                "replace": replace_m.group(1).strip(),
                "with_text": with_m.group(1).strip(),
                "reason": reason_m.group(1).strip() if reason_m else "",
            })

    return grafts


def _build_winner_flags_text(winner_scan: dict) -> str:
    """Turn the winner's scan_flagged_passages JSON into human-readable lines
    for injection into the line-graft prompt.
    """
    raw = (winner_scan or {}).get("scan_flagged_passages", "")
    if not raw:
        return "(No hard-cap violations flagged in the winner.)"
    try:
        flags = json.loads(raw)
    except Exception:
        return "(Winner flag data could not be parsed.)"
    if not flags:
        return "(No hard-cap violations flagged in the winner.)"
    lines = []
    for f in flags:
        rule = f.get("rule", "?")
        ctx = f.get("context", "").strip()
        lines.append(f"- [{rule}] …{ctx}…")
    return "\n".join(lines)


def _build_scanner_summary_text(drafts_ranked: list, scan_by_run_id: dict) -> str:
    """One line per draft with hard-cap counts, in rank order."""
    lines = []
    for i, d in enumerate(drafts_ranked, 1):
        scan = (scan_by_run_id or {}).get(d["run_id"], {})
        label = "WINNER" if i == 1 else f"RUNNER-UP #{i - 1}"
        if not scan:
            lines.append(f"Draft {i} ({label}, run_id: {d['run_id']}): scan unavailable")
            continue
        lines.append(
            f"Draft {i} ({label}, run_id: {d['run_id']}): "
            f"the_way={scan.get('scan_the_way_count', 0)}, "
            f"periphrastic={scan.get('scan_periphrastic_count', 0)}, "
            f"not_but={scan.get('scan_not_but_count', 0)}, "
            f"em_dash={scan.get('scan_em_dash_count', 0)} "
            f"({scan.get('scan_em_dash_per_1k', 0)}/1k), "
            f"emotion_naming={scan.get('scan_emotion_naming_count', 0)}"
        )
    return "\n".join(lines)


def _donor_sentence_is_clean(donor_sentence: str) -> bool:
    """Reject a proposed donor sentence if it itself contains any hard-cap
    pattern. This enforces the prompt's condition 3 deterministically in
    case the model misjudges its own candidate.
    """
    if THE_WAY_PATTERN.search(donor_sentence):
        return False
    if PERIPHRASTIC_PATTERN.search(donor_sentence):
        return False
    # "not X but Y" — same quote-count discipline as the main scanner
    for m in NOT_BUT_PATTERN.finditer(donor_sentence):
        before = donor_sentence[: m.start()]
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 == 0:
            return False
    if EMOTION_NAMING_PATTERN.search(donor_sentence):
        return False
    return True


def run_line_graft_experiment(
    client,
    eval_model: str,
    drafts_ranked: list,
    scan_by_run_id: dict,
    batch_stub: str,
) -> dict:
    """Identify runner-up sentences that improve the winner at the same beat,
    then apply via deterministic string replacement.

    Args:
        drafts_ranked: list of draft dicts in literary-ranking order
                       (index 0 = winner). Each has 'run_id' and 'text'.
        scan_by_run_id: per-draft mechanical scan dict. The winner's
                        scan_flagged_passages gets injected into the prompt
                        as "known weak spots"; donor sentences that themselves
                        carry hard-cap patterns get rejected.
        batch_stub: for file naming.

    Returns dict with:
      - grafted: bool
      - grafts: list of applied graft dicts
      - grafts_attempted: list of all identified graft dicts (before any filter)
      - grafts_rejected_dirty_donor: donor sentences rejected for carrying
                                     hard-cap patterns
      - grafts_rejected_no_match: REPLACE strings that didn't appear in winner
      - grafted_text: the modified winner text (empty if no grafts applied)
      - grafted_path: file path (empty if no grafts applied)
      - grafted_scan: mechanical scan of the grafted output (diagnostic only —
                      does not gate shipping)
      - raw: the identification model's full output
    """
    result = {
        "grafted": False,
        "grafts": [],
        "grafts_attempted": [],
        "grafts_rejected_dirty_donor": [],
        "grafts_rejected_no_match": [],
        "grafted_text": "",
        "grafted_path": "",
        "grafted_scan": None,
        "raw": "",
    }

    n = len(drafts_ranked)
    if n < 2:
        return result

    winner_run_id = drafts_ranked[0]["run_id"]
    winner_scan = (scan_by_run_id or {}).get(winner_run_id, {})
    winner_flags_text = _build_winner_flags_text(winner_scan)
    scanner_summary_text = _build_scanner_summary_text(drafts_ranked, scan_by_run_id)

    # Build prompt with all drafts in rank order, winner flags, and scanner summary
    prompt_text = LINE_GRAFT_IDENTIFICATION_PROMPT.format(
        N=n,
        winner_flags=winner_flags_text,
        scanner_summary=scanner_summary_text,
    )
    parts = [prompt_text]
    for i, d in enumerate(drafts_ranked, 1):
        label = "WINNER" if i == 1 else "RUNNER-UP"
        parts.append(f"\n\n=== DRAFT {i} ({label}, run_id: {d['run_id']}) ===\n\n{d['text']}")

    resp = client.messages.create(
        model=eval_model,
        max_tokens=MAX_EVAL_TOKENS,
        messages=[{"role": "user", "content": "".join(parts)}],
    )
    raw = "\n".join(b.text for b in resp.content if getattr(b, "text", None))
    result["raw"] = raw

    grafts = parse_line_grafts(raw)
    result["grafts_attempted"] = list(grafts)

    if not grafts:
        return result

    # Filter donor sentences that carry hard-cap patterns — enforces prompt
    # condition 3 deterministically. A dirty donor sentence can't cure a
    # winner's violation, and shouldn't be allowed to introduce a new one.
    clean_grafts = []
    for g in grafts:
        if _donor_sentence_is_clean(g["with_text"]):
            clean_grafts.append(g)
        else:
            result["grafts_rejected_dirty_donor"].append(g)

    if not clean_grafts:
        return result

    # Apply grafts via deterministic string replacement
    winner_text = drafts_ranked[0]["text"]
    grafted_text = winner_text
    applied = []
    for g in clean_grafts:
        if g["replace"] in grafted_text:
            grafted_text = grafted_text.replace(g["replace"], g["with_text"], 1)
            applied.append(g)
        else:
            result["grafts_rejected_no_match"].append(g)

    if not applied:
        return result

    result["grafted"] = True
    result["grafts"] = applied
    result["grafted_text"] = grafted_text

    # Diagnostic scan of the grafted output — reported but not gated.
    # Grafts are always shipped if they applied, because a graft with
    # one remaining violation can still be an improvement over the
    # winner's multiple violations at the same beats.
    result["grafted_scan"] = scan_draft(grafted_text)

    grafted_path = FINAL_DIR / f"TOP1_GRAFTED_{batch_stub}.txt"
    save_text(grafted_path, grafted_text)
    result["grafted_path"] = str(grafted_path)

    return result


# ============================================================================
# File naming
# ============================================================================

def make_file_stub(prompt_id: int, temperature: float, model: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.split("-")[-1][:6] if "-" in model else model[:6]
    return f"P{prompt_id} T{temperature} {model_short} {ts}"


def make_winner_filename(prompt_id: int, temperature: float, model: str) -> str:
    stub = make_file_stub(prompt_id, temperature, model)
    return f"WINNER {stub}.txt"


def make_batch_stub(batch_timestamp: str) -> str:
    return f"batch_{batch_timestamp}"


# ============================================================================
# Pipeline — Q1 quality floor, Q2 scanner-ranked TOP 1, Q3 graft
# ============================================================================

def _extract_target_word_count(outline_text: str) -> Optional[int]:
    """Scan the outline for a word-count target like '3800-4200' or
    '~4000 words'. Returns the midpoint of a range, or the single value,
    or None if no target is found.
    """
    if not outline_text:
        return None
    # Range form: "3800-4200 words", "3,800–4,200"
    range_m = re.search(
        r"(\d{1,2},?\d{3})\s*[–\-]\s*(\d{1,2},?\d{3})\s*words?",
        outline_text, re.IGNORECASE,
    )
    if range_m:
        lo = int(range_m.group(1).replace(",", ""))
        hi = int(range_m.group(2).replace(",", ""))
        return (lo + hi) // 2
    # Single form: "about 4000 words", "~4000 words"
    single_m = re.search(
        r"(?:about|approximately|around|~)?\s*(\d{1,2},?\d{3})\s*words?",
        outline_text, re.IGNORECASE,
    )
    if single_m:
        return int(single_m.group(1).replace(",", ""))
    return None


def _scanner_violation_score(scan: dict) -> int:
    """Sum the hard-cap violations for a draft. Lower is better for Q2
    ranking. Em-dashes above 12 count as (count - 12) to match the scanner's
    hard cap; at-or-below-cap counts as 0 for this purpose.
    """
    if not scan:
        return 9999
    em_over_cap = max(0, scan.get("scan_em_dash_count", 0) - 12)
    return (
        scan.get("scan_the_way_count", 0)
        + scan.get("scan_periphrastic_count", 0)
        + scan.get("scan_not_but_count", 0)
        + scan.get("scan_emotion_naming_count", 0)
        + em_over_cap
    )


def run_pipeline(
    client,
    eval_model: str,
    drafts: list,
    scan_by_run_id: dict,
    outline_text: str,
    top_n: int,
    batch_stub: str,
) -> dict:
    """Run the full pipeline on a batch of drafts.

    Flow:
      Q1. Literary evaluator applies a lenient quality floor to each draft.
          UNACCEPTABLE drafts are dropped. If zero drafts clear the floor,
          the pipeline halts — no FINAL is shipped.
      Q2. Among acceptable drafts, rank by scanner-violation count (ascending).
          Ties break on word-count distance from the outline's target, then
          on the evaluator's advisory literary ranking. The top-ranked draft
          is TOP 1.
      Q3. Sentence-graft pass looks for lines in the acceptable runners-up
          that would improve TOP 1's detector profile without hurting quality.
          Applied grafts produce TOP1_GRAFTED, which becomes FINAL.
          Otherwise TOP 1 ships unchanged.

    Returns dict with:
      - halt: bool — True if no draft cleared the quality floor
      - halt_reason: str
      - quality_by_run_id: {run_id: {"verdict": ..., "reason": ...}}
      - acceptable_run_ids: list of run_ids that cleared the floor
      - dropped_run_ids: list of run_ids marked UNACCEPTABLE
      - pipeline_ranking: list of run_ids in Q2 order (TOP 1 first); may be
                          shorter than `drafts` because unacceptable drafts
                          are excluded
      - literary_ranking: advisory ranking from the evaluator (acceptable
                          drafts only), used for tiebreak reporting
      - literary_winner_run_id: evaluator's advisory WINNER (may differ from
                                pipeline TOP 1)
      - top1_run_id: the pipeline's TOP 1 — what actually ships
      - final_text, final_path, final_source
      - top_paths: TOP1..TOP_N files for the acceptable drafts only
      - eval_raw: full evaluator output
      - line_graft: dict from the sentence-graft pass (empty if halted)
    """
    result = {
        "halt": False,
        "halt_reason": "",
        "quality_by_run_id": {},
        "acceptable_run_ids": [],
        "dropped_run_ids": [],
        "pipeline_ranking": [],
        "literary_ranking": [],
        "literary_winner_run_id": "",
        "top1_run_id": "",
        "final_text": "",
        "final_path": "",
        "final_source": "",
        "top_paths": [],
        "eval_raw": "",
        "line_graft": {},
    }

    # --- Stage A: Literary evaluation + quality floor ---
    lit = evaluate_drafts_with_anthropic(
        client, eval_model, drafts,
        outline_text=outline_text,
        scan_by_run_id=scan_by_run_id,
    )
    result["eval_raw"] = lit["raw_text"]
    result["quality_by_run_id"] = lit.get("quality_by_run_id", {})

    # Build the advisory literary ranking (run_ids, acceptable drafts only)
    # The evaluator has already excluded UNACCEPTABLE drafts from lit["ranking"].
    lit_ranking_ids = [drafts[i - 1]["run_id"] for i in lit["ranking"]]
    result["literary_ranking"] = lit_ranking_ids
    result["literary_winner_run_id"] = lit["winner_run_id"]

    # Partition drafts by verdict
    acceptable_ids = []
    dropped_ids = []
    for d in drafts:
        verdict = result["quality_by_run_id"].get(d["run_id"], {}).get("verdict", "")
        if verdict == "UNACCEPTABLE":
            dropped_ids.append(d["run_id"])
        else:
            acceptable_ids.append(d["run_id"])
    result["acceptable_run_ids"] = acceptable_ids
    result["dropped_run_ids"] = dropped_ids

    # --- Q1 halt: no drafts cleared the floor ---
    if not acceptable_ids:
        result["halt"] = True
        result["halt_reason"] = (
            "No draft cleared the quality floor. "
            "Regenerate this batch — the pipeline will not ship an unacceptable draft."
        )
        return result

    # --- Stage B: Q2 ranking — scanner-ascending, tiebreak on word-count
    # distance from outline target, final tiebreak on advisory literary rank ---
    target_wc = _extract_target_word_count(outline_text)

    literary_rank_by_id = {
        run_id: pos for pos, run_id in enumerate(lit_ranking_ids, 1)
    }

    def sort_key(run_id: str):
        scan = scan_by_run_id.get(run_id, {})
        draft_obj = next((d for d in drafts if d["run_id"] == run_id), None)
        wc = len(draft_obj["text"].split()) if draft_obj else 0
        violation_score = _scanner_violation_score(scan)
        wc_distance = abs(wc - target_wc) if target_wc is not None else 0
        lit_rank = literary_rank_by_id.get(run_id, 9999)
        return (violation_score, wc_distance, lit_rank)

    pipeline_ranking = sorted(acceptable_ids, key=sort_key)
    result["pipeline_ranking"] = pipeline_ranking
    result["top1_run_id"] = pipeline_ranking[0]

    # --- Stage B2: Top-N export (acceptable drafts only, in Q2 order) ---
    top_paths = []
    for rank_pos, run_id in enumerate(pipeline_ranking[:top_n], 1):
        draft_obj = next((d for d in drafts if d["run_id"] == run_id), None)
        if draft_obj is None:
            continue
        top_filename = f"TOP{rank_pos}_{batch_stub}_{run_id}.txt"
        top_path = FINAL_DIR / top_filename
        save_text(top_path, draft_obj["text"])
        top_paths.append(top_path)
    result["top_paths"] = top_paths

    # --- Stage C: Q3 sentence-graft pass ---
    top1_text = next(
        (d["text"] for d in drafts if d["run_id"] == result["top1_run_id"]),
        "",
    )

    if len(pipeline_ranking) >= 2:
        drafts_ranked = []
        for run_id in pipeline_ranking[:top_n]:
            draft_obj = next((d for d in drafts if d["run_id"] == run_id), None)
            if draft_obj:
                drafts_ranked.append(draft_obj)
        if len(drafts_ranked) >= 2:
            line_graft = run_line_graft_experiment(
                client, eval_model, drafts_ranked, scan_by_run_id, batch_stub,
            )
            result["line_graft"] = line_graft

    # --- Stage D: Ship ---
    lg = result.get("line_graft") or {}
    if lg.get("grafted") and lg.get("grafted_text"):
        result["final_source"] = "top1_grafted"
        result["final_text"] = lg["grafted_text"]
        final_path = FINAL_DIR / f"FINAL_{batch_stub}_top1_grafted.txt"
        save_text(final_path, lg["grafted_text"])
        result["final_path"] = str(final_path)
    else:
        result["final_source"] = "top1_ungrafted"
        result["final_text"] = top1_text
        final_path = FINAL_DIR / f"FINAL_{batch_stub}_top1_ungrafted.txt"
        save_text(final_path, top1_text)
        result["final_path"] = str(final_path)

    return result


def write_batch_summary(
    pipeline_result: dict,
    drafts: list,
    scan_by_run_id: dict,
    batch_stub: str,
    temperatures: list,
    prompts_used: list,
) -> Path:
    """Produce a human-readable summary of the batch."""
    lines = []
    lines.append(f"BATCH SUMMARY: {batch_stub}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Drafts in batch: {len(drafts)}")
    lines.append(f"Temperatures: {temperatures}")
    lines.append(f"Prompts used: P{', P'.join(str(p) for p in prompts_used)}")

    if pipeline_result.get("halt"):
        lines.append("")
        lines.append("=" * 60)
        lines.append("PIPELINE HALTED")
        lines.append("=" * 60)
        lines.append(pipeline_result.get("halt_reason", "Unknown halt."))

    lines.append("")
    lines.append("=" * 60)
    lines.append("Q1 — QUALITY FLOOR")
    lines.append("=" * 60)
    quality = pipeline_result.get("quality_by_run_id", {})
    acceptable_ids = pipeline_result.get("acceptable_run_ids", [])
    dropped_ids = pipeline_result.get("dropped_run_ids", [])
    lines.append(f"Acceptable: {len(acceptable_ids)} / {len(drafts)}")
    if dropped_ids:
        lines.append(f"Dropped (UNACCEPTABLE): {len(dropped_ids)}")
    lines.append("")
    for d in drafts:
        q = quality.get(d["run_id"], {})
        verdict = q.get("verdict", "?")
        reason = q.get("reason", "")
        lines.append(f"  {d['run_id']}: {verdict}")
        if reason:
            lines.append(f"    {reason}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("MECHANICAL SCAN RESULTS")
    lines.append("=" * 60)
    clean_count = sum(1 for d in drafts if scan_by_run_id.get(d["run_id"], {}).get("scan_hard_cap_pass"))
    lines.append(f"Clean (all hard caps held): {clean_count} / {len(drafts)}")
    lines.append("(Scan drives Q2 ranking among acceptable drafts.)")
    lines.append("")
    for d in drafts:
        scan = scan_by_run_id.get(d["run_id"], {})
        summary = format_scan_summary(scan) if scan else "not scanned"
        lines.append(f"  {d['run_id']}: {summary}")
        lines.append(
            f"    the-way={scan.get('scan_the_way_count', 0)}, "
            f"periphrastic={scan.get('scan_periphrastic_count', 0)}, "
            f"not-but={scan.get('scan_not_but_count', 0)}, "
            f"em-dash={scan.get('scan_em_dash_count', 0)} "
            f"({scan.get('scan_em_dash_per_1k', 0)}/1k), "
            f"emotion={scan.get('scan_emotion_naming_count', 0)}"
        )

    if pipeline_result.get("halt"):
        summary_path = FINAL_DIR / f"SUMMARY_{batch_stub}.txt"
        save_text(summary_path, "\n".join(lines))
        return summary_path

    lines.append("")
    lines.append("=" * 60)
    lines.append("Q2 — PIPELINE RANKING (scanner ascending)")
    lines.append("=" * 60)
    lines.append("Order: scanner-violation count → word-count distance → literary rank")
    lines.append("")
    top1_id = pipeline_result.get("top1_run_id", "")
    lit_winner = pipeline_result.get("literary_winner_run_id", "")
    for rank_pos, run_id in enumerate(pipeline_result["pipeline_ranking"], 1):
        scan = scan_by_run_id.get(run_id, {})
        violations = (
            scan.get("scan_the_way_count", 0)
            + scan.get("scan_periphrastic_count", 0)
            + scan.get("scan_not_but_count", 0)
            + scan.get("scan_emotion_naming_count", 0)
            + max(0, scan.get("scan_em_dash_count", 0) - 12)
        )
        marker = ""
        if run_id == top1_id:
            marker += " [TOP 1 — shipping base]"
        if run_id == lit_winner and run_id != top1_id:
            marker += " [literary winner — overridden]"
        elif run_id == lit_winner:
            marker += " [also literary winner]"
        lines.append(f"  {rank_pos}. {run_id} (violations={violations}){marker}")

    if lit_winner and lit_winner != top1_id:
        lines.append("")
        lines.append(
            f"Note: evaluator's advisory winner ({lit_winner}) was overridden by "
            f"the scanner-ranked TOP 1 ({top1_id}). Literary rank contributes only "
            f"to tiebreaks in Q2."
        )

    lines.append("")
    lines.append("=" * 60)
    lines.append("Q3 — SENTENCE-GRAFT PASS")
    lines.append("=" * 60)
    lg = pipeline_result.get("line_graft") or {}
    if not lg:
        lines.append("Not run (fewer than 2 acceptable drafts).")
    elif lg.get("grafted"):
        attempted = len(lg.get("grafts_attempted", []))
        applied = len(lg.get("grafts", []))
        rejected_dirty = len(lg.get("grafts_rejected_dirty_donor", []))
        rejected_no_match = len(lg.get("grafts_rejected_no_match", []))
        lines.append(
            f"Grafts applied: {applied} of {attempted} identified "
            f"({rejected_dirty} rejected — donor carried hard-cap pattern; "
            f"{rejected_no_match} rejected — REPLACE text not in TOP 1)"
        )
        for i, g in enumerate(lg["grafts"], 1):
            lines.append(f"  Graft {i} (from Draft {g['source_draft']}):")
            lines.append(f"    Replaced: {g['replace'][:120]}{'...' if len(g['replace']) > 120 else ''}")
            lines.append(f"    With:     {g['with_text'][:120]}{'...' if len(g['with_text']) > 120 else ''}")
            lines.append(f"    Reason:   {g['reason']}")
        if rejected_dirty:
            lines.append("")
            lines.append("  Rejected (donor carried hard-cap pattern):")
            for g in lg["grafts_rejected_dirty_donor"]:
                lines.append(f"    Draft {g['source_draft']}: {g['with_text'][:100]}{'...' if len(g['with_text']) > 100 else ''}")

        if lg.get("grafted_scan"):
            gs = lg["grafted_scan"]
            lines.append("")
            lines.append("Diagnostic scan of TOP1_GRAFTED (informational — not a gate):")
            lines.append(f"  {format_scan_summary(gs)}")
            lines.append(
                f"  the-way={gs['scan_the_way_count']}, "
                f"periphrastic={gs['scan_periphrastic_count']}, "
                f"not-but={gs['scan_not_but_count']}, "
                f"em-dash={gs['scan_em_dash_count']} "
                f"({gs['scan_em_dash_per_1k']}/1k), "
                f"emotion={gs['scan_emotion_naming_count']}"
            )
    elif lg.get("grafts_attempted"):
        attempted = len(lg["grafts_attempted"])
        rejected_dirty = len(lg.get("grafts_rejected_dirty_donor", []))
        rejected_no_match = len(lg.get("grafts_rejected_no_match", []))
        lines.append(f"Grafts identified: {attempted}, but none were applied.")
        if rejected_dirty:
            lines.append(f"  {rejected_dirty} rejected — donor carried a hard-cap pattern.")
        if rejected_no_match:
            lines.append(f"  {rejected_no_match} rejected — REPLACE text did not match TOP 1 verbatim.")
    else:
        lines.append("No runner-up sentence met all three graft conditions.")

    lines.append("")
    lines.append("=" * 60)
    lines.append("FINAL DELIVERABLE")
    lines.append("=" * 60)
    source = pipeline_result.get("final_source", "")
    if source == "top1_grafted":
        lines.append("TOP 1 with sentence-level grafts applied.")
    elif source == "top1_ungrafted":
        lines.append("TOP 1 unchanged — no grafts qualified.")
    else:
        lines.append("(source unknown)")

    lines.append("")
    lines.append("=" * 60)
    lines.append("FILES")
    lines.append("=" * 60)
    lines.append(f"Final deliverable: {pipeline_result['final_path']}")
    lines.append("Top-N drafts:")
    for p in pipeline_result["top_paths"]:
        lines.append(f"  {p}")
    if lg.get("grafted_path"):
        lines.append(f"Grafted winner: {lg['grafted_path']}")

    summary_path = FINAL_DIR / f"SUMMARY_{batch_stub}.txt"
    save_text(summary_path, "\n".join(lines))
    return summary_path


# ============================================================================
# Export
# ============================================================================

def export_zip(df: pd.DataFrame, file_paths: list) -> bytes:
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        zf.writestr("runs.csv", csv_buf.getvalue())
        for p in file_paths:
            if p.exists():
                zf.write(p, p.name)
    return buf.getvalue()


def gather_output_paths(df: pd.DataFrame) -> list:
    paths = []
    for col in ["output_file", "payload_file", "meta_file"]:
        if col in df.columns:
            for val in df[col].dropna():
                p = Path(str(val))
                if p.exists():
                    paths.append(p)
    # Also include final deliverables
    if FINAL_DIR.exists():
        for p in FINAL_DIR.iterdir():
            if p.is_file():
                paths.append(p)
    return paths


# ============================================================================
# GitHub sync
# ============================================================================

GITHUB_API_BASE = "https://api.github.com"
GITHUB_SYNC_STATUS_KEY = "github_sync_status"
GITHUB_PULLED_KEY = "github_pulled_this_session"


def load_github_config() -> dict:
    token = ""
    repo = ""
    branch = ""
    source = ""

    try:
        if "GITHUB_TOKEN" in st.secrets:
            token = str(st.secrets.get("GITHUB_TOKEN", "")).strip()
            repo = str(st.secrets.get("GITHUB_REPO", "")).strip()
            branch = str(st.secrets.get("GITHUB_BRANCH", "") or "main").strip()
            if token and repo:
                source = "Streamlit secrets"
    except Exception:
        token = ""
        repo = ""

    if not (token and repo):
        env_token = os.environ.get("GITHUB_TOKEN", "").strip()
        env_repo = os.environ.get("GITHUB_REPO", "").strip()
        env_branch = os.environ.get("GITHUB_BRANCH", "main").strip() or "main"
        if env_token and env_repo:
            token = env_token
            repo = env_repo
            branch = env_branch
            source = "environment variable"

    configured = bool(token and repo)
    return {
        "token": token,
        "repo": repo,
        "branch": branch or "main",
        "configured": configured,
        "source": source,
    }


def _gh_headers(token: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }


def _gh_record_status(message: str, kind: str = "info") -> None:
    st.session_state[GITHUB_SYNC_STATUS_KEY] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "message": message,
        "kind": kind,
    }


def _local_path_for_repo_path(repo_path: str) -> Path:
    return RUNS_DIR / repo_path


def _repo_path_for_local(local_path: Path) -> Optional[str]:
    try:
        rel = local_path.resolve().relative_to(RUNS_DIR.resolve())
    except Exception:
        return None
    return rel.as_posix()


def github_list_tree(cfg: dict) -> List[dict]:
    if not cfg.get("configured"):
        return []
    repo = cfg["repo"]
    branch = cfg["branch"]
    try:
        branch_resp = requests.get(
            f"{GITHUB_API_BASE}/repos/{repo}/branches/{branch}",
            headers=_gh_headers(cfg["token"]),
            timeout=15,
        )
    except requests.RequestException as exc:
        _gh_record_status(f"GitHub list failed: {exc}", kind="error")
        return []
    if branch_resp.status_code == 404:
        return []
    if not branch_resp.ok:
        _gh_record_status(f"GitHub list failed: {branch_resp.status_code}", kind="error")
        return []
    tree_sha = (
        branch_resp.json().get("commit", {}).get("commit", {}).get("tree", {}).get("sha")
    )
    if not tree_sha:
        return []
    try:
        tree_resp = requests.get(
            f"{GITHUB_API_BASE}/repos/{repo}/git/trees/{tree_sha}",
            params={"recursive": "1"},
            headers=_gh_headers(cfg["token"]),
            timeout=30,
        )
    except requests.RequestException as exc:
        _gh_record_status(f"GitHub tree read failed: {exc}", kind="error")
        return []
    if not tree_resp.ok:
        return []
    entries = tree_resp.json().get("tree", []) or []
    return [
        {"path": entry["path"], "sha": entry["sha"]}
        for entry in entries
        if entry.get("type") == "blob" and entry.get("path")
    ]


def github_get_file_bytes(cfg: dict, path: str) -> Optional[bytes]:
    if not cfg.get("configured"):
        return None
    try:
        resp = requests.get(
            f"{GITHUB_API_BASE}/repos/{cfg['repo']}/contents/{path}",
            params={"ref": cfg["branch"]},
            headers=_gh_headers(cfg["token"]),
            timeout=30,
        )
    except requests.RequestException:
        return None
    if not resp.ok:
        return None
    body = resp.json()
    if body.get("encoding") == "base64" and "content" in body:
        try:
            return base64.b64decode(body["content"])
        except Exception:
            return None
    download_url = body.get("download_url")
    if download_url:
        try:
            dl = requests.get(download_url, timeout=60)
            if dl.ok:
                return dl.content
        except requests.RequestException:
            return None
    return None


def github_get_file_sha(cfg: dict, path: str) -> Optional[str]:
    if not cfg.get("configured"):
        return None
    try:
        resp = requests.get(
            f"{GITHUB_API_BASE}/repos/{cfg['repo']}/contents/{path}",
            params={"ref": cfg["branch"]},
            headers=_gh_headers(cfg["token"]),
            timeout=15,
        )
    except requests.RequestException:
        return None
    if not resp.ok:
        return None
    return resp.json().get("sha")


def github_put_file(cfg: dict, path: str, data: bytes, message: str) -> bool:
    if not cfg.get("configured"):
        return False
    existing_sha = github_get_file_sha(cfg, path)
    payload = {
        "message": message,
        "content": base64.b64encode(data).decode("ascii"),
        "branch": cfg["branch"],
    }
    if existing_sha:
        payload["sha"] = existing_sha
    try:
        resp = requests.put(
            f"{GITHUB_API_BASE}/repos/{cfg['repo']}/contents/{path}",
            headers=_gh_headers(cfg["token"]),
            json=payload,
            timeout=30,
        )
    except requests.RequestException as exc:
        _gh_record_status(f"GitHub push failed for {path}: {exc}", kind="error")
        return False
    if not resp.ok:
        _gh_record_status(
            f"GitHub push failed for {path}: {resp.status_code} {resp.text[:200]}",
            kind="error",
        )
        return False
    return True


def github_pull_all(cfg: dict) -> dict:
    result = {"pulled": 0, "skipped": 0, "failed": 0}
    if not cfg.get("configured"):
        return result
    tree = github_list_tree(cfg)
    if not tree:
        _gh_record_status("Pull: no files in repo (or repo is empty).", kind="info")
        return result
    for entry in tree:
        repo_path = entry["path"]
        local_path = _local_path_for_repo_path(repo_path)
        data = github_get_file_bytes(cfg, repo_path)
        if data is None:
            result["failed"] += 1
            continue
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            result["pulled"] += 1
        except Exception:
            result["failed"] += 1
    _gh_record_status(
        f"Pulled {result['pulled']} file(s) from {cfg['repo']}@{cfg['branch']}.",
        kind="success" if result["failed"] == 0 else "warn",
    )
    return result


def github_push_paths(cfg: dict, local_paths: List[Path], commit_prefix: str) -> dict:
    result = {"pushed": 0, "failed": 0}
    if not cfg.get("configured"):
        return result
    for local_path in local_paths:
        if not local_path.exists() or not local_path.is_file():
            continue
        repo_path = _repo_path_for_local(local_path)
        if repo_path is None:
            continue
        try:
            data = local_path.read_bytes()
        except Exception:
            result["failed"] += 1
            continue
        commit_msg = f"{commit_prefix}: {repo_path}"
        ok = github_put_file(cfg, repo_path, data, commit_msg)
        if ok:
            result["pushed"] += 1
        else:
            result["failed"] += 1
    if result["failed"] == 0 and result["pushed"]:
        _gh_record_status(
            f"Pushed {result['pushed']} file(s) to {cfg['repo']}",
            kind="success",
        )
    return result


def github_push_after_generation(
    cfg: dict, csv_path: Path, output_path: Path, payload_path: Path, meta_path: Path,
) -> None:
    if not cfg.get("configured"):
        return
    github_push_paths(
        cfg, [csv_path, output_path, payload_path, meta_path],
        commit_prefix="generation",
    )


def github_push_after_pipeline(
    cfg: dict, csv_path: Path, pipeline_files: List[Path],
) -> None:
    if not cfg.get("configured"):
        return
    github_push_paths(cfg, [csv_path] + pipeline_files, commit_prefix="pipeline")


def github_pull_on_startup_if_needed(cfg: dict, csv_path: Path) -> None:
    if not cfg.get("configured"):
        return
    if st.session_state.get(GITHUB_PULLED_KEY):
        return
    local_empty = (not csv_path.exists()) or csv_path.stat().st_size == 0
    if local_empty:
        github_pull_all(cfg)
    st.session_state[GITHUB_PULLED_KEY] = True


# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(page_title="Micro-Prompt Harness", layout="wide")
st.title("Micro-Prompt Harness")
st.caption("Generate · Scan · Evaluate · Rank · Ship")

ensure_dirs()
csv_path = RUNS_DIR / CSV_FILENAME

github_cfg = load_github_config()
github_pull_on_startup_if_needed(github_cfg, csv_path)

auto_key, auto_key_source = load_api_key()

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    if auto_key:
        api_key = auto_key
        st.success(f"API key loaded from {auto_key_source}")
    else:
        manual_key = st.text_input("Anthropic API Key", type="password")
        api_key = clean_api_key(manual_key) if manual_key else ""
        if not api_key:
            st.warning("Set ANTHROPIC_API_KEY in Streamlit secrets or enter above.")

    st.markdown("---")

    gen_model = st.text_input("Generation model", value=DEFAULT_GEN_MODEL)
    eval_model = st.text_input("Evaluation model", value=DEFAULT_EVAL_MODEL)

    st.markdown("---")

    temps_input = st.text_input("Temperatures (comma-separated)", value="1.0")
    try:
        temperatures = [float(t.strip()) for t in temps_input.split(",") if t.strip()]
    except ValueError:
        temperatures = [0.7]
        st.warning("Could not parse temperatures. Using 0.7.")

    repetitions = st.number_input("Repetitions per prompt×temp", min_value=1, max_value=10, value=3)

    top_n = st.number_input(
        "Top-N drafts to export for external testing",
        min_value=1, max_value=10, value=3,
        help="After the pipeline runs, the top-N drafts from the literary ranking are saved as separate files so you can run them through external detectors. The sentence-graft pass also draws its donor pool from this top-N.",
    )

    st.markdown("---")

    st.subheader("Documents")
    st.caption("Upload the files the prompt references.")

    doc_uploads = {}
    outline_file = st.file_uploader("Outline", type=["txt", "docx"], key="outline")
    if outline_file:
        doc_uploads["Outline"] = extract_text_from_upload(outline_file)

    source_file = st.file_uploader("Source text (voice model)", type=["txt", "docx"], key="source")
    if source_file:
        doc_uploads["Source Text"] = extract_text_from_upload(source_file)

    profiles_file = st.file_uploader("Character profiles", type=["txt", "docx"], key="profiles")
    if profiles_file:
        doc_uploads["Character Profiles"] = extract_text_from_upload(profiles_file)

    st.markdown("---")

    st.subheader("GitHub sync")
    if github_cfg["configured"]:
        st.success(f"Repo: `{github_cfg['repo']}` ({github_cfg['source']})")
        sync_status = st.session_state.get(GITHUB_SYNC_STATUS_KEY)
        if sync_status:
            st.caption(f"{sync_status['timestamp']}: {sync_status['message']}")
        if st.button("Sync now (pull)"):
            github_pull_all(github_cfg)
            st.rerun()
    else:
        st.info("Set GITHUB_TOKEN and GITHUB_REPO in secrets to enable sync.")

    st.markdown("---")
    st.caption(f"Gen: `{gen_model}` · Eval: `{eval_model}`")
    st.caption(f"Temps: {temperatures} · Reps: {repetitions} · Top-N: {top_n}")
    if doc_uploads:
        st.caption(f"Docs: {', '.join(doc_uploads.keys())}")

prompts_df = load_prompts()

if prompts_df.empty:
    st.warning(
        f"No `{PROMPTS_CSV}` found or it has no rows. "
        f"Create a CSV with columns `id` and `text` (and optionally `category`)."
    )
    st.stop()

left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Prompt")

    # Pre-select prompt 62 only
    target_pid = 62
    target_row = prompts_df[prompts_df["id"].astype(int) == target_pid]
    if target_row.empty:
        st.error(f"Prompt {target_pid} not found in {PROMPTS_CSV}.")
        st.stop()
    selected_ids = [target_pid]

    with st.expander(f"P{target_pid} — {str(target_row.iloc[0].get('category', ''))}"):
        st.text(str(target_row.iloc[0]["text"]))

    total_runs = len(temperatures) * repetitions
    st.write(
        f"Prompt **P{target_pid}** × **{len(temperatures)}** temps × "
        f"**{repetitions}** reps = **{total_runs}** drafts"
    )

    if st.button("Generate & Evaluate", type="primary", disabled=not api_key or total_runs == 0):
        problems = []
        if not api_key:
            problems.append("No API key set.")
        if not temperatures:
            problems.append("No temperatures set.")
        if not doc_uploads:
            problems.append("No documents uploaded. The model needs at least the Outline and Source text.")
        else:
            if "Outline" not in doc_uploads:
                problems.append("Outline not uploaded. The prompt references it.")
            if "Source Text" not in doc_uploads:
                problems.append("Source text not uploaded. The prompt references it.")

        prompt_row = prompts_df[prompts_df["id"].astype(int) == target_pid].iloc[0]
        txt = str(prompt_row["text"]).strip()
        if not txt or txt == "nan":
            problems.append(f"P{target_pid} has no prompt text (empty or NaN in prompts.csv).")

        if problems:
            st.error("**Cannot generate — fix these first:**")
            for p in problems:
                st.warning(p)
        else:
            client = anthropic.Anthropic(api_key=api_key)
            progress = st.progress(0.0)
            status = st.empty()
            run_count = 0
            batch_run_ids_ordered = []
            batch_drafts = []
            batch_scans = {}

            prompt_text = str(prompt_row["text"])

            for temp in temperatures:
                for rep in range(1, repetitions + 1):
                    run_count += 1
                    status.info(f"Generating {run_count}/{total_runs}: P{target_pid} T{temp} R{rep:02d}")

                    payload_text = build_payload_text(prompt_text, doc_uploads)
                    message_blocks = build_message_blocks(prompt_text, doc_uploads)
                    stub = make_file_stub(target_pid, temp, gen_model)
                    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]

                    payload_path = OUTPUTS_DIR / f"{stub}_payload.txt"
                    save_text(payload_path, payload_text)

                    try:
                        output = generate_chapter(client, gen_model, temp, message_blocks)
                    except Exception as e:
                        st.error(f"Generation failed for P{target_pid} T{temp} R{rep}: {e}")
                        continue

                    output_path = OUTPUTS_DIR / f"{stub}_output.txt"
                    save_text(output_path, output)

                    # Run mechanical scanner immediately
                    scan_result = scan_draft(output)
                    batch_drafts.append({"run_id": run_id, "text": output})
                    batch_scans[run_id] = scan_result

                    meta = {
                        "run_id": run_id,
                        "prompt_id": target_pid,
                        "temperature": temp,
                        "model": gen_model,
                        "repetition": rep,
                        "timestamp": datetime.now().isoformat(),
                        "documents": list(doc_uploads.keys()),
                        "scan": scan_result,
                    }
                    meta_path = OUTPUTS_DIR / f"{stub}_meta.json"
                    save_text(meta_path, json.dumps(meta, indent=2))

                    record = RunRecord(
                        run_id=run_id,
                        timestamp=datetime.now().isoformat(),
                        prompt_id=target_pid,
                        prompt_text=prompt_text[:200],
                        temperature=temp,
                        model=gen_model,
                        output_file=str(output_path),
                        payload_file=str(payload_path),
                        meta_file=str(meta_path),
                        word_count=len(output.split()),
                        **{k: v for k, v in scan_result.items() if k in RUN_FIELDS},
                    )
                    append_record(csv_path, record)
                    batch_run_ids_ordered.append(run_id)

                    if github_cfg["configured"]:
                        try:
                            github_push_after_generation(
                                github_cfg, csv_path, output_path, payload_path, meta_path,
                            )
                        except Exception as push_exc:
                            st.warning(f"GitHub push failed: {push_exc}")

                    progress.progress(run_count / total_runs)
                    time.sleep(0.3)

            progress.empty()
            status.success(f"Generated {run_count} drafts. Running evaluation pipeline...")

            # --- Auto-run pipeline on this batch ---
            if len(batch_drafts) >= 2:
                outline_text = doc_uploads.get("Outline", "")
                temps_used = sorted(set(temperatures))
                batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_stub = make_batch_stub(batch_timestamp)

                # Record this batch's identifiers up front so the fallback
                # download path can scope to it even if the pipeline raises
                # before any FINAL file is written. Clear stale results/errors
                # from a previous batch at the same time.
                st.session_state["last_batch_stub"] = batch_stub
                st.session_state["last_batch_run_ids"] = batch_run_ids_ordered
                st.session_state["last_batch_size"] = total_runs
                st.session_state.pop("last_pipeline_result", None)
                st.session_state.pop("last_pipeline_summary_path", None)
                st.session_state.pop("last_pipeline_error", None)

                try:
                    result = run_pipeline(
                        client=client,
                        eval_model=eval_model,
                        drafts=batch_drafts,
                        scan_by_run_id=batch_scans,
                        outline_text=outline_text,
                        top_n=int(top_n),
                        batch_stub=batch_stub,
                    )

                    summary_path = write_batch_summary(
                        pipeline_result=result,
                        drafts=batch_drafts,
                        scan_by_run_id=batch_scans,
                        batch_stub=batch_stub,
                        temperatures=temps_used,
                        prompts_used=[target_pid],
                    )

                    # Update CSV with quality verdicts, ranking, and roles
                    evaluation_id = f"eval_{batch_timestamp}"
                    update_records_bulk(csv_path, batch_run_ids_ordered, {
                        "is_winner": False,
                        "evaluation_id": evaluation_id,
                        "evaluator_model": eval_model,
                        "evaluation_raw": result["eval_raw"][:8000],
                        "pipeline_role": "",
                        "quality_verdict": "",
                        "quality_reason": "",
                    })

                    # Write quality verdicts for every draft
                    quality_by_run = result.get("quality_by_run_id", {})
                    for run_id, q in quality_by_run.items():
                        update_record(csv_path, run_id, {
                            "quality_verdict": q.get("verdict", ""),
                            "quality_reason": (q.get("reason", ""))[:500],
                        })

                    # Mark dropped drafts with their pipeline role
                    for run_id in result.get("dropped_run_ids", []):
                        update_record(csv_path, run_id, {
                            "pipeline_role": "dropped_unacceptable",
                        })

                    if not result.get("halt"):
                        # Advisory literary rank — stored for auditing
                        for rank_pos, run_id in enumerate(result["literary_ranking"], 1):
                            update_record(csv_path, run_id, {"evaluation_rank": rank_pos})

                        # Pipeline TOP 1 is the scanner-ranked winner, not the evaluator's
                        top1_id = result.get("top1_run_id", "")
                        if top1_id:
                            update_record(csv_path, top1_id, {
                                "is_winner": True,
                                "pipeline_role": "top1_winner",
                            })

                        # Tag any donor whose line was transplanted into TOP 1.
                        # source_draft is a 1-indexed position into the pipeline_ranking
                        # (the order drafts were presented to the graft evaluator).
                        lg = result.get("line_graft") or {}
                        if lg.get("grafted"):
                            donor_positions = {g["source_draft"] for g in lg.get("grafts", [])}
                            ranking_ids = result.get("pipeline_ranking", [])
                            for pos in donor_positions:
                                if 1 <= pos <= len(ranking_ids):
                                    donor_run_id = ranking_ids[pos - 1]
                                    if donor_run_id != top1_id:
                                        update_record(csv_path, donor_run_id, {
                                            "pipeline_role": "graft_donor",
                                        })

                    # Store results for download section
                    st.session_state["last_pipeline_result"] = result
                    st.session_state["last_pipeline_summary_path"] = str(summary_path)
                    st.session_state["last_batch_stub"] = batch_stub
                    st.session_state["last_batch_run_ids"] = batch_run_ids_ordered
                    st.session_state["last_batch_size"] = total_runs

                    # Push to GitHub
                    if github_cfg["configured"]:
                        files_to_push = list(result.get("top_paths", [])) + [summary_path]
                        if result.get("final_path"):
                            files_to_push.insert(0, Path(result["final_path"]))
                        lg = result.get("line_graft", {})
                        if lg.get("grafted_path"):
                            files_to_push.append(Path(lg["grafted_path"]))
                        try:
                            github_push_after_pipeline(
                                github_cfg, csv_path, files_to_push,
                            )
                        except Exception as push_exc:
                            st.warning(f"GitHub push failed: {push_exc}")

                except Exception as e:
                    import traceback
                    # Persist the error across the st.rerun() below — otherwise
                    # st.error / st.code output is wiped from the screen on
                    # rerender and the failure is silent.
                    st.session_state["last_pipeline_error"] = {
                        "stub": batch_stub,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    }
            else:
                st.warning("Need at least 2 drafts for the pipeline. Some generations may have failed.")

            st.rerun()


with right_col:
    st.subheader("Run log")

    df = load_csv(csv_path)
    if df.empty:
        st.info("No runs yet. Generate some drafts.")
    else:
        display_cols = [
            "run_id", "prompt_id", "temperature", "word_count",
            "quality_verdict", "scan_hard_cap_pass",
            "scan_the_way_count", "scan_em_dash_count",
            "is_winner", "pipeline_role", "evaluation_rank",
        ]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True)

        # --- Show pipeline results if available ---
        result = st.session_state.get("last_pipeline_result")
        if result:
            st.markdown("---")
            st.subheader("Pipeline result")

            # --- Q1 halt case ---
            if result.get("halt"):
                st.error(
                    f"PIPELINE HALTED: {result.get('halt_reason', 'no acceptable draft')}"
                )
                dropped = result.get("dropped_run_ids", [])
                if dropped:
                    with st.expander(f"Drafts dropped as UNACCEPTABLE ({len(dropped)})"):
                        quality = result.get("quality_by_run_id", {})
                        for run_id in dropped:
                            q = quality.get(run_id, {})
                            st.markdown(f"**{run_id}**")
                            st.caption(q.get("reason", "(no reason)"))
                with st.expander("Literary evaluator reasoning"):
                    st.text(result["eval_raw"])
            else:
                top1_id = result.get("top1_run_id", "?")
                lit_winner = result.get("literary_winner_run_id", "")
                final_source = result.get("final_source", "")
                lg = result.get("line_graft") or {}

                # --- Headline status ---
                if final_source == "top1_grafted":
                    st.success(
                        f"TOP 1 GRAFTED: {len(lg.get('grafts', []))} line(s) "
                        f"transplanted from runners-up into `{top1_id}`. "
                        f"Shipped as FINAL."
                    )
                elif final_source == "top1_ungrafted":
                    if lg.get("grafts_attempted"):
                        rejected_dirty = len(lg.get("grafts_rejected_dirty_donor", []))
                        rejected_no_match = len(lg.get("grafts_rejected_no_match", []))
                        reasons = []
                        if rejected_dirty:
                            reasons.append(f"{rejected_dirty} donor sentence(s) carried hard-cap patterns")
                        if rejected_no_match:
                            reasons.append(f"{rejected_no_match} REPLACE text(s) did not match TOP 1")
                        reason_text = "; ".join(reasons) if reasons else "no grafts applied"
                        st.info(
                            f"TOP 1 SHIPPED UNCHANGED: `{top1_id}`. "
                            f"Graft candidates identified but not applied ({reason_text})."
                        )
                    else:
                        st.info(
                            f"TOP 1 SHIPPED UNCHANGED: `{top1_id}`. "
                            f"No runner-up sentence met the graft conditions."
                        )
                else:
                    st.info(f"Pipeline complete. TOP 1: `{top1_id}`.")

                # --- Q1 quality floor summary ---
                dropped = result.get("dropped_run_ids", [])
                acceptable = result.get("acceptable_run_ids", [])
                st.caption(
                    f"Q1 quality floor: {len(acceptable)}/{len(acceptable) + len(dropped)} acceptable"
                    + (f" — {len(dropped)} dropped as UNACCEPTABLE" if dropped else "")
                )
                if dropped:
                    with st.expander(f"Drafts dropped as UNACCEPTABLE ({len(dropped)})"):
                        quality = result.get("quality_by_run_id", {})
                        for run_id in dropped:
                            q = quality.get(run_id, {})
                            st.markdown(f"**{run_id}**")
                            st.caption(q.get("reason", "(no reason)"))

                # --- Q2 scanner-ranked TOP 1 ---
                if lit_winner and lit_winner != top1_id:
                    st.caption(
                        f"Q2 scanner-ranked TOP 1 `{top1_id}` overrides "
                        f"evaluator's advisory winner `{lit_winner}` (fewer hard-cap tells)."
                    )

                with st.expander("Literary evaluator reasoning"):
                    st.text(result["eval_raw"])

                # --- Q3 sentence-graft pass details ---
                if lg:
                    if lg.get("grafted"):
                        with st.expander("Graft details"):
                            for i, g in enumerate(lg["grafts"], 1):
                                st.markdown(f"**Graft {i}** (from Draft {g['source_draft']})")
                                st.text(f"  Replaced: {g['replace']}")
                                st.text(f"  With:     {g['with_text']}")
                                st.caption(f"  Reason: {g['reason']}")

                            rejected_dirty = lg.get("grafts_rejected_dirty_donor", [])
                            if rejected_dirty:
                                st.markdown("**Rejected — donor carried hard-cap pattern:**")
                                for g in rejected_dirty:
                                    st.text(f"  Draft {g['source_draft']}: {g['with_text']}")

                        rejected_no_match = lg.get("grafts_rejected_no_match", [])
                        if rejected_no_match:
                            st.markdown("**Rejected — REPLACE text not found in TOP 1:**")
                            for g in rejected_no_match:
                                st.text(f"  Draft {g['source_draft']}: {g['replace']}")

                        gs = lg.get("grafted_scan")
                        if gs:
                            st.caption(
                                f"Diagnostic scan of TOP1_GRAFTED (informational, not a gate): "
                                f"{format_scan_summary(gs)}"
                            )

                if lg.get("raw"):
                    with st.expander("Sentence-graft evaluator reasoning"):
                        st.text(lg["raw"])

                # --- Originality re-rank: override TOP 1 with color-based ranker ---
                st.markdown("---")
                st.subheader("Re-rank with Originality reports")
                st.caption(
                    "Export the TOP-N drafts to Originality, download the color-coded "
                    "docx reports, and upload them here. The ranker classifies each "
                    "highlighted run by its green–orange offset and scores drafts by "
                    "strong-orange cluster shape. Matches reports to drafts automatically "
                    "by text overlap — filenames are ignored."
                )

                orig_uploads = st.file_uploader(
                    "Originality .docx exports (multi-select)",
                    type=["docx"],
                    accept_multiple_files=True,
                    key=f"orig_uploads_{st.session_state.get('last_batch_stub', 'nobatch')}",
                )

                if orig_uploads and st.button(
                    "Compute Originality ranking",
                    key=f"orig_rerank_btn_{st.session_state.get('last_batch_stub', 'nobatch')}",
                ):
                    # Build candidate drafts from disk for matching
                    batch_run_ids = st.session_state.get("last_batch_run_ids", [])
                    df_now = load_csv(csv_path)
                    candidate_drafts = []
                    for run_id in batch_run_ids:
                        row = df_now[df_now["run_id"].astype(str) == str(run_id)]
                        if row.empty:
                            continue
                        of_path = Path(str(row.iloc[0].get("output_file", "")))
                        if not of_path.exists():
                            continue
                        try:
                            draft_text = of_path.read_text(encoding="utf-8", errors="replace")
                        except Exception:
                            continue
                        candidate_drafts.append({
                            "run_id": str(run_id),
                            "text": draft_text,
                        })

                    if not candidate_drafts:
                        st.error("Could not load any batch drafts from disk for matching.")
                    else:
                        reports = {}
                        unmatched = []
                        for uf in orig_uploads:
                            try:
                                docx_bytes = uf.read()
                            except Exception as e:
                                unmatched.append((uf.name, f"read error: {e}"))
                                continue
                            run_id = match_originality_docx_to_draft(
                                docx_bytes, candidate_drafts
                            )
                            if not run_id:
                                unmatched.append((uf.name, "no matching draft in batch"))
                                continue
                            try:
                                metrics = compute_originality_metrics(docx_bytes)
                            except Exception as e:
                                unmatched.append((uf.name, f"metrics error: {e}"))
                                continue
                            # Key on filename so raw and grafted variants of the
                            # same run_id produce separate rows.
                            reports[uf.name] = {
                                "run_id": run_id,
                                "metrics": metrics,
                                "filename": uf.name,
                            }

                        if not reports:
                            st.error(
                                "No uploaded reports could be matched to drafts in this "
                                "batch. Check that you are uploading the Originality "
                                "exports for the drafts just generated."
                            )
                        else:
                            # Verify the uploaded files actually contain color data.
                            # If every report has zero highlighted cells, the user
                            # almost certainly uploaded the plain draft exports
                            # instead of Originality-processed color reports.
                            total_cells = sum(
                                v["metrics"].get("total_runs", 0)
                                for v in reports.values()
                            )
                            if total_cells == 0:
                                st.error(
                                    "Uploaded files contain no color-coded cells. "
                                    "These look like plain drafts, not Originality "
                                    "reports. The correct files are the color-"
                                    "highlighted .docx exports you download from "
                                    "Originality.ai after scanning each draft — "
                                    "not the TOP-N files this app produces. "
                                    "Submit the TOP-N drafts to Originality, "
                                    "download the color reports, and upload those here."
                                )
                            else:
                                # Build ranking rows per-uploaded-file (not per-run_id),
                                # so raw vs grafted variants of the same run_id appear
                                # as separate rows. Sort by rank_score descending.
                                ranking = []
                                for fname, report in reports.items():
                                    ranking.append({
                                        "filename": fname,
                                        "run_id": report["run_id"],
                                        "rank_score": report["metrics"]["rank_score"],
                                        "metrics": report["metrics"],
                                    })
                                ranking.sort(
                                    key=lambda r: r["rank_score"], reverse=True,
                                )
                                for i, row in enumerate(ranking, 1):
                                    row["rank"] = i

                                # Store on session state so subsequent UI can read it
                                st.session_state["originality_ranking"] = ranking
                                st.session_state["originality_reports"] = reports
                                st.session_state["originality_unmatched"] = unmatched

                # --- Display Originality ranking if computed ---
                orig_ranking = st.session_state.get("originality_ranking")
                orig_reports = st.session_state.get("originality_reports", {})
                orig_unmatched = st.session_state.get("originality_unmatched", [])
                if orig_ranking:
                    scanner_top1 = result.get("top1_run_id", "")
                    orig_top1 = orig_ranking[0]["run_id"]

                    if scanner_top1 and orig_top1 != scanner_top1:
                        st.warning(
                            f"Originality TOP 1 (`{orig_top1}`) differs from scanner "
                            f"TOP 1 (`{scanner_top1}`). The scanner-ranked draft was "
                            f"the one graft-processed and shipped as FINAL. The "
                            f"Originality-ranked draft is the one you should actually "
                            f"submit."
                        )
                    elif scanner_top1:
                        st.success(
                            f"Originality TOP 1 (`{orig_top1}`) matches scanner TOP 1. "
                            f"Ship the current FINAL."
                        )

                    rank_rows = []
                    for row in orig_ranking:
                        m = row["metrics"]
                        fn = row.get("filename", "")
                        rank_rows.append({
                            "rank": row["rank"],
                            "run_id": row["run_id"],
                            "filename": fn,
                            "rank_score": row["rank_score"],
                            "longest_O": m["longest_strong_O"],
                            "total_O": m["strong_orange"],
                            "in_clusters": m["strong_O_in_clusters"],
                            "mild_G": m["mild_green"],
                            "mild_O": m["mild_orange"],
                            "is_scanner_top1": (row["run_id"] == scanner_top1),
                        })
                    st.dataframe(
                        pd.DataFrame(rank_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                    if orig_unmatched:
                        with st.expander(
                            f"Unmatched uploads ({len(orig_unmatched)})"
                        ):
                            for name, reason in orig_unmatched:
                                st.text(f"  {name}: {reason}")

                    st.caption(
                        "Ranking formula: -(longest_O²)·3 - in_clusters - total_O·0.3 "
                        "+ (mild_G - mild_O)·0.5. Strong-green count is deliberately "
                        "excluded — it is non-monotonic with true score and correlates "
                        "with concentrated orange clusters."
                    )

                    if st.button(
                        "Clear Originality ranking",
                        key=f"orig_clear_{st.session_state.get('last_batch_stub', 'nobatch')}",
                    ):
                        st.session_state.pop("originality_ranking", None)
                        st.session_state.pop("originality_reports", None)
                        st.session_state.pop("originality_unmatched", None)
                        st.rerun()

            summary_path_str = st.session_state.get("last_pipeline_summary_path")
            if summary_path_str:
                summary_path = Path(summary_path_str)
                if summary_path.exists():
                    with st.expander("Batch summary"):
                        st.text(summary_path.read_text(encoding="utf-8"))

            # --- Downloads scoped to this batch ---
            st.markdown("---")
            batch_stub = st.session_state.get("last_batch_stub", "batch")

            # Collect batch files: TOP-N drafts + FINAL + LINE-GRAFT + summary
            batch_paths = []
            for p in result.get("top_paths", []):
                pp = Path(p) if not isinstance(p, Path) else p
                if pp.exists():
                    batch_paths.append(pp)
            # Include line-graft file if produced
            lg_path_str = (result.get("line_graft") or {}).get("grafted_path", "")
            if lg_path_str:
                lg_p = Path(lg_path_str)
                if lg_p.exists():
                    batch_paths.append(lg_p)
            final_p = Path(result["final_path"]) if result.get("final_path") else None
            if final_p and final_p.exists():
                batch_paths.append(final_p)
            if summary_path_str:
                sp = Path(summary_path_str)
                if sp.exists():
                    batch_paths.append(sp)

            if batch_paths:
                # Build ZIP of just this batch
                batch_run_ids = st.session_state.get("last_batch_run_ids", [])
                batch_df = df[df["run_id"].astype(str).isin([str(r) for r in batch_run_ids])]
                zip_bytes = export_zip(batch_df, batch_paths)
                st.download_button(
                    f"Download {batch_stub} (ZIP)",
                    data=zip_bytes,
                    file_name=f"{batch_stub}.zip",
                    mime="application/zip",
                )

                csv_buf = io.StringIO()
                batch_df.to_csv(csv_buf, index=False)
                st.download_button(
                    f"Download {batch_stub}.csv",
                    data=csv_buf.getvalue(),
                    file_name=f"{batch_stub}.csv",
                    mime="text/csv",
                )
        else:
            # No pipeline result in session — either the pipeline hasn't run
            # yet this session, or it raised. Surface any persisted error and
            # scope the fallback download to THIS batch only (never to
            # accumulated FINAL_DIR history).
            st.markdown("---")

            err = st.session_state.get("last_pipeline_error")
            if err:
                st.error(
                    f"Pipeline failed on batch `{err['stub']}`: {err['message']}"
                )
                with st.expander("Traceback"):
                    st.code(err["traceback"])

            current_batch_stub = st.session_state.get("last_batch_stub")
            current_batch_ids = st.session_state.get("last_batch_run_ids", [])

            # FINAL_DIR files from THIS batch only, matched by stub in filename.
            batch_paths = []
            if current_batch_stub and FINAL_DIR.exists():
                for p in sorted(FINAL_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
                    if p.is_file() and current_batch_stub in p.name:
                        batch_paths.append(p)

            # Output files from THIS batch only, matched by run_id — not
            # df.tail(N), which drifts when other runs arrive.
            if current_batch_ids:
                recent_df = df[df["run_id"].astype(str).isin(
                    [str(r) for r in current_batch_ids]
                )].copy()
            else:
                last_n = st.session_state.get("last_batch_size", 4)
                recent_df = df.tail(last_n).copy()
            output_paths = []
            for _, row in recent_df.iterrows():
                op = Path(str(row.get("output_file", "")))
                if op.exists():
                    output_paths.append(op)

            all_dl_paths = output_paths + batch_paths
            if all_dl_paths:
                batch_label = current_batch_stub or datetime.now().strftime("%Y%m%d")
                zip_bytes = export_zip(recent_df, all_dl_paths)
                st.download_button(
                    f"Download latest batch (ZIP)",
                    data=zip_bytes,
                    file_name=f"batch_{batch_label}.zip",
                    mime="application/zip",
                )

                csv_buf = io.StringIO()
                recent_df.to_csv(csv_buf, index=False)
                st.download_button(
                    f"Download latest batch CSV",
                    data=csv_buf.getvalue(),
                    file_name=f"batch_{batch_label}.csv",
                    mime="text/csv",
                )
            else:
                st.info(
                    "No downloadable batch in this session yet. Click "
                    "Generate & Evaluate to produce one."
                )