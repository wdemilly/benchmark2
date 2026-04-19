"""
Micro-Prompt Harness — Sentence-Level Graft Pipeline
=====================================================
Generate chapter drafts. Scan each mechanically for AI-signature hard caps.
Evaluate on literary quality. TOP 1 (the literary winner) is the baseline
deliverable and ships unchanged if no graft is appropriate.

Then a single sentence-harvest pass: the evaluator looks for sentences in
the runners-up (Drafts 2..N) that do the same narrative work as a counterpart
in TOP 1 and are genuinely sharper, more economical, or more in-voice. Any
qualifying line is transplanted into TOP 1 via deterministic find-and-replace,
producing TOP1_GRAFTED. If nothing qualifies, NO_GRAFT is returned and TOP 1
alone ships.

The scanner informs selection in two ways: (1) donor sentences that themselves
contain hard-cap violations are rejected as graft candidates, and (2) the
winner's flagged passages are surfaced to the evaluator so it knows where to
hunt for clean replacements.

There is no draft-level clean/dirty classification. There are no pipeline
branches. TOP 1 is always the base; donors are always the other N-1 drafts.

Export: top-N literary drafts as separate files + TOP1_GRAFTED (when any
grafts applied) + a batch summary.

The generation prompt lives in prompts.csv. The app does not inject its own
drafting instructions.
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

YOUR METHOD — in this order:

1. WORD COUNTS. Note each draft's word count against the outline's target range. Flag any that are short or over.

2. MECHANICAL COMPLIANCE. The scanner results are provided below. For each draft, note the violation counts. Do not re-scan — use the provided numbers. The key caps: "the way X" (zero tolerance), periphrastic "as though/as if" observational framing (zero tolerance), "not X but Y" in narration (zero tolerance), em dashes (hard cap from the outline), emotion naming in narration (zero tolerance). Note which drafts are cleanest and which are most violated.

3. PROSE QUALITY (primary ranking signal). Evaluate each draft on:
   - Voice fidelity: Does the prose match the sentence movement, paragraph cadence, and dialogue rhythm of the voice-source authors named in the outline? Is the POV character's interior voice present, sharp, and wry where the outline asks for it?
   - Dialogue craft: Do exchanges carry subtext and tension? Are weapons and resolution carriers executed as the outline specifies? Do turns run short enough for the genre?
   - Interior voice: Is the POV character observing, judging, and acting — or is the narration doing interpretive work the reader should do?
   - Specific passage strength: Name the best individual moments (a line of dialogue, an image, an interior thought) in each draft. Quote a few words for identification.

4. BEAT EXECUTION. Check each draft against the outline's microbeats. Which beats are fully realized? Which are compressed or missing? Does the structural rhyme land? Does the thematic line land in dialogue as specified? Is the entry contract (interior voice within the first 200 words) met?

5. RANKING. Prose quality is the primary signal. Mechanical compliance is the secondary signal — a draft with stronger prose and more violations ranks above a mechanically cleaner but flatter draft, because violations are repairable and voice is not. Name the specific thing that tips each decision.

6. GRAFT CANDIDATES. From the non-winning drafts, name 1-3 specific lines or passages worth transplanting into the winner. Quote a few words for identification and name the beat where each would land.

OUTPUT FORMAT

For each draft, write a paragraph (3-5 sentences) covering voice quality, best moment, and notable weaknesses. Reference the scanner numbers.

Then a comparison paragraph naming the top 2-3 contenders, what tips the decision, and which beats carry the winner's strength.

Then a graft paragraph naming specific lines from non-winners worth transplanting, with beat locations.

Then on a line by itself:

RANKING: N, N, N, ...

(every draft number from strongest to weakest, separated by commas, each draft exactly once)

Then on the final line:

WINNER: N

Nothing after that line."""


EVALUATOR_SCANNER_BLOCK = """=== MECHANICAL SCANNER RESULTS ===

{scanner_text}

=== CHAPTER OUTLINE ===

{outline_text}

"""


# ============================================================================
# Line-graft prompt — identify runner-up sentences that improve the winner
# ============================================================================

LINE_GRAFT_IDENTIFICATION_PROMPT = """You are comparing {N} drafts of the same chapter. Draft 1 is the WINNER — the strongest overall. Drafts 2–{N} are runners-up, ranked by literary quality.

Your job: find every sentence in Drafts 2–{N} that is genuinely sharper, more economical, more in-voice, or carries more subtext than its counterpart in Draft 1 at the same beat. Transplanting it should improve the winner without disrupting voice or continuity.

Even weak drafts often contain one or two sentences that outperform the winner at a specific beat. Those are the ones you are looking for. Do not cap your list artificially. Return as many as you find; return zero if none qualify.

A candidate sentence qualifies ONLY if ALL of the following hold:

1. It does the same narrative work as a sentence in the winner (same beat, same moment in the scene).
2. It is clearly sharper, more vivid, more economical, or carries more subtext than the winner's version — not merely a different way of saying the same thing.
3. The donor sentence itself contains ZERO of these hard-cap patterns: "the way X" observational framing (e.g., "the way he held the spoon"), periphrastic "as though he were" / "in the manner of" observational framing, "not X but Y" negation pivots in narration, named emotions in narration (e.g., "a wave of sadness"). If the donor sentence has any of these patterns, it is disqualified even if its content is stronger — transplanting it would only move the problem.
4. Transplanting it would not break continuity with the sentences around it in the winner.

KNOWN WEAK SPOTS IN THE WINNER

The mechanical scanner flagged these passages in Draft 1 (the winner) as carrying hard-cap violations. These are the places where a clean donor replacement would help the most. Check each of these first; if a runner-up covers the same beat with a clean sentence, prefer that graft.

{winner_flags}

SCANNER COUNTS PER DRAFT

{scanner_summary}

OUTPUT FORMAT — follow exactly:

GRAFT 1:
SOURCE: Draft [number]
<replace>[exact text from the winner to remove]</replace>
<with>[exact text from the runner-up to insert]</with>
REASON: [one sentence]

GRAFT 2:
SOURCE: Draft [number]
<replace>[exact text from the winner to remove]</replace>
<with>[exact text from the runner-up to insert]</with>
REASON: [one sentence]

... (as many GRAFT entries as you find)

Or if nothing qualifies:

NO_GRAFT

Quote the EXACT text to replace and the EXACT replacement. These strings will be used for automated find-and-replace, so character-level precision matters — copy them exactly as they appear in the drafts, including punctuation and spacing.

Do NOT force grafts. Return NO_GRAFT if nothing in the runners-up clearly improves the winner at the same beat. Nothing after the last GRAFT entry or NO_GRAFT."""


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
    # Pipeline outcome for this draft
    pipeline_role: str = ""  # "literary_winner" / "graft_donor" / ""


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
        missing = [i for i in range(1, n + 1) if i not in seen]
        ranking = deduped + missing
        if missing:
            parse_status = "partial"
    else:
        parse_status = "no_ranking_line"

    winner_match = re.search(r"WINNER:\s*(\d+)", raw)
    if winner_match:
        winner_idx = int(winner_match.group(1))
    else:
        winner_idx = ranking[0] if ranking else 1
        if parse_status == "clean":
            parse_status = "no_winner_line"

    winner_idx = max(1, min(winner_idx, n))
    winner_run_id = drafts[winner_idx - 1]["run_id"]

    return {
        "winner_run_id": winner_run_id,
        "winner_index": winner_idx,
        "ranking": ranking,
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
# Pipeline — linear: evaluate, export top-N, sentence-graft, ship
# ============================================================================

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
      A. Literary evaluation ranks all drafts. TOP 1 is the baseline deliverable.
      B. Export the top-N drafts as separate files.
      C. Sentence-graft pass: look for lines in the runners-up that improve
         TOP 1 at the same beat. If any apply, produce TOP1_GRAFTED and ship
         that as FINAL. Otherwise ship TOP 1 unchanged.

    There is no draft-level clean/dirty gate, no rubric fallback, and no
    post-graft verification gate. The scanner informs the graft pass but
    does not block shipping.

    Returns dict with:
      - literary_ranking: list of run_ids best → worst
      - literary_winner_run_id
      - final_text: the chapter shipped as FINAL
      - final_path: where FINAL was written
      - final_source: "top1_ungrafted" or "top1_grafted"
      - top_paths: list of paths for TOP1..TOP_N files
      - eval_raw: literary evaluator output
      - line_graft: dict from the sentence-graft pass
    """
    result = {
        "literary_ranking": [],
        "literary_winner_run_id": "",
        "final_text": "",
        "final_path": "",
        "final_source": "",
        "top_paths": [],
        "eval_raw": "",
        "line_graft": {},
    }

    # --- Stage A: Literary evaluation, all drafts ---
    lit = evaluate_drafts_with_anthropic(
        client, eval_model, drafts,
        outline_text=outline_text,
        scan_by_run_id=scan_by_run_id,
    )
    result["eval_raw"] = lit["raw_text"]
    lit_ranking_ids = [drafts[i - 1]["run_id"] for i in lit["ranking"]]
    result["literary_ranking"] = lit_ranking_ids
    result["literary_winner_run_id"] = lit["winner_run_id"]

    # --- Stage B: Top-N export ---
    top_paths = []
    for rank_pos, run_id in enumerate(lit_ranking_ids[:top_n], 1):
        draft_obj = next((d for d in drafts if d["run_id"] == run_id), None)
        if draft_obj is None:
            continue
        top_filename = f"TOP{rank_pos}_{batch_stub}_{run_id}.txt"
        top_path = FINAL_DIR / top_filename
        save_text(top_path, draft_obj["text"])
        top_paths.append(top_path)
    result["top_paths"] = top_paths

    # --- Stage C: Sentence-graft pass ---
    # Always run when there are at least 2 drafts. The pass decides for
    # itself whether any grafts qualify; NO_GRAFT is a valid outcome.
    top1_text = next(
        (d["text"] for d in drafts if d["run_id"] == lit["winner_run_id"]),
        "",
    )

    if len(lit_ranking_ids) >= 2:
        drafts_ranked = []
        for run_id in lit_ranking_ids[:top_n]:
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
        # Grafts applied — ship the grafted winner as FINAL.
        result["final_source"] = "top1_grafted"
        result["final_text"] = lg["grafted_text"]
        final_path = FINAL_DIR / f"FINAL_{batch_stub}_top1_grafted.txt"
        save_text(final_path, lg["grafted_text"])
        result["final_path"] = str(final_path)
    else:
        # No qualifying grafts — ship TOP 1 unchanged as FINAL.
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
    lines.append("")
    lines.append("=" * 60)
    lines.append("MECHANICAL SCAN RESULTS")
    lines.append("=" * 60)

    clean_count = sum(1 for d in drafts if scan_by_run_id.get(d["run_id"], {}).get("scan_hard_cap_pass"))
    lines.append(f"Clean (all hard caps held): {clean_count} / {len(drafts)}")
    lines.append("(Scan is informational — every draft is kept regardless of violations.)")
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

    lines.append("")
    lines.append("=" * 60)
    lines.append("LITERARY RANKING")
    lines.append("=" * 60)
    for rank_pos, run_id in enumerate(pipeline_result["literary_ranking"], 1):
        scan_status = "clean" if scan_by_run_id.get(run_id, {}).get("scan_hard_cap_pass") else "dirty"
        marker = ""
        if run_id == pipeline_result["literary_winner_run_id"]:
            marker += " [TOP 1 / LITERARY WINNER]"
        lines.append(f"  {rank_pos}. {run_id} — {scan_status}{marker}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("SENTENCE-GRAFT PASS")
    lines.append("=" * 60)
    lg = pipeline_result.get("line_graft") or {}
    if not lg:
        lines.append("Not run (fewer than 2 drafts in batch).")
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
            lines.append(f"  {rejected_dirty} rejected because the donor sentence itself carried a hard-cap pattern.")
        if rejected_no_match:
            lines.append(f"  {rejected_no_match} rejected because the REPLACE text did not match TOP 1 verbatim.")
    else:
        lines.append("No lines in the runners-up were judged superior to TOP 1.")

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

                    # Update CSV with ranking + roles
                    evaluation_id = f"eval_{batch_timestamp}"
                    update_records_bulk(csv_path, batch_run_ids_ordered, {
                        "is_winner": False,
                        "evaluation_id": evaluation_id,
                        "evaluator_model": eval_model,
                        "evaluation_raw": result["eval_raw"][:8000],
                        "pipeline_role": "",
                    })

                    for rank_pos, run_id in enumerate(result["literary_ranking"], 1):
                        update_record(csv_path, run_id, {"evaluation_rank": rank_pos})

                    update_record(csv_path, result["literary_winner_run_id"], {
                        "is_winner": True,
                        "pipeline_role": "literary_winner",
                    })

                    # Tag any donor whose line was transplanted into TOP 1
                    lg = result.get("line_graft") or {}
                    if lg.get("grafted"):
                        donor_positions = {g["source_draft"] for g in lg.get("grafts", [])}
                        ranking_ids = result.get("literary_ranking", [])
                        for pos in donor_positions:
                            # pos is 1-indexed position in the ranked list shown to the evaluator
                            if 1 <= pos <= len(ranking_ids):
                                donor_run_id = ranking_ids[pos - 1]
                                if donor_run_id != result["literary_winner_run_id"]:
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
                        files_to_push = (
                            [Path(result["final_path"])]
                            + list(result["top_paths"])
                            + [summary_path]
                        )
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
            "scan_hard_cap_pass",
            "scan_the_way_count", "scan_em_dash_count",
            "is_winner", "evaluation_rank",
        ]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True)

        # --- Show pipeline results if available ---
        result = st.session_state.get("last_pipeline_result")
        if result:
            st.markdown("---")
            st.subheader("Pipeline result")

            winner_id = result.get("literary_winner_run_id", "?")
            final_source = result.get("final_source", "")
            lg = result.get("line_graft") or {}

            if final_source == "top1_grafted":
                st.success(
                    f"TOP 1 GRAFTED: {len(lg.get('grafts', []))} line(s) "
                    f"transplanted from runners-up into `{winner_id}`. "
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
                        f"TOP 1 SHIPPED UNCHANGED: `{winner_id}`. "
                        f"Graft candidates identified but not applied "
                        f"({reason_text})."
                    )
                else:
                    st.info(
                        f"TOP 1 SHIPPED UNCHANGED: `{winner_id}`. "
                        f"No runner-up sentences judged superior to the winner at the same beat."
                    )
            else:
                st.info(f"Pipeline complete. Winner: `{winner_id}`.")

            with st.expander("Literary evaluator reasoning"):
                st.text(result["eval_raw"])

            # --- Sentence-graft pass details ---
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