"""
Micro-Prompt Harness — Quality Floor + Scanner-Ranked Pipeline + Stage F + Stage G
==================================================================================
Generate chapter drafts and ship TOP 1, optionally with sentence-level grafts.
v16 adds Stage F: a ridge-regression predictor of the Originality human-score,
fit on labeled_corpus.json and run on the final shipped text. No API call; the
ridge lives inline. Output is a predicted score (0–100) and a recommendation
band (SHIP / RECONSIDER / REGENERATE). Advisory — does not gate shipping.
v17 adds Stage G: a line-edit pass that runs on the final shipped text.
G1 is a mechanical copyedit — one LLM call with strict punctuation-only
instructions and a word-sequence invariant check that rejects any edit that
changed words (not just punctuation). G2 is a deterministic AI-tell scan
against a module-level word list (starting with "particular" / "particularly").
G3 resolves each flagged sentence with a three-behavior cascade: (a) same-beat
graft from a runner-up acceptable draft where possible; (b) mechanical
deletion with a/an article repair where deletion leaves the sentence intact;
(c) flag for manual rewrite when neither path applies. Stage G writes
FINAL_<batch_stub>_LINEEDITED.txt and a LINEEDIT_REPORT_<batch_stub>.txt
audit file alongside the existing FINAL outputs. Stage F's prediction
subsequently runs on the line-edited text when one was produced.
v25 expands Stage G into a copy-edit pass against the v19 outline's residual
leak patterns. AI_TELL_WORDS grows from two single-word entries into a
construction catalogue with two entry classes: deletable single-word hedges
(particular, particularly, merely) and graft-only sentence-internal
constructions (as though, as if, conditional similes "as a man might,"
"the way [pron]" variants v19 enumerated, "of a [noun] who" portrait
constructions, "the kind of [X] that" classifiers, "the noise/sound a X
makes" sense-perception variants). Graft-only entries use a GRAFT_ONLY
sentinel in the replacement slot; G3b deletion is skipped for those because
removing the construction would damage the sentence — they fall through to
G3a graft and then to G3c manual flag if no clean graft is available.
v25 also adds G4: a single LLM call after G3 over the whole post-G3 text,
deletion-only, targeting three multi-sentence patterns that escape
sentence-by-sentence flagging — negation triplets running across consecutive
sentences, closing aphoristic gloss at scene/paragraph ends, and classify-
by-genre constructions that survived G3. G4 is protected by three invariants
(±2% word-count band, no whole-paragraph deletion, deletion-only word-set
check). Audit entries from G4 are appended to the same LINEEDIT_REPORT file.
The pipeline answers three questions in order:
  Q1. Is this draft acceptable as prose? A pass/fail quality floor on each
      draft. Voice intact, beats landed, dialogue working, no collapses or
      incoherences. Unacceptable drafts are dropped entirely — not shipped,
      not used as graft donors, not ranked. If zero drafts clear the floor,
      the pipeline halts and reports failure. The floor is lenient: it
      catches drafts you would be embarrassed to ship, not drafts that are
      merely different from the others.
  Q2. Among acceptable drafts, which ships? Literary ranking leads.
      The evaluator's top-ranked draft is TOP 1 unless it is a scanner
      outlier — defined as having more than double the batch median
      violation count. In that case the scanner vetoes the pick and the
      next literary-ranked draft that is not an outlier becomes TOP 1.
      This restores prose quality as the primary selection signal while
      protecting against the v9 failure mode (best prose = worst scan).
  Q3. Two-stage graft pass with two pathways and two graft units:
      Stage 1 (identification). Wide-net sweep of all runner-ups. For every
      sentence or clause that serves the same NARRATIVE FUNCTION as some
      text in TOP 1 — characterizing the same subject, marking the same
      interior movement, describing the same object or action — emit a
      candidate. Staging may differ; function must match.
      Stage 2 (commit). Each candidate is judged COMMIT or REJECT against
      three gates: donor is clean of hard-cap patterns, replacement
      preserves continuity with surrounding TOP 1 prose, and the graft
      genuinely improves on the TOP 1 text. Minimal seam edits at the
      boundary are permitted (at most one connecting word per side) and
      logged.
      Two pathways remain:
      Type A — FLAG REPAIR. TOP 1 text carries a flagged construction;
      donor is clean at the same function.
      Type B — QUALITY UPGRADE. TOP 1 text is acceptable but the donor
      is meaningfully better prose at the same function.
      Two units:
      Sentence-level: replace a whole TOP 1 sentence with a donor sentence.
      Phrase-level: replace a clause inside a TOP 1 sentence with a donor
      clause. Phrase grafts let a sharp clause from a divergent scene
      enter TOP 1 without importing the surrounding staging.
      Substitution is deterministic find-and-replace in Python, not an
      LLM pass. The model identifies, judges, and specifies seam edits;
      the code substitutes.
The scanner informs Q2 (veto only) and Q3. The literary evaluator drives
Q1 and Q2 selection.
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
import numpy as np
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
APP_VERSION = "v31"
RUNS_DIR = Path("micro_prompt_runs")
OUTPUTS_DIR = RUNS_DIR / "flat_outputs"
FINAL_DIR = RUNS_DIR / "final_deliverables"
CSV_FILENAME = "runs.csv"
PROMPTS_CSV = "prompts.csv"
DEFAULT_GEN_MODEL = "claude-opus-4-7"
DEFAULT_EVAL_MODEL = "claude-opus-4-6"
MAX_GEN_TOKENS = 16000
MAX_EVAL_TOKENS = 8000
QUALITY_GATE_MAX_TRIES = 5
# Stage F: ridge predictor of Originality human-score.
# The labeled corpus file lives at the repo root beside this script. If
# missing, Stage F degrades gracefully — the pipeline runs, the summary
# notes the predictor is unavailable.
LABELED_CORPUS_PATH = Path("labeled_corpus.json")
STAGE_F_RIDGE_LAMBDA = 3.0           # corpus-fit L2 regularization
STAGE_F_BAND_SHIP = 88               # pred ≥ 88 → SHIP
STAGE_F_BAND_CAUTION = 80            # 80 ≤ pred < 88 → RECONSIDER
# below STAGE_F_BAND_CAUTION → REGENERATE
# Stage G: line-edit pass.
# AI_TELL_WORDS drives the G2 identification pass and the G3b deletion
# heuristic. Each entry maps a canonical name (word or short construction
# label) to a list of (pattern, replacement) tuples. Patterns are
# case-insensitive. For deletable entries the replacement is a string; the
# patterns are applied in order and the a/an article repair runs afterward
# on the whole edited sentence. For graft-only entries the replacement is
# the GRAFT_ONLY sentinel — apply_deletion_heuristic skips those, and the
# flagged sentence falls through to G3a (same-beat graft) and, if no clean
# graft exists, to G3c (manual rewrite flag).
#
# To add a deletable single-word tell: append an entry whose patterns map
# the word to a deletion or substitution string. Start narrow.
#
# To add a graft-only construction: append an entry whose patterns use
# GRAFT_ONLY as the replacement. The pattern only needs to flag the
# construction; G3a will look for a clean alternative in the runner-ups.
STAGE_G_ENABLED = True
STAGE_G_MAX_WORD_DELTA = 0  # G1 mechanical pass rejected if word count changes
# v30: Line-graft pass — Type C (deletion) controls.
# Type C deletions are riskier than Type A/B grafts — an over-eager
# deletion pass can remove load-bearing prose. The feature is enabled
# by default with a cap on the number of deletions per chapter; the
# cap is conservative for the first few v30 runs and can be relaxed
# once the pass is shown to behave.
STAGE_G_TYPE_C_ENABLED = True
STAGE_G_TYPE_C_MAX_DELETIONS = 6
# v30 → v31 disabled: the 5% overshoot block in v30 caused +11% chapter
# length and an 80% increase in red word-count. Reverted to 0% — drafter
# now writes at the outline's stated target. Kept as a flag so the
# experiment can be re-run if a smaller overshoot is hypothesized.
STAGE_G_OVERSHOOT_PCT = 0.0  # was 0.05 in v30; disabled in v31 after regression
# Sentinel for graft-only patterns. apply_deletion_heuristic checks `is None`
# and skips deletion; G2 still flags the sentence; G3a still tries to graft.
GRAFT_ONLY = None
AI_TELL_WORDS = {
    # ── Class 2: deletable single-word hedges and modifiers ─────────────────
    # G3b removes these in place; the a/an article repair runs afterward.
    "particular": [
        # "a particular <X>"  → "a <X>"    (a/an fix-up runs afterward)
        # "the particular <X>" → "the <X>"
        (r"\b(a|an|the)\s+particular\s+", r"\1 "),
    ],
    "particularly": [
        # "particularly <word>" → "<word>"
        (r"\bparticularly\s+", ""),
    ],
    "merely": [
        # "merely <word>" → "<word>"  (almost always a hedging adverb)
        (r"\bmerely\s+", ""),
    ],
    # ── Class 1: graft-only sentence-internal constructions ────────────────
    # These are the v19-outline residual leaks observed in the v24 chapters.
    # Deletion would damage the sentence; G3a grafts from a clean runner-up
    # or G3c flags for manual rewrite.
    "as though": [
        (r"\bas though\b", GRAFT_ONLY),
    ],
    "as if": [
        (r"\bas if\b", GRAFT_ONLY),
    ],
    "as a [noun] might/would": [
        # Cap 1 conditional-characterisation variant: "as a man might touch
        # the flank of a horse," "as a clergyman would pause."
        (r"\bas (?:a|the) (?:man|woman|person|child|gentleman|lady|servant|soldier|priest|clergyman|stranger)\s+(?:might|would|will)\b", GRAFT_ONLY),
    ],
    "the way [pron]": [
        # Cap 1 base form: "the way he looked," "the way she held the cup."
        (r"\bthe way (?:he|she|it|they|I|we|you|one|a man|a woman|men|women|people)\b", GRAFT_ONLY),
    ],
    "the way [thing] wants/wanted": [
        # Cap 1 passive-intent variant: "the way the dough wanted to be
        # worked," "the way it wants to be laid."
        (r"\bthe way (?:\w+\s+){0,2}(?:wants?|wanted)\b", GRAFT_ONLY),
    ],
    "the noise/sound a [noun] makes": [
        # Cap 1 sense-perception variant: "the noise a man makes when,"
        # "the sound a horse gives at the bit."
        (r"\bthe (?:noise|sound|look|smell) (?:a|the|that) \w+\s+(?:makes|made|gives|gave)\b", GRAFT_ONLY),
    ],
    "of a [noun] who": [
        # Portrait construction: "the look of a man who," "the patience of
        # a woman who."
        (r"\bof (?:a|the) (?:man|woman|person|men|women|people|child)\s+who\b", GRAFT_ONLY),
    ],
    "the kind of [X] that": [
        # Classify-by-genre: "the kind of plain that costs money," "the kind
        # of silence that holds."
        (r"\bthe kind of \w+(?:\s+\w+){0,2}\s+(?:that|who)\b", GRAFT_ONLY),
    ],
    "Not the X; the Y / Not the X. The Y.": [
        # Cap 3 negation-pivot bridge variants v19 enumerated. This is a
        # sentence-initial rhetorical construction; require start-of-span
        # so the pattern does not fire on mid-sentence "not the X; the Y"
        # inside dialogue or longer prose. Semicolon bridge: "Not the
        # tired of a long night; the tired of a long October."  Full-stop
        # bridge: "Not the X. The Y." (G2 sees the first sentence; the
        # graft-only semantics handle resolution.)
        (r"^Not\s+the\s+\w+(?:\s+\w+){0,4}[;\.]\s+the\s+\w+", GRAFT_ONLY),
    ],
}
# ============================================================================
# Literary evaluator prompt — unchanged from original app
# ============================================================================
EVALUATOR_PROMPT = """You are evaluating {N} drafts of the same chapter against its outline. You have three inputs: the chapter outline, the mechanical scanner results for each draft, and the drafts themselves.
Read every draft in full. Do not skim.
Your job is to do three things in order:
(1) apply a lenient quality floor so the pipeline knows which drafts are fit to ship at all,
(2) assign each draft a prose-quality score,
and (3) rank only the top-scoring drafts.
The pipeline will keep ONLY the drafts that tie for the highest QUALITY_SCORE among ACCEPTABLE drafts. Every acceptable draft below that top score is discarded before the downstream AI ranking. So be willing to use ties when the writing quality is genuinely equal, but do not collapse distinct quality levels into a tie out of caution.
YOUR METHOD — in this order:
1. WORD COUNTS. Note each draft's word count against the outline's target range. Flag any that are materially short or over.
2. MECHANICAL COMPLIANCE. The scanner results are provided below. For each draft, note the violation counts. Do not re-scan — use the provided numbers. Reference them when assessing prose, but do not let them drive your quality verdict. Violations affect downstream diagnostics; your job here is writing quality.
3. QUALITY FLOOR — one verdict per draft. For each draft, decide ACCEPTABLE or UNACCEPTABLE. Apply a LENIENT standard: mark a draft ACCEPTABLE unless you would be embarrassed to ship it. UNACCEPTABLE means one or more of:
   - Voice collapse: the POV character's interior voice is absent, generic, or wrong register for long stretches.
   - Beats missing or compressed to the point of incoherence: a scene the outline requires is not on the page or is a throwaway line.
   - Dialogue that doesn't land: exchanges without subtext, without weapons, without stakes; turns that read like exposition dumps.
   - Structural failure: the chapter doesn't arrive where the outline says it arrives, or the ending doesn't close what was opened.
   - Prose-level damage: runs of flat summary where the outline asks for scene, long stretches of interpretive narration where the outline asks for observation and judgment, abandoned subplots, characters acting out of their profiles.
   Merely being less elegant than another draft is NOT grounds for UNACCEPTABLE. Stylistic difference is NOT grounds for UNACCEPTABLE. A draft can be ACCEPTABLE even if another draft is better at the same beats.
4. QUALITY SCORE — one integer score per draft, on a 1–10 scale, where 10 is the strongest prose in this batch and 1 is the weakest prose that still functions at all. Score on prose quality only: voice fidelity, dialogue craft, interior sharpness, beat execution, specificity, texture, rhythm, wit. Use the scale comparatively across THIS batch. If two drafts are genuinely equal in prose quality, give them the same score. UNACCEPTABLE drafts should get a score of 0.
5. TIE-ONLY RANKING. Rank ONLY the ACCEPTABLE drafts that received the highest QUALITY_SCORE. Omit every other draft from the ranking line, even if acceptable. The downstream AI ranking will break ties among these top-scoring drafts.
6. GRAFT CANDIDATES. From the non-winning top-scoring drafts, name specific lines or passages worth transplanting into the eventual winner. Quote a few words for identification and name the beat where each would land. This is advisory context for the downstream graft pass.
OUTPUT FORMAT
For each draft, write a paragraph (3-5 sentences) covering voice quality, best moment, notable weaknesses, and a one-sentence justification for your quality verdict. Reference the scanner numbers.
Then on a line by itself for each draft (one line per draft):
QUALITY: Draft N — ACCEPTABLE
or
QUALITY: Draft N — UNACCEPTABLE — [one-sentence reason]
Then on a line by itself for each draft:
QUALITY_SCORE: Draft N — S
(where S is an integer from 0 to 10. Use 0 only for UNACCEPTABLE drafts.)
Then a graft paragraph naming specific lines from non-winning top-scoring drafts worth transplanting, with beat locations.
Then on a line by itself:
RANKING: N, N, N, ...
(ONLY the ACCEPTABLE drafts tied at the highest QUALITY_SCORE, from strongest to weakest if there is still a distinction. Separated by commas. If only one draft has the top score, the line should contain only that draft number.)
Then on the final line:
WINNER: N
(the one draft from the RANKING line you would ship on literary grounds, before the downstream AI ranking breaks ties)
Nothing after that line."""
EVALUATOR_SCANNER_BLOCK = """=== MECHANICAL SCANNER RESULTS ===
{scanner_text}
=== CHAPTER OUTLINE ===
{outline_text}
"""
# ============================================================================
# Line-graft prompts — two-stage: candidate identification, then commit
# ============================================================================
LINE_GRAFT_CANDIDATE_PROMPT = """You are comparing {N} drafts of the same chapter. Draft 1 is TOP 1 — the shipping base. Drafts 2–{N} are acceptable runners-up.
Your job is to identify every sentence, clause, OR multi-sentence run in TOP 1 that should be REPLACED with a counterpart from a runner-up — OR DELETED outright if no runner-up has a clean equivalent and the passage is not load-bearing. This is a wide-net identification pass. Do not commit yet. Commit decisions happen in the next step.
Three kinds of candidate:
TYPE A — FLAG REPAIR. TOP 1's text carries one of the flagged patterns (listed below) and a runner-up has a clean version that does the same narrative function in the scene.
TYPE B — QUALITY UPGRADE. TOP 1's text is acceptable but a runner-up sentence or clause at the same narrative function is meaningfully better — sharper image, more specific physical detail, stronger interior voice, cleaner dialogue. The runner-up version is the one a reader would underline.
TYPE C — DELETION. TOP 1 carries a flagged construction (or a stretch of flat connective tissue), AND no runner-up has an equivalent passage that does the same narrative function, AND deleting the passage entirely does not break the scene's continuity. This is the right call when the winner has padded a transition or scene-setter with atmospheric/procedural prose that the other drafts simply skipped — meaning that beat is not load-bearing. Examples of strong deletion candidates: a single atmospheric sentence ("Fog sat over the yard in a low gray skin") that no runner-up reproduces; a procedural-action chain ("I shrugged into a jacket, took the keys, locked the deadbolt twice") that no runner-up matches; a sensory-fragment pair ("The sink had a drip. The kettle had a hairline crack I had been ignoring for eight months") that no other draft includes. If the deletion would leave a gap a reader would notice (a missing emotional beat, a missing piece of plot, a missing dialogue cue), do not propose it.
GRAFT UNITS. A candidate may be:
- SENTENCE: replace a whole TOP 1 sentence with a donor sentence (Type A or B), OR delete a whole TOP 1 sentence (Type C).
- PHRASE: replace a clause or phrase inside a TOP 1 sentence with a donor clause or phrase (Type A or B only — Type C does not use phrase units).
- RUN: replace a multi-sentence span (a staccato run, an anaphoric cluster, an aphoristic couplet, a sensory-fragment pair, a procedural chain, or a tautological restatement) with a SINGLE flowing donor sentence — OR delete the whole run (Type C). The donor for a RUN graft (Type A/B) is one sentence; the TOP1_TEXT is the verbatim multi-sentence span flagged.
MATCHING RULE. For Type A and Type B grafts, the donor and the TOP 1 text must serve the same NARRATIVE FUNCTION — characterize the same subject, mark the same interior movement, describe the same object or action. Surrounding staging MAY DIFFER; function must be the same. Do not reject a candidate because the scene frames the moment differently — only because the two texts do different work.
DELETION RULE. For Type C, before proposing deletion, verify by reading the runner-up drafts that NO runner-up renders the same narrative function. If even one runner-up has a clean equivalent, propose Type A or Type B graft instead, not deletion. Also verify that deletion does not break causal flow: the sentence(s) before and after the proposed deletion must still read as continuous prose.
CLEAN DONOR REQUIREMENT (Type A and Type B only). The donor text itself must contain ZERO flagged patterns:
- "the way X" observational framing
- Periphrastic observational framing ("as though he were," "like a woman who," "in the manner of," "as a man who")
- "not X but Y" negation pivots in narration (dialogue permitted)
- "Not X. Y." or "Not X; Y." bridge variants
- "a [specific/particular/certain] kind of [adj]" verdict
- Triple-noun-phrase escalation at sentence start or mid-sentence (three comma-separated phrases with evaluative heads)
- Named emotions in third-person-like form ("a wave of sadness"); first-person naming in the POV character's interior voice is PERMITTED
- "I named [the thing in my chest / the feeling / it / that]" tic
- Polysyndetic run-on (4+ coordinating "and"s in a single sentence)
- Tautological loop ("X did the thing Xs do")
- Aphoristic generalization templates ("People want to be told...", "It looks like a tide", "Dying people get quiet before they get tired", "the alternative is...")
- Em-dash over-cap (count exceeds 12 for the whole chapter)
- Staccato runs (≤ 5-word sentences appearing in 2+ consecutive sentences in narrative prose)
- Anaphoric escalation (3+ sentences in a 5-sentence window sharing the same opener)
- v30: Scene-set appositive ("X, which was Y" / ", and [Name] had been ___ing" / ", because ___ing was a ___" — declarative scene-set followed by an interpretive coda)
- v30: Tautological restatement (cross-sentence noun-restate "That was a real X. I could [verb] a real X." / within-sentence "X lived by Y because that's where Xs live" / triple self-affirmation fragments "These are my hands. I'm Nell. I'm scared.")
- v30: Sensory-fragment pair (two consecutive 4–12-word sentences both shaped "The/A [object] [verb-of-state] [predicate]" with no first-person interior between or inside them)
- v30: Procedural-action chain (3+ comma-separated past-tense action verbs in series, OR 3+ consecutive sentences each opening "I [action verb]" with no interior commentary)
- v31: "felt like a [abstract noun]" (commitment, decision, mistake, lie, surrender, etc.) — meta-categorization in copular-felt-like form
- v31: ", which was/is [meta-statement]" — appositive coda with non-comparative meta predicate ("which was the closest I had felt to whole", "which is how bodies are supposed to work", "which was without asking")
- v31: "but only because I [verb]" — meta-because in clause-final position
- v31: "the [adj] kind, the kind that" — verdict-escalation appositive
- v31: "X had gone cold. So had my Y." — sensory-pair with inverted-aux second sentence
- v31: "the person/stranger/man/woman watching/listening/holding/waiting" — periphrastic-observer
- v31: "a dog/man/woman who had decided/known/seen/chosen/understood" — portrait construction with cognition verb on animate subject
A donor that would introduce a new flag is disqualified. Note any such concern in JUSTIFICATION so the commit pass can address it.
SCANNER-FLAGGED PASSAGES IN TOP 1
{winner_flags}
SCANNER COUNTS PER DRAFT
{scanner_summary}
OUTPUT FORMAT — follow exactly. For each candidate, emit one block:
CANDIDATE <n>
TYPE: A | B | C
UNIT: sentence | phrase | run
TOP1_TEXT: "<exact text to replace or delete>"
DONOR_DRAFT: <draft number 2–{N}, or 0 for Type C deletion>
DONOR_TEXT: "<exact donor text, or empty string "" for Type C deletion>"
FUNCTION: <one line — the narrative function the TOP 1 text serves>
JUSTIFICATION: <one line — for Type A, name the flag; for Type B, name what makes the donor better; for Type C, name (1) the flag if any, (2) the runner-up drafts you confirmed do NOT render this beat, and (3) why deletion preserves continuity>
Cast a wide net. Do NOT cap the list. If a candidate looks marginal, include it and let the commit pass judge.
If no candidates exist at all, return exactly:
NO_CANDIDATES
Quote TOP1_TEXT and DONOR_TEXT EXACTLY as they appear in the drafts — character-level precision, including punctuation and spacing. The commit pass and the downstream substitution rely on verbatim matching. For Type C, DONOR_DRAFT must be 0 and DONOR_TEXT must be empty (the literal string "")."""
LINE_GRAFT_COMMIT_PROMPT = """You are reviewing graft candidates proposed in an earlier identification pass, and deciding which to commit.
TOP 1 is the shipping base. Below the candidates you will find all {N} drafts. Each candidate proposes either replacing some TOP1_TEXT with a DONOR_TEXT from a runner-up (Type A or B), or DELETING some TOP1_TEXT outright (Type C).
For each candidate you must decide COMMIT or REJECT.
COMMIT a Type A or B graft when:
- The graft genuinely improves TOP 1 (clears a flag, or imports a sharper sentence or clause)
- The donor text is clean of all flagged patterns
- Continuity is preserved — the replacement reads naturally with the sentences before and after it in TOP 1
COMMIT a Type C deletion when:
- The flagged passage in TOP 1 is genuinely not load-bearing (no plot beat, no emotional turn, no necessary dialogue cue is being removed)
- No runner-up renders the same narrative function (you have read the runner-ups and verified)
- Continuity is preserved — the sentence before the deletion and the sentence after it read as continuous prose
- Deleting moves the chapter closer to its target word count rather than further from it
REJECT (any type) when:
- The donor text itself carries a flagged pattern (any of those listed in the candidate-identification pass)
- The graft or deletion breaks continuity with the surrounding TOP 1 prose
- The donor is only marginally different, not meaningfully better
- The function match is superficial — the two texts do different narrative work despite looking similar
- For UNIT=run grafts (Type A/B): the donor is itself part of a staccato run, an anaphoric cluster, an aphoristic couplet, a sensory-fragment pair, a procedural chain, or a tautological restatement in the donor draft (it must be a SINGLE flowing sentence, not a fragment)
- For UNIT=run deletions (Type C): removing the run leaves a missing beat the reader will notice
SEAM EDITS (Type A/B). If the graft needs a minor adjustment at its boundary — a connecting word added, changed, or removed on either side to preserve grammar or flow — extend the TOP1_TEXT to include the adjusted word, and bake the adjustment into the DONOR_TEXT. Describe what changed in the SEAM_EDITS field. At most one connecting word or short phrase per side. If more is needed, reject the graft.
SEAM EDITS (Type C). For deletions, "seam" means making sure the surrounding paragraph still reads cleanly. If removing the sentence requires adjusting punctuation or capitalization on the next sentence (e.g. lowercase to uppercase, removing a stray space), extend the TOP1_TEXT to include the adjustment site and emit the corrected version as DONOR_TEXT. If a comma joins must be split into a period join, that counts as a permitted seam edit. If more than one seam edit is needed, reject the deletion.
If the candidate's original TOP1_TEXT or DONOR_TEXT is close but not verbatim to what appears in the drafts, correct it in your output. The final TOP1_TEXT and DONOR_TEXT you emit must match the drafts character-for-character, or the downstream substitution will fail.
CANDIDATES UNDER REVIEW
{candidates_block}
OUTPUT FORMAT — follow exactly. For each candidate, emit one block:
COMMIT_CANDIDATE <n>
DECISION: COMMIT | REJECT
TYPE: A | B | C
UNIT: sentence | phrase | run
TOP1_TEXT: "<exact text to replace or delete, verbatim from TOP 1>"
DONOR_DRAFT: <draft number, or 0 for Type C>
DONOR_TEXT: "<exact donor text with any seam edits baked in, or empty string "" for Type C>"
SEAM_EDITS: none | <one-line description of what changed at the boundary>
REASON: <one line>
After all blocks, emit a summary line:
FINAL_GRAFTS: <comma-separated candidate numbers that were COMMITted, or NONE>
Quote TOP1_TEXT and DONOR_TEXT EXACTLY. Do not paraphrase. Character-level precision is required."""
# ============================================================================
# Final pass — commercial vs literary pick across acceptable drafts
# ============================================================================
FINAL_PASS_PROMPT = """You will receive {N} drafts of the same chapter. The outline's GLOBAL DRAFTING CONTROLS section is the binding reference for register targets, hard caps, and per-beat contract requirements.
Read each draft end to end. Write a craft evaluation in prose, one paragraph per draft, noting what it does well and where the register drifts. Attend to sustained interior voice, whether each per-beat contract item named in the outline lands on the page with the specifics the outline asks for, whether the chapter's emotional channel is open in the beats the outline names for it, and whether the prose carries aphoristic closures, stacked periphrastic observation, or "the way X" constructions that cost the target register.
After the per-draft paragraphs, write a comparative paragraph that contrasts the two picks you will name and explains the trade each represents.
Close with exactly two lines in this format, with no other text after them:
MOST_LITERARY: T<n>
MOST_COMMERCIAL: T<n>
The literary pick is the draft that reads strongest as literary fiction within the outline's named tradition — richer prose texture, more willing flourish, more interior weight per sentence. The commercial pick is the draft that best fits the outline's market positioning and delivers the per-beat contract items with the cleanest interior voice.
OUTLINE (GLOBAL DRAFTING CONTROLS reference)
{outline_text}
"""
def run_final_pass(
    client,
    eval_model: str,
    acceptable_drafts: list,
    outline_text: str,
    batch_stub: str,
) -> dict:
    """Evaluate all acceptable drafts and pick one literary winner and one
    commercial winner. One LLM call, paragraph-per-draft reasoning, tagged
    picks at the tail for deterministic parsing.
    Args:
        acceptable_drafts: list of draft dicts that cleared Q1. Each has
                           'run_id' and 'text'. Position in this list is
                           the T<n> index used in the tagged output —
                           T1 is drafts[0], T2 is drafts[1], etc.
        outline_text: the chapter outline, injected into the prompt so
                      the evaluator has the GLOBAL DRAFTING CONTROLS
                      section to anchor register judgments.
        batch_stub: for file naming.
    Returns dict with:
      - ran: bool (False if fewer than 2 acceptable drafts)
      - literary_index: 1-indexed position of the literary pick in
                        acceptable_drafts, or 0 if unparsed
      - commercial_index: 1-indexed position of the commercial pick, or 0
      - literary_run_id: run_id of the literary pick, or ""
      - commercial_run_id: run_id of the commercial pick, or ""
      - literary_path: file path of the literary pick's saved text, or ""
      - commercial_path: file path of the commercial pick's saved text, or ""
      - reasoning_path: file path of the saved reasoning, or ""
      - raw: full model output (reasoning + tags)
    """
    result = {
        "ran": False,
        "literary_index": 0,
        "commercial_index": 0,
        "literary_run_id": "",
        "commercial_run_id": "",
        "literary_path": "",
        "commercial_path": "",
        "reasoning_path": "",
        "raw": "",
    }
    n = len(acceptable_drafts)
    if n < 2:
        return result
    prompt = FINAL_PASS_PROMPT.format(
        N=n,
        outline_text=(outline_text.strip() if outline_text
                      else "(no outline provided)"),
    )
    parts = [prompt]
    for i, d in enumerate(acceptable_drafts, 1):
        parts.append(
            f"\n\n=== T{i} (run_id: {d['run_id']}) ===\n\n{d['text']}"
        )
    resp = client.messages.create(
        model=eval_model,
        max_tokens=MAX_EVAL_TOKENS,
        messages=[{"role": "user", "content": "".join(parts)}],
    )
    raw = "\n".join(b.text for b in resp.content if getattr(b, "text", None))
    result["raw"] = raw
    result["ran"] = True
    lit_m = re.search(r"MOST_LITERARY:\s*T\s*(\d+)", raw, re.IGNORECASE)
    com_m = re.search(r"MOST_COMMERCIAL:\s*T\s*(\d+)", raw, re.IGNORECASE)
    if lit_m:
        idx = int(lit_m.group(1))
        if 1 <= idx <= n:
            result["literary_index"] = idx
            result["literary_run_id"] = acceptable_drafts[idx - 1]["run_id"]
    if com_m:
        idx = int(com_m.group(1))
        if 1 <= idx <= n:
            result["commercial_index"] = idx
            result["commercial_run_id"] = acceptable_drafts[idx - 1]["run_id"]
    # Save the two picks as separate files with the picks in the filenames.
    lit_idx = result["literary_index"]
    com_idx = result["commercial_index"]
    if lit_idx:
        lit_path = FINAL_DIR / (
            f"FINAL_{batch_stub}_LITERARY_PICK_T{lit_idx}_run-{result['literary_run_id']}"
            + (f"_COMMERCIAL_T{com_idx}" if com_idx else "")
            + ".txt"
        )
        save_text(lit_path, acceptable_drafts[lit_idx - 1]["text"])
        result["literary_path"] = str(lit_path)
    if com_idx:
        com_path = FINAL_DIR / (
            f"FINAL_{batch_stub}_COMMERCIAL_PICK_T{com_idx}_run-{result['commercial_run_id']}"
            + (f"_LITERARY_T{lit_idx}" if lit_idx else "")
            + ".txt"
        )
        save_text(com_path, acceptable_drafts[com_idx - 1]["text"])
        result["commercial_path"] = str(com_path)
    # Save the reasoning too, for auditability.
    reasoning_path = FINAL_DIR / f"FINAL_PASS_REASONING_{batch_stub}.txt"
    save_text(reasoning_path, raw)
    result["reasoning_path"] = str(reasoning_path)
    return result
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
    scan_aphoristic_count: int = 0
    scan_backfill_count: int = 0
    scan_verdict_count: int = 0
    scan_not_bridge_count: int = 0
    scan_verdict_kind_of_count: int = 0
    scan_triple_noun_count: int = 0
    scan_i_named_count: int = 0
    scan_staccato_count: int = 0
    scan_polysyndetic_count: int = 0
    scan_aphoristic_couplet_count: int = 0
    scan_mid_appositive_count: int = 0
    scan_tautology_count: int = 0
    scan_anaphoric_escalation_count: int = 0
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
    quality_score: int = 0
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
# Aphoristic standalone — short sentence, abstract weather/time/atmosphere
# subject, verdict verb. Matches things like "Morning light did not improve
# it.", "The silence offered nothing.", "The hour gave nothing back."
# Anchors on sentence start (after ./!/?/closing-quote or at string start).
# The verb-phrase vocabulary is kept tight so plain description ("the
# morning was cold") does not false-positive — only verdict forms match.
APHORISTIC_STANDALONE_PATTERN = re.compile(
    r"(?:^|(?<=[.!?\u201d\"])\s+)"
    r"(?:The\s+)?"
    r"(?:morning|evening|afternoon|night|dawn|dusk|day|silence|air|hour|"
    r"room|quiet|weather|house|year|dark|stillness|world|wind|rain|cold|"
    r"heat|light)"
    r"(?:\s+(?:light|air|wind|rain|cold|heat|quiet|silence|stillness))?"
    r"\s+"
    r"(?:did\s+not|offered\s+(?:no|nothing)|gave\s+(?:nothing|no|back)|"
    r"held\s+its|helped\s+nothing|was\s+no\s+(?:better|help|comfort|use|improvement)|"
    r"made\s+no\s+(?:difference|improvement))"
    r"\b",
    re.IGNORECASE,
)
# Explanatory backfill — ", because I had known / thought / realised / seen
# / understood / suspected / recognised / guessed / sensed". The model's
# reflex is to state an action and then explain it in the same sentence
# with a "because" tail. The comma-anchor and the "had + past-participle"
# form keep the pattern tight and distinctive.
EXPLANATORY_BACKFILL_PATTERN = re.compile(
    r",\s*(?:because|since)\s+(?:I|she|he)\s+(?:had|'d)\s+"
    r"(?:known|thought|realised|realized|seen|felt|understood|"
    r"suspected|recognised|recognized|guessed|sensed)\b",
    re.IGNORECASE,
)
# Verdict construction — "[noun] too [adjective] for [determiner] [noun]".
# Example failures: "the paper too good for the business", "the hands too
# clean for the work", "the coat too fine for the yard". Characters may
# use it in dialogue, so matches inside quotes are filtered out downstream
# the same way NOT_BUT_PATTERN matches are filtered.
VERDICT_TOO_FOR_PATTERN = re.compile(
    r"\btoo\s+\w+\s+for\s+(?:the|a|an|his|her|my|its|this|that|their|our)\s+\w+",
    re.IGNORECASE,
)
# v26 additions — Cap 3 bridge variants, Cap 11 triple-noun, "kind of"
# verdict variant, and the "I named [it/that/the feeling]" interior tic.
# Cap 3 bridge variants — "Not X. Y." and "Not X; Y." that v19 enumerated.
# The model reaches for these when the canonical "not X but Y" is closed.
# Quantifier-only X phrases ("not a lot," "not much") are excluded — those
# are quantity refinements, not category pivots, and score green in
# detector corpora. The pattern requires X to be a content noun phrase
# that Y can substitute for.
NOT_BRIDGE_PATTERN = re.compile(
    r"(?:^|(?<=[.!?\u201d\"])\s+|\n\s*)"
    r"Not\s+(?:a|an|the)\s+"
    r"(?!(?:lot|little|bit|few|many|much|moment|second|minute|while|long\s+time)\b)"
    r"(?:[a-z][\w']*\s+){0,8}?[a-z][\w']*"
    r"\s*[.;]\s+"
    r"(?:A|An|The)\s+[a-z]",
    re.MULTILINE,
)
# Verdict construction — "a [specific|particular|certain] kind of [adj]"
# closes a loophole in VERDICT_TOO_FOR. The "kind of" + bare-adjective
# form is the same verdict cadence the "too X for Y" cap targets, in a
# different syntactic dress. Excludes "kind of person/man/woman/thing"
# which are noun-phrase uses, not verdict uses.
VERDICT_KIND_OF_PATTERN = re.compile(
    r"\b(?:a|an)\s+"
    r"(?:specific|particular|certain|peculiar|special|distinct|different)"
    r"\s+kind\s+of\s+"
    r"(?:dangerous|cruel|terrible|awful|tired|sad|angry|afraid|scared|"
    r"wrong|strange|quiet|still|alone|broken|lost|lovely|beautiful|"
    r"hard|soft|mean|kind|gentle|fierce|wild|patient|empty|full|stupid|"
    r"clever|smart|brave|brittle|tender|ruthless|honest|exhausted)"
    r"\b(?!\s+(?:person|man|woman|thing|day|night|love|hate))",
    re.IGNORECASE,
)
# Cap 11 — Triple-noun-phrase escalation. Three or more comma-separated
# phrases at sentence start where the heads carry evaluative weight
# (suffix-evaluatives -ing/-ed/-y/-ic/-al/-ful/-less/-ous/-ish OR a short
# bare-adjective list). One of the most recognisable AI cadences in
# descriptive prose.
_EVAL_HEAD = (
    r"(?:[A-Za-z][\w']*"
    r"(?:ing|ed|y|ic|al|ful|less|ous|ish)"
    r"|"
    r"[Dd]ark|[Pp]ale|[Ss]oft|[Hh]ard|[Cc]old|[Ww]arm|[Ww]et|[Dd]ry|"
    r"[Tt]hin|[Tt]hick|[Ff]lat|[Ss]harp|[Bb]lunt|[Rr]aw|[Bb]are|[Ss]till|"
    r"[Qq]uiet|[Ll]oose|[Tt]ight|[Bb]roken|[Ww]hole|[Ss]low|[Ff]ast)"
)
TRIPLE_NOUN_PATTERN = re.compile(
    r"(?:^|(?<=[.!?\u201d\"])\s+|\n\s*)"
    + _EVAL_HEAD + r"(?:\s+[\w',]+){0,6}?,\s+"
    + _EVAL_HEAD + r"(?:\s+[\w',]+){0,6}?,\s+"
    + _EVAL_HEAD + r"(?:\s+[\w',]+){0,6}?[.!?]",
    re.MULTILINE,
)
# Chapter-specific tic — the "I named [it/that/the feeling]" construction.
# Not a global cap; the standing requirement asks the POV character to
# name a feeling per beat, and the drafter's failure mode is meta-narrating
# the act of naming rather than just naming. 2+ instances per chapter is
# fingerprint behavior. i_named is logged but does NOT trip hard_cap_pass.
# v27: broadened to catch the displaced "the thing in my chest" form.
I_NAMED_PATTERN = re.compile(
    r"\bI\s+named\s+"
    r"(?:that|it|this|the\s+feeling|all\s+(?:of\s+)?(?:them|those|three)|"
    r"the\s+(?:fear|grief|anger|wariness|fury|love|relief)|"
    r"the\s+thing\s+in\s+(?:my|her|his)\s+(?:chest|throat|stomach|belly|gut|head))"
    r"\b",
    re.IGNORECASE,
)
# v27 — Tautological loop. "X did the thing Xs do" / "her face did the
# thing faces do" / "her hand did the small thing hands do." Pure
# literary-AI cadence with no commercial-register twin. Direct flag.
TAUTOLOGY_PATTERN = re.compile(
    r"\b\w+\s+(?:did|does|do)\s+the\s+"
    r"(?:small\s+|big\s+|same\s+|usual\s+|exact\s+|right\s+|wrong\s+|"
    r"old\s+|new\s+|whole\s+|kind\s+of\s+)?"
    r"thing\s+\w+\s+(?:do|did|does)\b",
    re.IGNORECASE,
)
# v27 — Mid-sentence triple-appositive escalation. v26's TRIPLE_NOUN
# anchored at sentence start; the drafter routed around by putting the
# three-phrase escalation mid-sentence after a noun. Same fingerprint,
# new position. Pattern: ", phrase, phrase, phrase[end-punct]" where at
# least one phrase head carries evaluative weight.
MID_APPOSITIVE_PHRASE = r"[^,.!?;\n]{2,40}"
MID_APPOSITIVE_PATTERN = re.compile(
    r",\s+(" + MID_APPOSITIVE_PHRASE + r"),\s+"
    r"(" + MID_APPOSITIVE_PHRASE + r"),\s+"
    r"(" + MID_APPOSITIVE_PHRASE + r")(?=[.!?;])",
)
# v30 — Scene-set appositive. Declarative scene-set sentence followed by
# a comma and an interpretive coda. The coda explains, evaluates, or
# rephrases the first clause as a small aphorism. This is one of the
# heaviest single contributors to the red band in v29 chapters. The
# pattern is sentence-level — see _scan_scene_set_appositive — so the
# regex below only captures the marker phrases. Sentence-level filters
# (length, dialogue exclusion, opener exclusion) live in the scanner.
# Marker A: ", which was/is/were N more/fewer/larger/longer/etc. than ..."
# Marker B: ", and [Name/Title] had been ___ing ..."
# Marker C: "[,]? because ___ing ___ was a/an ..." (comma optional —
#           v29 produced both ", because" and "...because" forms, the
#           comma-less form was the dominant red opener pattern.)
# Marker D: ", because [a/the/this/that/saying it/crying it] [meta] ..."
SCENE_SET_APPOS_MARKER_A = re.compile(
    r",\s+which\s+(?:was|is|were|had)\s+\w+(?:\s+\w+){0,3}\s+(?:more|fewer|less|"
    r"larger|smaller|longer|shorter|further|closer|deeper|higher|lower)\b",
    re.IGNORECASE,
)
SCENE_SET_APPOS_MARKER_B = re.compile(
    r",\s+and\s+(?:[A-Z]\w*\.?\s+)?[A-Z]\w*\s+(?:had\s+been|was)\s+\w+ing\b",
)
# Marker C — meta-justifying because-clause. Commerical PNR uses because
# for actual causation ("I went because she called"); the AI cadence is
# because tied to a meta-statement that recasts the action as a category
# ("because sitting down was a commitment I wasn't prepared to make").
# We catch the meta form by requiring (gerund) + auxiliary + (article)
# + (noun) — that's a copular gerund-as-subject construction, the
# fingerprint of the cadence we're after.
SCENE_SET_APPOS_MARKER_C = re.compile(
    r",?\s+because\s+\w+ing\s+(?:\w+\s+){0,4}(?:was|is|were|had\s+been)\s+"
    r"(?:a|an|the)\s+\w+",
    re.IGNORECASE,
)
SCENE_SET_APPOS_MARKER_D = re.compile(
    r",\s+because\s+(?:that\s+is|that's|this\s+is|that\s+was|that\s+had|"
    r"saying\s+it|crying\s+(?:it|out)|hearing\s+(?:it|her|him))\s+",
    re.IGNORECASE,
)
# Marker E (v30) — "because [verb-ing] was [a thing] I [could/would/had to]"
# closes a Marker C variant where the noun is replaced with a relative
# clause: "because cleaning the mug was a thing I could do," "because
# tea was what my hands did." The structure is the same — gerund-as-
# subject plus meta-categorization — but the noun complement is
# elided to a "thing/what" + relative clause.
SCENE_SET_APPOS_MARKER_E = re.compile(
    r",?\s+because\s+\w+(?:\s+\w+){0,4}\s+(?:was|is|were)\s+(?:a\s+thing|"
    r"what|the\s+thing)\s+(?:\w+\s+){0,3}(?:\w+(?:'?s)?|I)\s+"
    r"(?:could|would|had\s+to|did|do|does)\b",
    re.IGNORECASE,
)
# v30 — Tautological restatement (v29-distinct). v29 catches the
# "X did the thing Xs do" form. v30 catches three additional forms:
#   Form 1 (cross-sentence noun-restate): "That was a real X. I could
#     [verb] a real X." — same noun phrase, second sentence restates.
#   Form 2 (within-sentence X-because-X-place): "The kit lived by the
#     door because that is where kits live." — subject restated as
#     plural nominal in a "where Xs live/do/are" tail.
#   Form 3 (triple self-affirmation fragments): three or more short
#     self-naming sentences in a row ("These are my hands. I'm Nell.
#     I'm scared.") — caught at the sentence-span level.
# The regex below catches Form 2 only; the cross-sentence and triple-
# fragment forms live in _scan_tautological_restatement.
TAUTOLOGICAL_PLACE_PATTERN = re.compile(
    r"\b(?:a|an|the)\s+(\w+)\s+(?:lived?|stood|sat|stayed|belonged|went|"
    r"slept|hung|sits?|lies?)\s+(?:by|in|at|near|on|over|under)\s+(?:a|an|the)\s+"
    r"\w+(?:\s+\w+){0,3}\s+because\s+that(?:'s|\s+is|\s+was)\s+where\s+\1s?\s+"
    r"(?:live|do|stay|sit|stand|belong|hang|lie|sleep|go)\b",
    re.IGNORECASE,
)
# v30 — Sensory-fragment pair. Two consecutive short sentences (4-12 words),
# both shaped as "[The/A] [object] [verb-of-state] [predicate]," neither
# containing first-person interior pronouns or active verbs of intention.
# Sentence-level scan; see _scan_sensory_fragment_pair.
SENSORY_FRAGMENT_OPENER = re.compile(
    r"^\s*(?:The|A|An)\s+\w+(?:\s+\w+){0,2}\s+"
    r"(?:had|was|were|sat|lay|stood|hung|smelled|smelt|felt|looked|tasted|"
    r"sounded|gave|came|went|stretched|stayed|spread|filled)\b",
)
INTERIOR_MARKERS = re.compile(
    r"\b(?:I|me|my|myself|we|our|us)\b",
    re.IGNORECASE,
)
# v30 — Procedural-action chain. Three or more comma-separated past-tense
# action verbs in series with first-person "I" subject and no interior
# commentary embedded. Examples:
#   "I shrugged into a jacket, took the keys, locked the deadbolt twice."
#   "I rebuilt the salt line. I put fresh water in both bowls. I pinned
#    the rowan twig back."
# Two forms: within-sentence comma-chain (regex below) and cross-sentence
# I-verb run (sentence-level scan; see _scan_procedural_chain).
PROCEDURAL_VERBS = (
    r"(?:shrugged|took|locked|grabbed|pulled|pushed|opened|closed|shut|"
    r"rebuilt|put|pinned|made|set|placed|drank|finished|checked|filled|"
    r"refreshed|changed|swept|wiped|tied|untied|zipped|unzipped|lifted|"
    r"dropped|carried|moved|turned|switched|started|stopped|left|"
    r"returned|walked|drove|rode|stepped|crossed|entered|exited)"
)
PROCEDURAL_CHAIN_PATTERN = re.compile(
    r"\bI\s+" + PROCEDURAL_VERBS + r"\b[^,.!?]{0,40},\s+"
    + PROCEDURAL_VERBS + r"\b[^,.!?]{0,40},\s+"
    + PROCEDURAL_VERBS + r"\b",
    re.IGNORECASE,
)
PROCEDURAL_I_VERB_OPENER = re.compile(
    r"^\s*I\s+" + PROCEDURAL_VERBS + r"\b",
    re.IGNORECASE,
)
# v31 — Evasion patterns observed in the v30 chapter. The v30 caps
# blocked the canonical forms ("X, which was Y" / "the kit lived by the
# door because that's where kits live") and the drafter routed around
# them with semantically equivalent shapes that no cap matched. Each
# v31 pattern targets one observed evasion. Patterns are deliberately
# narrow — broad pattern matching on these (e.g. any "felt like" or any
# "which was") false-positives on PNR-acceptable usage.
#
# v31-A — "felt like a/an [abstract noun]" (evades scene-set marker C).
# The bad form is "[main clause] because [gerund/clause] felt like a
# [abstract noun]" — the felt-like construction is the same meta-
# categorization as marker C's "was a commitment". Concrete-noun usage
# ("felt like a million bucks", "felt like a hot bath") is allowed by
# the abstract-noun whitelist.
FELT_LIKE_ABSTRACT = re.compile(
    r"\bfelt\s+like\s+(?:a|an)\s+(?:commitment|decision|mistake|lie|"
    r"choice|surrender|admission|confession|defeat|betrayal|warning|"
    r"failure|trap|punishment|rehearsal|reprieve|prayer|verdict|"
    r"promise|threat|test|loss|win|victory|gift|burden|reckoning)\b",
    re.IGNORECASE,
)
# v31-B — ", which was/is [meta-statement]" beyond the comparative form.
# Marker A in v30 required "more/fewer/larger/etc." after "which was";
# the v30 chapter produced four hits with non-comparative meta predicates
# ("which was the closest I had felt to whole", "which is how bodies
# are supposed to work", "which was without asking"). Restrict the
# predicate head to the marker words actually observed in AI cadence,
# so plain descriptive ", which was full" / "which was empty" don't
# false-positive.
WHICH_WAS_META = re.compile(
    r",\s+which\s+(?:was|is|were|had\s+been)\s+"
    r"(?:the\s+(?:closest|furthest|hardest|easiest|worst|best|least|most|"
    r"only|same|kind|sort|exact|whole)|"
    r"how\s+(?:bodies?|hands?|people|things?|life|death|love|grief|"
    r"this|that|it)\s+(?:are|is|do|does|work|works|goes|comes|gets)|"
    r"what\s+(?:happens|happened|she|he|it|they|we|I|you)|"
    r"without\s+\w+ing|"
    r"whatever\s+\w+|"
    r"the\s+part\s+of\s+\w+|"
    r"nothing\s+(?:I|you|he|she|we|they)\s+\w+)",
    re.IGNORECASE,
)
# v31-C — "but only because I [verb]" / "only because [pron] [verb]".
# Meta-because evasion in a different syntactic frame than marker C.
# Pattern: clause-final "but only because [pron] [verb]" — the "but
# only" prefix is the cadence marker; "because I [verb]" alone is
# permitted (legitimate causation in commercial PNR).
BUT_ONLY_BECAUSE = re.compile(
    r"\bbut\s+only\s+because\s+(?:I|she|he|they|we|you)\s+\w+",
    re.IGNORECASE,
)
# v31-D — "the [adj] kind, the kind that" — verdict-escalation variant.
# v26's VERDICT_KIND_OF_PATTERN catches "a particular kind of [adj]";
# v31 catches the appositive escalation form "the proper kind, the
# kind that takes the whole chest". The doubled "kind" is the marker.
KIND_THE_KIND = re.compile(
    r"\bthe\s+\w+\s+kind,\s+the\s+kind\s+(?:that|of|who)\b",
    re.IGNORECASE,
)
# v31-E — "X had gone cold. So had my Y." / "X was empty. So was Y."
# Sensory-fragment-pair variant the v30 scanner missed because the
# second sentence opens with "So [aux]" instead of "The/A [object]".
# Catches the inverted-aux flat-sensory pair specifically.
SO_AUX_INVERSION = re.compile(
    r"(?<=[.!?])\s+So\s+(?:had|was|were|did|is|are)\s+"
    r"(?:my|his|her|their|the|a|an)\s+\w+\.",
    re.IGNORECASE,
)
# v31-F — "the [observer-noun] [-ing verb]" periphrastic. The v29
# PERIPHRASTIC_PATTERN catches "as though he were", "like a woman who",
# "in the manner of", "as a man who". The v30 chapter produced "the
# person watching didn't have to see" — same observational-distance
# cadence in a new syntactic frame. Narrow to the specific
# observer-nouns + observing-verbs that AI cadence reaches for.
THE_OBSERVER_ING = re.compile(
    r"\bthe\s+(?:person|stranger|man|woman|people)\s+"
    r"(?:watching|listening|holding|waiting|standing|looking|"
    r"sitting|kneeling)\b",
    re.IGNORECASE,
)
# v31-G — "a/an [animate-noun] who had [cognition-verb]" — portrait
# construction with cognition verb. The v30 chapter had "a dog who had
# decided" — cognition verb on an animal is the AI-cadence marker (a
# dog "decides" is anthropomorphic abstraction; PNR uses concrete
# action). Narrow to specific cognition verbs.
WHO_HAD_DECIDED = re.compile(
    r"\b(?:a|an)\s+(?:dog|man|woman|person|stranger|child|kid|wolf|"
    r"girl|boy|cat|horse)\s+who\s+had\s+"
    r"(?:decided|known|seen|chosen|understood|realized|realised|"
    r"recognized|recognised|accepted|refused|forgiven|stopped)\b",
    re.IGNORECASE,
)
_EVAL_HEAD_RE = re.compile(_EVAL_HEAD)
# v27 — Aphoristic generalization templates. Used by the couplet detector
# to identify sentences that read as universal claims rather than scene
# beats. Hits a sentence if any template matches.
_APHORISM_TEMPLATES = [
    # "[X] people [universal verb]" — e.g. "Dying people get quiet"
    re.compile(
        r"^\s*[\"\u201c]?\s*\w+\s+people\s+"
        r"(?:get|are|don't|do\s+not|will|always|never|come|go|make|"
        r"see|hear|know|live|die|want|need|feel|think|tell|say|have|hate|love|fear)\b",
        re.IGNORECASE,
    ),
    # "It [sense-verb] like X" — e.g. "It looks like a tide."
    re.compile(
        r"^\s*[\"\u201c]?\s*It\s+(?:looks|sounds|feels|seems|tastes|smells)"
        r"\s+like\s+",
        re.IGNORECASE,
    ),
    # "People [universal verb]" — e.g. "People want to be told"
    re.compile(
        r"^\s*[\"\u201c]?\s*People\s+"
        r"(?:want|need|don't|do\s+not|always|never|feel|think|come|go|"
        r"are|will|have|hate|love|fear|tell|say|know)\b",
        re.IGNORECASE,
    ),
    # "the alternative is/was/to" — Cap 10 narrator-aphorism marker
    re.compile(r"\bthe\s+alternative\s+(?:is|was|to)\b", re.IGNORECASE),
    # "Every X is/has Y" — universal claim
    re.compile(r"^\s*[\"\u201c]?\s*Every\s+\w+\s+(?:is|has|wants|needs|gets|comes|goes)\b",
               re.IGNORECASE),
    # "There is/'s a X about Y" — sententious narrator opener
    re.compile(
        r"^\s*[\"\u201c]?\s*There(?:'s|\s+is|\s+was)\s+a(?:n)?\s+\w+\s+about\b",
        re.IGNORECASE,
    ),
    # "Both work the same" / "X works the same way" — verdict-aphorism
    re.compile(r"\b(?:both|all|either)\s+(?:work|do|are|come|go|feel)\s+the\s+same\b",
               re.IGNORECASE),
]
def _is_aphorism(sentence: str) -> bool:
    for pat in _APHORISM_TEMPLATES:
        if pat.search(sentence):
            return True
    return False
def _split_sentences_with_offsets(text: str) -> list:
    """Return a list of (start, end, sentence_text) tuples covering text.
    Uses the same sentence-end heuristic as the rest of the scanner."""
    spans = []
    cursor = 0
    for m in re.finditer(r"(?<=[.!?])\s+", text):
        end = m.start() + 1  # include the closing punctuation
        chunk = text[cursor:end].strip()
        if chunk:
            spans.append((cursor, end, chunk))
        cursor = m.end()
    if cursor < len(text):
        chunk = text[cursor:].strip()
        if chunk:
            spans.append((cursor, len(text), chunk))
    return spans
def _is_dialogue_sentence(sent_text: str, prefix: str) -> bool:
    """Heuristic: a sentence is dialogue if it starts with a quote mark
    or if an odd number of quote marks appears in everything before it.
    Both straight and curly quotes counted."""
    s = sent_text.lstrip()
    if s.startswith('"') or s.startswith('\u201c') or s.startswith('\u201d'):
        return True
    normalized = prefix.replace("\u201c", '"').replace("\u201d", '"')
    return normalized.count('"') % 2 == 1
def _word_count(s: str) -> int:
    return len(re.findall(r"\b[\w']+\b", s))
def _scan_staccato(text: str, sentence_spans: list) -> tuple:
    """v27 staccato detector.
    Definitions:
      - short sentence: ≤ 5 words after stripping (excluding dialogue)
      - run: any group of 1+ consecutive short sentences
      - cap: ≤ 1 run per 500 words
      - hard fail: any run of 3 or more consecutive shorts
    Dialogue sentences are excluded from short detection — character
    speech follows realism, not narrative-prose pacing.
    Returns (violations, total_runs, allowed).
    """
    short_indices = []
    for i, (start, end, s) in enumerate(sentence_spans):
        if _is_dialogue_sentence(s, text[:start]):
            continue
        wc = _word_count(s)
        if 1 <= wc <= 5:
            short_indices.append(i)
    # Group consecutive short-sentence indices into runs
    runs = []
    if short_indices:
        cur = [short_indices[0]]
        for idx in short_indices[1:]:
            if idx == cur[-1] + 1:
                cur.append(idx)
            else:
                runs.append(cur)
                cur = [idx]
        runs.append(cur)
    total_words = _word_count(text)
    allowed = max(1, total_words // 500)
    violations = []
    flagged_run_signatures = set()
    # Triplet+ runs always flagged (hard fail per user spec)
    for run in runs:
        if len(run) >= 3:
            sig = (run[0], run[-1])
            flagged_run_signatures.add(sig)
            first_start = sentence_spans[run[0]][0]
            last_end = sentence_spans[run[-1]][1]
            verbatim = text[first_start:last_end].strip()
            violations.append({
                "rule": "staccato_triplet",
                "verbatim": verbatim,
                "context": verbatim[:200],
                "indices": list(run),
            })
    # Density overflow — total runs exceeds the per-500-words cap.
    # The densest unflagged runs are flagged as overflow until count
    # drops to allowed.
    if len(runs) > allowed:
        overflow_count = len(runs) - allowed
        candidates = [
            r for r in runs
            if (r[0], r[-1]) not in flagged_run_signatures
        ]
        candidates.sort(key=lambda r: -len(r))
        for run in candidates[:overflow_count]:
            first_start = sentence_spans[run[0]][0]
            last_end = sentence_spans[run[-1]][1]
            verbatim = text[first_start:last_end].strip()
            violations.append({
                "rule": "staccato_density",
                "verbatim": verbatim,
                "context": verbatim[:200],
                "indices": list(run),
            })
    return violations, len(runs), allowed
def _scan_polysyndetic(sentence_spans: list, text: str) -> list:
    """v27 polysyndetic detector — single-sentence run-on with 4+
    coordinating 'and' joins. Threshold per user spec: 4+. Excludes
    'and's inside dialogue (we count the whole sentence's 'and's; the
    threshold is high enough that legitimate dialogue rarely trips it,
    but dialogue sentences are skipped explicitly)."""
    violations = []
    for i, (start, end, s) in enumerate(sentence_spans):
        if _is_dialogue_sentence(s, text[:start]):
            continue
        and_count = len(re.findall(r"\band\b", s, re.IGNORECASE))
        if and_count >= 4:
            violations.append({
                "rule": "polysyndetic",
                "context": s[:200],
                "indices": [i],
            })
    return violations
def _scan_aphoristic_couplets(sentence_spans: list, text: str) -> list:
    """v27 aphoristic-couplet detector — flag any 3-sentence sliding
    window containing 2+ generalization-template matches. Each matching
    sentence is flagged once even if it appears in multiple windows.
    Dialogue sentences excluded — characters can be sententious in their
    own speech without it being narrator aphorism."""
    n = len(sentence_spans)
    is_aph = []
    for i, (start, end, s) in enumerate(sentence_spans):
        if _is_dialogue_sentence(s, text[:start]):
            is_aph.append(False)
        else:
            is_aph.append(_is_aphorism(s))
    flagged = set()
    for i in range(n):
        window_end = min(i + 3, n)
        if sum(is_aph[i:window_end]) >= 2:
            for j in range(i, window_end):
                if is_aph[j]:
                    flagged.add(j)
    violations = []
    for j in sorted(flagged):
        s = sentence_spans[j][2]
        violations.append({
            "rule": "aphoristic_couplet",
            "context": s[:200],
            "indices": [j],
        })
    return violations
def _scan_anaphoric_escalation(sentence_spans: list) -> list:
    """v27 anaphoric-escalation detector — 3+ sentences within any
    5-sentence window sharing the same opening 1-2 words. Catches the
    'It looks like X. It looks like Y.' / 'already in to her X, already
    in to her Y' / 'Salt at the X, salt at the Y' cadence."""
    n = len(sentence_spans)
    if n < 3:
        return []
    openers = []
    for (_, _, s) in sentence_spans:
        cleaned = s.lstrip().lstrip('"\u201c\u201d').lstrip()
        words = re.findall(r"\b[\w']+\b", cleaned)
        if len(words) >= 2:
            key = (words[0].lower(), words[1].lower())
        elif words:
            key = (words[0].lower(),)
        else:
            key = ()
        openers.append(key)
    flagged_groups = []
    seen_signatures = set()
    for i in range(n):
        end = min(i + 5, n)
        window_keys = openers[i:end]
        # Find any 2-word opener occurring 3+ times in this window
        from collections import Counter
        counts = Counter(k for k in window_keys if len(k) == 2)
        for key, count in counts.items():
            if count >= 3:
                indices = [j for j in range(i, end) if openers[j] == key]
                if len(indices) >= 3:
                    sig = tuple(indices)
                    if sig not in seen_signatures:
                        seen_signatures.add(sig)
                        flagged_groups.append((indices, key))
    violations = []
    for indices, key in flagged_groups:
        examples = " | ".join(sentence_spans[j][2][:60] for j in indices[:3])
        violations.append({
            "rule": "anaphoric_escalation",
            "context": f"opener='{' '.join(key)}': {examples}",
            "indices": indices,
        })
    return violations
def _scan_mid_appositive(text: str) -> list:
    """v27 mid-sentence triple-appositive detector. Three comma-separated
    phrases mid-sentence where at least one phrase begins with an
    evaluative head (suffix or bare-adjective). Catches the 'a steady,
    working sadness, the same one I get every time, undimmed by
    repetition' cadence that displaced from v26's TRIPLE_NOUN."""
    violations = []
    for m in MID_APPOSITIVE_PATTERN.finditer(text):
        phrases = [m.group(1).strip(), m.group(2).strip(), m.group(3).strip()]
        eval_count = 0
        for p in phrases:
            words = p.split()
            if not words:
                continue
            head = words[0]
            if _EVAL_HEAD_RE.match(head):
                eval_count += 1
        if eval_count >= 1:
            start = max(0, m.start() - 30)
            end = min(len(text), m.end() + 10)
            violations.append({
                "rule": "mid_appositive_triple",
                "context": text[start:end].replace("\n", " ").strip(),
            })
    return violations
def _scan_scene_set_appositive(sentence_spans: list, text: str) -> list:
    """v30 — Scene-set appositive detector.
    Flags sentences that pair a declarative scene-set clause with an
    interpretive coda. Catches the dominant red-band cadence in v29
    chapters: openings and transitions where the model wraps a fact in
    a small aphorism. Sentence-level filters keep dialogue and direct
    first-person interior beats out of the catch-net.
    Examples flagged:
      "There were six squirrels lined up at the tree line at 7:42 in
       the morning, which was four more than the tree line had any
       business holding."  (marker A)
      "The Halloran farmhouse had been falling down politely since 1962,
       and Mrs. Halloran had been holding the porch up with paint and
       stubbornness."  (marker B)
      "I drank cold tea standing at the kitchen window because sitting
       down at this hour was a commitment I wasn't prepared to make."
       (marker C — no-comma form)
      "I made another cup of tea, because tea was what my hands did,
       and I set the mug on the counter and didn't drink it."
       (marker E)
    """
    violations = []
    # Markers that REQUIRE a leading comma (they target the appositive
    # cadence specifically). Markers that accept either form get a
    # broader treatment.
    COMMA_REQUIRED = {"scene_set_which_was", "scene_set_and_was", "scene_set_because_meta"}
    COMMA_OPTIONAL = {"scene_set_because_ing", "scene_set_because_thing"}
    for span_start, span_end, sent_text in sentence_spans:
        # Skip dialogue
        prefix = text[:span_start]
        if _is_dialogue_sentence(sent_text, prefix):
            continue
        # Length floor: this construction needs a real clause + coda
        if _word_count(sent_text) < 18:
            continue
        # For comma-required markers (A, B, D) the first comma must come
        # after at least 8 words. For comma-optional markers (C, E) the
        # construction can attach to a short main clause, so we only
        # require the head clause (everything before the marker) to be
        # at least 5 words.
        marker = None
        marker_idx = -1
        for label, pat in [
            ("scene_set_which_was", SCENE_SET_APPOS_MARKER_A),
            ("scene_set_and_was", SCENE_SET_APPOS_MARKER_B),
            ("scene_set_because_ing", SCENE_SET_APPOS_MARKER_C),
            ("scene_set_because_meta", SCENE_SET_APPOS_MARKER_D),
            ("scene_set_because_thing", SCENE_SET_APPOS_MARKER_E),
        ]:
            m = pat.search(sent_text)
            if m:
                marker = label
                marker_idx = m.start()
                break
        if not marker:
            continue
        head_clause = sent_text[:marker_idx].strip().rstrip(",")
        head_wc = _word_count(head_clause)
        if marker in COMMA_REQUIRED:
            # Require a comma at the marker join point
            join_segment = sent_text[max(0, marker_idx - 2): marker_idx + 1]
            if "," not in join_segment:
                continue
            if head_wc < 8:
                continue
        elif marker in COMMA_OPTIONAL:
            # Comma is optional; head clause must be at least 5 words
            if head_wc < 5:
                continue
        violations.append({
            "rule": "scene_set_appositive",
            "marker": marker,
            "context": sent_text.replace("\n", " ").strip()[:280],
            "verbatim": sent_text,
        })
    return violations
def _scan_tautological_restatement(sentence_spans: list, text: str) -> list:
    """v30 — Tautological restatement (v29-distinct) detector.
    Catches three forms not covered by v27 TAUTOLOGY_PATTERN:
      Form 1: cross-sentence noun-restate. Two consecutive sentences
              ≤ 12 words each, sharing the same head noun phrase, with
              the second sentence restating rather than advancing.
              "That was a real edge. I could hold onto a real edge."
      Form 2: within-sentence X-because-X-place. The kit lived by the
              door because that is where kits live.
      Form 3: triple self-affirmation fragments. Three or more short
              sentences in a row where the subject is "I" or a self-
              referential demonstrative, each ≤ 7 words.
              "These are my hands. I'm Nell. I'm scared."
    """
    violations = []
    # Form 2: within-sentence regex
    for m in TAUTOLOGICAL_PLACE_PATTERN.finditer(text):
        # Find the containing sentence for verbatim
        containing = ""
        for s, e, t in sentence_spans:
            if s <= m.start() < e:
                containing = t
                break
        start = max(0, m.start() - 20)
        end = min(len(text), m.end() + 20)
        violations.append({
            "rule": "tautological_restatement",
            "form": "within_sentence_place",
            "context": text[start:end].replace("\n", " ").strip(),
            "verbatim": containing or text[start:end].strip(),
        })
    # Form 1: cross-sentence noun-restate
    n = len(sentence_spans)
    for i in range(n - 1):
        a_text = sentence_spans[i][2]
        b_text = sentence_spans[i + 1][2]
        # Both sentences short
        if _word_count(a_text) > 12 or _word_count(b_text) > 12:
            continue
        # Skip dialogue
        if _is_dialogue_sentence(a_text, text[:sentence_spans[i][0]]):
            continue
        if _is_dialogue_sentence(b_text, text[:sentence_spans[i + 1][0]]):
            continue
        # Find an "a/the [adj]? [noun]" head in a; check for the same
        # noun in b. Require a content noun (≥ 4 chars, not a pronoun).
        a_head = re.search(
            r"\b(?:a|an|the)\s+(?:\w+\s+){0,2}(\w{4,})\b",
            a_text, re.IGNORECASE,
        )
        if not a_head:
            continue
        noun = a_head.group(1).lower()
        if noun in {"thing", "kind", "sort", "place", "time", "moment",
                    "minute", "second", "morning", "evening", "night"}:
            continue
        b_match = re.search(
            r"\b(?:a|an|the)\s+(?:\w+\s+){0,2}" + re.escape(noun) + r"\b",
            b_text, re.IGNORECASE,
        )
        if not b_match:
            continue
        violations.append({
            "rule": "tautological_restatement",
            "form": "cross_sentence_noun_restate",
            "context": (a_text + " " + b_text).replace("\n", " ").strip()[:240],
            "verbatim": a_text + " " + b_text,
        })
    # Form 3: triple self-affirmation fragments — three short
    # self-referential sentences in a row.
    SELF_OPEN = re.compile(
        r"^\s*[\"\u201c]?(I\s|I'?m\s|I'?ve\s|These\s+are\s+|This\s+is\s+|That\s+is\s+|That\s+was\s+)",
        re.IGNORECASE,
    )
    i = 0
    while i < n - 2:
        run_len = 0
        for k in range(i, n):
            t = sentence_spans[k][2]
            if _word_count(t) > 7:
                break
            if not SELF_OPEN.match(t):
                break
            if _is_dialogue_sentence(t, text[:sentence_spans[k][0]]):
                break
            run_len += 1
        if run_len >= 3:
            run_text = " ".join(
                sentence_spans[k][2] for k in range(i, i + run_len)
            )
            violations.append({
                "rule": "tautological_restatement",
                "form": "triple_self_affirmation",
                "context": run_text.replace("\n", " ").strip()[:240],
                "verbatim": run_text,
            })
            i += run_len
        else:
            i += 1
    return violations
def _scan_sensory_fragment_pair(sentence_spans: list, text: str) -> list:
    """v30 — Sensory-fragment pair detector.
    Flags two consecutive short sentences (4-14 words each), both shaped
    as "[The/A] [object] [verb-of-state] [predicate]," neither having
    "I/My/We" as the main clause subject. Reads as flat sensory
    observation with no interior pressure between beats.
    Examples flagged:
      "The sink had a drip. The kettle had a hairline crack I had been
       ignoring for eight months."
      "The truck started on the second turn. The back roads inland were
       narrow and wet and yellow with leaves."
    The "I" in "a hairline crack I had been ignoring" sits in a relative
    clause, not as the main subject — the sentence is still a flat
    sensory observation. We allow that. We reject sentences where I/My
    starts the sentence (those are interior beats, not sensory fragments).
    """
    violations = []
    n = len(sentence_spans)
    SUBJECT_INTERIOR = re.compile(r"^\s*(?:I|My|We|Our)\b", re.IGNORECASE)
    for i in range(n - 1):
        a_text = sentence_spans[i][2]
        b_text = sentence_spans[i + 1][2]
        # Length floor: 4-14 words each (slightly wider than v30 first cut)
        a_wc = _word_count(a_text)
        b_wc = _word_count(b_text)
        if a_wc < 4 or a_wc > 14:
            continue
        if b_wc < 4 or b_wc > 14:
            continue
        # Skip dialogue
        if _is_dialogue_sentence(a_text, text[:sentence_spans[i][0]]):
            continue
        if _is_dialogue_sentence(b_text, text[:sentence_spans[i + 1][0]]):
            continue
        # Reject if either sentence starts with first-person — that's an
        # interior beat, not a sensory fragment.
        if SUBJECT_INTERIOR.match(a_text):
            continue
        if SUBJECT_INTERIOR.match(b_text):
            continue
        # Both must match the sensory-fragment opener
        if not SENSORY_FRAGMENT_OPENER.search(a_text):
            continue
        if not SENSORY_FRAGMENT_OPENER.search(b_text):
            continue
        violations.append({
            "rule": "sensory_fragment_pair",
            "context": (a_text + " " + b_text).replace("\n", " ").strip()[:240],
            "verbatim": a_text + " " + b_text,
        })
    return violations
def _scan_procedural_chain(sentence_spans: list, text: str) -> list:
    """v30 — Procedural-action chain detector.
    Two forms:
      Form 1 (within-sentence): three or more past-tense procedural
        verbs in series with first-person "I" subject and no interior
        commentary embedded.
        "I shrugged into a jacket, took the keys, locked the deadbolt
         twice."
      Form 2 (cross-sentence run): three or more consecutive sentences
        each opening with "I [procedural verb]" and ending without
        interior commentary.
        "I rebuilt the salt line at the kitchen window where the morning
         had thinned it. I put fresh water in both bowls. I pinned the
         rowan twig back where it had slipped a quarter inch off true."
    """
    violations = []
    # Form 1: within-sentence chain via regex
    for m in PROCEDURAL_CHAIN_PATTERN.finditer(text):
        # Gate on dialogue
        before = text[: m.start()]
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 == 1:
            continue
        # Find the containing sentence for verbatim
        containing = ""
        for s, e, t in sentence_spans:
            if s <= m.start() < e:
                containing = t
                break
        start = max(0, m.start() - 20)
        end = min(len(text), m.end() + 40)
        violations.append({
            "rule": "procedural_chain",
            "form": "within_sentence",
            "context": text[start:end].replace("\n", " ").strip(),
            "verbatim": containing or text[start:end].strip(),
        })
    # Form 2: cross-sentence I-verb run (3 or more)
    n = len(sentence_spans)
    i = 0
    while i < n - 2:
        run_len = 0
        for k in range(i, n):
            t = sentence_spans[k][2]
            if not PROCEDURAL_I_VERB_OPENER.search(t):
                break
            if _is_dialogue_sentence(t, text[:sentence_spans[k][0]]):
                break
            run_len += 1
        if run_len >= 3:
            run_text = " ".join(
                sentence_spans[k][2] for k in range(i, i + run_len)
            )
            violations.append({
                "rule": "procedural_chain",
                "form": "cross_sentence_run",
                "context": run_text.replace("\n", " ").strip()[:300],
                "verbatim": run_text,
            })
            i += run_len
        else:
            i += 1
    return violations
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
    # Aphoristic standalone — abstract subject + verdict verb
    aphoristic_matches = list(APHORISTIC_STANDALONE_PATTERN.finditer(text))
    for m in aphoristic_matches[:15]:
        start = max(0, m.start() - 20)
        end = min(len(text), m.end() + 80)
        flagged.append({
            "rule": "aphoristic_standalone",
            "context": text[start:end].replace("\n", " ").strip(),
        })
    # Explanatory "because I had" backfill in interior
    backfill_matches = list(EXPLANATORY_BACKFILL_PATTERN.finditer(text))
    for m in backfill_matches[:15]:
        start = max(0, m.start() - 60)
        end = min(len(text), m.end() + 60)
        flagged.append({
            "rule": "explanatory_backfill",
            "context": text[start:end].replace("\n", " ").strip(),
        })
    # "X too Y for Z" verdict construction — skip if inside quotes (dialogue)
    verdict_matches = []
    for m in VERDICT_TOO_FOR_PATTERN.finditer(text):
        before = text[: m.start()]
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 == 0:  # even → outside dialogue
            verdict_matches.append(m)
    for m in verdict_matches[:15]:
        start = max(0, m.start() - 40)
        end = min(len(text), m.end() + 40)
        flagged.append({
            "rule": "verdict_too_for",
            "context": text[start:end].replace("\n", " ").strip(),
        })
    # v26: Cap 3 bridge variants ("Not X. Y." / "Not X; Y.")
    not_bridge_matches = list(NOT_BRIDGE_PATTERN.finditer(text))
    for m in not_bridge_matches[:15]:
        start = max(0, m.start() - 20)
        end = min(len(text), m.end() + 80)
        flagged.append({
            "rule": "not_bridge",
            "context": text[start:end].replace("\n", " ").strip(),
        })
    # v26: Verdict "kind of [adj]" construction — same dialogue filter as
    # VERDICT_TOO_FOR (characters may use it in spoken lines).
    kind_of_matches = []
    for m in VERDICT_KIND_OF_PATTERN.finditer(text):
        before = text[: m.start()]
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 == 0:
            kind_of_matches.append(m)
    for m in kind_of_matches[:15]:
        start = max(0, m.start() - 20)
        end = min(len(text), m.end() + 80)
        flagged.append({
            "rule": "verdict_kind_of",
            "context": text[start:end].replace("\n", " ").strip(),
        })
    # v26: Cap 11 triple-noun-phrase escalation
    triple_noun_matches = list(TRIPLE_NOUN_PATTERN.finditer(text))
    for m in triple_noun_matches[:15]:
        start = max(0, m.start() - 20)
        end = min(len(text), m.end() + 120)
        flagged.append({
            "rule": "triple_noun",
            "context": text[start:end].replace("\n", " ").strip(),
        })
    # v26: "I named [it/that/the feeling]" tic. Logged when count >= 2.
    # Not gated on hard_cap_pass — diagnostic and graft-target only.
    i_named_matches = list(I_NAMED_PATTERN.finditer(text))
    if len(i_named_matches) >= 2:
        for m in i_named_matches[:15]:
            start = max(0, m.start() - 10)
            end = min(len(text), m.end() + 80)
            flagged.append({
                "rule": "i_named_tic",
                "context": text[start:end].replace("\n", " ").strip(),
            })
    # v27: sentence-level scans. Build the sentence span list once and
    # share it across the staccato/polysyndetic/couplet/anaphora detectors.
    _v27_sentence_spans = _split_sentences_with_offsets(text)
    # v27: staccato density (≤1 run per 500 words; triplet+ always fails)
    staccato_violations, staccato_total_runs, staccato_allowed = _scan_staccato(
        text, _v27_sentence_spans
    )
    flagged.extend(staccato_violations[:25])
    # v27: polysyndetic single-sentence run-on (4+ coordinating 'and's)
    polysyndetic_violations = _scan_polysyndetic(_v27_sentence_spans, text)
    flagged.extend(polysyndetic_violations[:15])
    # v27: aphoristic couplet (2+ generalizations in any 3-sentence window)
    couplet_violations = _scan_aphoristic_couplets(_v27_sentence_spans, text)
    flagged.extend(couplet_violations[:15])
    # v27: mid-sentence triple appositive
    mid_appositive_violations = _scan_mid_appositive(text)
    flagged.extend(mid_appositive_violations[:15])
    # v27: tautological loop ("X did the thing Xs do")
    tautology_matches = list(TAUTOLOGY_PATTERN.finditer(text))
    for m in tautology_matches[:15]:
        start = max(0, m.start() - 20)
        end = min(len(text), m.end() + 40)
        flagged.append({
            "rule": "tautology",
            "context": text[start:end].replace("\n", " ").strip(),
        })
    # v27: anaphoric escalation (3+ same-opener sentences in 5-sent window)
    anaphoric_violations = _scan_anaphoric_escalation(_v27_sentence_spans)
    flagged.extend(anaphoric_violations[:15])
    # v30: scene-set appositive ("X, which was Y" / ", and Mrs. X had been ___ing"
    # / ", because ___ing was a [thing]"). Sentence-level scan; skips dialogue.
    scene_set_appos_violations = _scan_scene_set_appositive(
        _v27_sentence_spans, text,
    )
    flagged.extend(scene_set_appos_violations[:15])
    # v30: tautological restatement (cross-sentence noun-restate, X-because-X-place,
    # triple self-affirmation). v29-distinct from TAUTOLOGY_PATTERN.
    tautological_restate_violations = _scan_tautological_restatement(
        _v27_sentence_spans, text,
    )
    flagged.extend(tautological_restate_violations[:15])
    # v30: sensory-fragment pair (two flat sensory sentences in series with no
    # interior pressure).
    sensory_pair_violations = _scan_sensory_fragment_pair(
        _v27_sentence_spans, text,
    )
    flagged.extend(sensory_pair_violations[:15])
    # v30: procedural-action chain (I-verb chain ≥ 3 with no interior commentary).
    procedural_chain_violations = _scan_procedural_chain(
        _v27_sentence_spans, text,
    )
    flagged.extend(procedural_chain_violations[:15])
    # v31: evasion patterns observed in v30 chapter. Each pattern targets
    # one specific evasion the drafter used to route around v30 caps.
    felt_like_matches = list(FELT_LIKE_ABSTRACT.finditer(text))
    for m in felt_like_matches[:15]:
        # Skip dialogue
        before = text[: m.start()]
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 == 1:
            continue
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 30)
        flagged.append({
            "rule": "felt_like_abstract",
            "context": text[start:end].replace("\n", " ").strip(),
        })
    # Filter felt_like_matches to outside-dialogue only for the count
    felt_like_count = sum(
        1 for m in felt_like_matches
        if (text[:m.start()].replace("\u201c", '"').replace("\u201d", '"')).count('"') % 2 == 0
    )
    which_was_meta_matches = list(WHICH_WAS_META.finditer(text))
    for m in which_was_meta_matches[:15]:
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 60)
        flagged.append({
            "rule": "which_was_meta",
            "context": text[start:end].replace("\n", " ").strip(),
        })
    but_only_because_matches = list(BUT_ONLY_BECAUSE.finditer(text))
    for m in but_only_because_matches[:15]:
        before = text[: m.start()]
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 == 1:
            continue
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 40)
        flagged.append({
            "rule": "but_only_because",
            "context": text[start:end].replace("\n", " ").strip(),
        })
    but_only_because_count = sum(
        1 for m in but_only_because_matches
        if (text[:m.start()].replace("\u201c", '"').replace("\u201d", '"')).count('"') % 2 == 0
    )
    kind_the_kind_matches = list(KIND_THE_KIND.finditer(text))
    for m in kind_the_kind_matches[:15]:
        before = text[: m.start()]
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 == 1:
            continue
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 40)
        flagged.append({
            "rule": "kind_the_kind",
            "context": text[start:end].replace("\n", " ").strip(),
        })
    kind_the_kind_count = sum(
        1 for m in kind_the_kind_matches
        if (text[:m.start()].replace("\u201c", '"').replace("\u201d", '"')).count('"') % 2 == 0
    )
    so_aux_matches = list(SO_AUX_INVERSION.finditer(text))
    for m in so_aux_matches[:15]:
        # Find the sentence-pair this inversion completes (prior sentence)
        ctx_start = max(0, m.start() - 80)
        ctx_end = min(len(text), m.end() + 10)
        flagged.append({
            "rule": "so_aux_inversion",
            "context": text[ctx_start:ctx_end].replace("\n", " ").strip(),
            "verbatim": text[ctx_start:ctx_end].strip(),
        })
    observer_ing_matches = list(THE_OBSERVER_ING.finditer(text))
    for m in observer_ing_matches[:15]:
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 40)
        flagged.append({
            "rule": "the_observer_ing",
            "context": text[start:end].replace("\n", " ").strip(),
        })
    who_had_decided_matches = list(WHO_HAD_DECIDED.finditer(text))
    for m in who_had_decided_matches[:15]:
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 40)
        flagged.append({
            "rule": "who_had_decided",
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
        and len(aphoristic_matches) == 0
        and len(backfill_matches) == 0
        and len(verdict_matches) == 0
        and len(not_bridge_matches) == 0
        and len(kind_of_matches) == 0
        and len(triple_noun_matches) == 0
        # i_named_tic intentionally excluded — the standing requirement
        # legitimately asks for feeling-naming. Logged for grafter and
        # line-edit pass to use as a graft target when count is high.
        # v27 hard caps
        and len(staccato_violations) == 0
        and len(polysyndetic_violations) == 0
        and len(couplet_violations) == 0
        and len(mid_appositive_violations) == 0
        and len(tautology_matches) == 0
        and len(anaphoric_violations) == 0
        # v30 hard caps
        and len(scene_set_appos_violations) == 0
        and len(tautological_restate_violations) == 0
        and len(sensory_pair_violations) == 0
        and len(procedural_chain_violations) == 0
        # v31 hard caps — evasions from the v30 chapter
        and felt_like_count == 0
        and len(which_was_meta_matches) == 0
        and but_only_because_count == 0
        and kind_the_kind_count == 0
        and len(so_aux_matches) == 0
        and len(observer_ing_matches) == 0
        and len(who_had_decided_matches) == 0
    )
    return {
        "scan_the_way_count": len(the_way_matches),
        "scan_periphrastic_count": len(periphrastic_matches),
        "scan_not_but_count": len(not_but_matches),
        "scan_em_dash_count": em_dash_count,
        "scan_em_dash_per_1k": em_per_1k,
        "scan_emotion_naming_count": len(emotion_matches),
        "scan_aphoristic_count": len(aphoristic_matches),
        "scan_backfill_count": len(backfill_matches),
        "scan_verdict_count": len(verdict_matches),
        "scan_not_bridge_count": len(not_bridge_matches),
        "scan_verdict_kind_of_count": len(kind_of_matches),
        "scan_triple_noun_count": len(triple_noun_matches),
        "scan_i_named_count": len(i_named_matches),
        "scan_staccato_count": len(staccato_violations),
        "scan_staccato_total_runs": staccato_total_runs,
        "scan_staccato_allowed": staccato_allowed,
        "scan_polysyndetic_count": len(polysyndetic_violations),
        "scan_aphoristic_couplet_count": len(couplet_violations),
        "scan_mid_appositive_count": len(mid_appositive_violations),
        "scan_tautology_count": len(tautology_matches),
        "scan_anaphoric_escalation_count": len(anaphoric_violations),
        "scan_scene_set_appos_count": len(scene_set_appos_violations),
        "scan_tautological_restate_count": len(tautological_restate_violations),
        "scan_sensory_pair_count": len(sensory_pair_violations),
        "scan_procedural_chain_count": len(procedural_chain_violations),
        "scan_felt_like_abstract_count": felt_like_count,
        "scan_which_was_meta_count": len(which_was_meta_matches),
        "scan_but_only_because_count": but_only_because_count,
        "scan_kind_the_kind_count": kind_the_kind_count,
        "scan_so_aux_inversion_count": len(so_aux_matches),
        "scan_observer_ing_count": len(observer_ing_matches),
        "scan_who_had_decided_count": len(who_had_decided_matches),
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
    if scan.get("scan_aphoristic_count", 0):
        flags.append(f"aphoristic×{scan['scan_aphoristic_count']}")
    if scan.get("scan_backfill_count", 0):
        flags.append(f"backfill×{scan['scan_backfill_count']}")
    if scan.get("scan_verdict_count", 0):
        flags.append(f"verdict×{scan['scan_verdict_count']}")
    if scan.get("scan_not_bridge_count", 0):
        flags.append(f"not-bridge×{scan['scan_not_bridge_count']}")
    if scan.get("scan_verdict_kind_of_count", 0):
        flags.append(f"kind-of×{scan['scan_verdict_kind_of_count']}")
    if scan.get("scan_triple_noun_count", 0):
        flags.append(f"triple-noun×{scan['scan_triple_noun_count']}")
    if scan.get("scan_i_named_count", 0) >= 2:
        flags.append(f"i-named×{scan['scan_i_named_count']}")
    if scan.get("scan_staccato_count", 0):
        flags.append(f"staccato×{scan['scan_staccato_count']}")
    if scan.get("scan_polysyndetic_count", 0):
        flags.append(f"polysyndetic×{scan['scan_polysyndetic_count']}")
    if scan.get("scan_aphoristic_couplet_count", 0):
        flags.append(f"couplet×{scan['scan_aphoristic_couplet_count']}")
    if scan.get("scan_mid_appositive_count", 0):
        flags.append(f"mid-apposit×{scan['scan_mid_appositive_count']}")
    if scan.get("scan_tautology_count", 0):
        flags.append(f"tautology×{scan['scan_tautology_count']}")
    if scan.get("scan_anaphoric_escalation_count", 0):
        flags.append(f"anaphora×{scan['scan_anaphoric_escalation_count']}")
    if scan.get("scan_scene_set_appos_count", 0):
        flags.append(f"scene-appos×{scan['scan_scene_set_appos_count']}")
    if scan.get("scan_tautological_restate_count", 0):
        flags.append(f"taut-restate×{scan['scan_tautological_restate_count']}")
    if scan.get("scan_sensory_pair_count", 0):
        flags.append(f"sensory-pair×{scan['scan_sensory_pair_count']}")
    if scan.get("scan_procedural_chain_count", 0):
        flags.append(f"proc-chain×{scan['scan_procedural_chain_count']}")
    if scan.get("scan_felt_like_abstract_count", 0):
        flags.append(f"felt-like×{scan['scan_felt_like_abstract_count']}")
    if scan.get("scan_which_was_meta_count", 0):
        flags.append(f"which-meta×{scan['scan_which_was_meta_count']}")
    if scan.get("scan_but_only_because_count", 0):
        flags.append(f"but-only×{scan['scan_but_only_because_count']}")
    if scan.get("scan_kind_the_kind_count", 0):
        flags.append(f"kind-kind×{scan['scan_kind_the_kind_count']}")
    if scan.get("scan_so_aux_inversion_count", 0):
        flags.append(f"so-aux×{scan['scan_so_aux_inversion_count']}")
    if scan.get("scan_observer_ing_count", 0):
        flags.append(f"observer-ing×{scan['scan_observer_ing_count']}")
    if scan.get("scan_who_had_decided_count", 0):
        flags.append(f"who-decided×{scan['scan_who_had_decided_count']}")
    status = "PASS" if scan["scan_hard_cap_pass"] else "FAIL"
    return f"{status} ({', '.join(flags) if flags else 'clean'})"
# ============================================================================
# Stage F — ridge predictor of Originality human-score (deterministic, no LLM)
#
# Fits a compact ridge regression against labeled_corpus.json (doc text +
# known Originality human-score). The feature set was reworked against the
# supplied corpus to favor document-level movement and punctuation rhythm
# over the earlier broad 15-feature bundle.
#
# The predictor is advisory. It does not gate shipping. The goal is to
# surface a predicted Originality score in the batch summary so you can skip
# the manual Originality submission step when the prediction is clearly in
# or out of band.
# ============================================================================
# Six corpus-fit structural features:
#   1  sentence-length standard deviation
#   2  word count
#   3  semicolon rate per 1k words
#   4  em-dash rate per 1k words
#   5  periphrastic-observational rate per 1k words
#   6  mean commas per sentence
#
# On the uploaded labeled_corpus.json this revision materially outperformed
# the previous 15-feature Stage F ridge during leave-one-out testing.
STAGE_F_FEATURE_NAMES = [
    "sent_len_std",
    "word_count",
    "semicolon_per_1k",
    "em_dash_per_1k",
    "periphrastic_per_1k",
    "mean_commas_per_sentence",
]
def _stage_f_sentence_texts(text: str) -> list:
    """Sentence splitter aligned with scan_draft's heuristic."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
def _stage_f_sentence_lengths(text: str) -> list:
    sents = _stage_f_sentence_texts(text)
    return [len(re.findall(r"\b[\w']+\b", s)) for s in sents]
def _stage_f_periphrastic_count(text: str) -> int:
    return len(PERIPHRASTIC_PATTERN.findall(text))
def stage_f_extract_features(text: str) -> np.ndarray:
    """Return the Stage F feature vector for a draft.
    These features were chosen because they improved leave-one-out accuracy
    on the supplied labeled corpus while remaining cheap to compute at run
    time and independent of any API call.
    """
    words = re.findall(r"\b[\w']+\b", text.lower())
    wc = max(len(words), 1)
    sents = _stage_f_sentence_texts(text)
    sent_lengths = _stage_f_sentence_lengths(text) or [0]
    sent_arr = np.asarray(sent_lengths, dtype=float)
    mean_commas = float(np.mean([s.count(",") for s in sents] or [0]))
    semicolons_per_1k = text.count(";") / wc * 1000.0
    em_dashes_per_1k = text.count("\u2014") / wc * 1000.0
    periphrastic_per_1k = _stage_f_periphrastic_count(text) / wc * 1000.0
    feats = [
        float(sent_arr.std()),
        float(wc),
        float(semicolons_per_1k),
        float(em_dashes_per_1k),
        float(periphrastic_per_1k),
        float(mean_commas),
    ]
    return np.asarray(feats, dtype=float)
def _stage_f_fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge: β = (XᵀX + λI)⁻¹ Xᵀy. X includes a leading 1-column
    for the intercept; the intercept term is NOT penalized."""
    _n, p = X.shape
    reg = lam * np.eye(p)
    reg[0, 0] = 0.0  # don't regularize intercept
    beta = np.linalg.solve(X.T @ X + reg, X.T @ y)
    return beta
def _stage_f_loo_metrics(X_raw: np.ndarray, y: np.ndarray, lam: float) -> tuple:
    """Exact leave-one-out metrics with train-fold scaling.
    This is slower than the previous shortcut but the corpus is small enough
    that it remains cheap, and it reports a more honest advisory benchmark.
    """
    n = X_raw.shape[0]
    if n == 0:
        return float("nan"), float("nan")
    preds = []
    idx = np.arange(n)
    for i in range(n):
        mask = idx != i
        X_train = X_raw[mask]
        y_train = y[mask]
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0)
        sigma = np.where(sigma < 1e-9, 1.0, sigma)
        Xn_train = (X_train - mu) / sigma
        x_test = (X_raw[i] - mu) / sigma
        X_full = np.column_stack([np.ones(Xn_train.shape[0]), Xn_train])
        x_full = np.concatenate([[1.0], x_test])
        beta = _stage_f_fit_ridge(X_full, y_train, lam)
        preds.append(float(x_full @ beta))
    preds_arr = np.asarray(preds, dtype=float)
    mae = float(np.mean(np.abs(preds_arr - y)))
    if len(preds_arr) >= 2 and float(np.std(preds_arr)) > 0 and float(np.std(y)) > 0:
        loo_r = float(np.corrcoef(preds_arr, y)[0, 1])
    else:
        loo_r = float("nan")
    return mae, loo_r
@st.cache_resource
def stage_f_load_predictor(corpus_path_str: str, lam: float) -> dict:
    """Load labeled_corpus.json, extract features for every labeled record,
    fit ridge with feature standardization, and return a predictor dict.
    Returns a dict with:
      available: bool
      reason: str — explanation if not available
      beta: np.ndarray — ridge coefficients (length p+1 including intercept)
      mu, sigma: feature normalization terms
      n_train: int
      loo_mae: float
      loo_r: float
      feature_names: list[str]
    """
    corpus_path = Path(corpus_path_str)
    if not corpus_path.exists():
        return {"available": False, "reason": f"labeled_corpus.json not found at {corpus_path}"}
    try:
        with open(corpus_path, "r", encoding="utf-8") as fh:
            corpus = json.load(fh)
    except Exception as e:
        return {"available": False, "reason": f"failed to read corpus: {e}"}
    X_rows, y_rows = [], []
    for rec in corpus:
        text = rec.get("text") or ""
        score = rec.get("human_score")
        if not text or score is None:
            continue
        try:
            X_rows.append(stage_f_extract_features(text))
            y_rows.append(float(score))
        except Exception:
            continue
    if len(X_rows) < 10:
        return {"available": False, "reason": f"insufficient labeled docs ({len(X_rows)})"}
    X_raw = np.vstack(X_rows)
    y = np.asarray(y_rows, dtype=float)
    mu = X_raw.mean(axis=0)
    sigma = X_raw.std(axis=0)
    sigma = np.where(sigma < 1e-9, 1.0, sigma)
    Xn = (X_raw - mu) / sigma
    X_full = np.column_stack([np.ones(Xn.shape[0]), Xn])
    beta = _stage_f_fit_ridge(X_full, y, lam)
    loo_mae, loo_r = _stage_f_loo_metrics(X_raw, y, lam)
    return {
        "available": True,
        "reason": "",
        "beta": beta,
        "mu": mu,
        "sigma": sigma,
        "n_train": int(X_raw.shape[0]),
        "loo_mae": loo_mae,
        "loo_r": loo_r,
        "feature_names": list(STAGE_F_FEATURE_NAMES),
    }
def stage_f_predict_detailed(text: str, predictor: dict) -> dict:
    """Predict Originality human-score for `text` using the fitted ridge and
    return the full feature/contribution breakdown used in the score."""
    if not predictor or not predictor.get("available"):
        return {
            "available": False,
            "reason": (predictor or {}).get("reason", "predictor not loaded"),
            "predicted_score": None,
            "band": "UNAVAILABLE",
            "n_train": 0,
            "loo_mae": float("nan"),
            "loo_r": float("nan"),
            "features": {},
            "feature_z": {},
            "feature_contrib": {},
            "intercept": 0.0,
        }
    try:
        feats = stage_f_extract_features(text or "")
    except Exception as e:
        return {
            "available": False,
            "reason": f"feature extraction failed: {e}",
            "predicted_score": None,
            "band": "UNAVAILABLE",
            "n_train": predictor.get("n_train", 0),
            "loo_mae": predictor.get("loo_mae", float("nan")),
            "loo_r": predictor.get("loo_r", float("nan")),
            "features": {},
            "feature_z": {},
            "feature_contrib": {},
            "intercept": 0.0,
        }
    mu = predictor["mu"]
    sigma = predictor["sigma"]
    beta = predictor["beta"]
    x_norm = (feats - mu) / sigma
    contribs = x_norm * beta[1:]
    raw = float(beta[0] + np.sum(contribs))
    clamped = max(0, min(100, int(round(raw))))
    band = stage_f_band(clamped)
    return {
        "available": True,
        "reason": "",
        "predicted_score": clamped,
        "raw_score": round(raw, 2),
        "band": band,
        "n_train": predictor.get("n_train", 0),
        "loo_mae": round(predictor.get("loo_mae", float("nan")), 2),
        "loo_r": round(predictor.get("loo_r", float("nan")), 3),
        "features": {
            name: round(float(v), 3)
            for name, v in zip(STAGE_F_FEATURE_NAMES, feats)
        },
        "feature_z": {
            name: round(float(v), 3)
            for name, v in zip(STAGE_F_FEATURE_NAMES, x_norm)
        },
        "feature_contrib": {
            name: round(float(v), 3)
            for name, v in zip(STAGE_F_FEATURE_NAMES, contribs)
        },
        "intercept": round(float(beta[0]), 3),
    }
def stage_f_predict(text: str, predictor: dict) -> dict:
    """Compact wrapper for UI/summary use; retains the detailed breakdown too."""
    return stage_f_predict_detailed(text, predictor)
def write_stage_f_debug_report(
    predictor: dict,
    scored_items: list,
    batch_stub: str,
    top1_run_id: str = "",
    final_run_label: str = "",
) -> str:
    """Write a human-readable Stage F debug report for every scored draft in
    the batch, plus the final shipped text when present."""
    report_path = FINAL_DIR / f"STAGEF_DEBUG_{batch_stub}.txt"
    lines = []
    lines.append(f"STAGE F DEBUG REPORT — {batch_stub}")
    lines.append("=" * 60)
    if predictor and predictor.get("available"):
        lines.append(
            f"Corpus: {predictor.get('n_train', 0)} labeled docs · "
            f"LOO MAE {round(predictor.get('loo_mae', float('nan')), 2)} · "
            f"r {round(predictor.get('loo_r', float('nan')), 3)}"
        )
        lines.append(f"Lambda: {STAGE_F_RIDGE_LAMBDA}")
    lines.append("")
    for idx, item in enumerate(scored_items, 1):
        pred = item.get("prediction", {})
        label = item.get("label", item.get("run_id", f"item_{idx}"))
        marker_bits = []
        if item.get("run_id") and item.get("run_id") == top1_run_id:
            marker_bits.append("TOP 1")
        if final_run_label and label == final_run_label:
            marker_bits.append("FINAL scored text")
        marker = f" [{' · '.join(marker_bits)}]" if marker_bits else ""
        lines.append(f"{idx}. {label}{marker}")
        if not pred.get("available"):
            lines.append(f"   unavailable: {pred.get('reason', 'predictor not loaded')}")
            lines.append("")
            continue
        lines.append(
            f"   predicted={pred.get('predicted_score')} raw={pred.get('raw_score')} "
            f"band={pred.get('band')}"
        )
        lines.append(f"   intercept={pred.get('intercept', 0.0)}")
        feats = pred.get("features", {})
        zmap = pred.get("feature_z", {})
        cmap = pred.get("feature_contrib", {})
        feature_lines = []
        for name in STAGE_F_FEATURE_NAMES:
            feature_lines.append(
                (abs(float(cmap.get(name, 0.0))),
                 f"   {name}: value={feats.get(name)} z={zmap.get(name)} contrib={cmap.get(name)}")
            )
        for _abs_c, line in sorted(feature_lines, key=lambda t: t[0], reverse=True):
            lines.append(line)
        pos = [f"{k} {v:+.3f}" for k, v in sorted(cmap.items(), key=lambda kv: kv[1], reverse=True) if v > 0][:3]
        neg = [f"{k} {v:+.3f}" for k, v in sorted(cmap.items(), key=lambda kv: kv[1]) if v < 0][:3]
        if pos:
            lines.append("   strongest upward pushes: " + "; ".join(pos))
        if neg:
            lines.append("   strongest downward pushes: " + "; ".join(neg))
        lines.append("")
    save_text(report_path, "\n".join(lines))
    return str(report_path)
def stage_f_band(score: int) -> str:
    if score is None:
        return "UNAVAILABLE"
    if score >= STAGE_F_BAND_SHIP:
        return "SHIP"
    if score >= STAGE_F_BAND_CAUTION:
        return "RECONSIDER"
    return "REGENERATE"
# ============================================================================
# Stage G — line-edit pass (mechanical copyedit + AI-tell deletion/graft)
# ============================================================================
#
# Three internal steps run in sequence:
#
#   G1  Mechanical copyedit. One LLM call on the final text with
#       instructions limited to unambiguous punctuation fixes (missing
#       coordinator commas, comma splices, missing apostrophes, missing
#       end-of-sentence punctuation). A word-sequence invariant check
#       rejects the edit if any word was added, removed, or changed.
#
#   G2  AI-tell identification. Deterministic scan over AI_TELL_WORDS. For
#       each flagged word that appears in the text, the containing sentence
#       is captured.
#
#   G3  For each flagged sentence, a three-behavior cascade:
#         a. Same-beat graft. One LLM call looks in the runner-up drafts
#            for a verbatim sentence that does the same narrative work
#            without the flagged construction. If one is found, it replaces
#            the flagged sentence.
#         b. Deletion. If no graft is found, the deletion patterns from
#            AI_TELL_WORDS are applied to the sentence and the a/an article
#            agreement is repaired. If the deletion changed the sentence,
#            the change is applied.
#         c. Rewrite flag. If deletion produced no change, the sentence is
#            recorded in the audit report for manual review; no edit is
#            applied.
#
# Stage G outputs two files:
#   FINAL_<batch_stub>_LINEEDITED.txt  — the edited text, if anything changed
#   LINEEDIT_REPORT_<batch_stub>.txt    — an audit log of every action
#
# Stage G is advisory in the sense that the original FINAL file is not
# overwritten — both remain available. The edited text becomes the basis
# for Stage F's prediction.
# ============================================================================
LINE_EDIT_MECHANICAL_PROMPT = """You are a strict copyeditor. Your job is to fix unambiguous punctuation errors in the text below and nothing else.
Allowed edits:
- Insert a comma before a coordinating conjunction (and, but, or, nor, for, so, yet) when it joins two independent clauses.
- Fix comma splices (two independent clauses joined by only a comma) by replacing the comma with a semicolon or a period. If you use a period, capitalize the next word.
- Add a missing apostrophe in a contraction or possessive.
- Add missing end-of-sentence punctuation where the sentence structure clearly calls for it.
Forbidden edits:
- Do NOT change, add, remove, or reorder any words.
- Do NOT break a sentence apart or merge sentences, except the comma-splice fix above.
- Do NOT change spelling (British vs American, archaic vs modern).
- Do NOT make stylistic changes, smoothing, or rewording.
- Do NOT touch dialogue or internal quotes unless the fix is an unambiguous punctuation error.
Return ONLY the corrected text. No preamble, no commentary, no markdown fencing.
TEXT:
{text}
"""
LINE_EDIT_GRAFT_PROMPT = """A sentence in the TOP 1 draft contains an AI-tell construction that needs replacement. Your job is to find, in the runner-up drafts, a VERBATIM sentence that does the same narrative work but does not contain the flagged construction.
FLAGGED SENTENCE (from TOP 1):
{flagged_sentence}
FLAGGED CONSTRUCTION:
The word or phrase "{flagged_word}" used as an adjective, intensifier, or part of a named-state construction. The replacement must not reintroduce the same word or construction.
SURROUNDING CONTEXT IN TOP 1 (for beat identification only):
{context_before}
>>> [FLAGGED SENTENCE] <<<
{context_after}
RUNNER-UP DRAFTS (same chapter, same outline, different generations):
{alternative_drafts}
For each runner-up draft, locate the sentence or short passage that covers the same beat as the flagged sentence — the same moment in the chapter, the same narrative function. If any of those same-beat sentences is clean of the flagged construction and does equivalent work, choose the cleanest one and return it verbatim.
Return ONE of these two formats, with no other text:
REPLACEMENT: <the replacement sentence, verbatim from the named runner-up>
SOURCE: T<n>
OR, if no runner-up has a same-beat sentence that is both clean of the flagged construction and does the same work:
NO_REPLACEMENT
Do not invent, paraphrase, or compose. The replacement must be a sentence that already exists in one of the runner-up drafts, copied character-for-character.
"""
G4_MULTI_SENTENCE_PROMPT = """You are running a final mechanical pass over a chapter to remove three specific multi-sentence constructions. Your only operations are: delete a sentence in full, collapse a consecutive run of sentences by deleting the extensions and keeping the first, or leave a passage alone. You may not rewrite, paraphrase, or add new wording.
TARGETS
T1 — NEGATION FIGURE. Two or more consecutive sentences each carrying an explicit negation ("not," "no," "never," "did not," "could not," "would not," "had not," "was not," "is not"). The first sentence does the dramatic work; subsequent sentences function as rhetorical extension or amplification rather than introducing new information. Recognizable by structural parallelism between the sentences and by the absence of new content in the extensions. Do not flag two negation sentences that simply happen to be adjacent and carry independent content; flag only when the second (and any following) reads as scaffolding for the first. Action: keep the first sentence; delete every consecutive negation sentence that functions as extension.
T2 — CLOSING APHORISTIC GLOSS. A scene closes (paragraph break or scene break immediately follows) with an interpretive sentence that names what the scene meant rather than showing it. Common shapes: sentences beginning "It was the kind of...," "It was what...," "That was the X of...," "She/he/I was a woman/man who...," followed by a paragraph break or section break. Action: delete the closing interpretive sentence.
T3 — CLASSIFY-BY-GENRE RESIDUAL. A sentence whose function is to classify a person or thing as belonging to a category — "the kind of [X] that [Y]," "a [noun] who [verbs]," "the [adjective] of a [noun] who" — and whose removal does not leave the surrounding paragraph ungrammatical. Action: delete the classification sentence. If the classification is woven into a sentence whose other content is needed, leave it.
HARD CONSTRAINTS
- Delete only. Never rewrite. Never paraphrase. Never add new wording.
- Do not edit dialogue (anything inside quotation marks).
- Do not delete a whole paragraph. If a paragraph contains only a target sentence, leave it.
- The total word count must remain within 2 percent of the input word count.
- If you are not certain a candidate matches a target, leave it.
OUTPUT FORMAT
Return exactly two sections, in this order, with no other text:
EDITED_TEXT:
<the full chapter, with deletions applied. Preserve all paragraph breaks and other formatting verbatim except where a deletion removes a sentence.>
EDITS:
[
  {{"target": "T1|T2|T3", "deleted": "<verbatim sentence(s) removed>", "kept_neighbor": "<verbatim sentence kept (T1 only) or empty string>"}},
  ...
]
If no edits are warranted, return EDITED_TEXT identical to the input and EDITS as [].
CHAPTER:
{text}
"""
# ---- G1 helpers ------------------------------------------------------------
def _word_sequence(text: str) -> list:
    """Return the list of words in order, ignoring punctuation and whitespace.
    Used for the invariant check after G1 — a valid mechanical edit preserves
    this list exactly."""
    return re.findall(r"\b[\w']+\b", text or "")
def run_mechanical_copyedit(client, model: str, text: str) -> dict:
    """G1. One LLM call, punctuation-only edit, with word-sequence invariant
    check. Returns a dict:
        applied: bool         — whether the edit was accepted
        edited_text: str      — the edited text if accepted, else original
        reason: str           — explanation if rejected
        raw: str              — full model output (for audit)
    """
    out = {"applied": False, "edited_text": text, "reason": "", "raw": ""}
    if not text or not text.strip():
        out["reason"] = "empty text"
        return out
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_GEN_TOKENS,
            messages=[{
                "role": "user",
                "content": LINE_EDIT_MECHANICAL_PROMPT.format(text=text),
            }],
        )
    except Exception as e:
        out["reason"] = f"api error: {e}"
        return out
    raw = "\n".join(b.text for b in resp.content if getattr(b, "text", None))
    out["raw"] = raw
    candidate = raw.strip()
    # Strip accidental markdown fences if the model added any despite instructions
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[a-zA-Z]*\n?", "", candidate)
        candidate = re.sub(r"\n?```$", "", candidate)
        candidate = candidate.strip()
    if not candidate:
        out["reason"] = "empty response"
        return out
    # Invariant: word sequence must match exactly (punctuation-only edit).
    orig_words = _word_sequence(text)
    edit_words = _word_sequence(candidate)
    if orig_words != edit_words:
        # Locate the first divergence for the audit report
        first_diff = ""
        for i in range(min(len(orig_words), len(edit_words))):
            if orig_words[i] != edit_words[i]:
                first_diff = (
                    f"position {i}: '{orig_words[i]}' → '{edit_words[i]}'"
                )
                break
        if not first_diff and len(orig_words) != len(edit_words):
            first_diff = (
                f"word count changed: {len(orig_words)} → {len(edit_words)}"
            )
        out["reason"] = f"word-sequence invariant violated ({first_diff})"
        return out
    if candidate == text:
        out["reason"] = "no changes"
        return out
    out["applied"] = True
    out["edited_text"] = candidate
    return out
def _summarize_punctuation_diff(before: str, after: str) -> list:
    """Cheap summary of what punctuation was added/removed. Returns a list of
    short strings for the audit report."""
    changes = []
    punct_pairs = [
        (",", "comma"),
        (";", "semicolon"),
        (":", "colon"),
        (".", "period"),
        ("'", "apostrophe"),
    ]
    for sym, name in punct_pairs:
        delta = after.count(sym) - before.count(sym)
        if delta > 0:
            changes.append(f"+{delta} {name}{'s' if delta != 1 else ''}")
        elif delta < 0:
            changes.append(f"{delta} {name}{'s' if abs(delta) != 1 else ''}")
    return changes
# ---- G2 helpers ------------------------------------------------------------
def _split_sentences_with_spans(text: str) -> list:
    """Return list of (sentence_text, start_idx, end_idx) across the whole
    text. Sentence boundary: [.!?] followed by whitespace or end-of-text."""
    spans = []
    for m in re.finditer(r"[^.!?]+[.!?]+[\"\u201d)]*", text, flags=re.DOTALL):
        s = m.group(0)
        # Strip leading whitespace from the sentence but track its real start
        leading = len(s) - len(s.lstrip())
        start = m.start() + leading
        end = m.end()
        sent = text[start:end]
        if sent.strip():
            spans.append((sent, start, end))
    return spans
def find_ai_tell_sentences(text: str, ai_tell_words: dict) -> list:
    """G2. Return list of dicts for each sentence containing an AI-tell.
    Each dict has: sentence, start, end, flagged_word, match_text."""
    flagged = []
    spans = _split_sentences_with_spans(text)
    for sent_text, start, end in spans:
        for word, patterns in ai_tell_words.items():
            hit = None
            for pat, _repl in patterns:
                m = re.search(pat, sent_text, flags=re.IGNORECASE)
                if m:
                    hit = m.group(0)
                    break
            if hit:
                flagged.append({
                    "sentence": sent_text,
                    "start": start,
                    "end": end,
                    "flagged_word": word,
                    "match_text": hit,
                })
                # Don't double-count the same sentence under different words
                break
    return flagged
# ---- G3b helpers -----------------------------------------------------------
def apply_deletion_heuristic(sentence: str, ai_tell_words: dict) -> tuple:
    """G3b. Apply all deletion patterns to the sentence, run a/an article
    agreement repair, and collapse doubled whitespace. Patterns whose
    replacement is the GRAFT_ONLY sentinel (None) are skipped — those
    constructions cannot be safely deleted in place and must be resolved
    via G3a graft or G3c manual flag. Returns (modified_sentence,
    list_of_edits_made)."""
    edits = []
    result = sentence
    for word, patterns in ai_tell_words.items():
        for pat, repl in patterns:
            if repl is None:
                # graft-only: deletion is unsafe, leave the sentence alone
                continue
            new_result = re.sub(pat, repl, result, flags=re.IGNORECASE)
            if new_result != result:
                edits.append({
                    "word": word,
                    "pattern": pat,
                    "before": result,
                    "after": new_result,
                })
                result = new_result
    # a/an article repair. Only touch instances that are lowercase (leave
    # sentence-initial "A" alone unless it clearly needs repair — rare).
    result = re.sub(
        r"\b(a)\s+([aeiouAEIOU]\w)",
        lambda m: ("A" if m.group(1).isupper() else "a") + "n " + m.group(2),
        result,
    )
    result = re.sub(
        r"\b(an)\s+([^aeiouAEIOU\W\d]\w)",
        lambda m: ("A" if m.group(1)[0].isupper() else "a") + " " + m.group(2),
        result,
    )
    result = re.sub(r"  +", " ", result).strip() + (
        "" if result.endswith(("\n",)) else ""
    )
    # Preserve trailing whitespace/newlines the split captured
    trailing = len(sentence) - len(sentence.rstrip())
    if trailing:
        result = result.rstrip() + sentence[-trailing:]
    return result, edits
# ---- G3a helpers -----------------------------------------------------------
def _context_around(text: str, start: int, end: int, window_chars: int = 400) -> tuple:
    """Return (before_context, after_context) trimmed to the nearest paragraph
    or sentence boundary within window_chars of each side."""
    before_raw = text[max(0, start - window_chars):start]
    after_raw = text[end:end + window_chars]
    # Trim to the last/next double-newline or sentence boundary for readability
    if "\n\n" in before_raw:
        before_raw = before_raw.split("\n\n", 1)[1]
    if "\n\n" in after_raw:
        after_raw = after_raw.rsplit("\n\n", 1)[0]
    return before_raw.strip(), after_raw.strip()
def try_same_beat_graft(
    client,
    model: str,
    flagged_sentence: str,
    flagged_word: str,
    full_text: str,
    sentence_start: int,
    sentence_end: int,
    runner_up_drafts: list,
) -> dict:
    """G3a. One LLM call. Returns a dict:
        replacement: str      — the graft sentence, or "" if none
        source: str           — e.g., "T2"
        raw: str              — full model output
        reason: str           — short explanation of the outcome
    """
    out = {"replacement": "", "source": "", "raw": "", "reason": ""}
    if not runner_up_drafts:
        out["reason"] = "no runner-up drafts available"
        return out
    before_ctx, after_ctx = _context_around(
        full_text, sentence_start, sentence_end
    )
    alt_blocks = []
    for i, d in enumerate(runner_up_drafts, 1):
        alt_blocks.append(
            f"--- T{i} (run_id: {d.get('run_id', '')}) ---\n{d.get('text', '')}"
        )
    prompt = LINE_EDIT_GRAFT_PROMPT.format(
        flagged_sentence=flagged_sentence.strip(),
        flagged_word=flagged_word,
        context_before=before_ctx,
        context_after=after_ctx,
        alternative_drafts="\n\n".join(alt_blocks),
    )
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_EVAL_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
    except Exception as e:
        out["reason"] = f"api error: {e}"
        return out
    raw = "\n".join(b.text for b in resp.content if getattr(b, "text", None))
    out["raw"] = raw
    if re.search(r"\bNO_REPLACEMENT\b", raw):
        out["reason"] = "evaluator returned NO_REPLACEMENT"
        return out
    rep_m = re.search(
        r"REPLACEMENT:\s*(.+?)(?:\nSOURCE:|\Z)",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )
    src_m = re.search(r"SOURCE:\s*T\s*(\d+)", raw, flags=re.IGNORECASE)
    if not rep_m:
        out["reason"] = "could not parse REPLACEMENT line"
        return out
    candidate = rep_m.group(1).strip().strip('"').strip()
    if not candidate:
        out["reason"] = "empty REPLACEMENT"
        return out
    # Verify the candidate is verbatim in one of the runner-up drafts. This
    # guards against the model paraphrasing or inventing.
    verified_source = ""
    for i, d in enumerate(runner_up_drafts, 1):
        if candidate in d.get("text", ""):
            verified_source = f"T{i}"
            break
    if not verified_source:
        out["reason"] = "candidate not found verbatim in any runner-up draft"
        return out
    # Verify the replacement is clean of the flagged construction
    patterns = AI_TELL_WORDS.get(flagged_word, [])
    for pat, _repl in patterns:
        if re.search(pat, candidate, flags=re.IGNORECASE):
            out["reason"] = "candidate still contains the flagged construction"
            return out
    if src_m:
        claimed = f"T{src_m.group(1)}"
        out["source"] = claimed if claimed == verified_source else verified_source
    else:
        out["source"] = verified_source
    out["replacement"] = candidate
    out["reason"] = "graft accepted"
    return out
# ---- G4 helpers ------------------------------------------------------------
def _word_multiset(text: str) -> dict:
    """Return a multiset (dict word->count) of word-token occurrences,
    case-preserved. Used by G4's deletion-only invariant check: every
    word-token in the edited text must appear at least as many times in
    the original."""
    counts = {}
    for w in re.findall(r"\b[\w']+\b", text or ""):
        counts[w] = counts.get(w, 0) + 1
    return counts
def _split_paragraphs(text: str) -> list:
    """Return non-empty paragraphs split on blank lines. Used by G4's
    no-paragraph-deletion invariant check."""
    return [p for p in re.split(r"\n\s*\n", text or "") if p.strip()]
def run_g4_multisentence_pass(client, model: str, text: str) -> dict:
    """G4. One LLM call over the post-G3 text, deletion-only, targeting
    multi-sentence patterns: negation triplets, closing aphoristic gloss,
    and classify-by-genre residuals.
    Three invariants protect the output:
      1. ±2 percent word-count band (deletion of a small number of
         sentences should not change overall word count by more than 2%).
      2. No whole paragraph deleted (the model may collapse sentences but
         not remove a paragraph entirely).
      3. Deletion-only at the word level: every word-token in the edited
         text must appear at least as often in the original (the model is
         not allowed to introduce any new word).
    Returns a dict:
        applied: bool         — whether the edit was accepted
        edited_text: str      — edited text if accepted, else original
        reason: str           — short explanation of outcome
        edits: list           — parsed EDITS JSON from the model
        raw: str              — full model output (for audit)
    """
    out = {
        "applied": False, "edited_text": text,
        "reason": "", "edits": [], "raw": "",
    }
    if not text or not text.strip():
        out["reason"] = "empty text"
        return out
    try:
        resp = client.messages.create(
            model=model,
            max_tokens=MAX_GEN_TOKENS,
            messages=[{
                "role": "user",
                "content": G4_MULTI_SENTENCE_PROMPT.format(text=text),
            }],
        )
    except Exception as e:
        out["reason"] = f"api error: {e}"
        return out
    raw = "\n".join(b.text for b in resp.content if getattr(b, "text", None))
    out["raw"] = raw
    # Parse EDITED_TEXT block: everything between "EDITED_TEXT:" and the
    # later "EDITS:" marker. Tolerate optional surrounding whitespace.
    et_m = re.search(
        r"EDITED_TEXT:\s*\n?(.*?)\n\s*EDITS:",
        raw, flags=re.DOTALL,
    )
    if not et_m:
        out["reason"] = "could not parse EDITED_TEXT section"
        return out
    edited = et_m.group(1).strip("\n")
    # Parse EDITS block as JSON list. Be lenient about trailing whitespace
    # and accidental extra prose after the closing bracket.
    edits_list = []
    edits_m = re.search(r"EDITS:\s*(\[.*?\])\s*\Z", raw, flags=re.DOTALL)
    if edits_m:
        try:
            parsed = json.loads(edits_m.group(1))
            if isinstance(parsed, list):
                edits_list = parsed
        except json.JSONDecodeError:
            edits_list = []
    out["edits"] = edits_list
    if edited == text:
        out["reason"] = "no changes"
        return out
    # Invariant 1: word-count must stay within ±2% of the input
    in_words = len(text.split())
    out_words = len(edited.split())
    if in_words > 0:
        delta = abs(out_words - in_words) / in_words
        if delta > 0.02:
            out["reason"] = (
                f"word-count invariant violated (>2%): "
                f"{in_words} -> {out_words} ({delta * 100:.1f}%)"
            )
            return out
    # Invariant 2: no paragraph deleted entirely
    in_paras = _split_paragraphs(text)
    out_paras = _split_paragraphs(edited)
    if len(out_paras) < len(in_paras):
        out["reason"] = (
            f"paragraph-count invariant violated: "
            f"{len(in_paras)} -> {len(out_paras)} paragraphs"
        )
        return out
    # Invariant 3: deletion-only — no word-token introduced
    in_counts = _word_multiset(text)
    new_counts = dict(in_counts)
    for w in re.findall(r"\b[\w']+\b", edited):
        if new_counts.get(w, 0) <= 0:
            out["reason"] = f"new word-token introduced: '{w}'"
            return out
        new_counts[w] -= 1
    out["applied"] = True
    out["edited_text"] = edited
    out["reason"] = (
        f"applied {len(edits_list)} deletion(s)"
        if edits_list else "text changed"
    )
    return out
# ---- Stage G orchestrator --------------------------------------------------
def run_line_edit_pass(
    client,
    eval_model: str,
    final_text: str,
    all_acceptable_drafts: list,
    top1_run_id: str,
    batch_stub: str,
) -> dict:
    """Orchestrate G1 → G2 → G3 → G4 on `final_text`. Writes the edited text
    and an audit report to FINAL_DIR. Does not overwrite the original
    final_text file. Returns a dict usable by write_batch_summary and the UI.
    Args:
        final_text: the text produced by Stage D (TOP 1 or TOP1_GRAFTED).
        all_acceptable_drafts: drafts that cleared Q1 (used for G3a graft
                               candidates; TOP 1 itself is excluded).
        top1_run_id: run_id of the TOP 1 draft, so it can be excluded from
                     the graft pool.
        batch_stub: for output file naming.
    """
    result = {
        "ran": True,
        "enabled": STAGE_G_ENABLED,
        "original_text": final_text,
        "edited_text": final_text,
        "mechanical": {},
        "flagged_count": 0,
        "flagged_sentences": [],
        "g4": {},
        "edited_path": "",
        "report_path": "",
        "changed": False,
    }
    if not STAGE_G_ENABLED:
        result["ran"] = False
        return result
    if not final_text or not final_text.strip():
        result["ran"] = False
        return result
    # --- G1: mechanical copyedit ---
    mech = run_mechanical_copyedit(client, eval_model, final_text)
    result["mechanical"] = {
        "applied": mech.get("applied", False),
        "reason": mech.get("reason", ""),
        "raw": mech.get("raw", ""),
    }
    if mech.get("applied"):
        diff_summary = _summarize_punctuation_diff(final_text, mech["edited_text"])
        result["mechanical"]["diff_summary"] = diff_summary
        current_text = mech["edited_text"]
    else:
        current_text = final_text
    # --- G2: AI-tell identification ---
    flagged = find_ai_tell_sentences(current_text, AI_TELL_WORDS)
    result["flagged_count"] = len(flagged)
    # --- G3: resolve each flagged sentence ---
    runner_up_drafts = [
        d for d in (all_acceptable_drafts or [])
        if d.get("run_id") != top1_run_id
    ]
    # Re-compute sentence spans against current_text as edits are applied;
    # rebuild the flagged list from the updated text before each step so
    # positions stay correct.
    remaining = list(flagged)
    resolved = []
    while remaining:
        # Find the first flagged sentence in the current text
        fs = None
        for candidate_fs in remaining:
            if candidate_fs["sentence"] in current_text:
                fs = candidate_fs
                break
        if fs is None:
            # Residual flags whose text no longer appears (replaced earlier)
            for rem in remaining:
                resolved.append({
                    "original_sentence": rem["sentence"],
                    "flagged_word": rem["flagged_word"],
                    "match_text": rem["match_text"],
                    "action": "skipped_already_replaced",
                    "replacement": "",
                    "graft_raw": "",
                    "graft_reason": "",
                    "source": "",
                })
            break
        remaining.remove(fs)
        sent_text = fs["sentence"]
        # Locate sentence in current_text for context extraction
        idx = current_text.find(sent_text)
        if idx < 0:
            resolved.append({
                "original_sentence": sent_text,
                "flagged_word": fs["flagged_word"],
                "match_text": fs["match_text"],
                "action": "skipped_not_found",
                "replacement": "",
                "graft_raw": "",
                "graft_reason": "",
                "source": "",
            })
            continue
        entry = {
            "original_sentence": sent_text,
            "flagged_word": fs["flagged_word"],
            "match_text": fs["match_text"],
            "action": "",
            "replacement": "",
            "graft_raw": "",
            "graft_reason": "",
            "source": "",
        }
        # G3a: try same-beat graft
        graft = try_same_beat_graft(
            client, eval_model,
            sent_text, fs["flagged_word"],
            current_text, idx, idx + len(sent_text),
            runner_up_drafts,
        )
        entry["graft_raw"] = graft.get("raw", "")
        entry["graft_reason"] = graft.get("reason", "")
        if graft.get("replacement"):
            new_text = current_text.replace(sent_text, graft["replacement"], 1)
            if new_text != current_text:
                current_text = new_text
                entry["action"] = "graft"
                entry["replacement"] = graft["replacement"]
                entry["source"] = graft.get("source", "")
                resolved.append(entry)
                continue
        # G3b: deletion heuristic
        deleted, edits_made = apply_deletion_heuristic(sent_text, AI_TELL_WORDS)
        if edits_made and deleted != sent_text:
            new_text = current_text.replace(sent_text, deleted, 1)
            if new_text != current_text:
                current_text = new_text
                entry["action"] = "deletion"
                entry["replacement"] = deleted
                resolved.append(entry)
                continue
        # G3c: flag for manual review
        entry["action"] = "flag_for_rewrite"
        resolved.append(entry)
    result["flagged_sentences"] = resolved
    # --- G4: multi-sentence deletion pass over the post-G3 text ---
    g4 = run_g4_multisentence_pass(client, eval_model, current_text)
    result["g4"] = {
        "applied": g4.get("applied", False),
        "reason": g4.get("reason", ""),
        "edits": g4.get("edits", []),
        "raw": g4.get("raw", ""),
    }
    if g4.get("applied"):
        current_text = g4["edited_text"]
    result["edited_text"] = current_text
    result["changed"] = current_text != final_text
    # --- Save outputs ---
    try:
        ensure_dirs()
    except Exception:
        pass
    if result["changed"]:
        edited_path = FINAL_DIR / f"FINAL_{batch_stub}_RANK-01_WINNER_LINEEDITED_run-{top1_run_id}.txt"
        save_text(edited_path, current_text)
        result["edited_path"] = str(edited_path)
    # Always write the report, even when no edits were applied — it
    # documents what was looked at.
    report_lines = []
    report_lines.append(f"LINE-EDIT REPORT — {batch_stub}")
    report_lines.append("=" * 60)
    report_lines.append("")
    report_lines.append("G1 — Mechanical copyedit (punctuation only)")
    report_lines.append("-" * 60)
    mech_applied = result["mechanical"].get("applied", False)
    report_lines.append(f"Applied: {mech_applied}")
    report_lines.append(f"Reason:  {result['mechanical'].get('reason', '')}")
    diff_summary = result["mechanical"].get("diff_summary") or []
    if diff_summary:
        report_lines.append("Diff:    " + ", ".join(diff_summary))
    report_lines.append("")
    report_lines.append("G2 / G3 — AI-tell identification and resolution")
    report_lines.append("-" * 60)
    report_lines.append(f"Flagged sentences: {len(resolved)}")
    report_lines.append("")
    for i, entry in enumerate(resolved, 1):
        report_lines.append(f"[{i}] Flagged word: {entry['flagged_word']}  "
                            f"(match: '{entry['match_text']}')")
        report_lines.append(f"    Action: {entry['action']}")
        report_lines.append(f"    Original:    {entry['original_sentence'].strip()}")
        if entry["replacement"]:
            report_lines.append(
                f"    Replacement: {entry['replacement'].strip()}"
            )
        if entry["action"] == "graft" and entry.get("source"):
            report_lines.append(f"    Graft source: {entry['source']}")
        if entry["action"] == "flag_for_rewrite":
            report_lines.append(
                "    NOTE: neither graft nor deletion applied. "
                "Manual rewrite required."
            )
        if entry.get("graft_reason"):
            report_lines.append(f"    Graft notes: {entry['graft_reason']}")
        report_lines.append("")
    # G4 section of the audit report
    report_lines.append("G4 — Multi-sentence deletion pass (deletion-only)")
    report_lines.append("-" * 60)
    g4_applied = result["g4"].get("applied", False)
    report_lines.append(f"Applied: {g4_applied}")
    report_lines.append(f"Reason:  {result['g4'].get('reason', '')}")
    g4_edits = result["g4"].get("edits") or []
    report_lines.append(f"Edits:   {len(g4_edits)}")
    report_lines.append("")
    for i, edit in enumerate(g4_edits, 1):
        if not isinstance(edit, dict):
            continue
        target = edit.get("target", "?")
        deleted = (edit.get("deleted") or "").strip()
        kept = (edit.get("kept_neighbor") or "").strip()
        report_lines.append(f"[{i}] Target: {target}")
        if deleted:
            report_lines.append(f"    Deleted: {deleted}")
        if kept:
            report_lines.append(f"    Kept:    {kept}")
        report_lines.append("")
    report_path = FINAL_DIR / f"LINEEDIT_REPORT_{batch_stub}_RANK-01_WINNER_run-{top1_run_id}.txt"
    save_text(report_path, "\n".join(report_lines))
    result["report_path"] = str(report_path)
    return result
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
def _extract_global_drafting_controls(outline_text: str) -> str:
    """Extract the GLOBAL DRAFTING CONTROLS section from an outline so the
    caps can be re-surfaced at recency position in the drafter message.

    Returns the section body (without its header) if found, else "".
    The section is identified by a line beginning with "GLOBAL DRAFTING
    CONTROLS" (case-insensitive, allowing surrounding whitespace and
    optional trailing punctuation) and ends at the next top-level
    section header or end of document.

    Cap dilution mitigation: the v21 outline carries 19 hard caps inside
    its GLOBAL DRAFTING CONTROLS section. When the caps live only inside
    the outline document they compete with microbeats and source-voice
    reference for the drafter's attention. Extracting them and placing
    them as the terminal message block gives them recency primacy
    without touching the outline document itself.
    """
    if not outline_text:
        return ""
    lines = outline_text.splitlines()
    start = -1
    for i, ln in enumerate(lines):
        stripped = ln.strip().rstrip(":").rstrip(".")
        if re.match(r"(?i)^global drafting controls$", stripped):
            start = i + 1
            break
    if start < 0:
        return ""
    # Walk forward until we hit the next top-level section header. Top-
    # level headers in the v21 outline format are short ALL-CAPS lines
    # or lines beginning with "===" / "---" rule markers.
    end = len(lines)
    header_pat = re.compile(r"^(===+|---+)\s*$")
    for j in range(start, len(lines)):
        ln = lines[j].rstrip()
        stripped = ln.strip()
        if not stripped:
            continue
        if header_pat.match(stripped):
            end = j
            break
        # Detect a new ALL-CAPS section header (≥ 3 letters, no lower-
        # case anywhere). Cap-list bullet lines like "Cap 1 — Function:"
        # do not match because they contain lowercase letters.
        if (
            len(stripped) >= 3
            and any(c.isalpha() for c in stripped)
            and stripped.upper() == stripped
            and not stripped.startswith(("- ", "* ", "•"))
            and not stripped.lower().startswith("cap ")
        ):
            end = j
            break
    body = "\n".join(lines[start:end]).strip()
    return body


def _build_caps_recency_block(doc_texts: dict) -> str:
    """Build the terminal recency-boost block re-stating GLOBAL DRAFTING
    CONTROLS for the drafter. Returns "" if no caps section is present
    in any uploaded document."""
    for label, text in doc_texts.items():
        if not text or not text.strip():
            continue
        body = _extract_global_drafting_controls(text)
        if body:
            return (
                "[CRITICAL CONSTRAINTS — re-read before writing]\n\n"
                "These are the GLOBAL DRAFTING CONTROLS from the outline "
                "above. They are hard caps, not advisory. Every sentence "
                "you write must be screened against them. If the source "
                "voice and a cap conflict, the cap wins.\n\n"
                + body
            )
    return ""


def _build_word_count_overshoot_block(outline_text: str) -> str:
    """v30 — write a small recency-position instruction block telling the
    drafter to overshoot the outline's word-count target so the downstream
    line-edit pass (which now includes Type C deletions) lands the chapter
    at target after cuts. Returns an empty string if STAGE_G_OVERSHOOT_PCT
    is zero (v31 default — see regression note in constants block) or if
    the outline has no detectable word-count target.
    """
    if STAGE_G_OVERSHOOT_PCT <= 0:
        return ""
    target = _extract_target_word_count(outline_text or "")
    if not target:
        return ""
    pct = STAGE_G_OVERSHOOT_PCT
    overshoot_low = int(target * (1.0 + pct - 0.02))
    overshoot_high = int(target * (1.0 + pct + 0.02))
    pct_label = int(round(pct * 100))
    return (
        f"[v30 WORD COUNT GUIDANCE — read after the outline's word-count target]\n\n"
        f"The downstream line-edit pass deletes construction-flagged sentences "
        f"and procedural padding to reduce AI cadence. To land at the outline's "
        f"target word count after those deletions, draft this chapter at "
        f"{overshoot_low}–{overshoot_high} words — about {pct_label}% above the outline's "
        f"stated midpoint. Do NOT achieve the overshoot by adding atmospheric "
        f"description, procedural inventory, or scene-set commentary; those are "
        f"the categories the line-edit pass will cut. Achieve it by giving the "
        f"outline's required beats a little more interior weight or dialogue room."
    )


def build_payload_text(prompt_text: str, doc_texts: dict) -> str:
    parts = [prompt_text.strip()]
    outline_text_for_target = ""
    for label, text in doc_texts.items():
        if text.strip():
            parts.append(f"\n\n=== {label.upper()} ===\n\n{text.strip()}")
            if "outline" in label.lower() or "chapter" in label.lower():
                outline_text_for_target = text
    caps_block = _build_caps_recency_block(doc_texts)
    if caps_block:
        parts.append(f"\n\n=== {caps_block} ===")
    wc_block = _build_word_count_overshoot_block(outline_text_for_target)
    if wc_block:
        parts.append(f"\n\n{wc_block}")
    parts.append(
        "\n\nWrite the full chapter now. Return plain text only, "
        "with normal paragraph breaks and no commentary."
    )
    return "\n".join(parts)
def build_message_blocks(prompt_text: str, doc_texts: dict) -> list:
    blocks = [{"type": "text", "text": prompt_text.strip()}]
    outline_text_for_target = ""
    for label, text in doc_texts.items():
        if text.strip():
            blocks.append({
                "type": "text",
                "text": f"[{label.upper()}]\n\n{text.strip()}",
            })
            if "outline" in label.lower() or "chapter" in label.lower():
                outline_text_for_target = text
    caps_block = _build_caps_recency_block(doc_texts)
    if caps_block:
        blocks.append({
            "type": "text",
            "text": caps_block,
        })
    wc_block = _build_word_count_overshoot_block(outline_text_for_target)
    if wc_block:
        blocks.append({
            "type": "text",
            "text": wc_block,
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
def generate_quality_gated_batch(
    client,
    gen_model: str,
    eval_model: str,
    prompt_text: str,
    doc_uploads: dict,
    temperatures: list,
    repetitions: int,
    prompt_id: int,
    csv_path: Path,
    github_cfg: dict,
    progress,
    status,
    max_tries: int = QUALITY_GATE_MAX_TRIES,
) -> dict:
    """Generate drafts into fixed slots in a single pass. Keep every
    ACCEPTABLE draft for downstream ranking; drop UNACCEPTABLE drafts
    without retry. No target-score filtering, no slot refills — cost is
    bounded to one generation + one evaluation per slot (v23+ behavior)."""
    outline_text = doc_uploads.get("Outline", "")
    payload_text = build_payload_text(prompt_text, doc_uploads)
    message_blocks = build_message_blocks(prompt_text, doc_uploads)
    slots = []
    slot_no = 0
    for temp in temperatures:
        for rep in range(1, repetitions + 1):
            slot_no += 1
            slots.append({
                "slot_id": slot_no,
                "temp": temp,
                "rep": rep,
                "attempts": 0,
                "draft": None,
            })
    total_slots = len(slots)
    if total_slots == 0:
        return {
            "final_drafts": [],
            "scan_by_run_id": {},
            "all_run_ids": [],
            "retained_run_ids": [],
            "generated_count": 0,
            "attempt_rounds": 0,
            "quality_gate_history": [],
            "halt_reason": "No draft slots requested.",
        }
    scan_by_run_id = {}
    all_run_ids = []
    generated_count = 0
    round_no = 0
    quality_gate_history = []
    locked_target_quality_score = None
    while round_no < max_tries:
        round_no += 1
        open_slots = [s for s in slots if s["draft"] is None and s["attempts"] < max_tries]
        if not open_slots:
            break
        for slot in open_slots:
            slot["attempts"] += 1
            generated_count += 1
            status.info(
                f"Quality gate round {round_no}/{max_tries} · "
                f"slot {slot['slot_id']}/{total_slots} · "
                f"attempt {slot['attempts']}/{max_tries} · "
                f"generated {generated_count}"
            )
            stub = make_file_stub(prompt_id, slot["temp"], gen_model, outline_text)
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]
            payload_path = OUTPUTS_DIR / f"{stub}_run-{run_id}_PAYLOAD.txt"
            save_text(payload_path, payload_text)
            try:
                output = generate_chapter(client, gen_model, slot["temp"], message_blocks)
            except Exception as e:
                status.warning(
                    f"Generation failed for slot {slot['slot_id']} "
                    f"(T{slot['temp']} R{slot['rep']} A{slot['attempts']}): {e}"
                )
                slot["draft"] = None
                progress.progress(min(0.99, generated_count / max(total_slots * max_tries, 1)))
                continue
            output_path = OUTPUTS_DIR / f"{stub}_run-{run_id}_OUTPUT.txt"
            save_text(output_path, output)
            scan_result = scan_draft(output)
            scan_by_run_id[run_id] = scan_result
            all_run_ids.append(run_id)
            meta = {
                "run_id": run_id,
                "prompt_id": prompt_id,
                "temperature": slot["temp"],
                "model": gen_model,
                "repetition": slot["rep"],
                "attempt_round": round_no,
                "slot_id": slot["slot_id"],
                "slot_attempt": slot["attempts"],
                "timestamp": datetime.now().isoformat(),
                "documents": list(doc_uploads.keys()),
                "scan": scan_result,
            }
            meta_path = OUTPUTS_DIR / f"{stub}_run-{run_id}_META.json"
            save_text(meta_path, json.dumps(meta, indent=2))
            record = RunRecord(
                run_id=run_id,
                timestamp=datetime.now().isoformat(),
                prompt_id=prompt_id,
                prompt_text=prompt_text[:200],
                temperature=slot["temp"],
                model=gen_model,
                output_file=str(output_path),
                payload_file=str(payload_path),
                meta_file=str(meta_path),
                word_count=len(output.split()),
                pipeline_role="quality_gate_candidate",
                **{k: v for k, v in scan_result.items() if k in RUN_FIELDS},
            )
            append_record(csv_path, record)
            if github_cfg.get("configured"):
                try:
                    github_push_after_generation(
                        github_cfg, csv_path, output_path, payload_path, meta_path,
                    )
                except Exception as push_exc:
                    status.warning(f"GitHub push failed: {push_exc}")
            slot["draft"] = {
                "run_id": run_id,
                "text": output,
                "slot_id": slot["slot_id"],
                "attempt_round": round_no,
                "slot_attempt": slot["attempts"],
                "temperature": slot["temp"],
                "rep": slot["rep"],
            }
            progress.progress(min(0.99, generated_count / max(total_slots * max_tries, 1)))
            time.sleep(0.2)
        current_drafts = [s["draft"] for s in slots if s["draft"] is not None]
        if not current_drafts:
            continue
        try:
            lit = evaluate_drafts_with_anthropic(
                client, eval_model, current_drafts,
                outline_text=outline_text,
                scan_by_run_id=scan_by_run_id,
            )
        except Exception as eval_exc:
            import traceback as _tb
            _tb_text = _tb.format_exc()
            status.error(
                f"Quality-gate evaluator crashed. Failing open: marking all "
                f"{len(current_drafts)} drafts ACCEPTABLE so ranking can proceed.\n\n"
                f"Error: {eval_exc}\n\n"
                f"```\n{_tb_text[:3000]}\n```"
            )
            # v29 fail-open: synthesize a minimal lit-result so the rest of
            # the gate logic can run. Every draft becomes ACCEPTABLE with
            # score 0 and reason="gate_evaluator_failed_fail_open". The
            # downstream run_pipeline will still rank these drafts via
            # Stage F (and may call the evaluator again — if that call
            # also fails, it's caught by the existing try/except at the
            # run_pipeline call site, so this single failure mode is
            # fully covered).
            lit = {
                "raw_text": f"GATE_EVAL_FAILED: {eval_exc}\n\n{_tb_text[:2000]}",
                "ranking": list(range(1, len(current_drafts) + 1)),
                "winner_run_id": current_drafts[0]["run_id"],
                "quality_by_run_id": {
                    d["run_id"]: {
                        "verdict": "ACCEPTABLE",
                        "reason": "gate_evaluator_failed_fail_open",
                    }
                    for d in current_drafts
                },
                "quality_score_by_run_id": {
                    d["run_id"]: 0 for d in current_drafts
                },
                "top_quality_score": 0,
            }
        quality_by_run = lit.get("quality_by_run_id", {})
        quality_scores = lit.get("quality_score_by_run_id", {})
        acceptable_ids = [
            d["run_id"] for d in current_drafts
            if quality_by_run.get(d["run_id"], {}).get("verdict") != "UNACCEPTABLE"
        ]
        round_top_quality_score = max(
            (int(quality_scores.get(rid, 0) or 0) for rid in acceptable_ids),
            default=0,
        )
        if locked_target_quality_score is None:
            locked_target_quality_score = int(round_top_quality_score)
        top_quality_score = int(round_top_quality_score)
        target_quality_score = int(locked_target_quality_score or 0)
        # v23+ gate: retain every ACCEPTABLE draft. Scores are logged for
        # reference but no longer filter drafts out of the ranking pool.
        retained_ids = list(acceptable_ids)
        for slot in slots:
            d = slot.get("draft")
            if not d:
                continue
            rid = d["run_id"]
            update_record(csv_path, rid, {
                "quality_verdict": quality_by_run.get(rid, {}).get("verdict", ""),
                "quality_reason": (quality_by_run.get(rid, {}).get("reason", ""))[:500],
                "quality_score": int(quality_scores.get(rid, 0) or 0),
                "evaluation_id": f"qgate_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "evaluator_model": eval_model,
                "evaluation_raw": lit.get("raw_text", "")[:8000],
            })
            if rid in retained_ids:
                update_record(csv_path, rid, {"pipeline_role": "quality_gate_retained"})
            else:
                # Under the v23+ gate, the only non-retained case is UNACCEPTABLE.
                update_record(csv_path, rid, {"pipeline_role": "dropped_unacceptable"})
                slot["draft"] = None
        retained_count = len([s for s in slots if s["draft"] is not None])
        quality_gate_history.append({
            "round": round_no,
            "evaluated": len(current_drafts),
            "acceptable": len(acceptable_ids),
            "retained": retained_count,
            "top_quality_score": int(top_quality_score),
            "target_quality_score": int(target_quality_score),
            "retained_run_ids": retained_ids[:],
        })
        status.info(
            f"Quality gate round {round_no}/{max_tries} complete · "
            f"retained {retained_count}/{total_slots} at target writing score {target_quality_score} "
            f"(round top {top_quality_score})"
        )
        # v23+ gate: single pass. Dropped UNACCEPTABLE slots stay empty;
        # no refill, no retries. Cost is bounded and predictable.
        break
    final_drafts = [s["draft"] for s in slots if s["draft"] is not None]
    retained_run_ids = [d["run_id"] for d in final_drafts]
    halt_reason = ""
    if len(final_drafts) < total_slots:
        dropped = total_slots - len(final_drafts)
        halt_reason = (
            f"Quality gate retained {len(final_drafts)} of {total_slots} drafts "
            f"({dropped} dropped as UNACCEPTABLE). Shipping from retained set."
        )
    return {
        "final_drafts": final_drafts,
        "scan_by_run_id": scan_by_run_id,
        "all_run_ids": all_run_ids,
        "retained_run_ids": retained_run_ids,
        "generated_count": generated_count,
        "attempt_rounds": round_no,
        "quality_gate_history": quality_gate_history,
        "target_quality_score": int(locked_target_quality_score or 0),
        "halt_reason": halt_reason,
    }
# ============================================================================
# Literary evaluation — unchanged shape; adds strong-beat extraction
# ============================================================================
def evaluate_drafts_with_anthropic(
    client, model: str, drafts: list,
    outline_text: str = "", scan_by_run_id: dict = None,
) -> dict:
    n = len(drafts)
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
                f"aphoristic={scan.get('scan_aphoristic_count', '?')}, "
                f"backfill={scan.get('scan_backfill_count', '?')}, "
                f"verdict={scan.get('scan_verdict_count', '?')}, "
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
    for i in range(1, n + 1):
        if i not in quality_by_index:
            quality_by_index[i] = {
                "verdict": "ACCEPTABLE",
                "reason": "(no explicit verdict in evaluator output; defaulted to ACCEPTABLE)",
            }
    quality_score_by_index = {}
    score_pattern = re.compile(
        r"QUALITY_SCORE:\s*Draft\s*(\d+)\s*[—-]+\s*(-?\d+)(?=\n|$)",
        re.IGNORECASE,
    )
    for m in score_pattern.finditer(raw):
        idx = int(m.group(1))
        score = int(m.group(2))
        if 1 <= idx <= n:
            score = max(0, min(score, 10))
            if quality_by_index[idx]["verdict"] == "UNACCEPTABLE":
                score = 0
            quality_score_by_index[idx] = score
    parse_status = "clean"
    ranking = []
    rank_match = re.search(r"RANKING:\s*([0-9,\s]+)", raw)
    if rank_match:
        nums = [int(x.strip()) for x in rank_match.group(1).split(",") if x.strip().isdigit()]
        seen = set()
        deduped = []
        for x in nums:
            if 1 <= x <= n and x not in seen:
                seen.add(x)
                deduped.append(x)
        ranking = [x for x in deduped if quality_by_index[x]["verdict"] == "ACCEPTABLE"]
    else:
        parse_status = "no_ranking_line"
    if ranking:
        fallback_map = {}
        current = 10
        for idx in ranking:
            if idx not in fallback_map:
                fallback_map[idx] = current
                current = max(1, current - 1)
    else:
        acceptable_idxs = [
            i for i in range(1, n + 1)
            if quality_by_index[i]["verdict"] == "ACCEPTABLE"
        ]
        fallback_map = {idx: 10 for idx in acceptable_idxs}
    for i in range(1, n + 1):
        if i in quality_score_by_index:
            continue
        if quality_by_index[i]["verdict"] == "UNACCEPTABLE":
            quality_score_by_index[i] = 0
        else:
            quality_score_by_index[i] = fallback_map.get(i, 10)
            if parse_status == "clean":
                parse_status = "partial_missing_quality_score"
    acceptable_idxs = [
        i for i in range(1, n + 1)
        if quality_by_index[i]["verdict"] == "ACCEPTABLE"
    ]
    top_quality_score = max((quality_score_by_index[i] for i in acceptable_idxs), default=0)
    top_quality_idxs = [
        i for i in acceptable_idxs
        if quality_score_by_index[i] == top_quality_score
    ]
    # v23+ ranking: include every ACCEPTABLE draft in the ranking, preserving
    # the LLM's order. top_quality_idxs is still computed (and returned) for
    # reference, but it no longer filters the ranking pool.
    if ranking:
        ranking = [x for x in ranking if x in acceptable_idxs]
        missing_acceptable = [i for i in acceptable_idxs if i not in ranking]
        ranking += missing_acceptable
        if missing_acceptable and parse_status == "clean":
            parse_status = "partial"
    else:
        ranking = acceptable_idxs[:]
    if not ranking:
        ranking = acceptable_idxs[:]
    winner_match = re.search(r"WINNER:\s*(\d+)", raw)
    if winner_match:
        winner_idx = int(winner_match.group(1))
    else:
        winner_idx = ranking[0] if ranking else 1
        if parse_status == "clean":
            parse_status = "no_winner_line"
    if winner_idx not in ranking:
        winner_idx = ranking[0] if ranking else winner_idx
    winner_idx = max(1, min(winner_idx, n))
    winner_run_id = drafts[winner_idx - 1]["run_id"]
    quality_by_run_id = {}
    quality_score_by_run_id = {}
    for i, d in enumerate(drafts, 1):
        quality_by_run_id[d["run_id"]] = quality_by_index[i]
        quality_score_by_run_id[d["run_id"]] = int(quality_score_by_index[i])
    return {
        "winner_run_id": winner_run_id,
        "winner_index": winner_idx,
        "ranking": ranking,
        "quality_by_run_id": quality_by_run_id,
        "quality_by_index": quality_by_index,
        "quality_score_by_run_id": quality_score_by_run_id,
        "quality_score_by_index": quality_score_by_index,
        "top_quality_score": int(top_quality_score),
        "top_quality_indexes": top_quality_idxs,
        "raw_text": raw,
        "parse_status": parse_status,
        "model": model,
    }
# ============================================================================
# Line-graft — identify runner-up sentences, apply via string replacement
# ============================================================================
def parse_graft_candidates(raw: str) -> list:
    """Parse CANDIDATE blocks from the Stage-1 identification response.
    Returns list of dicts with keys:
      n, graft_type, unit, top1_text, donor_draft, donor_text,
      function, justification.
    v30: graft_type may be 'A', 'B', or 'C' (deletion). For Type C,
    donor_draft will be 0 and donor_text will be empty.
    """
    if "NO_CANDIDATES" in raw:
        return []
    candidates = []
    sections = re.split(r"CANDIDATE\s+(\d+)\s*\n", raw)
    # sections alternates: [preamble, "1", block1, "2", block2, ...]
    for i in range(1, len(sections), 2):
        try:
            n = int(sections[i])
        except ValueError:
            continue
        block = sections[i + 1] if i + 1 < len(sections) else ""
        type_m = re.search(r"TYPE:\s*([ABC])", block)
        unit_m = re.search(r"UNIT:\s*(sentence|phrase|run)", block, re.I)
        # TOP1_TEXT spans to the next DONOR_DRAFT: label
        top1_m = re.search(
            r'TOP1_TEXT:\s*"(.*?)"\s*(?=\n\s*DONOR_DRAFT:)',
            block, re.DOTALL,
        )
        donor_draft_m = re.search(r"DONOR_DRAFT:\s*(\d+)", block)
        # DONOR_TEXT spans to the next FUNCTION: label.
        # v30: Type C deletions emit DONOR_TEXT: "" — accept empty match.
        donor_text_m = re.search(
            r'DONOR_TEXT:\s*"(.*?)"\s*(?=\n\s*FUNCTION:)',
            block, re.DOTALL,
        )
        function_m = re.search(r"FUNCTION:\s*(.+)", block)
        justif_m = re.search(r"JUSTIFICATION:\s*(.+)", block)
        if top1_m and donor_text_m and donor_draft_m:
            candidates.append({
                "n": n,
                "graft_type": type_m.group(1) if type_m else "A",
                "unit": (unit_m.group(1).lower() if unit_m else "sentence"),
                "top1_text": top1_m.group(1).strip(),
                "donor_draft": int(donor_draft_m.group(1)),
                "donor_text": donor_text_m.group(1).strip(),
                "function": function_m.group(1).strip() if function_m else "",
                "justification": justif_m.group(1).strip() if justif_m else "",
            })
    return candidates
def parse_graft_commits(raw: str):
    """Parse COMMIT_CANDIDATE blocks and the FINAL_GRAFTS list from the
    Stage-2 commit response.
    Returns (commits, final_ids) where commits is a list of committed-graft
    dicts (DECISION=COMMIT only) with keys in the legacy shape:
      n, graft_type, unit, source_draft, replace, with_text, seam_edits,
      reason.
    v30: graft_type may be 'A', 'B', or 'C' (deletion). For Type C,
    source_draft will be 0 and with_text will be empty.
    """
    commits = []
    sections = re.split(r"COMMIT_CANDIDATE\s+(\d+)\s*\n", raw)
    for i in range(1, len(sections), 2):
        try:
            n = int(sections[i])
        except ValueError:
            continue
        block = sections[i + 1] if i + 1 < len(sections) else ""
        decision_m = re.search(r"DECISION:\s*(COMMIT|REJECT)", block, re.I)
        if not decision_m or decision_m.group(1).upper() != "COMMIT":
            continue
        type_m = re.search(r"TYPE:\s*([ABC])", block)
        unit_m = re.search(r"UNIT:\s*(sentence|phrase|run)", block, re.I)
        top1_m = re.search(
            r'TOP1_TEXT:\s*"(.*?)"\s*(?=\n\s*DONOR_DRAFT:)',
            block, re.DOTALL,
        )
        donor_draft_m = re.search(r"DONOR_DRAFT:\s*(\d+)", block)
        donor_text_m = re.search(
            r'DONOR_TEXT:\s*"(.*?)"\s*(?=\n\s*SEAM_EDITS:)',
            block, re.DOTALL,
        )
        seam_m = re.search(
            r"SEAM_EDITS:\s*(.+?)\s*(?=\n\s*REASON:|$)",
            block, re.DOTALL,
        )
        reason_m = re.search(r"REASON:\s*(.+)", block)
        if top1_m and donor_text_m and donor_draft_m:
            commits.append({
                "n": n,
                "graft_type": type_m.group(1) if type_m else "A",
                "unit": (unit_m.group(1).lower() if unit_m else "sentence"),
                "source_draft": int(donor_draft_m.group(1)),
                "replace": top1_m.group(1).strip(),
                "with_text": donor_text_m.group(1).strip(),
                "seam_edits": seam_m.group(1).strip() if seam_m else "none",
                "reason": reason_m.group(1).strip() if reason_m else "",
            })
    # FINAL_GRAFTS is authoritative when present.
    fg_m = re.search(r"FINAL_GRAFTS:\s*(.+?)(?:\n|$)", raw)
    if fg_m:
        fg_text = fg_m.group(1).strip()
        if fg_text.upper() == "NONE":
            return [], []
        final_ids = [int(x) for x in re.findall(r"\d+", fg_text)]
        commits = [c for c in commits if c["n"] in final_ids]
        return commits, final_ids
    # No FINAL_GRAFTS — trust per-candidate DECISIONs.
    return commits, [c["n"] for c in commits]
def _format_candidates_for_commit(candidates: list) -> str:
    """Render the Stage-1 candidate list into the CANDIDATES_BLOCK section
    injected into the Stage-2 commit prompt.
    """
    lines = []
    for c in candidates:
        lines.append(f"CANDIDATE {c['n']}")
        lines.append(f"TYPE: {c['graft_type']}")
        lines.append(f"UNIT: {c['unit']}")
        lines.append(f'TOP1_TEXT: "{c["top1_text"]}"')
        lines.append(f"DONOR_DRAFT: {c['donor_draft']}")
        lines.append(f'DONOR_TEXT: "{c["donor_text"]}"')
        lines.append(f"FUNCTION: {c['function']}")
        lines.append(f"JUSTIFICATION: {c['justification']}")
        lines.append("")
    return "\n".join(lines)
def _build_winner_flags_text(winner_scan: dict) -> str:
    """Turn the winner's scan_flagged_passages JSON into human-readable lines
    for injection into the line-graft prompt.
    v27: multi-sentence flags (staccato runs, anaphoric clusters) carry a
    'verbatim' field with the exact run text. The grafter is instructed to
    replace the entire run with one flowing sentence from a runner-up draft.
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
    multi_sent_rules = {
        "staccato_triplet", "staccato_density",
        "anaphoric_escalation", "aphoristic_couplet",
        # v30 — multi-sentence patterns the grafter must treat as one unit
        "tautological_restatement",  # cross-sentence noun-restate / triple self-affirmation
        "sensory_fragment_pair",     # always two sentences
        "procedural_chain",          # cross-sentence I-verb run
    }
    lines = []
    for f in flags:
        rule = f.get("rule", "?")
        ctx = f.get("context", "").strip()
        verbatim = (f.get("verbatim") or "").strip()
        if rule in multi_sent_rules and verbatim:
            lines.append(
                f"- [{rule}] MULTI-SENTENCE RUN — replace the WHOLE run as ONE "
                f"replacement (find a single flowing sentence from a runner-up "
                f"draft that delivers the same beat):"
            )
            lines.append(f"    VERBATIM TO REPLACE: {verbatim}")
        else:
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
            f"emotion_naming={scan.get('scan_emotion_naming_count', 0)}, "
            f"aphoristic={scan.get('scan_aphoristic_count', 0)}, "
            f"backfill={scan.get('scan_backfill_count', 0)}, "
            f"verdict={scan.get('scan_verdict_count', 0)}, "
            f"scene_appos={scan.get('scan_scene_set_appos_count', 0)}, "
            f"taut_restate={scan.get('scan_tautological_restate_count', 0)}, "
            f"sensory_pair={scan.get('scan_sensory_pair_count', 0)}, "
            f"proc_chain={scan.get('scan_procedural_chain_count', 0)}, "
            f"felt_like={scan.get('scan_felt_like_abstract_count', 0)}, "
            f"which_meta={scan.get('scan_which_was_meta_count', 0)}, "
            f"but_only={scan.get('scan_but_only_because_count', 0)}, "
            f"kind_kind={scan.get('scan_kind_the_kind_count', 0)}, "
            f"so_aux={scan.get('scan_so_aux_inversion_count', 0)}, "
            f"observer={scan.get('scan_observer_ing_count', 0)}, "
            f"who_decided={scan.get('scan_who_had_decided_count', 0)}"
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
    if APHORISTIC_STANDALONE_PATTERN.search(donor_sentence):
        return False
    if EXPLANATORY_BACKFILL_PATTERN.search(donor_sentence):
        return False
    # "X too Y for Z" — same quote-count discipline
    for m in VERDICT_TOO_FOR_PATTERN.finditer(donor_sentence):
        before = donor_sentence[: m.start()]
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 == 0:
            return False
    # v26 additions — reject donors carrying any of the new patterns
    if NOT_BRIDGE_PATTERN.search(donor_sentence):
        return False
    for m in VERDICT_KIND_OF_PATTERN.finditer(donor_sentence):
        before = donor_sentence[: m.start()]
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 == 0:
            return False
    if TRIPLE_NOUN_PATTERN.search(donor_sentence):
        return False
    if I_NAMED_PATTERN.search(donor_sentence):
        return False
    # v27 additions
    # Polysyndetic (4+ coordinating ands in this single sentence)
    and_count = len(re.findall(r"\band\b", donor_sentence, re.IGNORECASE))
    if and_count >= 4:
        return False
    # Mid-sentence triple appositive
    for m in MID_APPOSITIVE_PATTERN.finditer(donor_sentence):
        phrases = [m.group(1).strip(), m.group(2).strip(), m.group(3).strip()]
        if any(_EVAL_HEAD_RE.match(p.split()[0]) for p in phrases if p.split()):
            return False
    # Tautological loop
    if TAUTOLOGY_PATTERN.search(donor_sentence):
        return False
    # Aphoristic generalization template
    if _is_aphorism(donor_sentence):
        return False
    # v30 additions — reject donors carrying any of the new patterns.
    # Scene-set appositive: the five marker patterns at sentence level.
    # Length threshold is 18 (matches scanner). For comma-required markers
    # (A/B/D) we need a comma; for optional markers (C/E) we don't.
    if _word_count(donor_sentence) >= 18:
        # Comma-required markers
        if "," in donor_sentence:
            first_comma = donor_sentence.find(",")
            head_clause = donor_sentence[:first_comma]
            if _word_count(head_clause) >= 8:
                for pat in (
                    SCENE_SET_APPOS_MARKER_A,
                    SCENE_SET_APPOS_MARKER_B,
                    SCENE_SET_APPOS_MARKER_D,
                ):
                    if pat.search(donor_sentence):
                        return False
        # Comma-optional markers
        for pat in (SCENE_SET_APPOS_MARKER_C, SCENE_SET_APPOS_MARKER_E):
            m = pat.search(donor_sentence)
            if m:
                head_wc = _word_count(donor_sentence[:m.start()])
                if head_wc >= 5:
                    return False
    # Tautological place restatement (within-sentence)
    if TAUTOLOGICAL_PLACE_PATTERN.search(donor_sentence):
        return False
    # Procedural chain (within-sentence form)
    for m in PROCEDURAL_CHAIN_PATTERN.finditer(donor_sentence):
        before = donor_sentence[: m.start()]
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 == 0:
            return False
    # v31 evasions — same dialogue gating as base patterns
    for pat in (FELT_LIKE_ABSTRACT, BUT_ONLY_BECAUSE, KIND_THE_KIND):
        for m in pat.finditer(donor_sentence):
            before = donor_sentence[: m.start()]
            normalized = before.replace("\u201c", '"').replace("\u201d", '"')
            if normalized.count('"') % 2 == 0:
                return False
    # WHICH_WAS_META, THE_OBSERVER_ING, WHO_HAD_DECIDED — narration-only,
    # no dialogue allowance (these constructions are not used in PNR
    # dialogue, so any hit is a fail regardless of position)
    if WHICH_WAS_META.search(donor_sentence):
        return False
    if THE_OBSERVER_ING.search(donor_sentence):
        return False
    if WHO_HAD_DECIDED.search(donor_sentence):
        return False
    # SO_AUX_INVERSION is a multi-sentence pattern; not checked here,
    # only at passage level.
    # Note: staccato runs, anaphoric escalation, sensory pairs, cross-sentence
    # tautological restatement, and cross-sentence procedural runs are
    # multi-sentence properties — they cannot be evaluated against a single
    # isolated donor sentence. Run-level filtering happens in the grafter
    # when it composes a donor passage; see _donor_passage_is_clean below.
    return True
def _donor_passage_is_clean(donor_text: str) -> bool:
    """v27 — passage-level cleanliness check for multi-sentence donors,
    used when the grafter is replacing a staccato run or anaphoric
    cluster (one donor passage spanning 2+ sentences). Rejects passages
    that contain the same multi-sentence patterns we're trying to graft
    OUT of the winner, plus all single-sentence donor failures."""
    sentence_spans = _split_sentences_with_offsets(donor_text)
    # Single-sentence checks on every sentence
    for (_, _, s) in sentence_spans:
        if not _donor_sentence_is_clean(s):
            return False
    # Multi-sentence checks
    staccato, _, _ = _scan_staccato(donor_text, sentence_spans)
    if staccato:
        return False
    if _scan_anaphoric_escalation(sentence_spans):
        return False
    if _scan_aphoristic_couplets(sentence_spans, donor_text):
        return False
    # v30: passage-level v30 multi-sentence patterns
    if _scan_tautological_restatement(sentence_spans, donor_text):
        return False
    if _scan_sensory_fragment_pair(sentence_spans, donor_text):
        return False
    if _scan_procedural_chain(sentence_spans, donor_text):
        return False
    if _scan_scene_set_appositive(sentence_spans, donor_text):
        return False
    # v31: passage-level — so_aux_inversion is multi-sentence by nature
    if SO_AUX_INVERSION.search(donor_text):
        return False
    return True
def run_line_graft_experiment(
    client,
    eval_model: str,
    drafts_ranked: list,
    scan_by_run_id: dict,
    batch_stub: str,
) -> dict:
    """Identify runner-up sentences or clauses that improve TOP 1, judge
    each for commit, then apply the committed set via deterministic string
    replacement.
    Two-stage LLM pass + one deterministic substitution step:
      Stage 1 (LLM): wide-net candidate identification.
      Stage 2 (LLM): commit/reject per candidate with seam-edit handling.
      Stage 3 (code): find-and-replace on TOP 1 for each committed graft.
    Two graft pathways:
      Type A — Flag Repair: TOP 1 carries a flagged construction at the
               same narrative function as a clean donor.
      Type B — Quality Upgrade: donor is meaningfully better at the same
               narrative function, regardless of whether TOP 1 is flagged.
    Two graft units:
      sentence — whole-sentence replacement.
      phrase   — clause-level replacement inside a TOP 1 sentence.
    Args:
        drafts_ranked: list of draft dicts in ranking order
                       (index 0 = TOP 1). Each has 'run_id' and 'text'.
        scan_by_run_id: per-draft mechanical scan dict. The winner's
                        scan_flagged_passages gets injected into the
                        Stage-1 prompt as known weak spots; donor texts
                        that themselves carry hard-cap patterns get
                        rejected deterministically.
        batch_stub: for file naming.
    Returns dict with:
      - grafted: bool
      - grafts: list of applied graft dicts
      - grafts_attempted: Stage-1 candidates (before commit judging)
      - grafts_rejected_commit: candidates rejected at Stage 2
      - grafts_rejected_dirty_donor: commits rejected for donor flag
      - grafts_rejected_no_match: commits whose TOP1_TEXT didn't appear
                                  in TOP 1 verbatim
      - grafted_text: the modified TOP 1 text (empty if no grafts applied)
      - grafted_path: file path (empty if no grafts applied)
      - grafted_scan: mechanical scan of the grafted output (diagnostic)
      - raw_candidates: full Stage-1 model output
      - raw_commits: full Stage-2 model output
      - raw: concatenation of both, for the existing UI expander
    """
    result = {
        "grafted": False,
        "grafts": [],
        "grafts_attempted": [],
        "grafts_rejected_commit": [],
        "grafts_rejected_dirty_donor": [],
        "grafts_rejected_no_match": [],
        "grafted_text": "",
        "grafted_path": "",
        "grafted_scan": None,
        "raw_candidates": "",
        "raw_commits": "",
        "raw": "",
    }
    n = len(drafts_ranked)
    if n < 2:
        return result
    winner_run_id = drafts_ranked[0]["run_id"]
    winner_scan = (scan_by_run_id or {}).get(winner_run_id, {})
    winner_flags_text = _build_winner_flags_text(winner_scan)
    scanner_summary_text = _build_scanner_summary_text(
        drafts_ranked, scan_by_run_id,
    )
    # Build the drafts block once — reused for both Stage-1 and Stage-2 calls.
    drafts_block_parts = []
    for i, d in enumerate(drafts_ranked, 1):
        label = "WINNER" if i == 1 else "RUNNER-UP"
        drafts_block_parts.append(
            f"\n\n=== DRAFT {i} ({label}, run_id: {d['run_id']}) ===\n\n{d['text']}"
        )
    drafts_block = "".join(drafts_block_parts)
    # --- Stage 1: candidate identification ---
    cand_prompt = LINE_GRAFT_CANDIDATE_PROMPT.format(
        N=n,
        winner_flags=winner_flags_text,
        scanner_summary=scanner_summary_text,
    )
    resp1 = client.messages.create(
        model=eval_model,
        max_tokens=MAX_EVAL_TOKENS,
        messages=[{"role": "user", "content": cand_prompt + drafts_block}],
    )
    raw_candidates = "\n".join(
        b.text for b in resp1.content if getattr(b, "text", None)
    )
    result["raw_candidates"] = raw_candidates
    candidates = parse_graft_candidates(raw_candidates)
    result["grafts_attempted"] = list(candidates)
    if not candidates:
        result["raw"] = raw_candidates
        return result
    # --- Stage 2: commit decisions ---
    candidates_block = _format_candidates_for_commit(candidates)
    commit_prompt = LINE_GRAFT_COMMIT_PROMPT.format(
        N=n,
        candidates_block=candidates_block,
    )
    resp2 = client.messages.create(
        model=eval_model,
        max_tokens=MAX_EVAL_TOKENS,
        messages=[{"role": "user", "content": commit_prompt + drafts_block}],
    )
    raw_commits = "\n".join(
        b.text for b in resp2.content if getattr(b, "text", None)
    )
    result["raw_commits"] = raw_commits
    result["raw"] = (
        "=== STAGE 1: CANDIDATE IDENTIFICATION ===\n\n"
        + raw_candidates
        + "\n\n=== STAGE 2: COMMIT DECISIONS ===\n\n"
        + raw_commits
    )
    commits, _final_ids = parse_graft_commits(raw_commits)
    # Track candidates rejected at commit stage (identified but not committed).
    committed_n_set = {c["n"] for c in commits}
    for cand in candidates:
        if cand["n"] not in committed_n_set:
            result["grafts_rejected_commit"].append({
                "source_draft": cand["donor_draft"],
                "replace": cand["top1_text"],
                "with_text": cand["donor_text"],
                "reason": "rejected at commit stage",
                "graft_type": cand["graft_type"],
                "unit": cand["unit"],
            })
    if not commits:
        return result
    # Filter dirty donors — enforces the clean-donor rule deterministically
    # in case Stage 2 misjudges its own candidate.
    # v27: for unit=run grafts (multi-sentence replacements for staccato or
    # anaphoric flags) we use the passage-level cleanliness check, since a
    # multi-sentence donor can carry run-level patterns invisible to the
    # single-sentence screen. For unit=sentence and unit=phrase grafts, the
    # single-sentence screen is sufficient.
    # v30: Type C deletions skip the donor cleanliness check — there is no
    # donor text. The replace text just gets removed. We do still validate
    # that the deletion's with_text is empty (as the prompt mandates).
    # When STAGE_G_TYPE_C_ENABLED is False, Type C commits are coerced to
    # rejection so v30 can be run in graft-only mode for comparison.
    clean_commits = []
    type_c_committed = 0
    for c in commits:
        unit = (c.get("unit") or "").lower()
        donor = c["with_text"]
        graft_type = (c.get("graft_type") or "A").upper()
        if graft_type == "C":
            if not STAGE_G_TYPE_C_ENABLED:
                c_copy = dict(c)
                c_copy["reason"] = "type_c_disabled_by_flag"
                result["grafts_rejected_commit"].append(c_copy)
                continue
            if type_c_committed >= STAGE_G_TYPE_C_MAX_DELETIONS:
                c_copy = dict(c)
                c_copy["reason"] = (
                    f"type_c_cap_reached ({STAGE_G_TYPE_C_MAX_DELETIONS})"
                )
                result["grafts_rejected_commit"].append(c_copy)
                continue
            # Deletion: with_text must be empty (or near-empty whitespace);
            # accept seam-edited rephrases of trailing punctuation only.
            if donor.strip() == "" or len(donor.strip()) <= 3:
                clean_commits.append(c)
                type_c_committed += 1
            else:
                # If the model emitted a non-empty DONOR_TEXT for a Type C,
                # treat it as a Type-A/B replacement and run cleanliness
                # checks on it.
                if unit == "run":
                    ok = _donor_passage_is_clean(donor)
                else:
                    ok = _donor_sentence_is_clean(donor)
                if ok:
                    clean_commits.append(c)
                    type_c_committed += 1
                else:
                    result["grafts_rejected_dirty_donor"].append(c)
            continue
        if unit == "run":
            ok = _donor_passage_is_clean(donor)
        else:
            ok = _donor_sentence_is_clean(donor)
        if ok:
            clean_commits.append(c)
        else:
            result["grafts_rejected_dirty_donor"].append(c)
    if not clean_commits:
        return result
    # Apply grafts via deterministic string replacement.
    # v30: Type C deletions remove the matched text and tidy up paragraph
    # whitespace. Replacement strings for Type C are always the empty
    # string unless the model supplied a short seam-edit replacement.
    winner_text = drafts_ranked[0]["text"]
    grafted_text = winner_text
    applied = []
    for c in clean_commits:
        graft_type = (c.get("graft_type") or "A").upper()
        replace_text = c["replace"]
        with_text = c["with_text"]
        if graft_type == "C":
            # Strip a leading space from the replace target if removing
            # this passage would leave a double space in the surrounding
            # prose. Walk the surrounding text after substitution and
            # collapse any run of two or more spaces / blank lines to one.
            if replace_text in grafted_text:
                idx = grafted_text.find(replace_text)
                grafted_text = (
                    grafted_text[:idx]
                    + with_text
                    + grafted_text[idx + len(replace_text):]
                )
                # Collapse double spaces around the cut
                grafted_text = re.sub(r"  +", " ", grafted_text)
                # Collapse 3+ consecutive newlines to two (preserve para breaks)
                grafted_text = re.sub(r"\n{3,}", "\n\n", grafted_text)
                applied.append(c)
            else:
                result["grafts_rejected_no_match"].append(c)
            continue
        if replace_text in grafted_text:
            grafted_text = grafted_text.replace(replace_text, with_text, 1)
            applied.append(c)
        else:
            result["grafts_rejected_no_match"].append(c)
    if not applied:
        return result
    result["grafted"] = True
    result["grafts"] = applied
    result["grafted_text"] = grafted_text
    # Diagnostic scan of the grafted output — reported but not gated.
    result["grafted_scan"] = scan_draft(grafted_text)
    grafted_path = FINAL_DIR / f"WINNER_GRAFTED_{batch_stub}_RANK-01_WINNER_GRAFTED_run-{drafts_ranked[0]['run_id']}.txt"
    save_text(grafted_path, grafted_text)
    result["grafted_path"] = str(grafted_path)
    return result
# ============================================================================
# File naming
# ============================================================================
def sanitize_filename_part(value: str, max_len: int = 80) -> str:
    """Return a compact Windows-safe filename component."""
    value = str(value or "").strip()
    value = re.sub(r"[\r\n\t]+", " ", value)
    value = re.sub(r"[^A-Za-z0-9._() -]+", "", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._- ")
    if not value:
        value = "no_heading"
    return value[:max_len].strip("._- ") or "no_heading"


def extract_outline_label(outline_text: str) -> str:
    """Return a short outline label for filenames.

    Numbered chapter headings collapse to CH_<number>, so
    "Chapter 3: Departure" and "CH 3 - Departure" both become
    "CH_3" after filename sanitizing. If no chapter number is found,
    fall back to a compact sanitized heading.
    """
    if not outline_text:
        return "no_outline_heading"

    raw_lines = [ln.strip() for ln in str(outline_text).splitlines()]
    lines = [ln.strip().strip("#*-").strip() for ln in raw_lines if ln.strip()]

    numbered_chapter_patterns = [
        r"\bchapter\s*(\d{1,3})\b",
        r"\bch\.?\s*(\d{1,3})\b",
        r"^\s*(\d{1,3})\s*[.)\-:–—]\s+",
    ]
    for line in lines:
        if len(line) > 120:
            continue
        for pat in numbered_chapter_patterns:
            m = re.search(pat, line, flags=re.IGNORECASE)
            if m:
                return sanitize_filename_part(f"CH {m.group(1)}", 16)

    heading_skip = r"\b(words?|target|global drafting controls|drafting controls|outline)\b"
    for line in lines:
        if 3 <= len(line) <= 90 and not re.search(heading_skip, line, re.IGNORECASE):
            return sanitize_filename_part(line, 36)

    return "outline_heading"


def make_file_stub(prompt_id: int, temperature: float, model: str, outline_text: str = "") -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = sanitize_filename_part(model.split("-")[-1][:6] if "-" in model else model[:6], 20)
    temp_label = sanitize_filename_part(str(temperature).replace(".", "p"), 16)
    outline_label = extract_outline_label(outline_text)
    return f"{APP_VERSION}_P{prompt_id}_{outline_label}_DRAFT_T{temp_label}_{model_short}_{ts}"


def make_winner_filename(prompt_id: int, temperature: float, model: str, outline_text: str = "") -> str:
    stub = make_file_stub(prompt_id, temperature, model, outline_text)
    return f"{stub}_RANK-01_WINNER.txt"


def make_batch_stub(batch_timestamp: str, prompt_id: int = 0, outline_text: str = "") -> str:
    outline_label = extract_outline_label(outline_text)
    return f"{APP_VERSION}_P{prompt_id}_{outline_label}_{batch_timestamp}"
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
        + scan.get("scan_aphoristic_count", 0)
        + scan.get("scan_backfill_count", 0)
        + scan.get("scan_verdict_count", 0)
        + scan.get("scan_not_bridge_count", 0)
        + scan.get("scan_verdict_kind_of_count", 0)
        + scan.get("scan_triple_noun_count", 0)
        # i_named_count intentionally NOT summed — diagnostic only
        + scan.get("scan_staccato_count", 0)
        + scan.get("scan_polysyndetic_count", 0)
        + scan.get("scan_aphoristic_couplet_count", 0)
        + scan.get("scan_mid_appositive_count", 0)
        + scan.get("scan_tautology_count", 0)
        + scan.get("scan_anaphoric_escalation_count", 0)
        + scan.get("scan_scene_set_appos_count", 0)
        + scan.get("scan_tautological_restate_count", 0)
        + scan.get("scan_sensory_pair_count", 0)
        + scan.get("scan_procedural_chain_count", 0)
        + scan.get("scan_felt_like_abstract_count", 0)
        + scan.get("scan_which_was_meta_count", 0)
        + scan.get("scan_but_only_because_count", 0)
        + scan.get("scan_kind_the_kind_count", 0)
        + scan.get("scan_so_aux_inversion_count", 0)
        + scan.get("scan_observer_ing_count", 0)
        + scan.get("scan_who_had_decided_count", 0)
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
    """Run the full pipeline on a batch of already quality-gated drafts."""
    result = {
        "halt": False,
        "halt_reason": "",
        "quality_by_run_id": {},
        "quality_score_by_run_id": {},
        "acceptable_run_ids": [],
        "dropped_run_ids": [],
        "retained_run_ids": [],
        "discarded_below_top_quality_ids": [],
        "top_quality_score": 0,
        "pipeline_ranking": [],
        "literary_ranking": [],
        "literary_winner_run_id": "",
        "ai_scores_by_run_id": {},
        "top1_run_id": "",
        "final_text": "",
        "final_path": "",
        "final_source": "",
        "top_paths": [],
        "ranking_manifest_path": "",
        "eval_raw": "",
        "line_graft": {},
        "final_pass": {},
        "line_edit": {},
        "stage_f": {},
        "stage_f_batch": {},
    }
    lit = evaluate_drafts_with_anthropic(
        client, eval_model, drafts,
        outline_text=outline_text,
        scan_by_run_id=scan_by_run_id,
    )
    result["eval_raw"] = lit["raw_text"]
    result["quality_by_run_id"] = lit.get("quality_by_run_id", {})
    result["quality_score_by_run_id"] = lit.get("quality_score_by_run_id", {})
    result["top_quality_score"] = int(lit.get("top_quality_score", 0) or 0)
    lit_ranking_ids = [drafts[i - 1]["run_id"] for i in lit["ranking"]]
    result["literary_ranking"] = lit_ranking_ids
    result["literary_winner_run_id"] = lit["winner_run_id"]
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
    if not acceptable_ids:
        result["halt"] = True
        result["halt_reason"] = (
            "No draft cleared the quality floor. "
            "Regenerate this batch — the pipeline will not ship an unacceptable draft."
        )
        return result
    quality_scores = result["quality_score_by_run_id"]
    top_quality_score = max((int(quality_scores.get(rid, 0) or 0) for rid in acceptable_ids), default=0)
    # v23+ pipeline: retain every ACCEPTABLE draft for ranking and downstream
    # stages. The old locked-target filter has been removed to match the new
    # quality-gate semantics (gate keeps all acceptable; ship TOP 1 by ranking).
    retained_ids = acceptable_ids[:]
    discarded_below_top_quality_ids = []
    result["top_quality_score"] = int(top_quality_score)
    result["retained_run_ids"] = retained_ids
    result["discarded_below_top_quality_ids"] = discarded_below_top_quality_ids
    predictor = stage_f_load_predictor(str(LABELED_CORPUS_PATH), STAGE_F_RIDGE_LAMBDA)
    retained_drafts = [d for d in drafts if d["run_id"] in retained_ids]
    ai_scores = []
    if predictor.get("available"):
        for d in retained_drafts:
            pred = stage_f_predict_detailed(d.get("text", ""), predictor)
            ai_scores.append({
                "run_id": d.get("run_id", ""),
                "label": d.get("run_id", ""),
                "prediction": pred,
            })
        ai_scores.sort(
            key=lambda item: (
                item.get("prediction", {}).get("raw_score")
                if item.get("prediction", {}).get("raw_score") is not None else -1e9,
                item.get("prediction", {}).get("predicted_score")
                if item.get("prediction", {}).get("predicted_score") is not None else -1e9,
            ),
            reverse=True,
        )
        pipeline_ranking = [item["run_id"] for item in ai_scores]
        result["ai_scores_by_run_id"] = {
            item["run_id"]: item.get("prediction", {}) for item in ai_scores
        }
    else:
        pipeline_ranking = [rid for rid in lit_ranking_ids if rid in retained_ids]
        if not pipeline_ranking:
            pipeline_ranking = retained_ids[:]
        result["ai_scores_by_run_id"] = {}
    if not pipeline_ranking:
        pipeline_ranking = retained_ids[:]
    result["pipeline_ranking"] = pipeline_ranking
    result["top1_run_id"] = pipeline_ranking[0]
    top_paths = []
    ranking_lines = ["AI RANKING — BEST TO WORST", "=" * 60]
    for rank_pos, run_id in enumerate(pipeline_ranking[:top_n], 1):
        draft_obj = next((d for d in drafts if d["run_id"] == run_id), None)
        if draft_obj is None:
            continue
        top_filename = (
            f"{batch_stub}_RANK-01_WINNER_run-{run_id}.txt"
            if rank_pos == 1
            else f"{batch_stub}_RANK-{rank_pos:02d}_run-{run_id}.txt"
        )
        top_path = FINAL_DIR / top_filename
        save_text(top_path, draft_obj["text"])
        top_paths.append(top_path)
        pred_obj = result.get("ai_scores_by_run_id", {}).get(run_id, {}) or {}
        pred_score = pred_obj.get("predicted_score")
        raw_score = pred_obj.get("raw_score")
        band = pred_obj.get("band", "")
        bits = [f"#{rank_pos}", run_id]
        if pred_score is not None:
            bits.append(f"pred={pred_score}")
        if raw_score is not None:
            bits.append(f"raw={raw_score}")
        if band:
            bits.append(f"band={band}")
        bits.append(top_filename)
        ranking_lines.append(" | ".join(str(b) for b in bits))
    result["top_paths"] = top_paths
    ranking_lines.insert(2, f"WINNER_RUN_ID: {result["top1_run_id"]}")
    ranking_lines.insert(3, f"WINNER_FILE: {batch_stub}_RANK-01_WINNER_run-{result['top1_run_id']}.txt")
    ranking_manifest_path = FINAL_DIR / f"AI_RANKING_{batch_stub}.txt"
    save_text(ranking_manifest_path, "\n".join(ranking_lines))
    result["ranking_manifest_path"] = str(ranking_manifest_path)
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
    lg = result.get("line_graft") or {}
    if lg.get("grafted") and lg.get("grafted_text"):
        result["final_source"] = "top1_grafted"
        result["final_text"] = lg["grafted_text"]
        final_path = FINAL_DIR / f"FINAL_{batch_stub}_RANK-01_WINNER_GRAFTED_run-{result['top1_run_id']}.txt"
        save_text(final_path, lg["grafted_text"])
        result["final_path"] = str(final_path)
    else:
        result["final_source"] = "top1_ungrafted"
        result["final_text"] = top1_text
        final_path = FINAL_DIR / f"FINAL_{batch_stub}_RANK-01_WINNER_UNGRAFTED_run-{result['top1_run_id']}.txt"
        save_text(final_path, top1_text)
        result["final_path"] = str(final_path)
    acceptable_drafts_in_rank_order = []
    for run_id in pipeline_ranking:
        draft_obj = next((d for d in drafts if d["run_id"] == run_id), None)
        if draft_obj:
            acceptable_drafts_in_rank_order.append(draft_obj)
    if len(acceptable_drafts_in_rank_order) >= 2:
        final_pass = run_final_pass(
            client, eval_model, acceptable_drafts_in_rank_order,
            outline_text, batch_stub,
        )
        result["final_pass"] = final_pass
    try:
        top1_id = result.get("top1_run_id", "")
        le = run_line_edit_pass(
            client, eval_model,
            result.get("final_text", ""),
            acceptable_drafts_in_rank_order,
            top1_id,
            batch_stub,
        )
        result["line_edit"] = le
        if le.get("changed") and le.get("edited_text"):
            result["final_text_lineedited"] = le["edited_text"]
    except Exception as e:
        result["line_edit"] = {
            "ran": False,
            "enabled": STAGE_G_ENABLED,
            "error": f"Stage G crashed: {e}",
            "original_text": result.get("final_text", ""),
            "edited_text": result.get("final_text", ""),
            "mechanical": {},
            "flagged_count": 0,
            "flagged_sentences": [],
            "g4": {},
            "edited_path": "",
            "report_path": "",
            "changed": False,
        }
    try:
        predictor = stage_f_load_predictor(str(LABELED_CORPUS_PATH), STAGE_F_RIDGE_LAMBDA)
        batch_scores = []
        for d in drafts:
            batch_scores.append({
                "run_id": d.get("run_id", ""),
                "label": d.get("run_id", ""),
                "prediction": stage_f_predict_detailed(d.get("text", ""), predictor),
            })
        batch_scores.sort(
            key=lambda item: (
                item.get("prediction", {}).get("raw_score")
                if item.get("prediction", {}).get("raw_score") is not None else -1e9
            ),
            reverse=True,
        )
        text_for_prediction = result.get(
            "final_text_lineedited",
            result.get("final_text", ""),
        )
        result["stage_f"] = stage_f_predict(text_for_prediction, predictor)
        final_label = "FINAL_LINEEDITED" if result.get("final_text_lineedited") else "FINAL"
        batch_debug_items = list(batch_scores) + [{
            "run_id": "",
            "label": final_label,
            "prediction": result["stage_f"],
        }]
        debug_path = write_stage_f_debug_report(
            predictor=predictor,
            scored_items=batch_debug_items,
            batch_stub=batch_stub,
            top1_run_id=result.get("top1_run_id", ""),
            final_run_label=final_label,
        )
        result["stage_f_batch"] = {
            "available": predictor.get("available", False),
            "report_path": debug_path,
            "scores": batch_scores,
        }
    except Exception as e:
        result["stage_f"] = {
            "available": False,
            "reason": f"Stage F crashed: {e}",
            "predicted_score": None,
            "band": "UNAVAILABLE",
            "n_train": 0,
            "loo_mae": float("nan"),
            "features": {},
        }
        result["stage_f_batch"] = {
            "available": False,
            "reason": f"Stage F crashed: {e}",
            "report_path": "",
            "scores": [],
        }
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
    lines.append("Q1 — QUALITY FLOOR / TOP WRITING SCORE")
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
    lines.append("(Scan used for scanner veto check — literary ranking drives Q2 selection.)")
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
    lines.append("Q2 — AI RANKING OF RETAINED TOP-WRITING DRAFTS")
    lines.append("=" * 60)
    lines.append("Only drafts tied at the highest writing score are ranked here. AI ranking then orders that retained set.")
    lines.append("")
    top1_id = pipeline_result.get("top1_run_id", "")
    lit_winner = pipeline_result.get("literary_winner_run_id", "")
    scanner_veto = pipeline_result.get("scanner_veto", {})
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
        if run_id in scanner_veto.get("vetoed_run_ids", []):
            marker += " [SCANNER VETOED — literary winner but violations too high]"
        elif run_id == lit_winner and run_id == top1_id:
            marker += " [literary winner]"
        lines.append(f"  {rank_pos}. {run_id} (violations={violations}){marker}")
    if scanner_veto:
        lines.append("")
        lines.append(
            f"Scanner veto fired: literary winner had violations > "
            f"{scanner_veto.get('veto_threshold', '?')} "
            f"(threshold = 2× batch median of {scanner_veto.get('median_violations', '?')}). "
            f"Next literary-ranked draft promoted to TOP 1."
        )
    elif lit_winner and lit_winner == top1_id:
        lines.append("")
        lines.append("Literary winner shipped as TOP 1 (no scanner veto).")
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
        rejected_commit = len(lg.get("grafts_rejected_commit", []))
        rejected_dirty = len(lg.get("grafts_rejected_dirty_donor", []))
        rejected_no_match = len(lg.get("grafts_rejected_no_match", []))
        lines.append(
            f"Grafts applied: {applied} of {attempted} candidates identified "
            f"({rejected_commit} rejected at commit; "
            f"{rejected_dirty} rejected — donor carried hard-cap pattern; "
            f"{rejected_no_match} rejected — REPLACE text not in TOP 1)"
        )
        for i, g in enumerate(lg["grafts"], 1):
            gtype = g.get("graft_type", "A")
            unit = g.get("unit", "sentence")
            type_label = "Flag Repair" if gtype == "A" else "Quality Upgrade"
            lines.append(
                f"  Graft {i} — Type {gtype} ({type_label}), "
                f"{unit}-level, from Draft {g['source_draft']}:"
            )
            lines.append(f"    Replaced: {g['replace'][:120]}{'...' if len(g['replace']) > 120 else ''}")
            lines.append(f"    With:     {g['with_text'][:120]}{'...' if len(g['with_text']) > 120 else ''}")
            seam = g.get("seam_edits", "none")
            if seam and seam.lower() != "none":
                lines.append(f"    Seam:     {seam}")
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
        rejected_commit = len(lg.get("grafts_rejected_commit", []))
        rejected_dirty = len(lg.get("grafts_rejected_dirty_donor", []))
        rejected_no_match = len(lg.get("grafts_rejected_no_match", []))
        lines.append(f"Candidates identified: {attempted}, but none were applied.")
        if rejected_commit:
            lines.append(f"  {rejected_commit} rejected at commit stage.")
        if rejected_dirty:
            lines.append(f"  {rejected_dirty} rejected — donor carried a hard-cap pattern.")
        if rejected_no_match:
            lines.append(f"  {rejected_no_match} rejected — REPLACE text did not match TOP 1 verbatim.")
    else:
        lines.append("No runner-up sentence or clause met the graft conditions.")
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
    lines.append("FINAL PASS — COMMERCIAL vs LITERARY PICKS")
    lines.append("=" * 60)
    fp = pipeline_result.get("final_pass") or {}
    if not fp or not fp.get("ran"):
        lines.append("Not run (fewer than 2 acceptable drafts).")
    else:
        lit_idx = fp.get("literary_index", 0)
        com_idx = fp.get("commercial_index", 0)
        lit_rid = fp.get("literary_run_id", "")
        com_rid = fp.get("commercial_run_id", "")
        if lit_idx:
            lines.append(f"Most literary:   T{lit_idx}  (run_id: {lit_rid})")
        else:
            lines.append("Most literary:   (not parsed from response)")
        if com_idx:
            lines.append(f"Most commercial: T{com_idx}  (run_id: {com_rid})")
        else:
            lines.append("Most commercial: (not parsed from response)")
        if lit_idx and com_idx and lit_idx == com_idx:
            lines.append("(Same draft chosen for both registers.)")
    lines.append("")
    lines.append("=" * 60)
    lines.append("STAGE G — LINE-EDIT PASS (mechanical copyedit + AI-tell handling)")
    lines.append("=" * 60)
    le = pipeline_result.get("line_edit") or {}
    if not le.get("ran"):
        if le.get("error"):
            lines.append(f"Did not run: {le['error']}")
        else:
            lines.append("Did not run (disabled or no final text).")
    else:
        mech = le.get("mechanical") or {}
        mech_applied = mech.get("applied", False)
        lines.append(
            f"G1 mechanical copyedit: "
            f"{'APPLIED' if mech_applied else 'not applied'}"
            f" ({mech.get('reason', '')})"
        )
        if mech_applied and mech.get("diff_summary"):
            lines.append(f"  Diff: {', '.join(mech['diff_summary'])}")
        lines.append(f"G2 AI-tell flags: {le.get('flagged_count', 0)}")
        action_counts = {}
        for fs in le.get("flagged_sentences", []):
            action_counts[fs["action"]] = action_counts.get(fs["action"], 0) + 1
        if action_counts:
            for action, count in sorted(action_counts.items()):
                lines.append(f"  {action}: {count}")
        lines.append(f"G3 text changed: {le.get('changed', False)}")
        g4 = le.get("g4") or {}
        g4_applied = g4.get("applied", False)
        g4_edits = g4.get("edits") or []
        lines.append(
            f"G4 multi-sentence pass: "
            f"{'APPLIED' if g4_applied else 'not applied'}"
            f" ({g4.get('reason', '')})"
        )
        if g4_applied and g4_edits:
            target_counts = {}
            for ed in g4_edits:
                if isinstance(ed, dict):
                    t = ed.get("target", "?")
                    target_counts[t] = target_counts.get(t, 0) + 1
            for t, c in sorted(target_counts.items()):
                lines.append(f"  {t}: {c}")
        if le.get("edited_path"):
            lines.append(f"Edited file:   {le['edited_path']}")
        if le.get("report_path"):
            lines.append(f"Audit report:  {le['report_path']}")
        # Surface any flag_for_rewrite items inline for quick visibility
        rewrites = [
            fs for fs in le.get("flagged_sentences", [])
            if fs.get("action") == "flag_for_rewrite"
        ]
        if rewrites:
            lines.append("")
            lines.append("Sentences requiring manual rewrite:")
            for fs in rewrites:
                lines.append(
                    f"  [{fs['flagged_word']}] {fs['original_sentence'].strip()}"
                )
    lines.append("")
    lines.append("=" * 60)
    lines.append("STAGE F — PREDICTED ORIGINALITY HUMAN-SCORE (advisory)")
    lines.append("=" * 60)
    sf = pipeline_result.get("stage_f") or {}
    sfb = pipeline_result.get("stage_f_batch") or {}
    if not sf.get("available"):
        lines.append(f"Unavailable: {sf.get('reason', 'predictor not loaded')}")
        lines.append("(Pipeline ran normally; the predicted-score step is skipped.)")
    else:
        pred = sf.get("predicted_score")
        band = sf.get("band", "UNAVAILABLE")
        n_train = sf.get("n_train", 0)
        loo_mae = sf.get("loo_mae", float("nan"))
        lines.append(f"Predicted score:      {pred} / 100")
        lines.append(f"Recommendation band:  {band}")
        lines.append(
            f"  (SHIP ≥ {STAGE_F_BAND_SHIP}  ·  RECONSIDER {STAGE_F_BAND_CAUTION}–{STAGE_F_BAND_SHIP - 1}  ·  "
            f"REGENERATE < {STAGE_F_BAND_CAUTION})"
        )
        loo_r = sf.get("loo_r", float("nan"))
        lines.append(f"Corpus: {n_train} labeled docs · LOO MAE {loo_mae} · r {loo_r}")
        lines.append("Note: ridge regression on 6 corpus-fit structural features. Advisory only — "
                     "does not gate shipping.")
        if pipeline_result.get("final_text_lineedited"):
            lines.append("Scored on: LINE-EDITED text (Stage G output).")
        else:
            lines.append("Scored on: original FINAL text (Stage G produced no change).")
        scores = sfb.get("scores") or []
        if scores:
            lines.append("")
            lines.append("Batch-wide Stage F scores (highest raw first):")
            for idx, item in enumerate(scores, 1):
                pred_obj = item.get("prediction", {})
                rid = item.get("run_id", "")
                marker = ""
                if rid and rid == pipeline_result.get("top1_run_id", ""):
                    marker = " [TOP 1]"
                lines.append(
                    f"  {idx}. {rid}: pred={pred_obj.get('predicted_score')} "
                    f"raw={pred_obj.get('raw_score')} band={pred_obj.get('band')}{marker}"
                )
        if sfb.get("report_path"):
            lines.append(f"Debug report: {sfb['report_path']}")
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
    if fp.get("literary_path"):
        lines.append(f"Final pass — literary:   {fp['literary_path']}")
    if fp.get("commercial_path"):
        lines.append(f"Final pass — commercial: {fp['commercial_path']}")
    if fp.get("reasoning_path"):
        lines.append(f"Final pass — reasoning:  {fp['reasoning_path']}")
    if pipeline_result.get("ranking_manifest_path"):
        lines.append(f"AI ranking order:   {pipeline_result['ranking_manifest_path']}")
    if le.get("edited_path"):
        lines.append(f"Line-edited final:  {le['edited_path']}")
    if le.get("report_path"):
        lines.append(f"Line-edit report:   {le['report_path']}")
    sfb = pipeline_result.get("stage_f_batch") or {}
    if sfb.get("report_path"):
        lines.append(f"Stage F debug:      {sfb['report_path']}")
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
st.caption(f"App version: `{APP_VERSION}` · Generate · Quality Gate · AI Rank · Ship")
ensure_dirs()
csv_path = RUNS_DIR / CSV_FILENAME
github_cfg = load_github_config()
github_pull_on_startup_if_needed(github_cfg, csv_path)
auto_key, auto_key_source = load_api_key()
# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")
    st.caption(f"App version: `{APP_VERSION}`")
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
    st.caption("v18 outlines are self-contained. Only the Outline is required.")
    doc_uploads = {}
    outline_file = st.file_uploader("Outline", type=["txt", "docx"], key="outline")
    if outline_file:
        doc_uploads["Outline"] = extract_text_from_upload(outline_file)
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
    DEFAULT_PROMPT_ID = 63
    prompt_options_df = prompts_df.copy()
    prompt_options_df["id_numeric"] = pd.to_numeric(prompt_options_df["id"], errors="coerce")
    invalid_prompt_count = int(prompt_options_df["id_numeric"].isna().sum())
    prompt_options_df = prompt_options_df.dropna(subset=["id_numeric"]).copy()
    prompt_options_df["id_int"] = prompt_options_df["id_numeric"].astype(int)
    prompt_options_df = prompt_options_df.reset_index(drop=True)

    if prompt_options_df.empty:
        st.error(f"No usable numeric prompt IDs found in {PROMPTS_CSV}.")
        st.stop()
    if invalid_prompt_count:
        st.warning(
            f"Skipped {invalid_prompt_count} row(s) in {PROMPTS_CSV} because their `id` value is not numeric."
        )
    if prompt_options_df["id_int"].duplicated().any():
        st.warning(
            f"Duplicate prompt IDs found in {PROMPTS_CSV}. The selector still works, but duplicate IDs will share the same P# in output files."
        )

    default_matches = prompt_options_df.index[prompt_options_df["id_int"] == DEFAULT_PROMPT_ID].tolist()
    default_prompt_index = default_matches[0] if default_matches else 0

    def prompt_choice_label(row_index: int) -> str:
        row = prompt_options_df.iloc[row_index]
        category = row.get("category", "")
        if pd.isna(category):
            category = ""
        category = str(category).strip()
        preview = str(row.get("text", "")).replace("\n", " ").strip()
        if preview == "nan":
            preview = ""
        preview = preview[:80] + ("..." if len(preview) > 80 else "")
        bits = [f"P{int(row['id_int'])}"]
        if category:
            bits.append(category)
        if preview:
            bits.append(preview)
        return " — ".join(bits)

    selected_prompt_index = st.selectbox(
        "Prompt choice",
        options=list(range(len(prompt_options_df))),
        index=default_prompt_index,
        format_func=prompt_choice_label,
        help="This list is built from the current prompts.csv. You can add rows, delete old rows, or change the total number of prompts in that CSV.",
    )
    prompt_row = prompt_options_df.iloc[selected_prompt_index]
    target_pid = int(prompt_row["id_int"])
    selected_ids = [target_pid]

    if not default_matches:
        st.caption(f"Default P{DEFAULT_PROMPT_ID} is not present in {PROMPTS_CSV}; using the selected prompt instead.")
    st.caption(
        f"{len(prompt_options_df)} usable prompt(s) loaded from `{PROMPTS_CSV}`. "
        "Add/delete rows in the CSV, then rerun or restart the app to refresh this selector."
    )

    prompt_category = prompt_row.get("category", "")
    if pd.isna(prompt_category):
        prompt_category = ""
    with st.expander(f"P{target_pid} — {str(prompt_category).strip()}"):
        st.text(str(prompt_row["text"]))
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
            problems.append("No documents uploaded. The model needs the Outline.")
        else:
            if "Outline" not in doc_uploads:
                problems.append("Outline not uploaded. The prompt references it.")
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
            prompt_text = str(prompt_row["text"])
            try:
                gated = generate_quality_gated_batch(
                    client=client,
                    gen_model=gen_model,
                    eval_model=eval_model,
                    prompt_text=prompt_text,
                    doc_uploads=doc_uploads,
                    temperatures=temperatures,
                    repetitions=repetitions,
                    prompt_id=target_pid,
                    csv_path=csv_path,
                    github_cfg=github_cfg,
                    progress=progress,
                    status=status,
                    max_tries=QUALITY_GATE_MAX_TRIES,
                )
            except Exception as gate_exc:
                import traceback as _tb
                _tb_text = _tb.format_exc()
                progress.empty()
                st.error(
                    f"Quality gate crashed unrecoverably (uncaught exception "
                    f"inside generate_quality_gated_batch). Run aborted.\n\n"
                    f"Error: {gate_exc}\n\n"
                    f"```\n{_tb_text[:3000]}\n```"
                )
                st.session_state["last_pipeline_error"] = {
                    "stub": "quality_gate",
                    "message": str(gate_exc),
                    "traceback": _tb_text,
                }
                st.stop()
            progress.empty()
            batch_drafts = gated.get("final_drafts", [])
            batch_scans = gated.get("scan_by_run_id", {})
            batch_run_ids_all = gated.get("all_run_ids", [])
            batch_run_ids_ordered = [d["run_id"] for d in batch_drafts]
            if gated.get("halt_reason"):
                status.warning(gated["halt_reason"])
            else:
                status.success(
                    f"Quality gate complete. Retained {len(batch_drafts)}/{total_runs} top-writing drafts. "
                    f"Running AI ranking pipeline..."
                )
            if len(batch_drafts) >= 1:
                outline_text = doc_uploads.get("Outline", "")
                temps_used = sorted(set(temperatures))
                batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                batch_stub = make_batch_stub(batch_timestamp, target_pid, outline_text)
                st.session_state["last_batch_stub"] = batch_stub
                st.session_state["last_batch_run_ids"] = batch_run_ids_ordered
                st.session_state["last_batch_size"] = len(batch_drafts)
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
                        top_n=min(int(top_n), max(1, len(batch_drafts))),
                        batch_stub=batch_stub,
                    )
                    result["quality_gate_history"] = gated.get("quality_gate_history", [])
                    result["quality_gate_attempt_rounds"] = gated.get("attempt_rounds", 0)
                    result["quality_gate_requested_runs"] = total_runs
                    result["quality_gate_retained_runs"] = len(batch_drafts)
                    if gated.get("halt_reason"):
                        result["quality_gate_halt_reason"] = gated.get("halt_reason", "")
                    summary_path = write_batch_summary(
                        pipeline_result=result,
                        drafts=batch_drafts,
                        scan_by_run_id=batch_scans,
                        batch_stub=batch_stub,
                        temperatures=temps_used,
                        prompts_used=[target_pid],
                    )
                    evaluation_id = f"eval_{batch_timestamp}"
                    update_records_bulk(csv_path, batch_run_ids_all, {
                        "is_winner": False,
                        "evaluation_id": evaluation_id,
                        "evaluator_model": eval_model,
                        "pipeline_role": "",
                    })
                    quality_by_run = result.get("quality_by_run_id", {})
                    quality_score_by_run = result.get("quality_score_by_run_id", {})
                    for run_id, q in quality_by_run.items():
                        update_record(csv_path, run_id, {
                            "quality_verdict": q.get("verdict", ""),
                            "quality_reason": (q.get("reason", ""))[:500],
                            "quality_score": int(quality_score_by_run.get(run_id, 0) or 0),
                            "evaluation_raw": result.get("eval_raw", "")[:8000],
                        })
                    for run_id in result.get("dropped_run_ids", []):
                        update_record(csv_path, run_id, {
                            "pipeline_role": "dropped_unacceptable",
                        })
                    for run_id in result.get("discarded_below_top_quality_ids", []):
                        update_record(csv_path, run_id, {
                            "pipeline_role": "discarded_below_top_quality",
                        })
                    for rank_pos, run_id in enumerate(result.get("pipeline_ranking", []), 1):
                        update_record(csv_path, run_id, {"evaluation_rank": rank_pos})
                    top1_id = result.get("top1_run_id", "")
                    if top1_id:
                        update_record(csv_path, top1_id, {
                            "is_winner": True,
                            "pipeline_role": "top1_winner",
                        })
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
                    st.session_state["last_pipeline_result"] = result
                    st.session_state["last_pipeline_summary_path"] = str(summary_path)
                    st.session_state["last_batch_stub"] = batch_stub
                    st.session_state["last_batch_run_ids"] = batch_run_ids_ordered
                    st.session_state["last_batch_size"] = len(batch_drafts)
                    if github_cfg["configured"]:
                        files_to_push = list(result.get("top_paths", [])) + [summary_path]
                        if result.get("ranking_manifest_path"):
                            files_to_push.append(Path(result["ranking_manifest_path"]))
                        if result.get("final_path"):
                            files_to_push.insert(0, Path(result["final_path"]))
                        lg = result.get("line_graft", {})
                        if lg.get("grafted_path"):
                            files_to_push.append(Path(lg["grafted_path"]))
                        fp = result.get("final_pass", {})
                        if fp.get("literary_path"):
                            files_to_push.append(Path(fp["literary_path"]))
                        if fp.get("commercial_path"):
                            files_to_push.append(Path(fp["commercial_path"]))
                        if fp.get("reasoning_path"):
                            files_to_push.append(Path(fp["reasoning_path"]))
                        le = result.get("line_edit", {})
                        if le.get("edited_path"):
                            files_to_push.append(Path(le["edited_path"]))
                        if le.get("report_path"):
                            files_to_push.append(Path(le["report_path"]))
                        try:
                            github_push_after_pipeline(
                                github_cfg, csv_path, files_to_push,
                            )
                        except Exception as push_exc:
                            st.warning(f"GitHub push failed: {push_exc}")
                except Exception as e:
                    import traceback
                    st.session_state["last_pipeline_error"] = {
                        "stub": batch_stub,
                        "message": str(e),
                        "traceback": traceback.format_exc(),
                    }
            else:
                st.warning("Quality gate produced no retained drafts. Nothing to rank.")
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
        st.dataframe(df[available], width='stretch')
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
                        rejected_commit = len(lg.get("grafts_rejected_commit", []))
                        rejected_dirty = len(lg.get("grafts_rejected_dirty_donor", []))
                        rejected_no_match = len(lg.get("grafts_rejected_no_match", []))
                        reasons = []
                        if rejected_commit:
                            reasons.append(f"{rejected_commit} rejected at commit stage")
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
                            f"No runner-up sentence or clause met the graft conditions."
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
                # --- Q2 literary-ranked TOP 1 with scanner veto ---
                scanner_veto = result.get("scanner_veto", {})
                if scanner_veto:
                    vetoed = scanner_veto.get("vetoed_run_ids", [])
                    st.caption(
                        f"Scanner veto: literary winner `{vetoed[0] if vetoed else '?'}` had "
                        f"violations > {scanner_veto.get('veto_threshold', '?')} "
                        f"(2× batch median). TOP 1 promoted to `{top1_id}`."
                    )
                elif lit_winner and lit_winner == top1_id:
                    st.caption(f"Literary winner `{top1_id}` shipped as TOP 1 (no scanner veto).")
                st.markdown("**AI ranking — best to worst**")
                rank_rows = []
                ai_scores_by_run = result.get("ai_scores_by_run_id", {}) or {}
                for rank_pos, run_id in enumerate(result.get("pipeline_ranking", []), 1):
                    pred_obj = ai_scores_by_run.get(run_id, {}) or {}
                    pred_score = pred_obj.get("predicted_score")
                    raw_score = pred_obj.get("raw_score")
                    band = pred_obj.get("band", "")
                    rank_rows.append({
                        "rank": rank_pos,
                        "run_id": run_id,
                        "predicted_score": pred_score,
                        "raw_score": raw_score,
                        "band": band,
                    })
                if rank_rows:
                    st.dataframe(pd.DataFrame(rank_rows), width='stretch', hide_index=True)
                    ranking_manifest_path = result.get("ranking_manifest_path", "")
                    if ranking_manifest_path:
                        st.caption(f"AI ranking file: `{ranking_manifest_path}`")
                with st.expander("Literary evaluator reasoning"):
                    st.text(result["eval_raw"])
                # --- Q3 sentence-graft pass details ---
                if lg:
                    if lg.get("grafted"):
                        with st.expander("Graft details"):
                            for i, g in enumerate(lg["grafts"], 1):
                                gtype = g.get("graft_type", "A")
                                type_label = "Flag Repair" if gtype == "A" else "Quality Upgrade"
                                st.markdown(f"**Graft {i}** — Type {gtype} ({type_label}) from Draft {g['source_draft']}")
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
                # --- Stage E: Final pass — commercial vs literary picks ---
                fp = result.get("final_pass") or {}
                if fp.get("ran"):
                    st.markdown("---")
                    st.subheader("Final pass — commercial vs literary")
                    lit_idx = fp.get("literary_index", 0)
                    com_idx = fp.get("commercial_index", 0)
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if lit_idx:
                            st.markdown(
                                f"**Most literary:** T{lit_idx} "
                                f"(`{fp.get('literary_run_id', '')}`)"
                            )
                        else:
                            st.markdown("**Most literary:** _not parsed_")
                    with col_b:
                        if com_idx:
                            st.markdown(
                                f"**Most commercial:** T{com_idx} "
                                f"(`{fp.get('commercial_run_id', '')}`)"
                            )
                        else:
                            st.markdown("**Most commercial:** _not parsed_")
                    if lit_idx and com_idx and lit_idx == com_idx:
                        st.caption("Same draft chosen for both registers.")
                    if fp.get("raw"):
                        with st.expander("Final-pass evaluator reasoning"):
                            st.text(fp["raw"])
                # --- Stage G: Line-edit pass (copyedit + AI-tell handling) ---
                le = result.get("line_edit") or {}
                if le.get("ran"):
                    st.markdown("---")
                    st.subheader("Line-edit pass")
                    mech = le.get("mechanical") or {}
                    mech_applied = mech.get("applied", False)
                    col_le1, col_le2, col_le3 = st.columns(3)
                    with col_le1:
                        st.metric(
                            "G1 copyedit",
                            "applied" if mech_applied else "no change",
                        )
                    with col_le2:
                        st.metric("G2 flags", le.get("flagged_count", 0))
                    with col_le3:
                        action_counts = {}
                        for fs_ in le.get("flagged_sentences", []):
                            action_counts[fs_["action"]] = (
                                action_counts.get(fs_["action"], 0) + 1
                            )
                        grafted = action_counts.get("graft", 0)
                        deleted = action_counts.get("deletion", 0)
                        flagged = action_counts.get("flag_for_rewrite", 0)
                        st.metric(
                            "G3 actions",
                            f"{grafted}g / {deleted}d / {flagged}r",
                            help=(
                                "g = graft from runner-up · "
                                "d = deletion · "
                                "r = flagged for manual rewrite"
                            ),
                        )
                    if mech_applied and mech.get("diff_summary"):
                        st.caption(
                            f"Punctuation diff: {', '.join(mech['diff_summary'])}"
                        )
                    g4 = le.get("g4") or {}
                    g4_applied = g4.get("applied", False)
                    g4_edits = g4.get("edits") or []
                    if g4_applied or g4.get("reason"):
                        st.caption(
                            f"G4 multi-sentence pass: "
                            f"{'applied' if g4_applied else 'no change'} "
                            f"({g4.get('reason', '')}; "
                            f"{len(g4_edits) if isinstance(g4_edits, list) else 0} "
                            f"deletion(s))"
                        )
                    rewrites = [
                        fs_ for fs_ in le.get("flagged_sentences", [])
                        if fs_.get("action") == "flag_for_rewrite"
                    ]
                    if rewrites:
                        with st.expander(
                            f"Sentences requiring manual rewrite ({len(rewrites)})"
                        ):
                            for fs_ in rewrites:
                                st.markdown(
                                    f"**[{fs_['flagged_word']}]** "
                                    f"{fs_['original_sentence'].strip()}"
                                )
                    if le.get("edited_path"):
                        st.caption(f"Line-edited text: `{le['edited_path']}`")
                    if le.get("report_path"):
                        st.caption(f"Audit report:     `{le['report_path']}`")
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
            # Include Stage G line-edit outputs if produced
            le_res = result.get("line_edit") or {}
            le_edited = le_res.get("edited_path", "")
            if le_edited:
                le_ep = Path(le_edited)
                if le_ep.exists():
                    batch_paths.append(le_ep)
            le_report = le_res.get("report_path", "")
            if le_report:
                le_rp = Path(le_report)
                if le_rp.exists():
                    batch_paths.append(le_rp)
            final_p = Path(result["final_path"]) if result.get("final_path") else None
            if final_p and final_p.exists():
                batch_paths.append(final_p)
            if summary_path_str:
                sp = Path(summary_path_str)
                if sp.exists():
                    batch_paths.append(sp)
            ranking_manifest_path = result.get("ranking_manifest_path", "")
            if ranking_manifest_path:
                rmp = Path(ranking_manifest_path)
                if rmp.exists():
                    batch_paths.append(rmp)
            # Include final-pass register-pick outputs and reasoning. These
            # are written to disk in run_final_pass() but were previously
            # absent from the explicit batch_paths list, so they vanished
            # from the download zip even when present in FINAL_DIR.
            fp_res = result.get("final_pass") or {}
            for path_key in ("literary_path", "commercial_path", "reasoning_path"):
                pstr = fp_res.get(path_key, "")
                if pstr:
                    pp = Path(pstr)
                    if pp.exists():
                        batch_paths.append(pp)
            # Include Stage F debug report. Same prior omission as the
            # final-pass paths above.
            sfb_res = result.get("stage_f_batch") or {}
            sfb_report = sfb_res.get("report_path", "")
            if sfb_report:
                sfp = Path(sfb_report)
                if sfp.exists():
                    batch_paths.append(sfp)
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
                    file_name=f"{batch_label}.zip",
                    mime="application/zip",
                )
                csv_buf = io.StringIO()
                recent_df.to_csv(csv_buf, index=False)
                st.download_button(
                    f"Download latest batch CSV",
                    data=csv_buf.getvalue(),
                    file_name=f"{batch_label}.csv",
                    mime="text/csv",
                )
            else:
                st.info(
                    "No downloadable batch in this session yet. Click "
                    "Generate & Evaluate to produce one."
                )