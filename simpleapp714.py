"""
simpleapp714 — the three-window chapter pipeline
=================================================

Replaces the multi-draft harness of simpleapp_v22. One draft, three model
calls, no Originality anywhere in the loop.

    Window 1   Step_1_Normalize_Outline_v4.txt   outline -> drafting packet
    Window 2   Step_2_Draft_v2.txt               packet  -> raw chapter
    Window 3   Step_3_Harden_v1.txt              draft   -> hardened chapter

After Window 3 a warm-mode counter runs on the hardened chapter. It counts,
in narration only, the constructions Step 3 is ordered to remove. If the
density per 1,000 narration words meets the threshold, the script buys ONE
more Step 3 pass, pointed at the exact stretches the counter found, then
ships. The threshold is UNCALIBRATED: three candidate measures were tested
against the 114-document corpus on 2026-07-14 and none separates passing
from failing machine documents (ledger context: the constructions are not
what is being scored). The counter is a compliance check on Step 3's own
work order, not a score predictor. Installed on Walter's instruction,
2026-07-14.

The gate that would decide ship-or-rework from text alone does not exist
yet (LEDGER_BAND_PREDICTOR.md, Entry 11). The seam for it is
`predictor_verdict()` below, which currently always answers UNKNOWN.
Originality is used only for Walter's manual random audits, outside this
script.

The outline can arrive two ways. Load the whole-book outline file (a real
.docx or a text file wearing the extension) and pick a chapter: ONLY that
chapter goes to Step 1, with the Reference System from the front matter and
the Master Notes from the back riding along, because the chapter's beats
are written in those codes. If the file's chapter headings cannot be
recognized, the whole file goes with the chapter named instead. Or paste a
single chapter outline directly.

Run, from the folder containing this file and the three Step files:

    streamlit run simpleapp714.py

Requires: streamlit, anthropic. API key from the ANTHROPIC_API_KEY
environment variable or Streamlit secrets.

The model is called by streaming — client.messages.stream with model,
max_tokens, and the message, matching the proven simpleapp_v22 calls in
every other respect. No system prompt (the three Step files are the whole
instruction), default temperature, no effort or thinking parameters.
Streaming is required on the deployed library: a non-streaming call whose
max_tokens is large enough to run past the client's safety ceiling raises
ValueError before it is sent, so the draft and harden calls must stream.
"""

import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# ---------------------------------------------------------------- constants

HERE = Path(__file__).resolve().parent

STEP_FILES = {
    1: "Step_1_Normalize_Outline_v4.txt",
    2: "Step_2_Draft_v2.txt",
    3: "Step_3_Harden_v1.txt",
}

MODEL = "claude-opus-4-8"
MAX_TOKENS = {1: 20000, 2: 30000, 3: 30000}
PRICE_IN = 5.00 / 1_000_000     # dollars per input token, Opus 4.8 standard
PRICE_OUT = 25.00 / 1_000_000   # dollars per output token

DEFAULT_THRESHOLD = 12.0        # constructions per 1,000 narration words.
                                # A GUESS. See module docstring.
MAX_TARGET_STRETCHES = 40

CHAPTER_OPEN = re.compile(r"===\s*REVISED CHAPTER[^=]*===\s*", re.I)
CHAPTER_CLOSE = re.compile(r"\s*===\s*END REVISED CHAPTER\s*===", re.I)
LOG_OPEN = re.compile(r"===\s*CHANGE LOG\s*===\s*", re.I)
LOG_CLOSE = re.compile(r"\s*===\s*END CHANGE LOG\s*===", re.I)

# ------------------------------------------------------- warm-mode counter

QUOTE_RE = re.compile(r'["\u201c][^"\u201d]{2,400}?["\u201d]')
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
WORD = re.compile(r"[A-Za-z']+")

ECHO_STOP = set(
    "their there which about would could should where after before because "
    "these those something anything nothing without through against".split()
)

def _sentences(text):
    return [s.strip() for s in SENT_SPLIT.split(text) if s.strip()]

def warm_mode_count(full_text):
    """Count Step 3's target constructions in narration. Returns
    (counts dict, density per 1000 narration words, list of located
    stretches as (label, sentence) pairs)."""
    narration = QUOTE_RE.sub(" ", full_text)
    narr_words = len(WORD.findall(narration)) or 1
    counts = {}
    stretches = []

    def grab(label, pattern, text):
        n = 0
        for m in re.finditer(pattern, text, re.I):
            n += 1
            start = max(0, text.rfind(".", 0, m.start()) + 1)
            end = text.find(".", m.end())
            end = end + 1 if end != -1 else min(len(text), m.end() + 120)
            stretches.append((label, text[start:end].strip()[:240]))
        counts[label] = n

    grab("the-way construction", r"\bthe way\b", narration)
    grab("as-if / as-though comparison", r"\bas (?:if|though)\b", narration)
    grab("like-a simile", r"\blike (?:a|an|the|some)\b", narration)
    grab("not-X-but-Y pivot", r"\bnot\b[^.!?]{0,50}\bbut\b", narration)
    grab("explanatory colon", r": ", narration)

    sents = _sentences(narration)
    braided, prev_tokens = 0, set()
    echo = 0
    for s in sents:
        words_in = WORD.findall(s)
        if len(words_in) >= 30 and len(re.findall(r"\band\b", s, re.I)) >= 2:
            braided += 1
            stretches.append(("braided sentence", s[:240]))
        tokens = [w.lower() for w in words_in
                  if len(w) >= 5 and w.lower() not in ECHO_STOP]
        seen = set()
        for t in tokens:
            if t in seen or t in prev_tokens:
                echo += 1
                stretches.append(("echoed word: " + t, s[:240]))
                break
            seen.add(t)
        prev_tokens = set(tokens)
    counts["braided sentence"] = braided
    counts["echoed word"] = echo

    density = 1000.0 * sum(counts.values()) / narr_words
    return counts, density, stretches[:MAX_TARGET_STRETCHES]

# ------------------------------------------------------------ the book file

CHAPTER_HEAD = re.compile(
    r"^\s*#{0,4}\s*CHAPTER\s+(\d+)\b[:.\u2014\-\s]*(.*)$", re.I)
NOTES_HEAD = re.compile(r"^\s*#{1,3}\s+\S")

def read_book_file(data):
    """A real .docx (a zip) or a plain-text file under any name."""
    import html as _html
    import io
    import zipfile
    if data[:2] == b"PK":
        z = zipfile.ZipFile(io.BytesIO(data))
        xml = z.read("word/document.xml").decode("utf-8", "replace")
        xml = re.sub(r"</w:p>", "\n", xml)
        xml = re.sub(r"<[^>]+>", "", xml)
        return _html.unescape(xml)
    return data.decode("utf-8", "replace")

def split_book(text):
    """Split a whole-book outline into (preamble, {number: chapter text},
    {number: title}, trailing notes). Returns None if no CHAPTER headings
    are found."""
    lines = text.split("\n")
    marks = []
    for i, line in enumerate(lines):
        m = CHAPTER_HEAD.match(line)
        if m:
            marks.append((i, int(m.group(1)), m.group(2).strip()))
    if not marks:
        return None
    preamble = "\n".join(lines[:marks[0][0]]).strip()
    chapters, titles = {}, {}
    for k, (i, num, title) in enumerate(marks):
        end = marks[k + 1][0] if k + 1 < len(marks) else len(lines)
        chapters[num] = "\n".join(lines[i:end]).strip()
        titles[num] = title
    # peel trailing book-level notes off the final chapter: the first
    # heading after it that is not itself a CHAPTER heading
    last = max(chapters)
    body = chapters[last].split("\n")
    notes = ""
    for j in range(1, len(body)):
        if NOTES_HEAD.match(body[j]) and not CHAPTER_HEAD.match(body[j]):
            notes = "\n".join(body[j:]).strip()
            chapters[last] = "\n".join(body[:j]).strip()
            break
    return preamble, chapters, titles, notes

# ------------------------------------------------------------ the predictor

def predictor_verdict(chapter_text):
    """The seam for a future ship/rework predictor. Ledger Entry 11 proved
    no text-only predictor of the Originality verdict currently exists.
    Until one does, this answers UNKNOWN and the pipeline ships everything
    the counter has passed. Do not wire a predictor in here without
    matched-pair scan evidence behind it."""
    return "UNKNOWN"

# ----------------------------------------------------------------- api

def clean_api_key(value):
    return (value or "").strip().strip('"').strip("'")

def load_api_key():
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            key = clean_api_key(str(st.secrets["ANTHROPIC_API_KEY"]))
            if key:
                return key, "secrets"
    except Exception:
        pass
    key = clean_api_key(os.environ.get("ANTHROPIC_API_KEY", ""))
    if key:
        return key, "environment"
    return "", "none"

def call_model(client, prompt_text, max_tokens, status_slot):
    """One streamed call, otherwise shaped like the proven simpleapp_v22
    calls: model, max_tokens, and the message, with no effort setting, no
    thinking block, no system prompt, and default temperature. Streaming
    is required, not optional: on the deployed library a non-streaming
    request whose max_tokens is large enough to run past the client's
    safety ceiling raises ValueError from _calculate_nonstreaming_timeout
    before the call is even sent. Streaming lifts that ceiling and keeps
    the connection alive with a steady trickle of tokens. The result is
    accumulated and returned identically to a plain call."""
    status_slot.write("waiting for the model...")
    pieces = []
    with client.messages.stream(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt_text}],
    ) as stream:
        for chunk in stream.text_stream:
            pieces.append(chunk)
            if len(pieces) % 50 == 0:
                status_slot.write(
                    f"writing... {len(''.join(pieces).split())} words so far")
        final = stream.get_final_message()
    text = "\n".join(b.text for b in final.content
                     if getattr(b, "text", None))
    usage = final.usage
    return (text,
            getattr(usage, "input_tokens", 0),
            getattr(usage, "output_tokens", 0))

# ----------------------------------------------------------- prompt plumbing

def load_step(n):
    path = HERE / STEP_FILES[n]
    if not path.exists():
        st.error(f"Missing prompt file: {path}. This script must sit in the "
                 f"folder that holds the three Step files.")
        st.stop()
    return path.read_text(encoding="utf-8")

def build_step1(outline, chapter_number=None, chapter_title="",
                whole_book=False):
    head = load_step(1).rstrip()
    if whole_book and chapter_number is not None:
        name = (f"CHAPTER {chapter_number}"
                + (f": {chapter_title}" if chapter_title else ""))
        return (head
                + f"\n\nTHE CHAPTER TO CONVERT IS {name}. The rest of the"
                " document below is the full book outline; treat everything"
                " outside the named chapter as book-level notes, per the"
                " INPUT instructions."
                "\n\n=== EXISTING CHAPTER OUTLINE (FULL BOOK — CONVERT ONLY"
                " THE NAMED CHAPTER) ===\n"
                + outline.strip() + "\n")
    return (head + "\n\n=== EXISTING CHAPTER OUTLINE ===\n"
            + outline.strip() + "\n")

def build_step2(packet):
    return load_step(2).replace("[PASTE OUTLINE HERE]", packet.strip())

def build_step3(draft, packet, target_note=""):
    text = load_step(3)
    if target_note:
        text = text.replace(
            "DRAFT\n[PASTE DRAFT HERE]",
            target_note.rstrip() + "\n\nDRAFT\n[PASTE DRAFT HERE]")
    return (text.replace("[PASTE DRAFT HERE]", draft.strip())
                .replace("[PASTE OUTLINE HERE]", packet.strip()))

def targeting_note(stretches, density, threshold):
    lines = [
        "COUNTER-TARGETED SECOND PASS (added by the pipeline)",
        "This draft already went through this prompt once. A construction",
        f"counter found a residual density of {density:.1f} per 1,000",
        f"narration words against a working threshold of {threshold:.1f}.",
        "As in SECOND-RUN MODE, confine every edit to the stretches listed",
        "below, plus any category word Operation 1 can name anywhere in the",
        "narration. Re-derive the anchors and leave them alone. Touch",
        "nothing else.",
        "",
        "THE STRETCHES:",
    ]
    for label, sentence in stretches:
        lines.append(f"- [{label}] {sentence}")
    return "\n".join(lines)

def parse_step3(raw):
    """Split Step 3 output into (chapter, change_log). Falls back to the
    whole text if the delimiters did not survive."""
    chapter, log = raw, ""
    m1 = CHAPTER_OPEN.search(raw)
    m2 = CHAPTER_CLOSE.search(raw)
    if m1 and m2 and m2.start() > m1.end():
        chapter = raw[m1.end():m2.start()].strip()
    m3 = LOG_OPEN.search(raw)
    m4 = LOG_CLOSE.search(raw)
    if m3:
        log = raw[m3.end():m4.start()].strip() if m4 else raw[m3.end():].strip()
    return chapter, log

# ----------------------------------------------------------------- the app

def main():
    st.set_page_config(page_title="simpleapp714", layout="wide")
    st.title("simpleapp714 — outline to hardened chapter")
    st.caption("Three windows, one chapter. No Originality in the loop.")

    if not ANTHROPIC_AVAILABLE:
        st.error("The anthropic library is not installed. "
                 "Run: pip install anthropic")
        st.stop()

    key, source = load_api_key()
    with st.sidebar:
        st.subheader("Settings")
        if key:
            st.success(f"API key loaded from {source}.")
        else:
            key = clean_api_key(st.text_input("Anthropic API key",
                                              type="password"))
        st.write(f"Model: `{MODEL}`, streamed with no system prompt and "
                 "default temperature — the same call shape as the proven "
                 "simpleapp_v22.")
        threshold = st.number_input(
            "Warm-mode threshold, constructions per 1,000 narration words. "
            "UNCALIBRATED — a guess, installed on instruction. Lower means "
            "the second hardening pass runs more often. The pass is "
            "harmless to passing chapters and costs about eighteen cents.",
            min_value=0.0, max_value=60.0,
            value=DEFAULT_THRESHOLD, step=1.0)

    book = st.file_uploader(
        "Book outline file — load it and pick the chapter "
        "(.docx or a text file):", type=["docx", "txt", "md"])
    outline = ""
    pick = None
    pick_title = ""
    whole_book = False
    if book is not None:
        book_text = read_book_file(book.getvalue())
        parsed = split_book(book_text)
        if parsed:
            preamble, chapters, titles, notes = parsed
            numbers = sorted(chapters)
            pick = st.selectbox(
                "Chapter to send to Step 1:", numbers,
                format_func=lambda n: (f"Chapter {n} — {titles[n]}"
                                       if titles[n] else f"Chapter {n}"))
            pick_title = titles[pick]
            # Send ONLY the chosen chapter. The Reference System from the
            # front matter and the Master Notes from the back ride along,
            # because the chapter's beats are written in those codes; the
            # other 29 chapters do not.
            outline = "\n\n".join(
                p for p in (preamble, chapters[pick], notes) if p).strip()
            st.caption(
                f"{len(chapters)} chapters found. Chapter {pick} goes to "
                f"Step 1 on its own ({len(chapters[pick].split())} words), "
                f"with the Reference System and Master Notes "
                f"({len(preamble.split()) + len(notes.split())} words) "
                f"riding along so the chapter's codes resolve.")
        else:
            outline = book_text.strip()
            whole_book = True
            st.warning(
                "No CHAPTER headings were recognized, so the chapter cannot "
                "be isolated. The whole file goes to Step 1 with the chapter "
                "you name below.")
            pick = int(st.number_input("Chapter number to convert:",
                                       min_value=1, max_value=99,
                                       value=1, step=1))
            pick_title = st.text_input(
                "Chapter title (optional; helps Step 1 find it):").strip()
    if not outline:
        outline = st.text_area(
            "Or paste a single chapter outline (book-level notes may ride "
            "along before or after it):", height=300)

    if st.button("Run the pipeline", type="primary"):
        if not key:
            st.error("No API key.")
            st.stop()
        if not outline.strip():
            st.error("No outline.")
            st.stop()

        client = anthropic.Anthropic(api_key=key)
        run_dir = HERE / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        tokens_in = tokens_out = 0
        t0 = time.time()

        def save(name, text):
            (run_dir / name).write_text(text, encoding="utf-8")

        save("0_step1_input.txt",
             (f"[CHAPTER TO CONVERT: {pick} — {pick_title}]\n\n" if pick
              else "") + outline)

        # ---- Window 1: normalize
        with st.status("Window 1 — normalizing the outline...",
                       expanded=False) as s:
            packet, ti, to = call_model(
                client, build_step1(outline, pick, pick_title, whole_book),
                MAX_TOKENS[1], s)
            tokens_in += ti; tokens_out += to
            save("1_packet.txt", packet)
            s.update(label=f"Window 1 done — packet "
                           f"{len(packet.split())} words.", state="complete")

        # ---- Window 2: draft
        with st.status("Window 2 — drafting the chapter...",
                       expanded=False) as s:
            draft, ti, to = call_model(client, build_step2(packet),
                                       MAX_TOKENS[2], s)
            tokens_in += ti; tokens_out += to
            save("2_draft.txt", draft)
            s.update(label=f"Window 2 done — draft "
                           f"{len(draft.split())} words.", state="complete")

        # ---- Window 3: harden, pass 1
        with st.status("Window 3 — hardening pass one...",
                       expanded=False) as s:
            raw, ti, to = call_model(client, build_step3(draft, packet),
                                     MAX_TOKENS[3], s)
            tokens_in += ti; tokens_out += to
            chapter, log = parse_step3(raw)
            save("3_hardened_pass1.txt", chapter)
            save("3_changelog_pass1.txt", log)
            s.update(label=f"Window 3 pass one done — "
                           f"{len(chapter.split())} words.", state="complete")

        # ---- the counter
        counts, density, stretches = warm_mode_count(chapter)
        flagged = density >= threshold
        passes = 1

        st.subheader("Warm-mode counter, pass one")
        st.write(f"Density: **{density:.1f}** per 1,000 narration words. "
                 f"Threshold: {threshold:.1f}. "
                 f"Flag: **{'TRIPPED' if flagged else 'clear'}**.")
        st.table({"construction": list(counts.keys()),
                  "count": list(counts.values())})

        if flagged and stretches:
            with st.status("Flag tripped — Window 3, targeted second "
                           "pass...", expanded=False) as s:
                note = targeting_note(stretches, density, threshold)
                save("4_targeting_note.txt", note)
                raw2, ti, to = call_model(
                    client, build_step3(chapter, packet, note),
                    MAX_TOKENS[3], s)
                tokens_in += ti; tokens_out += to
                chapter2, log2 = parse_step3(raw2)
                save("4_hardened_pass2.txt", chapter2)
                save("4_changelog_pass2.txt", log2)
                counts2, density2, _ = warm_mode_count(chapter2)
                s.update(label=f"Second pass done — "
                               f"{len(chapter2.split())} words, density "
                               f"{density2:.1f} (was {density:.1f}).",
                         state="complete")
            chapter, log = chapter2, log + "\n\n--- PASS 2 ---\n\n" + log2
            passes = 2
            st.write(f"Density after the second pass: **{density2:.1f}**.")

        # ---- ship
        save("chapter_FINAL.txt", chapter)
        cost = tokens_in * PRICE_IN + tokens_out * PRICE_OUT
        verdict = predictor_verdict(chapter)
        report = {
            "when": datetime.now().isoformat(timespec="seconds"),
            "model": MODEL,
            "hardening_passes": passes,
            "counter_density_final": round(warm_mode_count(chapter)[1], 2),
            "threshold": threshold,
            "chapter_words": len(chapter.split()),
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "cost_dollars": round(cost, 2),
            "minutes": round((time.time() - t0) / 60, 1),
            "predictor_verdict": verdict,
        }
        save("report.json", json.dumps(report, indent=1))

        st.subheader("Shipped")
        st.write(f"{report['chapter_words']} words, {passes} hardening "
                 f"pass(es), {report['minutes']} minutes, about "
                 f"${report['cost_dollars']:.2f}. Files in `{run_dir}`.")
        st.write("Ship-or-rework verdict from text alone: **UNKNOWN** — no "
                 "honest predictor exists yet (ledger Entry 11). Audit "
                 "chapters manually at your own schedule.")
        st.download_button("Download the chapter", chapter,
                           file_name="chapter_FINAL.txt")
        with st.expander("The chapter"):
            st.text(chapter)
        with st.expander("Change log"):
            st.text(log)

if __name__ == "__main__":
    main()
