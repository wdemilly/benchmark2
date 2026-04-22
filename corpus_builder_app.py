"""
corpus_builder_app.py

Streamlit app that replaces build_labeled_corpus.py + add_new_sentences.py
as a single browser-based step.

Inputs (all uploaded through the UI):
    - New Originality .docx exports (loose multi-upload OR a .zip of them)
    - Folder tag for the new batch (becomes the id prefix so same-filename
      exports from different folders do not collide)
    - OPTIONAL: current labeled_corpus.json (baseline to merge into)
    - OPTIONAL: current sentence_dataset.json (baseline to merge into)

Outputs (download buttons):
    - Updated labeled_corpus.json
    - Updated sentence_dataset.json
    - Run summary (counts: docs in, docs added, dupes skipped, sentences
      added, score-label breakdown, score range)

Parses filename -> score exactly as build_labeled_corpus.py did:
    score_in_name, ai_in_name, top_rank, unlabeled.

Parses docx XML directly (no python-docx dependency) so the app runs on
Streamlit Cloud without extra native libs.

IMPROVEMENT over the old CLI scripts: sentence records are keyed on an
`id` field that carries the folder tag as a prefix. The audit surfaced
that keying on `source_file` alone caused cross-folder filename
collisions (e.g. two different export(89).docx files in two different
folders merging into one bucket). The folder-prefixed id fixes this.
"""

import io
import json
import re
import zipfile
import html
import hashlib
from collections import Counter
from datetime import datetime
from typing import Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Corpus Builder", layout="wide")
st.title("Corpus Builder")
st.caption(
    "Ingest Originality .docx exports into labeled_corpus.json and "
    "sentence_dataset.json. Replaces build_labeled_corpus.py + "
    "add_new_sentences.py."
)


# ---------------------------------------------------------------------------
# Docx XML parsing (no python-docx dependency)
# ---------------------------------------------------------------------------

# Matches the fill hex on any element that carries w:shd.
# Originality sets shading at the run level (<w:rPr><w:shd w:fill="RRGGBB"/></w:rPr>)
# and/or at the paragraph level. We look for any w:fill="...." attribute value.
_FILL_RE = re.compile(r'w:fill="([0-9A-Fa-f]{6})"')

# Word namespace (we strip all namespaces to simplify regex matching).
_NS_STRIP_RE = re.compile(r"\sxmlns(:[A-Za-z0-9]+)?=\"[^\"]*\"")
_TAG_NS_RE = re.compile(r"<(/?)[A-Za-z0-9]+:")


def _read_document_xml(docx_bytes: bytes) -> str:
    """Pull word/document.xml out of the .docx zip and return as string."""
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
        with zf.open("word/document.xml") as f:
            raw = f.read().decode("utf-8", errors="replace")
    return raw


def _strip_namespaces(xml: str) -> str:
    """Remove xmlns declarations and namespace prefixes so tags become
    <p>, <r>, <t>, <shd>, etc. Attributes like w:fill remain — we match
    them with _FILL_RE."""
    xml = _NS_STRIP_RE.sub("", xml)
    xml = _TAG_NS_RE.sub(r"<\1", xml)
    return xml


# ---------------------------------------------------------------------------
# Run extraction: each <r> carries text + (optional) fill hex
# ---------------------------------------------------------------------------

# Parse paragraphs, then the runs inside each paragraph. For every run we
# capture text and — if present on that run or inherited from paragraph —
# the fill hex.

_P_SPLIT_RE = re.compile(r"<p(?:\s[^>]*)?>(.*?)</p>", re.DOTALL)
_R_SPLIT_RE = re.compile(r"<r(?:\s[^>]*)?>(.*?)</r>", re.DOTALL)
_T_RE = re.compile(r"<t(?:\s[^>]*)?>(.*?)</t>", re.DOTALL)
_RPR_RE = re.compile(r"<rPr>(.*?)</rPr>", re.DOTALL)
_PPR_RE = re.compile(r"<pPr>(.*?)</pPr>", re.DOTALL)


def _text_of_run(run_xml: str) -> str:
    pieces = _T_RE.findall(run_xml)
    txt = "".join(pieces)
    return html.unescape(txt)


def _fill_of_run(run_xml: str, paragraph_fill: Optional[str]) -> Optional[str]:
    rpr_match = _RPR_RE.search(run_xml)
    if rpr_match:
        m = _FILL_RE.search(rpr_match.group(1))
        if m:
            return m.group(1).upper()
    return paragraph_fill


def extract_runs(docx_bytes: bytes) -> list[tuple[Optional[str], str]]:
    """Return ordered list of (fill_hex or None, text) for every run in the
    document. Paragraph boundaries are preserved by injecting a '\n'
    run between paragraphs."""
    xml = _strip_namespaces(_read_document_xml(docx_bytes))
    runs: list[tuple[Optional[str], str]] = []
    for i, para_inner in enumerate(_P_SPLIT_RE.findall(xml)):
        para_fill: Optional[str] = None
        ppr_match = _PPR_RE.search(para_inner)
        if ppr_match:
            pm = _FILL_RE.search(ppr_match.group(1))
            if pm:
                para_fill = pm.group(1).upper()

        for run_xml in _R_SPLIT_RE.findall(para_inner):
            txt = _text_of_run(run_xml)
            if not txt:
                continue
            fill = _fill_of_run(run_xml, para_fill)
            runs.append((fill, txt))
        # paragraph break marker
        if i < len(_P_SPLIT_RE.findall(xml)) - 1:
            runs.append((None, "\n"))
    return runs


# ---------------------------------------------------------------------------
# Color classification (R - G differential)
# Matches build_labeled_corpus.py + add_new_sentences.py convention.
# ---------------------------------------------------------------------------

def _rgb(hex_color: str) -> tuple[int, int, int]:
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return r, g, b


def classify_fill(hex_color: Optional[str]) -> str:
    """Five-class classifier based on R-G diff.

        diff <= -5   : STRONG_GREEN    (human)
        -4..0        : mild_green      (likely human)
        +1..+10      : neutral
        +11..+20     : mild_orange     (likely AI)
        >= +21       : STRONG_ORANGE   (AI)

    None fill -> 'no_fill'.
    """
    if hex_color is None:
        return "no_fill"
    try:
        r, g, _ = _rgb(hex_color)
    except ValueError:
        return "no_fill"
    diff = r - g
    if diff <= -5:
        return "STRONG_GREEN"
    if diff <= 0:
        return "mild_green"
    if diff <= 10:
        return "neutral"
    if diff <= 20:
        return "mild_orange"
    return "STRONG_ORANGE"


def rg_offset(hex_color: Optional[str]) -> Optional[int]:
    """R - G. Negative = greener/human-leaning, positive = redder/AI-leaning.
    None if no fill."""
    if hex_color is None:
        return None
    try:
        r, g, _ = _rgb(hex_color)
    except ValueError:
        return None
    return r - g


# ---------------------------------------------------------------------------
# Sentence splitting (same-color runs get concatenated into spans, then
# split on sentence punctuation)
# ---------------------------------------------------------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"\u201C\u2018])")


def split_into_sentences(runs: list[tuple[Optional[str], str]]) -> list[dict]:
    """Build sentence records from an ordered run list.

    Each sentence record contains:
        text:         str
        runs:         list of (fill_hex or None, substring) — the pieces
                      of this sentence with their source colors
        mean_offset:  char-weighted average of R-G across colored runs
                      (None if no run was colored)
        color_class:  classification of the weighted-mean color
                      (fell back to 'no_fill' if no colored run)
    """
    full_text = "".join(t for _, t in runs)
    # position -> fill map so we can slice out per-sentence colored runs
    positions: list[tuple[int, Optional[str]]] = []
    cursor = 0
    spans: list[tuple[int, int, Optional[str]]] = []  # (start, end, fill)
    for fill, text in runs:
        if not text:
            continue
        start = cursor
        end = cursor + len(text)
        spans.append((start, end, fill))
        cursor = end

    sentences: list[dict] = []
    sent_start = 0
    # walk through full_text and split on sentence boundaries
    for m in _SENT_SPLIT_RE.finditer(full_text):
        sent_end = m.start()
        stext = full_text[sent_start:sent_end].strip()
        if stext:
            sentences.append(_build_sentence_record(stext, sent_start, sent_end, spans, full_text))
        sent_start = m.end()
    # trailing sentence
    tail_text = full_text[sent_start:].strip()
    if tail_text:
        sentences.append(_build_sentence_record(tail_text, sent_start, len(full_text), spans, full_text))

    # drop empty / whitespace-only sentences
    return [s for s in sentences if s["text"].strip()]


def _build_sentence_record(text: str, start: int, end: int,
                            spans: list[tuple[int, int, Optional[str]]],
                            full_text: str) -> dict:
    piece_records: list[tuple[Optional[str], str]] = []
    total_chars = 0
    weighted_sum = 0
    colored_chars = 0
    for sp_start, sp_end, fill in spans:
        overlap_start = max(start, sp_start)
        overlap_end = min(end, sp_end)
        if overlap_end <= overlap_start:
            continue
        piece = full_text[overlap_start:overlap_end]
        if not piece:
            continue
        piece_records.append((fill, piece))
        off = rg_offset(fill)
        if off is not None:
            length = overlap_end - overlap_start
            weighted_sum += off * length
            colored_chars += length
        total_chars += len(piece)

    mean_offset = (weighted_sum / colored_chars) if colored_chars > 0 else None

    # color_class derived from the weighted-mean offset, not a majority vote
    if mean_offset is None:
        color_class = "no_fill"
    else:
        if mean_offset <= -5:
            color_class = "STRONG_GREEN"
        elif mean_offset <= 0:
            color_class = "mild_green"
        elif mean_offset <= 10:
            color_class = "neutral"
        elif mean_offset <= 20:
            color_class = "mild_orange"
        else:
            color_class = "STRONG_ORANGE"

    return {
        "text": text,
        "runs": piece_records,
        "mean_offset": round(mean_offset, 3) if mean_offset is not None else None,
        "color_class": color_class,
    }


# ---------------------------------------------------------------------------
# Doc-level color metrics (matches build_labeled_corpus.py convention)
# ---------------------------------------------------------------------------

def doc_color_metrics(runs: list[tuple[Optional[str], str]]) -> dict:
    classes = [classify_fill(f) for f, t in runs if f is not None and t.strip()]
    counts = Counter(classes)
    total = len(classes) or 1
    sg = counts.get("STRONG_GREEN", 0)
    mg = counts.get("mild_green", 0)
    n  = counts.get("neutral", 0)
    mo = counts.get("mild_orange", 0)
    so = counts.get("STRONG_ORANGE", 0)
    return {
        "segs": len(classes),
        "strong_green": sg,
        "mild_green": mg,
        "neutral": n,
        "mild_orange": mo,
        "strong_orange": so,
        "green_pct": round(100 * (sg + mg) / total, 1) if total else 0.0,
        "orange_pct": round(100 * (so + mo) / total, 1) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Filename -> score parser (copied verbatim from build_labeled_corpus.py)
# ---------------------------------------------------------------------------

def parse_score_from_filename(name: str) -> tuple[Optional[int], str]:
    """Return (human_score, label_type). label_type is one of:
        score_in_name, ai_in_name, top_rank, unlabeled.
    """
    stem = name.replace(".docx", "")

    # TOP ranking files: "export top 1.docx" -> rank, not score
    if re.search(r"\btop\s+\d+\b", stem, re.IGNORECASE):
        return (None, "top_rank")

    # AI score in name: "NNAI" or "NN AI"
    ai_match = re.search(r"(\d{1,3})\s*AI\b", stem, re.IGNORECASE)
    if ai_match:
        ai = int(ai_match.group(1))
        if 0 <= ai <= 100:
            return (100 - ai, "ai_in_name")

    # strip paren contents and replication tags
    cleaned = re.sub(r"\([^)]*\)", "", stem)
    cleaned = re.sub(r"\bR\d\b", "", cleaned, flags=re.IGNORECASE)

    nums = re.findall(r"\d{1,3}", cleaned)
    valid = [int(x) for x in nums if 0 <= int(x) <= 100]
    if valid:
        # last number in cleaned name is overwhelmingly the score
        return (valid[-1], "score_in_name")

    trail = re.search(r"(\d{1,3})$", cleaned.strip())
    if trail:
        v = int(trail.group(1))
        if 0 <= v <= 100:
            return (v, "score_in_name")

    return (None, "unlabeled")


# ---------------------------------------------------------------------------
# Per-doc ingest
# ---------------------------------------------------------------------------

def _safe_stem(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1]
    stem = stem.rsplit("\\", 1)[-1]
    if stem.lower().endswith(".docx"):
        stem = stem[:-5]
    return stem


def ingest_docx(filename: str, docx_bytes: bytes, folder_tag: str,
                include_unlabeled: bool) -> Optional[dict]:
    """Produce {'doc': <labeled_corpus record>, 'sentences': [<sent recs>]}
    for one docx. Returns None if the file is unlabeled and
    include_unlabeled is False."""
    stem = _safe_stem(filename)
    score, label_type = parse_score_from_filename(stem)
    if score is None and not include_unlabeled:
        return None

    try:
        runs = extract_runs(docx_bytes)
    except KeyError:
        # not a valid docx (no word/document.xml)
        return None
    except zipfile.BadZipFile:
        return None

    # full text
    text = "".join(t for _, t in runs)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    word_count = len(re.findall(r"\b\w+\b", text))

    # doc-level color metrics
    metrics = doc_color_metrics(runs)

    # sentence records
    sent_records_raw = split_into_sentences(runs)

    # folder-prefixed id prevents cross-folder collisions (audit finding)
    doc_id = f"{folder_tag}::{stem}"

    doc_record = {
        "id": doc_id,
        "source_file": f"{stem}.docx",
        "folder_tag": folder_tag,
        "label_type": label_type,
        "human_score": score,
        "word_count": word_count,
        "text": text,
        **metrics,
    }

    sentence_records: list[dict] = []
    for i, sr in enumerate(sent_records_raw):
        sentence_records.append({
            "id": f"{doc_id}#s{i:04d}",
            "doc_id": doc_id,
            "source_file": f"{stem}.docx",
            "folder_tag": folder_tag,
            "sent_index": i,
            "doc_score": score,
            "text": sr["text"],
            "mean_offset": sr["mean_offset"],
            "color_class": sr["color_class"],
        })

    return {"doc": doc_record, "sentences": sentence_records}


# ---------------------------------------------------------------------------
# Dedup helpers
# ---------------------------------------------------------------------------

def _text_fingerprint(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.subheader("1. New exports")
col1, col2 = st.columns([2, 1])
with col1:
    uploaded_docs = st.file_uploader(
        "Drop Originality .docx exports (multi-select) OR a .zip of them",
        type=["docx", "zip"],
        accept_multiple_files=True,
        help="Filenames should carry the score per the existing convention: "
             "'export 87.docx', '45 8 99.docx', '62AI.docx', etc.",
    )
with col2:
    folder_tag = st.text_input(
        "Folder tag for this batch",
        value=datetime.utcnow().strftime("batch_%Y%m%d"),
        help="Becomes the id prefix (e.g. 'validation_2' -> "
             "id='validation_2::export 94'). Prevents cross-folder "
             "filename collisions.",
    )

st.subheader("2. Baseline JSONs (optional — leave empty to build from scratch)")
col3, col4 = st.columns(2)
with col3:
    baseline_corpus = st.file_uploader("labeled_corpus.json", type=["json"], key="corpus_json")
with col4:
    baseline_sents = st.file_uploader("sentence_dataset.json", type=["json"], key="sent_json")

st.subheader("3. Options")
col5, col6 = st.columns(2)
with col5:
    include_unlabeled = st.checkbox(
        "Include unlabeled docs (human_score = null)",
        value=False,
        help="If on, exports whose filename carries no parseable score are "
             "still ingested — useful if you want the sentences for training "
             "but not the doc-level labels.",
    )
with col6:
    force_overwrite = st.checkbox(
        "Overwrite on id collision (default: keep baseline)",
        value=False,
        help="If a new doc has an id that already exists in the baseline, "
             "keep the baseline entry by default. Turn this on to replace "
             "the baseline entry with the new one.",
    )

go = st.button("Build corpus", type="primary", use_container_width=True)


# ---------------------------------------------------------------------------
# Main build step
# ---------------------------------------------------------------------------

def _load_baseline(file, label: str) -> list[dict]:
    if file is None:
        return []
    try:
        data = json.load(file)
        if not isinstance(data, list):
            st.error(f"{label}: expected a JSON list at top level.")
            return []
        return data
    except json.JSONDecodeError as e:
        st.error(f"{label}: invalid JSON — {e}")
        return []


def _collect_uploaded_docx(files) -> list[tuple[str, bytes]]:
    """Flatten the upload list into (filename, bytes) pairs. Zips are
    expanded in memory."""
    out: list[tuple[str, bytes]] = []
    for f in files:
        name = f.name
        if name.lower().endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(f.read())) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        if not info.filename.lower().endswith(".docx"):
                            continue
                        # skip Mac resource forks
                        if "__MACOSX" in info.filename:
                            continue
                        out.append((info.filename, zf.read(info.filename)))
            except zipfile.BadZipFile:
                st.warning(f"Skipping {name}: not a valid zip.")
        elif name.lower().endswith(".docx"):
            out.append((name, f.read()))
    return out


if go:
    if not uploaded_docs:
        st.error("Upload at least one .docx (or a zip of them) before building.")
        st.stop()
    if not folder_tag.strip():
        st.error("Folder tag is required.")
        st.stop()

    folder_tag_clean = folder_tag.strip()

    # load baselines
    baseline_docs = _load_baseline(baseline_corpus, "labeled_corpus.json")
    baseline_sents_list = _load_baseline(baseline_sents, "sentence_dataset.json")

    # index baselines for dedup / merge
    existing_ids = {d.get("id") for d in baseline_docs if d.get("id")}
    existing_fps = {}
    for d in baseline_docs:
        t = d.get("text", "")
        if t:
            existing_fps[_text_fingerprint(t)] = d.get("id")
    existing_sent_ids = {s.get("id") for s in baseline_sents_list if s.get("id")}

    # flatten uploads
    pairs = _collect_uploaded_docx(uploaded_docs)
    if not pairs:
        st.error("No .docx files found in your upload.")
        st.stop()

    progress = st.progress(0.0, text=f"Processing 0/{len(pairs)}")
    added_docs: list[dict] = []
    skipped_dupe_id: list[str] = []
    skipped_dupe_text: list[tuple[str, str]] = []  # (new_id, existing_id)
    skipped_unlabeled: list[str] = []
    added_sentences: list[dict] = []
    label_type_counts: Counter = Counter()

    for i, (fname, data) in enumerate(pairs, start=1):
        try:
            result = ingest_docx(fname, data, folder_tag_clean, include_unlabeled)
        except Exception as e:
            st.warning(f"Failed to parse {fname}: {e}")
            progress.progress(i / len(pairs), text=f"Processing {i}/{len(pairs)}")
            continue

        if result is None:
            skipped_unlabeled.append(fname)
            progress.progress(i / len(pairs), text=f"Processing {i}/{len(pairs)}")
            continue

        doc = result["doc"]
        label_type_counts[doc["label_type"]] += 1

        # id collision
        if doc["id"] in existing_ids and not force_overwrite:
            skipped_dupe_id.append(doc["id"])
            progress.progress(i / len(pairs), text=f"Processing {i}/{len(pairs)}")
            continue

        # text fingerprint dedup (catches re-uploads under a different filename)
        fp = _text_fingerprint(doc["text"])
        if fp in existing_fps:
            skipped_dupe_text.append((doc["id"], existing_fps[fp]))
            progress.progress(i / len(pairs), text=f"Processing {i}/{len(pairs)}")
            continue

        added_docs.append(doc)
        existing_ids.add(doc["id"])
        existing_fps[fp] = doc["id"]

        # sentence records — filter against existing ids
        for s in result["sentences"]:
            if s["id"] in existing_sent_ids and not force_overwrite:
                continue
            added_sentences.append(s)
            existing_sent_ids.add(s["id"])

        progress.progress(i / len(pairs), text=f"Processing {i}/{len(pairs)}")

    # build merged outputs
    if force_overwrite:
        # replace entries in baselines by id
        kept_docs = [d for d in baseline_docs if d.get("id") not in {a["id"] for a in added_docs}]
        merged_docs = kept_docs + added_docs
        kept_sents = [s for s in baseline_sents_list if s.get("id") not in {a["id"] for a in added_sentences}]
        merged_sents = kept_sents + added_sentences
    else:
        merged_docs = baseline_docs + added_docs
        merged_sents = baseline_sents_list + added_sentences

    # ─── report ───────────────────────────────────────────
    st.success(f"Ingested {len(added_docs)} new docs, {len(added_sentences)} new sentences.")

    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Uploaded files", len(pairs))
    col_b.metric("Docs added", len(added_docs))
    col_c.metric("ID dupes skipped", len(skipped_dupe_id))
    col_d.metric("Text dupes skipped", len(skipped_dupe_text))

    col_e, col_f, col_g, col_h = st.columns(4)
    col_e.metric("Unlabeled skipped", len(skipped_unlabeled))
    col_f.metric("Sentences added", len(added_sentences))
    col_g.metric("Total docs after", len(merged_docs))
    col_h.metric("Total sentences after", len(merged_sents))

    # label-type breakdown
    if label_type_counts:
        st.caption("New doc label types: " +
                   ", ".join(f"{k}={v}" for k, v in label_type_counts.most_common()))

    # score distribution
    scored = [d["human_score"] for d in merged_docs if d.get("human_score") is not None]
    if scored:
        bands = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
        for s in scored:
            if s <= 20: bands["0-20"] += 1
            elif s <= 40: bands["21-40"] += 1
            elif s <= 60: bands["41-60"] += 1
            elif s <= 80: bands["61-80"] += 1
            else: bands["81-100"] += 1
        st.subheader("Score distribution (merged corpus)")
        st.bar_chart(bands)
        st.caption(f"Range {min(scored)}–{max(scored)}, n={len(scored)}.")

    # skip lists (expandable)
    if skipped_dupe_text:
        with st.expander(f"Text duplicates skipped ({len(skipped_dupe_text)})"):
            for new_id, existing_id in skipped_dupe_text:
                st.text(f"{new_id}  →  matches  {existing_id}")
    if skipped_dupe_id:
        with st.expander(f"ID duplicates skipped ({len(skipped_dupe_id)})"):
            for did in skipped_dupe_id:
                st.text(did)
    if skipped_unlabeled:
        with st.expander(f"Unlabeled files skipped ({len(skipped_unlabeled)})"):
            for f in skipped_unlabeled:
                st.text(f)

    # ─── downloads ────────────────────────────────────────
    st.subheader("Download updated JSONs")
    col_dl1, col_dl2 = st.columns(2)
    with col_dl1:
        st.download_button(
            "labeled_corpus.json",
            data=json.dumps(merged_docs, indent=2, ensure_ascii=False),
            file_name="labeled_corpus.json",
            mime="application/json",
            use_container_width=True,
        )
    with col_dl2:
        st.download_button(
            "sentence_dataset.json",
            data=json.dumps(merged_sents, indent=2, ensure_ascii=False),
            file_name="sentence_dataset.json",
            mime="application/json",
            use_container_width=True,
        )

    # run summary (txt)
    summary_lines = [
        f"Corpus Builder run — {datetime.utcnow().isoformat()}Z",
        f"Folder tag: {folder_tag_clean}",
        f"Uploaded .docx files: {len(pairs)}",
        f"Docs added: {len(added_docs)}",
        f"Docs skipped (id dupe): {len(skipped_dupe_id)}",
        f"Docs skipped (text dupe): {len(skipped_dupe_text)}",
        f"Docs skipped (unlabeled): {len(skipped_unlabeled)}",
        f"Sentences added: {len(added_sentences)}",
        f"Total docs after merge: {len(merged_docs)}",
        f"Total sentences after merge: {len(merged_sents)}",
        "",
        "Label type breakdown (new docs):",
    ]
    for k, v in label_type_counts.most_common():
        summary_lines.append(f"  {k}: {v}")
    st.download_button(
        "run_summary.txt",
        data="\n".join(summary_lines),
        file_name=f"corpus_builder_summary_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain",
    )
else:
    st.info("Upload docs and click **Build corpus** to start.")
