"""
corpus_audit_app.py

Streamlit integrity auditor for The Last Retreat corpus zips
(v17_corpus_update.zip and successors).

Upload a corpus zip; the app unpacks labeled_corpus.json and
sentence_dataset.json and runs the seven checks:

  1. Total records and completeness of human_score.
  2. Distribution of human_score across bands.
  3. Filename-based score parsing: confirms every ai_in_name
     doc was converted via human = 100 - AI.
  4. Sign convention check: mean_offset direction vs doc_score.
  5. Filename collisions across source folders.
  6. Orphan detection: scored docs with no sentences; sentences
     without a corresponding labeled_corpus entry.
  7. Color-class consistency with mean_offset thresholds.

No API key required. No external services.

Deploy: push this file and a requirements.txt containing `streamlit`
to a GitHub repo, point Streamlit Cloud at it.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections import Counter, defaultdict
from statistics import mean

import streamlit as st


# --- Constants --------------------------------------------------------------

OFFSET_THRESHOLDS = [
    (15, "STRONG_GREEN"),
    (5, "mild_green"),
    (-5, "neutral"),
    (-15, "mild_orange"),
]


def classify_offset(offset):
    if offset is None:
        return "uncolored"
    for t, name in OFFSET_THRESHOLDS:
        if offset >= t:
            return name
    return "STRONG_ORANGE"


# --- Audit routines ---------------------------------------------------------

def load_zip(upload):
    """Unpack the two JSONs from an uploaded zip."""
    with zipfile.ZipFile(io.BytesIO(upload.read())) as z:
        names = z.namelist()
        lc_name = next((n for n in names if n.endswith("labeled_corpus.json")), None)
        sd_name = next((n for n in names if n.endswith("sentence_dataset.json")), None)
        if lc_name is None or sd_name is None:
            return None, None, names
        lc = json.loads(z.read(lc_name).decode("utf-8"))
        sd = json.loads(z.read(sd_name).decode("utf-8"))
    return lc, sd, names


def check_completeness(docs):
    total = len(docs)
    scored = sum(1 for d in docs if d.get("human_score") is not None)
    missing = [d.get("id", "?") for d in docs if d.get("human_score") is None]
    return total, scored, missing


def check_distribution(docs):
    bands = {"0-19": 0, "20-39": 0, "40-59": 0, "60-79": 0, "80-100": 0}
    for d in docs:
        s = d.get("human_score")
        if s is None:
            continue
        if s < 20:
            bands["0-19"] += 1
        elif s < 40:
            bands["20-39"] += 1
        elif s < 60:
            bands["40-59"] += 1
        elif s < 80:
            bands["60-79"] += 1
        else:
            bands["80-100"] += 1
    return bands


def check_ai_conversion(docs):
    """For every ai_in_name doc, confirm human_score == 100 - number_in_filename."""
    import re
    rows = []
    problems = []
    for d in docs:
        if d.get("label_type") != "ai_in_name":
            continue
        fn = d.get("source_file", "")
        m = re.search(r"(\d{1,3})\s*AI", fn, re.IGNORECASE)
        if not m:
            problems.append((fn, d.get("human_score"), "no number found"))
            continue
        ai = int(m.group(1))
        expected = 100 - ai
        stored = d.get("human_score")
        ok = stored == expected
        rows.append({
            "filename": fn,
            "parsed_AI": ai,
            "expected_human": expected,
            "stored_human": stored,
            "correct": "yes" if ok else "NO",
        })
        if not ok:
            problems.append((fn, stored, f"expected {expected}"))
    return rows, problems


def check_sign_convention(docs, sents):
    """Are mean_offset and doc_score in the same direction?"""
    human_by_file = {d["source_file"]: d["human_score"] for d in docs}
    per_file_offsets = defaultdict(list)
    for s in sents:
        if s.get("mean_offset") is not None and s.get("source_file") in human_by_file:
            per_file_offsets[s["source_file"]].append(s["mean_offset"])
    rows = []
    for f, offsets in sorted(per_file_offsets.items()):
        rows.append({
            "file": f,
            "human_score": human_by_file[f],
            "avg_mean_offset": round(mean(offsets), 2),
            "n_sentences": len(offsets),
        })
    # Pearson r manually (no numpy required)
    if len(rows) >= 2:
        xs = [r["human_score"] for r in rows]
        ys = [r["avg_mean_offset"] for r in rows]
        mx, my = mean(xs), mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        dx = sum((x - mx) ** 2 for x in xs) ** 0.5
        dy = sum((y - my) ** 2 for y in ys) ** 0.5
        r = num / (dx * dy) if dx and dy else 0.0
    else:
        r = 0.0
    return rows, r


def check_filename_collisions(docs):
    c = Counter(d["source_file"] for d in docs)
    dupes = {k: v for k, v in c.items() if v > 1}
    rows = []
    for fn, n in dupes.items():
        matches = [d for d in docs if d["source_file"] == fn]
        for d in matches:
            rows.append({
                "filename": fn,
                "id": d.get("id"),
                "source_folder": d.get("source_folder"),
                "human_score": d.get("human_score"),
                "word_count": d.get("word_count"),
            })
    return rows


def check_orphans(docs, sents):
    scored_files = {d["source_file"] for d in docs if d.get("human_score") is not None}
    sent_files = {s["source_file"] for s in sents}
    scored_without_sents = sorted(scored_files - sent_files)
    sents_without_labels = sorted(sent_files - scored_files)
    return scored_without_sents, sents_without_labels


def check_color_class_consistency(sents):
    mismatches = []
    for s in sents:
        off = s.get("mean_offset")
        got = s.get("color_class")
        expected = classify_offset(off)
        if got != expected:
            mismatches.append({
                "source_file": s.get("source_file"),
                "text": (s.get("text") or "")[:80],
                "mean_offset": off,
                "stored_class": got,
                "expected_class": expected,
            })
            if len(mismatches) >= 20:
                break
    return mismatches


# --- UI ---------------------------------------------------------------------

st.set_page_config(page_title="Corpus Audit", layout="wide")
st.title("The Last Retreat — Corpus Integrity Audit")
st.caption(
    "Upload v17_corpus_update.zip (or any future corpus zip). "
    "The app extracts labeled_corpus.json and sentence_dataset.json "
    "and runs seven structural checks. No API calls."
)

upload = st.file_uploader("Corpus zip file", type=["zip"])
if upload is None:
    st.stop()

with st.spinner("Unpacking and parsing..."):
    docs, sents, names = load_zip(upload)

if docs is None or sents is None:
    st.error("Could not find labeled_corpus.json and sentence_dataset.json in the zip.")
    st.write("Archive contents:")
    st.code("\n".join(names or []))
    st.stop()

st.success(f"Loaded {len(docs)} doc records and {len(sents):,} sentence records.")

# 1. Completeness
st.header("1. Label completeness")
total, scored, missing = check_completeness(docs)
st.write(f"**{scored}/{total}** records have a non-null `human_score`.")
if missing:
    st.error(f"Records missing human_score: {missing}")
else:
    st.success("No missing labels.")

# 2. Distribution
st.header("2. Score distribution")
bands = check_distribution(docs)
st.bar_chart(bands)
st.write({k: v for k, v in bands.items()})

# 3. AI → Human conversion
st.header("3. AI-to-Human conversion (ai_in_name docs)")
rows, problems = check_ai_conversion(docs)
if not rows:
    st.info("No ai_in_name docs in this corpus.")
else:
    st.dataframe(rows, use_container_width=True)
    if problems:
        st.error(f"{len(problems)} conversion problems found.")
        st.write(problems)
    else:
        st.success(f"All {len(rows)} ai_in_name docs correctly converted.")

# 4. Sign convention
st.header("4. Sign convention: mean_offset direction vs human_score")
sign_rows, r = check_sign_convention(docs, sents)
st.write(
    f"Pearson correlation between per-doc avg `mean_offset` and `human_score`: "
    f"**r = {r:+.3f}** across {len(sign_rows)} docs."
)
if r > 0.5:
    st.success("Strong positive correlation — fields are in the same direction (both high = human).")
elif r < -0.5:
    st.error("Strong NEGATIVE correlation — fields are inverted. Check the ingest logic.")
else:
    st.warning("Weak correlation — predictor upstream may be noisy; check individual rows.")
with st.expander("Per-file detail"):
    st.dataframe(sign_rows, use_container_width=True)

# 5. Filename collisions
st.header("5. Filename collisions across folders")
coll = check_filename_collisions(docs)
if not coll:
    st.success("No filename collisions.")
else:
    st.warning(
        f"{len(coll)} records share filenames with another record (usually across "
        f"validation folders). sentence_dataset.json keys by filename alone, so "
        f"colliding sentences will be merged. If scores match, no data corruption; "
        f"if they differ, labels are ambiguous."
    )
    st.dataframe(coll, use_container_width=True)

# 6. Orphans
st.header("6. Orphan check")
scored_no_sents, sents_no_labels = check_orphans(docs, sents)
col_a, col_b = st.columns(2)
with col_a:
    st.metric("Scored docs with zero sentences", len(scored_no_sents))
    if scored_no_sents:
        st.code("\n".join(scored_no_sents))
with col_b:
    st.metric("Unlabeled source_files in sentence data", len(sents_no_labels))
    if sents_no_labels:
        with st.expander(f"Show {len(sents_no_labels)} files"):
            st.code("\n".join(sents_no_labels))

# 7. Color-class consistency
st.header("7. color_class vs mean_offset consistency")
mism = check_color_class_consistency(sents)
if not mism:
    st.success("All sentences: color_class matches the mean_offset thresholds.")
else:
    st.warning(f"First {len(mism)} mismatches shown (stops at 20):")
    st.dataframe(mism, use_container_width=True)

# Summary
st.header("Summary")
issues = []
if missing:
    issues.append(f"{len(missing)} records missing human_score")
if problems:
    issues.append(f"{len(problems)} AI-conversion errors")
if r <= 0.5:
    issues.append(f"sign-convention correlation only {r:+.3f}")
if coll:
    issues.append(f"{len(coll)} filename-collision records")
if scored_no_sents:
    issues.append(f"{len(scored_no_sents)} scored docs have no sentences")
if mism:
    issues.append(f"color_class mismatches present")

if not issues:
    st.success("Corpus is structurally sound. Safe to use as training data.")
else:
    st.warning("Issues detected:")
    for i in issues:
        st.write(f"- {i}")
