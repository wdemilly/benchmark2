"""
streamlit_predictor_app.py

Streamlit wrapper around OriginalityPredictor. Two test modes:

  1) Test on corpus record (LOO)
     Pick any labeled record from labeled_corpus.json. The app excludes it
     from the neighbor pool, runs the prediction, and shows the LLM's
     prediction + ridge and NN-mean baselines alongside the known score.
     Use this to calibrate the predictor against known labels without
     uploading anything.

  2) Upload / paste a new draft
     The app auto-detects whether the uploaded text matches any corpus
     record (via char-n-gram overlap) and auto-excludes the match from
     the neighbor pool so the prediction isn't inflated by a self-match.

Files that must live in the same repo as this script:
    originality_predictor.py
    labeled_corpus.json

requirements.txt must include:
    streamlit, scikit-learn>=1.3, numpy>=1.24, anthropic>=0.39
"""

from __future__ import annotations

import html
import io
import json as _json
import re
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from originality_predictor import OriginalityPredictor, extract_style_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_uploaded(uploaded_file) -> str:
    if uploaded_file is None:
        return ""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()
    if name.endswith(".docx"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            with z.open("word/document.xml") as fh:
                xml = fh.read().decode("utf-8", errors="replace")
        text = re.sub(r"<[^>]+>", " ", xml)
        text = re.sub(r"\s+", " ", text).strip()
        return html.unescape(text)
    try:
        return data.decode("utf-8", errors="replace")
    except Exception:
        return data.decode("latin-1", errors="replace")


@st.cache_resource
def load_predictor(corpus_path: str) -> OriginalityPredictor:
    return OriginalityPredictor(corpus_path)


@st.cache_resource
def build_text_overlap_index(corpus_path: str):
    """Build a char-n-gram TF-IDF index over the corpus for duplicate detection.
    Unlike neighbor retrieval (which uses style features), this is specifically
    for detecting 'is this uploaded draft actually a corpus doc?'.
    """
    data = _json.loads(Path(corpus_path).read_text())
    records = [r for r in data if r.get("human_score") is not None]
    texts = [r["text"] for r in records]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), max_features=20000)
    mat = vec.fit_transform(texts)
    ids = [r["id"] for r in records]
    return vec, mat, ids


def detect_corpus_match(draft_text: str, corpus_path: str, threshold: float = 0.60):
    """Return (corpus_id, similarity) if the draft matches a corpus record,
    else (None, best_sim). Char-n-gram is used because shared content is
    the right signal for duplicate detection (unlike retrieval, where we
    deliberately avoid it)."""
    vec, mat, ids = build_text_overlap_index(corpus_path)
    q = vec.transform([draft_text])
    sims = cosine_similarity(q, mat).flatten()
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])
    if best_sim >= threshold:
        return ids[best_idx], best_sim
    return None, best_sim


def get_client(api_key: str):
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


def render_prediction(result: dict, predictor: OriginalityPredictor,
                       known_score: int | None = None):
    """Display a prediction result block."""
    pred = result["predicted_score"]
    status = result["parse_status"]
    if status != "ok":
        st.warning(f"Parse status: {status}")

    cols = st.columns(5 if known_score is not None else 4)
    i = 0
    if known_score is not None:
        cols[i].metric("Actual", f"{known_score}")
        i += 1
    cols[i].metric("LLM predicted", f"{pred}" if pred is not None else "—",
                   delta=(pred - known_score) if (known_score is not None and pred is not None) else None,
                   delta_color="off")
    i += 1
    cols[i].metric("Ridge baseline", f"{result['style_baseline_score']:.0f}")
    i += 1
    cols[i].metric(f"NN-{result['k']} mean", f"{result['nn_mean_baseline']:.0f}")
    i += 1
    cols[i].metric("Recommendation", predictor.recommendation(pred))

    st.subheader("Rationale")
    st.write(result["rationale"])

    st.subheader("Neighbors used")
    for nb in result["neighbors"]:
        st.write(
            f"- **sim={nb['similarity']:.3f}** score={nb['human_score']} "
            f"wc={nb['word_count']} · `{nb['source_file']}`"
        )

    with st.expander("Raw LLM response"):
        st.code(result["raw_response"])
    with st.expander("Full result JSON"):
        minimal = {k: v for k, v in result.items() if k != "raw_response"}
        st.code(_json.dumps(minimal, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Originality predictor", layout="wide")
st.title("Originality predictor — step 1")
st.caption(
    "Predicts the Originality.ai human-score for a draft before submission, "
    "using style-feature retrieval + LLM in-context prediction on the "
    "labeled corpus."
)

with st.sidebar:
    st.header("Config")
    corpus_path = st.text_input("Corpus path", value="labeled_corpus.json")
    model = st.selectbox(
        "Model",
        options=["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"],
        index=1,
        help="Sonnet is ~1/5 the cost of Opus per call.",
    )
    k = st.slider("Neighbors (k)", min_value=3, max_value=10, value=5)
    default_key = ""
    try:
        default_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass
    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        value=default_key,
        help="Set ANTHROPIC_API_KEY in Streamlit secrets to avoid re-entering."
    )

# Load predictor
try:
    predictor = load_predictor(corpus_path)
    st.caption(
        f"Corpus: {len(predictor.records)} labeled records loaded from "
        f"`{corpus_path}`."
    )
except Exception as e:
    st.error(f"Failed to load corpus: {e}")
    st.stop()


# --- Tabs for the two modes ---
tab_corpus, tab_upload = st.tabs([
    "Test on corpus record (LOO)",
    "Upload / paste draft",
])


# =========================================================================
# TAB 1 — LOO on corpus record
# =========================================================================
with tab_corpus:
    st.markdown(
        "Pick a labeled record from the corpus. The app excludes that record "
        "from the neighbor pool and predicts its score. Compare the "
        "prediction to the known score."
    )

    # Build a label-sortable list
    labeled = sorted(
        predictor.records,
        key=lambda r: (-r["human_score"], r["id"]),
    )
    options = [
        f"{r['human_score']:>3d}  |  {r['id']}  ({r['word_count']} wc)"
        for r in labeled
    ]
    choice = st.selectbox(
        "Corpus record",
        options=options,
        index=0,
        key="loo_choice",
    )
    chosen_idx = options.index(choice)
    chosen = labeled[chosen_idx]
    known_score = chosen["human_score"]

    st.caption(
        f"Selected: **{chosen['id']}** — known score **{known_score}**, "
        f"{chosen['word_count']} words, source `{chosen['source_file']}`"
    )

    # Closed-form baselines for this record (LOO: the record is excluded)
    rec_feat = extract_style_features(chosen["text"])
    ridge_base = predictor.ridge_baseline(rec_feat)
    nb_preview = predictor.find_neighbors(
        chosen["text"], k=k, exclude_ids={chosen["id"]}
    )
    nn_mean_preview = (
        sum(n.human_score for n in nb_preview) / len(nb_preview) if nb_preview else 0.0
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Known score", f"{known_score}")
    c2.metric("Ridge (LOO)", f"{ridge_base:.0f}", delta=f"{ridge_base - known_score:+.0f}")
    c3.metric(f"NN-{k} mean (LOO)", f"{nn_mean_preview:.0f}", delta=f"{nn_mean_preview - known_score:+.0f}")
    c4.write(" ")

    run_loo = st.button(
        "Predict with LLM (LOO)",
        type="primary",
        disabled=not api_key,
        key="loo_predict",
        use_container_width=True,
    )

    if run_loo:
        if not api_key:
            st.error("Need an Anthropic API key in the sidebar.")
            st.stop()
        client = get_client(api_key)
        with st.spinner(f"Calling {model} (corpus record excluded)..."):
            try:
                result = predictor.predict(
                    draft_text=chosen["text"],
                    client=client,
                    model=model,
                    k=k,
                    exclude_ids={chosen["id"]},
                )
            except Exception as e:
                st.error(f"API error: {e}")
                st.stop()
        render_prediction(result, predictor, known_score=known_score)


# =========================================================================
# TAB 2 — Upload / paste
# =========================================================================
with tab_upload:
    st.markdown(
        "Upload or paste a draft. If its text overlaps with a corpus record, "
        "the app auto-excludes the match so the prediction isn't inflated "
        "by a self-hit."
    )

    draft_text = ""
    draft_name = ""
    col_up, col_paste = st.columns(2)
    with col_up:
        f = st.file_uploader(
            "Draft file (.txt / .md / .docx)",
            type=["txt", "md", "docx"],
            key="upload_file",
        )
        if f is not None:
            draft_text = read_uploaded(f)
            draft_name = f.name
            st.caption(f"{draft_name}: {len(draft_text.split())} words")
    with col_paste:
        pasted = st.text_area("Or paste draft text", height=200, key="pasted_draft")
        if pasted.strip() and not draft_text:
            draft_text = pasted
            draft_name = "(pasted)"
            st.caption(f"{len(draft_text.split())} words")

    if draft_text.strip():
        # Detect corpus self-match
        match_id, match_sim = detect_corpus_match(draft_text, corpus_path)
        exclude_ids: set[str] = set()
        known_score_for_upload: int | None = None
        if match_id is not None:
            match_rec = next(
                (r for r in predictor.records if r["id"] == match_id), None
            )
            if match_rec is not None:
                known_score_for_upload = match_rec["human_score"]
                st.info(
                    f"This draft matches corpus record **{match_id}** "
                    f"(char-n-gram cosine **{match_sim:.3f}**, known score "
                    f"**{known_score_for_upload}**). Auto-excluding from "
                    f"neighbor pool — prediction will be leave-one-out."
                )
                exclude_ids = {match_id}
        else:
            st.caption(
                f"Max corpus overlap: {match_sim:.3f} (threshold 0.60) — treating as new draft."
            )

        # Closed-form baselines
        query_feat = extract_style_features(draft_text)
        ridge_base = predictor.ridge_baseline(query_feat)
        nb_preview = predictor.find_neighbors(draft_text, k=k, exclude_ids=exclude_ids)
        nn_mean_preview = (
            sum(n.human_score for n in nb_preview) / len(nb_preview)
            if nb_preview else 0.0
        )

        cols = st.columns(4)
        if known_score_for_upload is not None:
            cols[0].metric("Known score", f"{known_score_for_upload}")
        cols[1].metric("Ridge", f"{ridge_base:.0f}")
        cols[2].metric(f"NN-{k} mean", f"{nn_mean_preview:.0f}")
        cols[3].metric("Recommendation (ridge)", predictor.recommendation(int(round(ridge_base))))

        with st.expander("Query style metrics"):
            st.code("\n".join(
                f"  {name}: {query_feat[name]:.3f}"
                if isinstance(query_feat[name], float)
                else f"  {name}: {query_feat[name]}"
                for name in query_feat
            ))
        with st.expander(f"Top-{k} neighbors"):
            for nb in nb_preview:
                st.write(
                    f"- **sim={nb.similarity:.3f}** score={nb.human_score} "
                    f"wc={nb.word_count} · `{nb.source_file}`"
                )

        est_tokens = 22_000
        opus_cost = est_tokens / 1_000_000 * 15
        sonnet_cost = est_tokens / 1_000_000 * 3
        st.caption(
            f"Est. input tokens: ~{est_tokens:,}. "
            f"~${opus_cost:.2f} on Opus, ~${sonnet_cost:.2f} on Sonnet."
        )

        run_upload = st.button(
            "Predict with LLM",
            type="primary",
            disabled=not api_key,
            key="upload_predict",
            use_container_width=True,
        )
        if run_upload:
            if not api_key:
                st.error("Need an Anthropic API key in the sidebar.")
                st.stop()
            client = get_client(api_key)
            with st.spinner(f"Calling {model}..."):
                try:
                    result = predictor.predict(
                        draft_text=draft_text,
                        client=client,
                        model=model,
                        k=k,
                        exclude_ids=exclude_ids or None,
                    )
                except Exception as e:
                    st.error(f"API error: {e}")
                    st.stop()
            render_prediction(result, predictor, known_score=known_score_for_upload)
