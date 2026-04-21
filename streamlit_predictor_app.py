"""
streamlit_predictor_app.py

Minimal Streamlit wrapper around OriginalityPredictor. Upload a draft
(or paste text), pick a model, click Predict. Shows the predicted score,
the ridge and NN-mean baselines, the neighbors used, and the rationale.

Streamlit Cloud entry point. Point your app's "Main file path" at this
file in the app settings.

Files that must be in the same repo directory as this script:
    originality_predictor.py
    labeled_corpus.json

requirements.txt must contain:
    streamlit
    scikit-learn
    numpy
    anthropic
"""

from __future__ import annotations

import io
import re
import zipfile
import html
from pathlib import Path

import streamlit as st

from originality_predictor import OriginalityPredictor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_uploaded(uploaded_file) -> str:
    """Extract text from an uploaded .txt / .md / .docx."""
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
    """Instantiate once per app session. Reloads only if corpus_path changes."""
    return OriginalityPredictor(corpus_path)


def get_client(api_key: str):
    import anthropic
    return anthropic.Anthropic(api_key=api_key)


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
    api_key = st.text_input(
        "Anthropic API key",
        type="password",
        value=st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else "",
        help="Set ANTHROPIC_API_KEY in Streamlit secrets to avoid re-entering."
    )

# Load predictor up-front so corpus-path errors surface before the user works
try:
    predictor = load_predictor(corpus_path)
    st.caption(
        f"Corpus: {len(predictor.records)} labeled records loaded from "
        f"`{corpus_path}`."
    )
except Exception as e:
    st.error(f"Failed to load corpus: {e}")
    st.stop()

# --- Input ---
st.subheader("Draft")
tab_upload, tab_paste = st.tabs(["Upload file", "Paste text"])

draft_text = ""
draft_name = ""
with tab_upload:
    f = st.file_uploader("Draft file (.txt / .md / .docx)", type=["txt", "md", "docx"])
    if f is not None:
        draft_text = read_uploaded(f)
        draft_name = f.name
        st.caption(f"{draft_name}: {len(draft_text.split())} words")
with tab_paste:
    pasted = st.text_area("Paste draft text", height=200, key="pasted_draft")
    if pasted.strip() and not draft_text:
        draft_text = pasted
        draft_name = "(pasted)"
        st.caption(f"{len(draft_text.split())} words")

# --- Closed-form baselines (no API needed — run instantly) ---
if draft_text.strip():
    from originality_predictor import extract_style_features
    query_feat = extract_style_features(draft_text)
    ridge_base = predictor.ridge_baseline(query_feat)
    neighbors_preview = predictor.find_neighbors(draft_text, k=k)
    nn_mean_preview = sum(n.human_score for n in neighbors_preview) / len(neighbors_preview)

    st.subheader("Closed-form baselines (no LLM call)")
    c1, c2, c3 = st.columns(3)
    c1.metric("Ridge baseline", f"{ridge_base:.0f}",
              help="15 style features -> ridge regression. LOO MAE 18.83, r +0.504.")
    c2.metric(f"NN-{k} mean", f"{nn_mean_preview:.0f}",
              help="Mean human-score of the k nearest neighbors in style-feature space.")
    c3.metric("Recommendation (ridge)", predictor.recommendation(int(round(ridge_base))))

    with st.expander("Query style metrics"):
        st.code(
            "\n".join(
                f"  {name}: {query_feat[name]:.3f}" if isinstance(query_feat[name], float)
                else f"  {name}: {query_feat[name]}"
                for name in query_feat
            )
        )

    with st.expander(f"Top-{k} neighbors (style-feature cosine)"):
        for nb in neighbors_preview:
            st.write(
                f"- **sim={nb.similarity:.3f}** score={nb.human_score} "
                f"wc={nb.word_count} · `{nb.source_file}`"
            )

# --- LLM prediction ---
st.subheader("LLM prediction")
col_run, col_cost = st.columns([1, 2])
with col_run:
    go = st.button(
        "Predict with LLM",
        type="primary",
        disabled=not draft_text.strip() or not api_key,
        use_container_width=True,
    )
with col_cost:
    if draft_text.strip():
        # Rough cost estimate: 5 * ~3500 + query + prompt ~= 22k input tokens
        est_tokens = 22_000
        opus_cost = est_tokens / 1_000_000 * 15
        sonnet_cost = est_tokens / 1_000_000 * 3
        st.caption(
            f"Est. input tokens: ~{est_tokens:,}. "
            f"~${opus_cost:.2f} on Opus, ~${sonnet_cost:.2f} on Sonnet."
        )

if go:
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
            )
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

    pred = result["predicted_score"]
    status = result["parse_status"]
    if status != "ok":
        st.warning(f"Parse status: {status}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("LLM predicted", f"{pred}" if pred is not None else "—")
    m2.metric("Ridge baseline", f"{result['style_baseline_score']:.0f}")
    m3.metric(f"NN-{k} mean", f"{result['nn_mean_baseline']:.0f}")
    m4.metric("Recommendation", predictor.recommendation(pred))

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
        # Strip neighbor text which isn't in this result anyway, plus big raw
        import json as _json
        minimal = {k: v for k, v in result.items() if k != "raw_response"}
        st.code(_json.dumps(minimal, indent=2, ensure_ascii=False))
