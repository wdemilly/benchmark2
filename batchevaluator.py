"""
Chapter Batch Evaluator — Streamlit app

Uploads N chapter drafts, groups them into batches, sends each batch to Sonnet
for comparative ranking, and reports the ranking within each batch. Supports
a second "finals" round on the top performers from round one.

Dependencies (requirements.txt):
    streamlit
    anthropic
    pandas
    python-docx
"""

import io
import json
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
import anthropic

try:
    import docx as python_docx
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False


# ============================================================================
# Configuration
# ============================================================================

MODEL = "claude-sonnet-4-5"  # change to the Sonnet identifier current to your SDK
DEFAULT_BATCH_SIZE = 18
MIN_BATCH_SIZE = 5
MAX_BATCH_SIZE = 25
DEFAULT_TOP_K = 2            # how many to keep from each batch for finals
MAX_OUTPUT_TOKENS = 6000
TEMPERATURE = 0


# ============================================================================
# The evaluator prompt — comparative ranking within a batch
# ============================================================================

EVALUATOR_PROMPT = """You are reading {N} drafts of the same chapter of a novel. They come from different runs of a generation pipeline and may vary in prompt, temperature, outline version, or app version. Your job is to identify which drafts a serious reader of this specific genre would most want to keep reading.

Read every draft in full. Do not skim.

Infer the project register from the drafts themselves — genre, period, point of view, voice, narrator's class and position. Hold the drafts to their own standard, not to a generic "literary" ideal.

Judge as an experienced reader of this genre would, giving weight to:

- Specificity over atmosphere. Reward concrete observed detail — objects, gestures, numbers, names, prices. Penalize drafts that produce the sensation of good writing through cadence alone.
- Render over interpret. Penalize drafts that name emotions, summarize their own meaning, or close paragraphs with an interpretive sentence. Watch especially for "the way..." / "how..." observation framing, "as though..." similes at beat-ends, and negation pivots ("not X but Y").
- Dialogue doing dramatic work. Lines must carry tension, subtext, character, or forward motion — ideally more than one at once. Penalize polite turns stating positions, or exposition in quotation marks.
- Voice consistency. The established voice must hold throughout without drift, pastiche, or anachronism.
- Trust in the ending. The strongest draft ends on the beat it has earned and stops. Weaker drafts add a coda explaining or softening the beat.
- Restraint with simile, aphorism, and summary. Reward sentences that leave the reader to do the work.

Be demanding. Do not be diplomatic. If two drafts are close, name the specific thing that tips the decision.

OUTPUT FORMAT

For each draft, write a brief paragraph (2-4 sentences) citing a specific passage for your main observation.

Then a comparison paragraph naming the top 2-3 contenders and why the top one edges the others.

Then, on a line by itself, write exactly:

RANKING: N, N, N, ...

where the numbers are every draft number from strongest to weakest, separated by commas. Include every draft exactly once.

Then, on the final line, exactly:

WINNER: N

where N is the number of the strongest draft. Nothing after that line.
"""


# ============================================================================
# File reading
# ============================================================================

def extract_text(uploaded_file) -> str | None:
    """Extract plain text from an uploaded .txt or .docx file."""
    name = uploaded_file.name.lower()
    try:
        if name.endswith(".txt"):
            data = uploaded_file.read()
            if isinstance(data, bytes):
                return data.decode("utf-8", errors="replace")
            return data
        elif name.endswith(".docx"):
            if not DOCX_AVAILABLE:
                return None
            doc = python_docx.Document(uploaded_file)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        st.warning(f"Could not read {uploaded_file.name}: {e}")
        return None
    return None


# ============================================================================
# API call and parsing
# ============================================================================

def build_payload(drafts: list[tuple[str, str]]) -> str:
    """Build the user message with all drafts for one batch."""
    parts = [EVALUATOR_PROMPT.format(N=len(drafts)), "\n\n"]
    for i, (name, text) in enumerate(drafts, 1):
        parts.append(f"=== DRAFT {i} (source: {name}) ===\n\n{text}\n\n")
    return "".join(parts)


def evaluate_batch(client: anthropic.Anthropic, drafts: list[tuple[str, str]]) -> str:
    """Send one batch to the API; return raw response text."""
    content = build_payload(drafts)
    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_OUTPUT_TOKENS,
        temperature=TEMPERATURE,
        messages=[{"role": "user", "content": content}],
    )
    # Concatenate all text blocks in case the response streams in multiple
    return "\n".join(b.text for b in resp.content if getattr(b, "text", None))


def parse_ranking(raw: str, n_drafts: int) -> list[int] | None:
    """Extract RANKING: line and return a deduped, size-correct list."""
    match = re.search(r"RANKING:\s*([0-9,\s]+)", raw)
    if not match:
        return None
    nums = [int(x.strip()) for x in match.group(1).split(",") if x.strip().isdigit()]
    nums = [n for n in nums if 1 <= n <= n_drafts]
    seen = set()
    deduped = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            deduped.append(n)
    missing = [i for i in range(1, n_drafts + 1) if i not in seen]
    return deduped + missing


def parse_winner(raw: str) -> int | None:
    match = re.search(r"WINNER:\s*(\d+)", raw)
    return int(match.group(1)) if match else None


# ============================================================================
# Batch runner
# ============================================================================

def run_batches(
    client: anthropic.Anthropic,
    drafts: list[tuple[str, str]],
    batch_size: int,
    round_label: str,
    top_k: int,
) -> pd.DataFrame:
    """Chunk drafts, evaluate each batch, return a dataframe of results."""
    batches = [drafts[i:i + batch_size] for i in range(0, len(drafts), batch_size)]
    progress = st.progress(0.0)
    status = st.empty()
    rows = []

    for bi, batch in enumerate(batches):
        status.info(f"{round_label}: batch {bi + 1} of {len(batches)} ({len(batch)} drafts)")
        try:
            raw = evaluate_batch(client, batch)
            ranking = parse_ranking(raw, len(batch))
            if ranking is None:
                st.warning(f"{round_label} batch {bi + 1}: could not parse ranking")
                ranking = list(range(1, len(batch) + 1))
            for rank_pos, draft_num in enumerate(ranking, 1):
                filename, _ = batch[draft_num - 1]
                rows.append({
                    "round": round_label,
                    "batch": bi + 1,
                    "rank_in_batch": rank_pos,
                    "filename": filename,
                    "is_top_k": rank_pos <= top_k,
                    "raw_response": raw if rank_pos == 1 else "",
                })
        except Exception as e:
            st.error(f"{round_label} batch {bi + 1} failed: {e}")
            for draft_num, (filename, _) in enumerate(batch, 1):
                rows.append({
                    "round": round_label,
                    "batch": bi + 1,
                    "rank_in_batch": None,
                    "filename": filename,
                    "is_top_k": False,
                    "raw_response": str(e),
                })
        progress.progress((bi + 1) / len(batches))

    progress.empty()
    status.empty()
    return pd.DataFrame(rows)


# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(page_title="Chapter Batch Evaluator", layout="wide")
st.title("Chapter Batch Evaluator")
st.caption(f"Comparative ranking within batches · model: {MODEL}")

with st.sidebar:
    st.header("Configuration")
    default_key = ""
    try:
        default_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass
    api_key = st.text_input("Anthropic API Key", type="password", value=default_key)
    batch_size = st.slider("Batch size", MIN_BATCH_SIZE, MAX_BATCH_SIZE, DEFAULT_BATCH_SIZE)
    top_k = st.slider("Keep top K from each batch for finals", 1, 5, DEFAULT_TOP_K)
    st.markdown(f"**Model:** `{MODEL}`")
    st.markdown(f"**Temperature:** {TEMPERATURE}")
    if not DOCX_AVAILABLE:
        st.warning("python-docx not installed. Only .txt uploads will work.")

# State across reruns
if "round1_df" not in st.session_state:
    st.session_state.round1_df = None
if "finals_df" not in st.session_state:
    st.session_state.finals_df = None
if "drafts" not in st.session_state:
    st.session_state.drafts = []

uploaded_files = st.file_uploader(
    "Upload chapter drafts (.txt or .docx)",
    accept_multiple_files=True,
    type=["txt", "docx"],
)

if uploaded_files:
    n_batches = (len(uploaded_files) + batch_size - 1) // batch_size
    st.write(
        f"**{len(uploaded_files)}** files uploaded · "
        f"batch size {batch_size} → **{n_batches}** batches · "
        f"keeping top **{top_k}** per batch → ~{n_batches * top_k} survivors for finals"
    )

    run_col, reset_col = st.columns([1, 4])
    with run_col:
        if st.button("Run round one", type="primary", disabled=not api_key):
            client = anthropic.Anthropic(api_key=api_key)
            drafts = []
            for f in uploaded_files:
                text = extract_text(f)
                if text:
                    drafts.append((f.name, text))
            st.session_state.drafts = drafts
            st.session_state.round1_df = run_batches(
                client, drafts, batch_size, "round1", top_k
            )
            st.session_state.finals_df = None  # reset finals on new round one
    with reset_col:
        if st.button("Clear results"):
            st.session_state.round1_df = None
            st.session_state.finals_df = None
            st.session_state.drafts = []

# Round 1 results
if st.session_state.round1_df is not None and not st.session_state.round1_df.empty:
    df1 = st.session_state.round1_df
    st.subheader("Round one results")

    survivors = df1[df1["is_top_k"]].copy()
    st.markdown(f"**Survivors ({len(survivors)}):** top {top_k} from each batch")
    st.dataframe(
        survivors[["batch", "rank_in_batch", "filename"]].reset_index(drop=True),
        use_container_width=True,
    )

    with st.expander("Full round-one rankings"):
        st.dataframe(
            df1[["batch", "rank_in_batch", "filename"]].reset_index(drop=True),
            use_container_width=True,
        )

    with st.expander("Evaluator reasoning per batch"):
        for bi in sorted(df1["batch"].unique()):
            raw = df1[(df1["batch"] == bi) & (df1["raw_response"] != "")]
            if not raw.empty:
                st.markdown(f"**Batch {bi}**")
                st.text(raw.iloc[0]["raw_response"])
                st.markdown("---")

    # Finals trigger
    if len(survivors) >= 2 and st.session_state.finals_df is None:
        if st.button(f"Run finals round on {len(survivors)} survivors", type="primary"):
            client = anthropic.Anthropic(api_key=api_key)
            survivor_names = set(survivors["filename"].tolist())
            finals_drafts = [
                (name, text) for name, text in st.session_state.drafts
                if name in survivor_names
            ]
            st.session_state.finals_df = run_batches(
                client, finals_drafts, batch_size, "finals", top_k
            )

# Finals results
if st.session_state.finals_df is not None and not st.session_state.finals_df.empty:
    df2 = st.session_state.finals_df
    st.subheader("Finals round results")

    final_survivors = df2[df2["is_top_k"]].copy()
    st.markdown(f"**Final survivors ({len(final_survivors)}):**")
    st.dataframe(
        final_survivors[["batch", "rank_in_batch", "filename"]].reset_index(drop=True),
        use_container_width=True,
    )

    with st.expander("Full finals rankings"):
        st.dataframe(
            df2[["batch", "rank_in_batch", "filename"]].reset_index(drop=True),
            use_container_width=True,
        )

    with st.expander("Finals evaluator reasoning per batch"):
        for bi in sorted(df2["batch"].unique()):
            raw = df2[(df2["batch"] == bi) & (df2["raw_response"] != "")]
            if not raw.empty:
                st.markdown(f"**Finals batch {bi}**")
                st.text(raw.iloc[0]["raw_response"])
                st.markdown("---")

# Download combined CSV
if (
    st.session_state.round1_df is not None
    and not st.session_state.round1_df.empty
):
    frames = [st.session_state.round1_df]
    if st.session_state.finals_df is not None:
        frames.append(st.session_state.finals_df)
    combined = pd.concat(frames, ignore_index=True)

    buf = io.StringIO()
    combined.to_csv(buf, index=False)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        "Download full CSV",
        buf.getvalue(),
        file_name=f"chapter_rankings_{stamp}.csv",
        mime="text/csv",
    )