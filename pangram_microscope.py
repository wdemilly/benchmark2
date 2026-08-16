from __future__ import annotations

import io
import json
import os
import re
import sqlite3
import statistics
import time
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import streamlit as st
from docx import Document

try:
    from pangram import Pangram
except Exception:
    Pangram = None


APP_TITLE = "Pangram Microscope"
APP_VERSION = "v1.1 · Streamlit Cloud"
MIN_PANGRAM_WORDS = 50
DB_PATH = Path(__file__).with_name("pangram_microscope.db")
DEFAULT_SAMPLE_SIZES = [75, 100, 150, 200, 300, 500]
ALL_SAMPLE_SIZES = [50, 75, 100, 125, 150, 200, 250, 300, 400, 500, 750, 1000]


@dataclass
class SourceDoc:
    name: str
    expected_label: str
    text: str


@dataclass
class TextWindow:
    window_id: str
    source_name: str
    expected_label: str
    target_words: int
    actual_words: int
    sentence_start: int
    sentence_end: int
    text: str


# -------------------------
# Text handling
# -------------------------

WORD_RE = re.compile(r"\S+")


def count_words(text: str) -> int:
    return len(WORD_RE.findall(text or ""))


def clean_text(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    # Keep paragraph boundaries, but normalize internal whitespace.
    paragraphs = []
    for p in re.split(r"\n\s*\n", text):
        p = re.sub(r"[ \t\f\v]+", " ", p).strip()
        if p:
            paragraphs.append(p)
    return "\n\n".join(paragraphs)


def extract_text_from_upload(uploaded_file) -> str:
    suffix = Path(uploaded_file.name).suffix.lower()
    raw = uploaded_file.getvalue()

    if suffix == ".docx":
        doc = Document(io.BytesIO(raw))
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        return clean_text("\n\n".join(parts))

    if suffix in {".txt", ".md"}:
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                return clean_text(raw.decode(encoding))
            except UnicodeDecodeError:
                continue
        return clean_text(raw.decode("utf-8", errors="replace"))

    raise ValueError(f"Unsupported file type: {suffix}. Use DOCX, TXT, or MD.")


def split_sentences(text: str) -> list[str]:
    """A lightweight fiction-friendly sentence splitter.

    Pangram 4 is designed for complete-sentence prose. We therefore build test
    windows on sentence boundaries rather than cutting at an exact word index.
    This is deliberately dependency-light; it is not intended as a linguistic
    parser.
    """
    text = clean_text(text)
    if not text:
        return []

    sentences: list[str] = []
    # Work paragraph by paragraph so paragraph breaks remain natural stopping points.
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Capture through sentence-ending punctuation plus closing quote/bracket.
        # If a paragraph has no terminal punctuation, keep it as one unit.
        matches = re.findall(
            r".+?(?:[.!?]+(?:[\"'”’)]*)?(?=\s+|$)|$)",
            paragraph,
            flags=re.S,
        )
        for item in matches:
            item = item.strip()
            if item:
                sentences.append(item)

    return sentences


def _advance_start(sent_word_counts: list[int], start: int, stride_words: int) -> int:
    if start >= len(sent_word_counts) - 1:
        return len(sent_word_counts)
    total = 0
    i = start
    while i < len(sent_word_counts) and total < stride_words:
        total += sent_word_counts[i]
        i += 1
    return max(start + 1, i)


def build_sentence_windows(
    doc: SourceDoc,
    target_words: int,
    overlap_fraction: float,
) -> list[TextWindow]:
    sentences = split_sentences(doc.text)
    if not sentences:
        return []

    counts = [count_words(s) for s in sentences]
    stride_words = max(1, int(round(target_words * (1.0 - overlap_fraction))))
    min_acceptable = MIN_PANGRAM_WORDS

    windows: list[TextWindow] = []
    start = 0
    ordinal = 0

    while start < len(sentences):
        total = 0
        end = start
        while end < len(sentences) and total < target_words:
            total += counts[end]
            end += 1

        if total < min_acceptable:
            break

        ordinal += 1
        text = " ".join(sentences[start:end]).strip()
        safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(doc.name).stem)[:45]
        wid = f"{safe_name}__{target_words}w__{ordinal:04d}"
        windows.append(
            TextWindow(
                window_id=wid,
                source_name=doc.name,
                expected_label=doc.expected_label,
                target_words=target_words,
                actual_words=count_words(text),
                sentence_start=start + 1,
                sentence_end=end,
                text=text,
            )
        )

        start = _advance_start(counts, start, stride_words)

    return windows


def evenly_cap(items: list[TextWindow], cap: int) -> list[TextWindow]:
    if cap <= 0 or len(items) <= cap:
        return items
    if cap == 1:
        return [items[len(items) // 2]]
    indexes = [round(i * (len(items) - 1) / (cap - 1)) for i in range(cap)]
    seen = set()
    selected = []
    for idx in indexes:
        if idx not in seen:
            selected.append(items[idx])
            seen.add(idx)
    return selected


def make_calibration_windows(
    docs: list[SourceDoc],
    sample_sizes: list[int],
    overlap_fraction: float,
    cap_per_source_size: int,
) -> list[TextWindow]:
    out: list[TextWindow] = []
    for doc in docs:
        for size in sorted(sample_sizes):
            candidates = build_sentence_windows(doc, size, overlap_fraction)
            out.extend(evenly_cap(candidates, cap_per_source_size))
    return out


# -------------------------
# Pangram API
# -------------------------


def get_secret_api_key() -> str:
    env_key = os.getenv("PANGRAM_API_KEY", "").strip()
    if env_key:
        return env_key
    try:
        return str(st.secrets.get("PANGRAM_API_KEY", "")).strip()
    except Exception:
        return ""


def connect_pangram(api_key: str) -> tuple[Any, list[str]]:
    if Pangram is None:
        raise RuntimeError(
            "The pangram-sdk package is not installed. Make sure requirements.txt is in the GitHub repo, then reboot the Streamlit app."
        )
    client = Pangram(api_key=api_key) if api_key else Pangram()
    models = client.list_models()
    return client, list(models)


def weighted_window_metric(result: dict[str, Any], field: str) -> float | None:
    windows = result.get("windows") or []
    vals = []
    weights = []
    for w in windows:
        value = w.get(field)
        if value is None:
            continue
        weight = w.get("word_count") or count_words(w.get("text", "")) or 1
        vals.append(float(value))
        weights.append(float(weight))
    if not vals:
        return None
    return sum(v * wt for v, wt in zip(vals, weights)) / sum(weights)


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    windows = result.get("windows") or []
    confidences = [str(w.get("confidence", "")) for w in windows if w.get("confidence")]
    humanizer_scores = [
        float(w["humanizer_score"])
        for w in windows
        if w.get("humanizer_score") is not None
    ]
    return {
        "prediction": result.get("prediction_short"),
        "headline": result.get("headline"),
        "fraction_ai": result.get("fraction_ai"),
        "fraction_ai_assisted": result.get("fraction_ai_assisted"),
        "fraction_human": result.get("fraction_human"),
        "num_ai_segments": result.get("num_ai_segments"),
        "num_ai_assisted_segments": result.get("num_ai_assisted_segments"),
        "num_human_segments": result.get("num_human_segments"),
        "mean_ai_involvement": weighted_window_metric(result, "ai_assistance_score"),
        "max_humanizer_score": max(humanizer_scores) if humanizer_scores else None,
        "any_humanized": any(bool(w.get("is_humanized")) for w in windows),
        "window_confidence": ", ".join(sorted(set(confidences))),
        "version": result.get("version"),
        "dashboard_link": result.get("dashboard_link"),
    }


def run_bulk_scan(
    client: Any,
    model: str,
    windows: list[TextWindow],
    batch_size: int = 200,
    timeout: float = 3600,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Submit one or more Pangram bulk jobs and return successful and failed rows."""
    successes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    meta_by_id = {w.window_id: w for w in windows}
    progress = st.progress(0, text="Submitting Pangram bulk job…")
    total_batches = max(1, (len(windows) + batch_size - 1) // batch_size)

    for batch_no, offset in enumerate(range(0, len(windows), batch_size), start=1):
        chunk = windows[offset : offset + batch_size]
        items = [{"id": w.window_id, "text": w.text} for w in chunk]

        bulk = client.submit_bulk(items=items, model=model)
        bulk_id = bulk["bulk_id"]
        progress.progress(
            min(0.85, (batch_no - 1) / total_batches + 0.05),
            text=f"Pangram batch {batch_no}/{total_batches}: waiting for results…",
        )
        status = client.wait_for_bulk(bulk_id, timeout=timeout, poll_interval=0.5)
        results = client.get_bulk_results(bulk_id)

        for item in results.get("items", []):
            item_id = item.get("id")
            result = item.get("result")
            if result is None:
                failures.append(
                    {
                        "window_id": item_id,
                        "error": item.get("error") or f"No result; stage={item.get('stage')}",
                        "bulk_id": bulk_id,
                    }
                )
                continue

            meta = meta_by_id.get(item_id)
            if meta is None:
                failures.append(
                    {"window_id": item_id, "error": "Unknown result ID", "bulk_id": bulk_id}
                )
                continue

            row = asdict(meta)
            row.update(summarize_result(result))
            row["bulk_id"] = bulk_id
            row["raw_result"] = result
            successes.append(row)

        for failed in results.get("failed_items", []):
            failures.append(
                {
                    "window_id": failed.get("id"),
                    "error": failed.get("error") or "Pangram bulk item failed",
                    "bulk_id": bulk_id,
                }
            )

        progress.progress(
            min(0.98, batch_no / total_batches),
            text=f"Pangram batch {batch_no}/{total_batches} complete.",
        )

    progress.progress(1.0, text="Pangram analysis complete.")
    time.sleep(0.15)
    progress.empty()
    return successes, failures


# -------------------------
# Persistence
# -------------------------


def init_db() -> None:
    with sqlite3.connect(DB_PATH) as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                experiment_name TEXT,
                run_at TEXT NOT NULL,
                mode TEXT NOT NULL,
                model TEXT,
                source_name TEXT,
                expected_label TEXT,
                window_id TEXT,
                target_words INTEGER,
                actual_words INTEGER,
                sentence_start INTEGER,
                sentence_end INTEGER,
                prediction TEXT,
                headline TEXT,
                fraction_ai REAL,
                fraction_ai_assisted REAL,
                fraction_human REAL,
                mean_ai_involvement REAL,
                max_humanizer_score REAL,
                any_humanized INTEGER,
                window_confidence TEXT,
                version TEXT,
                dashboard_link TEXT,
                text TEXT,
                raw_json TEXT
            )
            """
        )
        con.commit()


def save_results(
    rows: list[dict[str, Any]],
    *,
    experiment_name: str,
    mode: str,
    model: str,
) -> str:
    init_db()
    experiment_id = str(uuid.uuid4())
    run_at = datetime.now(timezone.utc).isoformat()

    with sqlite3.connect(DB_PATH) as con:
        for r in rows:
            con.execute(
                """
                INSERT INTO scan_results (
                    experiment_id, experiment_name, run_at, mode, model,
                    source_name, expected_label, window_id, target_words, actual_words,
                    sentence_start, sentence_end, prediction, headline,
                    fraction_ai, fraction_ai_assisted, fraction_human,
                    mean_ai_involvement, max_humanizer_score, any_humanized,
                    window_confidence, version, dashboard_link, text, raw_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    experiment_id,
                    experiment_name,
                    run_at,
                    mode,
                    model,
                    r.get("source_name"),
                    r.get("expected_label"),
                    r.get("window_id"),
                    r.get("target_words"),
                    r.get("actual_words"),
                    r.get("sentence_start"),
                    r.get("sentence_end"),
                    r.get("prediction"),
                    r.get("headline"),
                    r.get("fraction_ai"),
                    r.get("fraction_ai_assisted"),
                    r.get("fraction_human"),
                    r.get("mean_ai_involvement"),
                    r.get("max_humanizer_score"),
                    1 if r.get("any_humanized") else 0,
                    r.get("window_confidence"),
                    r.get("version"),
                    r.get("dashboard_link"),
                    r.get("text"),
                    json.dumps(r.get("raw_result") or {}, ensure_ascii=False),
                ),
            )
        con.commit()
    return experiment_id


def load_history() -> pd.DataFrame:
    init_db()
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql_query(
            "SELECT * FROM scan_results ORDER BY id DESC",
            con,
        )


# -------------------------
# Analysis / display
# -------------------------


def results_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    display_rows = []
    for r in rows:
        d = {k: v for k, v in r.items() if k != "raw_result"}
        display_rows.append(d)
    return pd.DataFrame(display_rows)


def calibration_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    usable = df[df["expected_label"].isin(["Human", "AI"])].copy()
    if usable.empty:
        return pd.DataFrame()

    records = []
    for (label, size), group in usable.groupby(["expected_label", "target_words"]):
        n = len(group)
        records.append(
            {
                "Expected": label,
                "Target words": int(size),
                "N": n,
                "Human %": 100 * (group["prediction"] == "Human").mean(),
                "Mixed %": 100 * (group["prediction"] == "Mixed").mean(),
                "AI %": 100 * (group["prediction"] == "AI").mean(),
                "Mean human fraction": group["fraction_human"].mean(),
                "Mean AI fraction": group["fraction_ai"].mean(),
                "Mean AI involvement": group["mean_ai_involvement"].mean(),
            }
        )
    return pd.DataFrame(records).sort_values(["Target words", "Expected"])


def separation_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    records = []
    for size, group in df.groupby("target_words"):
        human = group[group["expected_label"] == "Human"]
        ai = group[group["expected_label"] == "AI"]
        if human.empty or ai.empty:
            continue
        human_correct = (human["prediction"] == "Human").mean()
        ai_correct = (ai["prediction"] == "AI").mean()
        records.append(
            {
                "Target words": int(size),
                "Human correctly Human %": 100 * human_correct,
                "AI correctly AI %": 100 * ai_correct,
                "Balanced decisive accuracy %": 100 * (human_correct + ai_correct) / 2,
                "Human N": len(human),
                "AI N": len(ai),
            }
        )
    return pd.DataFrame(records).sort_values("Target words")


def recommend_size(sep: pd.DataFrame, human_threshold: float, ai_threshold: float) -> str:
    if sep.empty:
        return "Not enough labeled Human and AI data to recommend a window size."
    candidates = sep[
        (sep["Human correctly Human %"] >= human_threshold * 100)
        & (sep["AI correctly AI %"] >= ai_threshold * 100)
    ]
    if candidates.empty:
        return (
            "No tested size met both thresholds. That is useful: either test larger windows, "
            "add more control texts, or accept a lower screening threshold for early experiments."
        )
    size = int(candidates.iloc[0]["Target words"])
    return (
        f"Smallest tested size meeting both thresholds: about **{size} words** "
        "(sentence-aligned, so individual windows may be somewhat longer)."
    )


def show_result_table(df: pd.DataFrame, key: str) -> None:
    if df.empty:
        st.info("No results yet.")
        return
    preferred = [
        "source_name",
        "expected_label",
        "target_words",
        "actual_words",
        "prediction",
        "fraction_human",
        "fraction_ai_assisted",
        "fraction_ai",
        "mean_ai_involvement",
        "max_humanizer_score",
        "window_confidence",
        "sentence_start",
        "sentence_end",
        "text",
    ]
    cols = [c for c in preferred if c in df.columns]
    st.dataframe(df[cols], use_container_width=True, hide_index=True, key=key)


def source_docs_from_uploads(files: Iterable[Any], expected_label: str) -> tuple[list[SourceDoc], list[str]]:
    docs = []
    errors = []
    for f in files or []:
        try:
            text = extract_text_from_upload(f)
            if count_words(text) < MIN_PANGRAM_WORDS:
                errors.append(f"{f.name}: fewer than {MIN_PANGRAM_WORDS} words after extraction")
                continue
            docs.append(SourceDoc(name=f.name, expected_label=expected_label, text=text))
        except Exception as exc:
            errors.append(f"{f.name}: {exc}")
    return docs, errors


def get_connected_client() -> tuple[Any | None, str | None]:
    return st.session_state.get("pangram_client"), st.session_state.get("pangram_model")


# -------------------------
# Streamlit app
# -------------------------

st.set_page_config(page_title=APP_TITLE, layout="wide")
init_db()

st.title(f"{APP_TITLE} {APP_VERSION}")
st.caption(
    "A small-sample laboratory for finding the shortest Pangram window that reliably separates "
    "known-human fiction from known-AI fiction, then using that window size for fast prompt experiments."
)

with st.sidebar:
    st.header("Pangram connection")
    api_key = get_secret_api_key()

    if Pangram is None:
        st.error(
            "pangram-sdk is not installed. Make sure requirements.txt is in the GitHub repo, "
            "then reboot the Streamlit app."
        )
    elif not api_key:
        st.error("PANGRAM_API_KEY is not set in Streamlit Secrets.")
        st.caption("In Streamlit Cloud: Manage app → Settings → Secrets, then add:")
        st.code('PANGRAM_API_KEY = "paste-your-key-here"', language="toml")
        st.caption("Save the secret and let Streamlit rerun the app. Do not put the key in GitHub.")
    else:
        # Auto-connect once per Streamlit session. No command line and no key-pasting in the app.
        if st.session_state.get("pangram_client") is None:
            try:
                with st.spinner("Connecting to Pangram…"):
                    client, models = connect_pangram(api_key)
                st.session_state["pangram_client"] = client
                st.session_state["pangram_models"] = models
                if "pangram-4" in models:
                    st.session_state["pangram_model"] = "pangram-4"
                elif models:
                    st.session_state["pangram_model"] = models[0]
            except Exception as exc:
                st.session_state.pop("pangram_client", None)
                st.session_state.pop("pangram_models", None)
                st.session_state.pop("pangram_model", None)
                st.error(f"Pangram connection failed: {exc}")

        if st.session_state.get("pangram_client") is not None:
            st.success("API key loaded from Streamlit Secrets.")

            if st.button("Refresh Pangram models", use_container_width=True):
                try:
                    with st.spinner("Refreshing Pangram models…"):
                        client, models = connect_pangram(api_key)
                    st.session_state["pangram_client"] = client
                    st.session_state["pangram_models"] = models
                    current = st.session_state.get("pangram_model")
                    if current not in models:
                        st.session_state["pangram_model"] = (
                            "pangram-4" if "pangram-4" in models else (models[0] if models else None)
                        )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not refresh models: {exc}")

    models = st.session_state.get("pangram_models", [])
    if models:
        current_model = st.session_state.get("pangram_model")
        idx = models.index(current_model) if current_model in models else 0
        selected = st.selectbox("Model", models, index=idx)
        st.session_state["pangram_model"] = selected
        st.caption("This list is read from the models currently enabled for your Pangram API key.")
    elif api_key and Pangram is not None:
        st.caption("No Pangram model list is available yet.")

    st.divider()
    st.caption(
        "Pangram 4 accepts prose samples of at least 50 words. This app enforces that minimum and "
        "builds windows on sentence boundaries."
    )

cal_tab, quick_tab, ab_tab, history_tab = st.tabs(
    ["1 · Corpus calibration", "2 · Quick scan", "3 · A/B prompt test", "4 · History"]
)


# -------------------------
# Tab 1: Calibration
# -------------------------
with cal_tab:
    st.subheader("Find the smallest useful test window")
    st.write(
        "Load writing you know is human and writing you know was generated by AI. The app creates "
        "mechanical, sentence-aligned windows at several sizes and sends them to Pangram in bulk."
    )

    c1, c2 = st.columns(2)
    with c1:
        human_files = st.file_uploader(
            "Known-human corpus",
            type=["docx", "txt", "md"],
            accept_multiple_files=True,
            key="human_uploads",
        )
    with c2:
        ai_files = st.file_uploader(
            "Known-AI corpus",
            type=["docx", "txt", "md"],
            accept_multiple_files=True,
            key="ai_uploads",
        )

    human_docs, human_errors = source_docs_from_uploads(human_files, "Human")
    ai_docs, ai_errors = source_docs_from_uploads(ai_files, "AI")
    for err in human_errors + ai_errors:
        st.warning(err)

    settings1, settings2, settings3 = st.columns(3)
    with settings1:
        sample_sizes = st.multiselect(
            "Target window sizes (words)",
            ALL_SAMPLE_SIZES,
            default=DEFAULT_SAMPLE_SIZES,
        )
    with settings2:
        overlap_pct = st.slider("Window overlap", 0, 75, 50, 25)
    with settings3:
        cap_per = st.number_input(
            "Max windows / file / size",
            min_value=1,
            max_value=200,
            value=12,
            step=1,
            help="Windows are selected evenly across each file when this cap is reached.",
        )

    docs = human_docs + ai_docs
    windows = make_calibration_windows(
        docs,
        [s for s in sample_sizes if s >= MIN_PANGRAM_WORDS],
        overlap_pct / 100.0,
        int(cap_per),
    ) if docs and sample_sizes else []

    total_words = sum(w.actual_words for w in windows)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Human files", len(human_docs))
    m2.metric("AI files", len(ai_docs))
    m3.metric("Pangram samples", len(windows))
    m4.metric("Words submitted", f"{total_words:,}")

    if windows:
        preview = pd.DataFrame([asdict(w) for w in windows])
        with st.expander("Preview the mechanical sample plan"):
            st.dataframe(
                preview[["source_name", "expected_label", "target_words", "actual_words", "sentence_start", "sentence_end"]],
                use_container_width=True,
                hide_index=True,
            )

    experiment_name = st.text_input(
        "Experiment name",
        value=f"Calibration {datetime.now().strftime('%Y-%m-%d')}",
        key="cal_experiment_name",
    )

    client, model = get_connected_client()
    run_disabled = not (client and model and windows and human_docs and ai_docs)
    if st.button("Run calibration through Pangram", type="primary", disabled=run_disabled):
        try:
            successes, failures = run_bulk_scan(client, model, windows)
            exp_id = save_results(
                successes,
                experiment_name=experiment_name,
                mode="Calibration",
                model=model,
            )
            st.session_state["cal_last_rows"] = successes
            st.session_state["cal_last_failures"] = failures
            st.session_state["cal_last_experiment_id"] = exp_id
        except Exception as exc:
            st.error(f"Pangram run failed: {exc}")

    rows = st.session_state.get("cal_last_rows", [])
    failures = st.session_state.get("cal_last_failures", [])
    if rows:
        df = results_dataframe(rows)
        st.divider()
        st.subheader("Calibration result")

        summary = calibration_summary(df)
        sep = separation_summary(df)

        if not sep.empty:
            st.markdown("**Separation by test length**")
            st.dataframe(sep, use_container_width=True, hide_index=True)
            chart = sep.set_index("Target words")[["Human correctly Human %", "AI correctly AI %"]]
            st.line_chart(chart)

            tc1, tc2 = st.columns(2)
            with tc1:
                human_thresh = st.slider("Required Human-control success", 0.50, 1.00, 0.90, 0.05)
            with tc2:
                ai_thresh = st.slider("Required AI-control success", 0.50, 1.00, 0.90, 0.05)
            st.info(recommend_size(sep, human_thresh, ai_thresh))

        with st.expander("Detailed calibration summary"):
            st.dataframe(summary, use_container_width=True, hide_index=True)

        with st.expander("Every tested window"):
            show_result_table(df, "cal_results_df")

        st.download_button(
            "Download calibration CSV",
            data=df.to_csv(index=False).encode("utf-8"),
            file_name="pangram_calibration_results.csv",
            mime="text/csv",
        )

        if failures:
            st.warning(f"{len(failures)} Pangram item(s) failed.")
            st.dataframe(pd.DataFrame(failures), use_container_width=True, hide_index=True)

    elif not run_disabled and not rows:
        st.caption("Ready. Run the calibration when you want to spend the API calls.")
    elif not client:
        st.info("Connect to Pangram in the sidebar first.")
    elif not (human_docs and ai_docs):
        st.info("Add at least one known-human and one known-AI document.")


# -------------------------
# Tab 2: Quick scan
# -------------------------
with quick_tab:
    st.subheader("Quick small-sample scan")
    st.write(
        "Use this while developing prompts: paste a new Claude passage, split it mechanically into small "
        "windows, and see whether the AI signal is already local or emerges only with more context."
    )

    quick_upload = st.file_uploader(
        "Optional DOCX / TXT / MD",
        type=["docx", "txt", "md"],
        accept_multiple_files=False,
        key="quick_upload",
    )
    default_quick_text = ""
    quick_name = "pasted_text.txt"
    if quick_upload is not None:
        try:
            default_quick_text = extract_text_from_upload(quick_upload)
            quick_name = quick_upload.name
        except Exception as exc:
            st.error(str(exc))

    quick_text = st.text_area(
        "Text to test",
        value=default_quick_text,
        height=260,
        key="quick_text",
    )

    q1, q2, q3 = st.columns(3)
    with q1:
        q_size = st.selectbox("Target words", ALL_SAMPLE_SIZES, index=ALL_SAMPLE_SIZES.index(200))
    with q2:
        q_overlap = st.slider("Overlap", 0, 75, 50, 25, key="quick_overlap")
    with q3:
        q_cap = st.number_input("Max windows", 1, 100, 20, 1, key="quick_cap")

    qdoc = SourceDoc(name=quick_name, expected_label="Unknown", text=clean_text(quick_text))
    qwindows = evenly_cap(
        build_sentence_windows(qdoc, int(q_size), q_overlap / 100.0),
        int(q_cap),
    ) if count_words(qdoc.text) >= MIN_PANGRAM_WORDS else []

    st.caption(
        f"{count_words(qdoc.text):,} source words → {len(qwindows)} Pangram window(s) → "
        f"{sum(w.actual_words for w in qwindows):,} submitted words."
    )

    qexp = st.text_input("Experiment name", value="Quick scan", key="quick_exp")
    client, model = get_connected_client()
    if st.button(
        "Scan these windows",
        type="primary",
        disabled=not (client and model and qwindows),
        key="quick_run",
    ):
        try:
            successes, failures = run_bulk_scan(client, model, qwindows)
            save_results(successes, experiment_name=qexp, mode="Quick", model=model)
            st.session_state["quick_last_rows"] = successes
            st.session_state["quick_last_failures"] = failures
        except Exception as exc:
            st.error(f"Pangram run failed: {exc}")

    qrows = st.session_state.get("quick_last_rows", [])
    if qrows:
        qdf = results_dataframe(qrows)
        qcounts = qdf["prediction"].value_counts().to_dict()
        cols = st.columns(3)
        cols[0].metric("Human windows", qcounts.get("Human", 0))
        cols[1].metric("Mixed windows", qcounts.get("Mixed", 0))
        cols[2].metric("AI windows", qcounts.get("AI", 0))
        show_result_table(qdf, "quick_results_df")
        st.download_button(
            "Download quick-scan CSV",
            qdf.to_csv(index=False).encode("utf-8"),
            file_name="pangram_quick_scan.csv",
            mime="text/csv",
        )


# -------------------------
# Tab 3: A/B prompt test
# -------------------------
with ab_tab:
    st.subheader("A/B prompt experiment")
    st.write(
        "Paste comparable output from the current prompt and one candidate prompt. The app uses the same "
        "window settings on both and compares their Pangram distributions."
    )

    a_col, b_col = st.columns(2)
    with a_col:
        control_text = st.text_area("Control output", height=300, key="control_text")
    with b_col:
        candidate_text = st.text_area("Candidate output", height=300, key="candidate_text")

    ab1, ab2, ab3 = st.columns(3)
    with ab1:
        ab_size = st.selectbox("Target words", ALL_SAMPLE_SIZES, index=ALL_SAMPLE_SIZES.index(200), key="ab_size")
    with ab2:
        ab_overlap = st.slider("Overlap", 0, 75, 50, 25, key="ab_overlap")
    with ab3:
        ab_cap = st.number_input("Max windows per side", 1, 100, 20, 1, key="ab_cap")

    cdoc = SourceDoc("Control", "Control", clean_text(control_text))
    ndoc = SourceDoc("Candidate", "Candidate", clean_text(candidate_text))
    cwindows = evenly_cap(build_sentence_windows(cdoc, int(ab_size), ab_overlap / 100.0), int(ab_cap)) if count_words(cdoc.text) >= MIN_PANGRAM_WORDS else []
    nwindows = evenly_cap(build_sentence_windows(ndoc, int(ab_size), ab_overlap / 100.0), int(ab_cap)) if count_words(ndoc.text) >= MIN_PANGRAM_WORDS else []
    abwindows = cwindows + nwindows

    abexp = st.text_input("Experiment name", value="Prompt A/B", key="ab_exp")
    client, model = get_connected_client()
    if st.button(
        "Run A/B Pangram test",
        type="primary",
        disabled=not (client and model and cwindows and nwindows),
        key="ab_run",
    ):
        try:
            successes, failures = run_bulk_scan(client, model, abwindows)
            save_results(successes, experiment_name=abexp, mode="A/B", model=model)
            st.session_state["ab_last_rows"] = successes
            st.session_state["ab_last_failures"] = failures
        except Exception as exc:
            st.error(f"Pangram run failed: {exc}")

    abrows = st.session_state.get("ab_last_rows", [])
    if abrows:
        abdf = results_dataframe(abrows)
        compare = []
        for label, group in abdf.groupby("expected_label"):
            compare.append(
                {
                    "Version": label,
                    "Windows": len(group),
                    "Human %": 100 * (group["prediction"] == "Human").mean(),
                    "Mixed %": 100 * (group["prediction"] == "Mixed").mean(),
                    "AI %": 100 * (group["prediction"] == "AI").mean(),
                    "Mean human fraction": group["fraction_human"].mean(),
                    "Mean AI fraction": group["fraction_ai"].mean(),
                    "Mean AI involvement": group["mean_ai_involvement"].mean(),
                }
            )
        cmp_df = pd.DataFrame(compare)
        st.dataframe(cmp_df, use_container_width=True, hide_index=True)
        if not cmp_df.empty:
            st.bar_chart(cmp_df.set_index("Version")[["Human %", "Mixed %", "AI %"]])
        with st.expander("Every A/B window"):
            show_result_table(abdf, "ab_results_df")


# -------------------------
# Tab 4: History
# -------------------------
with history_tab:
    st.subheader("Experiment history")
    st.caption(
        "Streamlit Cloud can recycle an app's local filesystem. Treat this built-in SQLite history as "
        "working history, not permanent storage; download CSVs you want to keep."
    )
    hist = load_history()
    if hist.empty:
        st.info("No saved Pangram runs yet.")
    else:
        e1, e2, e3 = st.columns(3)
        e1.metric("Saved windows", len(hist))
        e2.metric("Experiments", hist["experiment_id"].nunique())
        e3.metric("Latest run", str(hist.iloc[0]["run_at"])[:19].replace("T", " "))

        names = ["All"] + sorted([x for x in hist["experiment_name"].dropna().unique()])
        selected_name = st.selectbox("Experiment", names)
        shown = hist if selected_name == "All" else hist[hist["experiment_name"] == selected_name]

        display_cols = [
            "run_at",
            "experiment_name",
            "mode",
            "model",
            "source_name",
            "expected_label",
            "target_words",
            "actual_words",
            "prediction",
            "fraction_human",
            "fraction_ai_assisted",
            "fraction_ai",
            "mean_ai_involvement",
            "max_humanizer_score",
            "text",
        ]
        st.dataframe(
            shown[[c for c in display_cols if c in shown.columns]],
            use_container_width=True,
            hide_index=True,
        )
        st.download_button(
            "Download history CSV",
            shown.to_csv(index=False).encode("utf-8"),
            file_name="pangram_microscope_history.csv",
            mime="text/csv",
        )

st.divider()
st.caption(
    "Interpretation rule: this is an experimental screening tool, not a rewriting loop. Keep the drafting model "
    "blind to Pangram results; use the measurements to compare hypotheses and prompt versions."
)
