"""
Micro-Prompt Harness — Simplified
=================================
Generate chapter drafts from prompt variants × temperatures × repetitions.
Evaluate batches with Opus. Pick a winner.

The generation prompt lives in prompts.csv. The app does not inject its own
drafting instructions. Whatever the prompt says, the model gets — plus the
uploaded documents as context.

Architecture:
  prompts.csv       →  prompt variants (user-authored)
  uploaded docs     →  outline, source text, combined text (user uploads)
  app.py            →  runs the calls, saves everything, evaluates
"""

import io
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd
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
CSV_FILENAME = "runs.csv"
PROMPTS_CSV = "prompts.csv"

DEFAULT_GEN_MODEL = "claude-sonnet-4-5"
DEFAULT_EVAL_MODEL = "claude-opus-4-6"
MAX_GEN_TOKENS = 16000
MAX_EVAL_TOKENS = 6000


# ============================================================================
# Evaluator prompt — comparative ranking
# ============================================================================

EVALUATOR_PROMPT = """You are reading {N} drafts of the same chapter of a novel. They come from different generation runs and may vary in prompt or temperature. Your job is to identify which drafts a serious reader of this specific genre would most want to keep reading.

Read every draft in full. Do not skim. Infer the project register from the drafts themselves — genre, period, point of view, voice, narrator's class and position. Hold the drafts to their own standard.

Judge as an experienced reader of this genre would, giving weight to:

- Specificity over atmosphere. Reward concrete observed detail — objects, gestures, numbers, names, prices. Penalize drafts that produce the sensation of good writing through cadence alone.
- Render over interpret. Penalize drafts that name emotions, summarize their own meaning, or close paragraphs with an interpretive sentence. Watch for "the way..." / "how..." observation framing, "as though..." similes at beat-ends, and negation pivots ("not X but Y").
- Dialogue doing dramatic work. Lines must carry tension, subtext, character, or forward motion. Penalize polite turns stating positions, or exposition in quotation marks.
- Voice consistency. The established voice must hold throughout without drift, pastiche, or anachronism.
- Trust in the ending. The strongest draft ends on the beat it has earned and stops. Weaker drafts add a coda explaining or softening the beat.
- Restraint with simile, aphorism, and summary. Reward sentences that leave the reader to do the work.

Be demanding. Do not be diplomatic.

OUTPUT FORMAT

For each draft, write a brief paragraph (2-4 sentences) citing a specific passage.

Then a comparison paragraph naming the top 2-3 contenders and why the top one edges the others.

Then on a line by itself:

RANKING: N, N, N, ...

(every draft number from strongest to weakest, separated by commas, each draft exactly once)

Then on the final line:

WINNER: N

Nothing after that line."""


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


RUN_FIELDS = list(RunRecord.__dataclass_fields__.keys())


# ============================================================================
# File I/O
# ============================================================================

def ensure_dirs():
    RUNS_DIR.mkdir(exist_ok=True)
    OUTPUTS_DIR.mkdir(exist_ok=True)


def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def load_csv(path: Path) -> pd.DataFrame:
    if path.exists():
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
    """Extract text from .txt or .docx upload."""
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
# Prompt loading
# ============================================================================

def load_prompts() -> pd.DataFrame:
    """Load prompts.csv from repo root. Expects columns: id, text. Optional: category."""
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
# Payload construction — minimal
# ============================================================================

def build_payload(prompt_text: str, doc_texts: dict[str, str]) -> str:
    """
    Build the full message sent to the model.
    prompt_text: the micro-prompt from CSV
    doc_texts: dict of {label: text} for uploaded documents
    """
    parts = [prompt_text.strip()]

    for label, text in doc_texts.items():
        if text.strip():
            parts.append(f"\n\n=== {label.upper()} ===\n\n{text.strip()}")

    parts.append(
        "\n\nWrite the full chapter now. Return plain text only, "
        "with normal paragraph breaks and no commentary."
    )
    return "\n".join(parts)


# ============================================================================
# Generation
# ============================================================================

def generate_chapter(client, model: str, temperature: float, payload: str) -> str:
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_GEN_TOKENS,
        temperature=temperature,
        messages=[{"role": "user", "content": payload}],
    )
    return "\n".join(b.text for b in resp.content if getattr(b, "text", None))


# ============================================================================
# Evaluation
# ============================================================================

def evaluate_drafts_with_anthropic(client, model: str, drafts: list[dict]) -> dict:
    """
    drafts: list of {"run_id": str, "text": str}
    Returns: {"winner_run_id", "winner_index", "ranking", "raw_text", "parse_status", "model"}
    """
    n = len(drafts)
    parts = [EVALUATOR_PROMPT.format(N=n), "\n\n"]
    for i, d in enumerate(drafts, 1):
        parts.append(f"=== DRAFT {i} (run_id: {d['run_id']}) ===\n\n{d['text']}\n\n")

    resp = client.messages.create(
        model=model,
        max_tokens=MAX_EVAL_TOKENS,
        temperature=0,
        messages=[{"role": "user", "content": "".join(parts)}],
    )
    raw = "\n".join(b.text for b in resp.content if getattr(b, "text", None))

    # Parse RANKING
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

    # Parse WINNER
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
# File naming
# ============================================================================

def make_file_stub(prompt_id: int, temperature: float, model: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_short = model.split("-")[-1][:6] if "-" in model else model[:6]
    return f"P{prompt_id} T{temperature} {model_short} {ts}"


def make_winner_filename(prompt_id: int, temperature: float, model: str) -> str:
    stub = make_file_stub(prompt_id, temperature, model)
    return f"WINNER {stub}.txt"


# ============================================================================
# Export
# ============================================================================

def export_zip(df: pd.DataFrame, file_paths: list[Path]) -> bytes:
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


def gather_output_paths(df: pd.DataFrame) -> list[Path]:
    paths = []
    for col in ["output_file", "payload_file", "meta_file"]:
        if col in df.columns:
            for val in df[col].dropna():
                p = Path(str(val))
                if p.exists():
                    paths.append(p)
    return paths


# ============================================================================
# Streamlit UI
# ============================================================================

st.set_page_config(page_title="Micro-Prompt Harness", layout="wide")
st.title("Micro-Prompt Harness")
st.caption("Generate · Evaluate · Pick the winner")

ensure_dirs()
csv_path = RUNS_DIR / CSV_FILENAME

# --- Sidebar ---
with st.sidebar:
    st.header("Configuration")

    # API key
    default_key = ""
    try:
        default_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    except Exception:
        pass
    api_key = st.text_input("Anthropic API Key", type="password", value=default_key)

    st.markdown("---")

    # Model selection
    gen_model = st.text_input("Generation model", value=DEFAULT_GEN_MODEL)
    eval_model = st.text_input("Evaluation model", value=DEFAULT_EVAL_MODEL)

    st.markdown("---")

    # Temperature
    temps_input = st.text_input("Temperatures (comma-separated)", value="0.6, 0.7")
    try:
        temperatures = [float(t.strip()) for t in temps_input.split(",") if t.strip()]
    except ValueError:
        temperatures = [0.7]
        st.warning("Could not parse temperatures. Using 0.7.")

    # Repetitions
    repetitions = st.number_input("Repetitions per prompt×temp", min_value=1, max_value=10, value=3)

    st.markdown("---")

    # Document uploads
    st.subheader("Documents")
    st.caption("Upload the files the prompt references. Label them so you know what's what.")

    doc_uploads = {}
    outline_file = st.file_uploader("Outline", type=["txt", "docx"], key="outline")
    if outline_file:
        doc_uploads["Outline"] = extract_text_from_upload(outline_file)

    source_file = st.file_uploader("Source text (voice model)", type=["txt", "docx"], key="source")
    if source_file:
        doc_uploads["Source Text"] = extract_text_from_upload(source_file)

    combined_file = st.file_uploader("Combined text (optional)", type=["txt", "docx"], key="combined")
    if combined_file:
        doc_uploads["Combined Text"] = extract_text_from_upload(combined_file)

    extra_file = st.file_uploader("Additional doc (optional)", type=["txt", "docx"], key="extra")
    if extra_file:
        doc_uploads[extra_file.name] = extract_text_from_upload(extra_file)

    st.markdown("---")
    st.caption(f"Gen: `{gen_model}` · Eval: `{eval_model}`")
    st.caption(f"Temps: {temperatures} · Reps: {repetitions}")
    if doc_uploads:
        st.caption(f"Docs loaded: {', '.join(doc_uploads.keys())}")

# --- Load prompts ---
prompts_df = load_prompts()

if prompts_df.empty:
    st.warning(
        f"No `{PROMPTS_CSV}` found in the repo root, or it has no rows. "
        f"Create a CSV with columns `id` and `text` (and optionally `category`)."
    )
    st.stop()

# --- Main area: two columns ---
left_col, right_col = st.columns([1, 1])

with left_col:
    st.subheader("Prompts")

    # Prompt selection
    prompt_options = {
        f"P{row['id']}: {str(row['text'])[:80]}...": int(row["id"])
        for _, row in prompts_df.iterrows()
    }
    selected_labels = st.multiselect(
        "Select prompt(s) to run",
        options=list(prompt_options.keys()),
        default=list(prompt_options.keys()),
    )
    selected_ids = [prompt_options[label] for label in selected_labels]

    # Preview selected prompts
    for pid in selected_ids:
        row = prompts_df[prompts_df["id"].astype(int) == pid].iloc[0]
        with st.expander(f"P{pid} — {str(row.get('category', ''))}"):
            st.text(str(row["text"]))

    # Total runs
    total_runs = len(selected_ids) * len(temperatures) * repetitions
    st.write(
        f"**{len(selected_ids)}** prompts × **{len(temperatures)}** temps × "
        f"**{repetitions}** reps = **{total_runs}** drafts"
    )

    # Generate button
    if st.button("Generate", type="primary", disabled=not api_key or total_runs == 0):
        client = anthropic.Anthropic(api_key=api_key)
        progress = st.progress(0.0)
        status = st.empty()
        run_count = 0

        for pid in selected_ids:
            prompt_row = prompts_df[prompts_df["id"].astype(int) == pid].iloc[0]
            prompt_text = str(prompt_row["text"])

            for temp in temperatures:
                for rep in range(1, repetitions + 1):
                    run_count += 1
                    status.info(f"Run {run_count}/{total_runs}: P{pid} T{temp} R{rep:02d}")

                    payload = build_payload(prompt_text, doc_uploads)
                    stub = make_file_stub(pid, temp, gen_model)
                    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:20]

                    # Save payload
                    payload_path = OUTPUTS_DIR / f"{stub}_payload.txt"
                    save_text(payload_path, payload)

                    try:
                        output = generate_chapter(client, gen_model, temp, payload)
                    except Exception as e:
                        st.error(f"Generation failed for P{pid} T{temp} R{rep}: {e}")
                        continue

                    # Save output
                    output_path = OUTPUTS_DIR / f"{stub}_output.txt"
                    save_text(output_path, output)

                    # Save meta
                    meta = {
                        "run_id": run_id,
                        "prompt_id": pid,
                        "temperature": temp,
                        "model": gen_model,
                        "repetition": rep,
                        "timestamp": datetime.now().isoformat(),
                        "documents": list(doc_uploads.keys()),
                    }
                    meta_path = OUTPUTS_DIR / f"{stub}_meta.json"
                    save_text(meta_path, json.dumps(meta, indent=2))

                    # Append to CSV
                    record = RunRecord(
                        run_id=run_id,
                        timestamp=datetime.now().isoformat(),
                        prompt_id=pid,
                        prompt_text=prompt_text[:200],
                        temperature=temp,
                        model=gen_model,
                        output_file=str(output_path),
                        payload_file=str(payload_path),
                        meta_file=str(meta_path),
                        word_count=len(output.split()),
                    )
                    append_record(csv_path, record)

                    progress.progress(run_count / total_runs)
                    time.sleep(0.5)  # gentle rate buffer

        progress.empty()
        status.success(f"Done. {run_count} drafts generated.")
        st.rerun()


with right_col:
    st.subheader("Run log")

    df = load_csv(csv_path)
    if df.empty:
        st.info("No runs yet. Generate some drafts.")
    else:
        # Display table
        display_cols = ["run_id", "prompt_id", "temperature", "model", "word_count",
                        "is_winner", "evaluation_rank"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True)

        # --- Evaluate latest batch ---
        st.markdown("---")
        st.subheader("Evaluate")

        # Let user pick how many recent runs to evaluate
        max_eval = min(25, len(df))
        eval_count = st.slider("Drafts to evaluate (most recent N)", 2, max_eval, min(12, max_eval))

        if st.button("Evaluate", type="primary", disabled=not api_key):
            client = anthropic.Anthropic(api_key=api_key)

            # Take the most recent N runs
            batch_df = df.tail(eval_count).copy()
            drafts = []
            for _, row in batch_df.iterrows():
                output_path = Path(str(row["output_file"]))
                if output_path.exists():
                    text = output_path.read_text(encoding="utf-8")
                    drafts.append({"run_id": str(row["run_id"]), "text": text})

            if len(drafts) < 2:
                st.error("Need at least 2 readable drafts to evaluate.")
            else:
                with st.spinner(f"Evaluating {len(drafts)} drafts with {eval_model}..."):
                    try:
                        result = evaluate_drafts_with_anthropic(client, eval_model, drafts)

                        winner_run_id = result["winner_run_id"]
                        winner_row = batch_df[batch_df["run_id"].astype(str) == str(winner_run_id)].iloc[0]

                        # Save winner file
                        winner_text = Path(str(winner_row["output_file"])).read_text(encoding="utf-8")
                        winner_filename = make_winner_filename(
                            int(winner_row["prompt_id"]),
                            float(winner_row["temperature"]),
                            str(winner_row["model"]),
                        )
                        winner_path = OUTPUTS_DIR / winner_filename
                        save_text(winner_path, winner_text)

                        # Update records
                        evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                        batch_run_ids = [str(r) for r in batch_df["run_id"].astype(str).tolist()]
                        update_records_bulk(csv_path, batch_run_ids, {
                            "is_winner": False,
                            "evaluation_id": evaluation_id,
                            "evaluator_model": result["model"],
                            "evaluation_parse_status": result["parse_status"],
                            "evaluation_raw": result["raw_text"],
                        })
                        update_record(csv_path, str(winner_run_id), {"is_winner": True})

                        # Write per-run ranks
                        for rank_pos, draft_num in enumerate(result["ranking"], 1):
                            rid = drafts[draft_num - 1]["run_id"]
                            update_record(csv_path, rid, {"evaluation_rank": rank_pos})

                        st.success(f"Winner: {winner_run_id}. Saved to {winner_filename}.")

                        # Show ranking
                        st.markdown("**Ranking (best → worst):**")
                        for rank_pos, draft_num in enumerate(result["ranking"], 1):
                            d = drafts[draft_num - 1]
                            marker = " ★" if d["run_id"] == winner_run_id else ""
                            st.write(f"{rank_pos}. {d['run_id']}{marker}")

                        with st.expander("Evaluator reasoning"):
                            st.text(result["raw_text"])

                    except Exception as e:
                        st.error(f"Evaluation failed: {e}")

                st.rerun()

        # --- Downloads ---
        st.markdown("---")
        all_paths = gather_output_paths(df)
        if all_paths:
            zip_bytes = export_zip(df, all_paths)
            st.download_button(
                "Download all runs (ZIP)",
                data=zip_bytes,
                file_name=f"micro_prompt_runs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
            )

        # CSV only
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        st.download_button(
            "Download runs.csv",
            data=csv_buf.getvalue(),
            file_name="runs.csv",
            mime="text/csv",
        )
