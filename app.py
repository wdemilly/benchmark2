import io
import json
import time
import zipfile
import hashlib
import shutil
import re
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Iterable

import pandas as pd
import streamlit as st

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None

APP_TITLE = "Micro-Prompt Harness"
DATA_DIR = Path("micro_prompt_runs")
DATA_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR = DATA_DIR / "flat_outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

DEFAULT_BASE_PROMPT = '''You are not Claude. You are the author of the combined source texts document.

You wrote every passage in the combined source texts document. The character profiles are your notes. The outline is your plan for this chapter.

Read all attached documents from beginning to end. Do not sample them.

Then write the chapter from the outline exactly as you would write it yourself. Construct each sentence from within the habits of mind, sentence movement, and narrative logic already present in the source texts. Write the chapter straight through in one continuous pass, first sentence to last. Do not draft short and expand. Return plain text only, with normal prose paragraph breaks and no commentary.'''
DEFAULT_MODEL = "claude-opus-4-6"
DEFAULT_EVALUATOR_MODEL = "claude-opus-4-6"
DEFAULT_MAX_TOKENS = 12000
MAX_ALLOWED_TOKENS = 32000
MAX_CONTINUATIONS = 4
PROMPTS_CSV = Path("prompts.csv")
CURRENT_SESSION_KEY = "app_session_id"
CURRENT_SESSION_RUN_IDS_KEY = "app_session_run_ids"
LATEST_BATCH_RUN_IDS_KEY = "app_latest_batch_run_ids"

EVALUATOR_PROMPT = '''You are reading N drafts of the same chapter of a novel. The drafts were generated from the same source material and outline but with different micro-prompt variations. Your task is to select the single strongest draft on the basis of prose quality.

Read every draft in full before judging. Do not skim.

Before you judge, infer the project's register from the drafts and source material: genre and subgenre, period, point of view, tense, narrator's class and position, and the established voice. Hold the drafts to their own standard. A historical novel should sound like one; a contemporary literary novel should not be judged for failing to sound historical. A first-person working-class narrator should not be penalized for lacking formal diction. Judge each draft against what this book is trying to be.

Judge on these criteria, in roughly this order of importance:

1. RENDER VS. INTERPRET. The strongest draft shows; the weaker drafts explain. Reward drafts that trust the reader to draw meaning from concrete physical detail, gesture, and action. Penalize drafts that name emotions, summarize their own meaning, or close paragraphs with interpretive sentences that tell the reader what the scene meant. Polish that performs itself is a failure mode, not a strength.

2. DIALOGUE DOING DRAMATIC WORK. Dialogue must carry tension, subtext, emotion, or forward motion -- ideally more than one at once. Penalize lines that could be cut without loss, characters taking polite turns stating positions, or exposition delivered in quotation marks.

3. AVOIDANCE OF SPECIFIC TELLS. Penalize drafts that use any of these:
   - Oppositional / negation-pivot constructions: "not X, but Y"; "it wasn't that she was tired, it was that--"; split versions across adjacent sentences.
   - "The way..." or "how..." prefaces used to frame observation as significant ("the way the light fell," "how he held the reins").
   - Em dashes in the first paragraph (hard fail).
   - Frequent em dashes elsewhere.

4. VOICE CONSISTENCY AND PERIOD/REGISTER FIDELITY. The established voice must hold throughout without drift, pastiche, or anachronism. Whatever period, class, or register the project has set for itself, the draft must sustain it. Reward specificity of detail appropriate to the narrator's position and world; penalize generic atmosphere or register slippage.

5. SENTENCE CRAFT. Reward verbs that do work without adverbial propping; concrete nouns over abstract ones; rhythm that enacts rather than decorates; restraint from aphoristic or summary sentences; appropriate withholding.

You will receive the drafts labeled DRAFT 1, DRAFT 2, etc. Read every draft in full before beginning your analysis.

Then produce your evaluation in this exact format:

First, for each draft, write a short paragraph (3-6 sentences) assessing it against the criteria above. Quote specific sentences when flagging failures or praising successes. Do not be diplomatic — the weaker drafts should be identified as weaker and why.

Then write a comparison paragraph that names the two or three closest contenders and the specific reasons one edges out the others.

Then, on the final line of your response, write exactly:

WINNER: N

where N is the number of the winning draft. Nothing after that line. The word WINNER must appear in all caps followed by a colon and the integer.
'''


def ensure_session_state() -> str:
    if CURRENT_SESSION_KEY not in st.session_state:
        st.session_state[CURRENT_SESSION_KEY] = datetime.now().strftime("%Y%m%d_%H%M%S")
    if CURRENT_SESSION_RUN_IDS_KEY not in st.session_state:
        st.session_state[CURRENT_SESSION_RUN_IDS_KEY] = []
    if LATEST_BATCH_RUN_IDS_KEY not in st.session_state:
        st.session_state[LATEST_BATCH_RUN_IDS_KEY] = []
    return str(st.session_state[CURRENT_SESSION_KEY])


@dataclass
class RunRecord:
    run_id: str
    session_id: str
    timestamp: str
    batch_label: str
    prompt_id: int
    repetition_index: int
    category: str
    provider: str
    model: str
    temperature: float
    max_tokens: int
    continuation_rounds: int
    source_name: str
    outline_name: str
    profiles_name: str
    file_stub: str
    output_file: str
    payload_file: str
    micro_prompt_file: str
    meta_file: str
    output_sha256: str = ""
    stop_reason: str = ""
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    output_words: Optional[int] = None
    truncation_flag: bool = False
    evaluation_id: str = ""
    is_winner: bool = False
    evaluation_raw: str = ""
    evaluation_parse_status: str = ""
    evaluator_model: str = ""
    originality_label: str = ""
    originality_score: Optional[float] = None
    manual_rating: str = ""
    manual_notes: str = ""


def load_prompt_definitions(csv_path: Path) -> List[dict]:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {csv_path}. Place prompts.csv beside app.py."
        )

    df = pd.read_csv(csv_path)

    required_columns = ["id", "category", "text"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Prompt file is missing required column(s): {', '.join(missing)}"
        )

    if df.empty:
        raise ValueError("Prompt file is empty.")

    prompts: List[dict] = []
    seen_ids = set()

    for row_number, row in df.iterrows():
        raw_id = row["id"]
        raw_category = row["category"]
        raw_text = row["text"]

        if pd.isna(raw_id):
            raise ValueError(f"Row {row_number + 2}: id is blank.")
        if pd.isna(raw_category) or not str(raw_category).strip():
            raise ValueError(f"Row {row_number + 2}: category is blank.")
        if pd.isna(raw_text) or not str(raw_text).strip():
            raise ValueError(f"Row {row_number + 2}: text is blank.")

        try:
            prompt_id = int(raw_id)
        except Exception as exc:
            raise ValueError(f"Row {row_number + 2}: id must be an integer.") from exc

        if prompt_id in seen_ids:
            raise ValueError(f"Duplicate prompt id found: {prompt_id}")
        seen_ids.add(prompt_id)

        prompts.append(
            {
                "id": prompt_id,
                "category": str(raw_category).strip(),
                "text": str(raw_text).strip(),
            }
        )

    prompts.sort(key=lambda p: p["id"])
    return prompts


def normalize_text(text: str) -> str:
    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2013", "-")
        .replace("\u2014", "--")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00a0", " ")
    )


def normalize_output_text(text: str) -> str:
    text = normalize_text(text)
    lines = [line.rstrip() for line in text.split("\n")]
    return "\n".join(lines).strip() + "\n"


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def decode_uploaded_text(uploaded_file) -> str:
    raw = uploaded_file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return normalize_text(text)


def extract_text_from_response(resp) -> str:
    parts: List[str] = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "".join(parts)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_records(csv_path: Path) -> pd.DataFrame:
    columns = [field for field in RunRecord.__dataclass_fields__.keys()]

    text_columns = [
        "run_id",
        "session_id",
        "timestamp",
        "batch_label",
        "category",
        "provider",
        "model",
        "source_name",
        "outline_name",
        "profiles_name",
        "file_stub",
        "output_file",
        "payload_file",
        "micro_prompt_file",
        "meta_file",
        "output_sha256",
        "stop_reason",
        "evaluation_id",
        "evaluation_raw",
        "evaluation_parse_status",
        "evaluator_model",
        "originality_label",
        "manual_rating",
        "manual_notes",
    ]

    bool_columns = [
        "truncation_flag",
        "is_winner",
    ]

    numeric_columns = [
        "prompt_id",
        "repetition_index",
        "temperature",
        "max_tokens",
        "continuation_rounds",
        "input_tokens",
        "output_tokens",
        "output_words",
        "originality_score",
    ]

    if csv_path.exists():
        df = pd.read_csv(csv_path)

        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA

        for col in text_columns:
            if col in df.columns:
                df[col] = df[col].astype("object")

        for col in bool_columns:
            if col in df.columns:
                df[col] = df[col].astype("object")

        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df[columns]

    return pd.DataFrame(columns=columns)


def append_record(csv_path: Path, record: RunRecord) -> None:
    df = load_records(csv_path)
    new_row = pd.DataFrame([asdict(record)])

    for col in df.columns:
        if col in new_row.columns and df[col].dtype == object:
            new_row[col] = new_row[col].astype("object")

    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_path, index=False)


def update_record(csv_path: Path, run_id: str, updates: dict) -> None:
    df = load_records(csv_path)
    if df.empty:
        return
    mask = df["run_id"].astype(str) == str(run_id)
    if not mask.any():
        return
    for key, value in updates.items():
        if key in df.columns:
            if isinstance(value, str):
                df[key] = df[key].astype("object")
            df.loc[mask, key] = value
    df.to_csv(csv_path, index=False)


def clear_all_run_data(csv_path: Path, outputs_root: Path) -> None:
    if csv_path.exists():
        csv_path.unlink()
    if outputs_root.exists():
        for item in outputs_root.iterdir():
            if item.is_file() or item.is_symlink():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    outputs_root.mkdir(parents=True, exist_ok=True)
    st.session_state[CURRENT_SESSION_RUN_IDS_KEY] = []
    st.session_state[CURRENT_SESSION_KEY] = datetime.now().strftime("%Y%m%d_%H%M%S")


def build_payload(base_prompt: str, micro_prompt: str, source_text: str, outline_text: str, profiles_text: str) -> str:
    parts = [
        "BASE PROMPT",
        base_prompt.strip(),
        "",
        "TEST MICRO-PROMPT",
        micro_prompt.strip(),
        "",
        "BEGIN COMBINED SOURCE TEXTS",
        source_text.strip(),
        "END COMBINED SOURCE TEXTS",
        "",
        "BEGIN OUTLINE",
        outline_text.strip(),
        "END OUTLINE",
    ]
    if profiles_text.strip():
        parts.extend([
            "",
            "BEGIN CHARACTER PROFILES",
            profiles_text.strip(),
            "END CHARACTER PROFILES",
        ])
    parts.extend([
        "",
        "Write the full chapter now. Return plain text only, with normal paragraph breaks and no commentary."
    ])
    return "\n".join(parts)


def get_usage_tokens(resp) -> Tuple[Optional[int], Optional[int]]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None, None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    return input_tokens, output_tokens


def anthropic_messages_create_with_backoff(
    client,
    *,
    model: str,
    max_tokens: int,
    temperature: float,
    messages: list,
    max_attempts: int = 6,
    initial_delay: float = 2.0,
):
    last_exc = None
    delay = initial_delay

    for attempt in range(1, max_attempts + 1):
        try:
            return client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=messages,
            )
        except Exception as exc:
            last_exc = exc
            message = str(exc).lower()

            retryable = any(
                token in message
                for token in [
                    "rate limit",
                    "rate_limit",
                    "overloaded",
                    "timeout",
                    "timed out",
                    "connection",
                    "529",
                    "502",
                    "503",
                    "504",
                ]
            )

            if not retryable or attempt >= max_attempts:
                raise

            time.sleep(delay)
            delay = min(delay * 2, 20.0)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Unknown Anthropic API failure.")


def call_anthropic_with_continuation(
    api_key: str,
    model: str,
    payload: str,
    max_tokens: int,
    temperature: float,
    max_continuations: int = MAX_CONTINUATIONS,
) -> dict:
    if anthropic is None:
        raise RuntimeError("anthropic package is not installed.")

    client = anthropic.Anthropic(api_key=api_key)

    messages = [{"role": "user", "content": payload}]
    collected_parts: List[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    continuation_rounds = 0
    final_stop_reason = ""
    last_response_id = ""

    for round_index in range(max_continuations + 1):
        resp = anthropic_messages_create_with_backoff(
            client,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
        )

        text_chunk = extract_text_from_response(resp)
        if not text_chunk.strip():
            raise RuntimeError("Model returned an empty text block.")

        text_chunk = normalize_output_text(text_chunk)
        collected_parts.append(text_chunk)

        input_tokens, output_tokens = get_usage_tokens(resp)
        if input_tokens is not None:
            total_input_tokens += int(input_tokens)
        if output_tokens is not None:
            total_output_tokens += int(output_tokens)

        final_stop_reason = str(getattr(resp, "stop_reason", "") or "")
        last_response_id = str(getattr(resp, "id", "") or "")

        if final_stop_reason == "max_tokens":
            continuation_rounds += 1
            if round_index >= max_continuations:
                break

            messages.append({"role": "assistant", "content": text_chunk})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Continue exactly where you left off. Do not restart. "
                        "Do not summarize. Do not add commentary. "
                        "Continue the chapter in plain text only."
                    ),
                }
            )
            time.sleep(1.0)
            continue

        break

    combined_text = "".join(collected_parts)
    combined_text = normalize_output_text(combined_text)
    truncation_flag = final_stop_reason == "max_tokens"
    output_words = len(combined_text.split())

    return {
        "text": combined_text,
        "stop_reason": final_stop_reason,
        "input_tokens": total_input_tokens or None,
        "output_tokens": total_output_tokens or None,
        "output_words": output_words,
        "truncation_flag": truncation_flag,
        "continuation_rounds": continuation_rounds,
        "response_id": last_response_id,
    }


def parse_winner_integer(text: str, max_valid: int) -> Tuple[Optional[int], str]:
    cleaned = (text or "").strip()
    if not cleaned:
        return None, "failed"

    # Preferred: explicit 'WINNER: N' declaration (case-insensitive; last one wins)
    winner_matches = list(re.finditer(r"WINNER\s*[:\-]\s*(\d+)", cleaned, re.IGNORECASE))
    if winner_matches:
        value = int(winner_matches[-1].group(1))
        if 1 <= value <= max_valid:
            return value, "clean"

    if re.fullmatch(r"\d+[.\s]*", cleaned):
        value = int(re.search(r"\d+", cleaned).group(0))
        if 1 <= value <= max_valid:
            return value, "clean"
        return None, "failed"

    # Last-resort: the final integer in the response, if in valid range
    all_ints = list(re.finditer(r"\b(\d+)\b", cleaned))
    for match in reversed(all_ints):
        value = int(match.group(1))
        if 1 <= value <= max_valid:
            return value, "parsed"

    return None, "failed"


def evaluate_drafts_with_anthropic(
    api_key: str,
    model: str,
    drafts: List[Tuple[str, str]],
) -> dict:
    if anthropic is None:
        raise RuntimeError("anthropic package is not installed.")

    if not model or not model.strip():
        raise RuntimeError("Evaluator model is blank.")

    if len(drafts) < 2:
        raise RuntimeError("Evaluation requires at least 2 drafts.")

    client = anthropic.Anthropic(api_key=api_key)

    draft_blocks = []
    for index, (_run_id, text) in enumerate(drafts, start=1):
        draft_blocks.append(f"DRAFT {index}\n\n{text.strip()}")

    payload = (
        f"{EVALUATOR_PROMPT}\n\n"
        + "\n\n---\n\n".join(draft_blocks)
    )

    # Pre-flight size check. Claude Opus 4.6 has a 200k-token context window.
    # Reserve 8k tokens for the response and per-message overhead.
    # Rough char-to-token ratio for English prose is ~4 chars per token; we use 3.5
    # to be conservative (overestimate tokens so we warn earlier rather than later).
    CONTEXT_WINDOW_TOKENS = 200_000
    RESPONSE_RESERVE_TOKENS = 8_000
    INPUT_BUDGET_TOKENS = CONTEXT_WINDOW_TOKENS - RESPONSE_RESERVE_TOKENS

    estimated_input_tokens = int(len(payload) / 3.5)
    if estimated_input_tokens > INPUT_BUDGET_TOKENS:
        total_words = sum(len(t.split()) for _, t in drafts)
        avg_words = total_words // max(len(drafts), 1)
        raise RuntimeError(
            f"Evaluator input is too large for the model's context window. "
            f"Estimated {estimated_input_tokens:,} input tokens "
            f"(budget is {INPUT_BUDGET_TOKENS:,} after reserving "
            f"{RESPONSE_RESERVE_TOKENS:,} for the response). "
            f"Batch has {len(drafts)} drafts totaling {total_words:,} words "
            f"(avg {avg_words:,} words/draft). "
            f"Evaluate fewer drafts at once or shorten the inputs."
        )

    resp = anthropic_messages_create_with_backoff(
        client,
        model=model.strip(),
        max_tokens=4000,
        temperature=0,
        messages=[{"role": "user", "content": payload}],
        max_attempts=5,
        initial_delay=2.0,
    )

    raw_text = extract_text_from_response(resp)
    winner_index, parse_status = parse_winner_integer(raw_text, max_valid=len(drafts))

    if winner_index is None:
        raise RuntimeError(f"Could not parse a valid draft number from evaluator response: {raw_text!r}")

    winner_run_id = drafts[winner_index - 1][0]

    return {
        "winner_run_id": winner_run_id,
        "winner_index": winner_index,
        "raw_text": raw_text.strip(),
        "parse_status": parse_status,
        "model": model.strip(),
    }


def gather_paths_for_records(df: pd.DataFrame, columns: Iterable[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()
    for col in columns:
        if col not in df.columns:
            continue
        for raw in df[col].dropna().tolist():
            path = Path(str(raw))
            if path.exists() and path.is_file():
                resolved = str(path.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(path)
    return paths


def export_zip(df: pd.DataFrame, file_paths: List[Path]) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.csv", df.to_csv(index=False))
        for file_path in sorted(file_paths, key=lambda p: p.name):
            zf.write(file_path, arcname=file_path.name)
    mem.seek(0)
    return mem.read()


def make_file_stub(session_id: str, batch_label: str, prompt_id: int, repetition_index: int) -> str:
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_batch = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in batch_label.strip()) or "batch"
    return f"{session_id}_{timestamp_str}_{safe_batch}_p{prompt_id:02d}_r{repetition_index:02d}"


def short_model_slug(model: str) -> str:
    if not model:
        return "model"
    name = model.strip().lower()
    family_map = [
        ("opus", "O"),
        ("sonnet", "S"),
        ("haiku", "H"),
    ]
    prefix = None
    for keyword, letter in family_map:
        if keyword in name:
            prefix = letter
            break
    if prefix is None:
        safe = "".join(ch for ch in model if ch.isalnum() or ch in "-._")
        return safe or "model"
    match = re.search(r"(\d+)[-.](\d+)", name)
    if match:
        return f"{prefix}{match.group(1)}.{match.group(2)}"
    match = re.search(r"\d+", name)
    if match:
        return f"{prefix}{match.group(0)}"
    return prefix


def make_winner_filename(prompt_id: int, temperature: float, model: str) -> str:
    temp_str = f"{temperature:.1f}".lstrip("0") if temperature < 1 else f"{temperature:.1f}"
    safe_temp = temp_str if temp_str.startswith(".") or temp_str.startswith("-") else temp_str
    slug = short_model_slug(model)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"P{prompt_id} T{safe_temp} {slug} Winner {timestamp_str}.txt"


def update_records_bulk(csv_path: Path, run_ids: List[str], updates: dict) -> None:
    df = load_records(csv_path)
    if df.empty or not run_ids:
        return
    mask = df["run_id"].astype(str).isin([str(r) for r in run_ids])
    if not mask.any():
        return
    for key, value in updates.items():
        if key in df.columns:
            if isinstance(value, str):
                df[key] = df[key].astype("object")
            df.loc[mask, key] = value
    df.to_csv(csv_path, index=False)


def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def clean_api_key(value: str) -> str:
    if not value:
        return ""
    return re.sub(r"\s+", "", str(value)).strip()


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    session_id = ensure_session_state()
    st.title(APP_TITLE)
    st.caption("Run a controlled micro-prompt experiment against one fixed writing package.")

    csv_path = DATA_DIR / "runs.csv"

    try:
        prompt_defs = load_prompt_definitions(PROMPTS_CSV)
    except Exception as exc:
        st.error(f"Could not load prompt definitions from {PROMPTS_CSV}: {exc}")
        st.stop()

    with st.sidebar:
        st.header("Run setup")
        st.caption(f"Current app session: {session_id}")
        provider = st.selectbox("Provider", ["anthropic"], index=0)
        model = st.text_input("Model", value=DEFAULT_MODEL)

        api_key = ""
        api_key_source = ""

        try:
            if "ANTHROPIC_API_KEY" in st.secrets:
                api_key = clean_api_key(st.secrets["ANTHROPIC_API_KEY"])
                if api_key:
                    api_key_source = "Streamlit secrets"
        except Exception:
            api_key = ""
            api_key_source = ""

        if not api_key:
            api_key = clean_api_key(os.environ.get("ANTHROPIC_API_KEY", ""))
            if api_key:
                api_key_source = "environment variable"

        if not api_key:
            manual_key = st.text_input(
                "Anthropic API key",
                value="",
                type="password",
                help="Used only for this session if not found in Streamlit secrets or the environment.",
            )
            api_key = clean_api_key(manual_key)
            if api_key:
                api_key_source = "manual entry"

        if api_key:
            st.caption(f"API key loaded from {api_key_source}.")
        else:
            st.error(
                "API key not found. Set ANTHROPIC_API_KEY in Streamlit secrets, "
                "the environment, or enter it manually here."
            )

        temperature = st.slider("Temperature", 0.0, 1.5, 1.0, 0.1)
        max_tokens = st.number_input(
            "Max output tokens per API call",
            min_value=500,
            max_value=MAX_ALLOWED_TOKENS,
            value=DEFAULT_MAX_TOKENS,
            step=500,
            help="If the model hits this ceiling, the app will automatically ask it to continue.",
        )
        runs_per_prompt = st.number_input(
            "Runs per prompt",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Repeat each selected prompt up to 10 times in the same batch.",
        )
        batch_label = st.text_input(
            "Batch label",
            value="batch1",
            help="Optional for your own reference. Current-session export does not depend on this.",
        )
        evaluator_model = st.text_input(
            "Evaluator model",
            value=DEFAULT_EVALUATOR_MODEL,
            help="Claude model used by the 'Evaluate latest batch' button to pick the best draft.",
        )

        if max_tokens < 8000:
            st.warning("This token ceiling is on the low side for full chapter generation. The app can continue automatically, but larger per-call limits are safer.")

        st.markdown("---")
        st.subheader("Storage controls")
        show_current_only = st.checkbox("Show current app session runs only", value=True)
        if st.button("Start a fresh app session", help="Creates a new session ID so new exports include only future runs."):
            st.session_state[CURRENT_SESSION_KEY] = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.session_state[CURRENT_SESSION_RUN_IDS_KEY] = []
            st.rerun()
        if st.button("Clear all stored runs and files", type="secondary", help="Deletes runs.csv and every file in micro_prompt_runs/flat_outputs."):
            clear_all_run_data(csv_path, OUTPUTS_DIR)
            st.success("All stored run data was deleted. A new app session has started.")
            st.rerun()

    left, right = st.columns([1.15, 0.85])

    with left:
        st.subheader("Source package")
        base_prompt = st.text_area("Base prompt", value=DEFAULT_BASE_PROMPT, height=220)

        source_text = ""
        outline_text = ""
        profiles_text = ""
        source_name = ""
        outline_name = ""
        profiles_name = ""

        uploaded_source = st.file_uploader("Upload combined source texts (.txt/.md)", type=["txt", "md"], key="src")
        if uploaded_source is not None:
            source_name = uploaded_source.name
            source_text = decode_uploaded_text(uploaded_source)
            st.info(f"Loaded source text: {source_name}")

        uploaded_outline = st.file_uploader("Upload outline (.txt/.md)", type=["txt", "md"], key="out")
        if uploaded_outline is not None:
            outline_name = uploaded_outline.name
            outline_text = decode_uploaded_text(uploaded_outline)
            st.info(f"Loaded outline: {outline_name}")

        uploaded_profiles = st.file_uploader("Upload character profiles (.txt/.md, optional)", type=["txt", "md"], key="prof")
        if uploaded_profiles is not None:
            profiles_name = uploaded_profiles.name
            profiles_text = decode_uploaded_text(uploaded_profiles)
            st.info(f"Loaded profiles: {profiles_name}")

        st.markdown("### Prompt set")
        df_prompts = pd.DataFrame(prompt_defs)
        st.dataframe(df_prompts, use_container_width=True, hide_index=True)

        selected_ids = st.multiselect(
            "Select prompt IDs to run",
            options=[p["id"] for p in prompt_defs],
            default=[1, 2, 6, 10, 14, 16, 19, 21, 28],
        )

        run_selected = st.button("Run selected prompts", type="primary")

        if run_selected:
            if not api_key:
                st.error("Enter an API key.")
            elif not base_prompt.strip():
                st.error("Base prompt cannot be empty.")
            elif not source_text.strip():
                st.error("Combined source texts are required.")
            elif not outline_text.strip():
                st.error("Outline is required.")
            elif not selected_ids:
                st.error("Select at least one prompt ID.")
            else:
                session_id = ensure_session_state()
                selected_prompts = [p for p in prompt_defs if p["id"] in selected_ids]
                st.session_state[LATEST_BATCH_RUN_IDS_KEY] = []
                progress = st.progress(0)
                status = st.empty()
                failures = []
                warnings = []
                successes = 0
                total_runs = len(selected_prompts) * int(runs_per_prompt)
                completed_runs = 0

                for prompt_position, prompt_obj in enumerate(selected_prompts, start=1):
                    payload = build_payload(
                        base_prompt=base_prompt,
                        micro_prompt=prompt_obj["text"],
                        source_text=source_text,
                        outline_text=outline_text,
                        profiles_text=profiles_text,
                    )

                    for repetition_index in range(1, int(runs_per_prompt) + 1):
                        file_stub = make_file_stub(session_id, batch_label, prompt_obj["id"], repetition_index)
                        run_id = file_stub

                        output_slug = short_model_slug(model)
                        output_temp_str = f"{float(temperature):.1f}"
                        output_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        output_filename = (
                            f"P{prompt_obj['id']} T{output_temp_str} {output_slug} "
                            f"R{repetition_index:02d} {output_ts}.txt"
                        )

                        payload_path = OUTPUTS_DIR / f"{file_stub}_payload.txt"
                        output_path = OUTPUTS_DIR / output_filename
                        micro_prompt_path = OUTPUTS_DIR / f"{file_stub}_prompt.txt"
                        meta_path = OUTPUTS_DIR / f"{file_stub}_meta.json"

                        try:
                            status.write(
                                f"Running prompt {prompt_obj['id']} rep {repetition_index}/{int(runs_per_prompt)} "
                                f"(prompt {prompt_position}/{len(selected_prompts)}, overall {completed_runs + 1}/{total_runs})..."
                            )

                            save_text(payload_path, payload)
                            save_text(micro_prompt_path, prompt_obj["text"])

                            generation = call_anthropic_with_continuation(
                                api_key=api_key,
                                model=model,
                                payload=payload,
                                max_tokens=int(max_tokens),
                                temperature=float(temperature),
                            )

                            output_text = generation["text"]
                            save_text(output_path, output_text)

                            output_hash = sha256_text(output_text)

                            meta = {
                                "run_id": run_id,
                                "session_id": session_id,
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "batch_label": batch_label,
                                "prompt_id": prompt_obj["id"],
                                "repetition_index": repetition_index,
                                "category": prompt_obj["category"],
                                "provider": provider,
                                "model": model,
                                "temperature": float(temperature),
                                "max_tokens": int(max_tokens),
                                "continuation_rounds": generation["continuation_rounds"],
                                "source_name": source_name,
                                "outline_name": outline_name,
                                "profiles_name": profiles_name,
                                "file_stub": file_stub,
                                "payload_file": str(payload_path),
                                "output_file": str(output_path),
                                "micro_prompt_file": str(micro_prompt_path),
                                "meta_file": str(meta_path),
                                "output_sha256": output_hash,
                                "stop_reason": generation["stop_reason"],
                                "input_tokens": generation["input_tokens"],
                                "output_tokens": generation["output_tokens"],
                                "output_words": generation["output_words"],
                                "truncation_flag": generation["truncation_flag"],
                            }
                            save_text(meta_path, json.dumps(meta, indent=2))

                            append_record(
                                csv_path,
                                RunRecord(
                                    run_id=run_id,
                                    session_id=session_id,
                                    timestamp=meta["timestamp"],
                                    batch_label=batch_label,
                                    prompt_id=prompt_obj["id"],
                                    repetition_index=repetition_index,
                                    category=prompt_obj["category"],
                                    provider=provider,
                                    model=model,
                                    temperature=float(temperature),
                                    max_tokens=int(max_tokens),
                                    continuation_rounds=int(generation["continuation_rounds"]),
                                    source_name=source_name,
                                    outline_name=outline_name,
                                    profiles_name=profiles_name,
                                    file_stub=file_stub,
                                    output_file=str(output_path),
                                    payload_file=str(payload_path),
                                    micro_prompt_file=str(micro_prompt_path),
                                    meta_file=str(meta_path),
                                    output_sha256=output_hash,
                                    stop_reason=str(generation["stop_reason"]),
                                    input_tokens=generation["input_tokens"],
                                    output_tokens=generation["output_tokens"],
                                    output_words=generation["output_words"],
                                    truncation_flag=bool(generation["truncation_flag"]),
                                ),
                            )

                            st.session_state[CURRENT_SESSION_RUN_IDS_KEY].append(run_id)
                            st.session_state[LATEST_BATCH_RUN_IDS_KEY].append(run_id)
                            successes += 1

                        except Exception as exc:
                            failures.append(
                                f"Prompt {prompt_obj['id']} rep {repetition_index}: {exc}"
                            )

                        completed_runs += 1
                        progress.progress(min(completed_runs / total_runs, 1.0))
                        time.sleep(1.25)

                status.write("Run complete.")

                if successes:
                    st.success(f"Completed {successes} run(s).")
                if warnings:
                    st.warning("\n".join(warnings))
                if failures:
                    st.error("\n".join(failures))

    with right:
        st.subheader("Run log")

        df = load_records(csv_path)
        session_run_ids = list(st.session_state.get(CURRENT_SESSION_RUN_IDS_KEY, []))

        if df.empty:
            st.info("No runs logged yet.")
        else:
            display_df = df.copy()
            if "truncation_flag" in display_df.columns:
                display_df["truncation_flag"] = display_df["truncation_flag"].apply(coerce_bool)

            session_df = display_df[display_df["session_id"].astype(str) == session_id].copy() if "session_id" in display_df.columns else display_df.iloc[0:0].copy()
            if session_run_ids:
                session_df = display_df[display_df["run_id"].astype(str).isin(session_run_ids)].copy()

            table_df = session_df if show_current_only else display_df
            if table_df.empty and show_current_only:
                st.info("No runs yet in the current app session.")
            else:
                st.dataframe(table_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

            selectable_df = table_df if not table_df.empty else display_df
            selected_run = st.selectbox("Select run", selectable_df["run_id"].astype(str).tolist())
            current = df[df["run_id"].astype(str) == str(selected_run)].iloc[0]

            with st.form("score_form"):
                originality_label = st.text_input(
                    "Originality label",
                    value=str(current.get("originality_label", "") or ""),
                )
                originality_score = st.text_input(
                    "Originality score",
                    value="" if pd.isna(current.get("originality_score")) else str(current.get("originality_score")),
                )
                manual_rating = st.selectbox(
                    "Manual rating",
                    ["", "strong", "decent", "weak"],
                    index=["", "strong", "decent", "weak"].index(str(current.get("manual_rating", "") or ""))
                    if str(current.get("manual_rating", "") or "") in ["", "strong", "decent", "weak"]
                    else 0,
                )
                manual_notes = st.text_area(
                    "Manual notes",
                    value=str(current.get("manual_notes", "") or ""),
                    height=120,
                )
                submitted = st.form_submit_button("Save score")
                if submitted:
                    parsed_score = None
                    raw = originality_score.strip()
                    if raw:
                        parsed_score = float(raw)
                    update_record(
                        csv_path,
                        selected_run,
                        {
                            "originality_label": originality_label,
                            "originality_score": parsed_score,
                            "manual_rating": manual_rating,
                            "manual_notes": manual_notes,
                        },
                    )
                    st.success("Saved.")
                    st.rerun()

            for label, col in [("Output", "output_file"), ("Micro-prompt", "micro_prompt_file"), ("Payload", "payload_file")]:
                path_str = str(current.get(col, "") or "")
                if path_str and Path(path_str).exists():
                    st.markdown(f"### {label}")
                    content = Path(path_str).read_text(encoding="utf-8")
                    st.text_area(f"{label} preview", value=content, height=320, key=f"preview_{label}_{selected_run}")

            st.markdown("### Selected run metadata")
            st.json({
                "run_id": str(current.get("run_id", "")),
                "session_id": str(current.get("session_id", "")),
                "batch_label": str(current.get("batch_label", "")),
                "prompt_id": int(current.get("prompt_id", 0)),
                "repetition_index": int(current.get("repetition_index", 0)) if not pd.isna(current.get("repetition_index", 0)) else 0,
                "category": str(current.get("category", "")),
                "file_stub": str(current.get("file_stub", "")),
                "stop_reason": str(current.get("stop_reason", "")),
                "output_words": None if pd.isna(current.get("output_words")) else int(current.get("output_words")),
                "continuation_rounds": None if pd.isna(current.get("continuation_rounds")) else int(current.get("continuation_rounds")),
                "truncation_flag": coerce_bool(current.get("truncation_flag")),
                "is_winner": coerce_bool(current.get("is_winner")),
                "evaluation_id": str(current.get("evaluation_id", "") or ""),
                "evaluator_model": str(current.get("evaluator_model", "") or ""),
                "evaluation_parse_status": str(current.get("evaluation_parse_status", "") or ""),
                "evaluation_raw": str(current.get("evaluation_raw", "") or ""),
                "output_sha256": str(current.get("output_sha256", "")),
            })

            history_file_paths = gather_paths_for_records(display_df, ["output_file", "payload_file", "micro_prompt_file", "meta_file"])
            history_zip_bytes = export_zip(display_df, history_file_paths)

            latest_batch_ids = list(st.session_state.get(LATEST_BATCH_RUN_IDS_KEY, []))
            batch_count = len(latest_batch_ids)
            evaluate_clicked = st.button(
                f"Evaluate latest batch ({batch_count} draft{'s' if batch_count != 1 else ''})",
                disabled=batch_count < 2,
                help="Send all drafts from the most recent 'Run selected prompts' click to the evaluator. Opus picks the strongest on literary grounds.",
            )

            if evaluate_clicked:
                if not api_key:
                    st.error("Enter an API key in the sidebar.")
                else:
                    eval_df = load_records(csv_path)
                    batch_rows = eval_df[eval_df["run_id"].astype(str).isin([str(r) for r in latest_batch_ids])].copy()

                    if len(batch_rows) < 2:
                        st.error("Need at least 2 drafts in the latest batch to evaluate.")
                    else:
                        drafts: List[Tuple[str, str]] = []
                        missing: List[str] = []
                        for _, row in batch_rows.iterrows():
                            run_id_val = str(row["run_id"])
                            output_file_val = str(row.get("output_file", "") or "")
                            if output_file_val and Path(output_file_val).exists():
                                draft_text = Path(output_file_val).read_text(encoding="utf-8")
                                drafts.append((run_id_val, draft_text))
                            else:
                                missing.append(run_id_val)

                        if missing:
                            st.warning(f"Skipping {len(missing)} run(s) with missing output files: {', '.join(missing)}")

                        if len(drafts) < 2:
                            st.error("Fewer than 2 drafts have readable output files. Cannot evaluate.")
                        else:
                            with st.spinner(f"Evaluating {len(drafts)} drafts with {evaluator_model}..."):
                                try:
                                    result = evaluate_drafts_with_anthropic(
                                        api_key=api_key,
                                        model=evaluator_model,
                                        drafts=drafts,
                                    )

                                    winner_run_id = result["winner_run_id"]
                                    winner_row = batch_rows[batch_rows["run_id"].astype(str) == str(winner_run_id)].iloc[0]
                                    winner_prompt_id = int(winner_row["prompt_id"])
                                    winner_temperature = float(winner_row["temperature"])
                                    winner_model = str(winner_row["model"])
                                    winner_output_file = str(winner_row["output_file"])

                                    winner_text = Path(winner_output_file).read_text(encoding="utf-8")
                                    winner_filename = make_winner_filename(
                                        prompt_id=winner_prompt_id,
                                        temperature=winner_temperature,
                                        model=winner_model,
                                    )
                                    winner_path = OUTPUTS_DIR / winner_filename
                                    save_text(winner_path, winner_text)

                                    evaluation_id = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

                                    batch_run_id_list = [str(r) for r in batch_rows["run_id"].astype(str).tolist()]
                                    update_records_bulk(
                                        csv_path,
                                        batch_run_id_list,
                                        {
                                            "is_winner": False,
                                            "evaluation_id": evaluation_id,
                                            "evaluator_model": result["model"],
                                            "evaluation_parse_status": result["parse_status"],
                                            "evaluation_raw": result["raw_text"],
                                        },
                                    )
                                    update_record(
                                        csv_path,
                                        str(winner_run_id),
                                        {"is_winner": True},
                                    )

                                    st.success(
                                        f"Winner: {winner_run_id} (draft {result['winner_index']} of {len(drafts)}). "
                                        f"Parse: {result['parse_status']}. Saved to {winner_filename}."
                                    )
                                    if result["parse_status"] != "clean":
                                        st.info(f"Raw evaluator response: {result['raw_text']!r}")
                                    st.rerun()

                                except Exception as eval_exc:
                                    st.error(f"Evaluation failed: {eval_exc}")

            st.download_button(
                "Download all history",
                data=history_zip_bytes,
                file_name="micro_prompt_runs_export_all.zip",
                mime="application/zip",
            )

            current_session_export_df = session_df if not session_df.empty else display_df.iloc[0:0].copy()
            current_session_file_paths = gather_paths_for_records(current_session_export_df, ["output_file", "payload_file", "micro_prompt_file", "meta_file"])
            current_session_zip_bytes = export_zip(current_session_export_df, current_session_file_paths)
            st.download_button(
                "Download current app session only",
                data=current_session_zip_bytes,
                file_name=f"micro_prompt_runs_session_{session_id}.zip",
                mime="application/zip",
                disabled=current_session_export_df.empty,
            )


if __name__ == "__main__":
    main()