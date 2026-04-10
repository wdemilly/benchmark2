import io
import json
import time
import zipfile
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

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

CHAPTER_COMPLETE_MARKER = "<<CHAPTER_COMPLETE>>"
DEFAULT_BASE_PROMPT = f"""You are not Claude. You are the author of the combined source texts document.

You wrote every passage in the combined source texts document. The character profiles are your notes. The outline is your plan for this chapter.

Read all attached documents from beginning to end. Do not sample them.

Then write the chapter from the outline exactly as you would write it yourself. Construct each sentence from within the habits of mind, sentence movement, and narrative logic already present in the source texts. Write the chapter straight through in one continuous pass, first sentence to last. Do not draft short and expand. Return plain text only, with normal prose paragraph breaks and no commentary.

Important completion rule: after you have fully completed the chapter and reached the final word required by the outline, output this marker on its own final line:
{CHAPTER_COMPLETE_MARKER}

Do not output that marker early. Do not omit it once the chapter is complete."""
DEFAULT_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 64000
MAX_ALLOWED_TOKENS = 64000
MAX_CONTINUATIONS = 6
PROMPTS_CSV = Path("prompts.csv")


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


@dataclass
class RunRecord:
    run_id: str
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
    completion_marker_found: bool = False
    truncation_flag: bool = False
    truncation_reason: str = ""
    response_ids: str = ""
    originality_label: str = ""
    originality_score: Optional[float] = None
    manual_rating: str = ""
    manual_notes: str = ""


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


def strip_completion_marker(text: str) -> Tuple[str, bool]:
    normalized = normalize_output_text(text)
    if CHAPTER_COMPLETE_MARKER in normalized:
        cleaned = normalized.replace(CHAPTER_COMPLETE_MARKER, "")
        cleaned = normalize_output_text(cleaned)
        return cleaned, True
    return normalized, False


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
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return pd.DataFrame(columns=[field for field in RunRecord.__dataclass_fields__.keys()])


def append_record(csv_path: Path, record: RunRecord) -> None:
    df = load_records(csv_path)
    df = pd.concat([df, pd.DataFrame([asdict(record)])], ignore_index=True)
    df.to_csv(csv_path, index=False)


def update_record(csv_path: Path, run_id: str, updates: dict) -> None:
    df = load_records(csv_path)
    if df.empty:
        return
    mask = df["run_id"] == run_id
    if not mask.any():
        return
    for key, value in updates.items():
        if key in df.columns:
            df.loc[mask, key] = value
    df.to_csv(csv_path, index=False)


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
        "Write the full chapter now.",
        f"When the chapter is truly complete, output {CHAPTER_COMPLETE_MARKER} on its own final line.",
        "Return plain text only, with normal paragraph breaks and no commentary.",
    ])
    return "\n".join(parts)


def get_usage_tokens(resp) -> Tuple[Optional[int], Optional[int]]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None, None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    return input_tokens, output_tokens


def stream_anthropic_message(client, model: str, messages: List[dict], max_tokens: int, temperature: float):
    with client.messages.stream(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=messages,
    ) as stream:
        final_message = stream.get_final_message()
    return final_message


def call_anthropic_until_complete(
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

    messages: List[dict] = [{"role": "user", "content": payload}]
    collected_parts: List[str] = []
    response_ids: List[str] = []
    total_input_tokens = 0
    total_output_tokens = 0
    continuation_rounds = 0
    final_stop_reason = ""
    completion_marker_found = False
    truncation_reason = ""

    for round_index in range(max_continuations + 1):
        resp = stream_anthropic_message(
            client=client,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        text_chunk = extract_text_from_response(resp)
        if not text_chunk.strip():
            raise RuntimeError("Model returned an empty text block.")

        normalized_chunk = normalize_output_text(text_chunk)
        cleaned_chunk, found_marker_in_chunk = strip_completion_marker(normalized_chunk)
        collected_parts.append(cleaned_chunk)

        input_tokens, output_tokens = get_usage_tokens(resp)
        if input_tokens is not None:
            total_input_tokens += int(input_tokens)
        if output_tokens is not None:
            total_output_tokens += int(output_tokens)

        final_stop_reason = str(getattr(resp, "stop_reason", "") or "")
        response_ids.append(str(getattr(resp, "id", "") or ""))

        if found_marker_in_chunk:
            completion_marker_found = True
            truncation_reason = ""
            break

        if round_index >= max_continuations:
            truncation_reason = "completion_marker_missing_after_continuations"
            break

        continuation_rounds += 1
        messages.append({"role": "assistant", "content": normalized_chunk})
        messages.append(
            {
                "role": "user",
                "content": (
                    "Continue exactly where you left off. Do not restart. Do not summarize. "
                    "Do not add commentary. Continue the chapter in plain text only. "
                    f"When and only when the chapter is fully complete, end by placing {CHAPTER_COMPLETE_MARKER} "
                    "on its own final line."
                ),
            }
        )

    combined_text = normalize_output_text("".join(collected_parts))
    truncation_flag = not completion_marker_found
    output_words = len(combined_text.split())

    if truncation_flag and not truncation_reason:
        truncation_reason = "completion_marker_missing"

    return {
        "text": combined_text,
        "stop_reason": final_stop_reason,
        "input_tokens": total_input_tokens or None,
        "output_tokens": total_output_tokens or None,
        "output_words": output_words,
        "completion_marker_found": completion_marker_found,
        "truncation_flag": truncation_flag,
        "truncation_reason": truncation_reason,
        "continuation_rounds": continuation_rounds,
        "response_ids": response_ids,
    }


def export_zip(df: pd.DataFrame, outputs_root: Path) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.csv", df.to_csv(index=False))
        for file_path in sorted(outputs_root.glob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)
    mem.seek(0)
    return mem.read()


def make_file_stub(batch_label: str, prompt_id: int, repetition_index: int) -> str:
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_batch = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in batch_label.strip())
    return f"{timestamp_str}_{safe_batch}_p{prompt_id:02d}_r{repetition_index:02d}"


def coerce_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
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
        provider = st.selectbox("Provider", ["anthropic"], index=0)
        model = st.text_input("Model", value=DEFAULT_MODEL)
        api_key = st.text_input("API key", value="", type="password")
        temperature = st.slider("Temperature", 0.0, 1.5, 1.0, 0.1)
        max_tokens = st.number_input(
            "Max output tokens per API call",
            min_value=1000,
            max_value=MAX_ALLOWED_TOKENS,
            value=DEFAULT_MAX_TOKENS,
            step=1000,
            help="The app will continue across turns until the completion marker appears or the continuation budget is exhausted.",
        )
        runs_per_prompt = st.number_input(
            "Runs per prompt",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Repeat each selected prompt up to 10 times in the same batch.",
        )
        batch_label = st.text_input("Batch label", value="batch1", help="Required. Used in every filename.")

    left, right = st.columns([1.15, 0.85])

    with left:
        st.subheader("Source package")
        base_prompt = st.text_area("Base prompt", value=DEFAULT_BASE_PROMPT, height=260)

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
            elif not batch_label.strip():
                st.error("Batch label is required.")
            elif not base_prompt.strip():
                st.error("Base prompt cannot be empty.")
            elif not source_text.strip():
                st.error("Combined source texts are required.")
            elif not outline_text.strip():
                st.error("Outline is required.")
            elif not selected_ids:
                st.error("Select at least one prompt ID.")
            else:
                selected_prompts = [p for p in prompt_defs if p["id"] in selected_ids]
                progress = st.progress(0)
                status = st.empty()
                failures: List[str] = []
                warnings: List[str] = []
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
                        file_stub = make_file_stub(batch_label, prompt_obj["id"], repetition_index)
                        run_id = file_stub

                        payload_path = OUTPUTS_DIR / f"{file_stub}_payload.txt"
                        output_path = OUTPUTS_DIR / f"{file_stub}_output.txt"
                        micro_prompt_path = OUTPUTS_DIR / f"{file_stub}_prompt.txt"
                        meta_path = OUTPUTS_DIR / f"{file_stub}_meta.json"

                        try:
                            status.write(
                                f"Running prompt {prompt_obj['id']} rep {repetition_index}/{int(runs_per_prompt)} "
                                f"(prompt {prompt_position}/{len(selected_prompts)}, overall {completed_runs + 1}/{total_runs})..."
                            )
                            save_text(payload_path, payload)
                            save_text(micro_prompt_path, prompt_obj["text"])

                            generation = call_anthropic_until_complete(
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
                                "timestamp": datetime.now().isoformat(timespec="seconds"),
                                "batch_label": batch_label,
                                "prompt_id": prompt_obj["id"],
                                "repetition_index": repetition_index,
                                "category": prompt_obj["category"],
                                "provider": provider,
                                "model": model,
                                "temperature": float(temperature),
                                "max_tokens": int(max_tokens),
                                "continuation_rounds": int(generation["continuation_rounds"]),
                                "source_name": source_name,
                                "outline_name": outline_name,
                                "profiles_name": profiles_name,
                                "file_stub": file_stub,
                                "payload_file": str(payload_path),
                                "micro_prompt_file": str(micro_prompt_path),
                                "output_file": str(output_path),
                                "output_sha256": output_hash,
                                "stop_reason": generation["stop_reason"],
                                "input_tokens": generation["input_tokens"],
                                "output_tokens": generation["output_tokens"],
                                "output_words": generation["output_words"],
                                "completion_marker_found": generation["completion_marker_found"],
                                "truncation_flag": generation["truncation_flag"],
                                "truncation_reason": generation["truncation_reason"],
                                "response_ids": generation["response_ids"],
                            }
                            save_text(meta_path, json.dumps(meta, indent=2))

                            append_record(
                                csv_path,
                                RunRecord(
                                    run_id=run_id,
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
                                    completion_marker_found=bool(generation["completion_marker_found"]),
                                    truncation_flag=bool(generation["truncation_flag"]),
                                    truncation_reason=str(generation["truncation_reason"]),
                                    response_ids=json.dumps(generation["response_ids"]),
                                ),
                            )

                            if generation["truncation_flag"]:
                                warnings.append(
                                    f"Prompt {prompt_obj['id']} rep {repetition_index}: completion marker not found after {generation['continuation_rounds']} continuation round(s)."
                                )

                            successes += 1

                        except Exception as exc:
                            failures.append(f"Prompt {prompt_obj['id']} rep {repetition_index}: {exc}")

                        completed_runs += 1
                        progress.progress(completed_runs / total_runs)
                        time.sleep(0.1)

                if successes:
                    st.success(f"Completed {successes} run(s). Files written to: {OUTPUTS_DIR}")
                if warnings:
                    st.warning("\n".join(warnings))
                if failures:
                    st.error("\n".join(failures))

    with right:
        st.subheader("Run log")
        df = load_records(csv_path)
        if df.empty:
            st.info("No runs logged yet.")
        else:
            display_df = df.copy()
            for bool_col in ["completion_marker_found", "truncation_flag"]:
                if bool_col in display_df.columns:
                    display_df[bool_col] = display_df[bool_col].apply(coerce_bool)
            st.dataframe(display_df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

            selected_run = st.selectbox("Select run", df["run_id"].tolist())
            current = df[df["run_id"] == selected_run].iloc[0]

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
                "batch_label": str(current.get("batch_label", "")),
                "prompt_id": int(current.get("prompt_id", 0)),
                "repetition_index": int(current.get("repetition_index", 0)) if not pd.isna(current.get("repetition_index", 0)) else 0,
                "category": str(current.get("category", "")),
                "file_stub": str(current.get("file_stub", "")),
                "stop_reason": str(current.get("stop_reason", "")),
                "output_words": None if pd.isna(current.get("output_words")) else int(current.get("output_words")),
                "continuation_rounds": None if pd.isna(current.get("continuation_rounds")) else int(current.get("continuation_rounds")),
                "completion_marker_found": coerce_bool(current.get("completion_marker_found")),
                "truncation_flag": coerce_bool(current.get("truncation_flag")),
                "truncation_reason": str(current.get("truncation_reason", "") or ""),
                "output_sha256": str(current.get("output_sha256", "")),
            })

            zip_bytes = export_zip(df, OUTPUTS_DIR)
            st.download_button(
                "Download outputs + CSV",
                data=zip_bytes,
                file_name="micro_prompt_runs_export.zip",
                mime="application/zip",
            )


if __name__ == "__main__":
    main()
