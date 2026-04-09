
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

DEFAULT_BASE_PROMPT = """You are not Claude. You are the author of the combined source texts document.

You wrote every passage in the combined source texts document. The character profiles are your notes. The outline is your plan for this chapter.

Read all attached documents from beginning to end. Do not sample them.

Then write the chapter from the outline exactly as you would write it yourself. Construct each sentence from within the habits of mind, sentence movement, and narrative logic already present in the source texts. Write the chapter straight through in one continuous pass, first sentence to last. Do not draft short and expand. Return plain text only, with no commentary. Use normal prose formatting with paragraph breaks. Separate paragraphs with a blank line. Do not collapse the chapter into a single block of text."""
DEFAULT_MODEL = "claude-sonnet-4-6"
PROMPTS_CSV = Path("prompts.csv")

DEFAULT_MAX_TOKENS = 12000
UI_MAX_TOKENS = 32000
AUTO_RETRY_LIMIT = 2
MIN_RECOMMENDED_TOKENS = 8000


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
    attempts_used: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    output_words: Optional[int] = None
    truncation_flag: bool = False
    truncation_reason: str = ""
    originality_label: str = ""
    originality_score: Optional[float] = None
    manual_rating: str = ""
    manual_notes: str = ""


def normalize_text(text: str) -> str:
    return (
        text.replace("\u2013", "-")
        .replace("\u2014", "--")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
        .replace("\u00a0", " ")
    )


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


def normalize_generated_output(text: str) -> str:
    text = normalize_text(text).replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()


def normalize_anthropic_text(resp) -> str:
    parts: List[str] = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            text = getattr(block, "text", "")
            if text:
                parts.append(text)
    return normalize_generated_output("".join(parts))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def count_words(text: str) -> int:
    return len(text.split())


def parse_boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


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
        "Now write the chapter. Return plain text only, using normal prose paragraphing. Separate paragraphs with a blank line and preserve scene-break spacing if present."
    ])
    return "\n".join(parts)


def call_anthropic_once(
    api_key: str,
    model: str,
    payload: str,
    max_tokens: int,
    temperature: float,
):
    if anthropic is None:
        raise RuntimeError("anthropic package is not installed.")

    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": payload}],
    )
    return resp


def extract_usage(resp) -> Tuple[Optional[int], Optional[int]]:
    usage = getattr(resp, "usage", None)
    if usage is None:
        return None, None
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    return input_tokens, output_tokens


def generate_with_retry(
    api_key: str,
    model: str,
    payload: str,
    requested_max_tokens: int,
    temperature: float,
    retry_limit: int = AUTO_RETRY_LIMIT,
) -> dict:
    attempt = 0
    current_max_tokens = requested_max_tokens
    last_resp = None

    while True:
        attempt += 1
        resp = call_anthropic_once(
            api_key=api_key,
            model=model,
            payload=payload,
            max_tokens=current_max_tokens,
            temperature=temperature,
        )
        last_resp = resp
        output_text = normalize_anthropic_text(resp)
        stop_reason = str(getattr(resp, "stop_reason", "") or "")
        input_tokens, output_tokens = extract_usage(resp)
        output_words = count_words(output_text)

        hit_token_ceiling = stop_reason == "max_tokens"
        should_retry = (
            hit_token_ceiling
            and attempt <= retry_limit
            and current_max_tokens < UI_MAX_TOKENS
        )

        if should_retry:
            current_max_tokens = min(current_max_tokens * 2, UI_MAX_TOKENS)
            continue

        truncation_flag = hit_token_ceiling
        truncation_reason = "Model stopped at max_tokens." if hit_token_ceiling else ""

        return {
            "text": output_text,
            "stop_reason": stop_reason,
            "attempts_used": attempt,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "output_words": output_words,
            "final_max_tokens": current_max_tokens,
            "truncation_flag": truncation_flag,
            "truncation_reason": truncation_reason,
            "response_id": str(getattr(last_resp, "id", "") or ""),
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


def build_meta(
    *,
    run_id: str,
    batch_label: str,
    prompt_obj: dict,
    repetition_index: int,
    provider: str,
    model: str,
    temperature: float,
    requested_max_tokens: int,
    generation_result: dict,
    source_name: str,
    outline_name: str,
    profiles_name: str,
    file_stub: str,
    payload_path: Path,
    micro_prompt_path: Path,
    output_path: Path,
) -> dict:
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "batch_label": batch_label,
        "prompt_id": prompt_obj["id"],
        "repetition_index": repetition_index,
        "category": prompt_obj["category"],
        "provider": provider,
        "model": model,
        "temperature": float(temperature),
        "requested_max_tokens": int(requested_max_tokens),
        "final_max_tokens": int(generation_result["final_max_tokens"]),
        "stop_reason": generation_result["stop_reason"],
        "attempts_used": generation_result["attempts_used"],
        "input_tokens": generation_result["input_tokens"],
        "output_tokens": generation_result["output_tokens"],
        "output_words": generation_result["output_words"],
        "truncation_flag": generation_result["truncation_flag"],
        "truncation_reason": generation_result["truncation_reason"],
        "response_id": generation_result["response_id"],
        "source_name": source_name,
        "outline_name": outline_name,
        "profiles_name": profiles_name,
        "file_stub": file_stub,
        "payload_file": str(payload_path),
        "micro_prompt_file": str(micro_prompt_path),
        "output_file": str(output_path),
        "output_sha256": sha256_text(generation_result["text"]),
        "output_head": generation_result["text"][:200],
        "output_tail": generation_result["text"][-800:],
    }


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
            "Max output tokens",
            min_value=1000,
            max_value=UI_MAX_TOKENS,
            value=DEFAULT_MAX_TOKENS,
            step=500,
            help="Use a generous ceiling for full chapter generation. 12,000 is the new default; the app will auto-retry higher if the model hits the token ceiling.",
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

        if int(max_tokens) < MIN_RECOMMENDED_TOKENS:
            st.warning(
                f"{int(max_tokens)} max tokens is risky for full chapter output. Use at least {MIN_RECOMMENDED_TOKENS}, preferably {DEFAULT_MAX_TOKENS}+."
            )

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

                            generation_result = generate_with_retry(
                                api_key=api_key,
                                model=model,
                                payload=payload,
                                requested_max_tokens=int(max_tokens),
                                temperature=float(temperature),
                            )

                            output_text = generation_result["text"]
                            save_text(output_path, output_text)
                            output_hash = sha256_text(output_text)

                            meta = build_meta(
                                run_id=run_id,
                                batch_label=batch_label,
                                prompt_obj=prompt_obj,
                                repetition_index=repetition_index,
                                provider=provider,
                                model=model,
                                temperature=float(temperature),
                                requested_max_tokens=int(max_tokens),
                                generation_result=generation_result,
                                source_name=source_name,
                                outline_name=outline_name,
                                profiles_name=profiles_name,
                                file_stub=file_stub,
                                payload_path=payload_path,
                                micro_prompt_path=micro_prompt_path,
                                output_path=output_path,
                            )
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
                                    max_tokens=int(meta["final_max_tokens"]),
                                    source_name=source_name,
                                    outline_name=outline_name,
                                    profiles_name=profiles_name,
                                    file_stub=file_stub,
                                    output_file=str(output_path),
                                    payload_file=str(payload_path),
                                    micro_prompt_file=str(micro_prompt_path),
                                    meta_file=str(meta_path),
                                    output_sha256=output_hash,
                                    stop_reason=meta["stop_reason"],
                                    attempts_used=int(meta["attempts_used"]),
                                    input_tokens=meta["input_tokens"],
                                    output_tokens=meta["output_tokens"],
                                    output_words=int(meta["output_words"]) if meta["output_words"] is not None else None,
                                    truncation_flag=bool(meta["truncation_flag"]),
                                    truncation_reason=str(meta["truncation_reason"]),
                                ),
                            )
                            successes += 1

                            if generation_result["truncation_flag"]:
                                warnings.append(
                                    f"Prompt {prompt_obj['id']} rep {repetition_index}: {generation_result['truncation_reason']}"
                                )

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
            sort_cols = [c for c in ["timestamp", "prompt_id", "repetition_index"] if c in df.columns]
            st.dataframe(df.sort_values(sort_cols, ascending=False), use_container_width=True, hide_index=True)

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

            if str(current.get("stop_reason", "") or "") == "max_tokens":
                st.error("Run hit the token ceiling and should be treated as truncated.")

            metadata_items = {
                "run_id": str(current.get("run_id", "")),
                "batch_label": str(current.get("batch_label", "")),
                "prompt_id": int(current.get("prompt_id", 0)),
                "repetition_index": int(current.get("repetition_index", 0)) if not pd.isna(current.get("repetition_index", 0)) else 0,
                "category": str(current.get("category", "")),
                "file_stub": str(current.get("file_stub", "")),
                "output_sha256": str(current.get("output_sha256", "")),
                "stop_reason": str(current.get("stop_reason", "")),
                "attempts_used": int(current.get("attempts_used", 0)) if not pd.isna(current.get("attempts_used", 0)) else 0,
                "input_tokens": None if pd.isna(current.get("input_tokens")) else int(current.get("input_tokens")),
                "output_tokens": None if pd.isna(current.get("output_tokens")) else int(current.get("output_tokens")),
                "output_words": None if pd.isna(current.get("output_words")) else int(current.get("output_words")),
                "truncation_flag": parse_boolish(current.get("truncation_flag", False)),
                "truncation_reason": str(current.get("truncation_reason", "")),
            }

            for label, col in [("Output", "output_file"), ("Micro-prompt", "micro_prompt_file"), ("Payload", "payload_file")]:
                path_str = str(current.get(col, "") or "")
                if path_str and Path(path_str).exists():
                    st.markdown(f"### {label}")
                    content = Path(path_str).read_text(encoding="utf-8")
                    st.caption(f"{len(content):,} characters | {count_words(content):,} words")
                    st.text_area(f"{label} preview", value=content[:12000], height=420)

            st.markdown("### Selected run metadata")
            st.json(metadata_items)

            zip_bytes = export_zip(df, OUTPUTS_DIR)
            st.download_button(
                "Download outputs + CSV",
                data=zip_bytes,
                file_name="micro_prompt_runs_export.zip",
                mime="application/zip",
            )


if __name__ == "__main__":
    main()
