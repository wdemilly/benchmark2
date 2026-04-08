import io
import json
import time
import zipfile
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
import streamlit as st

try:
    import anthropic  # type: ignore
except Exception:
    anthropic = None

APP_TITLE = "Micro-Prompt Harness"
DATA_DIR = Path("micro_prompt_runs")
DATA_DIR.mkdir(exist_ok=True)

# Flat directory for all files, per your request
OUTPUTS_DIR = DATA_DIR / "flat_outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

DEFAULT_BASE_PROMPT = """You are not Claude. You are the author of the combined source texts document.

You wrote every passage in the combined source texts document. The character profiles are your notes. The outline is your plan for this chapter.

Read all attached documents from beginning to end. Do not sample them.

Then write the chapter from the outline exactly as you would write it yourself. Construct each sentence from within the habits of mind, sentence movement, and narrative logic already present in the source texts. Write the chapter straight through in one continuous pass, first sentence to last. Do not draft short and expand. Return plain text only, with no commentary."""
DEFAULT_MODEL = "claude-sonnet-4-6"

PROMPTS = [
    {"id": 1, "category": "identity_lock", "text": "You are the author of the combined source texts document. You wrote every passage in it."},
    {"id": 2, "category": "identity_lock", "text": "You are not Claude. You are the author of the combined source texts document."},
    {"id": 3, "category": "identity_lock", "text": "Write this chapter as the author of the combined source texts document would write it."},
    {"id": 4, "category": "identity_lock", "text": "You wrote the combined source texts document. Continue that writing from the outline."},
    {"id": 5, "category": "identity_lock", "text": "Treat the combined source texts document as your own prior work and the outline as your plan for the next chapter."},
    {"id": 6, "category": "sentence_construction", "text": "Construct each sentence in the way the writer would. Before producing it, check whether it matches the sentence movement and prose logic found in the combined text document."},
    {"id": 7, "category": "sentence_construction", "text": "Construct each sentence in the way the writer would. Make each sentence answer to the habits of movement, emphasis, and phrasing already present in the combined text document."},
    {"id": 8, "category": "sentence_construction", "text": "Construct each sentence in the way the writer would, using the same kind of sentence movement and unfolding of thought found in the combined text document."},
    {"id": 9, "category": "sentence_construction", "text": "Construct each sentence in the way the writer would. Do not settle for approximate tone; match the writer's actual sentence habits from the combined text document."},
    {"id": 10, "category": "sentence_construction", "text": "Construct each sentence in the way the writer would. Let each sentence follow the writer's prose logic, not a generalized imitation of style."},
    {"id": 11, "category": "sentence_construction", "text": "Construct each sentence in the way the writer would. Before producing it, check whether its emphasis, pacing, and shape belong to the patterns already present in the combined text document."},
    {"id": 12, "category": "sentence_construction", "text": "Construct each sentence in the way the writer would. Keep each sentence inside the writer's own habits of phrasing, pressure, and release as they appear in the combined text document."},
    {"id": 13, "category": "sentence_construction", "text": "Construct each sentence in the way the writer would. Match not just the tone, but the writer's actual way of building thought from one phrase to the next."},
    {"id": 14, "category": "sentence_construction", "text": "Construct each sentence in the way the writer would. Before producing it, check whether it could sit naturally among the sentences in the combined text document without drawing attention to itself."},
    {"id": 15, "category": "sentence_construction", "text": "Construct each sentence in the way the writer would. Build each sentence from the writer's own pattern of emphasis and progression, as shown in the combined text document."},
    {"id": 16, "category": "anti_explanatory", "text": "Do not shift into explanatory prose."},
    {"id": 17, "category": "anti_explanatory", "text": "Do not summarize the meaning of events."},
    {"id": 18, "category": "anti_explanatory", "text": "Do not generalize beyond the immediate scene."},
    {"id": 19, "category": "anti_explanatory", "text": "Do not produce thematic or interpretive closure."},
    {"id": 20, "category": "anti_explanatory", "text": "Do not smooth transitions for elegance if the writer's own prose would not do so."},
    {"id": 21, "category": "process_control", "text": "Write the chapter in one continuous pass from first sentence to last."},
    {"id": 22, "category": "process_control", "text": "Do not draft short and expand."},
    {"id": 23, "category": "process_control", "text": "Move from beat to beat without summarizing between them."},
    {"id": 24, "category": "process_control", "text": "Do not pause to explain what the scene means; keep writing the scene itself."},
    {"id": 25, "category": "process_control", "text": "Write straight through from the opening sentence to the final sentence without stepping outside the chapter."},
    {"id": 26, "category": "positive_grounding", "text": "Let handling, interruption, and task carry the scene."},
    {"id": 27, "category": "positive_grounding", "text": "Keep the prose inside practical observation and response."},
    {"id": 28, "category": "positive_grounding", "text": "Let action, speech, and local thought carry meaning without explanation."},
    {"id": 29, "category": "positive_grounding", "text": "Favor local physical business over interpretive phrasing."},
    {"id": 30, "category": "positive_grounding", "text": "Keep meaning embedded in action, speech, and routine rather than stated directly."},
]


@dataclass
class RunRecord:
    run_id: str
    timestamp: str
    batch_label: str
    prompt_id: int
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


def normalize_anthropic_text(resp) -> str:
    parts: List[str] = []
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "text":
            parts.append(block.text)
    return "\n".join(parts).strip()


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
        "Now write the chapter. Return plain text only."
    ])
    return "\n".join(parts)


def call_anthropic(api_key: str, model: str, payload: str, max_tokens: int, temperature: float) -> str:
    if anthropic is None:
        raise RuntimeError("anthropic package is not installed.")
    client = anthropic.Anthropic(api_key=api_key)
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": payload}],
    )
    return normalize_anthropic_text(resp)


def export_zip(df: pd.DataFrame, outputs_root: Path) -> bytes:
    mem = io.BytesIO()
    with zipfile.ZipFile(mem, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("results.csv", df.to_csv(index=False))
        for file_path in sorted(outputs_root.glob("*")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)
    mem.seek(0)
    return mem.read()


def make_file_stub(batch_label: str, prompt_id: int, run_number: int) -> str:
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_batch = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in batch_label.strip())
    return f"{timestamp_str}_{safe_batch}_p{prompt_id:02d}_r{run_number:02d}"


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.caption("Run a controlled micro-prompt experiment against one fixed writing package.")

    csv_path = DATA_DIR / "runs.csv"

    with st.sidebar:
        st.header("Run setup")
        provider = st.selectbox("Provider", ["anthropic"], index=0)
        model = st.text_input("Model", value=DEFAULT_MODEL)
        api_key = st.text_input("API key", value="", type="password")
        temperature = st.slider("Temperature", 0.0, 1.5, 1.0, 0.1)
        max_tokens = st.number_input("Max output tokens", min_value=200, max_value=20000, value=3000, step=100)
        batch_label = st.text_input("Batch label", value="batch1", help="Required. Used in every filename.")

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
        df_prompts = pd.DataFrame(PROMPTS)
        st.dataframe(df_prompts, use_container_width=True, hide_index=True)

        selected_ids = st.multiselect(
            "Select prompt IDs to run",
            options=[p["id"] for p in PROMPTS],
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
                selected_prompts = [p for p in PROMPTS if p["id"] in selected_ids]
                progress = st.progress(0)
                status = st.empty()
                failures = []
                successes = 0

                for run_number, prompt_obj in enumerate(selected_prompts, start=1):
                    file_stub = make_file_stub(batch_label, prompt_obj["id"], run_number)
                    run_id = file_stub

                    payload = build_payload(
                        base_prompt=base_prompt,
                        micro_prompt=prompt_obj["text"],
                        source_text=source_text,
                        outline_text=outline_text,
                        profiles_text=profiles_text,
                    )

                    payload_path = OUTPUTS_DIR / f"{file_stub}_payload.txt"
                    output_path = OUTPUTS_DIR / f"{file_stub}_output.txt"
                    micro_prompt_path = OUTPUTS_DIR / f"{file_stub}_prompt.txt"
                    meta_path = OUTPUTS_DIR / f"{file_stub}_meta.json"

                    try:
                        status.write(f"Running prompt {prompt_obj['id']} ({run_number} of {len(selected_prompts)})...")
                        save_text(payload_path, payload)
                        save_text(micro_prompt_path, prompt_obj["text"])

                        output_text = call_anthropic(
                            api_key=api_key,
                            model=model,
                            payload=payload,
                            max_tokens=int(max_tokens),
                            temperature=float(temperature),
                        )
                        save_text(output_path, output_text)

                        output_hash = sha256_text(output_text)

                        meta = {
                            "run_id": run_id,
                            "timestamp": datetime.now().isoformat(timespec="seconds"),
                            "batch_label": batch_label,
                            "prompt_id": prompt_obj["id"],
                            "category": prompt_obj["category"],
                            "provider": provider,
                            "model": model,
                            "temperature": float(temperature),
                            "max_tokens": int(max_tokens),
                            "source_name": source_name,
                            "outline_name": outline_name,
                            "profiles_name": profiles_name,
                            "file_stub": file_stub,
                            "payload_file": str(payload_path),
                            "micro_prompt_file": str(micro_prompt_path),
                            "output_file": str(output_path),
                            "output_sha256": output_hash,
                        }
                        save_text(meta_path, json.dumps(meta, indent=2))

                        append_record(
                            csv_path,
                            RunRecord(
                                run_id=run_id,
                                timestamp=meta["timestamp"],
                                batch_label=batch_label,
                                prompt_id=prompt_obj["id"],
                                category=prompt_obj["category"],
                                provider=provider,
                                model=model,
                                temperature=float(temperature),
                                max_tokens=int(max_tokens),
                                source_name=source_name,
                                outline_name=outline_name,
                                profiles_name=profiles_name,
                                file_stub=file_stub,
                                output_file=str(output_path),
                                payload_file=str(payload_path),
                                micro_prompt_file=str(micro_prompt_path),
                                meta_file=str(meta_path),
                                output_sha256=output_hash,
                            ),
                        )
                        successes += 1

                    except Exception as exc:
                        failures.append(f"Prompt {prompt_obj['id']}: {exc}")

                    progress.progress(run_number / len(selected_prompts))
                    time.sleep(0.1)

                if successes:
                    st.success(f"Completed {successes} run(s). Files written to: {OUTPUTS_DIR}")
                if failures:
                    st.error("\n".join(failures))

    with right:
        st.subheader("Run log")
        df = load_records(csv_path)
        if df.empty:
            st.info("No runs logged yet.")
        else:
            st.dataframe(df.sort_values("timestamp", ascending=False), use_container_width=True, hide_index=True)

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
                    st.code(content[:5000], language="text")

            st.markdown("### Selected run metadata")
            st.json({
                "run_id": str(current.get("run_id", "")),
                "batch_label": str(current.get("batch_label", "")),
                "prompt_id": int(current.get("prompt_id", 0)),
                "category": str(current.get("category", "")),
                "file_stub": str(current.get("file_stub", "")),
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