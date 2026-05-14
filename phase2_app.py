"""
phase2_app.py — Streamlit wrapper for the Phase 2 orchestrator
==============================================================

Provides a browser UI for running orchestrator.run_pipeline() the same
way simpleapp does for the Phase 1 flow. Drop this file next to
orchestrator.py, originality_api.py, stage_g_interface.py, and
reader_prompt_v1.txt, then run:

    streamlit run phase2_app.py

The app does not modify or import simpleapp_v29. It is a sibling tool.
"""

from __future__ import annotations

import io
import json
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

import streamlit as st

import orchestrator


# ============================================================================
# Page config
# ============================================================================

st.set_page_config(
    page_title="Phase 2 pipeline",
    layout="wide",
)
st.title("Phase 2 — multi-draft loop")
st.caption(
    "Generates N drafts per packet, reads each against the pattern "
    "rubric, scores PASS drafts on Originality.ai, ships the first ≥95."
)


# ============================================================================
# Prerequisite check
# ============================================================================

cwd = Path.cwd()
reader_prompt_path = cwd / "reader_prompt_v1.txt"
if not reader_prompt_path.exists():
    st.error(
        f"reader_prompt_v1.txt not found in current directory "
        f"({cwd}). Make sure all Phase 2 files are in the same "
        f"folder as this app."
    )
    st.stop()


# ============================================================================
# Packet input
# ============================================================================

st.subheader("Chapter packet")

packet_mode = st.radio(
    "Source",
    ["Upload .txt", "Paste text"],
    horizontal=True,
    label_visibility="collapsed",
)

packet_text = None
if packet_mode == "Upload .txt":
    uploaded = st.file_uploader(
        "Packet file", type=["txt"], label_visibility="collapsed",
    )
    if uploaded is not None:
        packet_text = uploaded.read().decode("utf-8")
        st.caption(
            f"Loaded {uploaded.name} — {len(packet_text):,} characters"
        )
else:
    packet_text = st.text_area(
        "Paste packet text",
        height=200,
        label_visibility="collapsed",
        placeholder="Paste the full chapter packet here…",
    )
    if packet_text:
        st.caption(f"{len(packet_text):,} characters")


# ============================================================================
# Configuration
# ============================================================================

st.subheader("Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    chapter_label = st.text_input(
        "Chapter label",
        value="ch01",
        help="Used in draft IDs and output directory name.",
    )
    n = st.number_input(
        "Drafts per iteration (N)",
        min_value=1, max_value=20, value=5,
    )
    drafter_model = st.text_input(
        "Drafter model",
        value=orchestrator.DRAFTER_MODEL_DEFAULT,
    )

with col2:
    temperature = st.slider(
        "Temperature",
        min_value=0.0, max_value=1.5, value=0.8, step=0.05,
    )
    max_iterations = st.number_input(
        "Max iterations",
        min_value=1, max_value=20, value=8,
    )
    reader_model = st.text_input(
        "Reader model",
        value=orchestrator.READER_MODEL_DEFAULT,
    )

with col3:
    ship_score = st.number_input(
        "Ship score (≥)",
        min_value=0, max_value=100, value=95,
    )
    stage_g_low = st.number_input(
        "Stage G band low",
        min_value=0, max_value=100, value=85,
    )
    stage_g_high = st.number_input(
        "Stage G band high",
        min_value=0, max_value=100, value=94,
    )

st.subheader("Scoring backend")
scorer_choice = st.radio(
    "Where do scores come from?",
    ["local", "originality", "both"],
    format_func=lambda s: {
        "local": "Local scorer (free; LOO r=0.944 against your corpus)",
        "originality": "Originality.ai API (paid)",
        "both": "Both — route on Originality, log local for validation",
    }[s],
    horizontal=True,
    label_visibility="collapsed",
)

calibration_path_input = None
if scorer_choice in ("local", "both"):
    default_cal = cwd / "calibration.json"
    calibration_str = st.text_input(
        "Path to calibration.json",
        value=str(default_cal),
        help="Produced by corpus_calibrator.py. If the file does not "
             "exist, the local scorer uses placeholder coefficients "
             "and predictions will be unreliable.",
    )
    calibration_path_input = Path(calibration_str) if calibration_str else None
    if calibration_path_input and calibration_path_input.exists():
        st.success(f"Calibration file found: {calibration_path_input.name}")
    else:
        st.warning(
            "Calibration file not found at that path — local scorer "
            "will use placeholder coefficients. Run corpus_calibrator.py "
            "first if you want real predictions."
        )

with st.expander("Override drafting prompt (optional)"):
    st.caption(
        "By default the embedded DRAFTING_PROMPT in orchestrator.py "
        "is used. To override for this run only, paste a replacement "
        "below. Leave empty to use the embedded prompt."
    )
    drafting_prompt_override = st.text_area(
        "Drafting prompt override",
        height=120,
        label_visibility="collapsed",
        placeholder="(empty = use embedded DRAFTING_PROMPT)",
    )


# ============================================================================
# Run
# ============================================================================

st.markdown("---")
run_clicked = st.button("Run pipeline", type="primary", use_container_width=True)

if run_clicked:
    if not packet_text or not packet_text.strip():
        st.error("No packet provided. Upload or paste a packet first.")
        st.stop()
    if not chapter_label.strip():
        st.error("Chapter label is required.")
        st.stop()

    # Set up output directory.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_label = "".join(
        c for c in chapter_label if c.isalnum() or c in ("_", "-")
    ) or "chapter"
    output_dir = cwd / "phase2_runs" / f"{safe_label}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist packet so the orchestrator can read by path.
    packet_path = output_dir / "packet.txt"
    packet_path.write_text(packet_text, encoding="utf-8")

    # Persist optional drafting prompt override.
    drafting_prompt_path = None
    if drafting_prompt_override and drafting_prompt_override.strip():
        drafting_prompt_path = output_dir / "drafting_prompt_override.txt"
        drafting_prompt_path.write_text(
            drafting_prompt_override, encoding="utf-8",
        )

    config = orchestrator.OrchestratorConfig(
        packet_path=packet_path,
        drafting_prompt_path=drafting_prompt_path,
        reader_prompt_path=reader_prompt_path,
        output_dir=output_dir,
        n=int(n),
        temperature=float(temperature),
        ship_score=float(ship_score),
        stage_g_low=float(stage_g_low),
        stage_g_high=float(stage_g_high),
        max_iterations=int(max_iterations),
        drafter_model=drafter_model,
        reader_model=reader_model,
        chapter_label=safe_label,
        scorer=scorer_choice,
        calibration_path=calibration_path_input,
    )

    with st.status(
        f"Running pipeline for {safe_label}…", expanded=True
    ) as status:
        st.write(f"**Output directory:** `{output_dir}`")
        st.write(
            f"**Expected duration:** roughly "
            f"{n * max_iterations * 0.5:.0f}–"
            f"{n * max_iterations * 2:.0f} minutes "
            f"(N×max_iterations × per-draft latency)."
        )
        st.write(
            "Progress streams to `orchestrator.log` in the output "
            "directory while the pipeline runs."
        )
        start = time.time()
        try:
            result = orchestrator.run_pipeline(config)
        except Exception as e:
            status.update(label="Pipeline failed", state="error")
            st.error(f"Pipeline raised an exception: {e}")
            with st.expander("Traceback"):
                st.code(traceback.format_exc())
            st.stop()
        elapsed = time.time() - start
        status.update(
            label=f"Pipeline complete in {elapsed/60:.1f} minutes",
            state="complete",
        )

    # ========================================================================
    # Results
    # ========================================================================

    st.markdown("---")
    st.subheader("Result")

    shipped = result.get("shipped")
    if shipped:
        score = shipped.get("originality_score")
        st.success(
            f"**SHIPPED** — draft `{shipped['draft_id']}` "
            f"at Originality {score:.1f}, iteration {shipped['iteration']}"
        )
        if shipped.get("parent_draft_id"):
            st.caption(
                f"Repaired from parent draft "
                f"`{shipped['parent_draft_id']}` via Stage G."
            )
        shipped_path = Path(shipped["text_path"])
        if shipped_path.exists():
            shipped_text = shipped_path.read_text(encoding="utf-8")
            st.download_button(
                "Download shipped chapter",
                data=shipped_text,
                file_name=f"{shipped['draft_id']}.txt",
                mime="text/plain",
                type="primary",
            )
            with st.expander("Preview shipped chapter (first 3,000 chars)"):
                preview = shipped_text[:3000]
                if len(shipped_text) > 3000:
                    preview += "\n\n[... truncated ...]"
                st.text(preview)
    else:
        st.warning(
            f"**MANUAL QUEUE** — no draft cleared the ship score of "
            f"{ship_score} in {result['iterations_used']} iterations. "
            f"See `result.json` and individual reader reports for "
            f"diagnosis."
        )

    # Drafts summary.
    st.subheader("All drafts")
    drafts_summary = result.get("drafts_summary", [])
    if drafts_summary:
        st.dataframe(drafts_summary, use_container_width=True)
    else:
        st.caption("No drafts in summary.")

    # Full result JSON.
    with st.expander("Full result.json"):
        st.json(result)

    # Orchestrator log preview.
    log_path = output_dir / "orchestrator.log"
    if log_path.exists():
        with st.expander("orchestrator.log"):
            st.code(log_path.read_text(encoding="utf-8"))

    # ZIP everything for download.
    st.subheader("All artifacts")
    st.caption(
        "ZIP contains drafts/, reader_reports/, originality/, "
        "stage_g/, log.jsonl, orchestrator.log, and result.json."
    )

    zip_buf = io.BytesIO()
    with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in output_dir.rglob("*"):
            if f.is_file():
                zf.write(f, f.relative_to(output_dir.parent))
    zip_buf.seek(0)
    st.download_button(
        f"Download {safe_label}_{timestamp}.zip",
        data=zip_buf.getvalue(),
        file_name=f"{safe_label}_{timestamp}.zip",
        mime="application/zip",
    )


# ============================================================================
# Sidebar — environment status
# ============================================================================

with st.sidebar:
    st.markdown("### Environment")

    import os
    has_anthropic = bool(
        orchestrator.ANTHROPIC_API_KEY
        or os.environ.get("ANTHROPIC_API_KEY")
    )
    if has_anthropic:
        st.success("ANTHROPIC_API_KEY available")
    else:
        st.error(
            "No Anthropic key — paste into orchestrator.py "
            "or set ANTHROPIC_API_KEY"
        )

    # Only check for Originality.ai key when the selected scorer needs it
    if scorer_choice in ("originality", "both"):
        try:
            import originality_api
            has_originality = bool(
                originality_api.API_KEY
                or os.environ.get("ORIGINALITY_API_KEY")
            )
        except Exception:
            has_originality = False
        if has_originality:
            st.success("Originality.ai key available")
        else:
            st.error(
                "No Originality.ai key — required for the selected "
                "scorer. Paste into originality_api.py or set "
                "ORIGINALITY_API_KEY."
            )

    st.markdown("### Files in working dir")
    required = (
        "orchestrator.py",
        "local_scorer.py",
        "extended_band_features.py",
        "band_classifier.py",
        "reader_prompt_v1.txt",
    )
    for fname in required:
        if (cwd / fname).exists():
            st.success(fname)
        else:
            st.error(f"{fname} missing")
    # originality_api.py is optional — only needed if you pick that scorer
    if (cwd / "originality_api.py").exists():
        st.caption("originality_api.py present (optional)")

    st.markdown("### Recent runs")
    runs_dir = cwd / "phase2_runs"
    if runs_dir.exists():
        runs = sorted(
            [p for p in runs_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:10]
        for run in runs:
            st.caption(f"`{run.name}`")
    else:
        st.caption("(no runs yet)")
