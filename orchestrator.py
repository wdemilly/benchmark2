"""
orchestrator.py — Phase 2 pipeline
==================================

Runs the full Phase 2 loop on a single chapter packet:

  1. Generate N drafts via the Anthropic API at configurable temperature.
  2. For each draft, invoke a Claude reader instance against the
     consolidated pattern rubric (reader_prompt_v1.txt +
     reader_rubric_v1.txt). Parse the structured report.
  3. Branch by classification:
       PASS       → submit to Originality.ai for turbo score
       REPAIRABLE → route to Stage G with the violation list as
                    repair brief
       REJECT     → discard
  4. Among Originality-scored drafts, branch by score:
       >= 95      → ship
       85-94      → route to Stage G with no specific brief
                    (mechanical residue clean-up)
       < 85       → discard
  5. Stage G output re-enters at step 2 (reader scan, then
     Originality if still PASS).
  6. Cap total iterations per chapter at 8 (configurable). On
     cap exhaustion, route the chapter to a manual queue
     (writes a MANUAL_QUEUE marker file alongside the artifacts).

Notes on integration with simpleapp_v29.py:

  The orchestrator does NOT wrap simpleapp's Streamlit pipeline.
  It calls anthropic.Messages.create() directly, using the same
  drafting prompt that simpleapp loads from prompts.csv. This
  isolates the loop architecture from simpleapp's UI and post-
  draft ranking machinery, which the Phase 2 architecture
  intentionally bypasses.

  The drafter sampling parameters in simpleapp_v29 default to
  temperature 0.7 (dataclass) / 1.0 (UI). The orchestrator's
  TEMPERATURE_DEFAULT below is set to 0.8 — the briefing's named
  midpoint — and is configurable.

Usage:

  python orchestrator.py \\
      --packet path/to/packet.txt \\
      --drafting-prompt path/to/drafting_prompt.txt \\
      --reader-prompt path/to/reader_prompt_v1.txt \\
      --output-dir runs/ch01/ \\
      --n 5 \\
      --temperature 0.8

Outputs land in --output-dir:
  drafts/draft_<idx>.txt
  reader_reports/draft_<idx>.json
  originality/draft_<idx>.json
  stage_g/draft_<idx>.txt
  log.jsonl                  (one line per routing decision)
  result.json                (final winner or MANUAL_QUEUE marker)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

try:
    import anthropic
except ImportError:
    print(
        "ERROR: the anthropic package is required. "
        "Install with: pip install anthropic",
        file=sys.stderr,
    )
    raise

import stage_g_interface
import local_scorer

# ============================================================================
# Configuration defaults — all overridable via CLI
# ============================================================================

N_DEFAULT = 5
TEMPERATURE_DEFAULT = 0.8
SHIP_SCORE_DEFAULT = 95
STAGE_G_BAND_LOW_DEFAULT = 85   # 85-94 inclusive → Stage G
STAGE_G_BAND_HIGH_DEFAULT = 94
DISCARD_BELOW_DEFAULT = 85
MAX_ITERATIONS_DEFAULT = 8

# Reader-side classification thresholds (these are FIXED by the
# Phase 2 design, not configurable — changing them changes the
# architectural contract with the reader prompt).
READER_PASS_MAX = 1        # 0-1 categories → PASS
READER_REPAIRABLE_MAX = 4  # 2-4 categories → REPAIRABLE
                           # 5+ categories → REJECT

DRAFTER_MODEL_DEFAULT = "claude-opus-4-7"
READER_MODEL_DEFAULT = "claude-opus-4-7"
MAX_DRAFTER_TOKENS = 16000
MAX_READER_TOKENS = 8000

# ============================================================================
# Embedded drafting prompt
# ============================================================================
#
# The drafter's system prompt is embedded here rather than loaded from a
# separate file. The orchestrator passes this string to the Anthropic API
# on every draft generation call. To change the drafter's instruction set,
# edit this constant directly OR override at runtime via
# --drafting-prompt path/to/file.txt.

DRAFTING_PROMPT = (
    "Write this chapter from the outline.  Follow the outline's "
    "instructions.  Make sure that you always conform to the GLOBAL "
    "DRAFTING CONTROLS. "
)

# ============================================================================
# >>> PASTE YOUR ANTHROPIC API KEY HERE <<<
# ============================================================================
#
# Replace None with your key as a string, e.g.:
#     ANTHROPIC_API_KEY = "sk-ant-..."
#
# Leaving this as None falls back to the ANTHROPIC_API_KEY environment
# variable (the standard place the anthropic SDK looks). The inline
# constant is checked first.
# ============================================================================

ANTHROPIC_API_KEY = None

# ============================================================================
# Data classes
# ============================================================================

@dataclass
class DraftRecord:
    draft_id: str
    iteration: int        # 1-indexed; iteration 1 is the initial batch
    text: str
    text_path: Optional[Path] = None
    parent_draft_id: Optional[str] = None   # set when this is a Stage G output
    reader_report: Optional[dict] = None
    reader_report_path: Optional[Path] = None
    classification: Optional[str] = None    # "PASS" | "REPAIRABLE" | "REJECT"
    originality_score: Optional[float] = None
    originality_response: Optional[dict] = None
    originality_response_path: Optional[Path] = None
    final_disposition: Optional[str] = None
    # "SHIPPED" | "STAGE_G_QUEUED" | "DISCARDED_BY_READER" |
    # "DISCARDED_BY_SCORE" | "STAGE_G_FAILED"


@dataclass
class OrchestratorConfig:
    packet_path: Path
    drafting_prompt_path: Optional[Path]   # None → use embedded DRAFTING_PROMPT
    reader_prompt_path: Path
    output_dir: Path
    n: int = N_DEFAULT
    temperature: float = TEMPERATURE_DEFAULT
    ship_score: float = SHIP_SCORE_DEFAULT
    stage_g_low: float = STAGE_G_BAND_LOW_DEFAULT
    stage_g_high: float = STAGE_G_BAND_HIGH_DEFAULT
    discard_below: float = DISCARD_BELOW_DEFAULT
    max_iterations: int = MAX_ITERATIONS_DEFAULT
    drafter_model: str = DRAFTER_MODEL_DEFAULT
    reader_model: str = READER_MODEL_DEFAULT
    chapter_label: str = "chapter"
    scorer: str = "originality"   # "originality" | "local" | "both"
    calibration_path: Optional[Path] = None   # used when scorer != "originality"

    def validate(self) -> None:
        if not self.packet_path.exists():
            raise FileNotFoundError(f"Packet not found: {self.packet_path}")
        if self.drafting_prompt_path is not None and not self.drafting_prompt_path.exists():
            raise FileNotFoundError(
                f"Drafting prompt not found: {self.drafting_prompt_path}"
            )
        if not self.reader_prompt_path.exists():
            raise FileNotFoundError(
                f"Reader prompt not found: {self.reader_prompt_path}"
            )
        if self.n < 1:
            raise ValueError("n must be >= 1")
        if not (0.0 <= self.temperature <= 1.5):
            raise ValueError("temperature out of expected range 0.0-1.5")
        if not (0 <= self.discard_below <= self.stage_g_low <=
                self.stage_g_high < self.ship_score <= 100):
            raise ValueError(
                "score thresholds must satisfy "
                "0 <= discard_below <= stage_g_low <= stage_g_high "
                "< ship_score <= 100"
            )
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if self.scorer not in ("originality", "local", "both"):
            raise ValueError(
                "scorer must be one of: originality, local, both"
            )


# ============================================================================
# Logging
# ============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "orchestrator.log"
    logger = logging.getLogger("phase2.orchestrator")
    logger.setLevel(logging.INFO)
    # Clear any prior handlers (re-runs in a single Python process).
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s"
    ))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def jsonl_append(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")


# ============================================================================
# Drafter — calls Anthropic API directly with the existing drafting prompt
# ============================================================================

def generate_draft(
    client: anthropic.Anthropic,
    drafting_system_prompt: str,
    packet_text: str,
    model: str,
    temperature: float,
    logger: logging.Logger,
) -> str:
    """
    Single draft generation call. Mirrors simpleapp_v29.generate_chapter()
    but isolated from the Streamlit pipeline. Returns the draft text.
    """
    response = client.messages.create(
        model=model,
        max_tokens=MAX_DRAFTER_TOKENS,
        temperature=temperature,
        system=drafting_system_prompt,
        messages=[
            {
                "role": "user",
                "content": packet_text,
            }
        ],
    )
    # Concatenate text blocks. Tool-use blocks are not expected here
    # because the drafter does not use tools.
    chunks = [
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    ]
    text = "".join(chunks).strip()
    logger.info(
        "drafter call complete: model=%s temp=%.2f output_chars=%d",
        model, temperature, len(text),
    )
    return text


# ============================================================================
# Reader — calls Anthropic API with reader_prompt_v1.txt as system prompt
# ============================================================================

def read_draft(
    client: anthropic.Anthropic,
    reader_system_prompt: str,
    draft_text: str,
    packet_text: str,
    draft_id: str,
    model: str,
    logger: logging.Logger,
) -> dict:
    """
    Calls a Claude reader instance and parses the JSON report.
    Raises if the reader returns non-JSON output (the reader prompt
    instructs JSON-only output; non-conformance is a reader-side bug
    worth surfacing).
    """
    user_content = (
        f"draft_id: {draft_id}\n\n"
        "=== CHAPTER PACKET ===\n"
        f"{packet_text}\n\n"
        "=== DRAFT TO SCAN ===\n"
        f"{draft_text}\n"
    )

    response = client.messages.create(
        model=model,
        max_tokens=MAX_READER_TOKENS,
        temperature=0.0,   # reader is deterministic; pattern-matching, not generation
        system=reader_system_prompt,
        messages=[
            {"role": "user", "content": user_content}
        ],
    )

    raw = "".join(
        block.text for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()

    # The reader prompt instructs JSON-only output with no fence.
    # Tolerate a stray ```json wrapper if it shows up, but log it.
    cleaned = raw
    if cleaned.startswith("```"):
        logger.warning(
            "reader produced a code fence; stripping. draft_id=%s", draft_id,
        )
        # Drop opening fence line and trailing fence.
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        report = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(
            "reader returned non-JSON output. draft_id=%s error=%s "
            "first_300_chars=%r",
            draft_id, e, raw[:300],
        )
        raise
    return report


def classify_from_report(report: dict) -> str:
    """
    Trusts the reader's own classification block if present and
    sensible; otherwise recomputes from the categories_triggered_count.
    """
    classification_block = report.get("classification", {})
    routing = classification_block.get("routing")
    if routing in ("PASS", "REPAIRABLE", "REJECT"):
        return routing

    # Fallback: recompute from count.
    count = classification_block.get("categories_triggered_count")
    if count is None:
        # Last-resort fallback: count categories with count >= 1 across
        # sections A and B.
        count = 0
        for section_key in ("section_A_regex_detectable",
                            "section_B_reading_required"):
            section = report.get(section_key, {})
            for cat_block in section.values():
                if isinstance(cat_block, dict) and cat_block.get("count", 0) >= 1:
                    count += 1

    if count <= READER_PASS_MAX:
        return "PASS"
    if count <= READER_REPAIRABLE_MAX:
        return "REPAIRABLE"
    return "REJECT"


def score_draft(
    text: str,
    config: "OrchestratorConfig",
    logger: logging.Logger,
) -> tuple[float, dict]:
    """
    Unified scoring entry point. Routes to the configured scorer
    backend and returns (score, response_dict).

    scorer="originality" — call originality_api.score_text
    scorer="local"       — call local_scorer.score_text
    scorer="both"        — call both, use Originality's score for
                           routing, persist local prediction alongside
                           for validation analysis
    """
    if config.scorer == "originality":
        import originality_api
        score, response = originality_api.score_text(text)
        return score, {"backend": "originality", "originality": response}

    if config.scorer == "local":
        score, response = local_scorer.score_text(text)
        return score, {"backend": "local", "local": response}

    # scorer == "both" — call both, route on Originality, log delta.
    import originality_api
    orig_score, orig_response = originality_api.score_text(text)
    try:
        local_score, local_response = local_scorer.score_text(text)
        delta = orig_score - local_score
        logger.info(
            "scorer=both: originality=%.1f local=%.1f delta=%.1f",
            orig_score, local_score, delta,
        )
    except Exception as e:
        logger.warning("local scorer failed in 'both' mode: %s", e)
        local_score, local_response = None, {"error": str(e)}
    return orig_score, {
        "backend": "both",
        "originality": orig_response,
        "local": local_response,
        "local_score": local_score,
    }


def extract_violation_list_for_stage_g(report: dict) -> list[dict]:
    """
    Stage G receives a structured list of named violations as its
    repair brief — one entry per triggered category, with quoted
    instances.
    """
    violations = []
    for section_key in ("section_A_regex_detectable",
                        "section_B_reading_required"):
        section = report.get(section_key, {})
        for cat_name, cat_block in section.items():
            if not isinstance(cat_block, dict):
                continue
            if cat_block.get("count", 0) < 1:
                continue
            violations.append({
                "category": cat_name,
                "count": cat_block["count"],
                "severity": cat_block.get("severity", "unknown"),
                "instances": cat_block.get("instances", []),
            })
    return violations


# ============================================================================
# The pipeline loop
# ============================================================================

def run_pipeline(config: OrchestratorConfig) -> dict:
    config.validate()
    logger = setup_logging(config.output_dir)
    logger.info("=== Phase 2 orchestrator start ===")
    logger.info("chapter=%s n=%d temperature=%.2f max_iterations=%d scorer=%s",
                config.chapter_label, config.n, config.temperature,
                config.max_iterations, config.scorer)

    # Configure local scorer calibration path if local scoring is requested.
    if config.scorer in ("local", "both"):
        if config.calibration_path is not None:
            local_scorer.set_calibration_path(config.calibration_path)
            logger.info("local scorer calibration: %s",
                        config.calibration_path)
        else:
            logger.info("local scorer calibration: using default "
                        "./calibration.json (or placeholder if absent)")

    routing_log = config.output_dir / "log.jsonl"

    # Load inputs once.
    packet_text = config.packet_path.read_text(encoding="utf-8")
    if config.drafting_prompt_path is not None:
        drafting_prompt = config.drafting_prompt_path.read_text(encoding="utf-8")
        logger.info("drafting prompt loaded from override file: %s",
                    config.drafting_prompt_path)
    else:
        drafting_prompt = DRAFTING_PROMPT
        logger.info("drafting prompt: using embedded DRAFTING_PROMPT constant")
    reader_prompt = config.reader_prompt_path.read_text(encoding="utf-8")

    # Set up subdirectories.
    drafts_dir = config.output_dir / "drafts"
    reports_dir = config.output_dir / "reader_reports"
    originality_dir = config.output_dir / "originality"
    stage_g_dir = config.output_dir / "stage_g"
    for d in (drafts_dir, reports_dir, originality_dir, stage_g_dir):
        d.mkdir(parents=True, exist_ok=True)

    client = (
        anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        if ANTHROPIC_API_KEY
        else anthropic.Anthropic()
    )  # picks up ANTHROPIC_API_KEY env var if no inline key set

    all_drafts: list[DraftRecord] = []
    iteration = 0
    shipped: Optional[DraftRecord] = None

    while iteration < config.max_iterations and shipped is None:
        iteration += 1
        logger.info("--- iteration %d / %d ---",
                    iteration, config.max_iterations)

        # On iteration 1, generate N fresh drafts. On later iterations,
        # only generate a fresh draft if no Stage G work was queued
        # in the prior iteration. (Stage G output re-enters at step 2
        # in the next iteration via the dedicated Stage G branch below.)
        #
        # For simplicity in v1 of the orchestrator: each iteration
        # generates a fresh batch of N. Stage G is invoked inline within
        # the same iteration on any REPAIRABLE or score-band-G drafts.
        # The iteration counter caps total drafter-call rounds, not
        # total Stage G calls.
        new_drafts = []
        for i in range(config.n):
            draft_id = f"{config.chapter_label}_iter{iteration}_d{i+1}_{uuid.uuid4().hex[:6]}"
            try:
                text = generate_draft(
                    client, drafting_prompt, packet_text,
                    config.drafter_model, config.temperature, logger,
                )
            except Exception as e:
                logger.error("drafter call failed: %s", e)
                jsonl_append(routing_log, {
                    "iteration": iteration,
                    "draft_id": draft_id,
                    "event": "drafter_error",
                    "error": str(e),
                })
                continue

            text_path = drafts_dir / f"{draft_id}.txt"
            text_path.write_text(text, encoding="utf-8")
            record = DraftRecord(
                draft_id=draft_id,
                iteration=iteration,
                text=text,
                text_path=text_path,
            )
            new_drafts.append(record)
            all_drafts.append(record)

        # Reader pass.
        for record in new_drafts:
            try:
                report = read_draft(
                    client, reader_prompt, record.text, packet_text,
                    record.draft_id, config.reader_model, logger,
                )
            except Exception as e:
                logger.error("reader call failed for %s: %s",
                             record.draft_id, e)
                record.final_disposition = "READER_ERROR"
                jsonl_append(routing_log, {
                    "iteration": iteration,
                    "draft_id": record.draft_id,
                    "event": "reader_error",
                    "error": str(e),
                })
                continue

            record.reader_report = report
            report_path = reports_dir / f"{record.draft_id}.json"
            report_path.write_text(
                json.dumps(report, indent=2), encoding="utf-8",
            )
            record.reader_report_path = report_path
            record.classification = classify_from_report(report)

            jsonl_append(routing_log, {
                "iteration": iteration,
                "draft_id": record.draft_id,
                "event": "reader_classified",
                "classification": record.classification,
                "categories_triggered": report.get("classification", {}).get(
                    "categories_triggered_names", []
                ),
            })
            logger.info("reader → %s for %s",
                        record.classification, record.draft_id)

        # Route by classification.
        pass_drafts = [r for r in new_drafts if r.classification == "PASS"]
        repairable_drafts = [r for r in new_drafts
                             if r.classification == "REPAIRABLE"]
        # REJECTed drafts get marked and dropped here.
        for r in new_drafts:
            if r.classification == "REJECT":
                r.final_disposition = "DISCARDED_BY_READER"
                jsonl_append(routing_log, {
                    "iteration": iteration,
                    "draft_id": r.draft_id,
                    "event": "discarded_by_reader",
                })

        # Submit PASS drafts to the configured scorer.
        for record in pass_drafts:
            try:
                score, response = score_draft(record.text, config, logger)
            except Exception as e:
                logger.error("scorer call failed for %s: %s",
                             record.draft_id, e)
                jsonl_append(routing_log, {
                    "iteration": iteration,
                    "draft_id": record.draft_id,
                    "event": "scorer_error",
                    "error": str(e),
                })
                continue

            record.originality_score = score
            record.originality_response = response
            orig_path = originality_dir / f"{record.draft_id}.json"
            orig_path.write_text(json.dumps(response, indent=2),
                                 encoding="utf-8")
            record.originality_response_path = orig_path

            jsonl_append(routing_log, {
                "iteration": iteration,
                "draft_id": record.draft_id,
                "event": "originality_scored",
                "score": score,
            })

            if score >= config.ship_score:
                record.final_disposition = "SHIPPED"
                shipped = record
                logger.info("SHIP at score %.1f: %s", score, record.draft_id)
                break  # first ship-band draft wins
            elif config.stage_g_low <= score <= config.stage_g_high:
                # Falls into the Stage G mechanical band.
                logger.info(
                    "score %.1f in Stage G band for %s — queueing",
                    score, record.draft_id,
                )
                # Append to repairable_drafts with no specific brief.
                record.classification = "REPAIRABLE_BY_SCORE"
                repairable_drafts.append(record)
            else:
                # below stage_g_low
                record.final_disposition = "DISCARDED_BY_SCORE"
                jsonl_append(routing_log, {
                    "iteration": iteration,
                    "draft_id": record.draft_id,
                    "event": "discarded_by_score",
                    "score": score,
                })

        if shipped is not None:
            break

        # Stage G pass for REPAIRABLE drafts.
        for record in repairable_drafts:
            violations = (
                extract_violation_list_for_stage_g(record.reader_report)
                if record.classification == "REPAIRABLE"
                else []   # REPAIRABLE_BY_SCORE: no reader violations, just mechanical band
            )
            try:
                repaired_text = stage_g_interface.repair(
                    draft_text=record.text,
                    violations=violations,
                    metadata={
                        "draft_id": record.draft_id,
                        "iteration": iteration,
                        "originality_score": record.originality_score,
                    },
                )
            except NotImplementedError:
                logger.info(
                    "Stage G not yet implemented; skipping repair for %s",
                    record.draft_id,
                )
                jsonl_append(routing_log, {
                    "iteration": iteration,
                    "draft_id": record.draft_id,
                    "event": "stage_g_skipped_not_implemented",
                })
                record.final_disposition = "STAGE_G_QUEUED"
                continue
            except Exception as e:
                logger.error("Stage G failed for %s: %s",
                             record.draft_id, e)
                record.final_disposition = "STAGE_G_FAILED"
                jsonl_append(routing_log, {
                    "iteration": iteration,
                    "draft_id": record.draft_id,
                    "event": "stage_g_failed",
                    "error": str(e),
                })
                continue

            # Stage G output becomes a new draft that re-enters at the
            # reader pass on the NEXT iteration. We persist it as a
            # child draft so the next iteration's reader pass picks
            # it up.
            child_id = record.draft_id + "_G"
            child_path = stage_g_dir / f"{child_id}.txt"
            child_path.write_text(repaired_text, encoding="utf-8")
            child_record = DraftRecord(
                draft_id=child_id,
                iteration=iteration + 1,   # will be processed next iter
                text=repaired_text,
                text_path=child_path,
                parent_draft_id=record.draft_id,
            )
            # Re-enter the loop: classify and (if PASS) score immediately
            # rather than waiting for next iteration. This bounds Stage G
            # output to a single reader+score round per repair.
            try:
                child_report = read_draft(
                    client, reader_prompt, child_record.text, packet_text,
                    child_record.draft_id, config.reader_model, logger,
                )
            except Exception as e:
                logger.error("post-Stage-G reader failed for %s: %s",
                             child_record.draft_id, e)
                child_record.final_disposition = "READER_ERROR"
                all_drafts.append(child_record)
                continue

            child_record.reader_report = child_report
            child_report_path = reports_dir / f"{child_record.draft_id}.json"
            child_report_path.write_text(
                json.dumps(child_report, indent=2), encoding="utf-8",
            )
            child_record.reader_report_path = child_report_path
            child_record.classification = classify_from_report(child_report)
            all_drafts.append(child_record)

            jsonl_append(routing_log, {
                "iteration": iteration,
                "draft_id": child_record.draft_id,
                "parent_draft_id": record.draft_id,
                "event": "stage_g_complete",
                "post_stage_g_classification": child_record.classification,
            })

            if child_record.classification != "PASS":
                child_record.final_disposition = "STAGE_G_INSUFFICIENT"
                continue

            # PASS after Stage G — submit to the configured scorer.
            try:
                score, response = score_draft(
                    child_record.text, config, logger,
                )
            except Exception as e:
                logger.error(
                    "scorer call failed for %s: %s",
                    child_record.draft_id, e,
                )
                continue
            child_record.originality_score = score
            child_record.originality_response = response
            orig_path = originality_dir / f"{child_record.draft_id}.json"
            orig_path.write_text(json.dumps(response, indent=2),
                                 encoding="utf-8")
            child_record.originality_response_path = orig_path

            jsonl_append(routing_log, {
                "iteration": iteration,
                "draft_id": child_record.draft_id,
                "event": "post_stage_g_scored",
                "score": score,
            })

            if score >= config.ship_score:
                child_record.final_disposition = "SHIPPED"
                shipped = child_record
                logger.info("SHIP (post-Stage-G) at score %.1f: %s",
                            score, child_record.draft_id)
                break
            else:
                child_record.final_disposition = "STAGE_G_INSUFFICIENT"

        if shipped is not None:
            break

    # End of loop.
    result = build_result(config, all_drafts, shipped, iteration)
    result_path = config.output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, default=str),
                           encoding="utf-8")

    if shipped is None:
        manual_marker = config.output_dir / "MANUAL_QUEUE"
        manual_marker.write_text(
            f"Chapter {config.chapter_label} exhausted "
            f"{config.max_iterations} iterations without producing a "
            f"draft at or above the ship score of {config.ship_score}. "
            f"Manual review required.\n",
            encoding="utf-8",
        )
        logger.warning(
            "MANUAL_QUEUE: chapter %s exhausted %d iterations",
            config.chapter_label, config.max_iterations,
        )

    logger.info("=== Phase 2 orchestrator end ===")
    return result


def build_result(
    config: OrchestratorConfig,
    all_drafts: list[DraftRecord],
    shipped: Optional[DraftRecord],
    iterations_used: int,
) -> dict:
    return {
        "chapter": config.chapter_label,
        "iterations_used": iterations_used,
        "max_iterations": config.max_iterations,
        "total_drafts": len(all_drafts),
        "shipped": (
            {
                "draft_id": shipped.draft_id,
                "text_path": str(shipped.text_path),
                "originality_score": shipped.originality_score,
                "iteration": shipped.iteration,
                "parent_draft_id": shipped.parent_draft_id,
            }
            if shipped is not None else None
        ),
        "manual_queue": shipped is None,
        "drafts_summary": [
            {
                "draft_id": r.draft_id,
                "iteration": r.iteration,
                "parent_draft_id": r.parent_draft_id,
                "classification": r.classification,
                "originality_score": r.originality_score,
                "final_disposition": r.final_disposition,
            }
            for r in all_drafts
        ],
    }


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 pipeline orchestrator."
    )
    parser.add_argument("--packet", required=True, type=Path,
                        help="Path to the chapter packet (text file).")
    parser.add_argument("--drafting-prompt", required=False, type=Path,
                        default=None,
                        help="Optional override. Path to a drafter system "
                             "prompt text file. If omitted, the embedded "
                             "DRAFTING_PROMPT constant at the top of "
                             "orchestrator.py is used.")
    parser.add_argument("--reader-prompt", required=True, type=Path,
                        help="Path to reader_prompt_v1.txt.")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Output directory (will be created).")
    parser.add_argument("--chapter-label", default="chapter",
                        help="Label used in draft IDs (e.g. 'ch01').")
    parser.add_argument("--n", type=int, default=N_DEFAULT,
                        help=f"Drafts per iteration (default {N_DEFAULT}).")
    parser.add_argument("--temperature", type=float,
                        default=TEMPERATURE_DEFAULT,
                        help=f"Sampling temperature (default {TEMPERATURE_DEFAULT}).")
    parser.add_argument("--ship-score", type=float,
                        default=SHIP_SCORE_DEFAULT,
                        help=f"Ship band floor (default {SHIP_SCORE_DEFAULT}).")
    parser.add_argument("--stage-g-low", type=float,
                        default=STAGE_G_BAND_LOW_DEFAULT,
                        help=f"Stage G band floor (default {STAGE_G_BAND_LOW_DEFAULT}).")
    parser.add_argument("--stage-g-high", type=float,
                        default=STAGE_G_BAND_HIGH_DEFAULT,
                        help=f"Stage G band ceiling (default {STAGE_G_BAND_HIGH_DEFAULT}).")
    parser.add_argument("--discard-below", type=float,
                        default=DISCARD_BELOW_DEFAULT,
                        help=f"Below this score, discard (default {DISCARD_BELOW_DEFAULT}).")
    parser.add_argument("--max-iterations", type=int,
                        default=MAX_ITERATIONS_DEFAULT,
                        help=f"Cap on iterations per chapter "
                             f"(default {MAX_ITERATIONS_DEFAULT}).")
    parser.add_argument("--drafter-model", default=DRAFTER_MODEL_DEFAULT,
                        help=f"Drafter model (default {DRAFTER_MODEL_DEFAULT}).")
    parser.add_argument("--reader-model", default=READER_MODEL_DEFAULT,
                        help=f"Reader model (default {READER_MODEL_DEFAULT}).")
    parser.add_argument("--scorer", default="originality",
                        choices=["originality", "local", "both"],
                        help="Which scorer to route PASS drafts through. "
                             "'originality' (default) — paid API. "
                             "'local' — band-classifier-based local "
                             "predictor (requires calibration.json). "
                             "'both' — call both, route on Originality, "
                             "log delta for validation.")
    parser.add_argument("--calibration", default=None, type=Path,
                        help="Path to calibration.json produced by "
                             "corpus_calibrator.py. Used only when "
                             "--scorer is local or both.")

    args = parser.parse_args()

    config = OrchestratorConfig(
        packet_path=args.packet,
        drafting_prompt_path=args.drafting_prompt,
        reader_prompt_path=args.reader_prompt,
        output_dir=args.output_dir,
        n=args.n,
        temperature=args.temperature,
        ship_score=args.ship_score,
        stage_g_low=args.stage_g_low,
        stage_g_high=args.stage_g_high,
        discard_below=args.discard_below,
        max_iterations=args.max_iterations,
        drafter_model=args.drafter_model,
        reader_model=args.reader_model,
        chapter_label=args.chapter_label,
        scorer=args.scorer,
        calibration_path=args.calibration,
    )

    result = run_pipeline(config)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
