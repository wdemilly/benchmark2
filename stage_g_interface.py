"""
stage_g_interface.py — Stage G repair interface (stub)
=======================================================

Stage G is the post-draft grafting/repair pass. Phase 2 ships with
the INTERFACE defined and the IMPLEMENTATION deferred — drafts that
route here get a NotImplementedError, which the orchestrator catches
and logs as "stage_g_skipped_not_implemented." The pipeline still
flows; Stage G simply becomes a no-op until its implementation
lands.

Once Stage G is implemented (against the current drafter's actual
fingerprints — see CHANGES_FROM_PHASE_1.txt for the Phase 3
recalibration note), this file is the single entry point the
orchestrator calls. The signature is stable; implementations may
change freely behind it.

The interface contract:

  Input
  -----
  draft_text : str
      The full draft prose to repair.
  violations : list[dict]
      The structured violation list produced by the reader, one
      entry per triggered Section A or B category. Schema:

        [
          {
            "category": "A1_the_way_X",          # rubric key
            "count": 3,
            "severity": "medium",
            "instances": [
              {
                "quote": "the way her wolf moved when she was tired",
                "locator": "P14",
                "note": ""
              },
              ...
            ]
          },
          ...
        ]

      May be empty when Stage G is invoked on a score-band draft
      (Originality 85-94) rather than a reader-flagged REPAIRABLE
      draft. In that case the implementation should fall back to
      its own mechanical scan (the v25/v29 Stage G word-list /
      G3/G4 logic the operator already has).
  metadata : dict
      Diagnostic context: draft_id, iteration, originality_score
      (when present). Implementations may persist this alongside
      their audit output.

  Output
  ------
  str
      The repaired draft text. Must be a valid prose string the
      orchestrator can re-submit to the reader and (if PASS) the
      Originality API.

  Contracts the implementation must honour
  ----------------------------------------
  1. Word count delta is bounded. Recommended: stay within ±5% of
     input word count unless the implementation explicitly
     documents larger deltas as part of its mechanical model.
     The orchestrator does NOT enforce this; the implementation
     is responsible.
  2. Voice and beat coverage are preserved. Stage G is mechanical
     and graft-based; it does not rewrite plot or alter the
     chapter's narrative shape.
  3. The function is allowed to raise. The orchestrator catches
     and logs without halting the pipeline.
"""

from __future__ import annotations

from typing import Optional


def repair(
    draft_text: str,
    violations: list[dict],
    metadata: Optional[dict] = None,
) -> str:
    """
    Repair a draft using Stage G's grafting/copy-edit logic.

    Not yet implemented — raises NotImplementedError. The orchestrator
    is designed to handle this gracefully (the draft is marked
    STAGE_G_QUEUED and the iteration moves on).

    When this function is implemented, the implementer should:

      1. Read each violation entry and decide which to address.
         The simplest mapping is: address every Section A category
         that has a regex-deletable mechanical fix (A1, A3, A9,
         parts of A8), and route the rest to mechanical graft
         from clean runner-up drafts (the v25/v29 G3a pathway).

      2. Honour the contracts in this module's docstring.

      3. Write an audit file under metadata-named output paths if
         desired. The orchestrator does not require it.

      4. Return the repaired prose as a single string.
    """
    if metadata is None:
        metadata = {}

    raise NotImplementedError(
        "Stage G is not yet implemented in Phase 2. The interface "
        "exists; the implementation is deferred. See "
        "CHANGES_FROM_PHASE_1.txt for the Phase 3 recalibration plan."
    )


# ============================================================================
# Helper utilities the implementation can reuse (kept minimal)
# ============================================================================

def violations_have_category(
    violations: list[dict],
    category_key: str,
) -> bool:
    """Convenience: did the reader trigger this specific category?"""
    return any(v.get("category") == category_key for v in violations)


def get_category_block(
    violations: list[dict],
    category_key: str,
) -> Optional[dict]:
    """Return the full violation block for one category, or None."""
    for v in violations:
        if v.get("category") == category_key:
            return v
    return None
