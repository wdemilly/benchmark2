"""
local_scorer.py — predict an Originality.ai turbo score from a draft, locally
==============================================================================

Pipeline:
  1. Load calibration (anchors + regression coefficients) from
     calibration.json if present; fall back to defaults baked into
     extended_band_features.DEFAULT_REGRESSION otherwise.
  2. Run the 6-band classifier on the draft.
  3. Compute the 5 production features (last_third_green_pct,
     deep_green_pct, longest_warm_run, orange_to_green_recoveries,
     last_third_orange_pct).
  4. Apply the regression to produce a predicted score in [0, 100].

Accuracy on the operator's 38-export cross-register corpus:
  LOO Pearson r = 0.944, RMSE = 5.17 points.

This module is the orchestrator-facing entry point. The orchestrator
calls local_scorer.score_text(text) the same way it calls
originality_api.score_text(text), and gets back a (score, details)
tuple.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import band_classifier
import extended_band_features


logger = logging.getLogger("phase2.local_scorer")


# ============================================================================
# Calibration loading — cached at module level
# ============================================================================

_CALIBRATION_CACHE: Optional[dict] = None
_CALIBRATION_PATH = Path("calibration.json")


def set_calibration_path(path: Path) -> None:
    """Override the default calibration.json location."""
    global _CALIBRATION_PATH, _CALIBRATION_CACHE
    _CALIBRATION_PATH = path
    _CALIBRATION_CACHE = None


def _load_calibration() -> Optional[dict]:
    """
    Load calibration.json from the configured path. Returns the parsed
    calibration dict, or an empty dict if the file doesn't exist (in
    which case the module falls back to
    extended_band_features.DEFAULT_REGRESSION, which carries the same
    coefficients).
    """
    global _CALIBRATION_CACHE
    if _CALIBRATION_CACHE is not None:
        return _CALIBRATION_CACHE
    if not _CALIBRATION_PATH.exists():
        logger.info(
            "No calibration.json at %s — using DEFAULT_REGRESSION "
            "from extended_band_features (LOO r=0.944, n=38).",
            _CALIBRATION_PATH,
        )
        _CALIBRATION_CACHE = {}
        return {}
    _CALIBRATION_CACHE = json.loads(
        _CALIBRATION_PATH.read_text(encoding="utf-8")
    )
    logger.info("loaded calibration from %s", _CALIBRATION_PATH)
    return _CALIBRATION_CACHE


# ============================================================================
# Public API
# ============================================================================

def score_text(
    text: str,
    classifier: Optional[band_classifier.BandClassifier] = None,
) -> tuple[float, dict]:
    """
    Score a draft locally. Returns (predicted_score, details_dict).

    The details dict mirrors what originality_api.score_text returns
    in shape:
      {
        "score":              0-100 predicted turbo score,
        "bands":              [{"text": "...", "band": "DEEP_GREEN"}, ...],
        "features":           {... 5 production features + diagnostics ...},
        "calibration_used":   "calibration.json" | "default_baked_in",
        "regression_metrics": {... fit quality info ...}
      }
    """
    calibration = _load_calibration() or {}
    anchors = calibration.get("anchors")
    regression = calibration.get("regression")
    using_file_calibration = bool(regression and regression.get("calibrated"))

    # Build classifier with anchors from calibration if present;
    # otherwise the classifier uses its built-in 6-band defaults.
    if classifier is None:
        config = band_classifier.BandClassifierConfig(anchors=anchors)
        classifier = band_classifier.BandClassifier(config=config)

    # Classify every sentence into one of the 6 bands.
    bands = classifier.classify(text)
    prose_bands = [(t, b) for t, b in bands if b is not None]

    # Compute the 5 production features.
    features = extended_band_features.compute_features(prose_bands)

    # Predict score using the regression (file's if loaded, else
    # the DEFAULT_REGRESSION baked into extended_band_features).
    predicted = extended_band_features.predict_score(features, regression)

    # Pick the regression metrics to report.
    if using_file_calibration:
        regression_metrics = regression.get("metrics", {})
    else:
        regression_metrics = extended_band_features.DEFAULT_REGRESSION.get(
            "metrics", {}
        )

    details = {
        "score": predicted,
        "bands": [{"text": t, "band": b} for t, b in prose_bands],
        "features": features.as_dict(),
        "calibration_used": (
            str(_CALIBRATION_PATH) if using_file_calibration
            else "default_baked_in"
        ),
        "regression_metrics": regression_metrics,
    }
    return predicted, details


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Run the local scorer on a draft and print the "
                    "predicted score plus per-sentence band breakdown."
    )
    parser.add_argument("draft_path", type=Path,
                        help="Path to draft text file.")
    parser.add_argument("--calibration", type=Path,
                        default=Path("calibration.json"),
                        help="Path to calibration.json "
                             "(default: ./calibration.json). If absent, "
                             "uses DEFAULT_REGRESSION baked into "
                             "extended_band_features.")
    parser.add_argument("--bands-out", type=Path, default=None,
                        help="Optional path to write per-sentence "
                             "band JSON.")
    args = parser.parse_args()

    set_calibration_path(args.calibration)
    text = args.draft_path.read_text(encoding="utf-8")
    score, details = score_text(text)

    print(f"=== PREDICTED SCORE: {score:.1f} ===")
    print(f"Calibration: {details['calibration_used']}")
    rm = details["regression_metrics"]
    if rm:
        loo_r = rm.get("loo_correlation", "?")
        loo_rmse = rm.get("loo_rmse", "?")
        n = rm.get("n_samples", "?")
        print(f"  Fit quality: LOO r={loo_r}, RMSE={loo_rmse}, n={n}")
    print()

    print("=== FEATURES ===")
    for k, v in details["features"].items():
        print(f"  {k:<30} {v}")

    # Quick band share summary
    bands = [b["band"] for b in details["bands"]]
    if bands:
        print(f"\n=== BAND SHARES ({len(bands)} sentences) ===")
        from collections import Counter
        counter = Counter(bands)
        for band in ["DEEP_GREEN", "LIGHT_GREEN", "YELLOW",
                     "YELLOW_ORANGE", "LIGHT_ORANGE", "DEEP_ORANGE"]:
            c = counter.get(band, 0)
            if c > 0:
                pct = c / len(bands) * 100
                print(f"  {band:<15} {c:>4}  ({pct:>5.1f}%)")

    if args.bands_out:
        args.bands_out.write_text(
            json.dumps(details["bands"], indent=2), encoding="utf-8",
        )
        print(f"\nWrote band breakdown to {args.bands_out}")


if __name__ == "__main__":
    main()
