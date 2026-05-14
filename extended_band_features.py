"""
extended_band_features.py — compute production-model features from a band sequence
====================================================================================

Replaces the simpler contagion_metrics feature set with the more
powerful one fit to the corpus. The production model uses 5 features
and achieves leave-one-out r=0.960 on the operator's scored corpus.

The features can all be computed from a 6-band categorical sequence:
   DEEP_GREEN, LIGHT_GREEN, YELLOW, YELLOW_ORANGE, LIGHT_ORANGE, DEEP_ORANGE

The bands can come from either:
  - docx_band_extractor (ground-truth extraction from Originality .docx)
  - band_classifier (Claude-based prediction on fresh drafts)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional


BANDS_6 = (
    "DEEP_GREEN", "LIGHT_GREEN", "YELLOW",
    "YELLOW_ORANGE", "LIGHT_ORANGE", "DEEP_ORANGE",
)

GREEN_BANDS = {"DEEP_GREEN", "LIGHT_GREEN"}
ORANGE_BANDS = {"LIGHT_ORANGE", "DEEP_ORANGE"}
WARM_BANDS = {"YELLOW_ORANGE", "LIGHT_ORANGE", "DEEP_ORANGE"}


@dataclass
class ProductionFeatures:
    """The 5 features the production regression uses."""
    last_third_green_pct:        float
    deep_green_pct:              float
    longest_warm_run:            int
    orange_to_green_recoveries:  int
    last_third_orange_pct:       float

    # Reported but not used by the regression:
    sentence_count:              int
    contagion_3plus:             int    # diagnostic only
    green_chars_pct:             float
    orange_chars_pct:            float

    def as_dict(self) -> dict:
        return asdict(self)


def _bands_only(sentences) -> list[str]:
    """Strip any (text, band) pairs down to a list of bands."""
    out = []
    for item in sentences:
        if isinstance(item, tuple):
            text, band = item
            if band is None:
                continue
            out.append(band)
        elif isinstance(item, str):
            out.append(item)
    return out


def _is_green(band: str) -> bool:
    return band in GREEN_BANDS


def _is_orange(band: str) -> bool:
    return band in ORANGE_BANDS


def _is_warm(band: str) -> bool:
    return band in WARM_BANDS


def compute_features(sentences) -> ProductionFeatures:
    """
    Compute the production-model features from a sequence of bands
    (or (text, band) tuples). Tolerates both 4-band and 6-band inputs;
    in 4-band input, DEEP_GREEN/LIGHT_GREEN collapse to GREEN, so
    deep_green_pct will be 0.
    """
    bands = _bands_only(sentences)
    n = len(bands)
    if n == 0:
        return ProductionFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0)

    # Sentence-level percentages
    deep_green_count = sum(1 for b in bands if b == "DEEP_GREEN")

    # Last third
    third_start = (2 * n) // 3
    last_third = bands[third_start:]
    last_third_n = len(last_third) or 1
    last_third_green = sum(1 for b in last_third if _is_green(b))
    last_third_orange = sum(1 for b in last_third if _is_orange(b))

    # Run-length encoding for warm runs (YO + ORANGE bands combined)
    longest_warm = 0
    current_warm = 0
    for b in bands:
        if _is_warm(b):
            current_warm += 1
            longest_warm = max(longest_warm, current_warm)
        else:
            current_warm = 0

    # Orange → green recoveries (any ORANGE-family band followed by
    # any GREEN-family band)
    recoveries = 0
    for i in range(1, n):
        if _is_orange(bands[i-1]) and _is_green(bands[i]):
            recoveries += 1

    # Contagion zones (diagnostic; not in the regression)
    contagion = 0
    current_orange = 0
    for b in bands:
        if _is_orange(b):
            current_orange += 1
        else:
            if current_orange >= 3:
                contagion += 1
            current_orange = 0
    if current_orange >= 3:
        contagion += 1

    # Per-band raw counts for diagnostic reporting
    green_count = sum(1 for b in bands if _is_green(b))
    orange_count = sum(1 for b in bands if _is_orange(b))

    return ProductionFeatures(
        last_third_green_pct=round(last_third_green / last_third_n * 100, 2),
        deep_green_pct=round(deep_green_count / n * 100, 2),
        longest_warm_run=longest_warm,
        orange_to_green_recoveries=recoveries,
        last_third_orange_pct=round(last_third_orange / last_third_n * 100, 2),
        sentence_count=n,
        contagion_3plus=contagion,
        green_chars_pct=round(green_count / n * 100, 2),
        orange_chars_pct=round(orange_count / n * 100, 2),
    )


def predict_score(
    features: ProductionFeatures,
    regression: Optional[dict] = None,
) -> float:
    """
    Apply the production regression coefficients to compute a
    predicted Originality.ai turbo score in [0, 100].
    """
    if regression is None:
        regression = DEFAULT_REGRESSION
    coefs = regression["coefficients"]
    means = regression.get("feature_means", {})
    stds = regression.get("feature_stds", {})
    intercept = regression["intercept"]

    f = features.as_dict()
    score = intercept
    for feat_name, coef in coefs.items():
        value = f.get(feat_name, 0)
        mean = means.get(feat_name, 0)
        std = stds.get(feat_name, 1) or 1
        z = (value - mean) / std
        score += coef * z

    return max(0.0, min(100.0, round(score, 1)))


# ============================================================================
# Production regression — fit on the operator's corpus (n=32 after dropping
# the export(51) outlier cluster). LOO r = 0.960, LOO RMSE = 4.51 points.
# ============================================================================

DEFAULT_REGRESSION = {
    "intercept": 83.44707212543099,
    "coefficients": {
        "last_third_green_pct":       3.997262085648683,
        "deep_green_pct":             5.403776898037795,
        "longest_warm_run":          -3.585729268272636,
        "orange_to_green_recoveries": -6.077802478729715,
        "last_third_orange_pct":      1.7956253937321024,
    },
    "feature_means": {
        "last_third_green_pct":       49.99052631578947,
        "deep_green_pct":             34.05710526315789,
        "longest_warm_run":           6.394736842105263,
        "orange_to_green_recoveries": 4.342105263157895,
        "last_third_orange_pct":      14.892894736842105,
    },
    "feature_stds": {
        "last_third_green_pct":       14.50676226345628,
        "deep_green_pct":             10.86283519834456,
        "longest_warm_run":           3.135725692938502,
        "orange_to_green_recoveries": 5.388826116720495,
        "last_third_orange_pct":      18.7886301180423,
    },
    "calibrated": True,
    "metrics": {
        "in_sample_correlation": 0.959,
        "in_sample_rmse": 4.44,
        "loo_correlation": 0.944,
        "loo_rmse": 5.17,
        "n_samples": 38,
        "n_features": 5,
    },
    "note": "Production regression fit on book-agnostic corpus (n=38): "
            "regency, paranormal, contemporary, literary registers. "
            "Excludes label-conflict pair + export(51) cluster.",
}
